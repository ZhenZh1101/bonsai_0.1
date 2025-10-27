"""Quantize a teacher for a small Mac (plan sections 3, 11; runs/remote_gen/): Qwen3-1.7B -> teachers/Qwen3-1.7B-8bit.

    python -m scripts.quantize_teacher                        # 8-bit, group size 64, then the check below
    python -m scripts.quantize_teacher --cpu                  # while the GPU is busy (pretraining)
    python -m scripts.quantize_teacher --check-only --cpu     # re-run the check on an existing quantized teacher

On an 8 GB Mac the bf16 Qwen3-1.7B (3.8 GB of weights) leaves room for a batch of ~8 and reads 3.8 GB per decode
step (~100 tok/s on an M2); 8-bit weights (~2.2 GB) about double the batch and halve the bytes per step, while
staying close to bf16 (the check measures how close). The off-policy generation only uses the teacher's text, so
the stored samples do not depend on the weight format; the on-policy teacher stays bf16 Qwen3-0.6B.

mlx_lm.convert writes its own tokenizer files; the original Qwen3 files are copied back so the hash check of
plan section 6 (bonsai.teacher.check_tokenizer) still holds. QUANT.json records the source commit and settings.

The check scores the same token sequences (kept off-policy samples, else built-in prompts) with both models and
reports the mean per-token KL(bf16 || quantized), the share of positions where the greedy next token agrees, and
the bf16 / quantized log-probability of the sequences.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from bonsai.common import DATA_ROOT, TEACHER_DIR, iter_jsonl, log, write_json
from bonsai.data.sources import TEACHERS
from bonsai.tokenizer import TOKENIZER_FILES, sha256

CHECK_TEXTS = [
    "Sarah has 3 boxes of 12 apples and gives away 10. How many apples are left?",
    "Explain in two sentences why the sky is blue.",
    "Write a Python function that returns the larger of two numbers.",
    "Question: Which gas do plants take in from the air?\nA. Oxygen\nB. Carbon dioxide\nC. Nitrogen\nD. Helium",
]


def quantize(src, dst, bits, group):
    from mlx_lm.convert import convert
    if dst.exists():
        shutil.rmtree(dst)
    convert(hf_path=str(src), mlx_path=str(dst), quantize=True, q_bits=bits, q_group_size=group)
    for f in TOKENIZER_FILES + ("generation_config.json", "LICENSE"):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    write_json(dst / "QUANT.json", {"source": src.name, "revision": TEACHERS.get(src.name, (None, None))[1],
                                    "bits": bits, "group_size": group, "time": time.strftime("%F %T"),
                                    "tokenizer_sha256": {f: sha256(dst / f) for f in TOKENIZER_FILES}})
    size = sum(p.stat().st_size for p in dst.glob("*.safetensors")) / 1e9
    log(f"{src.name} -> {dst} ({bits}-bit, group {group}): {size:.2f} GB of weights")


def sequences(qwen, n, max_len=768):
    """Teacher ids of up to n kept off-policy samples (the format the teacher generates), else chat prompts."""
    from bonsai import chat
    out = []
    for d in (DATA_ROOT / "data" / "offpolicy", DATA_ROOT / "data" / "smoke" / "offpolicy"):
        for p in sorted(d.glob("kept_*.jsonl")):
            for r in iter_jsonl(p):
                tools = chat.TOOLS if r.get("tools") else None
                out.append(chat.render(qwen, r["messages"], tools)[0][:max_len])
                if len(out) >= n:
                    return out
    for t in CHECK_TEXTS[:max(0, n - len(out))]:
        out.append(chat.render(qwen, [{"role": "user", "content": t}], add_generation_prompt=True)[0])
    return out


def check(src, dst, n, max_len=768):
    import mlx.core as mx
    from bonsai.teacher import Teacher
    ref, q = Teacher(src.name), Teacher(dst.name)
    kls, agree, lp_ref, lp_q, total = 0.0, 0, 0.0, 0.0, 0
    for ids in sequences(ref.qwen, n, max_len):
        x = mx.array([ids])
        a = ref.logits(x)[0, :-1]
        b = q.logits(x)[0, :-1]
        la, lb = a - mx.logsumexp(a, -1, keepdims=True), b - mx.logsumexp(b, -1, keepdims=True)
        kl = (mx.exp(la) * (la - lb)).sum(-1)
        tgt = mx.array(ids[1:])
        mx.eval(kl)
        kls += kl.sum().item()
        agree += (a.argmax(-1) == b.argmax(-1)).sum().item()
        lp_ref += mx.take_along_axis(la, tgt[:, None], -1).sum().item()
        lp_q += mx.take_along_axis(lb, tgt[:, None], -1).sum().item()
        total += len(ids) - 1
    res = {"sequences": n, "tokens": total, "mean_kl_nats": kls / total, "greedy_agreement": agree / total,
           "logprob_per_token_bf16": lp_ref / total, "logprob_per_token_quantized": lp_q / total}
    log(f"check {dst.name} vs {src.name} on {total:,} tokens: mean KL {res['mean_kl_nats']:.4f} nats/token, "
        f"greedy next token agrees {res['greedy_agreement']:.1%}, log p per token {res['logprob_per_token_bf16']:.3f} "
        f"(bf16) vs {res['logprob_per_token_quantized']:.3f}")
    write_json(dst / "QUANT_CHECK.json", res)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="Qwen3-1.7B")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--check", type=int, default=24, help="sequences for the bf16-vs-quantized check (0 = skip)")
    ap.add_argument("--check-len", type=int, default=768, help="tokens per check sequence")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="run on the CPU (the GPU is busy)")
    args = ap.parse_args()
    import mlx.core as mx
    if args.cpu:
        mx.set_default_device(mx.cpu)
    src = TEACHER_DIR / args.teacher
    dst = TEACHER_DIR / f"{args.teacher}-{args.bits}bit"
    if not (src / "config.json").exists():
        raise SystemExit(f"{src} missing: python -m scripts.download --only teachers --teachers {args.teacher}")
    if not args.check_only:
        quantize(src, dst, args.bits, args.group_size)
    if args.check:
        check(src, dst, args.check, args.check_len)


if __name__ == "__main__":
    main()
