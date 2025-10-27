"""Day 0: measure the vocabulary the corpus actually uses and fix the student's subset S (plan section 6).

    python -m scripts.vocab_subset                  # ~100M-token sample, 99.99% coverage target, cap 65,536
    python -m scripts.vocab_subset --tokens 2e7     # quicker
    python -m scripts.vocab_subset --max 94208      # no cap below the coverage rule (uint32 shards)

The sample follows the training mix itself: S0 5%, S1 70%, S2 25% of the tokens, each stage split by the
source weights of configs/s{0,1,2}.yaml, and every record is rendered exactly as the shards render it
(`bonsai.data.render.Renderer` over an identity-mapped full vocabulary, so ChatML samples are counted with the
template tokens, thinking blocks, soft switches and the tools block). Teacher ids are counted and the coverage
curve printed.

Forced into S: the 256 byte tokens, the 26 control tokens, and every token of the template's own constants
(`<|im_start|>assistant\\n`, the tools system block, the tool-call JSON frame, the empty think block, THINK_STOP,
the soft switches, `Answer:` forms): a split there would make every conversation non-canonical for the
teacher. N = the smallest n with coverage >= target plus the forced tokens, rounded up to a multiple of 2048 and
capped at --max (65,536 = uint16 shards; the user's choice of 2026-10-25 after the measurement showed 99.99%
needs ~93k). Student ids are sorted by frequency; tokens outside S are spelled losslessly with in-S pieces.

Writes tokenizer/vocab_subset.npz (s2t, counts, meta), tokenizer/sample_counts.npy, tokenizer/vocab_report.json.
Changing S changes `vocab_sha`: rebuild every shard afterwards.
"""

import argparse
import json
import time

import numpy as np

from bonsai import chat
from bonsai.common import CONFIG_DIR, FILTERED_DIR, TOKENIZER_DIR, load_yaml, log, write_json
from bonsai.tokenizer import CONTROL_IDS, N_BPE, TEACHER_VOCAB, QwenTokenizer, Tokenizer, VocabMap

STAGE_SHARES = {"s0": 0.05, "s1": 0.70, "s2": 0.25}


def template_ids(qwen):
    """Teacher ids of the template constants and fixed answer forms (content kept trivial)."""
    q, out = "What is 2 + 3?", set()
    convs = [
        (chat.tool_trajectory(q, "print(2 + 3)", "5", "Answer: 5", mode="no_think"), chat.TOOLS),
        (chat.tool_trajectory(q, "print(2 + 3)", "5", "Answer: 5", reasoning="Add them.", mode="think",
                              system="Be brief."), chat.TOOLS),
        ([{"role": "system", "content": "Be brief."}, {"role": "user", "content": chat.with_soft_switch(q, "think")},
          {"role": "assistant", "content": "Answer: 5", "reasoning_content": "2 + 3 = 5." + chat.THINK_STOP}], None),
        ([{"role": "user", "content": chat.with_soft_switch(q, "no_think")},
          {"role": "assistant", "content": "Answer: 5", "reasoning_content": ""},
          {"role": "user", "content": "Thanks"}, {"role": "assistant", "content": "You're welcome."}], None),
    ]
    for msgs, tools in convs:
        out |= set(chat.render(qwen, msgs, tools)[0])
        for et in (True, False):
            out |= set(chat.render(qwen, msgs[:-1] if msgs[-1]["role"] == "assistant" else msgs, tools,
                                   add_generation_prompt=True, enable_thinking=et)[0])
    for a in ["A", "B", "C", "D", "E", "PROVED", "DISPROVED", "UNKNOWN"] + [str(i) for i in range(10)]:
        out |= set(qwen.encode(f"Answer: {a}")) | set(qwen.encode(f"\n\nAnswer: {a}"))
    return {int(i) for i in out}


def sample_stage(stage, tok, budget, counts, per_source):
    cfg = load_yaml(CONFIG_DIR / f"{stage}.yaml")
    from bonsai.data.render import Renderer
    for name, spec in cfg["sources"].items():
        target = budget * STAGE_SHARES[stage] * float(spec["weight"])
        path = FILTERED_DIR / f"{spec['pool']}.jsonl"
        if not path.exists():
            log(f"  warning: {path} missing, {stage}/{name} not sampled")
            continue
        renderer = Renderer(tok, spec["format"], spec)
        c = np.zeros(TEACHER_VOCAB, dtype=np.int64)
        n_docs = 0
        with open(path) as f:
            while c.sum() < target:
                lines = [f.readline() for _ in range(2000)]
                lines = [l for l in lines if l]
                if not lines:
                    break
                for r in renderer.batch([json.loads(l) for l in lines]):
                    if r is not None:
                        c += np.bincount(np.asarray(r[0], dtype=np.int64), minlength=TEACHER_VOCAB)
                        n_docs += 1
        counts += c
        per_source[f"{stage}/{name}"] = c
        log(f"  {stage}/{name:20s} {n_docs:8,} docs  {c.sum() / 1e6:7.2f}M tokens (target {target / 1e6:.2f}M)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=float, default=1e8)
    ap.add_argument("--target", type=float, default=0.9999)
    ap.add_argument("--multiple", type=int, default=2048)
    ap.add_argument("--max", type=int, default=65536, help="upper bound on N (65,536 keeps uint16 shards)")
    ap.add_argument("--size", type=int, default=0, help="force this subset size instead of the measured one")
    ap.add_argument("--fallback", choices=["pieces", "bytes"], default="pieces",
                    help="how out-of-subset tokens are spelled (lossless either way)")
    args = ap.parse_args()

    qwen = QwenTokenizer()
    full = Tokenizer(qwen, VocabMap(list(range(N_BPE + len(CONTROL_IDS))), qwen, {"note": "identity map"}))
    t0 = time.time()
    counts = np.zeros(TEACHER_VOCAB, dtype=np.int64)
    per_source = {}
    for stage in ("s1", "s2", "s0"):
        sample_stage(stage, full, args.tokens, counts, per_source)
    total = int(counts.sum())
    log(f"sample: {total / 1e6:.1f}M tokens, {int((counts > 0).sum()):,} distinct ids, {time.time() - t0:.0f}s")

    tmpl = template_ids(qwen)
    forced = set(int(i) for i in qwen.byte_ids) | set(CONTROL_IDS.values()) | tmpl
    order = np.argsort(-counts, kind="stable")
    order = order[counts[order] > 0]
    cum = np.cumsum(counts[order]) / total
    curve = {n: float(cum[min(n, len(cum)) - 1]) for n in (16384, 32768, 40960, 49152, 57344, 65536, 81920, 98304)}
    for n, c in curve.items():
        log(f"  top {n:6d}: coverage {c:.5%}")
    n_forced_extra = len(forced - set(order[:int(np.searchsorted(cum, args.target)) + 1].tolist()))
    need = {t: int(np.searchsorted(cum, t)) + 1 + len(forced - set(order[:int(np.searchsorted(cum, t)) + 1].tolist()))
            for t in (0.9995, 0.9999, 0.99999)}
    for t, n in need.items():
        log(f"  coverage {t}: N = {n:,} (incl. forced tokens outside the top ids)")
    n = args.size or min(args.max, int(np.ceil(need[args.target] / args.multiple) * args.multiple))

    def subset(size):
        members = set(forced)
        for tid in order:
            if len(members) >= size:
                break
            members.add(int(tid))
        # student ids by frequency (forced tokens with zero count at the end, by teacher id)
        return sorted(members, key=lambda t: (-int(counts[t]), t))

    # what each candidate size costs: out-of-subset rate and how much the lossless split lengthens the data
    candidates = {}
    for size in sorted({49152, 57344, 65536, 81920, n, need[args.target] + (-need[args.target]) % args.multiple}):
        vm = VocabMap(subset(size), qwen, {}, fallback="pieces")
        in_s = vm.t2s >= 0
        bad = np.flatnonzero(~in_s & (counts > 0))
        extra_pieces = sum(int(counts[t]) * (len(vm.split(t)) - 1) for t in bad if t < N_BPE)
        extra_bytes = sum(int(counts[t]) * (len(qwen.token_bytes(t)) - 1) for t in bad if t < N_BPE)
        py = per_source.get("s1/stack_python")
        candidates[size] = {"oov_fraction": float(counts[~in_s].sum() / total),
                            "length_increase_pieces": extra_pieces / total, "length_increase_bytes": extra_bytes / total,
                            "stack_python_oov": float(py[~in_s].sum() / max(py.sum(), 1)) if py is not None else None}
        c = candidates[size]
        log(f"  N={size:6d}: out-of-subset {c['oov_fraction']:.4%} (python {c['stack_python_oov'] or 0:.3%}); "
            f"data +{c['length_increase_pieces']:.3%} tokens with piece split, +{c['length_increase_bytes']:.3%} "
            f"with byte split")

    s2t = subset(n)
    assert len(s2t) == n, (len(s2t), n)
    in_s = np.zeros(TEACHER_VOCAB, dtype=bool)
    in_s[s2t] = True
    assert in_s[list(tmpl)].all()
    oov = float(counts[~in_s].sum() / total)
    src_oov = {k: float(v[~in_s].sum() / max(v.sum(), 1)) for k, v in per_source.items()}
    top_oov = [(qwen.decode([int(t)]), int(counts[t])) for t in order[~in_s[order]][:40]]
    meta = {"size": n, "target_coverage": args.target, "sample_tokens": total, "oov_fraction": oov,
            "stage_shares": STAGE_SHARES, "forced": {"bytes": 256, "control": len(CONTROL_IDS), "template": len(tmpl)},
            "coverage_curve": curve, "n_for_coverage": need, "candidates": candidates, "fallback": args.fallback,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    vmap = VocabMap(s2t, qwen, meta, fallback=args.fallback)
    path = vmap.save(TOKENIZER_DIR, counts[np.array(s2t)])
    np.save(TOKENIZER_DIR / "sample_counts.npy", counts)
    write_json(TOKENIZER_DIR / "vocab_report.json", {**meta, "oov_fraction_by_source": src_oov,
                                                     "top_oov_tokens": top_oov})
    log(f"subset S: {n:,} ids ({len(tmpl)} template tokens forced, {n_forced_extra} of the forced tokens outside "
        f"the coverage set), out-of-subset fraction on the sample {oov:.5%}, vocab_sha {vmap.sha()} -> {path}")
    for k, v in src_oov.items():
        log(f"  oov {k:28s} {v:.5%}")


if __name__ == "__main__":
    main()
