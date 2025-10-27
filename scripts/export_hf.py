"""Export a Bonsai model to the Hugging Face Qwen3 layout (plan section 18), loadable by transformers / mlx-lm.

    python -m scripts.export_hf --run bonsai-0.1 --source final            # -> export/bonsai-0.1/
    python -m scripts.export_hf --run bonsai-0.1 --source base             # -> export/bonsai-0.1-base/

Writes config.json (model_type qwen3, _name_or_path bonsai-0.1, Qwen3ForCausalLM), model.safetensors (bf16, HF
names with the "model." prefix), generation_config.json, the Qwen3 tokenizer files, bonsai_vocab.json (the subset
S) and README.md (model card) into a temporary dir, loads it back with mlx-lm and checks it against the student,
and only then moves it into place: a failed check (ExportError, exit 1) leaves nothing behind.

Vocabulary. The student has N rows (subset S of Qwen3's ids); the export has Qwen3's 151,936 rows, so the unchanged
tokenizer works:
  * embed_tokens: row s2t[s] = student row s; rows outside S are 0. Bonsai's own pipeline never feeds an
    out-of-S id (it spells such tokens with in-S pieces, bonsai.tokenizer.VocabMap.split); the HF tokenizer
    produces them for ~0.3% of tokens, which the model then sees as zero vectors. For exact student behaviour
    feed `tok.to_teacher(tok.encode(text))` (in-S pieces, as Qwen ids).
  * lm_head (untied): row s2t[s] = student row s (the student is tied), so logits over S are exactly the
    student's. A linear head cannot give a row a negative logit for every hidden state (h and -h are both
    possible after RMSNorm), so the rows outside S share one vector w_out fitted on calibration hidden states:
    the final hidden states h of tok/val rows (else built-in text) and of chat-formatted conversations (the
    Qwen3 template with BOS, thinking on / off, the python tool, each followed by the model's own greedy reply,
    so the positions after "assistant\n" and "<think>\n" are covered). With a = log(n_out) + h.w_out - lse_S(h)
    the log-odds of the total out-of-S probability, w_out minimises a smooth maximum of a over the positions
    (logsumexp + mean softplus; convex, projected Adam from the mean in-S row) with |w_out| <= k x the largest
    in-S row norm, k = 1, 2, 4, 8: the smallest k whose held-out mass is <= --oov-mass (1e-6) everywhere wins
    (else the best one). The norm bound keeps the row's bf16 rounding and its worst-case logit on the scale of the
    in-S rows (a mean-direction row pushed to norm 1e4 failed on a trained Qwen3-architecture model: 3.5% of its
    positions, all BOS ones among them, point against the mean, and got ~100% out-of-S mass). A quarter of the
    rows of each source are held out; when the held-out maximum, or the mlx-lm check's maximum
    on raw and chat text, exceeds --max-oov-mass (1e-4), the export fails. The achieved masses go into
    bonsai_vocab.json and the model card; generation_config.json also lists the out-of-S ids as
    `suppress_tokens` (HF generate never samples them). For exact student probabilities renormalise over S with
    bonsai_vocab.json's s2t.
  * prompts: every Bonsai document / conversation starts with <|endoftext|> (Qwen3 has no BOS); the exported
    chat template prepends it (--keep-template keeps Qwen3's template unchanged), raw-text users must too.
"""

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from bonsai import chat
from bonsai.common import DATA_ROOT, LOG_DIR, TOK_DIR, TOKENIZER_DIR, log, write_json
from bonsai.tokenizer import BOS, CONTROL_IDS, TEACHER_VOCAB, TOKENIZER_FILES

CALIB_TEXT = [
    "The water cycle describes how water evaporates from the surface of the earth, rises into the atmosphere, "
    "cools and condenses into clouds, and falls again as rain or snow.",
    "To find the area of a rectangle, multiply its length by its width. A rectangle that is 5 meters long and 3 "
    "meters wide has an area of 15 square meters.",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
    "In 1492, Columbus sailed across the Atlantic Ocean. His voyages opened the way for European exploration.",
    "Photosynthesis is the process plants use to turn light, water and carbon dioxide into sugar and oxygen.",
    "Sarah has 3 boxes of 12 apples and gives away 10. How many apples are left? She has 36 - 10 = 26 apples.",
]
# chat calibration: (user message, enable_thinking, python tool offered); each gets the model's own greedy reply
CALIB_CHAT = [
    ("What is the capital of France?", True, False), ("Write a Python function that adds two numbers.", False, False),
    ("Hi!", True, False), ("Why do leaves change color in autumn? /no_think", True, False),
    ("A shop sells pens at 3 dollars each. How much do 7 pens cost?", True, True), ("Who are you?", False, False),
    ("Give me three tips for better sleep.", True, False), ("What is 17 * 23?", False, True),
    ("Explain what a prime number is. /think", True, False), ("Translate 'good morning' into French.", False, False),
    ("Is 91 a prime number? Check it with code.", True, True), ("Tell me a short story about a robot.", True, False),
]
REPLY_TOKENS = 64


class ExportError(RuntimeError):
    """The export would put too much probability on tokens outside the subset (nothing is written)."""


def hf_config(cfg, name, dtype="bfloat16"):
    return {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "_name_or_path": name,
            "attention_bias": False, "attention_dropout": 0.0, "bos_token_id": CONTROL_IDS[BOS],
            "eos_token_id": CONTROL_IDS["<|im_end|>"], "head_dim": cfg.head_dim, "hidden_act": "silu",
            "hidden_size": cfg.d_model, "initializer_range": cfg.init_std, "intermediate_size": cfg.mlp_hidden,
            "max_position_embeddings": cfg.seq_len, "max_window_layers": cfg.n_layer,
            "num_attention_heads": cfg.n_head, "num_hidden_layers": cfg.n_layer, "num_key_value_heads": cfg.n_kv_head,
            "rms_norm_eps": cfg.norm_eps, "rope_scaling": None, "rope_theta": cfg.rope_theta, "sliding_window": None,
            "tie_word_embeddings": False, "torch_dtype": dtype, "transformers_version": "4.51.0", "use_cache": True,
            "use_sliding_window": False, "vocab_size": TEACHER_VOCAB}


def calibration_ids(tok, calib_dir, max_tokens):
    """Rows of student ids (BOS first) for the out-of-S calibration."""
    if calib_dir is not None and (Path(calib_dir) / "manifest.json").exists():
        from bonsai.dataloader import val_batches
        return [r for x, _ in val_batches(calib_dir, 8, 256, max_tokens, tok.vmap.sha()) for r in x], str(calib_dir)
    seqs = [tok.encode(t, bos=True) for t in CALIB_TEXT]
    return seqs, "built-in text"


def chat_ids(tok, messages, tools=None, enable_thinking=True, add_generation_prompt=True):
    """Student ids of BOS + a conversation in the Qwen3 template (as Bonsai sees it)."""
    tids, _ = chat.render(tok.qwen, messages, tools=tools, add_generation_prompt=add_generation_prompt,
                          enable_thinking=enable_thinking)
    return [tok.bos_id] + tok.from_teacher(tids).tolist()


def chat_calibration(model, tok, reply_tokens=REPLY_TOKENS):
    """CALIB_CHAT prompts, each followed by the model's own greedy reply, plus one full tool conversation."""
    from bonsai.train import greedy
    stop = {tok.special("<|im_end|>"), tok.bos_id}
    seqs = []
    for q, think, tools in CALIB_CHAT:
        ids = chat_ids(tok, [{"role": "user", "content": q}], chat.TOOLS if tools else None, think)
        seqs.append(ids + greedy(model, ids, reply_tokens, stop))
    seqs.append(chat_ids(tok, chat.tool_trajectory("What is 12 squared?", "print(12 ** 2)", "144", "12 squared is 144.",
                                                   reasoning="I can compute this with Python."), chat.TOOLS,
                         add_generation_prompt=False))
    return seqs


def hidden_states(model, seqs, head_bf16, batch=8):
    """(H, lse_S, sequence index) over every position of `seqs` (equal-length neighbours run as one batch)."""
    H, lse, owner = [], [], []
    i = 0
    while i < len(seqs):
        j = i + 1
        while j < min(len(seqs), i + batch) and len(seqs[j]) == len(seqs[i]):
            j += 1
        x = np.stack([np.asarray(r) for r in seqs[i:j]])
        h = model.hidden(mx.array(x)).astype(mx.float32).reshape(-1, model.cfg.d_model)
        lse.append(np.array(mx.logsumexp(h @ head_bf16.T, axis=-1)))
        H.append(np.array(h))
        owner.append(np.repeat(np.arange(i, j), x.shape[1]))
        i = j
    return np.concatenate(H), np.concatenate(lse), np.concatenate(owner)


def fit_out_row(H, lse, c, w0, radius, steps=500):
    """argmin over |w| <= radius of logsumexp(a) + mean(softplus(a)), a = c + H w - lse (projected Adam)."""
    Hm, Lm = mx.array(H), mx.array(lse)
    p = {"w": mx.array(w0)}

    def objective(p):
        a = c + Hm @ p["w"] - Lm
        return mx.logsumexp(a) + mx.mean(mx.logaddexp(0.0, a))

    lr = 0.2 * radius / math.sqrt(H.shape[1])
    opt = optim.Adam(learning_rate=optim.cosine_decay(lr, steps, lr * 0.01))
    grad = mx.value_and_grad(objective)
    for _ in range(steps):
        _, g = grad(p)
        opt.update(p, g)
        p["w"] = p["w"] * mx.minimum(1.0, radius / (mx.linalg.norm(p["w"]) + 1e-12))
        mx.eval(p, opt.state)
    return np.array(p["w"].astype(mx.bfloat16).astype(mx.float32))  # as exported


def _mass(H, lse, c, w):
    return np.exp(-np.logaddexp(0.0, -(c + H @ w - lse).astype(np.float64)))  # sigmoid(a) = out-of-S mass


def _summary(ms):
    return {"positions": int(len(ms)), "mass_max": float(ms.max()), "mass_mean": float(ms.mean()),
            "mass_p999": float(np.quantile(ms, 0.999))}


def out_of_subset_rows(model, tok, sources, n_out, target=1e-6, radii=(1, 2, 4, 8), holdout=4):
    """(w_out row, stats) from {source: [id sequences]}; every `holdout`-th sequence of each source is held out
    (see the module docstring)."""
    head = (model.embed_tokens.weight if model.cfg.tie_embeddings else model.lm_head.weight).astype(mx.float32)
    head_bf16 = head.astype(mx.bfloat16).astype(mx.float32)  # the exported rows are bf16
    max_norm = float(mx.linalg.norm(head_bf16, axis=1).max())
    c = math.log(n_out)
    data = {k: hidden_states(model, seqs, head_bf16) for k, seqs in sources.items()}
    fit = {k: (o % holdout != holdout - 1) for k, (_, _, o) in data.items()}
    H = np.concatenate([d[0][fit[k]] for k, d in data.items()])
    L = np.concatenate([d[1][fit[k]] for k, d in data.items()])
    w = np.array(head.mean(axis=0))
    tried, best = [], None
    for k in radii:
        w = fit_out_row(H, L, c, w, k * max_norm)
        held = np.concatenate([_mass(d[0][~fit[s]], d[1][~fit[s]], c, w) for s, d in data.items()])
        tried.append({"norm_mult": k, "held_out_mass_max": float(held.max())})
        if best is None or held.max() < best[1]:
            best = (w, float(held.max()), k)
        if held.max() <= target:
            break
    w, _, k = best
    stats = {"method": "fitted shared row", "norm_mult": k, "w_norm": float(np.linalg.norm(w)),
             "max_in_subset_row_norm": max_norm, "target_mass": target, "tried": tried, "sources": {}}
    for s, (Hs, Ls, o) in data.items():
        stats["sources"][s] = {"fit": _summary(_mass(Hs[fit[s]], Ls[fit[s]], c, w)),
                               "held_out": _summary(_mass(Hs[~fit[s]], Ls[~fit[s]], c, w)) if (~fit[s]).any() else None}
    held = [v["held_out"] for v in stats["sources"].values() if v["held_out"]]
    n = sum(h["positions"] for h in held)
    stats["held_out"] = {"positions": n, "mass_max": max(h["mass_max"] for h in held),
                         "mass_mean": sum(h["mass_mean"] * h["positions"] for h in held) / n}
    return w, stats


def export(model, tok, meta, out_dir, name, calib_dir=None, calib_tokens=2 ** 16, oov_mass=1e-6, keep_template=False,
           verify=True, max_oov_mass=1e-4):
    """Write the export to `out_dir` (replacing it) and return the report; ExportError (nothing written) when the
    held-out or the mlx-lm-checked out-of-S mass exceeds `max_oov_mass`."""
    out = Path(out_dir)
    cfg, s2t = model.cfg, tok.vmap.s2t
    inside = np.zeros(TEACHER_VOCAB, dtype=bool)
    inside[s2t] = True
    n_out = int((~inside).sum())
    rows, calib_src = calibration_ids(tok, calib_dir, calib_tokens)
    calib_src = f"{calib_src} + {len(CALIB_CHAT) + 1} chat conversations"
    w_out, oov = out_of_subset_rows(model, tok, {"text": rows, "chat": chat_calibration(model, tok)}, n_out, oov_mass)
    log(f"out-of-subset rows: {json.dumps(oov)}")
    if oov["held_out"]["mass_max"] > max_oov_mass:
        raise ExportError(f"out-of-subset probability up to {oov['held_out']['mass_max']:.2e} on held-out calibration "
                          f"positions (> --max-oov-mass {max_oov_mass:g}); nothing written. Fit: {oov['tried']}")
    tmp = out.parent / f".tmp_{out.name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        report = _write(model, tok, meta, tmp, name, w_out, oov, calib_src, inside, keep_template, verify)
        ver = report.get("verify")
        if ver and ver["out_of_subset_mass_max"] > max_oov_mass:
            raise ExportError(f"the exported model (mlx-lm) puts up to {ver['out_of_subset_mass_max']:.2e} "
                              f"probability outside the subset (> --max-oov-mass {max_oov_mass:g}); nothing written")
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    return report


def _write(model, tok, meta, out, name, w_out, oov, calib_src, inside, keep_template, verify):
    cfg, s2t = model.cfg, tok.vmap.s2t
    n_out = int((~inside).sum())
    weights = {}
    for k, v in tree_flatten(model.parameters()):
        if k == "embed_tokens.weight":
            emb = np.zeros((TEACHER_VOCAB, cfg.d_model), dtype=np.float32)
            emb[s2t] = np.array(v)
            weights["model.embed_tokens.weight"] = mx.array(emb).astype(mx.bfloat16)
        elif k == "lm_head.weight":
            continue
        else:
            weights[f"model.{k}"] = v.astype(mx.bfloat16)
    head = np.repeat(w_out[None].astype(np.float32), TEACHER_VOCAB, axis=0)
    src = model.embed_tokens.weight if cfg.tie_embeddings else model.lm_head.weight
    head[s2t] = np.array(src)
    weights["lm_head.weight"] = mx.array(head).astype(mx.bfloat16)
    mx.save_safetensors(str(out / "model.safetensors"), weights, metadata={"format": "mlx"})

    write_json(out / "config.json", hf_config(cfg, name))
    (out / "generation_config.json").write_text(json.dumps(  # compact: suppress_tokens lists every out-of-S id
        {"bos_token_id": CONTROL_IDS[BOS], "eos_token_id": [CONTROL_IDS["<|im_end|>"], CONTROL_IDS[BOS]],
         "pad_token_id": CONTROL_IDS[BOS], "do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
         "suppress_tokens": np.flatnonzero(~inside).tolist(), "transformers_version": "4.51.0"}))
    for f in TOKENIZER_FILES:
        shutil.copy(TOKENIZER_DIR / f, out / f)
    if not keep_template:
        tc = json.loads((out / "tokenizer_config.json").read_text())
        tc["chat_template"] = "{{- '" + BOS + "' }}" + tc["chat_template"]
        (out / "tokenizer_config.json").write_text(json.dumps(tc, indent=2, ensure_ascii=False))
    vocab = {"vocab_sha": tok.vmap.sha(), "n_subset": int(tok.vocab_size), "s2t": s2t.tolist(),
             "out_of_subset_rows": {**oov, "calibration": calib_src, "n_rows": n_out}, "bos": BOS,
             "bos_in_template": not keep_template}
    write_json(out / "bonsai_vocab.json", vocab)
    report = {"name": name, "oov": oov, "calibration": calib_src}
    if verify:
        report["verify"] = verify_export(out, model, tok)
    (out / "README.md").write_text(model_card(name, cfg, model.num_params(), meta, report))
    return report


def verify_export(out, model, tok, text="The capital of France is Paris. Water boils at 100 degrees Celsius."):
    """Load the export with mlx-lm and compare its logits over S with the student's, on raw text and on a chat
    prompt (BOS + template, thinking on and off); out-of-S mass from the export's full-vocabulary softmax."""
    from mlx_lm.utils import load_model as mlx_lm_load
    hf, _ = mlx_lm_load(Path(out))
    q = [{"role": "user", "content": "What is the capital of France?"}]
    diffs, agree, mass, n = [], [], [], 0
    for ids in (tok.encode(text, bos=True), chat_ids(tok, q, enable_thinking=True),
                chat_ids(tok, q, enable_thinking=False)):
        student = np.array(model(mx.array([ids]))[0].astype(mx.float32))
        full = np.array(hf(mx.array([tok.to_teacher(ids).tolist()]))[0].astype(mx.float32))
        sub = full[:, tok.vmap.s2t]
        lse_full = np.logaddexp.reduce(full.astype(np.float64), axis=-1)
        lse_sub = np.logaddexp.reduce(sub.astype(np.float64), axis=-1)
        diffs.append(np.abs(sub - student).max())
        agree.append((sub.argmax(-1) == student.argmax(-1)).sum())
        mass.append((-np.expm1(lse_sub - lse_full)).max())
        n += len(ids)
    res = {"positions": n, "max_abs_logit_diff_over_S": float(max(diffs)), "argmax_agree": float(sum(agree) / n),
           "out_of_subset_mass_max": float(max(mass))}
    log(f"export check (mlx-lm): {res}")
    return res


def model_card(name, cfg, n, meta, report):
    oov, ver = report["oov"], report.get("verify", {})
    held = oov["held_out"]
    evals = ""
    run = meta.get("run")
    for f in sorted((LOG_DIR / run).glob("eval_*.json")) if run else []:
        evals += f"\n- `{f.name}`: see logs (not copied)"
    return f"""---
library_name: transformers
pipeline_tag: text-generation
language: [en]
tags: [qwen3, bonsai, from-scratch, distillation, mlx]
---

# {name}

A {n['total'] / 1e6:.1f}M-parameter English language model ({n['non_embedding'] / 1e6:.1f}M non-embedding),
pretrained from scratch with MLX on one MacBook Pro (M4 Pro, 48 GB).

* **Architecture**: Qwen3-0.6B's structure scaled down ({cfg.n_layer} layers x {cfg.d_model}, {cfg.n_head} query /
  {cfg.n_kv_head} KV heads, head_dim {cfg.head_dim}, SwiGLU {cfg.mlp_hidden}, QK-Norm, learnable RMSNorm, RoPE
  theta {cfg.rope_theta:g}, {cfg.seq_len}-token context). `model_type` is `qwen3`.
* **Tokenizer and chat template**: Qwen3's (Apache-2.0), unchanged except that the chat template prepends
  `<|endoftext|>`: Bonsai sees it at position 0 of every document and conversation (Qwen3 has no BOS). Prepend it
  to raw-text prompts too. Thinking (`<think>`, budget 128 tokens), `/think` / `/no_think` and the Python tool
  protocol (`<tool_call>` / `<tool_response>`) follow Qwen3.
* **Training**: pretrained from scratch (no Qwen weights): formal-language pre-pretraining (S0), general text (S1),
  reasoning anneal (S2); then off-policy distillation from Qwen3-1.7B (teacher generations, SFT) and on-policy
  distillation from Qwen3-0.6B (reverse KL on the student's samples).
* **Vocabulary**: the model was trained on a {meta.get('model_config', {}).get('vocab_size', '?')}-token subset of
  Qwen3's vocabulary (`bonsai_vocab.json`: `s2t`). Rows outside it are zero in the embedding; in the untied
  lm_head they share one row fitted so that their total probability stays small (calibrated on
  {report['calibration']}; on {held['positions']} held-out positions: max {held['mass_max']:.2e}, mean
  {held['mass_mean']:.2e}), and `generation_config.json` lists them in `suppress_tokens`.
  Logits over the subset are exactly the trained model's{'' if not ver else f" (mlx-lm check: max |diff| {ver['max_abs_logit_diff_over_S']:.3g}, argmax agreement {ver['argmax_agree']:.0%}, out-of-subset probability at most {ver['out_of_subset_mass_max']:.1e})"}.
  Text with tokens outside the subset (~0.3% of tokens) is best fed as in-subset pieces (see `bonsai_vocab.json`).
* **Limits**: about 1e18 training FLOPs; English only; small-model knowledge and reasoning. It is Bonsai-0.1, not
  Qwen and not ChatGPT.

Source: `{meta.get('path', '?')}` (vocab_sha `{meta.get('vocab_sha')}`).{evals}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="bonsai-0.1")
    ap.add_argument("--source", default="final", help="base | latest | sft | onpolicy | final | a path")
    ap.add_argument("--name", default="", help="_name_or_path (default: the run, + -<source> unless final)")
    ap.add_argument("--out", default="", help="default export/<name>")
    ap.add_argument("--calib-dir", default="val", help="calibration shards for the out-of-subset rows (tok/<dir>)")
    ap.add_argument("--calib-tokens", type=int, default=2 ** 16)
    ap.add_argument("--oov-mass", type=float, default=1e-6, help="target out-of-subset mass (held-out max)")
    ap.add_argument("--max-oov-mass", type=float, default=1e-4, help="fail above this out-of-subset mass")
    ap.add_argument("--keep-template", action="store_true", help="do not prepend <|endoftext|> in the chat template")
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    args = ap.parse_args()
    from bonsai.load import load_model
    model, tok, meta = load_model(args.run, args.source)
    meta.setdefault("run", args.run)
    name = args.name or (args.run if args.source == "final" else f"{args.run}-{Path(args.source).stem}")
    out = Path(args.out) if args.out else DATA_ROOT / "export" / name
    calib = Path(args.calib_dir) if Path(args.calib_dir).is_dir() else TOK_DIR / args.calib_dir
    try:
        rep = export(model, tok, meta, out, name, calib, args.calib_tokens, args.oov_mass, args.keep_template,
                     args.verify, args.max_oov_mass)
    except ExportError as e:
        raise SystemExit(f"export of {name} FAILED: {e}")
    log(f"exported {name} -> {out}: {json.dumps(rep, default=str)}")


if __name__ == "__main__":
    main()
