"""On-policy distillation from Qwen3-0.6B over the student's vocabulary subset S (plan sections 12, 14, 15, 16).

One step (plan 12):
  1. `batch` fresh prompts from data/prompts_onpolicy.jsonl (the section 11 pool resampled, identity prompts
     removed; the teacher's off-policy answers are never used), a seeded permutation consumed in order (resume =
     the step index). Each prompt gets a random mode with the off-policy data's signals: thinking = no switch or
     " /think" (30%), non-thinking = " /no_think" or the enable_thinking=False prefill (50 / 50); math prompts get
     the python tool (`math_tools` of them; prompts marked tools always).
  2. The student samples with its current weights, no gradients (bonsai.engine.chat_batch): thinking budget 128
     (then chat.think_stop(tools) + </think> is forced), answers <= 300 tokens, at most 2 tool calls run in
     bonsai.sandbox, Qwen3's sampling preset of the mode (thinking T 0.6 / top-p 0.95, non-thinking T 0.7 / top-p
     0.8, top-k 20, presence penalty 1.5).
  3. Teacher context and vocabulary adaptation: the teacher scores in the context it generated the off-policy data
     in (plan 11), which the SFT student imitates: the teacher-only system prompt of the prompt's kind
     (`system_prompts`, those of configs/offpolicy.yaml; math_tool when the tool is offered) before the student's
     messages, rendered like bonsai.teacher's generation prompts (its build_job + chat.render, generation prompt,
     the empty think block for enable_thinking=False), then s2t of the student's completion ids (generated +
     forced + tool continuations). The completion is identical, so there is no span alignment: student position
     t is teacher position t + offset, offset = teacher prompt length - student prompt length (BOS included),
     constant per sequence (`teacher_sequences`). The teacher (bf16) prefills them and
     Teacher.iter_subset_logprobs gives its fp32 log-softmax over S (the s2t columns, mass outside S dropped and
     renormalised). Without the system prompt Qwen3 rejects the SFT formats (<tool_call>, "Answer:") by 7-20
     nats a token.
  4. Loss = exact reverse KL(pi_student || pi_teacher) summed over S, temperature 1, at the positions whose token
     the student sampled and that are neither forced (budget stop text, tool continuations), nor <tool_response>
     content, nor from an out-of-subset split, nor a stand-alone non-ASCII byte token (such bytes spell characters
     outside S, where the renormalised teacher distribution is not comparable); the mean over all such positions
     of the step. Student log-probs: the head in fp32 on the gathered positions, fp32 log-softmax.
  5. One update per step: the off-policy stage's optimizer (meta["optim"] of sft_final: Muon + AdamW, or Qwen-style
     AdamW) at `lr_frac` (0.5) x its LRs, short warmup, clip 1.0. The sequences are processed in micro-batches of
     <= `micro_tokens` scored positions (teacher scoring + the student's backward, gradients accumulated).
Every `eval_every` (25) steps: val KL (the on-policy KL on a fixed held-out subset with a fixed sampling seed),
the GSM8K-style monitor (`gsm8k_n` of the held-out math prompts with a numeric answer, data/prompts_heldout.jsonl,
non-thinking + tool, greedy; evals.eval_gsm8k_tool on these items, never the GSM8K test set, which stays for
plan 15's reported number), the 20-conversation format check (evals.eval_mode_switch on its dev prompts, not the
frozen eval prompts: think block well formed, ends with <|im_end|>, mode obeyed; answers up to
`format_max_answer_tokens` (400, the SFT cap), and a reply cut by that cap is not a format error), the mean
training KL. Stop rules (plan 12): no val-KL improvement for `patience` (50) steps, the monitor below the start by
more than `gsm8k_tol`, format errors above the start by more than `format_tol`; a monitor that fails at step 0
stops the run at once (the rules would never fire). The final weights are the best healthy eval (lowest val KL
among the evals where neither the monitor nor the format check regressed; step 0 is the SFT model).

Resume (`OnPolicy.run(resume=...)`): "auto" continues the newest checkpoint, but after a health stop or with a
changed `lr_frac` (plan 16 collapse: halve the LR, rerun) it goes back to the best healthy checkpoint (best.
safetensors with a fresh optimizer when that step has no full checkpoint, e.g. step 0); "latest", "best" or a
checkpoint dir force the choice. The stop reason is always cleared and the evals after the resumed step dropped;
later checkpoints are moved to onpolicy/abandoned_<time>/ so rotation and `latest` never mix trajectories. A fresh
run refuses an onpolicy/ dir that holds an earlier attempt (overwrite=True moves it aside, never deletes it).
SIGINT / SIGTERM / SIGHUP save a checkpoint after the current step.

Outputs: ckpt/<run>/onpolicy_final.safetensors (meta: model_config, vocab_sha, optim, base_optim, lr_scale, ...);
ckpt/<run>/onpolicy/ (full checkpoints every `ckpt_every` steps, newest `keep`, + best.safetensors);
logs/<run>/onpolicy.jsonl (a record per step, per eval and per resume), onpolicy.log, onpolicy_samples.jsonl,
onpolicy_summary.json.

`measure_kl` is plan 15's "KL to the teacher": the mean per-token KL(student || teacher) on the student's own
samples of held-out prompts, modes fixed per prompt id (scripts/teacher_kl.py, for base, sft and onpolicy).
"""

import hashlib
import json
import math
import random
import signal
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map

from bonsai import chat as C
from bonsai.common import CODE_ROOT, DATA_ROOT, iter_jsonl, load_yaml, log, write_json
from bonsai.engine import GREEDY, PRESETS, Sampling, chat_batch

DEFAULTS = {
    "teacher": "Qwen3-0.6B", "source": "sft", "system_prompts": "configs/offpolicy.yaml",
    "prompts": "data/prompts_onpolicy.jsonl", "heldout": "data/prompts_heldout.jsonl",
    "steps": 150, "batch": 256, "seed": 2026,
    "think_frac": 0.5, "think_soft_frac": 0.3, "no_think_soft_frac": 0.5, "math_tools": 1.0,
    "think_budget": C.THINK_BUDGET, "max_answer_tokens": 300, "max_tool_calls": 2,
    "sampling": "qwen", "presence_penalty": None, "sample_batch": 64,
    "micro_tokens": 2048, "micro_seqs": 8, "teacher_chunk": 4,
    "lr_frac": 0.5, "optimizer": "same", "warmup": 10, "clip": 1.0,
    "eval_every": 25, "ckpt_every": 25, "keep": 3,
    "val_prompts": 128, "gsm8k_n": 200, "format_n": 20, "format_max_answer_tokens": 400, "eval_batch": 32,
    "patience": 50, "kl_min_delta": 0.0, "gsm8k_tol": 0.03, "format_tol": 1, "final": "best",
    "memory_limit_gb": 40.0,
}


def stable_frac(key, salt=""):
    return int(hashlib.md5((salt + "\x00" + str(key)).encode()).hexdigest()[:8], 16) / 2 ** 32


def aside(path, prefix):
    """A new path next to `path` named <prefix><time>[_k] that does not exist yet."""
    base = Path(path).with_name(prefix + time.strftime("%Y%m%d-%H%M%S"))
    out, k = base, 1
    while out.exists():
        out, k = base.with_name(f"{base.name}_{k}"), k + 1
    return out


def load_records(path, n=0, seed=0):
    """Prompt records {"id", "kind", "messages", "tools", ...}; n > 0: a fixed random subset."""
    p = Path(path)
    recs = list(iter_jsonl(p if p.is_absolute() or p.exists() else DATA_ROOT / p))
    return random.Random(seed).sample(recs, n) if n and n < len(recs) else recs


def load_system_prompts(spec):
    """The teacher-only system prompts by kind: a dict, or a YAML file whose `system_prompts` they are (default
    configs/offpolicy.yaml, the off-policy generation's); None / {} = no system prompt."""
    if not spec or isinstance(spec, dict):
        return dict(spec or {})
    p = Path(spec)
    p = p if p.is_absolute() or p.exists() else CODE_ROOT / p
    sp = (load_yaml(p) or {}).get("system_prompts")
    if not sp:
        raise ValueError(f"{p}: no system_prompts")
    return sp


def math_items(records, n=0, seed=0):
    """GSM8K-style monitor items {"question", "answer"}: the single-turn math prompts with a numeric answer;
    n > 0: a fixed random subset."""
    from bonsai.sandbox import parse_number
    out = [{"question": r["messages"][-1]["content"], "answer": parse_number(r["answer"]), "id": r.get("id")}
           for r in records if r.get("kind") == "math" and len(r["messages"]) == 1 and
           r["messages"][0]["role"] == "user" and parse_number(r.get("answer")) is not None]
    return random.Random(seed).sample(out, n) if n and n < len(out) else out


# ------------------------------------------------------------------------------------------ prompts -> chat items


def sampling_for(cfg, mode):
    s = cfg.get("sampling", "qwen")
    s = PRESETS[mode] if s == "qwen" else GREEDY if s == "greedy" else Sampling(**s.get(mode, s))
    pp = cfg.get("presence_penalty")
    return s if pp is None else Sampling(s.temperature, s.top_p, s.top_k, float(pp))


def make_item(rec, cfg, u):
    """chat_batch item of a prompt record; u = three uniforms deciding the mode, its signal and the tool."""
    mode = "think" if u[0] < cfg["think_frac"] else "no_think"
    soft = u[1] < cfg["think_soft_frac" if mode == "think" else "no_think_soft_frac"]
    signal_ = "soft" if soft else ("default" if mode == "think" else "prefill")
    msgs = [{"role": m["role"], "content": m["content"]} for m in rec["messages"]]
    if soft:
        msgs[-1] = {**msgs[-1], "content": C.with_soft_switch(msgs[-1]["content"], mode)}
    tools = rec.get("kind") == "math" and (bool(rec.get("tools")) or u[2] < cfg["math_tools"])
    return {"messages": msgs, "enable_thinking": signal_ != "prefill", "tools": C.TOOLS if tools else None,
            "sampling": sampling_for(cfg, mode), "id": rec.get("id"), "kind": rec.get("kind"), "mode": mode,
            "signal": signal_}


def make_items(records, cfg, rng=None):
    """rng: numpy Generator (training: random per prompt); None: stable per prompt id (held-out KL: the same modes
    at every stage)."""
    return [make_item(r, cfg, rng.random(3) if rng is not None else
                      [stable_frac(r.get("id"), s) for s in ("mode", "signal", "tools")]) for r in records]


class PromptOrder:
    """Record indices in a seeded permutation, `batch` per step (step s takes s*batch ...); a new permutation per
    pass over the pool (epoch > 0 means prompts repeat)."""

    def __init__(self, n, batch, seed):
        self.n, self.batch, self.seed, self._perms = n, batch, seed, {}

    def __call__(self, step):
        out, epochs = [], set()
        for k in range(step * self.batch, (step + 1) * self.batch):
            e, i = divmod(k, self.n)
            if e not in self._perms:
                self._perms = {e: np.random.default_rng([self.seed, e]).permutation(self.n)}
            out.append(int(self._perms[e][i]))
            epochs.add(e)
        return out, max(epochs)


# ------------------------------------------------------------------------------------------ masks and the KL


def nonascii_byte_ids(tok):
    """Bool over student ids: the stand-alone byte tokens 0x80..0xFF."""
    bad = np.zeros(tok.vocab_size, dtype=bool)
    bad[tok.vmap.byte_student[0x80:]] = True
    return bad


def kl_mask(r, bad):
    """Bool per position of r.ids: True = the token there is scored (its distribution is logits row i - 1)."""
    ids = np.asarray(r.ids)
    m = r.sampled & ~r.forced & ~r.tool_output & ~r.oov & ~bad[ids]
    m[:1] = False
    return m


def mask_counts(results, masks, bad):
    c = Counter()
    for r, m in zip(results, masks):
        ids = np.asarray(r.ids)
        c.update({"positions": len(ids), "generated": len(ids) - r.prompt_len, "sampled": int(r.sampled.sum()),
                  "scored": int(m.sum()), "forced": int(r.forced.sum()), "tool_output": int(r.tool_output.sum()),
                  "oov": int(r.oov.sum()), "sampled_oov": int((r.sampled & r.oov).sum()),
                  "sampled_nonascii_byte": int((r.sampled & bad[ids]).sum())})
    return dict(c)


def reverse_kl(logp_s, logp_t):
    """KL(p_s || p_t) per row, exact over the last axis (both are log-probabilities over S)."""
    return (mx.exp(logp_s) * (logp_s - logp_t)).sum(axis=-1)


def student_logprobs(model, x, rows, cols):
    """fp32 log-softmax of the student at positions (rows, cols) of x; the head runs in fp32 on those rows only."""
    h = model.hidden(x)[rows, cols].astype(mx.float32)
    w = model.embed_tokens.weight if model.cfg.tie_embeddings else model.lm_head.weight
    z = h @ w.astype(mx.float32).T
    return z - mx.logsumexp(z, axis=-1, keepdims=True)


def _kl_sum(model, x, rows, cols, logp_t, scale):
    kl = reverse_kl(student_logprobs(model, x, rows, cols), logp_t)
    return kl.sum() * scale, kl


def teacher_prompt(qwen, item, system_prompts):
    """Teacher ids of an item's prompt in the teacher's generation context: bonsai.teacher.build_job's system
    prompt for its kind (math_tool when the tool is offered) before the item's messages (soft switch included),
    rendered as Teacher.chat_batch renders jobs, up to the generation prompt the student's prompt also ends with
    (+ the empty think block for enable_thinking=False; the student samples "<think>" itself otherwise)."""
    from bonsai.teacher import build_job
    rec = {"id": item.get("id"), "kind": item.get("kind") or "chat", "messages": item["messages"],
           "tools": bool(item.get("tools"))}
    job = build_job(rec, item.get("mode", "think"), system_prompts, signal=None)
    ids, _ = C.render(qwen, job.messages, tools=C.TOOLS if job.tools else None, add_generation_prompt=True,
                      enable_thinking=item.get("enable_thinking", True))
    return ids


def teacher_sequences(tok, items, results, system_prompts):
    """Per result (teacher ids, offset, start): teacher_prompt + s2t of the student's completion ids; student
    position t >= start (= its prompt length) is teacher position t + offset."""
    out = []
    for it, r in zip(items, results):
        tp = teacher_prompt(tok.qwen, it, system_prompts)
        comp = tok.vmap.s2t[np.asarray(r.ids[r.prompt_len:], dtype=np.int64)]
        out.append((np.concatenate([np.asarray(tp, dtype=np.int64), comp]).astype(np.int32),
                    len(tp) - r.prompt_len, r.prompt_len))
    return out


def teacher_rows(tok, seqs, targets, tseqs=None):
    """(teacher ids (B, T) right-padded, rows, cols): the teacher's input for the student sequences and the
    positions whose distributions predict targets[i]; tseqs[i] from teacher_sequences, None = s2t of the student
    ids (offset 0: the teacher sees exactly the student's context)."""
    s2t = tok.vmap.s2t
    if tseqs is None:
        tseqs = [(s2t[np.asarray(s, dtype=np.int64)], 0, 0) for s in seqs]
    x = np.full((len(seqs), max(len(t) for t, _, _ in tseqs) - 1), s2t[tok.bos_id], dtype=np.int32)
    rows, cols = [], []
    for r, ((tids, off, start), s, tg) in enumerate(zip(tseqs, seqs, targets)):
        tg = np.asarray(tg, dtype=np.int64)
        assert len(tids) == len(s) + off and (tg >= max(start, 1)).all() and (tg < len(s)).all(), \
            "scored positions must lie in the completion shared with the teacher"
        x[r, :len(tids) - 1] = tids[:-1]
        rows.append(np.full(len(tg), r, dtype=np.int32))
        cols.append((tg + off - 1).astype(np.int32))
    return x, np.concatenate(rows), np.concatenate(cols)


def teacher_logprobs(teacher, vmap, x, rows, cols, chunk):
    """The teacher's fp32 log-softmax over S at (rows, cols) of the teacher ids x, (M, N); rows ascending."""
    parts = []
    for a, lp in teacher.iter_subset_logprobs(x, vmap, chunk):
        sel = np.flatnonzero((rows >= a) & (rows < a + lp.shape[0]))
        if len(sel):
            parts.append(lp[mx.array(rows[sel] - a), mx.array(cols[sel])])
            mx.eval(parts[-1])
    return parts[0] if len(parts) == 1 else mx.concatenate(parts)


def micro_groups(lengths, counts, max_tokens, max_seqs):
    """Indices of the sequences with scored positions, by length, in groups of <= max_seqs sequences and
    <= max_tokens scored positions (a longer sequence is a group of its own)."""
    groups, cur, n = [], [], 0
    for i in sorted((i for i in range(len(lengths)) if counts[i]), key=lambda i: lengths[i]):
        if cur and (n + counts[i] > max_tokens or len(cur) >= max_seqs):
            groups.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += counts[i]
    return groups + [cur] if cur else groups


def score(model, teacher, tok, seqs, targets, cfg, grad_scale=None, tseqs=None):
    """Exact reverse KL at the positions targets[i] of seqs[i] (student ids), micro-batched; the teacher scores
    tseqs[i] (teacher_sequences: its own context + the same completion) or, None, s2t of the student ids.

    Returns (per-sequence KL arrays, gradients of grad_scale * sum(KL) accumulated over the micro-batches or None,
    seconds spent in the teacher)."""
    lengths, counts = [len(s) for s in seqs], [len(t) for t in targets]
    out = [np.zeros(0, dtype=np.float32) for _ in seqs]
    grad_fn = nn.value_and_grad(model, _kl_sum) if grad_scale is not None else None
    grads, t_teacher = None, 0.0
    for g in micro_groups(lengths, counts, cfg["micro_tokens"], cfg["micro_seqs"]):
        # right padding (causal: never changes earlier rows) to a multiple of 32: very short query lengths take
        # MLX's SDPA vector kernel, which has no VJP
        T = max(lengths[i] for i in g) - 1
        x = np.full((len(g), -(-T // 32) * 32), tok.bos_id, dtype=np.int32)
        rows, cols = [], []
        for r, i in enumerate(g):
            x[r, :lengths[i] - 1] = seqs[i][:-1]
            rows.append(np.full(counts[i], r, dtype=np.int32))
            cols.append(np.asarray(targets[i], dtype=np.int32) - 1)
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        xt, trows, tcols = teacher_rows(tok, [seqs[i] for i in g], [targets[i] for i in g],
                                        None if tseqs is None else [tseqs[i] for i in g])
        t0 = time.time()
        lp_t = teacher_logprobs(teacher, tok.vmap, xt, trows, tcols, cfg["teacher_chunk"])
        t_teacher += time.time() - t0
        xa, ra, ca = mx.array(x), mx.array(rows), mx.array(cols)
        if grad_fn is None:
            kl = reverse_kl(student_logprobs(model, xa, ra, ca), lp_t)
            mx.eval(kl)
        else:
            (_, kl), gr = grad_fn(model, xa, ra, ca, lp_t, mx.array(grad_scale, dtype=mx.float32))
            grads = gr if grads is None else tree_map(mx.add, grads, gr)
            mx.eval(grads, kl)
        kl, k = np.array(kl), 0
        for i in g:
            out[i] = kl[k:k + counts[i]]
            k += counts[i]
        del lp_t
    return out, grads, t_teacher


def sample(model, tok, items, cfg, seed, executor=None):
    kw = {} if executor is None else {"executor": executor}
    return chat_batch(model, tok, items, think_budget=cfg["think_budget"], max_answer_tokens=cfg["max_answer_tokens"],
                      max_tool_calls=cfg["max_tool_calls"], seed=seed, max_batch=cfg["sample_batch"], **kw)


def _format_ok(tok, items, results):
    """evals.mode_check on the conversations without tools (a tool turn may legitimately end with an empty block);
    a reply cut by the answer cap is not counted."""
    want = {"default": "think", "soft": None, "prefill": "prefilled"}
    try:
        from bonsai.tasks.evals import mode_check
        checks = [mode_check(tok, r, want[it["signal"]] or it["mode"]) for it, r in zip(items, results)
                  if not it["tools"]]
    except Exception:   # a statistic only
        return None
    oks = [ok for ok, why in checks if ok or "length" not in str(why)]
    return sum(oks) / len(oks) if oks else None


def summarize(tok, items, results, kls, masks, bad):
    n = sum(len(k) for k in kls)
    allk = np.concatenate([k for k in kls if len(k)] or [np.zeros(1, dtype=np.float32)])
    by = {}
    for it, k in zip(items, kls):
        for key in (f"mode/{it['mode']}", f"signal/{it['signal']}", f"kind/{it['kind']}{'+tools' if it['tools'] else ''}"):
            g = by.setdefault(key, [0.0, 0])
            g[0] += float(k.sum(dtype=np.float64))
            g[1] += len(k)
    m = max(len(results), 1)
    return {"kl": float(allk.sum(dtype=np.float64)) / max(n, 1), "tokens": n, "seqs": len(results),
            "seq_kl": float(np.mean([k.mean() for k in kls if len(k)])) if n else 0.0,
            "kl_max": float(allk.max()), "kl_p99": float(np.quantile(allk, 0.99)),
            "by": {k: {"kl": a / max(b, 1), "tokens": b} for k, (a, b) in sorted(by.items())},
            "gen_tokens": sum(len(r.ids) - r.prompt_len for r in results) / m,
            "finished": sum(r.finished for r in results) / m,
            "tool_use": sum(r.n_tool_calls > 0 for r in results) / m,
            "hit_budget": sum(r.hit_budget for r in results) / m,
            "think_tokens": sum(r.think_tokens for r in results) / m,
            "format_ok": _format_ok(tok, items, results), "mask": mask_counts(results, masks, bad)}


def conversation_samples(tok, items, results, kls, n):
    out = []
    for it, r, k in list(zip(items, results, kls))[:n]:
        out.append({"id": it["id"], "mode": it["mode"], "signal": it["signal"], "tools": bool(it["tools"]),
                    "prompt": it["messages"][-1]["content"][-400:], "completion": tok.decode(r.completion_ids)[:2000],
                    "kl": float(k.mean()) if len(k) else None, "finished": r.finished,
                    "tool_calls": r.tool_calls})
    return out


def measure_kl(model, tok, teacher, records, cfg, seed=0, executor=None, n_samples=0, system_prompts=None):
    """Mean per-token KL(student || teacher) on the student's own samples of `records` (modes fixed per prompt id,
    sampling seeded; the teacher in its generation context, `system_prompts` or cfg's): plan 15's KL to the
    teacher and the on-policy run's val KL."""
    cfg = {**DEFAULTS, **cfg}
    sp = load_system_prompts(cfg["system_prompts"]) if system_prompts is None else system_prompts
    items = make_items(records, cfg)
    t0 = time.time()
    results = sample(model, tok, items, cfg, seed, executor)
    t_sample = time.time() - t0
    mx.clear_cache()
    bad = nonascii_byte_ids(tok)
    masks = [kl_mask(r, bad) for r in results]
    kls, _, t_teacher = score(model, teacher, tok, [r.ids for r in results], [np.flatnonzero(m) for m in masks], cfg,
                              tseqs=teacher_sequences(tok, items, results, sp))
    mx.clear_cache()
    out = summarize(tok, items, results, kls, masks, bad)
    out.update(t_sample=t_sample, t_teacher=t_teacher, t_total=time.time() - t0, teacher_context=sorted(sp))
    if n_samples:
        out["samples"] = conversation_samples(tok, items, results, kls, n_samples)
    return out


# ------------------------------------------------------------------------------------------ optimizer config


def onpolicy_tc(meta, d_model, cfg):
    """(optimizer config, note): `lr_frac` x the off-policy stage's LRs (plan 12). meta = the start weights' meta:
    sft_final carries the scaled config it used (meta["optim"]); other start weights (a base model) get 0.15 x
    the pretraining peak as the assumed off-policy LR. optimizer "adamw" swaps Muon for Qwen-style AdamW at the
    same fraction of its own peak."""
    from bonsai.optim import resolve_tc, scale_tc
    from bonsai.sft import base_optim
    lr_scale = float(meta.get("lr_scale", 1.0) or 1.0)
    if meta.get("kind") == "sft" and "groups" in (meta.get("optim") or {}):
        off, frac, note = meta["optim"], float(meta.get("lr_frac", 0.15)) * lr_scale, "the SFT run's optimizer"
    else:
        frac = 0.15 * lr_scale
        off, note = scale_tc(base_optim(meta, d_model)[0], frac), "not an SFT output: off-policy LR = 0.15 x peak"
    if cfg.get("optimizer", "same") == "adamw" and off["optimizer"] != "adamw":
        off, note = scale_tc(resolve_tc({"optimizer": "adamw"}, d_model), frac), note + "; Qwen-style AdamW"
    return scale_tc(off, float(cfg["lr_frac"])), note


def forwarded_base_optim(meta):
    """The pretraining peak config for later stages (bonsai.sft.base_optim prefers meta["base_optim"])."""
    if meta.get("base_optim"):
        return meta["base_optim"]
    return meta.get("optim") if meta.get("kind") not in ("sft", "identity_sft", "onpolicy") else None


# ------------------------------------------------------------------------------------------ stop rules (plan 12)


def health(rec, base, cfg):
    """(healthy, why) of an eval record against the start (step 0): the GSM8K-style monitor not below it by more
    than gsm8k_tol, format errors not above it by more than format_tol."""
    if rec.get("gsm8k") is not None and base.get("gsm8k") is not None and \
            rec["gsm8k"] < base["gsm8k"] - cfg["gsm8k_tol"]:
        return False, f"GSM8K-style monitor dropped: {base['gsm8k']:.3f} -> {rec['gsm8k']:.3f}"
    if rec.get("format_errors") is not None and base.get("format_errors") is not None and \
            rec["format_errors"] > base["format_errors"] + cfg["format_tol"]:
        return False, f"format errors rose: {base['format_errors']} -> {rec['format_errors']} of {rec.get('format_n')}"
    return True, ""


def is_better(rec, best, cfg):
    """A healthy eval with a lower val KL than the best so far (without a val KL: every healthy eval)."""
    if not rec["healthy"]:
        return False
    if best is None or rec.get("val_kl") is None or best.get("val_kl") is None:
        return True
    return rec["val_kl"] < best["val_kl"] - cfg["kl_min_delta"]


def stop_reason(last, best, cfg):
    if not last["healthy"]:
        return last["why"]
    if best is not None and best.get("val_kl") is not None and last["step"] - best["step"] >= cfg["patience"]:
        return f"no val-KL improvement for {last['step'] - best['step']} steps (best {best['val_kl']:.4f} at " \
               f"step {best['step']})"
    return None


# ------------------------------------------------------------------------------------------ the training loop


class OnPolicy:
    """The plan 12 loop. model (holding the start weights) / tok / teacher / prompt records are given
    (scripts/distill_onpolicy.py loads them); `teacher` needs `iter_subset_logprobs(teacher_ids, vmap, chunk)` and
    `name` (tests use fakes); `executor` replaces bonsai.sandbox.run_tool_call (tests); `monitor` = the records
    of the GSM8K-style monitor (default: cfg["heldout"] read in full)."""

    def __init__(self, cfg, model, tok, teacher, prompts, heldout, meta, out_dir, log_dir, src="", executor=None,
                 monitor=None):
        from bonsai.optim import build_optimizer, set_muon_momentum
        self.cfg = {**DEFAULTS, **cfg}
        self.model, self.tok, self.teacher, self.executor = model, tok, teacher, executor
        self.prompts, self.heldout, self.meta, self.src = prompts, heldout, meta, str(src)
        self.system_prompts = load_system_prompts(self.cfg["system_prompts"])
        n = int(self.cfg["gsm8k_n"])
        self.math_items = math_items(load_records(self.cfg["heldout"]) if monitor is None else monitor, n) if n else []
        self.tc, self.tc_note = onpolicy_tc(meta, model.cfg.d_model, self.cfg)
        self.opt, self.groups = build_optimizer(model, self.tc)
        set_muon_momentum(self.opt, 0.95)
        self.out_dir, self.log_dir = Path(out_dir), Path(log_dir)
        self.ckpt_dir = self.out_dir / "onpolicy"
        self.best_path = self.ckpt_dir / "best.safetensors"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file, self.jsonl = self.log_dir / "onpolicy.log", self.log_dir / "onpolicy.jsonl"
        self.order = PromptOrder(len(prompts), int(self.cfg["batch"]), int(self.cfg["seed"]))
        self.bad = nonascii_byte_ids(tok)
        # train_kl: one entry per step (None = nothing scored); warmup_from: the step a fresh optimizer started at
        self.state = {"step": 0, "evals": [], "best": None, "train_kl": [], "bad_steps": 0, "stop": None,
                      "warmup_from": 0}
        self._signal = None

    @property
    def step(self):
        return self.state["step"]

    def log(self, msg):
        log(msg, self.log_file)

    def record(self, rec, path=None):
        with open(path or self.jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def weights_meta(self, kind, extra=None):
        return {"model_config": self.model.cfg.to_dict(), "vocab_sha": self.tok.vmap.sha(), "kind": kind,
                "run": self.cfg.get("run"), "base": self.src, "teacher": getattr(self.teacher, "name", None),
                "optim": self.tc, "base_optim": forwarded_base_optim(self.meta),
                "lr_scale": float(self.meta.get("lr_scale", 1.0) or 1.0), "lr_frac": self.cfg["lr_frac"],
                "sft_optim": self.meta.get("optim"), "teacher_context": sorted(self.system_prompts), **(extra or {})}

    def lr_mult(self, step):
        w = int(self.cfg["warmup"])
        return min(1.0, (step - self.state.get("warmup_from", 0) + 1) / w) if w else 1.0

    # --- one step ------------------------------------------------------------------------------------
    def train_step(self):
        from bonsai.optim import clip_by_global_norm, set_lr
        cfg, step = self.cfg, self.step
        idx, epoch = self.order(step)
        items = make_items([self.prompts[i] for i in idx], cfg, np.random.default_rng([int(cfg["seed"]), step]))
        mx.reset_peak_memory()
        t0 = time.time()
        results = sample(self.model, self.tok, items, cfg, int(cfg["seed"]) * 1000003 + step, self.executor)
        t_sample = time.time() - t0
        mx.clear_cache()
        masks = [kl_mask(r, self.bad) for r in results]
        targets = [np.flatnonzero(m) for m in masks]
        n = sum(len(t) for t in targets)
        set_lr(self.opt, self.groups, self.lr_mult(step))
        rec = {"kind": "train", "step": step + 1, "lr_mult": self.lr_mult(step), "epoch": epoch}
        gn, t_teacher, kls = float("nan"), 0.0, [np.zeros(0, dtype=np.float32)] * len(results)
        t1 = time.time()
        if n:
            kls, grads, t_teacher = score(self.model, self.teacher, self.tok, [r.ids for r in results], targets, cfg,
                                          grad_scale=1.0 / n,
                                          tseqs=teacher_sequences(self.tok, items, results, self.system_prompts))
            grads, norm = clip_by_global_norm(grads, float(cfg["clip"]))
            mx.eval(grads, norm)
            gn = norm.item()
            if math.isfinite(gn):
                self.opt.update(self.model, grads)
                mx.eval(self.model.parameters(), self.opt.state)
            del grads
        ok = n > 0 and math.isfinite(gn)
        self.state["bad_steps"] = 0 if ok else self.state["bad_steps"] + 1
        t_train = time.time() - t1
        mx.clear_cache()
        s = summarize(self.tok, items, results, kls, masks, self.bad)
        rec.update(kl=s["kl"], tokens=n, grad_norm=gn, updated=ok, kl_max=s["kl_max"], kl_p99=s["kl_p99"],
                   seq_kl=s["seq_kl"], by=s["by"], gen_tokens=s["gen_tokens"], finished=s["finished"],
                   tool_use=s["tool_use"], hit_budget=s["hit_budget"], format_ok=s["format_ok"], mask=s["mask"],
                   t_sample=round(t_sample, 2), t_teacher=round(t_teacher, 2), t_student=round(t_train - t_teacher, 2),
                   peak_gb=round(mx.get_peak_memory() / 1e9, 3))
        self.state["step"] = step + 1
        self.state["train_kl"].append(s["kl"] if n else None)
        self.record(rec)
        self.log(f"onpolicy step {step + 1}/{cfg['steps']} | kl {s['kl']:.4f} (think {s['by'].get('mode/think', {}).get('kl', 0):.3f}"
                 f" / no_think {s['by'].get('mode/no_think', {}).get('kl', 0):.3f}, max {s['kl_max']:.2f}) | "
                 f"{n:,} scored tok | gnorm {gn:.3f} | lr x{rec['lr_mult']:.2f} | gen {s['gen_tokens']:.0f} tok, "
                 f"fin {s['finished']:.2f}, tools {s['tool_use']:.2f} | sample {t_sample:.1f}s teacher {t_teacher:.1f}s "
                 f"student {t_train - t_teacher:.1f}s | peak {rec['peak_gb']:.2f} GB" + ("" if ok else " | NO UPDATE"))
        return rec

    # --- monitoring and stop rules -------------------------------------------------------------------
    def monitor_evals(self):
        """The GSM8K-style monitor on the held-out math items (never the GSM8K test set) and the format check on the
        mode-switch dev prompts (never the frozen eval prompts); a reply cut by the answer cap is counted apart
        (format_truncated), not as a format error."""
        cfg, out = self.cfg, {}
        if not (cfg["gsm8k_n"] or cfg["format_n"]):
            return out
        try:
            from bonsai.tasks import evals
        except ImportError as e:
            return {"monitor_error": repr(e)}
        if cfg["gsm8k_n"]:
            try:
                if not self.math_items:
                    raise ValueError(f"no math prompts with a numeric answer in {cfg['heldout']}")
                r = evals.eval_gsm8k_tool(self.model, self.tok, n=len(self.math_items), items=self.math_items,
                                          batch=int(cfg["eval_batch"]))
                out.update(gsm8k=r["accuracy"], gsm8k_n=r["n"], gsm8k_tool_use=r["tool_use_frac"],
                           gsm8k_finished=r["finished_frac"])
            except Exception as e:  # after step 0, monitoring never kills the run
                out["gsm8k_error"] = repr(e)
                self.log(f"GSM8K monitor failed: {e!r}")
        if cfg["format_n"]:
            try:
                r = evals.eval_mode_switch(self.model, self.tok, n=int(cfg["format_n"]), split="dev",
                                           max_answer_tokens=int(cfg["format_max_answer_tokens"]),
                                           batch=int(cfg["format_n"]))
                fails = r.get("failures")       # evals.TRUNCATED ("not finished (length)") is cut before any answer
                cut = sum(c for why, c in (fails or {}).items() if "length" in str(why))
                errs = int(round((1 - r["accuracy"]) * r["n"])) if fails is None else sum(fails.values()) - cut
                out.update(format_errors=errs, format_n=r["n"], format_truncated=r.get("truncated", cut),
                           format_finished=r.get("finished_frac"), format_failures=fails or {})
            except Exception as e:
                out["format_error"] = repr(e)
                self.log(f"format check failed: {e!r}")
        mx.clear_cache()
        return out

    def evaluate(self):
        cfg, step = self.cfg, self.step
        t0 = time.time()
        rec = {"kind": "eval", "step": step}
        if self.heldout:
            v = measure_kl(self.model, self.tok, self.teacher, self.heldout, cfg, seed=int(cfg["seed"]) + 1,
                           executor=self.executor, n_samples=4)
            rec.update(val_kl=v["kl"], val_tokens=v["tokens"], val_seq_kl=v["seq_kl"], val_by=v["by"],
                       val_kl_p99=v["kl_p99"], val_finished=v["finished"], val_tool_use=v["tool_use"],
                       val_format_ok=v["format_ok"], val_mask=v["mask"])
            self.record({"step": step, "samples": v.pop("samples", [])}, self.log_dir / "onpolicy_samples.jsonl")
        rec.update(self.monitor_evals())
        failed = {k: rec[k] for k in ("monitor_error", "gsm8k_error", "format_error") if k in rec}
        if failed and not self.state["evals"]:
            raise SystemExit(f"the step-0 monitor failed, so its stop rules could never fire: {failed} "
                             f"(fix it, or set gsm8k_n / format_n to 0 to run without them)")
        recent = [k for k in self.state["train_kl"][-int(cfg["eval_every"]):] if k is not None]
        rec["train_kl_mean"] = float(np.mean(recent)) if recent else None
        rec["healthy"], rec["why"] = health(rec, self.state["evals"][0] if self.state["evals"] else rec, cfg)
        best = self.state["best"]
        v = rec.get("val_kl")
        if is_better(rec, best, cfg):
            from bonsai.checkpoint import save_weights
            path = save_weights(self.best_path, self.model, self.weights_meta("onpolicy", {"step": step, "val_kl": v}))
            self.state["best"] = {"step": step, "val_kl": v, "gsm8k": rec.get("gsm8k"),
                                  "format_errors": rec.get("format_errors"), "path": str(path)}
            rec["new_best"] = True
        rec["t_eval"] = round(time.time() - t0, 1)
        self.state["evals"].append({k: rec[k] for k in ("step", "val_kl", "gsm8k", "format_errors", "format_n",
                                                         "format_truncated", "train_kl_mean", "healthy", "why")
                                    if k in rec})
        self.record(rec)
        fmt = lambda x, f: "-" if x is None else format(x, f)
        self.log(f"onpolicy eval step {step} | val KL {fmt(v, '.4f')} | GSM8K-style {fmt(rec.get('gsm8k'), '.3f')} | "
                 f"format errors {fmt(rec.get('format_errors'), 'd')}/{rec.get('format_n', '-')} "
                 f"({rec.get('format_truncated', 0)} cut at the answer cap, not errors) | train KL (last "
                 f"{len(recent)}) {fmt(rec['train_kl_mean'], '.4f')} | {'best' if rec.get('new_best') else ''}"
                 f"{'' if rec['healthy'] else 'UNHEALTHY: ' + rec['why']} ({rec['t_eval']:.0f}s)")
        return rec

    def check_stop(self):
        return stop_reason(self.state["evals"][-1], self.state["best"], self.cfg)

    # --- checkpoints ---------------------------------------------------------------------------------
    def save_checkpoint(self):
        from bonsai.checkpoint import save_full
        meta = self.weights_meta("onpolicy_full", {"state": self.state, "cfg": self.cfg})
        path = save_full(self.ckpt_dir, self.step, self.model, self.opt, meta, keep=int(self.cfg["keep"]))
        self.log(f"saved {path}")
        return path

    def resume(self, which="auto"):
        """Restores a checkpoint of ckpt/<run>/onpolicy/; which = "auto" | "latest" | "best" | a checkpoint dir (see
        the module docstring). Returns False when there is none."""
        from bonsai import checkpoint as Ck
        path = Ck.latest(self.ckpt_dir) if which in ("auto", "latest", "best") else Path(which)
        if path is None:
            self.log(f"no checkpoint in {self.ckpt_dir}: starting at step 0")
            return False
        meta = Ck.load_meta(path)
        if meta.get("vocab_sha") != self.tok.vmap.sha():
            raise SystemExit(f"{path}: vocabulary {meta.get('vocab_sha')} != tokenizer {self.tok.vmap.sha()}")
        last = meta["state"]
        old_lr = (meta.get("cfg") or {}).get("lr_frac", meta.get("lr_frac"))
        lr_changed = old_lr is not None and float(old_lr) != float(self.cfg["lr_frac"])
        unhealthy = bool(last["evals"]) and not last["evals"][-1].get("healthy", True)
        best, state, fresh_opt = last.get("best"), last, False
        if best is not None and best["step"] < last["step"] and \
                (which == "best" or (which == "auto" and (unhealthy or lr_changed))):
            b = best["step"]
            bdir = self.ckpt_dir / f"step_{b:06d}"
            if (bdir / "meta.json").exists():
                path, state = bdir, Ck.load_meta(bdir)["state"]
            else:   # step 0 (never a full checkpoint) or rotated away: the best weights with a fresh optimizer
                if not self.best_path.exists() or Ck.load_meta(self.best_path).get("step") != b:
                    raise SystemExit(f"cannot go back to the best step {b}: neither {bdir} nor a matching "
                                     f"{self.best_path}; pass --resume latest or a checkpoint dir")
                path, fresh_opt = self.best_path, True
                state = {**last, "step": b, "evals": [e for e in last["evals"] if e["step"] <= b],
                         "train_kl": list(last["train_kl"][:b]), "warmup_from": b}
        state = {**state, "stop": None, "bad_steps": 0}
        resave = self._sync_best(state)
        Ck.load_weights(self.model, path)
        if not fresh_opt:
            Ck.load_optimizer(self.opt, path)
        if resave:
            Ck.save_weights(self.best_path, self.model, self.weights_meta(
                "onpolicy", {"step": state["step"], "val_kl": state["best"].get("val_kl")}))
        self.state = state
        later = sorted(d for d in self.ckpt_dir.glob("step_*") if d.is_dir() and int(d.name[5:]) > self.step)
        moved = aside(self.ckpt_dir / "x", "abandoned_") if later else None
        for d in later:     # never deleted; out of the way of rotation and `latest`
            moved.mkdir(exist_ok=True)
            d.rename(moved / d.name)
        dropped = [e["step"] for e in last["evals"] if e["step"] > self.step]
        self.record({"kind": "resume", "step": self.step, "from": str(path), "latest_step": last["step"],
                     "old_stop": last.get("stop"), "dropped_evals": dropped, "lr_frac": self.cfg["lr_frac"],
                     "old_lr_frac": old_lr, "fresh_optimizer": fresh_opt, "moved": str(moved) if moved else None})
        self.log(f"resumed {path} at step {self.step} (newest checkpoint: step {last['step']}"
                 f"{', stopped: ' + last['stop'] if last.get('stop') else ''}"
                 f"{'; lr_frac ' + str(old_lr) + ' -> ' + str(self.cfg['lr_frac']) if lr_changed else ''}"
                 f"{'; evals dropped: ' + str(dropped) if dropped else ''}{'; fresh optimizer' if fresh_opt else ''}"
                 f"{'; later checkpoints moved to ' + str(moved) if moved else ''}; LR from the current config: "
                 f"{self.tc['groups']})")
        return True

    def _sync_best(self, state):
        """best.safetensors must hold state["best"]'s weights (an older checkpoint's best may have been replaced):
        re-saved here from the start weights (best = step 0; the model still holds them) or by resume() from the
        loaded checkpoint (best = its step, returns True); otherwise the checkpoint's own healthy eval becomes the
        best (or none)."""
        from bonsai import checkpoint as Ck
        best = state.get("best")
        if best is None:
            return False
        best["path"] = str(self.best_path)
        on_disk = Ck.load_meta(self.best_path).get("step") if self.best_path.exists() else None
        if on_disk == best["step"]:
            return False
        if best["step"] == state["step"]:
            return True
        if best["step"] == 0:
            Ck.save_weights(self.best_path, self.model,
                            self.weights_meta("onpolicy", {"step": 0, "val_kl": best.get("val_kl")}))
            return False
        ev = next((e for e in state["evals"] if e["step"] == state["step"] and e.get("healthy")), None)
        self.log(f"WARNING: {self.best_path} holds step {on_disk}, not the checkpoint's best step {best['step']}: "
                 f"the best is now {'step ' + str(ev['step']) if ev else 'unset'}")
        state["best"] = None if ev is None else {
            "step": ev["step"], "val_kl": ev.get("val_kl"), "gsm8k": ev.get("gsm8k"),
            "format_errors": ev.get("format_errors"), "path": str(self.best_path)}
        return ev is not None

    def _fresh_start(self, overwrite=False):
        """A fresh run never mixes with an earlier attempt's checkpoints (rotation by step number, best.safetensors):
        refused unless `overwrite`, which moves onpolicy/ (+ onpolicy_final) to onpolicy.<time>/ and the jsonl logs
        to *.<time>.json*."""
        from bonsai import checkpoint as Ck
        if Ck.latest(self.ckpt_dir) is None and not self.best_path.exists():
            return
        if not overwrite:
            raise SystemExit(f"{self.ckpt_dir} holds an earlier attempt (checkpoints / best.safetensors): "
                             f"--resume auto continues it, --overwrite moves it aside and starts over")
        dest = aside(self.ckpt_dir, "onpolicy.")
        stamp = dest.name[len("onpolicy."):]
        self.ckpt_dir.rename(dest)
        final = self.out_dir / "onpolicy_final.safetensors"
        if final.exists():
            final.rename(dest / final.name)
        for p in (self.jsonl, self.log_dir / "onpolicy_samples.jsonl", self.log_dir / "onpolicy_summary.json"):
            if p.exists():
                p.rename(p.with_name(f"{p.stem}.{stamp}{p.suffix}"))
        self.log(f"moved the earlier attempt aside to {dest} (logs: onpolicy*.{stamp}.json*); starting at step 0")

    def _on_signal(self, signum, frame):
        self._signal = signum
        self.log(f"signal {signum}: stopping after this step")

    # --- run -----------------------------------------------------------------------------------------
    def run(self, resume=None, overwrite=False):
        """resume: None = a fresh run (see _fresh_start); "auto" | "latest" | "best" | a checkpoint dir: resume()
        (nothing to resume: a fresh run that moves leftovers aside)."""
        cfg = self.cfg
        if not (resume and self.resume(resume)):
            self._fresh_start(overwrite or bool(resume))
        self.log(f"on-policy distillation: {len(self.prompts):,} prompts, batch {cfg['batch']}, steps {cfg['steps']}, "
                 f"teacher {getattr(self.teacher, 'name', '?')} (system prompts: "
                 f"{', '.join(sorted(self.system_prompts)) or 'none'}), start {self.src}; LRs {self.tc['groups']} "
                 f"({cfg['lr_frac']} x {self.tc_note}); held-out {len(self.heldout)} for val KL, "
                 f"{len(self.math_items)} math items for the monitor")
        old = {}
        try:
            old = {s: signal.signal(s, self._on_signal) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        except ValueError:   # not the main thread: no Ctrl-C checkpoint
            pass
        try:
            if not self.state["evals"]:
                self.evaluate()
            stop = self.state["stop"]
            while stop is None and self.step < int(cfg["steps"]):
                self.train_step()
                if self.state["bad_steps"] >= 3:
                    stop = "3 steps in a row without an update (non-finite gradients or nothing to score)"
                if self.step % int(cfg["eval_every"]) == 0 or self.step == int(cfg["steps"]):
                    self.evaluate()
                    stop = stop or self.check_stop()
                self.state["stop"] = stop
                if self._signal is not None:
                    self.save_checkpoint()
                    return {"stopped": "signal", "step": self.step}
                if self.step % int(cfg["ckpt_every"]) == 0 or stop or self.step == int(cfg["steps"]):
                    self.save_checkpoint()
        finally:
            for s, h in old.items():
                signal.signal(s, h)
        return self.finish(stop or f"{cfg['steps']} steps done")

    def finish(self, reason):
        from bonsai import checkpoint as Ck
        best, step = self.state["best"], self.step
        use = step
        if self.cfg["final"] == "best" and best is not None and best["step"] != step:
            on_disk = Ck.load_meta(self.best_path).get("step") if self.best_path.exists() else None
            if on_disk == best["step"]:
                Ck.load_weights(self.model, self.best_path)
                use = best["step"]
            else:
                self.log(f"WARNING: {self.best_path} holds step {on_disk}, not the best step {best['step']}: the final "
                         f"weights are the last step's")
        ev = next((e for e in self.state["evals"] if e["step"] == use), {})
        start = self.state["evals"][0] if self.state["evals"] else {}
        extra = {"step": use, "steps_run": step, "stop_reason": reason, "val_kl": ev.get("val_kl"),
                 "val_kl_start": start.get("val_kl"), "gsm8k": ev.get("gsm8k"), "gsm8k_start": start.get("gsm8k"),
                 "format_errors": ev.get("format_errors")}
        path = Ck.save_weights(self.out_dir / "onpolicy_final.safetensors", self.model, self.weights_meta("onpolicy", extra))
        summary = {**extra, "path": str(path), "evals": self.state["evals"], "optim": self.tc["groups"]}
        write_json(self.log_dir / "onpolicy_summary.json", summary)
        self.log(f"on-policy done: {reason}; final = step {use} (val KL {start.get('val_kl')} -> {ev.get('val_kl')}, "
                 f"GSM8K {start.get('gsm8k')} -> {ev.get('gsm8k')}) -> {path}"
                 + ("  [WARNING: no eval beat the start model; the final weights are the SFT model's]" if use == 0 else ""))
        return summary
