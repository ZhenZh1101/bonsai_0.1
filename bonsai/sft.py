"""Off-policy distillation SFT (plan sections 11, 14) and the identity mini-SFT after on-policy distillation
(plan section 18).

Sets (scripts/build_sft.py, configs/sft.yaml), pagoda's SFT format (DESIGN 1.2):
  tok/<dir>/{train,val}_00000.{bin,idx,mask} + {train,val}_manifest.json; student ids in the manifest's dtype
  (tok.vmap.shard_dtype), every conversation starts with BOS (<|endoftext|>), mask = the chat.render mask
  (train_on="last": every assistant turn after the last user query, incl. think blocks and tool-calling steps,
  never tool output; BOS 0), expanded over the pieces of out-of-subset tokens. Conversations longer than
  seq_len + 1 tokens are dropped (one packed row holds at most seq_len + 1). Val = a stable hash of the prompt
  id, so both modes of one prompt land in the same split.

  tok/offpolicy     the kept teacher samples (data/offpolicy/kept_*.jsonl) + `identity.frac` (2%) scripted
                    identity conversations (raw/filtered/identity_chat.jsonl, never from the teacher): the last
                    assistant turn of a stable-hashed cut, in both modes; thinking ones get
                    bonsai.identity.short_reasoning, non-thinking ones the empty think block. Mode signalling as
                    in generation: thinking = no switch or " /think" (30%); non-thinking = " /no_think" (50%) or
                    the enable_thinking=False prefill (think_prefilled, the empty block not supervised). A small
                    share gets the Bonsai system prompt (bonsai.identity.with_system_prompt, pagoda practice).
  tok/identity_sft  every identity conversation in both modes + `replay_ratio` x as many off-policy samples
                    (general replay), for the 100-200 step identity fix after on-policy distillation.

Training (`run_sft`; scripts/sft.py, scripts/identity_sft.py): from ckpt/<run>/base.safetensors (bonsai.load,
or a local reader of a weights file with meta), loss on supervised tokens only (thinking counts), best-fit
packing of whole conversations into seq_len + 1 rows (pagoda), the base run's optimizer (meta["optim"]: Muon +
AdamW or Qwen-style AdamW) at `lr_frac` (0.15) x its peak LRs (x its spike-recovery lr_scale), Muon momentum 0.95,
warmup 50 steps then linear decay to 0, clip 1.0, 64k tokens per step. Epoch mode (SFT): 2 epochs, each saved as
sft_epoch{k}.safetensors; if the val loss rises in an epoch the previous one is kept, sft_final.safetensors
links to the best epoch. Step mode (identity mini-SFT): `steps` steps (data cycles), writes final.safetensors.
Resumable: a full checkpoint (fp32 weights, optimizer state, step, epoch + batch within it, history, the settings
that fix order and schedule) goes to ckpt/<run>/<sft|identity_sft>/step_* every `save_every_min` (30) minutes, at
each epoch end and at the end of step mode (newest 3 kept, bonsai.checkpoint.save_full); `resume="auto"`
continues the latest exactly (the row order is rows(epoch, seed), so the same batches follow; it refuses changed
settings). SIGINT / SIGTERM / SIGHUP save one after the current step and stop (the scripts exit with code 3). A
fresh start refuses a directory that already holds checkpoints, so two attempts never mix in its rotation.
Weights files carry meta: model_config, vocab_sha, kind, optim (the scaled config actually used: on-policy
distillation takes 0.5 x these LRs, `scale_tc(meta["optim"], 0.5)`), base_optim (the pretraining peak; later
stages should forward it with lr_scale so the identity mini-SFT finds it), lr_scale, base, epoch, step, val_loss.
"""

import bisect
import hashlib
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

from bonsai import chat
from bonsai.common import CKPT_DIR, LOG_DIR, TOK_DIR, iter_jsonl, log, write_json

SHARD_TOKENS = 2 ** 24


def stable_frac(key, salt=""):
    return int(hashlib.md5((salt + "\x00" + str(key)).encode()).hexdigest()[:8], 16) / 2 ** 32


# ------------------------------------------------------------------------------------------ encoding


def to_student(vmap, teacher_ids, mask):
    """(student ids int32, mask uint8 repeated over split pieces, number of out-of-subset teacher tokens)."""
    t = np.asarray(teacher_ids, dtype=np.int64)
    m = np.asarray(mask, dtype=np.uint8)
    bad = np.flatnonzero(vmap.t2s[t] < 0)
    ids = vmap.to_student(t).astype(np.int32)
    if len(bad) == 0:
        return ids, m, 0
    lens = np.ones(len(t), dtype=np.int64)
    lens[bad] = [len(vmap.split(x)) for x in t[bad]]
    return ids, np.repeat(m, lens), len(bad)


def encode_chats(tok, convs, train_on="last"):
    """[(ids, mask, n_oov)] for conversations {"messages", "tools": bool}: BOS + chat.render, student ids."""
    items = [{"messages": c["messages"], "tools": chat.TOOLS if c.get("tools") else None} for c in convs]
    out = []
    for ids, mask in chat.render_batch(tok.qwen, items, train_on=train_on):
        s, m, n = to_student(tok.vmap, ids, mask)
        out.append((np.concatenate([[tok.bos_id], s]).astype(np.int32), np.concatenate([[0], m]).astype(np.uint8), n))
    return out


def write_split(out_dir, prefix, items, tok, seq_len, extra=None):
    """Writes <prefix>_XXXXX.{bin,idx,mask} + <prefix>_manifest.json; items = [(ids, mask)]."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob(f"{prefix}_*.*"):
        if old.suffix in (".bin", ".idx", ".mask"):
            old.unlink()
    dtype = tok.vmap.shard_dtype
    shards, buf = [], []

    def flush():
        if not buf:
            return
        name = f"{prefix}_{len(shards):05d}"
        np.concatenate([i for i, _ in buf]).astype(dtype).tofile(d / f"{name}.bin")
        np.concatenate([m for _, m in buf]).astype(np.uint8).tofile(d / f"{name}.mask")
        off = np.zeros(len(buf) + 1, dtype=np.uint64)
        off[1:] = np.cumsum([len(i) for i, _ in buf])
        off.tofile(d / f"{name}.idx")
        shards.append({"name": name, "tokens": int(off[-1]), "docs": len(buf)})
        buf.clear()

    n = 0
    for ids, mask in items:
        if n + len(ids) > SHARD_TOKENS and buf:
            flush()
            n = 0
        buf.append((ids, mask))
        n += len(ids)
    flush()
    man = {"bos_id": tok.bos_id, "vocab_size": tok.vocab_size, "vocab_sha": tok.vmap.sha(),
           "dtype": np.dtype(dtype).name, "seq_len": seq_len, "has_mask": True,
           "total_tokens": sum(s["tokens"] for s in shards), "total_docs": sum(s["docs"] for s in shards),
           "supervised_tokens": int(sum(int(m.sum()) for _, m in items)), "shards": shards, **(extra or {})}
    write_json(d / f"{prefix}_manifest.json", man)
    return man


# ------------------------------------------------------------------------------------------ sets


def _identity():
    try:
        import bonsai.identity as identity
        return identity
    except ImportError:
        return None


def identity_samples(convs, n, cfg, salt="sft"):
    """`n` training conversations from identity_chat records (cycled), the mode and signal by stable hash."""
    ident = _identity()
    if ident is None and convs and n:
        raise SystemExit("bonsai.identity is missing (identity builder): it provides short_reasoning for the "
                         "thinking identity samples; build with --no-identity to skip them")
    think_frac = float(cfg.get("think_frac", 0.5))
    out, k = [], 0
    order = list(range(len(convs)))
    random.Random(salt).shuffle(order)
    while len(out) < n and convs:
        rec = convs[order[k % len(order)]]
        key = f"{rec.get('id', order[k % len(order)])}|{k // len(order)}|{salt}"
        k += 1
        mode = "think" if stable_frac(key, "mode") < think_frac else "no_think"
        out.append(identity_sample(rec, mode, key, cfg, ident))
    return out


def identity_sample(rec, mode, key, cfg, ident):
    msgs = [{"role": m["role"], "content": m["content"]} for m in rec["messages"]]
    asst = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    cut = asst[int(stable_frac(key, "cut") * len(asst))]          # multi-turn: supervise one stable-hashed turn
    msgs = msgs[:cut + 1]
    user_text = msgs[cut - 1]["content"]
    u = stable_frac(key, "signal")
    if mode == "think":
        signal = "soft" if u < cfg.get("think_soft_frac", 0.3) else "default"
    else:
        signal = "soft" if u < cfg.get("no_think_soft_frac", 0.5) else "prefill"
    if signal == "soft":
        msgs[cut - 1] = {**msgs[cut - 1], "content": chat.with_soft_switch(msgs[cut - 1]["content"], mode)}
    # the category describes the conversation's first exchange (in_task: every turn), so pass it only then
    category = rec.get("category") if cut == 1 or rec.get("category") == "in_task" else None
    reasoning = ident.short_reasoning(user_text, key=key, category=category, reply=msgs[cut]["content"]) \
        if mode == "think" else ""
    msgs[cut] = {"role": "assistant", "content": msgs[cut]["content"], "reasoning_content": reasoning}
    if signal == "prefill":
        msgs[cut]["think_prefilled"] = True
    frac = float(cfg.get("system_prompt_frac", 0.0))
    if frac and ident is not None:
        msgs = ident.with_system_prompt(msgs, frac, key=key)
    return {"id": f"identity:{key}", "prompt_id": f"identity:{rec.get('id')}", "messages": msgs, "tools": False,
            "mode": mode, "signal": signal, "kind": "identity"}


def with_general_system_prompt(sample, frac):
    ident = _identity()
    if not frac or ident is None:
        return sample
    return {**sample, "messages": ident.with_system_prompt(sample["messages"], frac, key=sample["id"])}


def encode_set(tok, samples, seq_len, chunk=4096):
    """[(ids, mask, sample)] of samples that fit one row, plus counts."""
    kept, cnt = [], {"too_long": 0, "oov_tokens": 0}
    for a in range(0, len(samples), chunk):
        part = samples[a:a + chunk]
        for s, (ids, mask, n_oov) in zip(part, encode_chats(tok, part)):
            cnt["oov_tokens"] += n_oov
            if len(ids) > seq_len + 1:
                cnt["too_long"] += 1
                continue
            kept.append((ids, mask, s))
    return kept, cnt


def build_offpolicy_sft(cfg, tok, identity_ok=True, log_file=None):
    """tok/offpolicy (+ tok/identity_sft) from configs/sft.yaml; returns the report."""
    from bonsai.common import DATA_ROOT, FILTERED_DIR
    t0 = time.time()
    seq_len = int(cfg.get("seq_len", 1024))
    from bonsai.teacher import iter_samples        # skips a line cut by a hard kill of gen_offpolicy
    paths = sorted(DATA_ROOT.glob(cfg.get("offpolicy_glob", "data/offpolicy/kept_*.jsonl")))
    general = [r for p in paths for r in iter_samples(p, log_file)]
    if not general:
        raise SystemExit(f"no off-policy samples match {cfg.get('offpolicy_glob')}: run scripts.gen_offpolicy first")
    seen, dedup = set(), []
    for r in general:
        if r["id"] not in seen:
            seen.add(r["id"])
            dedup.append(r)
    general = dedup
    icfg = dict(cfg.get("identity", {}))
    id_path = FILTERED_DIR / f"{icfg.get('pool', 'identity_chat')}.jsonl"
    convs = list(iter_jsonl(id_path)) if identity_ok and id_path.exists() else []
    if identity_ok and not convs:
        raise SystemExit(f"{id_path} missing: run `python -m scripts.gen_identity pools` (or pass --no-identity)")
    frac = float(icfg.get("frac", 0.02)) if convs else 0.0
    n_id = round(frac / (1 - frac) * len(general)) if frac else 0
    gfrac = float(cfg.get("system_prompt_frac", 0.0))
    general = [with_general_system_prompt(s, gfrac) for s in general]
    ident = identity_samples(convs, n_id, icfg, salt="sft")
    report = {"offpolicy_files": [str(p) for p in paths], "general_samples": len(general),
              "identity_samples": len(ident), "identity_convs": len(convs)}
    items, cnt = encode_set(tok, general + ident, seq_len)
    report.update(cnt)
    vfrac = float(cfg.get("val_fraction", 0.02))
    train, val = [], []
    for ids, mask, s in items:
        (val if stable_frac(s.get("prompt_id", s["id"]), "val") < vfrac else train).append((ids, mask, s))
    rng = random.Random(int(cfg.get("seed", 0)))
    rng.shuffle(train)
    rng.shuffle(val)
    out_dir = TOK_DIR / cfg.get("shard_dir", "offpolicy")
    report["by_kind"] = _counts(train)
    man = write_split(out_dir, "train", [(i, m) for i, m, _ in train], tok, seq_len,
                      {"config": cfg, "report": report})
    write_split(out_dir, "val", [(i, m) for i, m, _ in val], tok, seq_len, {"by_kind": _counts(val)})
    log(f"sft set {out_dir}: train {man['total_docs']:,} convs / {man['total_tokens'] / 1e6:.1f}M tokens "
        f"({man['supervised_tokens'] / 1e6:.1f}M supervised), val {len(val):,}; identity {len(ident):,}, "
        f"too long {cnt['too_long']:,} ({time.time() - t0:.0f}s)", log_file)
    report["train"] = {k: man[k] for k in ("total_docs", "total_tokens", "supervised_tokens")}
    report["val_docs"] = len(val)
    isft = cfg.get("identity_sft")
    if isft and convs:
        report["identity_sft"] = build_identity_sft(isft, tok, convs, [s for _, _, s in train], seq_len, log_file)
    return report


def build_identity_sft(cfg, tok, convs, general, seq_len, log_file=None):
    """tok/identity_sft: each identity conversation in both modes + replay_ratio x general samples."""
    ident_mod = _identity()
    icfg = dict(cfg)
    ident = [identity_sample(rec, mode, f"{rec.get('id', i)}|{mode}|isft", icfg, ident_mod)
             for i, rec in enumerate(convs) for mode in ("think", "no_think")]
    n_rep = min(len(general), int(round(float(cfg.get("replay_ratio", 3)) * len(ident))))
    replay = random.Random(int(cfg.get("seed", 5))).sample(general, n_rep)
    items, cnt = encode_set(tok, ident + replay, seq_len)
    vfrac = float(cfg.get("val_fraction", 0.05))
    train = [(i, m, s) for i, m, s in items if stable_frac(s["id"], "ival") >= vfrac]
    val = [(i, m, s) for i, m, s in items if stable_frac(s["id"], "ival") < vfrac]
    random.Random(int(cfg.get("seed", 5))).shuffle(train)
    out_dir = TOK_DIR / cfg.get("shard_dir", "identity_sft")
    rep = {"identity": len(ident), "replay": len(replay), **cnt, "by_kind": _counts(train)}
    man = write_split(out_dir, "train", [(i, m) for i, m, _ in train], tok, seq_len, {"config": cfg, "report": rep})
    write_split(out_dir, "val", [(i, m) for i, m, _ in val], tok, seq_len, {"by_kind": _counts(val)})
    log(f"identity sft set {out_dir}: train {man['total_docs']:,} convs / {man['total_tokens'] / 1e6:.2f}M tokens, "
        f"val {len(val):,}", log_file)
    return rep


def _counts(items):
    from collections import Counter
    return dict(Counter(f"{s.get('kind')}{'+tools' if s.get('tools') else ''}/{s.get('mode')}/{s.get('signal')}"
                        for _, _, s in items).most_common())


# ------------------------------------------------------------------------------------------ training


class SFTData:
    """All conversations of one split (tokens + masks), best-fit packed into rows per epoch (pagoda)."""

    def __init__(self, split, seq_len, pad_id, data_dir, vocab_sha=None):
        d = Path(data_dir)
        man = json.loads((d / f"{split}_manifest.json").read_text())
        if vocab_sha is not None and man.get("vocab_sha") != vocab_sha:
            raise SystemExit(f"{d}: built for vocabulary {man.get('vocab_sha')}, the tokenizer is {vocab_sha}")
        dtype = np.dtype(man.get("dtype", "uint16"))
        self.convs = []
        for s in man["shards"]:
            toks = np.fromfile(d / f"{s['name']}.bin", dtype=dtype)
            mask = np.fromfile(d / f"{s['name']}.mask", dtype=np.uint8)
            idx = np.fromfile(d / f"{s['name']}.idx", dtype=np.uint64).astype(np.int64)
            for i in range(len(idx) - 1):
                self.convs.append((toks[idx[i]:idx[i + 1]], mask[idx[i]:idx[i + 1]]))
        self.cap, self.pad = seq_len + 1, pad_id
        self.tokens = sum(len(t) for t, _ in self.convs)
        self.manifest = man

    def rows(self, epoch, seed):
        """Best-fit packing of a seeded permutation -> (inputs, targets) of shape (rows, seq_len); targets -1
        where the mask is 0 (prompt, tool output, BOS, padding)."""
        order = np.random.default_rng([seed, epoch]).permutation(len(self.convs))
        free, bins = [], []                    # free: sorted (space_left, bin index)
        for k in order:
            n = len(self.convs[k][0])
            if n > self.cap:
                continue
            j = bisect.bisect_left(free, (n, -1))
            if j < len(free):
                space, b = free.pop(j)
                bins[b].append(k)
                if space - n > 0:
                    bisect.insort(free, (space - n, b))
            else:
                bins.append([k])
                if self.cap - n > 0:
                    bisect.insort(free, (self.cap - n, len(bins) - 1))
        ids = np.full((len(bins), self.cap), self.pad, dtype=np.int32)
        sup = np.zeros((len(bins), self.cap), dtype=bool)
        for b, members in enumerate(bins):
            pos = 0
            for k in members:
                t, m = self.convs[k]
                ids[b, pos:pos + len(t)] = t
                sup[b, pos:pos + len(t)] = m.astype(bool)
                pos += len(t)
        perm = np.random.default_rng([seed, epoch, 1]).permutation(len(bins))
        ids, sup = ids[perm], sup[perm]
        return ids[:, :-1], np.where(sup[:, 1:], ids[:, 1:], -1)


def load_base(run, source="base", path=None, tok=None):
    """(model, tok, meta, path): bonsai.load.load_model, or a local reader of a weights file with meta."""
    try:
        from bonsai.load import load_model
    except ImportError:
        load_model = None
    if load_model is not None:
        model, tok, meta = load_model(run, str(path) if path else source, tok=tok)
        return model, tok, meta, Path(meta.get("path", path or ""))
    from bonsai import checkpoint as C
    from bonsai.model import build_model
    from bonsai.tokenizer import get_tokenizer
    p = Path(path) if path else CKPT_DIR / run / {"base": "base.safetensors", "onpolicy": "onpolicy_final.safetensors",
                                                    "sft": "sft_final.safetensors"}[source]
    meta = C.load_meta(p)
    tok = tok or get_tokenizer()
    if meta.get("vocab_sha") not in (None, tok.vmap.sha()):
        raise SystemExit(f"{p}: vocabulary {meta.get('vocab_sha')} != tokenizer {tok.vmap.sha()}")
    model = build_model(meta["model_config"])
    C.load_weights(model, p)
    return model, tok, meta, p


def base_optim(meta, d_model):
    """The pretraining run's resolved optimizer config at its peak LRs: meta["base_optim"] when the start weights
    come from a post-training stage (sft / onpolicy files forward it), else meta["optim"] of a pretraining file;
    defaults if neither is there."""
    from bonsai.optim import resolve_tc
    tc = meta.get("base_optim") or meta.get("optim") or meta.get("tc")
    if tc and "groups" in tc:
        return tc, True
    args = meta.get("args") or {}
    keys = ("optimizer", "matrix_lr", "embed_lr", "norm_lr", "adamw_lr", "weight_decay")
    return resolve_tc({k: args.get(k) for k in keys}, d_model), False


def _step_fns(model, opt, clip, token_bytes, compile=True):
    try:
        from bonsai.train import build_step_fns
    except ImportError:
        build_step_fns = None
    if build_step_fns is not None:
        return build_step_fns(model, opt, clip, token_bytes, compile)
    import mlx.nn as nn
    from bonsai.model import token_losses
    from bonsai.optim import clip_by_global_norm

    def loss_sum(model, x, y, scale):
        ce, valid = token_losses(model, x, y)
        return ce.sum() * scale, valid.sum()

    grad_fn = nn.value_and_grad(model, loss_sum)

    def micro(x, y, scale):
        (loss, n), grads = grad_fn(model, x, y, scale)
        return loss, n, grads

    def apply(grads):
        grads, norm = clip_by_global_norm(grads, clip)
        opt.update(model, grads)
        return norm

    def evaluate(x, y):
        ce, valid = token_losses(model, x, y)
        return ce.sum(), valid.sum(), ce.sum(), valid.sum()

    if not compile:
        return micro, apply, evaluate
    state = [model.state, opt.state]
    return (mx.compile(micro, inputs=[model.state], outputs=[model.state]),
            mx.compile(apply, inputs=state, outputs=state), mx.compile(evaluate, inputs=[model.state]))


def run_sft(args, tok=None):
    """SFT (epoch mode, args.steps == 0) or the identity mini-SFT (step mode). See the module docstring.

    args: run, base_run, source, base (path), data (dir), out_dir, log_dir, epochs, steps, lr_frac, warmup,
    batch_tokens, micro_batch, clip, log_every, eval_every, seed, memory_limit_gb, compile, final_name, and
    (optional) resume ("" | "auto" | a checkpoint dir), save_every_min (30), keep (3). Returns the history; on
    SIGINT / SIGTERM / SIGHUP it saves a full checkpoint after the current step and returns it with
    "stopped": "signal" (the scripts exit with code 3)."""
    import signal
    import sys
    from bonsai import checkpoint as C
    from bonsai.optim import build_optimizer, linear_warmup_decay, scale_tc, set_lr, set_muon_momentum

    mx.set_memory_limit(int(args.memory_limit_gb * 1024 ** 3))
    mx.random.seed(args.seed)
    model, tok, meta, src = load_base(args.base_run, args.source, args.base or None, tok)
    cfg = model.cfg
    tc0, from_meta = base_optim(meta, cfg.d_model)
    lr_scale = float(meta.get("lr_scale", 1.0) or 1.0)
    tc = scale_tc(tc0, args.lr_frac * lr_scale)
    opt, groups = build_optimizer(model, tc)
    set_muon_momentum(opt, 0.95)
    out_dir = Path(args.out_dir) if args.out_dir else CKPT_DIR / args.run
    log_dir = Path(args.log_dir) if args.log_dir else LOG_DIR / args.run
    log_dir.mkdir(parents=True, exist_ok=True)
    stage = "identity_sft" if args.steps else "sft"
    ckpt_dir = out_dir / stage
    resume, keep = getattr(args, "resume", "") or "", int(getattr(args, "keep", 3))
    save_every = 60 * float(getattr(args, "save_every_min", 30.0))
    log_file, metrics = log_dir / f"{stage}.log", log_dir / "metrics.jsonl"
    log(f"{stage} from {src} ({cfg.n_layer}x{cfg.d_model}); LRs {groups} = {args.lr_frac} x lr_scale {lr_scale:.3f} x "
        f"the base run's peak {tc0['groups']}" + ("" if from_meta else " (base meta has no 'optim': defaults)"),
        log_file)
    token_bytes = mx.array(tok.token_bytes(), dtype=mx.float32)
    micro, apply, evaluate = _step_fns(model, opt, args.clip, token_bytes, args.compile)

    data_dir = Path(args.data)
    train = SFTData("train", cfg.seq_len, tok.bos_id, data_dir, tok.vmap.sha())
    val = SFTData("val", cfg.seq_len, tok.bos_id, data_dir, tok.vmap.sha())
    B = args.micro_batch
    accum = max(1, args.batch_tokens // (B * cfg.seq_len))
    rows_per_epoch = len(train.rows(0, args.seed)[0])
    steps_per_epoch = max(1, rows_per_epoch // (B * accum))
    total_steps = args.steps or steps_per_epoch * args.epochs
    log(f"{stage}: {len(train.convs):,} train convs ({train.tokens / 1e6:.1f}M tokens) -> {rows_per_epoch:,} rows/epoch, "
        f"{steps_per_epoch} steps/epoch, {total_steps} steps, {B * accum * cfg.seq_len:,} tokens/step; val "
        f"{len(val.convs):,} convs", log_file)
    vx, vy = val.rows(0, 0)
    # what fixes the data order and the LR schedule: a resumed run must match it exactly
    plan = {"data": str(data_dir), "train_convs": len(train.convs), "train_tokens": int(train.tokens), "seed": args.seed,
            "seq_len": cfg.seq_len, "micro_batch": B, "accum": accum, "epochs": args.epochs, "steps": args.steps,
            "total_steps": total_steps, "warmup": args.warmup, "lr_frac": args.lr_frac, "clip": args.clip,
            "base": str(src)}
    wmeta = {"model_config": cfg.to_dict(), "vocab_sha": tok.vmap.sha(), "kind": stage, "run": args.run,
             "base": str(src), "optim": tc, "base_optim": tc0, "lr_scale": lr_scale, "lr_frac": args.lr_frac}

    def val_loss():
        tot = n = 0.0
        for i in range(0, len(vx), B):
            ce, cnt, _, _ = evaluate(mx.array(vx[i:i + B]), mx.array(vy[i:i + B]))
            mx.eval(ce, cnt)
            tot += ce.item()
            n += cnt.item()
        return tot / max(n, 1.0)

    def save(path, extra):
        return C.save_weights(path, model, {**wmeta, **extra})

    def link_final(epoch):
        final = out_dir / "sft_final.safetensors"
        if final.exists() or final.is_symlink():
            final.unlink()
        final.symlink_to(f"sft_epoch{epoch}.safetensors")

    # position: `step` optimizer steps done; the next one is batch `batch` of epoch `epoch` (rows(epoch, seed) is
    # deterministic and training draws no random numbers, so (epoch, batch) restores the order exactly)
    history = {"val": [], "epochs": [], "stage": stage}
    step, epoch, batch, done = 0, 0, 0, False
    path = (C.latest(ckpt_dir) if resume == "auto" else Path(resume)) if resume else None
    if resume == "auto" and path is None:
        log(f"--resume auto: no checkpoint in {ckpt_dir}, starting fresh", log_file)
    if not resume and C.latest(ckpt_dir) is not None:
        raise SystemExit(f"{ckpt_dir} has checkpoints of an earlier attempt: pass --resume auto to continue it, or "
                         f"move / delete that directory to start over")
    if path is not None:
        ck = C.load_meta(path)
        if ck.get("vocab_sha") != tok.vmap.sha() or ck.get("model_config") != cfg.to_dict():
            raise SystemExit(f"{path}: another model or vocabulary than {src}")
        diff = {k: (ck["plan"].get(k), v) for k, v in plan.items() if ck["plan"].get(k) != v}
        if diff:
            raise SystemExit(f"cannot resume {path} exactly: settings changed (checkpoint, now): {diff}; rerun "
                             f"with the original ones, or move {ckpt_dir} away to start over")
        C.load_weights(model, path)
        C.load_optimizer(opt, path)
        st = ck["state"]
        step, epoch, batch, done, history = st["step"], st["epoch"], st["batch"], st["done"], st["history"]
        log(f"resumed {path}: step {step}/{total_steps}, epoch {epoch + 1} at batch {batch}", log_file)
    else:
        v0 = val_loss()
        history["val"].append({"step": 0, "val_loss": v0})
        log(f"step 0 | {stage} val loss {v0:.4f}", log_file)

    stop, saved = [], [step if path is not None else -1]

    def on_signal(signum, frame):
        if stop:
            sys.exit(130)
        stop.append(signum)            # first: logging can fail if the terminal / tee is already gone
        log(f"signal {signum}: saving a checkpoint after this step, then exiting (again to abort)", log_file)

    def checkpoint():
        t0 = time.time()
        p = C.save_full(ckpt_dir, step, model, opt, {**wmeta, "kind": f"{stage}_full", "plan": plan, "state": {
            "step": step, "epoch": epoch, "batch": batch, "done": done, "history": history}}, keep=keep)
        saved[0], last_save[0] = step, time.time()
        log(f"saved {p} ({time.time() - t0:.1f}s)", log_file)

    def stopped():
        log(f"stopped by signal {stop[0]} at step {step}/{total_steps}: continue with --resume auto", log_file)
        return {**history, "stopped": "signal", "step": step}

    old = {}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            old[sig] = signal.signal(sig, on_signal)
        except ValueError:             # not the main thread: no checkpoint on a signal
            pass
    last_save, t_start = [time.time()], time.time()
    try:
        while not done and step < total_steps and (args.steps or epoch < args.epochs):
            x_all, y_all = train.rows(epoch, args.seed)
            n_steps = len(x_all) // (B * accum)
            if args.steps and n_steps == 0:
                raise SystemExit(f"{data_dir}: fewer rows than one step ({B * accum})")
            while batch < n_steps and step < total_steps:
                set_lr(opt, groups, linear_warmup_decay(step, total_steps, args.warmup))
                t0 = time.time()
                acc, loss_sum, n_sum = None, 0.0, 0.0
                for a in range(accum):
                    i = (batch * accum + a) * B
                    loss, n, grads = micro(mx.array(x_all[i:i + B]), mx.array(y_all[i:i + B]), mx.array(1.0))
                    acc = grads if acc is None else tree_map(mx.add, acc, grads)
                    mx.eval(acc, loss, n)
                    loss_sum += loss.item()
                    n_sum += n.item()
                norm = apply(tree_map(lambda g: g / max(n_sum, 1.0), acc))
                mx.eval(norm, model.parameters(), opt.state)
                step += 1
                batch += 1
                if step % args.log_every == 0 or step == 1:
                    tl = loss_sum / max(n_sum, 1.0)
                    log(f"{stage} step {step}/{total_steps} (epoch {epoch + 1}) | loss {tl:.4f} | gnorm "
                        f"{norm.item():.3f} | {time.time() - t0:.2f}s/step", log_file)
                    with open(metrics, "a") as f:
                        f.write(json.dumps({"kind": f"{stage}_train", "step": step, "loss": tl,
                                            "grad_norm": norm.item()}) + "\n")
                if args.eval_every and step % args.eval_every == 0 and step < total_steps:
                    v = val_loss()
                    history["val"].append({"step": step, "val_loss": v})
                    log(f"{stage} step {step} | val loss {v:.4f}", log_file)
                if stop or time.time() - last_save[0] >= save_every:
                    checkpoint()
                    if stop:
                        return stopped()
            epoch, batch = epoch + 1, 0
            if args.steps:
                continue
            v = val_loss()
            history["val"].append({"step": step, "val_loss": v})
            history["epochs"].append({"epoch": epoch, "step": step, "val_loss": v})
            path = save(out_dir / f"sft_epoch{epoch}.safetensors", {"epoch": epoch, "step": step, "val_loss": v})
            log(f"epoch {epoch} done | val loss {v:.4f} | saved {path}", log_file)
            link_final(min(history["epochs"], key=lambda e: e["val_loss"])["epoch"])  # a crash later keeps a final
            write_json(log_dir / "sft_history.json", history)
            if epoch > 1 and v > history["epochs"][-2]["val_loss"]:
                log("val loss rose in this epoch: keeping the previous epoch as final (plan sections 11, 16)", log_file)
                done = True
            checkpoint()               # epoch boundary: a resume starts the next epoch (or finishes)
            if stop:
                return stopped()
        if args.steps and saved[0] != step:
            checkpoint()               # a rerun with --resume auto then only rewrites the final model
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)
    v0 = history["val"][0]["val_loss"]
    if args.steps:
        v = val_loss()
        history["val"].append({"step": step, "val_loss": v})
        path = save(out_dir / (args.final_name or "final.safetensors"), {"step": step, "val_loss": v,
                                                                         "val_loss_start": v0})
        history["final"] = {"path": str(path), "val_loss": v}
        write_json(log_dir / f"{stage}_history.json", history)
        log(f"{stage} done in {(time.time() - t_start) / 60:.1f} min: val loss {v0:.4f} -> {v:.4f}; saved {path}",
            log_file)
        return history
    best = min(history["epochs"], key=lambda e: e["val_loss"])
    link_final(best["epoch"])
    history["final"] = {"epoch": best["epoch"], "path": str(out_dir / f"sft_epoch{best['epoch']}.safetensors")}
    write_json(log_dir / "sft_history.json", history)
    log(f"SFT done in {(time.time() - t_start) / 3600:.2f} h; final = epoch {best['epoch']} -> "
        f"{out_dir / 'sft_final.safetensors'}", log_file)
    return history
