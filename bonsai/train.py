"""Pretraining: S0 -> S1 -> S2 in one run on one LR curve (plan sections 4, 5, 7, 8, 9, 14).

Steps of 2^17 tokens (micro-batches + gradient accumulation; S0 micro-batches have more, shorter rows). The
schedule (bonsai.optim.RunPlan): warmup 300 steps from step 0, constant, linear decay to 0 over S2; S0 is the
first 5% of the steps (hard switch to S1), S2 the last 25%, with a 200-step 50/50 S1/S2 transition before the
decay. Muon + AdamW (or Qwen-style AdamW), global-norm clipping at 1.0, bf16 compute with fp32 master weights.
At start every phase's token need (RunPlan.phase_tokens) is checked against its shards: a phase that would
repeat data is refused unless --allow-repeat.

Monitoring: train loss / grad norm / tok/s / peak memory / phase every `log_every` steps; val loss + bpb on
tok/val every `eval_every`; S0 val loss on tok/val_s0 while in S0; at the S0 -> S1 boundary the S0 val loss,
the S1 val loss and bonsai.formal.dyck.dyck_accuracy (plan 14 gate: k-Dyck and k-Shuffle-Dyck closing accuracy
> 95%; the valid rates are logged with their constant-predictor baselines when dyck_accuracy reports them) plus
a full checkpoint; fixed-prompt samples every `sample_every`; CORE at every 25% of progress (bonsai.core_eval,
when present). Optional monitors (engine, CORE, Dyck) are imported lazily: a missing module is skipped, a failing
one is logged and never stops the run.

Health (plan section 7): a non-finite loss / grad norm, or a grad norm > --spike-norm (3) that is a jump over the
recent median on 3 consecutive post-warmup steps, rolls back to the last full checkpoint this process loaded or
saved with LR x0.7 and 200 skipped steps of data (at most --auto-recover 2 times, then exit 2 with the manual
command). Scheduled saves wait while a spike streak is open, so the rollback target never contains its updates.
The first --switch-grace steps after the S0 -> S1 switch do not count as spikes: the switch to natural language
makes the gradient norm jump by design (it is still clipped).

Checkpoints: full every `save_every_min` minutes and at step `first_save` (newest 3 kept), bf16 milestones every
10%, a save on SIGINT / SIGTERM / SIGHUP. Checkpoints newer than the step training starts from (an abandoned
branch after --resume <older dir>) are moved to ckpt/<run>/abandoned_<time>/. At the end
ckpt/<run>/base.safetensors (bf16, meta incl. model_config and vocab_sha) = Bonsai-0.1-base, and
logs/<run>/pretrain_done.json.
"""

import importlib
import json
import math
import signal
import statistics
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from bonsai import checkpoint as C
from bonsai.common import CKPT_DIR, LOG_DIR, PROMPT_DIR, TOK_DIR, load_yaml, log, write_json
from bonsai.dataloader import Prefetcher, StagedBatches, read_manifest, val_batches
from bonsai.model import Bonsai, ModelConfig, make_caches, token_losses
from bonsai.optim import (RunPlan, build_optimizer, clip_by_global_norm, muon_momentum, resolve_tc, round_steps,
                          set_lr, set_muon_momentum)

DTYPES = {"bf16": mx.bfloat16, "fp32": mx.float32}
STOPPED = "stopped"
DONE = "done"


def resolve_shard_dir(spec):
    """configs/sN.yaml -> tok/<shard_dir>; an existing directory as is; otherwise a directory name under tok/."""
    if not spec:
        return None
    p = Path(spec)
    if p.suffix in (".yaml", ".yml"):
        return TOK_DIR / load_yaml(p)["shard_dir"]
    return p if p.is_dir() else TOK_DIR / spec


def optional_import(module, name):
    """getattr(module, name), or None when the module (another builder's) does not exist."""
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as e:
        if e.name and (module == e.name or module.startswith(e.name + ".")):
            return None
        raise
    return getattr(mod, name, None)


def display_name(run):
    return run.replace("bonsai", "Bonsai", 1) if run.startswith("bonsai") else run


def data_shortfalls(plan, tokens_per_step, available, s2_share=0.5):
    """[(phase, tokens the run plan needs, tokens its shards hold)] for every phase that would repeat data;
    `available`: {phase: total_tokens} of the phases that have shards."""
    need = plan.phase_tokens(tokens_per_step, s2_share)
    return [(k, need[k], available[k]) for k in need if available.get(k) is not None and need[k] > available[k]]


def build_step_fns(model, opt, clip, token_bytes, compile=True):
    """(micro, apply, evaluate) for gradient accumulation; shared by pretraining and SFT.

    micro(x, y, scale) -> (sum of token losses * scale, n targets, grads of that sum)
    apply(grads)       -> global grad norm (before clipping); clips and updates in place
    evaluate(x, y)     -> (sum ce, n targets, sum ce over byte-carrying targets, n bytes)
    """
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
        nbytes = token_bytes[mx.where(valid, y, 0)] * valid
        return ce.sum(), valid.sum(), (ce * (nbytes > 0)).sum(), nbytes.sum()

    if not compile:
        return micro, apply, evaluate
    state = [model.state, opt.state]
    # nn.value_and_grad writes the traced parameters into the model, so `micro` must
    # also list the model state as an output or the model is left holding tracers
    return (mx.compile(micro, inputs=[model.state], outputs=[model.state]),
            mx.compile(apply, inputs=state, outputs=state),
            mx.compile(evaluate, inputs=[model.state]))


def greedy(model, ids, max_new_tokens, stop_ids):
    """Tiny KV-cache greedy loop (used when bonsai.engine is not there)."""
    caches = make_caches(model)
    logits = model(mx.array([ids]), caches)
    out = []
    for _ in range(max_new_tokens):
        t = int(mx.argmax(logits[0, -1]).item())
        if t in stop_ids:
            break
        out.append(t)
        logits = model(mx.array([[t]]), caches)
    return out


class Trainer:
    def __init__(self, args, tok=None):
        self.args = a = args
        mx.set_memory_limit(int(a.memory_limit_gb * 1024 ** 3))
        mx.random.seed(a.seed)
        if tok is None:
            from bonsai.tokenizer import get_tokenizer
            tok = get_tokenizer()
        self.tok, self.vocab_sha = tok, tok.vmap.sha()
        self.cfg = ModelConfig.from_preset(a.model, vocab_size=tok.vocab_size, seq_len=a.seq)
        self.model = Bonsai(self.cfg, compute_dtype=DTYPES[a.dtype])
        mx.eval(self.model.parameters())

        self.tokens_per_step = a.batch_tokens
        self.micro_tokens = a.micro_batch * a.seq
        if self.tokens_per_step % self.micro_tokens:
            raise SystemExit(f"--batch-tokens {a.batch_tokens} must be a multiple of micro-batch x seq {self.micro_tokens}")
        self.accum = self.tokens_per_step // self.micro_tokens
        total = a.max_steps or round_steps(a.tokens, self.tokens_per_step)
        self.plan = RunPlan.build(total, a.s0_frac, a.s2_frac, a.transition_steps, a.warmup, a.schedule)
        self.run_dir = Path(a.ckpt_root or CKPT_DIR) / a.run
        self.log_dir = Path(a.log_root or LOG_DIR) / a.run
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "train.log"
        self.metrics_file = self.log_dir / "metrics.jsonl"

        self.tc = resolve_tc({"optimizer": a.optimizer, "matrix_lr": a.matrix_lr, "embed_lr": a.embed_lr,
                              "norm_lr": a.norm_lr, "adamw_lr": a.adamw_lr, "weight_decay": a.weight_decay},
                             self.cfg.d_model)
        self.opt, self.lr_groups = build_optimizer(self.model, self.tc)

        dirs = {k: resolve_shard_dir(getattr(a, k)) for k in ("s0", "s1", "s2")}
        try:
            self.data = StagedBatches(self.plan, dirs, a.micro_batch, a.seq, a.seed, self.vocab_sha)
        except FileNotFoundError as e:
            hint = {"s0": " (--s0-frac 0 trains without S0)", "s2": " (--s2-frac 0 trains without S2)"}
            which = next((k for k, d in dirs.items() if d and str(d) in str(e)), "")
            raise SystemExit(f"{e}{hint.get(which, '')}")
        except ValueError as e:
            raise SystemExit(str(e))
        B = a.micro_batch
        short = data_shortfalls(self.plan, self.tokens_per_step,
                                {k: ld.shards.total_tokens for k, ld in self.data.loaders.items()}, (B - B // 2) / B)
        if short:
            msg = "; ".join(f"{k.upper()} needs {n / 1e6:.1f}M tokens but {dirs[k]} holds {h / 1e6:.1f}M"
                            for k, n, h in short)
            if not a.allow_repeat:
                raise SystemExit(f"{msg} ({self.plan.describe()}): the loader would start a second epoch. Build more "
                                 f"(configs/<phase>.yaml total_tokens), lower --tokens, or pass --allow-repeat.")
            log(f"WARNING: {msg}: those phases repeat data (--allow-repeat)", self.log_file)
        self.val_dir = resolve_shard_dir(a.val_dir)
        self.val_s0_dir = resolve_shard_dir(a.val_s0_dir)
        self.token_bytes = mx.array(tok.token_bytes(), dtype=mx.float32)

        self.step = 0
        self.lr_scale = 1.0 if a.lr_scale is None else a.lr_scale
        self.stats = {"tokens_seen": 0, "train_time": 0.0, "best_val_bpb": None, "recoveries": 0, "skipped_steps": 0,
                      "core_done": 0, "milestones_done": 0, "skip_armed": None, "s0_done": False}
        self._dirty = False  # startup changed the state (skip / LR override): save before the first step
        self._skipped_monitors = set()
        self._last_ckpt = None  # the checkpoint this process last loaded or saved = the spike rollback target
        if a.resume:
            self._resume(a.resume)
        moved = C.set_aside_newer(self.run_dir, self.step)
        if moved:
            log(f"WARNING: training continues from step {self.step}, so the newer checkpoints "
                f"{', '.join(p.name for p in moved)} are an abandoned branch: moved to {moved[0].parent}",
                self.log_file)
        if a.skip_steps:
            self._skip_once(a.skip_steps)
        else:
            self.stats["skip_armed"] = None  # a launch without the flag re-arms it
        self._build_compiled()
        self._stop_requested = False
        self._last_save = time.time()

    # ------------------------------------------------------------------ setup
    def _on_signal(self, signum, frame):
        if self._stop_requested:
            sys.exit(130)
        self._stop_requested = True  # first: logging can fail if the terminal / tee is already gone
        log(f"signal {signum} received: will save a checkpoint after this step and exit (press again to abort)",
            self.log_file)

    def _build_compiled(self):
        self._micro, self._apply, self._eval = build_step_fns(self.model, self.opt, self.args.clip,
                                                              self.token_bytes, self.args.compile)

    def _resume(self, which):
        path = C.latest(self.run_dir) if which == "auto" else Path(which)
        if path is None:
            log(f"--resume auto: no checkpoint in {self.run_dir}, starting fresh", self.log_file)
            return
        meta = C.load_meta(path)
        if meta["model_config"] != self.cfg.to_dict():
            raise SystemExit(f"checkpoint config {meta['model_config']} != current {self.cfg.to_dict()}")
        if meta.get("vocab_sha") != self.vocab_sha:
            raise SystemExit(f"checkpoint {path} was trained with vocabulary subset {meta.get('vocab_sha')!r}, "
                             f"the tokenizer has {self.vocab_sha!r}")
        ckpt_opt = (meta.get("optim") or {}).get("optimizer", "muon")
        if ckpt_opt != self.tc["optimizer"]:
            raise SystemExit(f"checkpoint {path} uses --optimizer {ckpt_opt}, not {self.tc['optimizer']}")
        C.load_weights(self.model, path)
        C.load_optimizer(self.opt, path)
        for w in self.data.load_state_dict(meta["data_state"]):
            log(f"WARNING: {w}", self.log_file)
        self.step = meta["step"]
        self._last_ckpt = path
        self.stats.update(meta.get("stats", {}))
        # --lr-scale is absolute and never compounds. Re-running the command that wrote this checkpoint keeps
        # the checkpoint's value (which may include later automatic 0.7x reductions).
        cli, prev = self.args.lr_scale, (meta.get("args") or {}).get("lr_scale")
        self.lr_scale = meta.get("lr_scale", 1.0) if cli is None or cli == prev else cli
        if self.lr_scale != meta.get("lr_scale", 1.0):
            self._dirty = True
        old = RunPlan(**meta["plan"])
        if old != self.plan:
            moved = [k for k in ("s0_end", "transition", "decay", "total")
                     if getattr(old, k) != getattr(self.plan, k) and min(getattr(old, k), getattr(self.plan, k)) <= self.step]
            if (old.warmup, old.schedule) != (self.plan.warmup, self.plan.schedule) and self.step > 0:
                moved.append("warmup/schedule")
            if moved and not self.args.force_reschedule:
                raise SystemExit(
                    f"refusing to change the run plan at step {self.step}: {', '.join(moved)} already reached "
                    f"(checkpoint: {old.describe()}; now: {self.plan.describe()}). Resume with the original "
                    f"--tokens / --max-steps / --s0-frac / --s2-frac, or pass --force-reschedule.")
            log(f"note: run plan changed: {old.describe()} -> {self.plan.describe()}", self.log_file)
            self._dirty = True
        log(f"resumed from {path} at step {self.step} (lr_scale {self.lr_scale:.3f})", self.log_file)

    def _skip_data(self, n_steps):
        """Advance the data cursors by n optimizer steps without training (spike recovery)."""
        for s in range(self.step, self.step + n_steps):
            for _ in range(self.accum):
                self.data.batch(s)
        self.stats["skipped_steps"] += n_steps
        log(f"skipped {n_steps} steps of data", self.log_file)

    def _skip_once(self, n_steps):
        """Apply --skip-steps unless the run that wrote this checkpoint already applied the same request. The
        request is (N, --lr-scale) and is persisted, so re-running the same command after a crash does not skip
        again, while the command of a later STOP line (always a new --lr-scale) does. Relaunch once without the
        flag, or use another N, to repeat a request."""
        key = [n_steps, self.args.lr_scale]
        if self.stats.get("skip_armed") == key:
            log(f"--skip-steps {n_steps} --lr-scale {self.args.lr_scale} was already applied by the run that wrote "
                f"this checkpoint; not skipping again", self.log_file)
            return
        self._skip_data(n_steps)
        self.stats["skip_armed"] = key
        self._dirty = True

    # ---------------------------------------------------------------- helpers
    def meta(self, data_state):
        return {"model_config": self.cfg.to_dict(), "vocab_sha": self.vocab_sha, "data_state": data_state,
                "stats": self.stats, "lr_scale": self.lr_scale, "plan": self.plan.to_dict(),
                "total_steps": self.plan.total, "optim": self.tc, "args": vars(self.args),
                "tokens_per_step": self.tokens_per_step, "phase": self.plan.phase(min(self.step, self.plan.total - 1))}

    def weights_meta(self, kind):
        return {"model_config": self.cfg.to_dict(), "vocab_sha": self.vocab_sha, "kind": kind,
                "name": f"{display_name(self.args.run)}-{kind}", "run": self.args.run, "step": self.step,
                "tokens": self.stats["tokens_seen"], "plan": self.plan.to_dict(), "optim": self.tc,
                "lr_scale": self.lr_scale, "args": vars(self.args)}

    def metric(self, **kv):
        kv["step"] = self.step
        kv["time"] = time.time()
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(kv) + "\n")

    def _monitor(self, what, fn):
        """Run an optional monitor; never let it kill the run."""
        try:
            return fn()
        except Exception as e:
            log(f"monitor '{what}' failed at step {self.step}: {type(e).__name__}: {e}", self.log_file)
            with open(self.log_dir / "monitor_errors.log", "a") as f:
                f.write(f"--- step {self.step} {what}\n{traceback.format_exc()}\n")
            return None

    def _skip_note(self, what, why):
        if what not in self._skipped_monitors:
            self._skipped_monitors.add(what)
            log(f"{what} skipped: {why}", self.log_file)

    def evaluate_set(self, d, max_tokens, s0=False):
        """(loss, bpb, targets) on a fixed shard dir; None when it does not exist."""
        if d is None or not (Path(d) / "manifest.json").exists():
            self._skip_note(f"val on {d}", "no shards")
            return None
        seq = (read_manifest(d).get("seq_len") if s0 else None) or self.args.seq
        rows = max(1, self.micro_tokens // seq)
        tot_ce = tot_n = tot_ce_b = tot_bytes = 0.0
        for x, y in val_batches(d, rows, seq, max_tokens, self.vocab_sha):
            ce, n, ce_b, nb = self._eval(mx.array(x), mx.array(y))
            mx.eval(ce, n, ce_b, nb)
            tot_ce += ce.item(); tot_n += n.item(); tot_ce_b += ce_b.item(); tot_bytes += nb.item()
        return tot_ce / max(tot_n, 1), tot_ce_b / (math.log(2) * max(tot_bytes, 1)), tot_n

    def evaluate_val(self):
        res = self.evaluate_set(self.val_dir, self.args.eval_tokens)
        if res is None:
            return None
        vl, bpb, _ = res
        best = self.stats["best_val_bpb"]
        self.stats["best_val_bpb"] = bpb if best is None else min(best, bpb)
        log(f"step {self.step} | val loss {vl:.4f} | val bpb {bpb:.4f}", self.log_file)
        self.metric(kind="val", val_loss=vl, val_bpb=bpb, phase=self.plan.phase(max(0, self.step - 1)))
        return vl, bpb

    def evaluate_val_s0(self):
        res = self.evaluate_set(self.val_s0_dir, self.args.eval_tokens, s0=True)
        if res is None:
            return None
        log(f"step {self.step} | S0 val loss {res[0]:.4f}", self.log_file)
        self.metric(kind="val_s0", val_loss=res[0], targets=res[2])
        return res[0]

    def sample(self):
        path = PROMPT_DIR / "fixed_prompts.json"
        if not path.exists():
            return self._skip_note("samples", f"{path} missing")
        prompts = json.loads(path.read_text()).get("pretrain", [])
        generate = optional_import("bonsai.engine", "generate")
        stop = [self.tok.bos_id]
        out = []
        for p in prompts:
            ids = self.tok.encode(p, bos=True)
            if generate is not None:
                gen = generate(self.model, self.tok, ids, max_new_tokens=80, temperature=0.0, stop_ids=stop)
                gen = gen[:-1] if gen and gen[-1] in stop else gen  # the engine keeps the stop id
            else:
                gen = greedy(self.model, ids, 80, set(stop))
            out.append(f"--- {p!r}\n{self.tok.decode(gen)}")
        text = f"===== step {self.step} ({self.plan.phase(max(0, self.step - 1))}) =====\n" + "\n".join(out) + "\n"
        with open(self.log_dir / "samples.txt", "a") as f:
            f.write(text)
        log(text.rstrip(), None)

    def run_core(self):
        evaluate_core = optional_import("bonsai.core_eval", "evaluate_core")
        if evaluate_core is None:
            return self._skip_note("CORE", "bonsai.core_eval not available")
        t0 = time.monotonic()
        res = evaluate_core(self.model, self.tok, max_per_task=self.args.core_max_per_task)
        took = time.monotonic() - t0
        self.metric(kind="core", core=res.get("core"), tasks=res.get("tasks"), seconds=took)
        log(f"step {self.step} | CORE {res.get('core')} ({took:.0f}s)", self.log_file)

    def run_dyck(self):
        dyck_accuracy = optional_import("bonsai.formal.dyck", "dyck_accuracy")
        if dyck_accuracy is None:
            return self._skip_note("Dyck accuracy", "bonsai.formal.dyck not available")
        res = dyck_accuracy(self.model, self.tok)
        self.metric(kind="dyck", **{k: v for k, v in res.items() if isinstance(v, (int, float, str, bool))})
        log(f"step {self.step} | Dyck {json.dumps(res, default=str)[:400]}", self.log_file)
        # plan 14 gate on the closing positions: an 'always open' constant predictor already scores ~97% (k-Dyck)
        # / ~94% (shuffle) *_valid_rate, so the valid rates are only reported, next to their baselines if given
        gate = [res.get(k) for k in ("kdyck_close_acc", "shuffle_close_acc")]
        if all(isinstance(g, (int, float)) for g in gate):
            valid = []
            for name in ("kdyck", "shuffle"):
                v = res.get(f"{name}_valid_rate")
                base = next((b for k, b in res.items() if k.startswith(name) and "valid" in k and "baseline" in k
                             and isinstance(b, (int, float))), None)
                if isinstance(v, (int, float)):
                    valid.append(f"{name} {v:.1%}" + (f" (constant predictor {base:.1%})" if base is not None else ""))
            log(f"step {self.step} | S0 gate (plan 14, > 95%): k-Dyck closing {gate[0]:.1%}, shuffle-Dyck closing "
                f"{gate[1]:.1%} -> {'PASS' if min(gate) > 0.95 else 'BELOW TARGET'}; valid rate "
                f"{', '.join(valid) or '-'}", self.log_file)

    def s0_boundary(self):
        """End of S0 (plan 14 gate: Dyck accuracy > 95%): S0 val, S1 val, Dyck, then a full checkpoint."""
        log(f"step {self.step} | S0 -> S1 switch", self.log_file)
        self.metric(kind="phase_switch", frm="s0", to="s1")
        self._monitor("S0 val", self.evaluate_val_s0)
        self._monitor("val", self.evaluate_val)
        self._monitor("dyck", self.run_dyck)
        self.stats["s0_done"] = True

    def save(self, data_state, milestone=False):
        t0 = time.time()
        path = self._last_ckpt = C.save_full(self.run_dir, self.step, self.model, self.opt, self.meta(data_state),
                                             keep=self.args.keep)
        msg = f"saved {path.name} ({time.time() - t0:.1f}s)"
        if milestone:
            C.save_milestone(self.run_dir, self.step, self.model, self.weights_meta("milestone"))
            msg += " + bf16 milestone"
        log(msg, self.log_file)

    def recover(self, reason):
        """Plan section 7: reload the last full checkpoint of this branch (the one this process last loaded or
        saved, never one written during the spike streak), LR x0.7, skip 200 steps of data.

        Each recovery compounds on the LR actually in use and is saved right away, so a second spike
        rewinds to the reduced state and skips fresh data instead of replaying the first recovery."""
        path = self._last_ckpt
        if path is None or not Path(path).exists() or self.stats["recoveries"] >= self.args.auto_recover:
            log(f"STOP: {reason}. No automatic recovery left. Resume manually with `--resume auto --lr-scale "
                f"{0.7 * self.lr_scale:.3f} --skip-steps {self.args.recover_skip_steps}`.", self.log_file)
            sys.exit(2)
        meta = C.load_meta(path)
        C.load_weights(self.model, path)
        C.load_optimizer(self.opt, path)
        self.data.load_state_dict(meta["data_state"])
        self.step = meta["step"]
        self.stats = {**self.stats, **meta["stats"], "recoveries": self.stats["recoveries"] + 1}
        self.lr_scale = min(self.lr_scale, meta.get("lr_scale", 1.0)) * 0.7
        log(f"RECOVER ({reason}): back to step {self.step}, lr_scale -> {self.lr_scale:.3f}", self.log_file)
        self.metric(kind="recover", reason=reason, lr_scale=self.lr_scale)
        self._skip_data(self.args.recover_skip_steps)
        self.save(self.data.state_dict())  # persist the reduced LR and the advanced cursor
        self._last_save = time.time()

    def _cadence(self, frac):
        return max(1, int(round(1 / frac))) if frac and frac > 0 else 0

    def _crossed(self, done, n):
        """Has the run reached the (done+1)-th of n equally spaced points of the total?"""
        return n and done < n and self.step >= math.ceil((done + 1) * self.plan.total / n)

    def finish(self):
        base = C.save_weights(self.run_dir / "base.safetensors", self.model, self.weights_meta("base"))
        log(f"training done: {self.stats['tokens_seen'] / 1e9:.3f}B tokens, "
            f"{self.stats['train_time'] / 3600:.2f} h of step time; {display_name(self.args.run)}-base -> {base}",
            self.log_file)
        write_json(self.log_dir / "pretrain_done.json",
                   {"step": self.step, "total_steps": self.plan.total, "tokens": self.stats["tokens_seen"],
                    "best_val_bpb": self.stats["best_val_bpb"], "time": time.time(), "base": str(base),
                    "checkpoint": str(C.latest(self.run_dir))})

    # -------------------------------------------------------------- main loop
    def train(self):
        """Returns DONE when the last step finished, STOPPED after a signal-triggered save."""
        old = {sig: signal.signal(sig, self._on_signal) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        try:
            return self._train()
        finally:
            for sig, h in old.items():
                signal.signal(sig, h)

    def _train(self):
        a, plan = self.args, self.plan
        n = self.model.num_params()
        log(f"model {a.model} {self.cfg.to_dict()}", self.log_file)
        log(f"params total {n['total'] / 1e6:.1f}M, non-embedding {n['non_embedding'] / 1e6:.1f}M; {plan.total} steps x "
            f"{self.tokens_per_step} tokens (micro {a.micro_batch}x{a.seq}, S0 rows "
            f"{self.data.rows.get('s0', '-')}, accum {self.accum}); {plan.describe()}", self.log_file)
        log(f"optimizer {self.tc['optimizer']}, peak LRs {self.lr_groups}, lr_scale {self.lr_scale:.3f}", self.log_file)
        n_milestone = self._cadence(a.milestone_frac)
        n_core = self._cadence(a.core_frac)
        if self._dirty:  # a resume-time skip or LR change must survive a crash before the next timed save
            self.save(self.data.state_dict())
            self._dirty = False
        self._last_save = time.time()
        window = {"t": time.time(), "tokens": 0, "loss": 0.0, "n": 0}
        gnorms = deque(maxlen=50)  # recent gradient norms, the baseline for spike detection
        mx.reset_peak_memory()
        scale = mx.array(1.0)

        while self.step < plan.total:
            prefetch = Prefetcher(self.data, self.step, self.accum)
            trouble = None
            spikes = 0
            save_pending = False
            try:
                while self.step < plan.total:
                    step = self.step
                    phase = plan.phase(step)
                    lrm = plan.lr(step) * self.lr_scale
                    set_lr(self.opt, self.lr_groups, lrm)
                    set_muon_momentum(self.opt, muon_momentum(step, a.muon_warmup))

                    t0 = time.time()
                    acc, loss_sum, tok_sum, data_state = None, 0.0, 0.0, None
                    for _ in range(self.accum):
                        s, _, (x, y), data_state = prefetch.get()
                        assert s == step
                        loss, ntok, grads = self._micro(mx.array(x), mx.array(y), scale)
                        acc = grads if acc is None else tree_map(mx.add, acc, grads)
                        mx.eval(acc, loss, ntok)
                        loss_sum += loss.item()
                        tok_sum += ntok.item()
                    norm = self._apply(tree_map(lambda g: g / max(tok_sum, 1.0), acc))
                    mx.eval(norm, self.model.parameters(), self.opt.state)
                    dt = time.time() - t0

                    train_loss = loss_sum / max(tok_sum, 1)
                    gnorm = norm.item()
                    self.step += 1
                    self.stats["tokens_seen"] += self.tokens_per_step
                    self.stats["train_time"] += dt

                    # --- health checks (plan section 7)
                    if not math.isfinite(train_loss) or not math.isfinite(gnorm):
                        trouble = f"non-finite loss/grad at step {step}"
                        break
                    # a fresh model's gradient norm is naturally large and is clipped anyway: count spikes only after
                    # warmup (and the S0 -> S1 grace window), and only as a jump over the recent median
                    if plan.s0_end and step == plan.s0_end:
                        gnorms.clear()  # S1 gets its own baseline
                    grace = plan.s0_end and plan.s0_end <= step < plan.s0_end + a.switch_grace
                    jump = len(gnorms) < 20 or gnorm > 3 * statistics.median(gnorms)
                    gnorms.append(gnorm)
                    if gnorm > a.spike_norm and step >= a.warmup and not grace and jump:
                        spikes += 1
                        log(f"warning: grad norm {gnorm:.2f} > {a.spike_norm} at step {step} ({spikes} in a row)", self.log_file)
                        if spikes >= 3:
                            trouble = f"grad norm > {a.spike_norm} for 3 consecutive steps at step {step}"
                            break
                    else:
                        spikes = 0

                    window["tokens"] += self.tokens_per_step
                    window["loss"] += train_loss
                    window["n"] += 1
                    if self.step % a.log_every == 0 or self.step <= 3:
                        el = time.time() - window["t"]
                        tps = window["tokens"] / max(el, 1e-9)
                        peak = mx.get_peak_memory() / 1024 ** 3
                        avg = window["loss"] / window["n"]
                        log(f"step {self.step:6d}/{plan.total} | loss {avg:.4f} | gnorm {gnorm:.3f} | lr x{lrm:.3f} | "
                            f"{phase} | {tps:,.0f} tok/s | {dt:.2f}s/step | peak {peak:.1f}GB | "
                            f"{self.stats['tokens_seen'] / 1e9:.3f}B tok", self.log_file)
                        self.metric(kind="train", loss=avg, step_loss=train_loss, grad_norm=gnorm, lr_mult=lrm,
                                    tok_per_s=tps, step_time=dt, peak_mem_gb=peak, phase=phase, targets=tok_sum,
                                    tokens=self.stats["tokens_seen"], epochs=self.data.epochs(), total_steps=plan.total)
                        window = {"t": time.time(), "tokens": 0, "loss": 0.0, "n": 0}

                    last = self.step == plan.total
                    t_block = time.time()
                    boundary = bool(plan.s0_end) and self.step >= plan.s0_end and not self.stats["s0_done"]
                    if boundary:
                        self.s0_boundary()
                    if a.eval_every and (self.step % a.eval_every == 0 or last) and not boundary:
                        self._monitor("val", self.evaluate_val)
                        if phase == "s0":
                            self._monitor("S0 val", self.evaluate_val_s0)
                    if a.sample_every and (self.step % a.sample_every == 0 or last):
                        self._monitor("samples", self.sample)
                    if self._crossed(self.stats["core_done"], n_core):
                        self._monitor("CORE", self.run_core)
                        self.stats["core_done"] += 1
                    milestone = bool(self._crossed(self.stats["milestones_done"], n_milestone))
                    timed = (time.time() - self._last_save) / 60 >= a.save_every_min
                    stepped = (a.save_every and self.step % a.save_every == 0) or self.step == a.first_save
                    due = milestone or timed or stepped or save_pending
                    if (due and not spikes) or last or boundary or self._stop_requested:
                        if milestone:
                            self.stats["milestones_done"] += 1
                        self.save(data_state, milestone=milestone or boundary)
                        self._last_save = time.time()
                        save_pending = False
                    elif due:  # inside a spike streak: a checkpoint now could become the rollback target
                        save_pending = True
                    window["t"] += time.time() - t_block  # tok/s in the log excludes eval / CORE / save time
                    if self._stop_requested:
                        log("stopped by signal after saving", self.log_file)
                        return STOPPED
            finally:
                prefetch.close()  # the loader thread must be stopped before its state is rewound
            if trouble is None:
                break
            self.recover(trouble)
            gnorms.clear()
        self.finish()
        return DONE
