"""Checkpoint recoverability check (plan sections 8, 14: "存读续训通过").

    python -m scripts.ckpt_check --run bonsai-0.1 [--steps 10] [--checkpoint ckpt/bonsai-0.1/step_000500]

Loads the newest (or the given) full checkpoint -- weights, optimizer state, the loader state of every phase --
into a fresh process with the arguments the run was launched with, runs N training steps on exactly the batches
training would see next (LR and Muon momentum from the run plan), and compares the losses with the run's own
metrics: exactly (< 1e-3) for resumed steps the run logged, else the first loss against the last train loss the
run logged in the same phase (--tolerance; a loss from another phase, e.g. S0 before a checkpoint at the
S0 -> S1 switch, says nothing). Without such a loss, the val loss the run logged at the checkpoint's own step
(the S0 -> S1 switch always has one) is recomputed from the loaded weights (< 1e-3); with neither, the result is
INCONCLUSIVE (exit 0). Exit 1 on a mismatch or a non-finite loss. Nothing is written (logs go to a scratch run
dir that is removed). Running it while training is active steals GPU time for about a minute.
"""

import argparse
import json
import math
import shutil
from pathlib import Path

from bonsai import checkpoint as C
from bonsai.common import CKPT_DIR, LOG_DIR


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="", help="a full checkpoint dir (default: the newest of the run)")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--tolerance", type=float, default=0.25)
    args = ap.parse_args()

    path = Path(args.checkpoint) if args.checkpoint else C.latest(CKPT_DIR / args.run)
    if path is None:
        raise SystemExit(f"no checkpoint in {CKPT_DIR / args.run}")
    meta = C.load_meta(path)
    from scripts.train import build_parser
    targs = build_parser().parse_args([])
    vars(targs).update(meta["args"])  # flags added since the run started keep their defaults
    targs.resume = str(path)
    targs.skip_steps = 0
    targs.lr_scale = None          # the checkpoint's effective LR scale
    log_root = Path(targs.log_root or LOG_DIR)
    run_log = log_root / targs.run
    targs.run = targs.run + "__ckpt_check"   # logs go to a scratch run dir; checkpoints are never written

    import mlx.core as mx
    from mlx.utils import tree_map
    from bonsai.dataloader import Prefetcher
    from bonsai.optim import muon_momentum, set_lr, set_muon_momentum
    from bonsai.train import Trainer

    tr = Trainer(targs)
    start = meta["step"]
    steps = min(args.steps, tr.plan.total - tr.step)
    if steps <= 0:
        raise SystemExit(f"{path.name} is the end of the run ({tr.plan.total} steps): nothing to continue")
    by_step, val_at = {}, None
    mfile = run_log / "metrics.jsonl"
    for line in (mfile.read_text().splitlines() if mfile.exists() else []):
        m = json.loads(line)
        if m.get("kind") == "train":
            by_step[m["step"]] = m
        elif m.get("kind") == "val" and m["step"] == start:
            val_at = m["val_loss"]
    # a train metric at step s is the loss of 0-based step s - 1; the first resumed step is `start`
    phase = tr.plan.phase(start)
    logged = next((by_step[s].get("step_loss", by_step[s]["loss"]) for s in sorted(by_step, reverse=True)
                   if s <= start and by_step[s].get("phase", tr.plan.phase(s - 1)) == phase), None)
    exact_steps = [start + i + 1 for i in range(steps) if "step_loss" in by_step.get(start + i + 1, {})]
    val_now = None
    if not exact_steps and logged is None and val_at is not None:  # before any update changes the weights
        val_now = (tr.evaluate_set(tr.val_dir, targs.eval_tokens) or [None])[0]
    pf = Prefetcher(tr.data, tr.step, tr.accum)
    losses = []
    try:
        for i in range(steps):
            step = tr.step + i
            set_lr(tr.opt, tr.lr_groups, tr.plan.lr(step) * tr.lr_scale)
            set_muon_momentum(tr.opt, muon_momentum(step, targs.muon_warmup))
            acc, ls, ns = None, 0.0, 0.0
            for _ in range(tr.accum):
                _, _, (x, y), _ = pf.get()
                loss, n, g = tr._micro(mx.array(x), mx.array(y), mx.array(1.0))
                acc = g if acc is None else tree_map(mx.add, acc, g)
                mx.eval(acc, loss, n)
                ls += loss.item(); ns += n.item()
            norm = tr._apply(tree_map(lambda v: v / max(ns, 1.0), acc))
            mx.eval(norm, tr.model.parameters(), tr.opt.state)
            losses.append(ls / max(ns, 1.0))
            print(f"step {step} ({tr.plan.phase(step)}): loss {losses[-1]:.4f} gnorm {norm.item():.3f}")
    finally:
        pf.close()
        shutil.rmtree(log_root / targs.run, ignore_errors=True)

    # steps the run itself logged after this checkpoint: the resumed losses must reproduce them exactly
    exact = [(s, losses[s - start - 1], by_step[s]["step_loss"]) for s in exact_steps]
    finite = all(math.isfinite(v) for v in losses)
    print(f"checkpoint {path.name} (step {start}, {phase}): resumed losses {losses[0]:.4f} -> {losses[-1]:.4f}")
    if exact:
        worst = max(abs(a - b) for _, a, b in exact)
        ok = finite and worst < 1e-3
        print(f"{len(exact)} resumed steps were logged by the run; max |diff| {worst:.2e}")
    elif logged is not None:
        ok = finite and abs(losses[0] - logged) < args.tolerance
        print(f"last train loss the run logged in {phase}: {logged:.4f} (tolerance {args.tolerance})")
    elif val_now is not None:
        ok = finite and abs(val_now - val_at) < 1e-3
        print(f"val loss at step {start}: run {val_at:.5f}, recomputed from the checkpoint {val_now:.5f}")
    else:
        if not finite:
            raise SystemExit("PROBLEM: non-finite resumed loss")
        print(f"INCONCLUSIVE: the run logged no {phase} train loss before step {start} and no val loss at it "
              f"(losses are finite); check an older checkpoint (--checkpoint) or the next one")
        return
    print("RECOVERABLE" if ok else "PROBLEM: resumed loss does not continue the logged curve")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
