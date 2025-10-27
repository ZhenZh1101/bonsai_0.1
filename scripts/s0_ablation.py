"""S0 go / no-go (plan section 9): two Bonsai-0.1-probe runs (12 x 256), with and without S0, same total tokens.

    python -m scripts.s0_ablation                        # 100M tokens each (about 1-2 h each)
    python -m scripts.s0_ablation --tokens 2e6 --model tiny --s0 tok/smoke/s0 --s1 tok/smoke/s1 ...   # smoke

Both runs use the same seed, S1 / S2 shards, schedule and token budget; with S0, its 5% of the steps come out of
S1 (natural-language tokens are deducted, not added). The schedule is the 5-day run's scaled to the probe's
length: LR warmup, Muon momentum warmup and the 50/50 transition keep their share of the run (300 / 300 / 200 of
10,758 steps -> 21 / 21 / 14 of the probe's 763), S2 = the last 25% with the LR decay. With the unscaled 300-step
warmup the probe's whole S0 phase would run at ~6% of the peak LR (it would hardly train), unlike the main
run's S0, which is mostly at peak; a probe checkpoint with another schedule is refused on resume (delete
ckpt/bonsai-0.1-probe-* to start over). Both final models are evaluated on the same fixed val set (tok/val, the
first --final-eval-tokens tokens). S0 stays in the main run only if it is at least 0.02 nat better; otherwise
its budget goes back to S1 (train the main run with --s0-frac 0). Result: logs/s0_ablation.json (incl. the
schedule).

Unknown flags go to scripts.train (shard dirs, --micro-batch, --seq, --ckpt-root, ...). Runs are named
<prefix>-s0 / <prefix>-nos0 and resume automatically: re-run the same command after an interruption.
"""

import argparse
import sys
import time

from bonsai.common import LOG_DIR, log, write_json
from bonsai.optim import round_steps

# the 5-day run whose schedule the probe reproduces in proportion (scripts.train defaults)
MAIN = {"steps": round_steps(1.41e9, 2 ** 17), "warmup": 300, "muon_warmup": 300, "transition_steps": 200}


def probe_schedule(total):
    """Warmup / Muon momentum warmup / transition steps for a `total`-step probe, as shares of the main run."""
    return {k: max(1, round(MAIN[k] * total / MAIN["steps"])) for k in ("warmup", "muon_warmup", "transition_steps")}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)  # "--s0 DIR" belongs to scripts.train, not to --s0-frac
    ap.add_argument("--tokens", type=float, default=1e8, help="total tokens of each run")
    ap.add_argument("--model", default="bonsai-0.1-probe")
    ap.add_argument("--s0-frac", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=None, help="LR warmup steps (default: scaled, see above)")
    ap.add_argument("--muon-warmup", type=int, default=None, help="Muon momentum warmup steps (default: scaled)")
    ap.add_argument("--transition-steps", type=int, default=None, help="50/50 S1/S2 steps (default: scaled)")
    ap.add_argument("--threshold", type=float, default=0.02, help="nat of val loss S0 must win by")
    ap.add_argument("--final-eval-tokens", type=int, default=2 ** 23, help="val tokens for the final comparison")
    ap.add_argument("--run-prefix", default="bonsai-0.1-probe")
    ap.add_argument("--out", default=str(LOG_DIR / "s0_ablation.json"))
    return ap


def probe_args(opts, train_argv, name, frac):
    """(scripts.train args, schedule) of one probe run: the schedule scaled to its length unless given."""
    from scripts.train import build_parser as train_parser
    argv = ["--model", opts.model, "--tokens", str(opts.tokens), "--s0-frac", str(frac),
            "--run", f"{opts.run_prefix}-{name}", "--resume", "auto", "--core-frac", "0", "--sample-every", "0",
            *train_argv]
    args = train_parser().parse_args(argv)
    args.s0_frac, args.run = frac, f"{opts.run_prefix}-{name}"  # never overridden by the pass-through flags
    total = args.max_steps or round_steps(args.tokens, args.batch_tokens)
    schedule = {k: v if getattr(opts, k) is None else getattr(opts, k) for k, v in probe_schedule(total).items()}
    vars(args).update(schedule)
    return args, schedule


def ablation(opts, train_argv, tok=None):
    from bonsai.train import STOPPED, Trainer

    runs, schedule = {}, None
    for name, frac in (("s0", opts.s0_frac), ("nos0", 0.0)):
        args, schedule = probe_args(opts, train_argv, name, frac)
        tr = Trainer(args, tok=tok)
        log(f"S0 ablation: run {args.run} ({tr.plan.describe()})")
        if tr.train() == STOPPED:
            log("interrupted: re-run the same command to resume")
            sys.exit(3)
        loss, bpb, n = tr.evaluate_set(tr.val_dir, opts.final_eval_tokens) or (None, None, 0)
        s0 = tr.evaluate_set(tr.val_s0_dir, opts.final_eval_tokens, s0=True)
        runs[name] = {"run": args.run, "s0_frac": frac, "val_loss": loss, "val_bpb": bpb, "val_targets": n,
                      "val_s0_loss": s0[0] if s0 else None, "steps": tr.plan.total, "plan": tr.plan.to_dict(),
                      "tokens": tr.stats["tokens_seen"]}
        log(f"S0 ablation: {args.run} val loss {loss} (bpb {bpb}), S0 val loss {runs[name]['val_s0_loss']}")
    if runs["s0"]["val_loss"] is None or runs["s0"]["val_targets"] != runs["nos0"]["val_targets"]:
        raise SystemExit("no comparable val set (tok/val missing?)")
    delta = runs["nos0"]["val_loss"] - runs["s0"]["val_loss"]
    keep = delta >= opts.threshold
    out = {"model": opts.model, "tokens": opts.tokens, "threshold_nat": opts.threshold, "runs": runs,
           "schedule": {"steps": tr.plan.total, **schedule, "scaled_from_main_run": MAIN},
           "delta_nat": delta, "keep_s0": keep, "val_dir": str(tr.val_dir), "time": time.time(),
           "verdict": (f"keep S0: {delta:.4f} nat better (>= {opts.threshold}); main run with --s0-frac {opts.s0_frac}"
                       if keep else f"drop S0: {delta:+.4f} nat (< {opts.threshold}); main run with --s0-frac 0, "
                                    f"the S0 budget goes to S1")}
    write_json(opts.out, out)
    log(f"S0 ablation: {out['verdict']} -> {opts.out}")
    return out


def main(argv=None):
    opts, rest = build_parser().parse_known_args(argv)
    ablation(opts, rest)


if __name__ == "__main__":
    main()
