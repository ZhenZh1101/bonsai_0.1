"""Pretraining S0 -> S1 -> S2 in one run (plan sections 8, 9, 14). Budgets from the day-0 benchmark (scripts.bench):

    # 5-day: Bonsai-0.1, 14 x 640, 1.41B tokens (S0 70M, S1 0.99B, S2 353M)
    python -m scripts.train --model bonsai-0.1 --tokens 1.41e9 --run bonsai-0.1

    # 3-day: Bonsai-0.1-mini, 12 x 576, 1.0B tokens (S0 50M, S1 700M, S2 250M)
    python -m scripts.train --model bonsai-0.1-mini --tokens 1.0e9 --run bonsai-0.1-mini

    # without S0 (the S0 ablation said no): its budget goes to S1
    python -m scripts.train ... --s0-frac 0

    # resume after a crash / sleep (exact data position of every phase, LR continues, no re-warmup)
    python -m scripts.train ... --resume auto
    # loss spike / NaN procedure (plan section 7): use the --lr-scale value printed on the STOP line
    # (0.7 x the LR scale that failed); --lr-scale is absolute, --skip-steps is applied once per
    # (N, --lr-scale) request, so re-running the same command after a crash does not skip again
    python -m scripts.train ... --resume auto --lr-scale 0.49 --skip-steps 200
    # rewind by hand (e.g. a slow divergence): newer checkpoints are moved to ckpt/<run>/abandoned_<time>/
    python -m scripts.train ... --resume ckpt/bonsai-0.1/step_004934

    # Qwen-style AdamW alternative (plan section 8 table)
    python -m scripts.train ... --optimizer adamw --schedule cosine

Shard dirs: a name under tok/ (default s0 / s1 / s2 / val / val_s0), a directory, or a configs/*.yaml (its
`shard_dir`). Shards must carry the tokenizer's vocab_sha, and each phase's shards must hold the tokens the run
plan gives it (S2 incl. half the transition), else the run is refused (--allow-repeat: repeat data). Outputs: ckpt/<run>/ (full checkpoints, milestones,
base.safetensors at the end = <Run>-base), logs/<run>/ (train.log, metrics.jsonl, samples.txt).

Exit codes: 0 finished, 2 stopped after unrecoverable spikes, 3 interrupted by a signal (checkpoint saved).
"""

import argparse
import sys


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("model")
    g.add_argument("--model", default="bonsai-0.1",
                   help="preset: bonsai-0.1 (14x640), bonsai-0.1-mini (12x576), bonsai-0.1-probe (12x256), "
                        "14x576, 12x640, 16x640, tiny")
    g.add_argument("--seq", type=int, default=1024)
    g.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"], help="compute dtype (weights stay fp32)")

    g = ap.add_argument_group("data")
    g.add_argument("--s0", default="s0", help="S0 shards (formal languages, seq_len 512 from the manifest)")
    g.add_argument("--s1", default="s1", help="S1 shards (general)")
    g.add_argument("--s2", default="s2", help="S2 shards (reasoning anneal)")
    g.add_argument("--val-dir", default="val", help="fixed S1 validation set (val loss + bpb)")
    g.add_argument("--val-s0-dir", default="val_s0", help="fixed S0 validation set")
    g.add_argument("--seed", type=int, default=1337)
    g.add_argument("--allow-repeat", action="store_true",
                   help="train even if a phase needs more tokens than its shards hold (the loader repeats data)")

    g = ap.add_argument_group("budget and schedule")
    g.add_argument("--tokens", type=float, default=1.41e9, help="total S0 + S1 + S2 tokens")
    g.add_argument("--max-steps", type=int, default=0, help="override the step count (smoke tests)")
    g.add_argument("--batch-tokens", type=int, default=2 ** 17, help="tokens per optimizer step")
    g.add_argument("--micro-batch", type=int, default=16, help="rows of --seq per micro-step (S0: 2x rows of 512)")
    g.add_argument("--s0-frac", type=float, default=0.05, help="fraction of steps in S0 (0 = no S0)")
    g.add_argument("--s2-frac", type=float, default=0.25, help="fraction of steps in S2 = the LR decay")
    g.add_argument("--transition-steps", type=int, default=200, help="50/50 S1/S2 steps right before S2")
    g.add_argument("--warmup", type=int, default=300)
    g.add_argument("--muon-warmup", type=int, default=300, help="Muon momentum warmup 0.85 -> 0.95 (steps)")
    g.add_argument("--schedule", default="wsd", choices=["wsd", "cosine"], help="cosine: to 10%% of the peak")
    g.add_argument("--optimizer", default="muon", choices=["muon", "adamw"],
                   help="muon = Muon (block matrices) + AdamW (embedding, norms); adamw = Qwen-style AdamW")
    g.add_argument("--matrix-lr", type=float, default=0.02, help="Muon LR at width 768, scaled by (768/d)^0.5")
    g.add_argument("--embed-lr", type=float, default=1e-3, help="AdamW LR for the tied embedding")
    g.add_argument("--norm-lr", type=float, default=1e-3, help="AdamW LR for the RMSNorm weights (see bonsai.optim)")
    g.add_argument("--adamw-lr", type=float, default=None, help="--optimizer adamw peak LR (default by width)")
    g.add_argument("--weight-decay", type=float, default=0.1, help="--optimizer adamw, 2-D weights only")
    g.add_argument("--clip", type=float, default=1.0)
    g.add_argument("--lr-scale", type=float, default=None,
                   help="absolute LR multiplier (default: 1.0, or the checkpoint's value on resume); not compounded")

    g = ap.add_argument_group("monitoring and checkpoints")
    g.add_argument("--run", default="bonsai-0.1", help="bonsai-0.1 (5-day), bonsai-0.1-mini (3-day), ...")
    g.add_argument("--log-every", type=int, default=100)
    g.add_argument("--eval-every", type=int, default=500)
    g.add_argument("--eval-tokens", type=int, default=2 ** 21, help="tokens of the fixed val set per eval")
    g.add_argument("--sample-every", type=int, default=2000)
    g.add_argument("--core-frac", type=float, default=0.25, help="run CORE every this fraction of progress (0=off)")
    g.add_argument("--core-max-per-task", type=int, default=500)
    g.add_argument("--save-every-min", type=float, default=30.0)
    g.add_argument("--save-every", type=int, default=0, help="also save every N steps")
    g.add_argument("--first-save", type=int, default=100, help="extra early checkpoint so spikes can rewind")
    g.add_argument("--keep", type=int, default=3, help="full checkpoints to keep")
    g.add_argument("--milestone-frac", type=float, default=0.1, help="bf16 weight milestone every this fraction")
    g.add_argument("--resume", default="", help="'auto' (latest in ckpt/<run>) or a checkpoint dir")
    g.add_argument("--skip-steps", type=int, default=0,
                   help="skip N steps of data after resuming; applied once per (N, --lr-scale): not repeated when "
                        "the same command is re-run (relaunch once without it, or use another N, to skip again)")
    g.add_argument("--force-reschedule", action="store_true",
                   help="allow a different run plan on resume even after a changed boundary was reached")
    g.add_argument("--spike-norm", type=float, default=3.0, help="plan 7: 3 without softcap (V1: 5)")
    g.add_argument("--switch-grace", type=int, default=200, help="steps after the S0 -> S1 switch not counted as spikes")
    g.add_argument("--auto-recover", type=int, default=2, help="automatic spike recoveries before stopping")
    g.add_argument("--recover-skip-steps", type=int, default=200)
    g.add_argument("--ckpt-root", default="", help="default ckpt/")
    g.add_argument("--log-root", default="", help="default logs/")

    g = ap.add_argument_group("system")
    g.add_argument("--memory-limit-gb", type=float, default=40.0)
    g.add_argument("--no-compile", dest="compile", action="store_false")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    from bonsai.train import STOPPED, Trainer
    if Trainer(args).train() == STOPPED:
        sys.exit(3)  # interrupted (checkpoint saved), distinct from a finished run


if __name__ == "__main__":
    main()
