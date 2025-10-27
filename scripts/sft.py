"""Off-policy distillation SFT (plan sections 11, 14): Bonsai-0.1-base -> Bonsai-0.1-sft.

    python -m scripts.sft --data tok/offpolicy --epochs 2                  # run bonsai-0.1 (5-day build)
    python -m scripts.sft --run bonsai-0.1-mini --data tok/offpolicy --epochs 2
    python -m scripts.sft --data tok/offpolicy --epochs 2 --resume auto    # after a crash / Ctrl-C (exit code 3)

Starts from ckpt/<base-run>/base.safetensors (or --base <file>), LR = --lr-frac (0.15) x the base run's peak
LRs, warmup 50, linear decay, 2 epochs; keeps epoch 1 if the val loss rises. Outputs
ckpt/<run>/sft_epoch{1,2}.safetensors and sft_final.safetensors (-> the best epoch); logs/<run>/sft.log.
Full checkpoints every --save-every-min minutes and at each epoch end in ckpt/<run>/sft/ (newest --keep);
--resume auto continues the latest exactly (same batches, LR schedule and optimizer state; starts fresh if there
is none). Without --resume the run refuses a ckpt/<run>/sft/ that already holds checkpoints. Exit codes: 0
finished, 3 stopped by SIGINT / SIGTERM / SIGHUP (checkpoint saved).
"""

import argparse
import sys

from bonsai.common import TOK_DIR


def add_args(ap, epochs=2, lr_frac=0.15, warmup=50, batch_tokens=65536):
    ap.add_argument("--run", default="bonsai-0.1", help="output run (ckpt/<run>/, logs/<run>/)")
    ap.add_argument("--base-run", default="", help="run to start from (default: --run)")
    ap.add_argument("--source", default="base", help="bonsai.load source of the start weights")
    ap.add_argument("--base", default="", help="explicit start weights (.safetensors or checkpoint dir)")
    ap.add_argument("--data", default="tok/offpolicy", help="SFT set dir (tok/<name> or a path)")
    ap.add_argument("--out-dir", default="", help="default ckpt/<run>")
    ap.add_argument("--log-dir", default="", help="default logs/<run>")
    ap.add_argument("--epochs", type=int, default=epochs)
    ap.add_argument("--lr-frac", type=float, default=lr_frac, help="x the pretraining peak LRs (plan: 0.1-0.2)")
    ap.add_argument("--warmup", type=int, default=warmup)
    ap.add_argument("--batch-tokens", type=int, default=batch_tokens)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--memory-limit-gb", type=float, default=40.0)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--resume", default="", help="auto (latest in ckpt/<run>/<stage>/) or a checkpoint dir")
    ap.add_argument("--save-every-min", type=float, default=30.0, help="full checkpoint interval (plan 8: 30 min)")
    ap.add_argument("--keep", type=int, default=3, help="full checkpoints kept")


def resolve_data(data):
    from pathlib import Path
    p = Path(data)
    if p.is_absolute() or p.exists():
        return str(p)
    return str(TOK_DIR / (data[4:] if data.startswith("tok/") else data))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    args = ap.parse_args()
    args.base_run = args.base_run or args.run
    args.data = resolve_data(args.data)
    args.steps, args.final_name = 0, ""
    from bonsai.sft import run_sft
    sys.exit(3 if run_sft(args).get("stopped") == "signal" else 0)


if __name__ == "__main__":
    main()
