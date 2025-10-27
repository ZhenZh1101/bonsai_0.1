"""Identity mini-SFT after on-policy distillation (plan section 18): onpolicy_final -> final (Bonsai-0.1).

    python -m scripts.identity_sft                        # run bonsai-0.1, 150 steps on tok/identity_sft
    python -m scripts.identity_sft --run bonsai-0.1-mini --steps 200

On-policy distillation pulls the student towards the Qwen teacher's identity; 100-200 small steps on identity
conversations + general replay (tok/identity_sft, built by scripts.build_sft) fix it. Starts from
ckpt/<run>/onpolicy_final.safetensors (--source / --base to override), LR = --lr-frac (0.075) x the base run's
pretraining peak (half the SFT LR), 16k tokens per step, warmup 10, linear decay. Writes
ckpt/<run>/final.safetensors; check it with scripts.identity_eval. Checkpoints and --resume auto as scripts.sft
(in ckpt/<run>/identity_sft/; exit code 3 = stopped by a signal, checkpoint saved).
"""

import argparse
import sys

from scripts.sft import add_args, resolve_data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap, epochs=1, lr_frac=0.075, warmup=10, batch_tokens=16384)
    ap.add_argument("--steps", type=int, default=150, help="plan 18: 100-200")
    ap.add_argument("--final-name", default="final.safetensors")
    ap.set_defaults(source="onpolicy", data="tok/identity_sft", eval_every=50)
    args = ap.parse_args()
    args.base_run = args.base_run or args.run
    args.data = resolve_data(args.data)
    from bonsai.sft import run_sft
    sys.exit(3 if run_sft(args).get("stopped") == "signal" else 0)


if __name__ == "__main__":
    main()
