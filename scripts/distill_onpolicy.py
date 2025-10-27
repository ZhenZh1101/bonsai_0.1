"""On-policy distillation (plan sections 12, 14): Bonsai-0.1-sft -> onpolicy_final, then scripts.identity_sft.

    python -m scripts.distill_onpolicy --teacher Qwen/Qwen3-0.6B --steps 150 --batch 256     # run bonsai-0.1
    python -m scripts.distill_onpolicy --run bonsai-0.1-mini
    python -m scripts.distill_onpolicy --resume auto                  # after Ctrl-C / SIGHUP / a crash (exit 3)
    python -m scripts.distill_onpolicy --resume auto --lr-frac 0.25   # plan 16 collapse: half the LR, rerun from the
                                                                      #   best healthy checkpoint
    python -m scripts.distill_onpolicy --overwrite                    # start over (the earlier attempt is moved aside)
    python -m scripts.distill_onpolicy --config configs/smoke/onpolicy.yaml --base <sft.safetensors> \\
        --ckpt-root <dir> --log-root <dir>                            # smoke test

Settings come from configs/onpolicy.yaml (--config), then the flags, then --set key=value. Starts from
ckpt/<run>/sft_final.safetensors (--source / --base); prompts data/prompts_onpolicy.jsonl (the plan 14 command's
`--prompts data/prompts.jsonl` would reuse the off-policy prompts: allowed, with a warning); val KL on
data/prompts_heldout.jsonl (also the GSM8K-style monitor's math items); the teacher scores in its off-policy
generation context (the system prompts of configs/offpolicy.yaml, config key system_prompts). Writes
ckpt/<run>/onpolicy_final.safetensors, ckpt/<run>/onpolicy/ (checkpoints, best.safetensors),
logs/<run>/onpolicy.{log,jsonl}, onpolicy_samples.jsonl, onpolicy_summary.json.
A run without --resume refuses a ckpt/<run>/onpolicy/ that holds an earlier attempt (--overwrite moves it, its
onpolicy_final and its jsonl logs aside), so two attempts never mix in the checkpoint rotation or best.safetensors.
--resume auto continues the newest checkpoint; after a health stop (the GSM8K-style monitor or the format check
regressed) or with a changed --lr-frac it goes back to the best healthy checkpoint and drops the later evals;
--resume latest / best / <dir> force the choice. The stop reason is cleared, the LR is recomputed from the
current settings; the step, history and optimizer state come from the checkpoint.
"""

import argparse
import sys
from pathlib import Path

import yaml

from bonsai.common import CKPT_DIR, CONFIG_DIR, LOG_DIR, load_yaml


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_DIR / "onpolicy.yaml"))
    ap.add_argument("--run", default="bonsai-0.1")
    ap.add_argument("--base-run", default="", help="run whose weights to start from (default: --run)")
    ap.add_argument("--source", default=None, help="bonsai.load source of the start weights (config: sft)")
    ap.add_argument("--base", default="", help="explicit start weights (.safetensors or checkpoint dir)")
    ap.add_argument("--teacher", default=None, help="Qwen/Qwen3-0.6B or Qwen3-0.6B (teachers/<name>)")
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--heldout", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr-frac", type=float, default=None, help="x the off-policy stage's LRs (plan: 0.5)")
    ap.add_argument("--micro-tokens", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="any config key (YAML value)")
    ap.add_argument("--resume", default="", help="auto | latest | best | a checkpoint dir (ckpt/<run>/onpolicy/...)")
    ap.add_argument("--overwrite", action="store_true", help="without --resume: move an earlier attempt aside")
    ap.add_argument("--ckpt-root", default="", help="default ckpt/ (outputs go to <root>/<run>)")
    ap.add_argument("--log-root", default="", help="default logs/")
    return ap.parse_args(argv)


def build_config(args):
    from bonsai.distill import DEFAULTS
    cfg = {**DEFAULTS, **(load_yaml(args.config) or {})}
    for key in ("source", "teacher", "prompts", "heldout", "steps", "batch", "lr_frac", "micro_tokens", "eval_every",
                "seed"):
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v
    for kv in args.set:
        k, _, v = kv.partition("=")
        if k not in DEFAULTS:
            raise SystemExit(f"--set {k}: unknown key (known: {', '.join(DEFAULTS)})")
        cfg[k] = yaml.safe_load(v)
    cfg["run"] = args.run
    return cfg


def main(argv=None):
    args = parse_args(argv)
    cfg = build_config(args)
    import mlx.core as mx
    from bonsai.common import log
    from bonsai.distill import OnPolicy, load_records
    from bonsai.load import load_model
    from bonsai.teacher import Teacher, resolve_name

    mx.set_memory_limit(int(float(cfg["memory_limit_gb"]) * 1024 ** 3))
    mx.random.seed(int(cfg["seed"]))
    ckpt_root = Path(args.ckpt_root) if args.ckpt_root else CKPT_DIR
    log_root = Path(args.log_root) if args.log_root else LOG_DIR
    model, tok, meta = load_model(args.base_run or args.run, args.base or cfg["source"], ckpt_dir=ckpt_root)
    if Path(str(cfg["prompts"])).name == "prompts.jsonl":
        log("WARNING: --prompts is the off-policy pool (data/prompts.jsonl); plan 12 resamples it: the default "
            "data/prompts_onpolicy.jsonl holds fresh prompts")
    prompts = load_records(cfg["prompts"])
    heldout = load_records(cfg["heldout"], int(cfg["val_prompts"]), seed=0) if cfg["val_prompts"] else []
    teacher = Teacher(resolve_name(cfg["teacher"]))
    trainer = OnPolicy(cfg, model, tok, teacher, prompts, heldout, meta, ckpt_root / args.run, log_root / args.run,
                       src=meta.get("path", ""))
    out = trainer.run(resume=args.resume or None, overwrite=args.overwrite)
    sys.exit(3 if out.get("stopped") == "signal" else 0)


if __name__ == "__main__":
    main()
