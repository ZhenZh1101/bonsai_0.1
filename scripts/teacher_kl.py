"""KL to the teacher (plan section 15): mean per-token KL(student || teacher) on held-out prompts, recorded for the
base, sft and onpolicy stages (and final).

    python -m scripts.teacher_kl --run bonsai-0.1 --source base
    python -m scripts.teacher_kl --run bonsai-0.1 --source sft
    python -m scripts.teacher_kl --run bonsai-0.1 --source onpolicy
    python -m scripts.teacher_kl --run bonsai-0.1 --source final --n 200

The student samples its own answers to data/prompts_heldout.jsonl (bonsai.distill.measure_kl: the mode and
signal of each prompt fixed by its id, math prompts with the python tool, Qwen3 sampling presets with a fixed
seed, the on-policy run's scoring and masks from configs/onpolicy.yaml); the teacher (Qwen3-0.6B) scores them
over the student's vocabulary subset in its off-policy generation context (the teacher-only system prompt of
each prompt's kind, configs/offpolicy.yaml; `teacher_context` in the output lists the kinds, [] = none). Writes
logs/<run>/teacher_kl_<source>.json and merges the headline numbers into logs/<run>/teacher_kl.json
({source: {...}}).
"""

import argparse
from pathlib import Path

from bonsai.common import CONFIG_DIR, LOG_DIR, load_yaml, log, read_json, write_json


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="bonsai-0.1")
    ap.add_argument("--source", default="sft", help="base | sft | onpolicy | final | a weights path")
    ap.add_argument("--teacher", default=None, help="default: the config's teacher (Qwen3-0.6B)")
    ap.add_argument("--prompts", default=None, help="default: the config's heldout (data/prompts_heldout.jsonl)")
    ap.add_argument("--n", type=int, default=0, help="a fixed random subset of the prompts (0 = all)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--config", default=str(CONFIG_DIR / "onpolicy.yaml"))
    ap.add_argument("--ckpt-root", default="")
    ap.add_argument("--log-root", default="")
    args = ap.parse_args(argv)

    import mlx.core as mx
    from bonsai.common import CKPT_DIR
    from bonsai.distill import DEFAULTS, load_records, measure_kl
    from bonsai.load import load_model
    from bonsai.teacher import Teacher, resolve_name

    cfg = {**DEFAULTS, **(load_yaml(args.config) or {})}
    mx.set_memory_limit(int(float(cfg["memory_limit_gb"]) * 1024 ** 3))
    model, tok, meta = load_model(args.run, args.source, ckpt_dir=Path(args.ckpt_root) if args.ckpt_root else CKPT_DIR)
    records = load_records(args.prompts or cfg["heldout"], args.n, seed=0)
    teacher = Teacher(resolve_name(args.teacher or cfg["teacher"]))
    log_dir = (Path(args.log_root) if args.log_root else LOG_DIR) / args.run
    label = args.source if args.source in ("base", "latest", "sft", "onpolicy", "final") else Path(args.source).stem
    log(f"teacher KL: {label} ({meta['path']}) vs {teacher.name} on {len(records)} prompts")
    res = measure_kl(model, tok, teacher, records, cfg, seed=args.seed, n_samples=8)
    res.update(source=label, path=meta["path"], teacher=teacher.name, prompts=str(args.prompts or cfg["heldout"]),
               seed=args.seed)
    write_json(log_dir / f"teacher_kl_{label}.json", res)
    merged_path = log_dir / "teacher_kl.json"
    merged = read_json(merged_path) if merged_path.exists() else {}
    merged[label] = {k: res[k] for k in ("kl", "seq_kl", "tokens", "seqs", "by", "finished", "path", "teacher", "seed",
                                         "teacher_context")}
    if any(v.get("teacher_context") != res["teacher_context"] for v in merged.values()):
        log(f"WARNING: {merged_path} mixes teacher contexts (an entry without teacher_context was scored without "
            f"the system prompts): re-run those sources")
    write_json(merged_path, merged)
    by = res["by"]
    log(f"KL(student || teacher) = {res['kl']:.4f} nats/token over {res['tokens']:,} tokens of {res['seqs']} answers "
        f"(think {by.get('mode/think', {}).get('kl', float('nan')):.4f}, no_think "
        f"{by.get('mode/no_think', {}).get('kl', float('nan')):.4f}); finished {res['finished']:.2f}; "
        f"{res['t_total']:.0f}s -> {log_dir / f'teacher_kl_{label}.json'}")


if __name__ == "__main__":
    main()
