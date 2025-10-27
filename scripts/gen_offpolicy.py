"""Off-policy distillation set: the Qwen3 teacher answers every prompt in both modes (plan sections 11, 14).

    python -m scripts.gen_offpolicy --teacher Qwen/Qwen3-1.7B --prompts data/prompts.jsonl --modes think,no_think --budget 128
    python -m scripts.gen_offpolicy ... --shard 0/2      # this machine; the M2 runs --shard 1/2 --teacher Qwen/Qwen3-0.6B
    python -m scripts.gen_offpolicy ... --limit 48 --out /tmp/bench   # throughput check

Resumable: samples already in <out>/kept_*.jsonl or rejected_*.jsonl (id "<prompt id>:<mode>") are skipped, so
rerunning the same command continues, also after a hard kill (OOM, power loss): a partly written last line is
cut off (logged) and that sample generated again. Settings (system prompts, sampling, budgets, batch) come from
configs/offpolicy.yaml; see bonsai/teacher.py for the generation protocol and the filters. Output:
<out>/kept_s<i>of<n>.jsonl (Qwen-format messages without the teacher's system prompt), rejected_s<i>of<n>.jsonl
(reasons + the full sample), runs.jsonl (throughput of each run) and stats.json (totals, filter reasons, keep
rates). `--refilter` re-applies the filters (e.g. after changing `filters.think_answer`) without generating.
"""

import argparse
import os

from bonsai.common import CONFIG_DIR, DATA_ROOT, LOG_DIR, iter_jsonl, load_yaml, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_DIR / "offpolicy.yaml"))
    ap.add_argument("--teacher", default=None, help="Qwen/Qwen3-1.7B (default: config) or Qwen/Qwen3-0.6B")
    ap.add_argument("--prompts", default=None, help="default: the config's prompts.out (data/prompts.jsonl)")
    ap.add_argument("--modes", default=None, help="comma list, default think,no_think")
    ap.add_argument("--budget", type=int, default=None, help="thinking tokens per think block (128)")
    ap.add_argument("--shard", default="0/1", help="i/n: this process takes the prompts whose id hashes to i")
    ap.add_argument("--limit", type=int, default=0, help="at most this many prompts (x modes jobs)")
    ap.add_argument("--batch", type=int, default=None, help="rows decoded together (config: 64)")
    ap.add_argument("--chunk", type=int, default=None, help="jobs per continuous batch (config: 8192)")
    ap.add_argument("--out", default=None, help="output dir (config: data/offpolicy)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--memory-limit-gb", type=float, default=36.0)
    ap.add_argument("--retry", default="", help="comma list of rejection reasons to generate again (non_english)")
    ap.add_argument("--refilter", action="store_true",
                    help="no generation: re-apply the filters of --config to every stored sample of --out")
    args = ap.parse_args()

    import mlx.core as mx
    from bonsai.teacher import Teacher, generate_offpolicy, refilter, summarize

    cfg = load_yaml(args.config)
    g = cfg.get("generation", {})
    if args.refilter:
        st = refilter(DATA_ROOT / (args.out or g.get("out_dir", "data/offpolicy")), cfg,
                      log_file=LOG_DIR / "gen_offpolicy.log")
        log(f"kept {st['kept']:,} / {st['jobs']:,}; first rejection reasons: {st['first_reason']}")
        return
    i, n = (int(x) for x in args.shard.split("/"))
    if not 0 <= i < n:
        raise SystemExit(f"--shard {args.shard}: need 0 <= i < n")
    if os.environ.get("BONSAI_MLX_DEVICE") == "cpu":   # e.g. a check while the GPU is busy training
        mx.set_default_device(mx.cpu)
    mx.set_memory_limit(int(args.memory_limit_gb * 1024 ** 3))
    mx.random.seed(args.seed * 1000 + i)
    out = DATA_ROOT / (args.out or g.get("out_dir", "data/offpolicy"))
    prompts = list(iter_jsonl(DATA_ROOT / (args.prompts or cfg["prompts"]["out"])))
    modes = tuple((args.modes or ",".join(g.get("modes", ["think", "no_think"]))).split(","))
    teacher = Teacher(args.teacher or g.get("teacher", "Qwen3-1.7B"))
    log_file = LOG_DIR / "gen_offpolicy.log"
    log(f"teacher {teacher.name}, {len(prompts):,} prompts, modes {modes}, shard {i}/{n} -> {out}", log_file)
    run = generate_offpolicy(teacher, prompts, out, modes, cfg, (i, n), args.limit, args.batch or g.get("batch", 64),
                             args.chunk or g.get("chunk", 8192), args.budget or g.get("budget", 128),
                             log_file=log_file, retry=tuple(r for r in args.retry.split(",") if r))
    st = summarize(out)
    log(f"run: {run['jobs']:,} jobs, kept {run['kept']:,}, {run['gen_tokens']:,} generated tokens at "
        f"{run.get('tok_per_s', 0):.0f} tok/s; total kept {st['kept']:,} / {st['jobs']:,} "
        f"(first rejection reasons: {st['first_reason']})", log_file)


if __name__ == "__main__":
    main()
