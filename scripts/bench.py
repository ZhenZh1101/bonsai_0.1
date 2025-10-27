"""Day-0 throughput benchmark and model-size decision (plan section 4).

    python -m scripts.bench             # full: 5 configs x micro-batch 8/16/32, 3 min each after 1 min warm-up (~1 h)
    python -m scripts.bench --quick     # < 1 min smoke (bonsai-0.1-mini, micro 4, 2^14-token steps) -> bench_quick.json
    python -m scripts.bench --summarize logs/bench.json      # re-apply the rule to saved results

Each run does real compiled optimizer steps (2^17 tokens/step, Muon + AdamW, clipping, gradient accumulation)
on S1 shards (tok/s1, else tok/smoke/s1) or, with --synthetic or when no shards exist, on random-token batches
(throughput does not depend on the text). Reported: mean tok/s of the steps after the warm-up, step time, peak
memory. The summary applies the plan's rules for the 3-day (50 h) and 5-day (96 h) budgets:

  * D = 0.85 x measured tok/s x hours x 3600 (0.85 = thermal + eval overhead) = the --tokens budget.
  * ratio rule: a config passes when D / non-embedding params >= 18, compared after rounding to an integer as
    the plan's table does (its 5-day default 14x640 at the estimated 4.8k tok/s gives 17.8, listed as 18); the
    widest passing config wins. It decides the 3-day budget (no config passes: the plan default, 12x576).
  * 5-day rule (plan section 4, explicit): bonsai-0.1 (14x640) unless it measures < 4.6k tok/s, then 12x640 (at
    4.6k, 14x640 gets 17.1 tokens/param: the plan accepts it). It decides the 5-day budget, unless 14x640
    measures > 20% above the plan's estimate and a wider config passes the ratio rule ("switch").
  * measured vs the plan's estimate (table, section 4) off by more than 20% -> switch (flagged per config).
  * data: the phase needs of the recommended run (bonsai.optim.RunPlan with the scripts.train defaults: S0 5%,
    S2 25% incl. half the 200-step transition) against tok/s{0,1,2}/manifest.json (else configs/s*.yaml
    total_tokens, not built yet); --tokens is capped so that no phase repeats data (scripts.train refuses
    that), with a note saying which build to raise.

logs/bench.json: {"results": [every run], "summary": {"per_config": ..., "50h": B, "96h": B}}, where
B["recommended"] = {"config", "micro_batch", "tokens" (float, = --tokens), "tokens_flag" (the same as a string,
e.g. "1.38e9"), "run", "command" (the full scripts.train command), "decided_by", "passes_ratio_rule",
"tokens_per_param", "tok_per_s", "tokens_measured" (the uncapped budget D), "stages" / "data" (phase needs vs
shards), "note" (only when the shards capped --tokens)}.
"""

import argparse
import gc
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

from bonsai.common import CONFIG_DIR, LOG_DIR, TOK_DIR, load_yaml, log, read_json, write_json
from bonsai.dataloader import PackedLoader
from bonsai.model import Bonsai, ModelConfig
from bonsai.optim import RunPlan, build_optimizer, round_steps, set_lr
from bonsai.train import build_step_fns, data_shortfalls

CONFIGS = ["bonsai-0.1-mini", "14x576", "12x640", "bonsai-0.1", "16x640"]
ESTIMATE = {"bonsai-0.1-mini": 6500, "14x576": 5900, "12x640": 5300, "bonsai-0.1": 4800, "16x640": 4200}  # tok/s
PLAN_DEFAULT = {50: "bonsai-0.1-mini", 96: "bonsai-0.1"}
RUN_NAME = {50: "bonsai-0.1-mini", 96: "bonsai-0.1"}
FALLBACK_5D = ("bonsai-0.1", 4600, "12x640")  # plan 4: 14x640 unless it measures < 4.6k tok/s, then 12x640
TOKENS_PER_STEP = 2 ** 17


def shard_tokens(tok_dir=TOK_DIR, config_dir=CONFIG_DIR):
    """{phase: (tokens, source)}: tok/<phase>/manifest.json, else the configs/<phase>.yaml build target."""
    out = {}
    for k in ("s0", "s1", "s2"):
        m, c = Path(tok_dir) / k / "manifest.json", Path(config_dir) / f"{k}.yaml"
        if m.exists():
            out[k] = (int(read_json(m)["total_tokens"]), str(m.parent))
        elif c.exists():
            out[k] = (int(float(load_yaml(c)["total_tokens"])), f"{c} (not built)")
    return out


class Synthetic:
    """Random-token rows (BOS first); throughput does not depend on the text."""

    def __init__(self, vocab, bos, seq, seed=0):
        self.rng, self.vocab, self.bos, self.seq = np.random.default_rng(seed), vocab, bos, seq

    def next_xy(self, rows):
        ids = self.rng.integers(0, self.vocab, (rows, self.seq + 1), dtype=np.int32)
        ids[:, 0] = self.bos
        return ids[:, :-1], ids[:, 1:]


def data_source(args, tok, seq):
    if not args.synthetic:
        for d in ([Path(args.data)] if args.data else [TOK_DIR / "s1", TOK_DIR / "smoke" / "s1"]):
            if (d / "manifest.json").exists():
                return PackedLoader(d, seq, seed=0, vocab_sha=tok.vmap.sha()), str(d)
        if args.data:
            raise SystemExit(f"{args.data}: no manifest.json")
    return Synthetic(tok.vocab_size, tok.bos_id, seq), "synthetic"


def bench_one(preset, micro, args, tok, seq=1024):
    cfg = ModelConfig.from_preset(preset, vocab_size=tok.vocab_size, seq_len=seq)
    model = Bonsai(cfg)
    mx.eval(model.parameters())
    opt, groups = build_optimizer(model)
    set_lr(opt, groups, 0.1)  # small: random data must not diverge; the LR does not change the speed
    step_micro, apply, _ = build_step_fns(model, opt, 1.0, mx.zeros((tok.vocab_size,)), compile=True)
    src, src_name = data_source(args, tok, seq)
    accum = max(1, args.batch_tokens // (micro * seq))
    mx.reset_peak_memory()
    t_start = time.time()
    steps, losses, timed = 0, [], []
    one = mx.array(1.0)
    while time.time() - t_start < args.minutes * 60 or len(timed) < 1:
        t0 = time.time()
        acc, ls, ns = None, 0.0, 0.0
        for _ in range(accum):
            x, y = src.next_xy(micro)
            loss, n, g = step_micro(mx.array(x), mx.array(y), one)
            acc = g if acc is None else tree_map(mx.add, acc, g)
            mx.eval(acc, loss, n)
            ls += loss.item(); ns += n.item()
        norm = apply(tree_map(lambda v: v / ns, acc))
        mx.eval(norm, model.parameters(), opt.state)
        dt = time.time() - t0
        steps += 1
        losses.append(ls / ns)
        if steps > args.warm_steps and time.time() - t_start >= args.warm * 60:
            timed.append(dt)
    n = model.num_params()
    tps = accum * micro * seq * len(timed) / sum(timed)
    res = {"config": preset, "micro_batch": micro, "accum": accum, "batch_tokens": accum * micro * seq,
           "steps": steps, "timed_steps": len(timed), "tok_per_s": tps, "step_time_s": sum(timed) / len(timed),
           "peak_mem_gb": mx.get_peak_memory() / 1024 ** 3, "first_loss": losses[0], "last_loss": losses[-1],
           "non_embedding_params": n["non_embedding"], "total_params": n["total"], "data": src_name,
           "estimate_tok_per_s": ESTIMATE.get(preset)}
    if res["estimate_tok_per_s"]:
        res["vs_estimate"] = tps / res["estimate_tok_per_s"] - 1
    del model, opt, step_micro, apply
    gc.collect()
    mx.clear_cache()
    return res


def recommendation(r, h, ratio, passes, decided_by, shards):
    """The recommended run for config result `r` at `h` hours: --tokens (the budget, capped so that no phase
    repeats data: scripts.train would refuse it), the command, and the phase needs vs the shards."""
    D = 0.85 * r["tok_per_s"] * h * 3600
    budget = tokens = math.floor(D / 1e7) / 100  # in 1e9, rounded down to 10M
    built = {k: v[0] for k, v in shards.items()}

    def plan(t):
        return RunPlan.build(round_steps(t * 1e9, TOKENS_PER_STEP))

    short = first_short = data_shortfalls(plan(tokens), TOKENS_PER_STEP, built)
    while short and tokens > 0.01:
        tokens = round(tokens - 0.01, 2)
        short = data_shortfalls(plan(tokens), TOKENS_PER_STEP, built)
    need = plan(tokens).phase_tokens(TOKENS_PER_STEP)
    run = RUN_NAME.get(int(h), "bonsai-0.1")
    rec = {"config": r["config"], "micro_batch": r["micro_batch"], "tokens": tokens * 1e9,
           "tokens_flag": f"{tokens:.2f}e9", "run": run,
           "command": f"python -m scripts.train --model {r['config']} --micro-batch {r['micro_batch']} "
                      f"--tokens {tokens:.2f}e9 --run {run}",
           "decided_by": decided_by, "passes_ratio_rule": passes, "tokens_per_param": ratio,
           "tok_per_s": r["tok_per_s"], "tokens_measured": D, "stages": need,
           "data": {k: {"need": need[k], "built": shards[k][0], "source": shards[k][1]} for k in need if k in shards}}
    if first_short:
        rec["note"] = (f"--tokens capped at {tokens:.2f}e9 (budget {budget:.2f}e9) by the shards: " + "; ".join(
            f"{k.upper()} would need {n / 1e6:.1f}M tokens > the {b / 1e6:.1f}M in {shards[k][1]}"
            for k, n, b in first_short) + ". Raise total_tokens in configs/<phase>.yaml and rebuild to use the whole "
                                          "budget.")
    return rec


def summarize(results, hours=(50, 96), min_ratio=18, shards=None):
    shards = shard_tokens() if shards is None else shards
    best = {}
    for r in results:
        if r.get("tok_per_s") and (r["config"] not in best or r["tok_per_s"] > best[r["config"]]["tok_per_s"]):
            best[r["config"]] = r
    out = {"rule": f"D = 0.85 x tok/s x hours x 3600; ratio rule: pass if round(D / non-embedding) >= {min_ratio}, "
                   f"widest passing config (3-day); 5-day: bonsai-0.1 unless < 4.6k tok/s, then 12x640; "
                   f"|measured / plan estimate - 1| > 20% -> switch", "per_config": {}}
    for cfg, r in best.items():
        out["per_config"][cfg] = {"micro_batch": r["micro_batch"], "tok_per_s": r["tok_per_s"],
                                  "peak_mem_gb": r["peak_mem_gb"], "estimate_tok_per_s": ESTIMATE.get(cfg),
                                  "vs_estimate": r.get("vs_estimate"),
                                  "switch": r.get("vs_estimate") is not None and abs(r["vs_estimate"]) > 0.2}
    for h in hours:
        table, widest = [], None
        for cfg, r in best.items():
            D = 0.85 * r["tok_per_s"] * h * 3600
            ratio = D / r["non_embedding_params"]
            ok = math.floor(ratio + 0.5) >= min_ratio
            table.append({"config": cfg, "micro_batch": r["micro_batch"], "tok_per_s": r["tok_per_s"], "tokens": D,
                          "tokens_per_param": ratio, "passes": ok})
            if ok and (widest is None or r["non_embedding_params"] > best[widest]["non_embedding_params"]):
                widest = cfg
        entry = {"table": table, "ratio_rule": widest, "recommended": None}
        default, choice = PLAN_DEFAULT.get(int(h)), None
        if int(h) == 96 and FALLBACK_5D[0] in best:
            cfg, floor_tps, fallback = FALLBACK_5D
            tps, vs = best[cfg]["tok_per_s"], best[cfg].get("vs_estimate") or 0
            choice = cfg if tps >= floor_tps else fallback
            why = (f"plan 5-day rule: {cfg} measured {tps:,.0f} tok/s {'>=' if tps >= floor_tps else '<'} "
                   f"{floor_tps:,} -> {choice}")
            if (vs > 0.2 and widest is not None
                    and best[widest]["non_embedding_params"] > best[cfg]["non_embedding_params"]):
                choice, why = widest, (f"switch: {cfg} measured {vs:+.0%} vs the plan estimate (> 20%) and the wider "
                                       f"{widest} passes the ratio rule")
            if choice not in best:
                choice = None  # the fallback was not measured: the ratio rule below
        if choice is None and widest is not None:
            choice, why = widest, f"ratio rule: widest config with round(tokens / non-embedding params) >= {min_ratio:g}"
        if choice is None and table:
            choice = default if default in best else max(table, key=lambda t: t["tokens_per_param"])["config"]
            why = f"no config passes the ratio rule: {'the plan default' if choice == default else 'the closest'} {choice}"
        if choice is not None:
            row = next(t for t in table if t["config"] == choice)
            entry["recommended"] = recommendation(best[choice], h, row["tokens_per_param"], row["passes"], why, shards)
        out[f"{h:g}h"] = entry
    return out


def report(results, summary):
    lines = [f"{r['config']:16s} mb{r['micro_batch']:<3d} {r['tok_per_s']:8,.0f} tok/s  {r['step_time_s']:6.2f} s/step  "
             f"peak {r['peak_mem_gb']:5.1f} GB  (plan {r.get('estimate_tok_per_s') or 0:,} tok/s, "
             f"{100 * (r.get('vs_estimate') or 0):+.0f}%)" if r.get("tok_per_s") else
             f"{r['config']:16s} mb{r['micro_batch']:<3d} FAILED: {r.get('error')}" for r in results]
    for key in (k for k in summary if k.endswith("h")):
        rec = summary[key]["recommended"]
        lines.append(f"{key}: " + (f"{rec['config']} mb{rec['micro_batch']}, {rec['tokens'] / 1e9:.2f}B tokens "
                                   f"({rec['tokens_per_param']:.1f} tokens/param; {rec['decided_by']}): "
                                   f"{rec['command']}" + (f"\n    NOTE: {rec['note']}" if rec.get("note") else "")
                                   if rec else "no results"))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--micro", default="8,16,32")
    ap.add_argument("--minutes", type=float, default=3, help="per run, incl. the warm-up")
    ap.add_argument("--warm", type=float, default=1, help="ignore steps in the first N minutes (thermal steady state)")
    ap.add_argument("--warm-steps", type=int, default=2, help="... and at least this many steps (compilation)")
    ap.add_argument("--batch-tokens", type=int, default=2 ** 17)
    ap.add_argument("--hours", default="50,96", help="pretraining GPU hours of the 3-day / 5-day plans")
    ap.add_argument("--min-ratio", type=float, default=18)
    ap.add_argument("--synthetic", action="store_true", help="random tokens even if shards exist")
    ap.add_argument("--data", default="", help="shard dir (default tok/s1, then tok/smoke/s1)")
    ap.add_argument("--out", default="", help="file under logs/ (default bench.json; --quick: bench_quick.json)")
    ap.add_argument("--summarize", default="", help="only re-apply the rule to a saved bench json")
    ap.add_argument("--quick", action="store_true", help="sub-minute smoke run")
    ap.add_argument("--memory-limit-gb", type=float, default=40.0)
    args = ap.parse_args()
    hours = [float(h) for h in args.hours.split(",")]
    if args.summarize:
        results = read_json(args.summarize)["results"]
        summary = summarize(results, hours, args.min_ratio)
        write_json(LOG_DIR / (args.out or "bench.json"), {"results": results, "summary": summary})
        log("\n" + report(results, summary))
        return
    if args.quick:
        args.configs, args.micro, args.batch_tokens = "bonsai-0.1-mini", "4", 2 ** 14
        args.minutes, args.warm, args.warm_steps = 0.4, 0, 2
        args.memory_limit_gb = min(args.memory_limit_gb, 8.0)

    mx.set_memory_limit(int(args.memory_limit_gb * 1024 ** 3))
    from bonsai.tokenizer import get_tokenizer
    tok = get_tokenizer()
    results = []
    for preset in args.configs.split(","):
        for micro in [int(m) for m in args.micro.split(",")]:
            log(f"bench {preset} micro {micro} for {args.minutes} min ...")
            try:
                r = bench_one(preset, micro, args, tok)
            except RuntimeError as e:  # e.g. out of memory at micro 32
                r = {"config": preset, "micro_batch": micro, "error": str(e)[:300]}
                gc.collect()
                mx.clear_cache()
            log(json.dumps(r))
            results.append(r)
    summary = summarize(results, hours, args.min_ratio)
    out = LOG_DIR / (args.out or ("bench_quick.json" if args.quick else "bench.json"))
    write_json(out, {"results": results, "summary": summary, "quick": args.quick, "time": time.time()})
    log("\n" + report(results, summary) + f"\n-> {out}")


if __name__ == "__main__":
    main()
