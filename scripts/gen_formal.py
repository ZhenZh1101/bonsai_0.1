"""Generate the formal-language and synthetic-reasoning pools (plan sections 9, 10; DESIGN.md 1.1, producer "formal").

    python -m scripts.gen_formal [--smoke] [--only POOL ...] [--workers 8]

Writes raw/filtered/<pool>.jsonl for
    s0_shuffle_dyck, s0_dyck, s0_mp_struct   {"text"}                       (S0, full loss)
    s0_derivations                           {"text", "target_start", "source"}  (S0, loss after "next:")
    synth_reasoning                          {"text", "kind"}               (S1: arith, igsm, fld_text, logic_text)
    fld_chat                                 {"user", "reasoning", "answer", "label", "depth", "config"}  (S2)
    rc_short                                 {"user", "reasoning", "answer", "task", "source", "cut"}  (S2)
and raw/filtered/stats_formal.json; --smoke writes small versions of the same files to raw/filtered/smoke/.

Sizes are student tokens incl. BOS, counted with the real tokenizer (chat pools: the rendered conversation in
thinking mode). Targets are minimums: the brief's sizes, or configs/s{0,1,2}.yaml total_tokens x weight x 1.3 /
max_epochs when those configs ask for more. Source-bound parts (ProcPile, FOL-Traces, symbolic-reasoning-env,
FLD.v2) are used in full; the synthetic generators (Dyck, MP-STRUCT, logic, arith, iGSM) top each pool up to its
target. Every record is 13-gram decontaminated (bonsai.data.decontam; the index is built in the parent and
inherited by the forked workers); S0 pools must be ASCII. Records are shuffled with a fixed seed after being
collected in task order, so a run is deterministic and a prefix of a pool is a uniform sample.
"""

import argparse
import math
import multiprocessing as mp
import random
import time
from collections import Counter, defaultdict

from bonsai import chat
from bonsai.common import CONFIG_DIR, FILTERED_DIR, load_yaml, log, read_json, write_json, write_jsonl
from bonsai.data import decontam
from bonsai.formal import arith, derivations, dyck, fld, igsm, logic, mp_struct

POOLS = ["s0_shuffle_dyck", "s0_dyck", "s0_mp_struct", "s0_derivations", "synth_reasoning", "fld_chat", "rc_short"]
TARGETS = {"s0_shuffle_dyck": 31e6, "s0_dyck": 11e6, "s0_mp_struct": 21e6, "s0_derivations": 42e6,
           "synth_reasoning": 40e6, "fld_chat": 25e6, "rc_short": 22e6}
S0 = {"s0_shuffle_dyck", "s0_dyck", "s0_mp_struct", "s0_derivations"}
CHAT = {"fld_chat", "rc_short"}
SOURCE_KEY = {"s0_derivations": "source", "synth_reasoning": "kind", "fld_chat": "config", "rc_short": "source"}
SYNTH_MIX = {"arith": 0.4, "igsm": 0.4, "text": 0.2}     # synth_reasoning: fld_text + logic_text share "text"
FOL_TRACES_KEEP = 25                                     # percent of FOL-Traces rows (3 single-step patterns)
SMOKE_SCALE = 0.005
SMOKE_ROWS = {"procpile": 1500, "fol_traces": 600, "symbolic_env": 1500, "fld": 150}   # source rows per file
_TOK = None


def config_targets():
    """Pool -> tokens asked by the stage configs (x 1.3 headroom), when the configs exist."""
    out = {}
    for stage in ("s0", "s1", "s2"):
        path = CONFIG_DIR / f"{stage}.yaml"
        if not path.exists():
            continue
        cfg = load_yaml(path)
        for spec in (cfg.get("sources") or {}).values():
            pool = spec.get("pool")
            if pool in TARGETS:
                need = float(cfg["total_tokens"]) * float(spec["weight"]) * 1.3 / float(spec.get("max_epochs") or 1)
                out[pool] = max(out.get(pool, 0), need)
    return out


# ------------------------------------------------------------------ workers


def _tok():
    global _TOK
    if _TOK is None:
        from bonsai.tokenizer import get_tokenizer
        _TOK = get_tokenizer()
    return _TOK


def _generate(kind, key, args):
    tok = _tok()
    qwen = tok.qwen
    if kind in ("shuffle", "dyck"):
        pool = "s0_shuffle_dyck" if kind == "shuffle" else "s0_dyck"
        return {pool: [dyck.doc(kind, key, i) for i in range(args)]}, {}
    if kind == "mp_struct":
        return {"s0_mp_struct": [mp_struct.doc(key, i) for i in range(args)]}, {}
    if kind == "procpile":
        path, start, stop = args
        docs, rc, cnt = derivations.procpile_chunk(path, start, stop, tok, qwen)
        return {"s0_derivations": docs, "rc_short": rc}, cnt
    if kind == "fol_traces":
        path, start, stop = args
        docs, cnt = derivations.fol_traces_chunk(path, start, stop, tok, keep=FOL_TRACES_KEEP)
        return {"s0_derivations": docs}, {f"fol_traces_{k}": v for k, v in cnt.items()}
    if kind == "logic_fol":
        return {"s0_derivations": derivations.logic_chunk(key, args, tok)}, {}
    if kind == "symbolic_env":
        path, start, stop = args
        return {"rc_short": derivations.symbolic_env_chunk(path, start, stop, qwen)}, {}
    if kind == "fld":
        split, cfg, start, stop = args
        chats, texts = [], []
        for rec in fld.load_fld(split, (cfg,))[start:stop]:
            r = fld.chat_record(rec, qwen) if split == "train" else None
            if r is None:
                texts.append({"text": fld.fld_text(rec), "kind": "fld_text"})
            else:
                chats.append(r)
        return {"fld_chat": chats, "synth_reasoning": texts}, {f"fld_{split}_rows": stop - start}
    if kind == "logic_chat":
        recs = [logic.chat_record(key, i, qwen=qwen) for i in range(args)]
        return {"fld_chat": [r for r in recs if r]}, {"logic_chat_over_budget": sum(r is None for r in recs)}
    if kind in ("arith", "igsm"):
        mod = arith if kind == "arith" else igsm
        return {"synth_reasoning": [mod.doc(key, i) for i in range(args)]}, {}
    if kind == "logic_text":
        return {"synth_reasoning": [logic.text_record(key, i) for i in range(args)]}, {}
    raise ValueError(kind)


def _check_text(pool, r):
    return r["text"] if pool not in CHAT else f"{r['user']}\n{r['reasoning']}\n{r['answer']}"


def _count(pool, recs, tok):
    """Student tokens of each record incl. BOS."""
    if pool not in CHAT:
        return [len(ids) + 1 for ids in tok.encode_batch([r["text"] for r in recs])]
    convs = [[{"role": "user", "content": r["user"]},
              {"role": "assistant", "content": r["answer"], "reasoning_content": r["reasoning"]}] for r in recs]
    return [len(tok.from_teacher(ids)) + 1 for ids, _ in chat.render_batch(tok.qwen, convs)]


def work(task):
    """task = (index, kind, key, args) -> (index, {pool: (records, tokens per record)}, counts)."""
    idx, kind, key, args = task
    tok = _tok()
    out, cnt = {}, Counter()
    gen, extra = _generate(kind, key, args)
    cnt.update(extra)
    for pool, recs in gen.items():
        kept = []
        for r in recs:
            if pool in S0 and not r["text"].isascii():
                cnt[f"{pool}_non_ascii"] += 1
            elif decontam.is_contaminated(_check_text(pool, r)):
                cnt[f"{pool}_contaminated"] += 1
            else:
                kept.append(r)
        if kept:
            out[pool] = (kept, _count(pool, kept, tok))
    return idx, out, cnt


# ------------------------------------------------------------------ orchestration


class Collector:
    def __init__(self, pools):
        self.pools = set(pools)
        self.recs = defaultdict(dict)        # pool -> task index -> records
        self.tokens = Counter()
        self.by_source = defaultdict(Counter)
        self.by_source_recs = defaultdict(Counter)
        self.cnt = Counter()
        self.n_tasks = 0

    def run(self, pool, tasks):
        tasks = [(self.n_tasks + i, *t) for i, t in enumerate(tasks)]
        self.n_tasks += len(tasks)
        for idx, out, cnt in pool.imap_unordered(work, tasks):
            self.cnt.update(cnt)
            for name, (recs, counts) in out.items():
                if name not in self.pools:
                    continue
                self.recs[name][idx] = recs
                self.tokens[name] += sum(counts)
                key = SOURCE_KEY.get(name)
                for r, n in zip(recs, counts):
                    src = str(r.get(key)) if key else name
                    if name == "synth_reasoning" and src in ("fld_text", "logic_text"):
                        src = "text:" + src
                    self.by_source[name][src] += n
                    self.by_source_recs[name][src] += 1

    def synth_tokens(self, part):
        return sum(v for k, v in self.by_source["synth_reasoning"].items() if k == part or k.startswith(part + ":"))

    def records(self, name):
        return [r for idx in sorted(self.recs[name]) for r in self.recs[name][idx]]


def fill(col, pool, workers, name, kind, target_fn, per_chunk, est_tokens):
    """Dispatch synthetic chunks of `kind` until target_fn() <= 0 (waves of 2 x workers chunks)."""
    chunk = 0
    while target_fn() > 0:
        n = max(1, min(2 * workers, math.ceil(target_fn() / (per_chunk * est_tokens))))
        col.run(pool, [(kind, f"{name}-{chunk + i}", per_chunk) for i in range(n)])
        chunk += n
        log(f"  {name}: {col.tokens[name] / 1e6:.1f}M tokens after {chunk} {kind} chunks")


def source_tasks(want, smoke):
    """Source-bound tasks (kind, key, args) for the wanted pools."""
    rows = (lambda kind: SMOKE_ROWS[kind]) if smoke else (lambda kind: None)
    tasks = []
    if want & {"s0_derivations", "rc_short"}:
        for path in derivations.PROCPILE[: 1 if smoke else None]:
            n = rows("procpile") or 250_000
            tasks += [("procpile", path.name, (path, s, min(s + 25_000, n))) for s in range(0, n, 25_000)]
    if "s0_derivations" in want:
        for path in derivations.FOL_TRACES:
            n = rows("fol_traces") or sum(1 for _ in open(path))
            tasks += [("fol_traces", path.name, (path, s, min(s + 25_000, n))) for s in range(0, n, 25_000)]
    if "rc_short" in want:
        import pyarrow.parquet as pq
        for path in derivations.SYMBOLIC_ENV:
            n = rows("symbolic_env") or pq.ParquetFile(path).metadata.num_rows
            tasks += [("symbolic_env", path.name, (path, s, min(s + 20_000, n))) for s in range(0, n, 20_000)]
    if want & {"fld_chat", "synth_reasoning"}:
        for split, size in (("train", 30_000), ("validation", 5_000)):
            if split == "validation" and "synth_reasoning" not in want:
                continue
            for cfg in fld.CONFIGS:
                n = rows("fld") or size
                tasks += [("fld", f"{split}-{cfg}", (split, cfg, s, min(s + 5_000, n))) for s in range(0, n, 5_000)]
    return tasks


def run(pools, smoke=False, workers=8):
    out_dir = FILTERED_DIR / "smoke" if smoke else FILTERED_DIR
    cfg = config_targets()
    targets = {p: max(TARGETS[p], cfg.get(p, 0)) * (SMOKE_SCALE if smoke else 1) for p in pools}
    t0 = time.time()
    log(f"gen_formal {'smoke ' if smoke else ''}-> {out_dir}: {', '.join(pools)}; targets "
        + ", ".join(f"{p} {targets[p] / 1e6:.2f}M" for p in pools))
    decontam.build_index()       # before the fork: workers share it
    col = Collector(pools)
    want = set(pools)
    small = 0.02 if smoke else 1.0   # smoke chunks are small too
    with mp.get_context("fork").Pool(workers) as pool:
        tasks = source_tasks(want, smoke)
        if tasks:
            log(f"source-bound tasks: {len(tasks)}")
            col.run(pool, tasks)
            log(f"  done in {time.time() - t0:.0f}s: " + ", ".join(f"{p} {col.tokens[p] / 1e6:.1f}M" for p in pools))
        per = lambda n: max(5, int(n * small))
        for name, kind, n, est in (("s0_shuffle_dyck", "shuffle", 2000, 513), ("s0_dyck", "dyck", 2000, 513),
                                   ("s0_mp_struct", "mp_struct", 2000, 513), ("s0_derivations", "logic_fol", 2000, 600),
                                   ("fld_chat", "logic_chat", 4000, 260)):
            if name in want:
                fill(col, pool, workers, name, kind, lambda: targets[name] - col.tokens[name], per(n), est)
        if "synth_reasoning" in want:
            t = targets["synth_reasoning"]
            for part, kind, est in (("arith", "arith", 600), ("igsm", "igsm", 570), ("text", "logic_text", 450)):
                fill(col, pool, workers, "synth_reasoning", kind,
                     lambda: SYNTH_MIX[part] * t - col.synth_tokens(part), per(2000), est)
    stats_path = out_dir / "stats_formal.json"
    stats = read_json(stats_path) if stats_path.exists() else {}
    for name in pools:
        recs = col.records(name)
        random.Random(f"shuffle-{name}").shuffle(recs)
        write_jsonl(out_dir / f"{name}.jsonl", recs)
        stats[name] = {
            "records": len(recs), "tokens": int(col.tokens[name]), "target": int(targets[name]),
            "met": col.tokens[name] >= targets[name],
            "shortfall": int(max(0, targets[name] - col.tokens[name])),
            "sources": {k: {"records": col.by_source_recs[name][k], "tokens": v}
                        for k, v in sorted(col.by_source[name].items())},
            "counts": {k: v for k, v in sorted(col.cnt.items()) if k.startswith(name)},
        }
        log(f"{name}: {len(recs)} records, {col.tokens[name] / 1e6:.2f}M tokens (target {targets[name] / 1e6:.2f}M)")
    tok = _tok()
    stats["_meta"] = {"smoke": smoke, "vocab_size": tok.vocab_size, "vocab_sha": tok.vmap.sha(),
                      "counts": dict(sorted(col.cnt.items())), "seconds": round(time.time() - t0),
                      "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(stats_path, stats)
    log(f"gen_formal done in {time.time() - t0:.0f}s -> {stats_path}")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true", help="small pools into raw/filtered/smoke/")
    ap.add_argument("--only", nargs="+", choices=POOLS, help="regenerate only these pools")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run(args.only or POOLS, smoke=args.smoke, workers=args.workers)


if __name__ == "__main__":
    main()
