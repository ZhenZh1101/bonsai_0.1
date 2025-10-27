"""Mix by token targets, render + tokenize, write student-id shards (plan sections 9, 10; DESIGN 1.2, 1.3; ported
from pagoda).

Stage builds (configs/s0.yaml, s1.yaml, s2.yaml):
  * for each source, read its filtered pool raw/filtered/<pool>.jsonl from the pool cursor on, render and
    tokenize records (bonsai.data.render, in a spawn process pool, results consumed in order so the build is
    identical for any number of workers) until the source's token target (total * weight) is met; if the pool
    runs dry, repeat what was taken up to `max_epochs`
  * `val_fraction` first sets aside that share of every source -> `val_dir` (S1: tok/val, fixed for the
    project; S0: tok/val_s0); a stage without it drops training docs identical to a doc of `val_dir`
  * pools shared between stages (TinyGSM, SmolTalk, Stack-Edu, ...) are consumed through a persistent cursor
    (tok/pool_cursors.json; smoke builds: tok/smoke/pool_cursors.json), so S1, S2 and the prompt pool never see
    the same record; S0 (`use_cursors: false`) always starts its own pools at line 0
  * docs longer than seq_len + 1 are split into BOS-prefixed chunks (conversations longer than a row are dropped
    instead: a chunk without its start would break the chat format); all docs of a stage are shuffled together
    (seeded) and written as shard_XXXXX.bin (student ids, dtype = tok.vmap.shard_dtype, <= 2^24 tokens, cut at
    doc boundaries) + .idx (uint64 offsets + sentinel) + .mask (uint8, `mask: true` stages only: S0, and S2 for
    its tool_chat docs; a doc whose renderer gives no mask gets all ones)
  * manifest.json records bos_id, vocab_size, vocab_sha, dtype, seq_len, has_mask, per-source counts (target,
    deviation, cursors, out-of-subset teacher tokens) and the config
"""

import json
import multiprocessing as mp
import os
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bonsai.common import FILTERED_DIR, TOK_DIR, log, write_json
from bonsai.data.render import CHAT_FORMATS, Renderer, split_long

SHARD_TOKENS = 2 ** 24
CURSOR_FILE = "pool_cursors.json"
BATCH_LINES = 1000


# ------------------------------------------------------------------ cursors
def cursor_path(cfg):
    return TOK_DIR / cfg.get("cursor_file", CURSOR_FILE)


def load_cursors(cfg):
    p = cursor_path(cfg)
    return json.loads(p.read_text()) if p.exists() else {}


def save_cursors(cfg, c):
    write_json(cursor_path(cfg), c)


def restore_cursors_after(cfg, shard_dir):
    """Reset the pool cursors to where the build of `shard_dir` left them (its manifest's cursor_end).

    The cursor file holds the state after the last stage built, so rebuilding S2 alone would otherwise start
    after the previous S2 records instead of after S1. Pools the earlier stage never read start from 0."""
    man = json.loads((TOK_DIR / shard_dir / "manifest.json").read_text())
    cursors = {}
    for e in man["sources"].values():
        cursors[e["pool"]] = max(cursors.get(e["pool"], 0), int(e["cursor_end"]))
    save_cursors(cfg, cursors)
    return cursors


# ------------------------------------------------------------------ reading + rendering
def pool_path(pool):
    return FILTERED_DIR / f"{pool}.jsonl"


class PoolReader:
    """Raw lines of raw/filtered/<pool>.jsonl from a line cursor."""

    def __init__(self, pool, start=0):
        self.path = pool_path(pool)
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} missing: run scripts.prepare_data (or the pool's producer) first")
        self.f = open(self.path)
        self.eof = False
        for _ in range(start):
            if not self.f.readline():
                break

    def read(self, n):
        """Up to n lines ([] at the end of the file)."""
        lines = []
        while len(lines) < n and not self.eof:
            line = self.f.readline()
            if not line:
                self.eof = True
                self.f.close()
                break
            lines.append(line)
        return lines


_WORKER = {}


def _init_worker(directory, s2t, meta):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"   # one tokenizer thread per worker process
    from bonsai.tokenizer import QwenTokenizer, Tokenizer, VocabMap
    qwen = QwenTokenizer(directory)
    _WORKER["tok"] = Tokenizer(qwen, VocabMap(s2t, qwen, meta))
    _WORKER["renderers"] = {}


def _render_lines(fmt, spec, lines, record_filter=None):
    key = (fmt, json.dumps(spec, sort_keys=True, default=str))
    r = _WORKER["renderers"].get(key)
    if r is None:
        r = _WORKER["renderers"][key] = Renderer(_WORKER["tok"], fmt, spec)
    return _render_records(r, [json.loads(l) for l in lines], record_filter)


def _render_records(renderer, recs, record_filter=None):
    out = renderer.batch(recs, stats=True)
    if record_filter is not None:
        out = [o if record_filter(rec) else False for rec, o in zip(recs, out)]
    return out


class RenderPool:
    """Spawn process pool whose workers hold the same Tokenizer (same subset S) as the parent."""

    def __init__(self, tok, workers=None):
        self.n = workers or max(1, (os.cpu_count() or 4) - 2)
        self.ex = ProcessPoolExecutor(self.n, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                                      initargs=(str(tok.qwen.dir), tok.vmap.s2t, tok.vmap.meta))

    def submit(self, renderer, lines, record_filter=None):
        return self.ex.submit(_render_lines, renderer.fmt, renderer.spec, lines, record_filter)

    def close(self):
        self.ex.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class DocSet:
    """Token docs of one source: a list of arrays + lengths (+ optional uint8 masks)."""

    def __init__(self, with_mask=False):
        self.chunks, self.mask_chunks, self.lengths = [], [], []
        self.with_mask = with_mask
        self.tokens = 0

    def add(self, ids, mask=None):
        self.chunks.append(ids)
        if self.with_mask:
            self.mask_chunks.append(np.ones(len(ids), dtype=np.uint8) if mask is None else mask)
        self.lengths.append(len(ids))
        self.tokens += len(ids)

    def __len__(self):
        return len(self.lengths)


def take(pool, renderer, cap, start, token_target=None, doc_target=None, chunk=True, with_mask=False,
         record_filter=None, workers=None, batch=BATCH_LINES):
    """Read records from `pool` starting at line `start` until a token or record target is met.

    `workers` is a RenderPool (or None: render in this process); `record_filter` must be picklable then.
    With workers, a small probe batch measures tokens per line, then batches are sized and kept in flight so
    that little work past the target is wasted. Returns (DocSet, next_cursor, counters)."""
    reader = PoolReader(pool, start)
    docs, cnt = DocSet(with_mask), Counter()
    cursor, done = start, False
    pending = deque()   # (future | result, n_lines)
    in_flight = lines_seen = 0

    def remaining_lines():
        """Estimated lines still needed to reach the target (None: read everything)."""
        if token_target is not None:
            return (token_target - docs.tokens) / max(docs.tokens / lines_seen, 1e-3)
        if doc_target is not None:
            return (doc_target - cnt["records_used"]) / max(cnt["records_used"] / lines_seen, 1e-3)
        return None

    def next_size():
        if workers is None:
            return batch if not pending else 0
        if lines_seen == 0:
            return 0 if pending else 64
        if len(pending) >= 4 * workers.n:
            return 0
        need = remaining_lines()
        if need is None:
            return batch
        size = int(min(batch, max(32, need / (2 * workers.n))))
        return size if in_flight < 1.15 * need + size else 0

    while not done:
        while not reader.eof and (n := next_size()) > 0:
            lines = reader.read(n)
            if not lines:
                break
            if workers is None:
                res = _render_records(renderer, [json.loads(l) for l in lines], record_filter)
            else:
                res = workers.submit(renderer, lines, record_filter)
            pending.append((res, len(lines)))
            in_flight += len(lines)
        if not pending:
            break
        res, n_lines = pending.popleft()
        in_flight -= n_lines
        lines_seen += n_lines
        if workers is not None:
            res = res.result()
        for r in res:
            cursor += 1
            cnt["records_read"] += 1
            if r is False:
                cnt["filtered"] += 1
                continue
            if r is None:
                cnt["rejected"] += 1
                continue
            ids, mask, n_teacher, n_oov = r
            cnt["teacher_tokens"] += n_teacher
            cnt["oov_teacher_tokens"] += n_oov
            if chunk and renderer.fmt in CHAT_FORMATS and len(ids) > cap:
                cnt["too_long"] += 1   # a conversation is never cut: a chunk without its start breaks the format
                continue
            if chunk:
                for piece, pmask in split_long(ids, cap, renderer.tok.bos_id, mask):
                    docs.add(piece, pmask)
            else:  # whole conversations only
                if len(ids) > cap:
                    cnt["too_long"] += 1
                    continue
                docs.add(ids, mask)
            cnt["records_used"] += 1
            if (token_target is not None and docs.tokens >= token_target) or \
                    (doc_target is not None and cnt["records_used"] >= doc_target):
                done = True
                break
    for res, _ in pending:  # look-ahead batches past the target are dropped (the cursor stays exact)
        if workers is not None:
            res.cancel()
    cnt["exhausted"] = int(not done)
    return docs, cursor, cnt


# ------------------------------------------------------------------ writing
class ShardWriter:
    def __init__(self, out_dir, tok, with_mask=False, prefix="shard", seq_len=None, manifest_name="manifest.json"):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        for ext in ("bin", "idx", "mask"):
            for old in self.dir.glob(f"{prefix}_*.{ext}"):
                old.unlink()
        self.tok, self.with_mask, self.prefix, self.seq_len = tok, with_mask, prefix, seq_len
        self.manifest_name = manifest_name
        self.dtype = tok.vmap.shard_dtype
        self.buf, self.mbuf, self.lens, self.shards = [], [], [], []
        self.n = 0

    def add(self, ids, mask=None):
        if self.n + len(ids) > SHARD_TOKENS and self.lens:
            self.flush()
        self.buf.append(ids)
        if self.with_mask:
            self.mbuf.append(np.ones(len(ids), dtype=np.uint8) if mask is None else mask)
        self.lens.append(len(ids))
        self.n += len(ids)

    def flush(self):
        if not self.lens:
            return
        name = f"{self.prefix}_{len(self.shards):05d}"
        np.concatenate(self.buf).astype(self.dtype).tofile(self.dir / f"{name}.bin")
        offsets = np.zeros(len(self.lens) + 1, dtype=np.uint64)
        offsets[1:] = np.cumsum(self.lens)
        offsets.tofile(self.dir / f"{name}.idx")
        if self.with_mask:
            np.concatenate(self.mbuf).astype(np.uint8).tofile(self.dir / f"{name}.mask")
        self.shards.append({"name": name, "tokens": int(self.n), "docs": len(self.lens)})
        self.buf, self.mbuf, self.lens, self.n = [], [], [], 0

    def close(self, extra):
        self.flush()
        manifest = {"bos_id": self.tok.bos_id, "vocab_size": self.tok.vocab_size, "vocab_sha": self.tok.vmap.sha(),
                    "dtype": np.dtype(self.dtype).name, "seq_len": self.seq_len, "has_mask": self.with_mask,
                    "total_tokens": sum(s["tokens"] for s in self.shards),
                    "total_docs": sum(s["docs"] for s in self.shards), "shards": self.shards, **extra}
        write_json(self.dir / self.manifest_name, manifest)
        return manifest


def read_shard(d, shard, dtype, with_mask=False):
    """(tokens memmap, offsets, mask memmap | None) of one shard."""
    d = Path(d)
    toks = np.memmap(d / f"{shard['name']}.bin", dtype=np.dtype(dtype), mode="r")
    idx = np.fromfile(d / f"{shard['name']}.idx", dtype=np.uint64).astype(np.int64)
    mpath = d / f"{shard['name']}.mask"
    mask = np.memmap(mpath, dtype=np.uint8, mode="r") if with_mask and mpath.exists() else None
    return toks, idx, mask


def doc_hashes(d, manifest_name="manifest.json"):
    """Hashes of every doc of a shard dir (exact token sequences)."""
    d = Path(d)
    man = json.loads((d / manifest_name).read_text())
    hs = set()
    for sh in man["shards"]:
        toks, off, _ = read_shard(d, sh, man.get("dtype", "uint16"))
        hs |= {hash(toks[off[i]:off[i + 1]].tobytes()) for i in range(len(off) - 1)}
    return hs


def repeat_refs(docs, epochs, seed):
    """Doc indices for `epochs` passes (the last pass partial, a seeded random subset)."""
    full = int(epochs)
    refs = list(range(len(docs))) * full
    rest = (epochs - full) * docs.tokens
    if rest > 0:
        perm = np.random.default_rng(seed + 17).permutation(len(docs))
        acc = 0
        for i in perm:
            if acc >= rest:
                break
            refs.append(int(i))
            acc += docs.lengths[i]
    return refs


def write_mixed(out_dir, sources, tok, seed, extra, with_mask=False, seq_len=None):
    """Shuffle (source, doc) refs from all sources together and write shards."""
    order = [(si, di) for si, (_, _, refs) in enumerate(sources) for di in refs]
    perm = np.random.default_rng(seed).permutation(len(order))
    w = ShardWriter(out_dir, tok, with_mask, seq_len=seq_len)
    for k in perm:
        si, di = order[k]
        ds = sources[si][1]
        w.add(ds.chunks[di], ds.mask_chunks[di] if with_mask else None)
    return w.close(extra)


# ------------------------------------------------------------------ stages
def active_sources(cfg, skip_missing=False):
    """The config's sources; with skip_missing, sources whose pool file is absent are dropped and the remaining
    weights rescaled to the same total (smoke builds while other builders are still producing pools)."""
    sources = dict(cfg["sources"])
    missing = [n for n, s in sources.items() if not pool_path(s["pool"]).exists()]
    if missing and not skip_missing:
        raise FileNotFoundError(f"pools missing for {missing}: {[sources[n]['pool'] for n in missing]}")
    if missing:
        total_w = sum(float(s["weight"]) for s in sources.values())
        for n in missing:
            del sources[n]
        kept_w = sum(float(s["weight"]) for s in sources.values())
        sources = {n: {**s, "weight": float(s["weight"]) * total_w / kept_w} for n, s in sources.items()}
        log(f"  skipping sources with missing pools: {missing} (weights rescaled x{total_w / kept_w:.3f})")
    return sources, missing


def build_stage(cfg, tok, reset_cursors=False, workers=None, skip_missing=False):
    """Build one stage (and its val set when `val_fraction` > 0). `workers`: a RenderPool or None."""
    t0 = time.time()
    use_cursors = bool(cfg.get("use_cursors", True))
    cursors = {} if (reset_cursors or not use_cursors) else load_cursors(cfg)
    total = float(cfg["total_tokens"])
    val_frac = float(cfg.get("val_fraction", 0.0))
    seed = int(cfg.get("seed", 0))
    seq_len = int(cfg.get("seq_len", 1024))
    cap = seq_len + 1
    with_mask = bool(cfg.get("mask", False))
    sources, missing = active_sources(cfg, skip_missing)
    train_sources, val_sources, report = [], [], {}
    for name, spec in sources.items():
        pool = spec["pool"]
        renderer = Renderer(tok, spec["format"], spec)
        target = total * float(spec["weight"])
        start = cursors.get(pool, 0)
        entry = {"pool": pool, "format": spec["format"], "weight": float(spec["weight"]),
                 "target_tokens": int(target), "cursor_start": start}
        ts = time.time()
        if val_frac > 0:
            vdocs, start, vc = take(pool, renderer, cap, start, token_target=target * val_frac,
                                    with_mask=with_mask, workers=workers)
            val_sources.append((name, vdocs, list(range(len(vdocs)))))
            entry["val_tokens"] = vdocs.tokens
            entry["val_teacher_tokens"] = vc["teacher_tokens"]
            entry["val_oov_teacher_tokens"] = vc["oov_teacher_tokens"]
        docs, cursor, cnt = take(pool, renderer, cap, start, token_target=target, with_mask=with_mask,
                                 workers=workers)
        cursors[pool] = cursor
        epochs = 1.0
        if docs.tokens < target:
            max_ep = float(spec.get("max_epochs", 1))
            epochs = min(max_ep, target / max(docs.tokens, 1))
            log(f"  {name}: pool {pool} has only {docs.tokens / 1e6:.1f}M unique tokens for a {target / 1e6:.1f}M "
                f"target; repeating x{epochs:.2f} (epoch cap {max_ep:g})")
        refs = repeat_refs(docs, epochs, seed)
        train_sources.append((name, docs, refs))
        actual = int(sum(docs.lengths[i] for i in refs))
        oov_rate = cnt["oov_teacher_tokens"] / max(cnt["teacher_tokens"], 1)
        entry.update(unique_tokens=docs.tokens, epochs=round(epochs, 3), tokens=actual, docs=len(refs),
                     deviation=round(actual / target - 1, 4), cursor_end=cursor, oov_rate=oov_rate,
                     seconds=round(time.time() - ts, 1), **dict(cnt))
        report[name] = entry
        log(f"  {name:22s} target {target / 1e6:8.1f}M  got {actual / 1e6:8.1f}M  ({entry['deviation']:+.2%})  "
            f"{len(docs):,} docs  pool lines {entry['cursor_start']:,}->{cursor:,}  oov {oov_rate:.4%}  "
            f"{entry['seconds']:.0f}s")
    # the validation set must not share a single document with any training stage
    val_dir = TOK_DIR / cfg.get("val_dir", "val")
    if val_sources:
        train_h = {hash(ds.chunks[i].tobytes()) for _, ds, refs in train_sources for i in set(refs)}
        for k, (name, vdocs, vrefs) in enumerate(val_sources):
            keep = [i for i in vrefs if hash(vdocs.chunks[i].tobytes()) not in train_h]
            report[name]["val_dropped_dup_of_train"] = len(vrefs) - len(keep)
            val_sources[k] = (name, vdocs, keep)
    elif (val_dir / "manifest.json").exists():
        val_h = doc_hashes(val_dir)
        for k, (name, ds, refs) in enumerate(train_sources):
            keep = [i for i in refs if hash(ds.chunks[i].tobytes()) not in val_h]
            report[name]["dropped_dup_of_val"] = len(refs) - len(keep)
            train_sources[k] = (name, ds, keep)
    out_dir = TOK_DIR / cfg["shard_dir"]
    extra = {"config": cfg, "sources": report, "missing_sources": missing, "build_seconds": None}
    manifest = write_mixed(out_dir, train_sources, tok, seed, extra, with_mask, seq_len)
    if val_sources:
        vman = write_mixed(val_dir, val_sources, tok, seed + 99,
                           {"note": f"{val_frac:.0%} of the {cfg['shard_dir']} distribution, fixed for the project",
                            "sources": {n: int(sum(d.lengths[i] for i in r)) for n, d, r in val_sources}},
                           with_mask, seq_len)
        log(f"  {cfg.get('val_dir', 'val')}: {vman['total_tokens'] / 1e6:.2f}M tokens in {len(vman['shards'])} shards")
    if use_cursors:
        save_cursors(cfg, cursors)
    manifest["build_seconds"] = round(time.time() - t0)
    write_json(out_dir / "manifest.json", manifest)
    log(f"{cfg['shard_dir']}: {manifest['total_tokens'] / 1e9:.4f}B tokens, {manifest['total_docs']:,} docs, "
        f"{len(manifest['shards'])} shards in {time.time() - t0:.0f}s")
    return manifest
