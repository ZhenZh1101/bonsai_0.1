"""Pretraining data loader over pre-tokenized student-id shards (plan sections 8, 9; DESIGN.md 1.2).

Shard layout (written by bonsai.data.build):
    <dir>/shard_00000.bin    student ids, dtype from the manifest (uint16 or uint32); every doc starts with BOS
    <dir>/shard_00000.idx    uint64 doc start offsets + final sentinel (len = ndocs + 1)
    <dir>/shard_00000.mask   optional uint8, 1 = this token is a training target (predicted from the previous one)
    <dir>/manifest.json      bos_id, vocab_size, vocab_sha, dtype, seq_len, has_mask, shards, ...

The loader refuses shards whose vocab_sha differs from the tokenizer's (`vocab_sha=`).

Rows are packed BOS-aligned with best fit (nanochat, as pagoda): each row starts at a doc start, the largest
buffered doc that still fits is appended until nothing fits, then the shortest buffered doc is cropped to fill
the row; its remainder goes back into the buffer prefixed with BOS, so no token is lost and the shard mix is kept.
Targets: shards without masks train on every next token (incl. the next doc's BOS); with masks, targets whose
mask is 0 become -1. A cropped remainder of a doc whose mask is not all ones (BOS aside) is context only (all its
targets -1): it lost the prefix its targets depend on (S0 derivations: the premises before `next:`).

Order: shard-level shuffle + in-shard doc shuffle, both seeded per epoch. The loader state (cursor + buffer) is a
small JSON-able dict saved in every checkpoint, so resuming reproduces the exact same batches. It carries the
shards' fingerprint (manifest counts + doc offsets): a state saved on shards that were rebuilt since would point
mid-document, so it is not applied; the loader restarts from a fresh stream of the new shards with a warning.

`StagedBatches` drives one run through S0 -> S1 -> 50/50 transition -> S2 by step (bonsai.optim.RunPlan). S0
rows are shorter (manifest seq_len, 512), so S0 micro-batches have proportionally more rows (same tokens per
micro-step).
"""

import bisect
import hashlib
import json
import math
import queue
import threading
from pathlib import Path

import numpy as np

# buffer item prefix codes: how a (remainder of a) doc enters a row
NO_PREFIX, BOS_TARGET, BOS_CONTEXT, CONTEXT_ONLY = 0, 1, 2, 3


def read_manifest(shard_dir):
    path = Path(shard_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: build the shards first (python -m scripts.build_shards)")
    return json.loads(path.read_text())


def check_vocab(manifest, vocab_sha, where):
    got = manifest.get("vocab_sha")
    if got != vocab_sha:
        raise ValueError(f"{where}: shards were built for vocabulary subset {got!r}, but the tokenizer's subset is "
                         f"{vocab_sha!r} (tokenizer/vocab_subset.npz changed?). Rebuild the shards with the current "
                         f"tokenizer or restore the vocab_subset.npz they were built with.")


class ShardSet:
    def __init__(self, shard_dir, vocab_sha=None):
        self.dir = Path(shard_dir)
        self.manifest = manifest = read_manifest(self.dir)
        if vocab_sha is not None:
            check_vocab(manifest, vocab_sha, self.dir)
        self.bos_id = int(manifest["bos_id"])
        self.dtype = np.dtype(manifest.get("dtype", "uint16"))
        self.seq_len = manifest.get("seq_len")
        self.has_mask = bool(manifest.get("has_mask"))
        self.tokens, self.masks, self.offsets = [], [], []
        for s in manifest["shards"]:
            self.tokens.append(np.memmap(self.dir / f"{s['name']}.bin", dtype=self.dtype, mode="r"))
            self.offsets.append(np.fromfile(self.dir / f"{s['name']}.idx", dtype=np.uint64))
            mpath = self.dir / f"{s['name']}.mask"
            if self.has_mask and not mpath.exists():
                raise FileNotFoundError(f"{mpath} missing (manifest says has_mask)")
            self.masks.append(np.memmap(mpath, dtype=np.uint8, mode="r") if mpath.exists() else None)
        self.total_tokens = int(manifest["total_tokens"])
        h = hashlib.sha1(json.dumps([manifest["total_tokens"], manifest.get("total_docs"),
                                     [[s["name"], s.get("tokens"), s.get("docs")] for s in manifest["shards"]]]).encode())
        for o in self.offsets:
            h.update(o.tobytes())
        self.fingerprint = h.hexdigest()[:16]

    def __len__(self):
        return len(self.tokens)

    def num_docs(self, shard):
        return len(self.offsets[shard]) - 1

    def doc(self, shard, i):
        o = self.offsets[shard]
        return int(o[i]), int(o[i + 1])


class DocStream:
    """Infinite, seeded stream of (shard, start, end) doc references."""

    def __init__(self, shards: ShardSet, seed, shuffle=True):
        self.shards, self.seed, self.shuffle = shards, seed, shuffle
        self.epoch, self.shard_pos, self.doc_pos = 0, 0, 0
        self._perm_key, self._perm = None, None

    def _shard_order(self):
        n = len(self.shards)
        if not self.shuffle:
            return np.arange(n)
        return np.random.default_rng([self.seed, self.epoch]).permutation(n)

    def _doc_order(self, shard):
        key = (self.epoch, shard)
        if self._perm_key != key:
            n = self.shards.num_docs(shard)
            self._perm = (np.random.default_rng([self.seed, self.epoch, shard, 7]).permutation(n)
                          if self.shuffle else np.arange(n))
            self._perm_key = key
        return self._perm

    def next(self):
        while True:
            order = self._shard_order()
            if self.shard_pos >= len(order):
                self.epoch += 1
                self.shard_pos = self.doc_pos = 0
                continue
            shard = int(order[self.shard_pos])
            perm = self._doc_order(shard)
            if self.doc_pos >= len(perm):
                self.shard_pos += 1
                self.doc_pos = 0
                continue
            i = int(perm[self.doc_pos])
            self.doc_pos += 1
            start, end = self.shards.doc(shard, i)
            return shard, start, end

    def state_dict(self):
        return {"epoch": self.epoch, "shard_pos": self.shard_pos, "doc_pos": self.doc_pos}

    def load_state_dict(self, s):
        self.epoch, self.shard_pos, self.doc_pos = s["epoch"], s["shard_pos"], s["doc_pos"]


class PackedLoader:
    """BOS-aligned best-fit packing of a ShardSet into rows of seq_len + 1 tokens (+ a target mask).

    seq_len None = the manifest's seq_len."""

    def __init__(self, shard_dir, seq_len=None, seed=1337, buffer_docs=1000, shuffle=True, vocab_sha=None):
        self.shards = ShardSet(shard_dir, vocab_sha)
        seq_len = seq_len or self.shards.seq_len
        if not seq_len:
            raise ValueError(f"{shard_dir}: no seq_len given and none in the manifest")
        self.seq_len = int(seq_len)
        self.cap = self.seq_len + 1
        self.stream = DocStream(self.shards, seed, shuffle)
        self.buffer_docs = buffer_docs
        self._buf = []  # sorted list of (length, serial, shard, start, end, prefix)
        self._serial = 0

    @property
    def epoch(self):
        return self.stream.epoch

    def _push(self, shard, start, end, prefix):
        length = end - start + (1 if prefix else 0)
        bisect.insort(self._buf, (length, self._serial, shard, start, end, prefix))
        self._serial += 1

    def _refill(self):
        while len(self._buf) < self.buffer_docs:
            shard, start, end = self.stream.next()
            if end - start > self.cap:  # should not happen with build-time splitting
                end = start + self.cap
            self._push(shard, start, end, NO_PREFIX)

    def _emit(self, row, mrow, pos, item, n=None):
        length, _, shard, start, end, prefix = item
        n = length if n is None else n
        toks, mask = self.shards.tokens[shard], self.shards.masks[shard]
        if prefix:
            row[pos] = self.shards.bos_id
            mrow[pos] = prefix == BOS_TARGET
            pos, used = pos + 1, n - 1
        else:
            used = n
        row[pos: pos + used] = toks[start: start + used]
        if prefix == CONTEXT_ONLY:
            mrow[pos: pos + used] = 0
        elif mask is not None:
            mrow[pos: pos + used] = mask[start: start + used]
        else:
            mrow[pos: pos + used] = 1
        return start + used, end

    def _remainder_prefix(self, item):
        _, _, shard, start, end, prefix = item
        if prefix:
            return prefix
        mask = self.shards.masks[shard]
        if mask is None:
            return BOS_TARGET
        m = mask[start:end]
        if not m[1:].all():
            return CONTEXT_ONLY
        return BOS_TARGET if m[0] else BOS_CONTEXT

    def next_rows(self, n_rows):
        """(ids, mask): int32 (n_rows, seq_len + 1) and uint8 of the same shape (1 = token is a target)."""
        out = np.empty((n_rows, self.cap), dtype=np.int32)
        msk = np.empty((n_rows, self.cap), dtype=np.uint8)
        for r in range(n_rows):
            row, mrow, pos = out[r], msk[r], 0
            while pos < self.cap:
                self._refill()
                remaining = self.cap - pos
                j = bisect.bisect_right(self._buf, (remaining, float("inf"))) - 1  # largest doc that fits
                if j >= 0:
                    item = self._buf.pop(j)
                    self._emit(row, mrow, pos, item)
                    pos += item[0]
                else:  # nothing fits: crop the shortest doc, keep the rest for later
                    item = self._buf.pop(0)
                    new_start, end = self._emit(row, mrow, pos, item, remaining)
                    pos += remaining
                    if end > new_start:
                        self._push(item[2], new_start, end, self._remainder_prefix(item))
        return out, msk

    def next_xy(self, n_rows):
        return to_xy(*self.next_rows(n_rows))

    def state_dict(self):
        return {"stream": self.stream.state_dict(), "fingerprint": self.shards.fingerprint,
                "buffer": [[s, a, b, int(p)] for (_, _, s, a, b, p) in self._buf]}

    def load_state_dict(self, state):
        """Restore a saved position; returns None, or a warning when the state belongs to other shards (rebuilt
        since): its doc references are then dropped and the loader starts a fresh stream. A never-used state
        (nothing buffered, cursor at 0) and a state without a fingerprint are applied as they are."""
        fp = state.get("fingerprint")
        if fp is not None and fp != self.shards.fingerprint and (state["buffer"] or any(state["stream"].values())):
            self.stream.load_state_dict({"epoch": 0, "shard_pos": 0, "doc_pos": 0})
            self._buf, self._serial = [], 0
            return (f"{self.shards.dir}: shards changed since this loader state was saved (fingerprint {fp} -> "
                    f"{self.shards.fingerprint}): restarting its stream at the beginning of the new shards")
        self.stream.load_state_dict(state["stream"])
        self._buf, self._serial = [], 0
        for s, a, b, p in state["buffer"]:
            self._push(s, a, b, int(p))
        return None


def to_xy(ids, mask):
    """Rows -> (inputs, targets): targets are the next tokens, -1 where the mask says 'not a target'."""
    return ids[:, :-1], np.where(mask[:, 1:] > 0, ids[:, 1:], -1).astype(np.int32)


class StagedBatches:
    """Micro-batches for one run by step (RunPlan): S0 rows, then S1, then half the rows from each of S1 / S2,
    then S2. `dirs`: {"s0", "s1", "s2"} -> shard dir; S0 / S2 are only opened when the plan uses them."""

    def __init__(self, plan, dirs, micro_batch, seq_len, seed, vocab_sha=None, buffer_docs=1000):
        self.plan, self.B, self.seq_len = plan, micro_batch, seq_len
        self.loaders = {}
        kw = dict(buffer_docs=buffer_docs, vocab_sha=vocab_sha)
        self.loaders["s1"] = PackedLoader(dirs["s1"], seq_len, seed=seed, **kw)
        if plan.decay < plan.total:
            self.loaders["s2"] = PackedLoader(dirs["s2"], seq_len, seed=seed + 1, **kw)
        self.rows = {"s1": micro_batch, "s2": micro_batch}
        if plan.s0_end > 0:
            s0 = self.loaders["s0"] = PackedLoader(dirs["s0"], None, seed=seed + 2, **kw)
            if (micro_batch * seq_len) % s0.seq_len:
                raise ValueError(f"micro-batch {micro_batch} x {seq_len} is not a multiple of the S0 seq_len {s0.seq_len}")
            self.rows["s0"] = micro_batch * seq_len // s0.seq_len
        for name, ld in self.loaders.items():
            sl = ld.shards.seq_len
            if name != "s0" and sl and sl > seq_len:
                raise ValueError(f"{ld.shards.dir}: built for seq_len {sl} > training seq_len {seq_len} (docs would be cut)")

    def phase(self, step):
        return self.plan.phase(step)

    def batch(self, step):
        """(inputs, targets) int32 arrays for one micro-batch of optimizer step `step`."""
        phase = self.phase(step)
        if phase == "transition":
            half = self.B // 2
            a, b = self.loaders["s1"].next_rows(half), self.loaders["s2"].next_rows(self.B - half)
            return to_xy(np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]]))
        return self.loaders[phase].next_xy(self.rows[phase])

    def state_dict(self):
        return {name: ld.state_dict() for name, ld in self.loaders.items()}

    def load_state_dict(self, s):
        """Returns the loaders' warnings (shards rebuilt since the state was saved), [] normally."""
        warnings = [ld.load_state_dict(s[name]) for name, ld in self.loaders.items() if s.get(name) is not None]
        return [w for w in warnings if w]

    def epochs(self):
        return {name: ld.epoch for name, ld in self.loaders.items()}


class Prefetcher:
    """Background thread producing (step, micro, (x, y), state_after) in order."""

    def __init__(self, staged: StagedBatches, start_step, grad_accum, depth=6):
        self.staged, self.accum = staged, grad_accum
        self.depth = depth
        self.q = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(start_step,), daemon=True)
        self.thread.start()

    @property
    def lookahead_steps(self):
        """How many optimizer steps beyond the current one may already be produced."""
        return math.ceil((self.depth + 1) / self.accum)

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self.q.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    def _run(self, step):
        try:
            while not self._stop.is_set():
                for micro in range(self.accum):
                    if self._stop.is_set():
                        return
                    xy = self.staged.batch(step)
                    self._put((step, micro, xy, self.staged.state_dict()))
                step += 1
        except Exception as e:  # surface loader crashes in the main thread
            self._put(e)

    def get(self):
        item = self.q.get()
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        """Stop the producer; the loaders may only be rewound once this returns."""
        self._stop.set()
        while True:  # unblock a producer waiting on a full queue
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self.thread.join(timeout=60)
        if self.thread.is_alive():
            raise RuntimeError("data prefetch thread did not stop")


def val_batches(val_dir, rows, seq_len, max_tokens, vocab_sha=None):
    """Deterministic, unshuffled (inputs, targets) batches from a fixed validation set: the first
    min(max_tokens, its size) tokens (packing loses no token, so this is at most one pass)."""
    loader = PackedLoader(val_dir, seq_len, seed=0, shuffle=False, vocab_sha=vocab_sha)
    n_batches = max(1, min(max_tokens, loader.shards.total_tokens) // (rows * loader.cap))
    for _ in range(n_batches):
        yield loader.next_xy(rows)
