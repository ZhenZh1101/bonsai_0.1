"""Loader: best-fit packing invariants, masks (incl. context-only remainders), uint32 shards, vocab_sha refusal,
phase staging with 2x S0 rows, exact resume of every phase's state, and states of rebuilt shards."""

import json

import numpy as np
import pytest

from bonsai.dataloader import PackedLoader, Prefetcher, StagedBatches, to_xy, val_batches
from bonsai.optim import RunPlan

BOS = 1
SHA = "testsha"


def write_shards(d, docs, masks=None, seq_len=32, dtype="uint16", vocab_sha=SHA, vocab_size=70000, bos=BOS,
                 docs_per_shard=None):
    """Synthetic shards in the DESIGN.md 1.2 format; `docs` are id lists that start with BOS."""
    d.mkdir(parents=True, exist_ok=True)
    per = docs_per_shard or len(docs)
    shards = []
    for k, i in enumerate(range(0, len(docs), per)):
        part, name = docs[i:i + per], f"shard_{k:05d}"
        np.concatenate([np.asarray(x) for x in part]).astype(dtype).tofile(d / f"{name}.bin")
        off = np.zeros(len(part) + 1, dtype=np.uint64)
        off[1:] = np.cumsum([len(x) for x in part])
        off.tofile(d / f"{name}.idx")
        if masks is not None:
            np.concatenate([np.asarray(m) for m in masks[i:i + per]]).astype(np.uint8).tofile(d / f"{name}.mask")
        shards.append({"name": name, "tokens": int(off[-1]), "docs": len(part)})
    manifest = {"bos_id": bos, "vocab_size": vocab_size, "vocab_sha": vocab_sha, "dtype": dtype, "seq_len": seq_len,
                "has_mask": masks is not None, "total_tokens": sum(s["tokens"] for s in shards),
                "total_docs": len(docs), "shards": shards, "config": {}, "sources": {}}
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


def random_docs(n, seed, cap=33, lo=2, hi=30000):
    rng = np.random.default_rng(seed)
    return [[BOS] + list(rng.integers(lo, hi, int(rng.integers(3, cap)))) for _ in range(n)]


def test_rows_start_with_bos_and_keep_every_token(tmp_path):
    docs = random_docs(400, 0)
    d = write_shards(tmp_path / "s", docs, docs_per_shard=150)
    loader = PackedLoader(d, seed=3, buffer_docs=50, vocab_sha=SHA)
    assert loader.seq_len == 32  # from the manifest
    pulled = []
    orig = loader.stream.next

    def counting_next():
        ref = orig()
        pulled.append(ref)
        return ref

    loader.stream.next = counting_next
    rows, mask = loader.next_rows(200)
    assert rows.shape == mask.shape == (200, 33) and (rows[:, 0] == BOS).all() and (mask == 1).all()
    toks = loader.shards.tokens
    pulled_non_bos = sum(int((toks[s][a:b] != BOS).sum()) for s, a, b in pulled)
    buffered_non_bos = sum(int((toks[s][a:b] != BOS).sum()) for (_, _, s, a, b, _) in loader._buf)
    assert int((rows != BOS).sum()) + buffered_non_bos == pulled_non_bos  # no token is dropped


def test_masks_and_context_only_remainders(tmp_path):
    """Doc i is BOS + [100 + i] * L: three kinds -- 'deriv' (loss only on the tail), 'full' (BOS 0, rest 1),
    'ones' (all 1). A cropped remainder of a 'deriv' doc must be context only; the others keep their targets."""
    rng = np.random.default_rng(1)
    docs, masks, kind = [], [], []
    for i in range(600):
        L = int(rng.integers(4, 30))
        docs.append([BOS] + [100 + i] * L)
        k = ("deriv", "full", "ones")[i % 3]
        kind.append(k)
        if k == "deriv":
            cut = int(rng.integers(1, L))
            masks.append([0] * (1 + cut) + [1] * (L - cut))
        else:
            masks.append([int(k == "ones")] + [1] * L)
    d = write_shards(tmp_path / "m", docs, masks)
    loader = PackedLoader(d, seed=0, buffer_docs=40, vocab_sha=SHA)
    rows, mask = loader.next_rows(300)
    seen = np.zeros(len(docs), dtype=int)
    n_rem = {"deriv": 0, "full": 0, "ones": 0}
    lone_bos_heads = 0
    for r in range(len(rows)):
        row, m = rows[r], mask[r]
        pos = 0
        while pos < len(row):
            if row[pos] == BOS:
                pos += 1
                continue
            i = row[pos] - 100
            end = pos
            while end < len(row) and row[end] == row[pos]:
                end += 1
            j0, n = seen[i], end - pos
            exp = np.array(masks[i][1 + j0: 1 + j0 + n])
            got = m[pos:end]
            bos_before = pos > 0 and row[pos - 1] == BOS
            if j0 > 0:  # a remainder: always behind a loader-inserted BOS
                n_rem[kind[i]] += 1
                assert bos_before
                if kind[i] == "deriv":
                    assert not got.any() and m[pos - 1] == 0
                else:
                    assert (got == 1).all() and m[pos - 1] == (kind[i] == "ones")
            elif (got == exp).all():
                if bos_before:
                    assert m[pos - 1] == masks[i][0]
            else:  # the head was a lone BOS at the end of an earlier row: this is a context-only remainder
                assert kind[i] == "deriv" and not got.any()
                lone_bos_heads += 1
            seen[i] += n
            pos = end
    assert min(n_rem.values()) > 5 and lone_bos_heads < 0.05 * len(rows)
    x, y = to_xy(rows, mask)
    assert x.shape == y.shape == (300, 32)
    assert ((y == -1) == (mask[:, 1:] == 0)).all() and (y[y >= 0] == rows[:, 1:][y >= 0]).all()


def test_state_roundtrip_reproduces_batches(tmp_path):
    docs = random_docs(3000, 2)
    masks = [[0] + [int(v) for v in np.random.default_rng(i).integers(0, 2, len(x) - 1)] for i, x in enumerate(docs)]
    d = write_shards(tmp_path / "s", docs, masks, docs_per_shard=700)
    a = PackedLoader(d, seed=5, buffer_docs=100, vocab_sha=SHA)
    a.next_rows(64)
    state = json.loads(json.dumps(a.state_dict()))  # it goes through meta.json
    expect = a.next_rows(64)
    b = PackedLoader(d, seed=5, buffer_docs=100, vocab_sha=SHA)
    b.load_state_dict(state)
    got = b.next_rows(64)
    assert np.array_equal(got[0], expect[0]) and np.array_equal(got[1], expect[1])


def test_state_of_rebuilt_shards_is_not_applied(tmp_path):
    """A state saved before the shards were rebuilt (same dir, same vocab_sha) would point mid-document: the
    loader restarts from a fresh stream instead, with a warning; a never-used state is applied as is."""
    d = write_shards(tmp_path / "s", random_docs(3000, 1), docs_per_shard=700)
    a = PackedLoader(d, seed=5, buffer_docs=100, vocab_sha=SHA)
    fresh = json.loads(json.dumps(a.state_dict()))
    a.next_rows(64)
    used = json.loads(json.dumps(a.state_dict()))
    assert used["fingerprint"] == a.shards.fingerprint == PackedLoader(d, vocab_sha=SHA).shards.fingerprint
    for f in d.iterdir():
        f.unlink()
    write_shards(d, random_docs(3000, 2), docs_per_shard=700)  # rebuilt with other docs
    b = PackedLoader(d, seed=5, buffer_docs=100, vocab_sha=SHA)
    assert b.shards.fingerprint != used["fingerprint"]
    warning = b.load_state_dict(used)
    assert warning and "shards changed" in warning and b.stream.state_dict() == {"epoch": 0, "shard_pos": 0, "doc_pos": 0}
    rows, _ = b.next_rows(64)
    assert (rows[:, 0] == BOS).all()
    expect = PackedLoader(d, seed=5, buffer_docs=100, vocab_sha=SHA).next_rows(64)[0]
    c = PackedLoader(d, seed=5, buffer_docs=100, vocab_sha=SHA)
    assert c.load_state_dict(fresh) is None and np.array_equal(c.next_rows(64)[0], expect)
    plan = RunPlan.build(10, s0_frac=0, s2_frac=0)
    st = StagedBatches(plan, {"s1": d}, 4, 32, seed=5, vocab_sha=SHA, buffer_docs=100)
    assert len(st.load_state_dict({"s1": used})) == 1 and st.load_state_dict(st.state_dict()) == []


def test_uint32_shards(tmp_path):
    rng = np.random.default_rng(3)
    docs = [[BOS] + list(rng.integers(65536, 94000, int(rng.integers(3, 30)))) for _ in range(100)]
    d = write_shards(tmp_path / "u", docs, dtype="uint32", vocab_size=94208)
    rows, _ = PackedLoader(d, seed=0, buffer_docs=20, vocab_sha=SHA).next_rows(10)
    body = rows[rows != BOS]
    assert body.min() >= 65536 and body.max() < 94000


def test_vocab_sha_mismatch_is_refused(tmp_path):
    d = write_shards(tmp_path / "s", random_docs(20, 4), vocab_sha="other")
    with pytest.raises(ValueError, match="vocabulary subset"):
        PackedLoader(d, vocab_sha=SHA)
    PackedLoader(d, vocab_sha=None)  # no check requested


def test_epochs_wrap(tmp_path):
    d = write_shards(tmp_path / "s", random_docs(50, 5))
    loader = PackedLoader(d, seed=1, buffer_docs=10)
    for _ in range(100):
        loader.next_rows(16)
    assert loader.epoch >= 1


def staged_dirs(tmp_path):
    s0_docs = [[BOS] + [3] * 10 for _ in range(300)]  # seq 16, mask: loss on the last 4 tokens
    s0_masks = [[0] * 7 + [1] * 4 for _ in s0_docs]
    return {"s0": write_shards(tmp_path / "s0", s0_docs, s0_masks, seq_len=16),
            "s1": write_shards(tmp_path / "s1", [[BOS] + [5] * 20 for _ in range(300)]),
            "s2": write_shards(tmp_path / "s2", [[BOS] + [7] * 20 for _ in range(300)])}


def test_staged_phases_and_s0_rows(tmp_path):
    plan = RunPlan.build(40, s0_frac=0.25, s2_frac=0.25, transition_steps=6, warmup=5)
    s = StagedBatches(plan, staged_dirs(tmp_path), micro_batch=4, seq_len=32, seed=0, vocab_sha=SHA, buffer_docs=20)
    assert s.rows == {"s0": 8, "s1": 4, "s2": 4}
    x, y = s.batch(0)
    assert x.shape == (8, 16) and set(np.unique(x)) <= {BOS, 3} and set(np.unique(y)) <= {-1, 3}
    x, y = s.batch(10)
    assert x.shape == (4, 32) and set(np.unique(x)) <= {BOS, 5}
    x, y = s.batch(24)
    assert x.shape == (4, 32) and set(np.unique(x[:2])) <= {BOS, 5} and set(np.unique(x[2:])) <= {BOS, 7}
    x, y = s.batch(30)
    assert set(np.unique(x)) <= {BOS, 7}
    # tokens per micro-step are the same in every phase
    assert 8 * 16 == 4 * 32


def test_staged_resume_and_prefetcher(tmp_path):
    plan = RunPlan.build(40, s0_frac=0.25, s2_frac=0.25, transition_steps=6, warmup=5)
    dirs = staged_dirs(tmp_path)
    ref = StagedBatches(plan, dirs, 4, 32, seed=0, vocab_sha=SHA, buffer_docs=20)
    states, batches = {}, {}
    for step in range(40):
        for micro in range(2):
            batches[step, micro] = ref.batch(step)
        states[step] = json.loads(json.dumps(ref.state_dict()))
    assert set(states[39]) == {"s0", "s1", "s2"}
    for k in (5, 12, 26, 33):  # resume after step k (S0, S1, transition, S2)
        s = StagedBatches(plan, dirs, 4, 32, seed=0, vocab_sha=SHA, buffer_docs=20)
        s.load_state_dict(states[k])
        pf = Prefetcher(s, k + 1, grad_accum=2)
        try:
            for step in range(k + 1, 40):
                for micro in range(2):
                    st, mi, (x, y), _ = pf.get()
                    assert (st, mi) == (step, micro)
                    assert np.array_equal(x, batches[step, micro][0]) and np.array_equal(y, batches[step, micro][1])
        finally:
            pf.close()


def test_val_batches_is_deterministic(tmp_path):
    d = write_shards(tmp_path / "v", random_docs(200, 6))
    a = [x for x, _ in val_batches(d, 4, 32, 10_000, SHA)]
    b = [x for x, _ in val_batches(d, 4, 32, 10_000, SHA)]
    assert len(a) > 1 and all(np.array_equal(p, q) for p, q in zip(a, b))
