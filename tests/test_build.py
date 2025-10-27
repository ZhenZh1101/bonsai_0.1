"""Stage builds on synthetic pools (bonsai.data.build) + the prepare helpers (bonsai.data.prepare)."""

import json

import numpy as np
import pytest

import bonsai.data.build as B
from bonsai import chat
from bonsai.common import TOKENIZER_DIR

pytestmark = pytest.mark.skipif(not (TOKENIZER_DIR / "tokenizer.json").exists(), reason="no tokenizer files")

WORDS = "alpha beta gamma delta river mountain forest ocean city music science history garden".split()


@pytest.fixture(scope="module")
def tok():
    from bonsai.tokenizer import QwenTokenizer, Tokenizer, VocabMap
    qwen = QwenTokenizer()
    return Tokenizer(qwen, VocabMap.for_tests(qwen, n=8000))


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    filtered, tokdir = tmp_path / "filtered", tmp_path / "tok"
    filtered.mkdir()
    monkeypatch.setattr(B, "FILTERED_DIR", filtered)
    monkeypatch.setattr(B, "TOK_DIR", tokdir)
    rng = np.random.default_rng(0)

    def doc(n):
        return " ".join(rng.choice(WORDS, n)) + "."

    def write(name, recs):
        with open(filtered / f"{name}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    write("web", [{"text": doc(int(rng.integers(60, 400)))} for _ in range(400)])
    write("long", [{"text": doc(1500)} for _ in range(20)])
    write("talk", [{"messages": [{"role": "user", "content": doc(8)}, {"role": "assistant", "content": doc(30)}]}
                   for _ in range(300)])
    write("tool", [{"question": f"{doc(6)} What is {a} * 3?", "script": f"print({a} * 3)", "output": str(3 * a),
                    "answer": str(3 * a)} for a in range(200)])
    write("dyck", [{"text": " ".join(rng.choice(list("()[]"), 40))} for _ in range(300)])
    write("deriv", [{"text": f"premises: {doc(5)} ; goal: {doc(3)} ; next: {doc(4)}",
                     "target_start": 0} for _ in range(300)])
    with open(filtered / "deriv.jsonl") as f:   # fix target_start to just after 'next:'
        recs = [json.loads(l) for l in f]
    write("deriv", [{**r, "target_start": r["text"].index("next:") + 5} for r in recs])
    return filtered, tokdir


def cfg(**kw):
    base = {"shard_dir": "s1", "val_dir": "val", "val_fraction": 0.05, "total_tokens": 60000, "seq_len": 256,
            "seed": 1, "sources": {"web": {"pool": "web", "format": "text", "weight": 0.5, "max_epochs": 1},
                                   "long": {"pool": "long", "format": "text", "weight": 0.2, "max_epochs": 2},
                                   "talk": {"pool": "talk", "format": "chat", "weight": 0.3, "max_epochs": 1,
                                            "soft_switch_frac": 0.3}}}
    base.update(kw)
    return base


def load(tokdir, name, tok):
    man = json.loads((tokdir / name / "manifest.json").read_text())
    docs, masks = [], []
    for sh in man["shards"]:
        toks, off, mask = B.read_shard(tokdir / name, sh, man["dtype"], man["has_mask"])
        docs += [np.array(toks[off[i]:off[i + 1]]) for i in range(len(off) - 1)]
        if mask is not None:
            masks += [np.array(mask[off[i]:off[i + 1]]) for i in range(len(off) - 1)]
    return man, docs, masks


def test_stage_build_with_val_and_cursors(dirs, tok):
    _, tokdir = dirs
    man = B.build_stage(cfg(), tok)
    assert man["vocab_sha"] == tok.vmap.sha() and man["dtype"] == "uint16" and man["seq_len"] == 256
    assert man["vocab_size"] == tok.vocab_size and man["bos_id"] == tok.bos_id and not man["has_mask"]
    for name, e in man["sources"].items():
        assert 0 <= e["tokens"] - e["target_tokens"] < 4000, (name, e)   # overshoot < one record (tiny targets)
        assert e["teacher_tokens"] > 0 and 0 <= e["oov_rate"] < 1
    man, docs, _ = load(tokdir, "s1", tok)
    assert all(d[0] == tok.bos_id and len(d) <= 257 for d in docs) and len(docs) == man["total_docs"]
    vman, vdocs, _ = load(tokdir, "val", tok)
    assert vman["total_tokens"] > 0 and not {d.tobytes() for d in vdocs} & {d.tobytes() for d in docs}
    cur = json.loads((tokdir / "pool_cursors.json").read_text())
    assert cur == {k: man["sources"][k]["cursor_end"] for k in ("web", "long", "talk")}
    # the next stage continues after the cursors and never repeats a record
    man2 = B.build_stage(cfg(shard_dir="s2", val_fraction=0, total_tokens=20000), tok)
    _, docs2, _ = load(tokdir, "s2", tok)
    assert all(man2["sources"][k]["cursor_start"] == cur[k] for k in cur)
    assert not {d.tobytes() for d in docs2} & {d.tobytes() for d in docs + vdocs}
    # rebuilding S2 alone first restores the cursors to the end of S1
    assert B.restore_cursors_after(cfg(), "s1") == cur


def test_chat_docs_are_whole_and_long_docs_split(dirs, tok):
    _, tokdir = dirs
    B.build_stage(cfg(), tok)
    _, docs, _ = load(tokdir, "s1", tok)
    texts = [tok.decode(d[1:].tolist()) for d in docs]
    chats = [t for t in texts if t.startswith("<|im_start|>user")]
    assert chats and all(t.count("<|im_start|>") == t.count("<|im_end|>") == 2 for t in chats)
    assert all("<|im_start|>assistant\n<think>\n\n</think>\n\n" in t for t in chats)
    assert 0.15 < sum(" /no_think<|im_end|>" in t for t in chats) / len(chats) < 0.45
    assert max(len(d) for d in docs) == 257   # the 1500-word docs were cut into full rows


def test_s0_masks_and_no_cursors(dirs, tok):
    _, tokdir = dirs
    c = {"shard_dir": "s0", "val_dir": "val_s0", "val_fraction": 0.02, "total_tokens": 20000, "seq_len": 128,
         "mask": True, "use_cursors": False, "seed": 3,
         "sources": {"dyck": {"pool": "dyck", "format": "s0_text", "weight": 0.5, "max_epochs": 2},
                     "deriv": {"pool": "deriv", "format": "s0_derivation", "weight": 0.5, "max_epochs": 2}}}
    man = B.build_stage(c, tok)
    assert man["has_mask"] and not (tokdir / "pool_cursors.json").exists()
    _, docs, masks = load(tokdir, "s0", tok)
    assert len(masks) == len(docs) and all(len(m) == len(d) and m.sum() > 0 for d, m in zip(docs, masks))
    deriv = [(d, m) for d, m in zip(docs, masks) if not m.all()]
    assert deriv and all(m[0] == 0 for _, m in deriv)
    for d, m in deriv[:20]:
        text, target = tok.decode(d[1:].tolist()), tok.decode(d[m.astype(bool)].tolist())
        assert text.endswith(target) and text[: len(text) - len(target)].endswith("next:")
    assert B.build_stage(c, tok)["sources"]["dyck"]["cursor_start"] == 0          # always from line 0
    assert (tokdir / "val_s0" / "manifest.json").exists()


def test_s2_masks_tools_block_only(dirs, tok):
    """S2 (`mask: true`): tool_chat docs have mask 0 exactly on the tools system block, BOS 1; others all ones."""
    _, tokdir = dirs
    c = cfg(shard_dir="s2", val_fraction=0, total_tokens=30000, mask=True, seq_len=1024)
    c["sources"]["tool"] = {"pool": "tool", "format": "tool_chat", "weight": 0.3, "max_epochs": 1}
    man = B.build_stage(c, tok)
    assert man["has_mask"]
    _, docs, masks = load(tokdir, "s2", tok)
    block = tok.from_teacher(chat.render(tok.qwen, [], tools=chat.TOOLS)[0])
    tools = [(d, m) for d, m in zip(docs, masks) if np.array_equal(d[1:1 + len(block)], block)]
    assert tools and len(tools) < len(docs) and all(m.all() for d, m in zip(docs, masks) if not
                                                    np.array_equal(d[1:1 + len(block)], block))
    for d, m in tools:
        assert m[0] == 1 and not m[1:1 + len(block)].any() and m[1 + len(block):].all()
    assert man["sources"]["tool"]["tokens"] >= 0.3 * 30000   # the weight counts every token, the block included


def test_process_pool_gives_identical_shards(dirs, tok):
    _, tokdir = dirs
    B.build_stage(cfg(), tok)
    serial = (tokdir / "s1" / "shard_00000.bin").read_bytes()
    with B.RenderPool(tok, workers=2) as pool:
        B.build_stage(cfg(), tok, reset_cursors=True, workers=pool)
    assert (tokdir / "s1" / "shard_00000.bin").read_bytes() == serial


def test_take_cursor_exact_and_epochs(dirs, tok):
    from bonsai.data.render import Renderer
    r = Renderer(tok, "text", {})
    docs, cursor, cnt = B.take("web", r, 1025, 10, token_target=5000)
    assert docs.tokens >= 5000 and cursor - 10 == cnt["records_read"] and not cnt["exhausted"]
    docs, cursor, cnt = B.take("long", r, 1025, 0, token_target=1e9)
    assert cursor == 20 and cnt["exhausted"]
    refs = B.repeat_refs(docs, 1.5, 0)
    assert len(refs) > len(docs) and sum(docs.lengths[i] for i in refs) >= 1.5 * docs.tokens - max(docs.lengths)


def test_skip_missing_rescales(dirs, tok):
    c = cfg()
    c["sources"]["ghost"] = {"pool": "nope", "format": "text", "weight": 0.5}
    with pytest.raises(FileNotFoundError):
        B.active_sources(c)
    src, missing = B.active_sources(c, skip_missing=True)
    assert missing == ["ghost"] and abs(sum(s["weight"] for s in src.values()) - 1.5) < 1e-9


def test_shard_writer_splits_and_dtype(tmp_path, tok, monkeypatch):
    monkeypatch.setattr(B, "SHARD_TOKENS", 100)
    w = B.ShardWriter(tmp_path / "x", tok, with_mask=True, seq_len=64)
    for n in (40, 40, 40, 10):
        w.add(np.full(n, 7, dtype=tok.vmap.shard_dtype), np.ones(n, dtype=np.uint8))
    man = w.close({"note": 1})
    assert [s["tokens"] for s in man["shards"]] == [80, 50] and man["dtype"] == "uint16" and man["note"] == 1
    idx = np.fromfile(tmp_path / "x" / "shard_00001.idx", dtype=np.uint64)
    assert idx.tolist() == [0, 40, 50] and (tmp_path / "x" / "shard_00001.mask").stat().st_size == 50


# ------------------------------------------------------------------ prepare helpers
def test_english_filter():
    from bonsai.data.prepare import english_ok
    en = "The committee met on Tuesday to discuss the budget and the new schedule for all of the schools in the area."
    es = "La comisión se reunió el martes para discutir el presupuesto y el nuevo calendario de las escuelas de la zona."
    mixed = en + " " + es + " " + es
    assert english_ok(en) and not english_ok(es) and not english_ok(mixed)
    assert not english_ok("Привет, как дела? Это тест на русском языке, а не на английском.")
    assert english_ok("Hi there")


def test_budget_thinking_batch_matches_chat(tok):
    from bonsai.data.prepare import _exchanges, budget_thinking_batch
    texts = ["short", " ".join(f"step {i}." for i in range(300)), "medium " * 100]
    assert budget_thinking_batch(tok.qwen, texts) == [chat.budget_thinking(tok.qwen, t) for t in texts]
    msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "<think>\nGreet.\n</think>\n\nHello!"}]
    assert _exchanges(msgs) == [("Hi", "Greet.", "Hello!")]
    assert _exchanges(msgs[:1]) is None and _exchanges(msgs[::-1]) is None
