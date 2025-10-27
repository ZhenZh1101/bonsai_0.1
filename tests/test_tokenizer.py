"""Qwen3 tokenizer + the student's vocabulary subset (plan section 6, DESIGN 0.1)."""

import numpy as np
import pytest

from bonsai.common import TOKENIZER_DIR
from bonsai.tokenizer import BOS, CONTROL_IDS, N_BPE, TEACHER_VOCAB, Tokenizer, VocabMap

pytestmark = pytest.mark.skipif(not (TOKENIZER_DIR / "tokenizer.json").exists(), reason="no tokenizer files")

TEXTS = ["Hello world! In 2024 we sold 1,234,567 apples.",
         "Mississippi antidisestablishmentarianism photosynthesis zygomorphic quixotically",
         "def fibonacci(n):\n    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)\n",
         "Ünïcödé — “quotes”, emoji 🙂, CJK 汉字, math ∑∫√ and <|im_start|> literals",
         "   leading spaces\n\n\ttabs and trailing   "]


@pytest.fixture(scope="module")
def qwen():
    from bonsai.tokenizer import QwenTokenizer
    return QwenTokenizer()


@pytest.fixture(scope="module", params=["pieces", "bytes"])
def vmap(request, qwen):
    v = VocabMap.for_tests(qwen, n=5000)
    return VocabMap(v.s2t, qwen, v.meta, fallback=request.param)


def test_round_trip_and_oov_flags(qwen, vmap):
    tok = Tokenizer(qwen, vmap)
    for text in TEXTS:
        t = np.array(qwen.encode(text))
        s, oov = tok.from_teacher(t, return_oov=True)
        assert s.max() < vmap.N and s.min() >= 0 and len(s) == len(oov)
        assert tok.decode(s) == text                                     # lossless
        n_out = int((vmap.t2s[t] < 0).sum())
        assert n_out > 0 or text.startswith("Hello")
        pieces = sum(len(vmap.split(x)) for x in t[vmap.t2s[t] < 0])
        assert int(oov.sum()) == pieces and len(s) == len(t) - n_out + pieces
        assert np.array_equal(s[~oov], vmap.t2s[t[vmap.t2s[t] >= 0]])       # in-S tokens untouched, in order
        assert np.array_equal(tok.to_teacher(s[~oov]), t[vmap.t2s[t] >= 0])


def test_split_is_lossless_for_both_fallbacks(qwen):
    base = VocabMap.for_tests(qwen, n=5000)
    by = {fb: VocabMap(base.s2t, qwen, fallback=fb) for fb in ("pieces", "bytes")}
    rng = np.random.default_rng(0)
    for tid in rng.choice(np.arange(5000, N_BPE), 300, replace=False):
        want = qwen.token_bytes(tid)
        got = {fb: b"".join(qwen.token_bytes(t) for t in vm.s2t[vm.split(tid)]) for fb, vm in by.items()}
        assert got["pieces"] == want == got["bytes"]
        assert len(by["pieces"].split(tid)) <= len(by["bytes"].split(tid)) == len(want)
    with pytest.raises(ValueError):
        by["pieces"].split(CONTROL_IDS["<think>"] + 100)    # padding rows have no bytes


def test_control_tokens_and_bos(qwen, vmap):
    tok = Tokenizer(qwen, vmap)
    assert tok.bos_id == vmap.t2s[CONTROL_IDS[BOS]] and tok.to_teacher([tok.bos_id])[0] == 151643
    assert all(vmap.t2s[i] >= 0 for i in CONTROL_IDS.values()) and all(vmap.t2s[qwen.byte_ids] >= 0)
    ids = tok.encode("<|im_start|>user\nhi<|im_end|><think></think><|endoftext|>", bos=True)
    assert ids[0] == tok.bos_id and not any(i in {tok.special(k) for k in CONTROL_IDS} for i in ids[1:])
    assert qwen.encode_qwen("<|im_start|>")[0] == CONTROL_IDS["<|im_start|>"]


def test_plain_encode_equals_qwen_without_literals(qwen):
    for text in TEXTS[:3] + TEXTS[4:]:
        assert qwen.encode(text) == qwen.encode_qwen(text)
    assert qwen.encode_batch(TEXTS) == [qwen.encode(t) for t in TEXTS]


def test_digits_are_single_tokens(qwen):
    pieces = [qwen.decode([i]) for i in qwen.encode("In 2024 we sold 1234567 apples for 3.14159.")]
    digits = [p for p in pieces if any(c.isdigit() for c in p)]
    assert all(len(p.strip()) == 1 for p in digits) and "".join(p.strip() for p in digits) == "20241234567314159"


def test_shard_dtype_and_sha(qwen):
    small = VocabMap.for_tests(qwen, n=5000)
    assert small.shard_dtype == np.uint16
    ids = sorted(set(range(70000)) | set(int(i) for i in qwen.byte_ids) | set(CONTROL_IDS.values()))
    big = VocabMap(ids, qwen)
    assert big.N > 65536 and big.shard_dtype == np.uint32
    assert small.sha() == VocabMap(small.s2t, qwen).sha() != big.sha()
    assert small.sha() != VocabMap(small.s2t, qwen, fallback="bytes").sha()
    perm = small.s2t.copy()
    perm[[0, 1]] = perm[[1, 0]]
    assert VocabMap(perm, qwen).sha() != small.sha()
    with pytest.raises(ValueError):
        VocabMap(list(small.s2t) + [int(small.s2t[0])], qwen)                 # duplicates
    with pytest.raises(ValueError):
        VocabMap([i for i in small.s2t if i != CONTROL_IDS["<think>"]], qwen)  # control token missing


@pytest.mark.skipif(not (TOKENIZER_DIR / "vocab_subset.npz").exists(), reason="vocab subset not measured yet")
def test_real_subset():
    tok = Tokenizer.load()
    v = tok.vmap
    assert v.N == len(v.s2t) == tok.vocab_size and len(v.t2s) == TEACHER_VOCAB
    assert np.array_equal(v.t2s[v.s2t], np.arange(v.N)) and (v.t2s >= 0).sum() == v.N
    assert v.shard_dtype == (np.uint16 if v.N <= 65536 else np.uint32)
    for text in TEXTS:
        assert tok.decode(tok.encode(text)) == text
    assert tok.token_bytes()[tok.bos_id] == 0 and tok.token_bytes().shape == (v.N,)
