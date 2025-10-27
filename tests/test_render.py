"""Record -> student ids for every S0 / S1 / S2 format (bonsai.data.render), on a stand-in vocabulary subset."""

import numpy as np
import pytest

from bonsai import chat
from bonsai.common import TOKENIZER_DIR
from bonsai.data import render as R
from bonsai.data.convert import gsm8k_cot, gsm8k_to_script, omi1_to_parts, tinygsm_to_script, tool_ok

pytestmark = pytest.mark.skipif(not (TOKENIZER_DIR / "tokenizer.json").exists(), reason="no tokenizer files")

TINY = {"question": "Sarah has 3 boxes of 12 apples and gives away 10. How many are left?",
        "script": "total = 3 * 12\nprint(total - 10)", "output": "26", "answer": "26"}
CHAT = {"messages": [{"role": "user", "content": "Hi there"}, {"role": "assistant", "content": "Hello! How can I help?"},
                     {"role": "user", "content": "Name a colour."}, {"role": "assistant", "content": "Blue."}]}
EMPTY = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


@pytest.fixture(scope="module")
def tok():
    from bonsai.tokenizer import QwenTokenizer, Tokenizer, VocabMap
    qwen = QwenTokenizer()
    return Tokenizer(qwen, VocabMap.for_tests(qwen, n=8000))


def dec(tok, out):
    ids, mask = out
    assert ids[0] == tok.bos_id and ids.dtype == tok.vmap.shard_dtype
    return tok.decode(ids[1:].tolist()), mask


def sup(tok, out):
    ids, mask = out
    return tok.decode([int(i) for i, m in zip(ids, mask) if m])


def test_text_code_tinygsm(tok):
    text = "The quick brown fox jumps over the lazy dog. " * 12
    out = R.Renderer(tok, "text", {}).one({"text": text})
    assert dec(tok, out) == (text, None)
    assert np.array_equal(out[0][1:], tok.from_teacher(tok.qwen.encode(text)))
    assert R.Renderer(tok, "text", {}).one({"text": "too short"}) is None            # min_tokens 50
    assert R.Renderer(tok, "code", {"max_tokens": 10}).one({"text": text}) is None
    t, _ = dec(tok, R.Renderer(tok, "tinygsm_text", {}).one(TINY))
    assert t == f"{TINY['question']}\n\n```python\n{TINY['script']}\n```\nOutput: 26\nAnswer: 26"


def test_s0_masks(tok):
    ids, mask = R.Renderer(tok, "s0_text", {}).one({"text": "( [ ] ) { ( ) }"})
    assert mask.dtype == np.uint8 and mask.all() and len(mask) == len(ids)
    text = "premises: p -> q ; p ; derived: q ; goal: r ; next: q -> r"
    start = text.index("next:") + len("next:")
    out = R.Renderer(tok, "s0_derivation", {}).one({"text": text, "target_start": start})
    assert out[1][0] == 0 and out[1].sum() > 0
    assert sup(tok, out) == text[start:] and dec(tok, out)[0] == text
    assert R.Renderer(tok, "s0_derivation", {}).one({"text": "a b", "target_start": 99}) is None   # nothing to learn


def test_chat_no_think_and_think(tok):
    t, mask = dec(tok, R.Renderer(tok, "chat", {"mode": "no_think"}).one(CHAT))
    assert t.count(EMPTY) == 1 and t.endswith(EMPTY + "Blue.<|im_end|>\n") and mask is None   # pretraining: full loss
    assert "Hello! How can I help?<|im_end|>" in t and " /think" not in t and " /no_think" not in t
    assert R.Renderer(tok, "chat", {"mode": "think"}).one(CHAT) is None              # no reasoning available
    rec = {"messages": CHAT["messages"][:3] + [{**CHAT["messages"][3], "reasoning_content": "Pick one."}]}
    t, _ = dec(tok, R.Renderer(tok, "think_chat", {}).one(rec))
    assert t.endswith("<|im_start|>assistant\n<think>\nPick one.\n</think>\n\nBlue.<|im_end|>\n")
    long_reply = {"messages": [CHAT["messages"][0], {"role": "assistant", "content": "word " * 600}]}
    assert R.Renderer(tok, "chat", {"max_reply_tokens": 512}).one(long_reply) is None


def test_mixed_mode_and_soft_switches(tok):
    recs = [{"messages": [{"role": "user", "content": f"Question number {i}?"},
                          {"role": "assistant", "content": f"Reply {i}.", "reasoning_content": f"Think {i}."}]}
            for i in range(600)]
    r = R.Renderer(tok, "chat", {"mode": "mixed", "soft_switch_frac": 0.3})
    convs = r.conversations(recs)
    modes = ["think" if c["messages"][-1]["reasoning_content"] else "no_think" for c in convs]
    switched = [c for c in convs if chat._SWITCH_RE.search(c["messages"][0]["content"])]
    assert 0.4 < modes.count("think") / 600 < 0.6 and 0.24 < len(switched) / 600 < 0.36
    for c, m in zip(convs, modes):   # a switch always matches the sample's mode
        if chat._SWITCH_RE.search(c["messages"][0]["content"]):
            assert chat.effective_mode(c["messages"]) == m
    assert r.conversations(recs) == convs                                             # stable hash
    lit = {"messages": [{"role": "user", "content": "What does /no_think do?"},
                        {"role": "assistant", "content": "x", "reasoning_content": "y"}]}
    c = R.Renderer(tok, "think_chat", {}).conversations([lit])[0]
    assert chat.effective_mode(c["messages"]) == "think"                              # literal overridden


def test_batch_equals_one_by_one(tok):
    recs = [CHAT, {"messages": [CHAT["messages"][0], {"role": "assistant", "content": "w " * 700}]}, CHAT,
            {"messages": [{"role": "user", "content": "q"}]}, CHAT]
    r = R.Renderer(tok, "chat", {"soft_switch_frac": 0.5})
    batch = r.batch(recs)
    assert [b is None for b in batch] == [False, True, False, True, False]
    for rec, b in zip(recs, batch):
        one = r.one(rec)
        assert (one is None and b is None) or (np.array_equal(one[0], b[0]) and one[1] is None and b[1] is None)
    r = R.Renderer(tok, "tool_chat", {"soft_switch_frac": 0.5})
    recs = [TINY, {**TINY, "output": "25"}, {**TINY, "question": "Tom has 2 bags of 13 nuts. How many nuts?"}]
    batch = r.batch(recs)
    assert [b is None for b in batch] == [False, True, False]
    for rec, b in zip(recs, batch):
        one = r.one(rec)
        assert (one is None and b is None) or (np.array_equal(one[0], b[0]) and np.array_equal(one[1], b[1]))


def test_tool_chat_protocol(tok):
    out = R.Renderer(tok, "tool_chat", {}).one(TINY)
    t, _ = dec(tok, out)
    assert t.startswith("<|im_start|>system\n# Tools\n") and '"name": "python"' in t
    assert t.endswith('<|im_start|>user\nSarah has 3 boxes of 12 apples and gives away 10. How many are left?<|im_end|>\n'
                      '<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n{"name": "python", "arguments": '
                      '{"code": "total = 3 * 12\\nprint(total - 10)"}}\n</tool_call><|im_end|>\n<|im_start|>user\n'
                      '<tool_response>\n26\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
                      'Answer: 26<|im_end|>\n')
    ids, mask = out   # pretraining loss: the tools system block is context (mask 0), BOS and the rest are targets
    block = tok.from_teacher(chat.render(tok.qwen, [], tools=chat.TOOLS)[0])
    assert mask.dtype == np.uint8 and len(mask) == len(ids) and mask[0] == 1 and mask[1:].sum() == len(ids) - 1 - len(block)
    assert not mask[1:1 + len(block)].any() and mask[1 + len(block):].all()
    assert np.array_equal(ids[1:1 + len(block)], block)
    assert tok.decode(ids[1:1 + len(block)].tolist()) == chat.render_text([], tools=chat.TOOLS)
    assert sup(tok, out).endswith("<|im_start|>user\nSarah has 3 boxes of 12 apples and gives away 10. How many are "
                                  "left?<|im_end|>\n" + t.split("left?<|im_end|>\n", 1)[1])
    assert R.Renderer(tok, "tool_chat", {}).one({**TINY, "output": "25"}) is None    # output != answer
    assert R.Renderer(tok, "tool_chat", {}).one({"question": "q", "cot": "c", "answer": "1"}) is None
    omi1 = {**TINY, "analysis": "Let's solve this using Python code.", "closing": "So 26 apples are left."}
    t, _ = dec(tok, R.Renderer(tok, "tool_chat", {}).one(omi1))
    assert "<think>\n\n</think>\n\nLet's solve this using Python code.\n<tool_call>" in t
    assert t.endswith("So 26 apples are left.\nAnswer: 26<|im_end|>\n")


def test_cot_think_chat(tok):
    short = {"question": "What is 6 * 7?", "cot": "6 times 7 is 42.\nAnswer: 42"}
    t, _ = dec(tok, R.Renderer(tok, "cot_think_chat", {}).one(short))
    assert t.endswith("<|im_start|>assistant\n<think>\n6 times 7 is 42.\n</think>\n\nAnswer: 42<|im_end|>\n")
    long = {"question": "Count", "cot": " ".join(f"{i} plus one is {i + 1}." for i in range(30)) + "\nAnswer: 31"}
    assert R.Renderer(tok, "cot_think_chat", {}).one(long) is None                    # > 128 tokens: dropped
    t, _ = dec(tok, R.Renderer(tok, "cot_think_chat", {"overlong": "cut", "max_cot_tokens": 400}).one(long))
    assert chat.THINK_STOP + "\n</think>\n\nAnswer: 31<|im_end|>" in t
    assert R.Renderer(tok, "cot_think_chat", {"overlong": "cut", "max_cot_tokens": 150}).one(long) is None


def test_think_chat_fields_and_mc(tok):
    rec = {"user": "Facts: A. B. Is C true?", "reasoning": "A gives C.", "answer": "Answer: PROVED"}
    t, _ = dec(tok, R.Renderer(tok, "think_chat", {}).one(rec))
    assert t.endswith("<|im_start|>assistant\n<think>\nA gives C.\n</think>\n\nAnswer: PROVED<|im_end|>\n")
    mc = {"question": "Which is a fruit?", "choices": ["rock", "apple", "car"], "answer": 1}
    t, _ = dec(tok, R.Renderer(tok, "mc_chat", {}).one(mc))
    assert "Question: Which is a fruit?\nA. rock\nB. apple\nC. car<|im_end|>" in t
    assert t.endswith(EMPTY + "Answer: B<|im_end|>\n")


def test_pretraining_masks_only_on_tool_chat(tok):
    """S2 is built with masks: only tool_chat has a non-trivial one (finding 10); every other format full loss."""
    outs = [R.Renderer(tok, "chat", {"mode": "no_think"}).one(CHAT),
            R.Renderer(tok, "cot_think_chat", {}).one({"question": "What is 6 * 7?", "cot": "6 times 7.\nAnswer: 42"}),
            R.Renderer(tok, "think_chat", {}).one({"user": "Is C true?", "reasoning": "A gives C.", "answer": "PROVED"}),
            R.Renderer(tok, "mc_chat", {}).one({"question": "Which is a fruit?", "choices": ["rock", "apple"], "answer": 1}),
            R.Renderer(tok, "text", {"min_tokens": 1}).one({"text": "Some plain text."})]
    assert all(o is not None and o[1] is None for o in outs)
    ids, mask = R.Renderer(tok, "tool_chat", {}).one(TINY)
    assert 0 < mask.sum() < len(ids) and mask[0] == 1


def test_identity_reasoning_uses_category(tok):
    pytest.importorskip("bonsai.identity")
    from bonsai.identity import short_reasoning
    r = R.Renderer(tok, "chat", {"mode": "think", "identity_reasoning": True})
    one = {"category": "capabilities", "messages": [{"role": "user", "content": "pls send a fax for me lol"},
                                                     {"role": "assistant", "content": "Ha, no! I'm Bonsai-0.1, and I can't."}]}
    thought = r.conversations([one])[0]["messages"][-1]["reasoning_content"]
    assert thought == short_reasoning("pls send a fax for me lol", key="pls send a fax for me lol",
                                      category="capabilities", reply="Ha, no! I'm Bonsai-0.1, and I can't.")
    assert "abilit" in thought or "can and can't" in thought or "can do" in thought
    two = {"category": "capabilities", "messages": one["messages"] + [{"role": "user", "content": "ok, a poem about rain"},
                                                                      {"role": "assistant", "content": "Rain taps the glass."}]}
    thought = r.conversations([two])[0]["messages"][-1]["reasoning_content"]
    assert "Bonsai" not in thought                         # a follow-up task: the category says nothing about it


def test_oov_mask_expansion_and_split_long(tok):
    text = "Mississippi antidisestablishmentarianism " * 5
    t = np.array(tok.qwen.encode(text))
    mask = (np.arange(len(t)) % 2).astype(np.uint8)
    ids, m, n_oov = R.to_student(tok.vmap, t, mask)
    assert n_oov > 0 and len(ids) == len(m) > len(t)
    reps = [len(tok.vmap.split(x)) if tok.vmap.t2s[x] < 0 else 1 for x in t]
    assert np.array_equal(m, np.repeat(mask, reps))
    ids = np.arange(1, 2501, dtype=np.uint16)
    chunks = R.split_long(ids, 1025, 0)
    assert [len(c) for c, _ in chunks] == [1025, 1025, 452] and all(c[0] == 0 for c, _ in chunks[1:])
    assert np.array_equal(np.concatenate([chunks[0][0]] + [c[1:] for c, _ in chunks[1:]]), ids)
    mk = np.zeros(2500, dtype=np.uint8)
    mk[1100:1200] = 1
    chunks = R.split_long(ids, 1025, 0, mk)
    assert len(chunks) == 2 and chunks[1][1].sum() == 100                                # target-free chunk dropped


def test_convert():
    assert tinygsm_to_script("def simple_math_problem():\n    \"\"\"doc\"\"\"\n    a = 2\n    return a * 3\n") == \
        "a = 2\nprint(a * 3)"
    ans = "He has 48/2 = <<48/2=24>>24 apples.\nThen 24+1 = <<24+1=25>>25.\n#### 25"
    assert gsm8k_to_script(ans) == "# He has 48/2 = 24 apples.\nv1 = 48/2\n# Then 24+1 = 25.\nv2 = v1+1\nprint(v2)"
    assert gsm8k_cot(ans).endswith("Answer: 25")
    pre, script, post = omi1_to_parts("Let's code.\n<llm-code>\nx = 5\nx * 2\n</llm-code>\n<llm-code-output>\n10\n"
                                      "</llm-code-output>\nSo \\boxed{10}.")
    assert (pre, script, post) == ("Let's code.", "x = 5\nprint(x * 2)", "So 10.")
    assert tool_ok(TINY) and not tool_ok({**TINY, "answer": "27"}) and not tool_ok({"answer": "1"})
