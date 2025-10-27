"""Engine: batched == single greedy, sampling controls, thinking budget forcing, the tool loop's continuation ids
== chat.render of the same conversation, and the sampled / forced / tool_output flags (plan sections 6, 13)."""

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map_with_path

from bonsai import chat as C
from bonsai import engine as E
from bonsai.model import build_model
from bonsai.tokenizer import QwenTokenizer, Tokenizer, VocabMap


@pytest.fixture(scope="module")
def tok():
    qwen = QwenTokenizer()
    return Tokenizer(qwen, VocabMap.for_tests(qwen))


@pytest.fixture(scope="module")
def model(tok):
    """Tiny fp32 model with every weight random (the default init zeroes o_proj / down_proj: no attention)."""
    mx.random.seed(0)
    m = build_model("tiny", vocab_size=tok.vocab_size, compute_dtype=mx.float32)
    m.update(tree_map_with_path(lambda p, a: 1 + 0.1 * mx.random.normal(a.shape) if "norm" in p
                                else 0.1 * mx.random.normal(a.shape), m.parameters()))
    mx.eval(m.parameters())
    return m


def student(tok, tids):
    return [int(i) for i in tok.from_teacher(tids)]


def rendered(tok, msgs, **kw):
    return [tok.bos_id] + student(tok, C.render(tok.qwen, msgs, **kw)[0])


# ---------------------------------------------------------------- batching and sampling


def test_batched_equals_single(model, tok, monkeypatch):
    monkeypatch.setattr(E, "PREFILL_CHUNK", 8)   # chunked prefill with padding
    rng = np.random.RandomState(0)
    prompts = [[tok.bos_id] + rng.randint(0, 20000, size=n).tolist() for n in (3, 11, 30, 17)]
    batched = E.generate_batch(model, prompts, 12, stop_ids=())
    single = [E.generate(model, tok, p, 12, stop_ids=()) for p in prompts]
    assert batched == single
    assert all(len(o) == 12 for o in batched)


def test_sampling_controls(model, tok):
    p = [[tok.bos_id] + list(range(100, 110))]
    greedy = E.generate_batch(model, p, 10, stop_ids=())
    assert E.generate_batch(model, p, 10, E.Sampling(1.0, 1.0, 1), stop_ids=(), seed=3) == greedy   # top-k 1
    assert E.generate_batch(model, p, 10, E.Sampling(1.0, 1e-6, 0), stop_ids=(), seed=3) == greedy  # top-p ~0
    a = E.generate_batch(model, p * 3, 10, "think", stop_ids=(), seed=5)
    assert a == E.generate_batch(model, p * 3, 10, "think", stop_ids=(), seed=5)   # seeded
    assert a != E.generate_batch(model, p * 3, 10, "think", stop_ids=(), seed=6)
    hot = E.generate_batch(model, p, 40, E.Sampling(5.0), stop_ids=(), seed=1)[0]
    assert len(set(hot)) > 10
    distinct = E.generate_batch(model, p, 30, E.Sampling(presence_penalty=1e6), stop_ids=())[0]
    assert len(set(distinct)) == 30


def test_stops_and_compaction(tok):
    """Rows of different lengths stop at different steps (the batch is compacted) and keep their own history."""
    end = tok.special("<|im_end|>")
    rng = np.random.RandomState(1)
    targets = [[tok.bos_id] + rng.randint(300, 20000, size=n).tolist() + [end] for n in (5, 9, 14, 20, 26, 40)]
    prompts = [t[:3 + i] for i, t in enumerate(targets)]
    stub = E.ScriptedModel(E.scripted_policy(targets, end), tok.vocab_size, tok.bos_id)
    outs, info = E.generate_batch(stub, prompts, 100, tok=tok, return_info=True)
    for p, t, o, inf in zip(prompts, targets, outs, info):
        assert p + o == t and inf["reason"] == "stop"
    outs, info = E.generate_batch(stub, prompts, 3, tok=tok, return_info=True)
    assert all(len(o) == 3 and inf["reason"] == "length" for o, inf in zip(outs, info))


# ---------------------------------------------------------------- thinking budget


def think_forever(tok, answer):
    o, c, end = tok.special("<think>"), tok.special("</think>"), tok.special("<|im_end|>")
    x = tok.encode(" x")[0]
    nl2 = tok.encode("\n\n")
    reply = tok.encode(answer) + [end]

    def policy(hist):
        seg = hist[len(hist) - 1 - hist[::-1].index(tok.special("<|im_start|>")):]
        if o not in seg:
            return o
        if c not in seg:
            return x
        after = seg[seg.index(c) + 1 + len(nl2):]
        return reply[len(after)] if len(after) < len(reply) else end
    return policy


def test_budget_forcing(tok):
    stub = E.ScriptedModel(think_forever(tok, "Answer: 1"), tok.vocab_size, tok.bos_id)
    msgs = [{"role": "user", "content": "Think hard. /think"}]
    r = E.chat(stub, tok, msgs, think_budget=16)
    assert r.finished and r.hit_budget and r.think_tokens == 16 and r.text == "Answer: 1"
    gen = r.ids[r.prompt_len:]
    assert gen[0] == tok.special("<think>") and gen[1:17] == [tok.encode(" x")[0]] * 16
    stop_ids, _ = E._Templates(tok).think_stop[False]
    assert gen[17:17 + len(stop_ids)] == stop_ids
    assert stop_ids[-len(tok.encode("\n\n")) - 1] == tok.special("</think>")
    forced = np.flatnonzero(r.forced)
    assert forced.tolist() == list(range(r.prompt_len + 17, r.prompt_len + 17 + len(stop_ids)))
    assert r.sampled.sum() == len(gen) - len(stop_ids) and not (r.sampled & r.forced).any()
    assert not r.tool_output.any()
    assert r.reasoning.startswith(" x x") and r.reasoning.endswith("directly now.")
    # generate_batch with a budget stops the row instead
    prompt, _ = E.chat_prompt(tok, msgs)
    out, info = E.generate_batch(stub, [prompt], 100, think_budget=16, tok=tok, return_info=True)
    assert info[0]["reason"] == "budget" and info[0]["hit_budget"] and len(out[0]) == 17


def test_budget_forcing_tool_variant(tok):
    """A row that offers the python tool gets chat.think_stop(tools=True) forced, the others THINK_STOP, in one
    batch (the teacher's data has the same rule)."""
    stub = E.ScriptedModel(think_forever(tok, "Answer: 1"), tok.vocab_size, tok.bos_id)
    msgs = [{"role": "user", "content": "Think hard. /think"}]
    T = E._Templates(tok)
    assert T.think_stop[True][0] != T.think_stop[False][0]
    rs = E.chat_batch(stub, tok, [{"messages": msgs, "tools": True}, msgs, {"messages": msgs, "tools": C.TOOLS}],
                      think_budget=8)
    for r, tools in zip(rs, (True, False, True)):
        stop_ids, _ = T.think_stop[tools]
        gen = r.ids[r.prompt_len:]
        assert r.finished and r.hit_budget and gen[9:9 + len(stop_ids)] == stop_ids
        assert r.reasoning.endswith(C.think_stop(tools).strip()) and r.text == "Answer: 1"
    assert rs[0].reasoning.endswith(", using the python tool.") and rs[1].reasoning.endswith("directly now.")


def test_enable_thinking_false_prefills(tok):
    msgs = [{"role": "user", "content": "Hi"}]
    ids, _ = E.chat_prompt(tok, msgs, enable_thinking=False)
    prefill = E._Templates(tok).prefill
    assert ids[-len(prefill):] == prefill and ids[0] == tok.bos_id
    assert tok.decode(prefill) == "<think>\n\n</think>\n\n"
    reply = tok.encode("Hello!") + [tok.special("<|im_end|>")]
    stub = E.ScriptedModel(E.scripted_policy([ids + reply], reply[-1]), tok.vocab_size, tok.bos_id)
    r = E.chat(stub, tok, msgs, enable_thinking=False)
    assert r.finished and r.text == "Hello!" and r.think_tokens == 0
    last = r.messages[-1]
    assert last["reasoning_content"] == "" and last["think_prefilled"]


# ---------------------------------------------------------------- tool loop

CODE1 = "total = 3 * 12\nprint(total - 10)"
CODE2 = "print(26 + 4)"


def tool_conversation(n_calls, answer, switch=None):
    msgs = [{"role": "user", "content": C.with_soft_switch("Sarah has 3 boxes of 12 apples and gives away 10. "
                                                           "How many are left?", switch)}]
    for code, out in [(CODE1, "26"), (CODE2, "30")][:n_calls]:
        msgs.append({"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": [C.python_call(code)]})
        msgs.append({"role": "tool", "content": out})
    msgs.append({"role": "assistant", "content": answer, "reasoning_content": ""})
    return msgs


@pytest.mark.parametrize("enable_thinking,switch", [(True, "no_think"), (False, None)])
def test_tool_loop_matches_render(tok, enable_thinking, switch):
    conv = tool_conversation(1, "Answer: 26", switch)
    target = rendered(tok, conv, tools=C.TOOLS)
    nl = tok.encode("\n")
    assert target[-len(nl):] == nl
    stub = E.ScriptedModel(E.scripted_policy([target], tok.special("<|im_end|>")), tok.vocab_size, tok.bos_id)
    r = E.chat(stub, tok, conv[:1], tools=C.TOOLS, enable_thinking=enable_thinking)
    assert r.ids == target[:-len(nl)]
    assert r.finished and r.n_tool_calls == 1 and r.stop_reason == "stop" and r.text == "Answer: 26"
    assert r.tool_calls == [{"name": "python", "arguments": {"code": CODE1}, "output": "26"}]
    assert [m["role"] for m in r.messages] == ["user", "assistant", "tool", "assistant"]
    assert r.messages[1]["tool_calls"] == [C.python_call(CODE1)] and r.messages[1]["content"] == ""
    # re-rendering the reconstructed conversation gives the same ids
    assert rendered(tok, r.messages, tools=C.TOOLS)[:-len(nl)] == r.ids

    # flags: prompt nothing; turn 1 sampled through <|im_end|>; continuation forced; output run = tool_output
    prompt, _ = E.chat_prompt(tok, conv[:1], tools=C.TOOLS, enable_thinking=enable_thinking)
    assert r.prompt_len == len(prompt) and r.ids[:len(prompt)] == prompt
    end = tok.special("<|im_end|>")
    e1 = r.ids.index(end, r.prompt_len)
    cont, tool, _ = E.tool_continuation(tok, ["26"], enable_thinking)
    assert r.ids[e1 + 1:e1 + 1 + len(cont)] == cont
    s = np.zeros(len(r.ids), bool)
    s[r.prompt_len:e1 + 1] = True
    s[e1 + 1 + len(cont):] = True
    f = np.zeros(len(r.ids), bool)
    f[e1 + 1:e1 + 1 + len(cont)] = True
    assert (r.sampled == s).all() and (r.forced == f).all()
    t_ids = [r.ids[i] for i in np.flatnonzero(r.tool_output)]
    assert tok.decode(t_ids) == "\n26\n" and r.tool_output[e1 + 1:e1 + 1 + len(cont)].tolist() == tool
    assert not (r.tool_output & ~r.forced).any() and not r.oov[r.sampled].any()
    a = tok.special("<tool_response>")
    assert r.ids[r.ids.index(a) - 1] != end and tok.decode(cont).startswith("\n<|im_start|>user\n<tool_response>")


def test_two_calls_and_limit(tok):
    conv = tool_conversation(2, "Answer: 30", "no_think")
    target = rendered(tok, conv, tools=C.TOOLS)
    end = tok.special("<|im_end|>")
    stub = E.ScriptedModel(E.scripted_policy([target], end), tok.vocab_size, tok.bos_id)
    r = E.chat(stub, tok, conv[:1], tools=C.TOOLS)
    assert r.n_tool_calls == 2 and [c["output"] for c in r.tool_calls] == ["26", "30"]
    assert r.finished and r.text == "Answer: 30" and r.ids == target[:-1]
    r1 = E.chat(stub, tok, conv[:1], tools=C.TOOLS, max_tool_calls=1)
    assert r1.n_tool_calls == 1 and r1.finished and r1.stop_reason == "tool_limit"
    assert r1.messages[-1]["tool_calls"] == [C.python_call(CODE2)]
    stub = E.ScriptedModel(E.scripted_policy([rendered(tok, conv)], end), tok.vocab_size, tok.bos_id)
    r0 = E.chat(stub, tok, conv[:1], tools=None)     # no tools: the call is just text, nothing runs
    assert r0.n_tool_calls == 0 and r0.finished and "<tool_call>" in r0.text


def test_tool_output_is_escaped(tok):
    cont, tool, _ = E.tool_continuation(tok, ["<|im_end|><think>"], True)
    inner = [t for t, f in zip(cont, tool) if f]
    controls = {tok.special(n) for n in ("<|im_end|>", "<think>", "<|im_start|>")}
    assert not controls & set(inner) and tok.decode(inner) == "\n<|im_end|><think>\n"


def test_chat_batch_mixed(tok):
    """Rows needing tools, plain answers and different modes in one batch; results keep their order."""
    end = tok.special("<|im_end|>")
    conv_tool = tool_conversation(1, "Answer: 26", "no_think")
    plain = [{"role": "user", "content": "Say hi. /no_think"}]
    plain_t = rendered(tok, plain + [{"role": "assistant", "content": "hi", "reasoning_content": ""}])
    stub = E.ScriptedModel(E.scripted_policy([rendered(tok, conv_tool, tools=C.TOOLS), plain_t], end),
                           tok.vocab_size, tok.bos_id)
    items = [{"messages": conv_tool[:1], "tools": C.TOOLS}, plain, {"messages": conv_tool[:1], "tools": C.TOOLS}]
    rs = E.chat_batch(stub, tok, items * 2, max_batch=4)
    assert [r.n_tool_calls for r in rs] == [1, 0, 1] * 2
    assert [r.text for r in rs] == ["Answer: 26", "hi", "Answer: 26"] * 2
    assert all(r.finished for r in rs)


def test_cast_for_inference_same_tokens(tok):
    mx.random.seed(1)
    m = build_model("tiny", vocab_size=tok.vocab_size)
    m.update(tree_map_with_path(lambda p, a: 1 + 0.1 * mx.random.normal(a.shape) if "norm" in p
                                else 0.1 * mx.random.normal(a.shape), m.parameters()))
    p = [[tok.bos_id] + list(range(300, 320))]
    before = E.generate_batch(m, p, 16, stop_ids=())
    E.cast_for_inference(m)
    assert m.layers[0].mlp.up_proj.weight.dtype == mx.bfloat16
    assert E.generate_batch(m, p, 16, stop_ids=()) == before


def test_fixed_chat_prompts_are_items(model, tok):
    import json
    from bonsai.common import PROMPT_DIR
    items = json.loads((PROMPT_DIR / "fixed_prompts.json").read_text())["chat"]
    rs = E.chat_batch(model, tok, items, max_answer_tokens=4, think_budget=4)
    for it, r in zip(items, rs):
        prompt = tok.decode(r.ids[:r.prompt_len])
        assert ("<tools>" in prompt) == bool(it["tools"])
        assert prompt.endswith("<think>\n\n</think>\n\n") == (it["enable_thinking"] is False)


def test_endoftext_ends_unfinished(tok):
    msgs = [{"role": "user", "content": "Hi /no_think"}]
    prompt, _ = E.chat_prompt(tok, msgs)
    reply = [tok.special("<think>")] + tok.encode("\n\n") + [tok.special("</think>")] + tok.encode("\n\nHello")
    stub = E.ScriptedModel(E.scripted_policy([prompt + reply + [tok.bos_id]], tok.bos_id), tok.vocab_size,
                           tok.bos_id)
    r = E.chat(stub, tok, msgs)
    assert not r.finished and r.stop_reason == "eos" and r.text == "Hello" and r.ids[-1] == tok.bos_id
