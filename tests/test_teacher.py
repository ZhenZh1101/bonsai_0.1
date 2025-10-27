"""Teacher scoring over S, the sampler, batched decoding (left padding, refill), thinking-budget forcing, tool
turns and the off-policy filters (bonsai/teacher.py). Tiny random mlx-lm Qwen3 models and scripted stubs only."""

import mlx.core as mx
import numpy as np
import pytest

from bonsai import chat
from bonsai import teacher as T
from bonsai.common import TEACHER_DIR
from bonsai.tokenizer import QwenTokenizer, VocabMap


@pytest.fixture(scope="module")
def qwen():
    return QwenTokenizer()


@pytest.fixture(scope="module")
def vmap(qwen):
    return VocabMap.for_tests(qwen, n=3000)


def _tiny(qwen, tied):
    """A 2-layer fp32 mlx-lm Qwen3 with random weights large enough that greedy outputs depend on the context."""
    from mlx_lm.models import qwen3
    args = qwen3.ModelArgs(model_type="qwen3", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
                           num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=151936, num_key_value_heads=2,
                           max_position_embeddings=2048, head_dim=8, tie_word_embeddings=tied, rope_theta=1e6)
    mx.random.seed(0)
    m = qwen3.Model(args)
    m.model.embed_tokens.weight = mx.random.normal((151936, 32))
    if not tied:
        m.lm_head.weight = mx.random.normal((151936, 32))
    for layer in m.model.layers:
        layer.self_attn.o_proj.weight = mx.random.normal(layer.self_attn.o_proj.weight.shape) * 0.5
    mx.eval(m.parameters())
    return T.Teacher("tiny", model=m, qwen=qwen)


@pytest.fixture(scope="module")
def tiny(qwen):
    return _tiny(qwen, tied=False)


@pytest.fixture(scope="module")
def tiny_tied(qwen):
    return _tiny(qwen, tied=True)


# ------------------------------------------------------------------------------------------- scoring


def test_subset_log_softmax_matches_naive():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(2, 5, 300)).astype(np.float32) * 3
    s2t = rng.choice(300, size=40, replace=False)
    got = np.array(T.subset_log_softmax(mx.array(logits), s2t))
    sub = logits[..., s2t].astype(np.float64)
    want = sub - np.log(np.exp(sub).sum(-1, keepdims=True))
    assert np.allclose(got, want, atol=1e-5)
    assert np.allclose(np.exp(got).sum(-1), 1.0, atol=1e-5)


@pytest.mark.parametrize("tied", [True, False])
def test_subset_logprobs_matches_full_vocab(tiny, tiny_tied, vmap, tied):
    t = tiny_tied if tied else tiny
    rng = np.random.default_rng(1)
    ids = rng.integers(0, 3000, size=(3, 12)).astype(np.int32)
    naive = T.subset_log_softmax(t.logits(ids), vmap.s2t)
    got = t.subset_logprobs(ids, vmap, chunk=2)
    assert got.shape == (3, 12, vmap.N) and got.dtype == mx.float32
    assert mx.abs(got - naive).max().item() < 1e-3          # fp32 tiny model: only the matmul order differs
    assert mx.abs(mx.logsumexp(got, axis=-1)).max().item() < 1e-4
    # right padding never changes the valid prefix
    padded = ids.copy()
    padded[1, 7:] = 0
    lp = t.subset_logprobs(padded, vmap)
    assert mx.abs(lp[1, :7] - got[1, :7]).max().item() < 1e-4
    parts = list(t.iter_subset_logprobs(ids, vmap, chunk=2))
    assert [a for a, _ in parts] == [0, 2] and parts[1][1].shape[0] == 1


# ------------------------------------------------------------------------------------------- sampler


def test_sampler_greedy_topk_topp():
    logits = mx.array([[0.0, 3.0, 1.0, 2.0]] * 4)
    temp = mx.array([0.0, 1.0, 1.0, 0.7])
    top_p = mx.array([1.0, 1.0, 1e-6, 1.0])
    out = T.sample_next(logits, temp, top_p, top_k=1).tolist()
    assert out == [1, 1, 1, 1]                                  # temp 0 / top-k 1 -> argmax
    out = T.sample_next(logits, mx.array([1.0] * 4), mx.array([1e-6] * 4), top_k=0).tolist()
    assert out == [1, 1, 1, 1]                                  # top-p tiny -> argmax, also without top-k
    mx.random.seed(3)
    big = mx.broadcast_to(mx.array([[0.0, 3.0, 1.0, 2.9]]), (2000, 4))
    s = np.array(T.sample_next(big, mx.ones(2000), mx.ones(2000), top_k=2))
    assert set(np.unique(s)) == {1, 3}                          # top-k 2 keeps tokens 1 and 3 only
    # top-p 0.6 after temperature: p(1) = 0.52 at T=1 -> tokens 1 and 3 needed
    s = np.array(T.sample_next(big, mx.ones(2000), mx.full((2000,), 0.6), top_k=0))
    assert set(np.unique(s)) == {1, 3}
    # temperature before top-p: at T=0.1 token 1 alone has > 0.6 of the mass
    s = np.array(T.sample_next(big, mx.full((2000,), 0.1), mx.full((2000,), 0.6), top_k=0))
    assert set(np.unique(s)) == {1}


def test_presence_penalty_vectorised():
    logits = mx.array([[0.0, 3.0, 2.0, 0.0], [0.0, 3.0, 2.0, 0.0]])
    seen = mx.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    out = T.sample_next(logits, mx.zeros(2), mx.ones(2), 0, mx.array([1.5, 1.5]), seen).tolist()
    assert out == [2, 1]


# ---------------------------------------------------------------------------------- batched decoding


def test_batched_greedy_matches_single(tiny, qwen):
    prompts = [qwen.encode(t) for t in ("Hello there, how are", "A", "The quick brown fox jumps over the lazy",
                                        "One two", "x y z w")]
    single = [tiny.generate_batch([p], T.GREEDY, max_tokens=6, stop=())[0] for p in prompts]
    assert len({tuple(s) for s in single}) == 5 and all(len(set(s)) > 3 for s in single)   # not degenerate
    # batch 2 with prefill groups of 1: left padding, BatchKVCache.extend on refill and filter on finish
    got = tiny.generate_batch(prompts, T.GREEDY, max_tokens=6, stop=(), batch=2, prefill_batch=1)
    assert got == single
    got = tiny.generate_batch(prompts, T.GREEDY, max_tokens=6, stop=(), batch=8, prefill_batch=3)
    assert got == single
    assert tiny.stats["gen_tokens"] == 5 * 6 - 5               # the first token of each row comes from its prefill


def test_stop_tokens_end_rows(tiny, qwen):
    p = qwen.encode("Hello there")
    full = tiny.generate_batch([p], T.GREEDY, max_tokens=6, stop=())[0]
    cut = tiny.generate_batch([p, p], T.GREEDY, max_tokens=6, stop={full[2]})
    assert cut[0] == full[:full.index(full[2]) + 1] == cut[1]


class _StubCache:
    def __init__(self, hist):
        self.hist = hist

    def filter(self, keep):
        self.hist = [self.hist[i] for i in keep]

    def extend(self, other):
        self.hist = self.hist + other.hist


class StubTeacher(T.Teacher):
    """Scripted 'model': policy(history) -> next token id, through the real decode loop."""

    def __init__(self, qwen, policy):
        self.name, self.qwen, self.policy = "stub", qwen, policy
        self.vocab_size, self.n_layers, self.stats = 151936, 1, {}

    def _logits(self, hists):
        x = np.full((len(hists), self.vocab_size), -1e4, dtype=np.float32)
        for i, h in enumerate(hists):
            x[i, self.policy(h)] = 0.0
        return mx.array(x)

    def _prefill(self, seqs, step=1024):
        c = _StubCache([list(s) for s in seqs])
        return [c], self._logits(c.hist)

    def _step(self, cache, tokens):
        c = cache[0]
        for h, t in zip(c.hist, tokens):
            h.append(t)
        return self._logits(c.hist)


def _after(h, marker):
    """Tokens of h after the last occurrence of `marker` (a token id), or None."""
    return None if marker not in h else len(h) - 1 - h[::-1].index(marker)


def test_think_budget_forcing(qwen):
    a = qwen.encode(" a")[0]
    answer = qwen.encode("Answer: 9")

    def policy(h):
        k = _after(h, T.THINK_CLOSE)
        if k is None:
            return a                                            # never closes the think block itself
        n = len(h) - k - 1 - 2                                  # after "</think>" "\n\n"
        return answer[n] if 0 <= n < len(answer) else (T.IM_END if n >= 0 else qwen.encode("\n\n")[0])

    t = StubTeacher(qwen, policy)
    job = T.ChatJob([{"role": "user", "content": "What is 4+5?"}], "think", sampling=T.GREEDY)
    out = t.chat_batch([job, job], budget=16)[0]
    assert out.finish == "stop" and len(out.turns) == 1
    turn = out.turns[0]
    assert turn["hit_budget"] and turn["think_tokens"] == 16
    assert turn["think"].endswith(chat.THINK_STOP) and turn["text"].strip() == "Answer: 9"
    gen = out.ids[out.prompt_len:]
    stop = T.template_ids(qwen, (chat.THINK_STOP,), "\n</think>\n\n")
    assert gen[:2] == T.template_ids(qwen, T.THINK_PREFILL)
    assert gen[2:18] == [a] * 16 and gen[18:18 + len(stop)] == stop
    assert out.kinds[18:18 + len(stop)] == [2] * len(stop) and out.kinds[2:18] == [1] * 16
    # under the budget the teacher's own </think> ends thinking, nothing is forced
    short = StubTeacher(qwen, lambda h: T.THINK_CLOSE if len(h) - out.prompt_len >= 5 and T.THINK_CLOSE not in h
                        else policy(h))
    o2 = short.chat_batch([job], budget=16)[0]
    assert not o2.turns[0]["hit_budget"] and o2.turns[0]["think_tokens"] == 3 and 2 not in o2.kinds[2:]
    # a job that offers the python tool gets the tool variant (else the teacher answers without calling it)
    tj = T.ChatJob([{"role": "user", "content": "What is 4+5?"}], "think", tools=True, sampling=T.GREEDY)
    o3 = t.chat_batch([tj, job], budget=16)
    tstop = T.template_ids(qwen, (chat.think_stop(tools=True),), "\n</think>\n\n")
    g3 = o3[0].ids[o3[0].prompt_len:]
    assert chat.think_stop(tools=True) == chat.THINK_STOP_TOOL != chat.think_stop() == chat.THINK_STOP
    assert g3[18:18 + len(tstop)] == tstop and o3[0].kinds[18:18 + len(tstop)] == [2] * len(tstop)
    assert o3[0].turns[0]["think"].endswith(chat.THINK_STOP_TOOL) and o3[0].turns[0]["hit_budget"]
    assert o3[1].ids == out.ids and o3[1].turns[0]["think"].endswith(chat.THINK_STOP)    # non-tool jobs unchanged


def test_tool_turn_renders_like_the_template(qwen):
    call = T.template_ids(qwen, "<tool_call>", ('\n{"name": "python", "arguments": {"code": "print(6*7)"}}\n',),
                          "</tool_call>")
    answer = qwen.encode("Answer: 42")

    def policy(h):
        k = _after(h, qwen.control_ids["</tool_response>"])
        base = _after(h, T.THINK_CLOSE)
        n = len(h) - base - 2                                   # tokens after "</think>\n\n"
        script = call if k is None else answer
        return script[n] if n < len(script) else T.IM_END

    t = StubTeacher(qwen, policy)
    calls = []
    job = T.ChatJob([{"role": "user", "content": "What is 6 times 7?"}], "no_think", tools=True,
                    sampling=T.GREEDY, key="k")
    out = t.chat_batch([job], run_tool=lambda c: calls.append(c) or "42")[0]
    assert out.finish == "stop" and out.n_tool_calls == 1 and len(out.turns) == 2
    assert calls == [{"name": "python", "arguments": {"code": "print(6*7)"}}]
    msgs = T.generated_messages(out, "no_think", "soft")
    assert [m["role"] for m in msgs] == ["assistant", "tool", "assistant"]
    assert msgs[0]["tool_calls"] == calls and msgs[0]["content"] == "" and msgs[2]["content"] == "Answer: 42"
    ids, mask = chat.render(qwen, job.messages + msgs, tools=chat.TOOLS)
    assert ids[:len(out.ids)] == out.ids and ids[len(out.ids):] == qwen.encode("\n")
    # supervised: both assistant turns incl. their empty think blocks, never the tool output
    tool_out = qwen.decode([i for i, m in zip(ids, mask) if m])
    assert "42\n</tool_response>" not in tool_out and tool_out.count("<think>") == 2
    # a third call is refused
    loop = StubTeacher(qwen, lambda h: call[min(len(h) - _after(h, T.THINK_CLOSE) - 2, len(call))]
                       if len(h) - _after(h, T.THINK_CLOSE) - 2 < len(call) else T.IM_END)
    o3 = loop.chat_batch([job], run_tool=lambda c: "1")[0]
    assert o3.finish == "too_many_calls" and o3.n_tool_calls == 2


def test_answer_cap(qwen):
    a = qwen.encode(" a")[0]
    t = StubTeacher(qwen, lambda h: a)
    out = t.chat_batch([T.ChatJob([{"role": "user", "content": "hi"}], "no_think", sampling=T.GREEDY)],
                       max_answer=20)[0]
    assert out.finish == "answer_too_long" and out.turns[0]["answer_tokens"] == 21


def test_template_ids_equal_render(qwen):
    msgs = [{"role": "user", "content": "Hi <think> there"}]
    ids, _ = chat.render(qwen, msgs, add_generation_prompt=True)
    assert ids == T.template_ids(qwen, "<|im_start|>user\n", ("Hi <think> there",), "<|im_end|>\n<|im_start|>"
                                 "assistant\n")


@pytest.mark.skipif(not (TEACHER_DIR / "Qwen3-0.6B" / "tokenizer.json").exists(), reason="teachers not downloaded")
def test_teacher_tokenizer_hashes():
    for name in T.TEACHERS:
        if (TEACHER_DIR / name).exists():
            T.check_tokenizer(TEACHER_DIR / name)


# ------------------------------------------------------------------------------------------- filters


def _sample(content, reasoning="", mode="think", kind="math", answer="9", tools=False, gen=None, extra=()):
    msgs = [{"role": "user", "content": "Mia had 16 slices and ate 7. How many are left?"}, *extra,
            {"role": "assistant", "content": content, "reasoning_content": reasoning}]
    return {"messages": msgs, "mode": mode, "kind": kind, "answer": answer, "tools": tools,
            "gen": gen or {"finish": "stop", "hit_budget": [False], "answer_tokens": [10]}}


def _probs(s):
    return T.sample_problems(s, contaminated=lambda text: "LEAKED" in text)


def test_filters_keep_good_samples():
    assert _probs(_sample("16 - 7 = 9 slices remain.\nAnswer: 9", "16 - 7 = 9. So 9 are left.")) == []
    assert _probs(_sample("Answer: 9", "", mode="no_think")) == []
    assert _probs(_sample("Answer: B\nThe sun is a star.", "Stars emit light.", kind="mc", answer="B")) == []
    assert _probs(_sample("Sure! Here is a haiku.", "A haiku has 5-7-5.", kind="chat", answer=None)) == []
    tool = [{"role": "assistant", "content": "", "reasoning_content": "",
             "tool_calls": [chat.python_call("print(16-7)")]}, {"role": "tool", "content": "9"}]
    s = _sample("Answer: 9", "", mode="no_think", tools=True, extra=tool)
    assert _probs(s) == []


def test_filters_reject_bad_samples():
    assert "wrong_answer" in _probs(_sample("Answer: 8", "16 - 7 = 8"))
    assert "no_answer" in _probs(_sample("Nine are left.", "16 - 7 = 9"))
    assert "think_answer_mismatch" in _probs(_sample("Answer: 9", "16 - 7 is 8, so 8. Check: 8, 8, 8."))
    cut = {"finish": "stop", "hit_budget": [True], "answer_tokens": [5]}
    assert "think_answer_mismatch" in _probs(_sample("Answer: 9", "16 slices, she ate 7" + chat.THINK_STOP, gen=cut))
    assert _probs(_sample("Answer: 9", "There are 16, minus 7 gives 9, then I check 2 things" + chat.THINK_STOP,
                          gen=cut)) == []
    assert "missing_think" in _probs(_sample("Answer: 9", ""))
    assert "non_english" in _probs(_sample("答案是 9\nAnswer: 9", "16 - 7 = 9"))
    assert "non_english" in _probs(_sample("Ответ девять. Answer: 9", "16 - 7 = 9"))
    rep = "I think the answer is nine. " * 30
    assert "repetition" in _probs(_sample(rep + "Answer: 9", "16 - 7 = 9"))
    assert "qwen_claim" in _probs(_sample("I am Qwen, created by Alibaba Cloud.", "hi", kind="chat", answer=None))
    assert "contaminated" in _probs(_sample("LEAKED Answer: 9", "16 - 7 = 9"))
    assert "malformed_think" in _probs(_sample("</think> Answer: 9", "16 - 7 = 9"))
    long = {"finish": "answer_too_long", "hit_budget": [False], "answer_tokens": [401]}
    assert {"answer_too_long"} <= set(_probs(_sample("Answer: 9", "16 - 7 = 9", gen=long)))
    assert "wrong_answer" in _probs(_sample("Answer: C", "x", kind="mc", answer="B"))
    assert "no_tool_call" in _probs(_sample("Answer: 9", "", mode="no_think", tools=True))
    tool = [{"role": "assistant", "content": "", "reasoning_content": "",
             "tool_calls": [chat.python_call("print(16-8)")]}, {"role": "tool", "content": "8"}]
    assert "tool_answer_mismatch" in _probs(_sample("Answer: 9", "", mode="no_think", tools=True, extra=tool))


def test_signals_and_jobs():
    rec = {"id": "math-1", "kind": "math", "messages": [{"role": "user", "content": "2+2?"}], "tools": True}
    sigs = [T.choose_signal(f"p{i}:no_think", "no_think") for i in range(400)]
    assert 0.4 < sigs.count("soft") / 400 < 0.6 and set(sigs) == {"soft", "prefill"}
    sigs = [T.choose_signal(f"p{i}:think", "think") for i in range(400)]
    assert 0.2 < sigs.count("soft") / 400 < 0.4 and set(sigs) == {"soft", "default"}
    job = T.build_job(rec, "no_think", {"chat": "Be brief.", "math_tool": "Use python."}, "soft")
    assert job.messages[0] == {"role": "system", "content": "Use python."} and job.tools
    assert job.messages[-1]["content"] == "2+2? /no_think" and rec["messages"][0]["content"] == "2+2?"
    assert job.key == "math-1:no_think"


def test_blocked_topk_equals_argpartition():
    for V in (151936, 1001, 64):
        z = mx.random.normal((7, V))
        got = mx.sort(T.topk_indices(z, 20), axis=-1)
        want = mx.sort(mx.argpartition(-z, kth=19, axis=-1)[:, :20], axis=-1)
        assert mx.array_equal(got, want).item()


def test_open_think_after_tool_is_salvaged(qwen):
    """Qwen3-1.7B after a tool result in thinking mode: "<think>\\n...Answer: 42<|im_end|>" without </think>."""
    call = T.template_ids(qwen, "<tool_call>", ('\n{"name": "python", "arguments": {"code": "print(6*7)"}}\n',),
                          "</tool_call>")
    tail = T.template_ids(qwen, "<think>", ("\nSo 6 times 7 is 42.\n\nAnswer: 42",), "<|im_end|>")

    first = qwen.encode("I will use the tool.\n") + [T.THINK_CLOSE] + qwen.encode("\n\n") + call + [T.IM_END]

    def policy(h):
        k = _after(h, qwen.control_ids["</tool_response>"])
        seq, n = (first, len(h) - start[0]) if k is None else \
            (tail, len(h) - k - 1 - len(T.template_ids(qwen, "<|im_end|>\n<|im_start|>assistant\n")))
        return seq[n] if 0 <= n < len(seq) else T.IM_END       # (ignored while forced tokens are fed)

    job = T.ChatJob([{"role": "user", "content": "6*7?"}], "think", tools=True, sampling=T.GREEDY)
    start = [len(chat.render(qwen, job.messages, tools=chat.TOOLS, add_generation_prompt=True)[0]) + 2]
    out = StubTeacher(qwen, policy).chat_batch([job], run_tool=lambda c: "42")[0]
    assert out.finish == "stop" and out.turns[-1]["salvaged"]
    msgs = T.generated_messages(out, "think", "default")
    assert msgs[-1]["content"] == "Answer: 42" and msgs[-1]["reasoning_content"] == "So 6 times 7 is 42."
    s = {"messages": [job.messages[0]] + msgs, "mode": "think", "kind": "math", "answer": "42", "tools": True,
         "gen": {"finish": out.finish, "hit_budget": [False, False], "answer_tokens": [20, 5]}}
    assert msgs[0]["reasoning_content"] == "I will use the tool." and msgs[0]["tool_calls"]
    assert T.sample_problems(s, contaminated=lambda t: False) == []


def _math_stub(qwen):
    """A stub teacher that answers 'Answer: 9' (its thinking never mentions 9) + 5 math prompts + a config."""
    think = qwen.encode("I count: 16 slices, 7 eaten.\n") + [T.THINK_CLOSE] + qwen.encode("\n\n16 - 7 = 9.\nAnswer: 9")
    plain = qwen.encode("16 - 7 = 9.\nAnswer: 9")

    def policy(h):
        k = _after(h, T.THINK_OPEN)
        if k is None:                                           # the warm-up prompt
            return T.IM_END
        no_think = len(h) > k + 2 and h[k + 2] == T.THINK_CLOSE
        seq, n = (plain, len(h) - k - 4) if no_think else (think, len(h) - k - 2)
        return seq[n] if 0 <= n < len(seq) else T.IM_END

    prompts = [{"id": f"m{i}", "kind": "math", "answer": "9", "tools": False, "source": "t",
                "messages": [{"role": "user", "content": f"Mia had 16 slices and ate 7 ({i}). How many are left?"}]}
               for i in range(5)]
    cfg = {"generation": {"think_soft_frac": 0.3, "no_think_soft_frac": 0.5},
           "system_prompts": {"math": "Answer with 'Answer: N'."}}
    return StubTeacher(qwen, policy), prompts, cfg


def test_generate_offpolicy_resume_and_refilter(qwen, tmp_path):
    t, prompts, cfg = _math_stub(qwen)
    run = T.generate_offpolicy(t, prompts, tmp_path, cfg=cfg, batch=4, chunk=3, contaminated=lambda x: False)
    st = T.summarize(tmp_path)
    assert run["jobs"] == 10 and st["kept"] == 5 and st["rejected"] == 5      # thinking: 9 is not in the thinking
    assert st["first_reason"] == {"think_answer_mismatch": 5}
    kept = list(T.iter_jsonl(tmp_path / "kept_s0of1.jsonl"))
    assert all(s["mode"] == "no_think" and s["messages"][-1]["content"].endswith("Answer: 9") for s in kept)
    assert all(m["role"] != "system" for s in kept for m in s["messages"])          # teacher prompt not stored
    assert {s["signal"] for s in kept} <= {"soft", "prefill"}
    again = T.generate_offpolicy(t, prompts, tmp_path, cfg=cfg, batch=4, contaminated=lambda x: False)
    assert again["jobs"] == 0                                                          # resumable by sample id
    retry = T.generate_offpolicy(t, prompts, tmp_path, cfg=cfg, batch=4, contaminated=lambda x: False,
                                 retry=("think_answer_mismatch",))
    assert retry["jobs"] == 5 and T.summarize(tmp_path)["rejected"] == 5                # regenerated, not doubled
    st = T.refilter(tmp_path, {"filters": {"think_answer": "off"}}, contaminated=lambda x: False)
    assert st["kept"] == 10 and st["rejected"] == 0
    shard = T.generate_offpolicy(t, prompts, tmp_path / "sh", cfg=cfg, shard=(1, 2), contaminated=lambda x: False)
    assert shard["jobs"] == 2 * sum(T.shard_of(p["id"], 2) == 1 for p in prompts)


def test_generate_offpolicy_resumes_after_a_truncated_line(qwen, tmp_path):
    """A hard kill mid-write leaves a partial last line: reading skips it, the next run cuts it off and
    generates that sample again (appending after it would corrupt the next record too)."""
    t, prompts, cfg = _math_stub(qwen)
    T.generate_offpolicy(t, prompts, tmp_path, cfg=cfg, batch=4, contaminated=lambda x: False)
    kp = tmp_path / "kept_s0of1.jsonl"
    lines = kp.read_bytes().splitlines(keepends=True)
    lost = T.json.loads(lines[-1])["id"]
    kp.write_bytes(b"".join(lines[:-1]) + lines[-1][:40] + "é".encode()[:1])    # cut inside a UTF-8 character
    assert lost not in T.done_ids(tmp_path) and len(T.done_ids(tmp_path)) == 9
    assert T.summarize(tmp_path)["kept"] == 4
    again = T.generate_offpolicy(t, prompts, tmp_path, cfg=cfg, batch=4, contaminated=lambda x: False)
    assert again["jobs"] == 1 and kp.read_bytes().endswith(b"\n")
    recs = list(T.iter_jsonl(kp))                                               # strict reader: the file is clean
    assert len(recs) == 5 and len({r["id"] for r in recs}) == 5 and lost in {r["id"] for r in recs}
    assert T.repair_tail(kp) == 0 and T.summarize(tmp_path)["kept"] == 5


def test_prefix_cache_matches_plain_prefill(tiny):
    sys_a = {"role": "system", "content": "You are a helpful assistant. Always reply in English."}
    jobs = [T.ChatJob([sys_a, {"role": "user", "content": u}], mode, tools=tools, sampling=T.GREEDY)
            for u in ("Hi!", "What is two plus two, my friend?", "Name a color.")
            for mode, tools in (("think", False), ("no_think", False), ("no_think", True))]
    jobs.append(T.ChatJob([{"role": "user", "content": "No system turn here."}], "no_think", sampling=T.GREEDY))
    kw = dict(budget=4, max_answer=5, batch=4, prefill_batch=3, run_tool=lambda c: "0")
    rows = [T._ChatRow(tiny.qwen, j, 4, 5, 2, 2048, None) for j in jobs]
    assert [r.prefix_len > 0 for r in rows] == [True] * 9 + [False]
    a = tiny.chat_batch(jobs, prefix_cache=True, **kw)
    b = tiny.chat_batch(jobs, prefix_cache=False, **kw)
    assert [o.ids for o in a] == [o.ids for o in b]
    assert len({tuple(o.ids[o.prompt_len:]) for o in a}) > 3          # the outputs are not all the same


def test_prompt_records():
    from scripts.build_prompts import is_identity, make_prompt
    conv = {"messages": [{"role": "user", "content": "Why is the sky blue?"}, {"role": "assistant", "content": "A"},
                         {"role": "user", "content": "And at sunset?"}, {"role": "assistant", "content": "B"}]}
    msgs, ans = make_prompt({"pool": "sft_chat", "explain": True, "first_turn_only": True}, conv, 1)
    assert msgs == conv["messages"][:1] and ans is None
    assert make_prompt({"pool": "sft_chat", "explain": False}, conv, 1) == (None, "no_match")
    multi = [make_prompt({"pool": "sft_chat", "multi_turn_frac": 1.0}, conv, i)[0] for i in range(5)]
    assert all(len(m) == 3 and m[-1]["content"] == "And at sunset?" for m in multi)
    mc = {"question": "Which is a star?", "choices": ["Moon", "Sun", "Mars"], "answer": 1, "source": "arc-easy"}
    msgs, ans = make_prompt({"pool": "mc", "match": {"source": ["arc-easy"]}}, mc, 3)
    assert msgs[0]["content"] == "Question: Which is a star?\nA. Moon\nB. Sun\nC. Mars" and ans == "B"
    assert make_prompt({"pool": "mc", "match": {"source": ["arc-challenge"]}}, mc, 3) == (None, "no_match")
    spec = {"pool": "omi2_cot", "integer_answer": True, "no_latex": True, "max_ref_chars": 50}
    assert make_prompt(spec, {"question": "2+2?", "cot": "2+2=4\nAnswer: 4"}, 0) == \
        ([{"role": "user", "content": "2+2?"}], "4")
    assert make_prompt(spec, {"question": "$x$?", "cot": "Answer: 4"}, 0) == (None, "latex")
    assert make_prompt(spec, {"question": "1/3?", "cot": "Answer: 0.333"}, 0) == (None, "non_integer_answer")
    assert make_prompt(spec, {"question": "q?", "cot": "x" * 60 + "Answer: 4"}, 0) == (None, "long_reference")
    assert make_prompt({"pool": "tinygsm"}, {"question": "q?", "answer": "2.5"}, 0)[1] == "2.5"
    assert is_identity("Who are you?") and not is_identity("Write a haiku about the sea.")


def test_prompt_pool_overlap_with_s1_s2(tmp_path, monkeypatch):
    """Cursor check for every source, exclude_seen (SmolTalk duplicates), prefer_unseen / after_cursor (ARC)."""
    import json
    import bonsai.data.decontam as decontam
    import scripts.build_prompts as B
    pools = tmp_path / "filtered"
    pools.mkdir()

    def pool(name, recs):
        (pools / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))

    q = lambda i: f"Tell me a short story about the number {i}."
    pool("sft_chat", [{"messages": [{"role": "user", "content": q(i)}, {"role": "assistant", "content": "x"}]}
                      for i in range(20)])
    pool("smoltalk_chat", [{"messages": [{"role": "user", "content": "  " + q(i).upper()}]} for i in range(10)])
    mc = lambda text, src: {"question": text, "choices": ["a", "b", "c"], "answer": 1, "source": src}
    pool("mc", [mc(f"Q{i}", "arc-easy" if i % 2 == 0 else "mmlu_aux") for i in range(40)]
         + [mc(f"Q{i - 40}", "arc-easy") for i in range(40, 50)] + [mc(f"U{i}", "arc-easy") for i in range(50, 60)])
    (tmp_path / "cursors.json").write_text(json.dumps({"smoltalk_chat": 5, "mc": 40}))
    monkeypatch.setattr(B, "FILTERED_DIR", pools)
    monkeypatch.setattr(decontam, "build_index", lambda: None)
    monkeypatch.setattr(decontam, "is_contaminated", lambda t: False)
    out = {k: str(tmp_path / f"{k}.jsonl") for k in ("off", "on", "held")}

    def build(arc):
        cfg = {"prompts": {"out": out["off"], "onpolicy_out": out["on"], "heldout_out": out["held"],
                           "stats_out": str(tmp_path / "stats.json"), "total": 20, "onpolicy": 10, "heldout": 4,
                           "seed": 1, "cursor_file": str(tmp_path / "cursors.json"), "sources": {
                               "chat": {"pool": "sft_chat", "kind": "chat", "share": 0.5, "explain": False,
                                        "exclude_seen": ["smoltalk_chat"]},
                               "arc": {"pool": "mc", "kind": "mc", "share": 0.5,
                                       "match": {"source": ["arc-easy"]}, **arc}}}}
        return B.build_pools(cfg)["sources"]

    st = build({"prefer_unseen": True, "allow_overlap": True})
    recs = {k: [json.loads(x) for x in open(v)] for k, v in out.items()}
    chat = [r["messages"][0]["content"] for v in recs.values() for r in v if r["kind"] == "chat"]
    assert st["chat"]["seen_in_s1_s2"] == 5 and not {q(i) for i in range(5)} & set(chat)   # S2 trained on them
    assert {q(i) for i in range(5, 10)} <= set(chat)                       # after the smoltalk_chat cursor: kept
    arc = {k: [r["messages"][0]["content"].split("\n")[0] for r in v if r["kind"] == "mc"] for k, v in recs.items()}
    unseen = {k: sum(x.startswith("Question: U") for x in v) for k, v in arc.items()}
    assert unseen == {"held": 2, "off": 8, "on": 0}                        # unseen questions first (held-out)
    assert st["arc"]["s1_s2_overlap"] == {"cursor": 40, "below_cursor": {"held": 0, "off": 0, "on": 0},
                                          "seen_question": {"held": 0, "off": 2, "on": 5}}
    st = build({"after_cursor": True})                                     # strict: lines after the cursor only
    assert st["arc"]["lines_from"] == 40 and st["arc"]["s1_s2_overlap"]["below_cursor"] == {"held": 0, "off": 0,
                                                                                            "on": 0}
    with pytest.raises(SystemExit, match="below the S1/S2 cursor"):        # the check covers every source now
        build({})
