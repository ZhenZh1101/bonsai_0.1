"""On-policy distillation (bonsai/distill.py): the exact reverse KL against a naive implementation, the loss mask
(tool output, forced text, out-of-subset splits, stand-alone non-ASCII bytes), student row i <-> teacher row i
through s2t (padding and micro-batching invisible), the teacher's generation context (system prompt; student
position t = teacher position t + offset, exact with a context-dependent fake teacher; real Qwen3-0.6B accepts the
SFT formats in it), gradient accumulation, prompt modes, stop rules, the monitors' items, a 3-step run with a fake
teacher that lowers the val KL (and resumes), resume after a stop, the fresh-run guard and SIGHUP. Tiny models and
fake teachers, except one test with the real teacher."""

import json
import os
import signal

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map_with_path

from bonsai import chat as C
from bonsai import distill as D
from bonsai import engine as E
from bonsai.common import DATA_ROOT, TEACHER_DIR, TOKENIZER_DIR, iter_jsonl
from bonsai.model import build_model
from bonsai.optim import resolve_tc
from bonsai.tokenizer import QwenTokenizer, Tokenizer, VocabMap

SYSTEM = D.load_system_prompts("configs/offpolicy.yaml")


@pytest.fixture(scope="module")
def tok():
    qwen = QwenTokenizer()
    return Tokenizer(qwen, VocabMap.for_tests(qwen, n=3000))


def random_model(tok, seed=0):
    """Tiny fp32 model with every weight random (the default init zeroes o_proj / down_proj)."""
    mx.random.seed(seed)
    m = build_model("tiny", vocab_size=tok.vocab_size, compute_dtype=mx.float32)
    m.update(tree_map_with_path(lambda p, a: 1 + 0.1 * mx.random.normal(a.shape) if "norm" in p
                                else 0.1 * mx.random.normal(a.shape), m.parameters()))
    mx.eval(m.parameters())
    return m


class BigramTeacher:
    """Fake teacher: the log-softmax over S at row t is a function of the teacher id at t only (a low-rank bigram
    model over student ids, plus a fixed bias; teacher ids outside S, e.g. of its system prompt, act as id 0).
    Records the teacher ids it was given."""

    name = "fake"

    def __init__(self, vmap, rank=8, scale=2.0, bias=None, seed=0):
        rng = np.random.default_rng(seed)
        self.A = mx.array(rng.normal(size=(vmap.N, rank)).astype(np.float32) * scale)
        self.B = mx.array(rng.normal(size=(rank, vmap.N)).astype(np.float32))
        self.bias = mx.array(np.zeros(vmap.N, np.float32) if bias is None else bias.astype(np.float32))
        self.seen = []

    def logprobs_of(self, student_ids):
        z = self.A[mx.array(np.maximum(np.asarray(student_ids), 0))] @ self.B + self.bias
        return z - mx.logsumexp(z, axis=-1, keepdims=True)

    def iter_subset_logprobs(self, teacher_ids, vmap, chunk=8):
        ids = np.asarray(teacher_ids)
        assert ids.max() < 151936, "not teacher ids"
        self.seen.append(ids)
        s = vmap.t2s[ids]
        for a in range(0, len(ids), chunk):
            yield a, self.logprobs_of(s[a:a + chunk])


class ContextTeacher:
    """Fake teacher whose distribution at row t depends on the whole prefix ids[:t + 1] (a running hash picks one
    of K rows, the id at t another): any shift or a different context changes the numbers."""

    name = "ctx"

    def __init__(self, vmap, K=17, seed=0):
        rng = np.random.default_rng(seed)
        self.E = rng.normal(size=(K, vmap.N)) * 2
        self.F = rng.normal(size=(K, vmap.N))
        self.K, self.seen = K, []

    def logprobs_of(self, teacher_ids):
        """(T, N) float64 for one unpadded sequence of teacher ids."""
        ids = np.asarray(teacher_ids, dtype=np.int64)
        return np_log_softmax(self.E[np.cumsum(ids % 101) % self.K] + self.F[ids % self.K])

    def iter_subset_logprobs(self, teacher_ids, vmap, chunk=8):
        ids = np.asarray(teacher_ids)
        self.seen.append(ids)
        for a in range(0, len(ids), chunk):
            yield a, mx.array(np.stack([self.logprobs_of(r) for r in ids[a:a + chunk]]).astype(np.float32))


def np_log_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def np_kl(lps, lpt):
    return (np.exp(lps) * (lps - lpt)).sum(-1)


# ------------------------------------------------------------------------------------------ the KL itself


def test_reverse_kl_matches_naive(tok):
    from bonsai.teacher import subset_log_softmax
    rng = np.random.default_rng(0)
    s2t = tok.vmap.s2t
    zs = rng.normal(size=(5, tok.vocab_size)).astype(np.float32) * 3
    zt = rng.normal(size=(5, 151936)).astype(np.float32) * 3      # full-vocabulary teacher logits
    lps = mx.array(zs) - mx.logsumexp(mx.array(zs), axis=-1, keepdims=True)
    got = np.array(D.reverse_kl(lps, subset_log_softmax(mx.array(zt), s2t)))
    sub = zt[:, s2t].astype(np.float64)                            # plan 12: logits[..., s2t], renormalised over S
    want = np_kl(np_log_softmax(zs), np_log_softmax(sub))
    assert np.allclose(got, want, rtol=1e-4, atol=1e-4) and (want > 0).all()
    assert np.abs(np.array(D.reverse_kl(lps, lps))).max() < 1e-5


def test_kl_gradient_is_analytic():
    rng = np.random.default_rng(1)
    zs, lpt = rng.normal(size=(3, 50)) * 2, np_log_softmax(rng.normal(size=(3, 50)))

    def f(z):
        return D.reverse_kl(z - mx.logsumexp(z, axis=-1, keepdims=True), mx.array(lpt, dtype=mx.float32)).sum()

    g = np.array(mx.grad(f)(mx.array(zs, dtype=mx.float32)))
    lps = np_log_softmax(zs)
    kl = np_kl(lps, lpt)[:, None]
    assert np.allclose(g, np.exp(lps) * (lps - lpt - kl), atol=1e-5)    # d KL / d z = p (log p - log q - KL)


# ------------------------------------------------------------------------------------------ the mask


def test_mask_excludes_tool_output_forced_oov_and_bytes(tok):
    qwen, vmap = tok.qwen, tok.vmap
    assert vmap.t2s[qwen.encode("中")[0]] < 0, "the test needs '中' outside the test subset"
    out = "26 ½ 中"                                                  # tool output with out-of-subset tokens
    user = "Sarah has 3 boxes of 12 apples and gives away 10. How many are left? 中文 /no_think"
    conv = [{"role": "user", "content": user},
            {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": [C.python_call("print(26)")]},
            {"role": "tool", "content": out},
            {"role": "assistant", "content": "Answer: 26", "reasoning_content": ""}]
    target = [tok.bos_id] + [int(i) for i in tok.from_teacher(C.render(qwen, conv, tools=C.TOOLS)[0])]
    end = tok.special("<|im_end|>")
    last = len(target) - 1 - target[::-1].index(end)
    b1, b2 = int(vmap.byte_student[0xC3]), int(vmap.byte_student[0xA9])   # "é" spelled as two raw bytes
    target = target[:last] + [b1, b2] + target[last:]
    stub = E.ScriptedModel(E.scripted_policy([target], end), tok.vocab_size, tok.bos_id)
    r = E.chat(stub, tok, conv[:1], tools=C.TOOLS, executor=lambda call: out)
    assert r.n_tool_calls == 1 and r.finished and r.ids == target[:len(r.ids)]
    bad = D.nonascii_byte_ids(tok)
    m = D.kl_mask(r, bad)
    ids = np.array(r.ids)
    assert r.oov[:r.prompt_len].any() and (r.oov & r.tool_output).any()       # the test exercises both
    assert not m[:r.prompt_len].any() and not (m & (r.forced | r.tool_output | r.oov)).any()
    byte_pos = np.flatnonzero(r.sampled & np.isin(ids, [b1, b2]))
    assert len(byte_pos) == 2 and not m[byte_pos].any()
    assert m.sum() == r.sampled.sum() - 2 and (m <= r.sampled).all()
    assert tok.decode(ids[r.tool_output]) == "\n" + out + "\n"
    counts = D.mask_counts([r], [m], bad)
    assert counts["sampled_nonascii_byte"] == 2 and counts["scored"] == m.sum() and counts["sampled_oov"] == 0
    # the scored rows are target - 1: the last scored target is <|im_end|>, predicted from the row before it
    tg = np.flatnonzero(m)
    assert ids[tg[-1]] == end and tg.min() >= r.prompt_len


# ------------------------------------------------------------------------------------------ alignment


def test_score_aligns_student_and_teacher_rows(tok):
    """KL from `score` (right padding, several micro-batches, teacher chunks of 1) == per sequence: student row
    i - 1 of an unpadded full forward vs the teacher's distribution given the token at i - 1."""
    model, teacher = random_model(tok), BigramTeacher(tok.vmap)
    rng = np.random.default_rng(3)
    seqs = [[tok.bos_id] + rng.integers(0, tok.vocab_size, size=n).tolist() for n in (6, 14, 9, 21, 3)]
    targets = [np.sort(rng.choice(np.arange(1, len(s)), size=min(4, len(s) - 1), replace=False)) for s in seqs]
    targets[2] = np.zeros(0, dtype=np.int64)                         # nothing to score: skipped
    cfg = {**D.DEFAULTS, "micro_tokens": 5, "micro_seqs": 2, "teacher_chunk": 1}
    got, grads, _ = D.score(model, teacher, tok, seqs, targets, cfg)
    assert grads is None and len(got[2]) == 0
    for s, t, k in zip(seqs, targets, got):
        if not len(t):
            continue
        logits = np.array(model(mx.array([s]))[0])
        lps = np_log_softmax(logits[t - 1])
        lpt = np.array(teacher.logprobs_of(np.array(s)[t - 1]), dtype=np.float64)
        assert np.allclose(k, np_kl(lps, lpt), atol=2e-4), (k, np_kl(lps, lpt))
    # the teacher saw exactly s2t of the student ids (first len - 1 tokens of each row)
    flat = np.concatenate([x.reshape(-1) for x in teacher.seen])
    assert set(tok.vmap.t2s[flat].tolist()) <= set(sum((s for s in seqs), []))
    # with an offset of one the numbers would differ: sanity-check that the test can tell
    s, t = seqs[3], targets[3]
    shifted = np_kl(np_log_softmax(np.array(model(mx.array([s]))[0])[t - 1]),
                    np.array(teacher.logprobs_of(np.array(s)[t]), dtype=np.float64))
    assert not np.allclose(got[3], shifted, atol=1e-2)


def sampled(tok, n=12, seed=0):
    """Items (math with the tool, all modes and signals) and the tiny model's own samples of them."""
    model = random_model(tok, seed=2)
    cfg = {**D.DEFAULTS, "think_budget": 6, "max_answer_tokens": 10, "sample_batch": 8}
    items = D.make_items(records(n), cfg, np.random.default_rng(seed))
    return model, cfg, items, D.sample(model, tok, items, cfg, seed=1, executor=lambda c: "7")


def test_teacher_prompt_is_its_generation_prompt(tok):
    """teacher_prompt == bonsai.teacher's generation prompt of the same record / mode / signal (build_job's system
    prompt, math_tool with the tool), up to the think prefill the student samples itself."""
    from bonsai import teacher as T
    recs = records(24)
    items = D.make_items(recs, D.DEFAULTS, np.random.default_rng(3))
    assert {(it["mode"], it["signal"]) for it in items} == {("think", "default"), ("think", "soft"),
                                                           ("no_think", "soft"), ("no_think", "prefill")}
    for rec, it in zip(recs, items):
        job = T.build_job({**rec, "tools": bool(it["tools"])}, it["mode"], SYSTEM, it["signal"])
        row = T._ChatRow(tok.qwen, job, 128, 400, 2, 2048, lambda c: "")
        tp = D.teacher_prompt(tok.qwen, it, SYSTEM)
        rest = row.ids[len(tp):]
        assert row.ids[:len(tp)] == tp
        assert rest == ([] if it["signal"] == "prefill" else row.prefill)
        kind = "math_tool" if it["tools"] else it["kind"]
        assert tok.qwen.decode(tp).startswith("<|im_start|>system\n" + SYSTEM[kind])
        assert tok.qwen.decode(tp) == C.render_text([{"role": "system", "content": SYSTEM[kind]}] + it["messages"],
                                                    tools=it["tools"], add_generation_prompt=True,
                                                    enable_thinking=it["enable_thinking"])


def test_score_in_teacher_context_aligns_exactly(tok):
    """The teacher scores its system prompt + the student's completion: KL from `score` (micro-batches, padding,
    chunks of 1) == the student's unpadded rows t - 1 vs a context-dependent teacher's rows t + offset - 1 of the
    unpadded teacher sequence; the context changes the numbers, a context-free teacher's do not move."""
    model, cfg, items, results = sampled(tok)
    cfg = {**cfg, "micro_tokens": 40, "micro_seqs": 2, "teacher_chunk": 1}
    bad = D.nonascii_byte_ids(tok)
    targets = [np.flatnonzero(D.kl_mask(r, bad)) for r in results]
    seqs = [r.ids for r in results]
    tseqs = D.teacher_sequences(tok, items, results, SYSTEM)
    teacher = ContextTeacher(tok.vmap)
    got, _, _ = D.score(model, teacher, tok, seqs, targets, cfg, tseqs=tseqs)
    assert sum(len(t) for t in targets) > 50 and any(it["tools"] for it in items)
    for r, t, (tids, off, start), k in zip(results, targets, tseqs, got):
        assert start == r.prompt_len and len(tids) == len(r.ids) + off
        assert tids[start + off:].tolist() == tok.vmap.s2t[r.ids[start:]].tolist()     # the same completion
        if not len(t):
            continue
        lps = np_log_softmax(np.array(model(mx.array([r.ids]))[0])[t - 1])
        lpt = teacher.logprobs_of(tids[:-1])[t + off - 1]
        assert np.allclose(k, np_kl(lps, lpt), atol=2e-4), (k, np_kl(lps, lpt))
    plain, _, _ = D.score(model, ContextTeacher(tok.vmap), tok, seqs, targets, cfg)    # no context, offset 0
    assert not all(np.allclose(a, b, atol=1e-2) for a, b in zip(got, plain) if len(a))
    bigram = BigramTeacher(tok.vmap)
    a, _, _ = D.score(model, bigram, tok, seqs, targets, cfg, tseqs=tseqs)
    b, _, _ = D.score(model, bigram, tok, seqs, targets, cfg)
    assert all(np.allclose(x, y, atol=1e-5) for x, y in zip(a, b))


REAL = [p for p in (TEACHER_DIR / "Qwen3-0.6B" / "config.json", TOKENIZER_DIR / "vocab_subset.npz",
                    DATA_ROOT / "data/smoke/offpolicy/kept_s0of1.jsonl") if not p.exists()]


@pytest.mark.skipif(bool(REAL), reason=f"missing {REAL}")
def test_real_teacher_accepts_the_sft_formats_in_its_context():
    """Qwen3-0.6B scoring an SFT tool sample (generated with the math_tool system prompt) as distill does: the
    KL of a student sure of the SFT token (= -log p_teacher) at <tool_call> and at "Answer" is < 1 nat in the
    teacher's generation context, several nats without it (the off-policy formats would be trained away)."""
    from bonsai.teacher import Teacher
    from bonsai.tokenizer import get_tokenizer
    tok = get_tokenizer()
    s = next(r for r in iter_jsonl(DATA_ROOT / "data/smoke/offpolicy/kept_s0of1.jsonl")
             if r["tools"] and r["signal"] == "prefill" and r["gen"]["n_tool_calls"] == 1)
    conv = s["messages"]
    target = [tok.bos_id] + [int(i) for i in tok.from_teacher(C.render(tok.qwen, conv, tools=C.TOOLS)[0])]
    out = next(m["content"] for m in conv if m["role"] == "tool")
    stub = E.ScriptedModel(E.scripted_policy([target], tok.special("<|im_end|>")), tok.vocab_size, tok.bos_id)
    r = E.chat(stub, tok, conv[:1], enable_thinking=False, tools=C.TOOLS, executor=lambda c: out)
    assert r.finished and r.n_tool_calls == 1 and r.ids == target[:len(r.ids)]
    item = {"messages": conv[:1], "enable_thinking": False, "tools": C.TOOLS, "kind": "math", "mode": "no_think",
            "signal": "prefill", "id": s["prompt_id"]}
    ids, scored = np.asarray(r.ids), np.flatnonzero(D.kl_mask(r, D.nonascii_byte_ids(tok)))
    pos = [p for p in scored if ids[p] == tok.special("<tool_call>")][:1] + \
          [p for p in scored if tok.decode([int(ids[p])]).strip() == "Answer"][:1]
    assert len(pos) == 2
    teacher = Teacher("Qwen3-0.6B")

    def nll(system_prompts):
        tseqs = D.teacher_sequences(tok, [item], [r], system_prompts)
        x, rows, cols = D.teacher_rows(tok, [r.ids], [np.array(pos)], tseqs)
        lp = np.array(D.teacher_logprobs(teacher, tok.vmap, x, rows, cols, 1))
        return -lp[np.arange(len(pos)), ids[pos]]

    with_ctx, without = nll(SYSTEM), nll({})
    assert (with_ctx < 1.0).all() and without.max() > 5.0, (with_ctx, without)


def test_gradient_accumulation_matches_one_batch(tok):
    model, teacher = random_model(tok, seed=1), BigramTeacher(tok.vmap, seed=1)
    rng = np.random.default_rng(4)
    seqs = [[tok.bos_id] + rng.integers(0, tok.vocab_size, size=n).tolist() for n in (8, 12, 5, 10)]
    targets = [np.arange(1, len(s)) for s in seqs]
    n = sum(len(t) for t in targets)
    one = {**D.DEFAULTS, "micro_tokens": 10 ** 6, "micro_seqs": 100}
    many = {**D.DEFAULTS, "micro_tokens": 7, "micro_seqs": 1, "teacher_chunk": 1}
    k1, g1, _ = D.score(model, teacher, tok, seqs, targets, one, grad_scale=1.0 / n)
    k2, g2, _ = D.score(model, teacher, tok, seqs, targets, many, grad_scale=1.0 / n)
    for a, b in zip(k1, k2):
        assert np.allclose(a, b, atol=1e-5)
    f1, f2 = dict(tree_flatten(g1)), dict(tree_flatten(g2))
    assert f1.keys() == f2.keys()
    for key in f1:
        assert np.allclose(np.array(f1[key]), np.array(f2[key]), atol=1e-5, rtol=1e-3), key
    assert float(mx.abs(f1["embed_tokens.weight"]).sum()) > 0


def test_micro_groups_cover_once():
    lengths, counts = [5, 30, 12, 7, 40, 3], [4, 25, 0, 6, 39, 2]
    groups = D.micro_groups(lengths, counts, max_tokens=30, max_seqs=2)
    flat = sorted(i for g in groups for i in g)
    assert flat == [0, 1, 3, 4, 5]
    for g in groups:
        assert len(g) <= 2 and (len(g) == 1 or sum(counts[i] for i in g) <= 30)


# ------------------------------------------------------------------------------------------ prompts and modes


def records(n):
    out = []
    for i in range(n):
        kind = ["chat", "math", "explain", "mc"][i % 4]
        out.append({"id": f"p{i}", "kind": kind, "tools": kind == "math" and i % 8 == 1, "answer": "3",
                    "messages": [{"role": "user", "content": f"What is {i} plus {i}?"}]})
    return out


def test_make_items_modes_signals_tools():
    cfg = {**D.DEFAULTS, "math_tools": 0.5}
    recs = records(4000)
    items = D.make_items(recs, cfg, np.random.default_rng(0))
    think = [it for it in items if it["mode"] == "think"]
    assert 0.45 < len(think) / len(items) < 0.55
    assert 0.25 < np.mean([it["signal"] == "soft" for it in think]) < 0.35
    no = [it for it in items if it["mode"] == "no_think"]
    assert 0.45 < np.mean([it["signal"] == "prefill" for it in no]) < 0.55
    for it, rec in zip(items, recs):
        u = it["messages"][-1]["content"]
        assert it["enable_thinking"] == (it["signal"] != "prefill")
        assert C.effective_mode(it["messages"], it["enable_thinking"]) == it["mode"]
        assert u.endswith(" /think") == (it["signal"] == "soft" and it["mode"] == "think")
        assert u.endswith(" /no_think") == (it["signal"] == "soft" and it["mode"] == "no_think")
        assert it["sampling"] == (E.THINK_SAMPLING if it["mode"] == "think" else E.NO_THINK_SAMPLING)
        if rec["kind"] != "math":
            assert it["tools"] is None
        elif rec["tools"]:
            assert it["tools"] == C.TOOLS
        assert rec["messages"][-1]["content"] == f"What is {rec['id'][1:]} plus {rec['id'][1:]}?"   # not mutated
    math_plain = [it for it, r in zip(items, recs) if r["kind"] == "math" and not r["tools"]]
    assert 0.4 < np.mean([it["tools"] is not None for it in math_plain]) < 0.6
    assert all(it["tools"] for it in D.make_items(recs[:40], D.DEFAULTS, np.random.default_rng(1)) if it["kind"] == "math")
    # held-out: stable per prompt id; training: a new draw per step
    assert [i["signal"] for i in D.make_items(recs[:50], cfg)] == [i["signal"] for i in D.make_items(recs[:50], cfg)]
    a = D.make_items(recs[:50], cfg, np.random.default_rng([1, 0]))
    b = D.make_items(recs[:50], cfg, np.random.default_rng([1, 1]))
    assert [i["mode"] for i in a] != [i["mode"] for i in b]
    pp = D.sampling_for({"sampling": "qwen", "presence_penalty": 0.0}, "think")
    assert pp.temperature == 0.6 and pp.presence_penalty == 0.0


def test_prompt_order_fresh_then_repeats():
    order = D.PromptOrder(10, 4, seed=3)
    seen = [order(s) for s in range(5)]
    first = sum((idx for idx, _ in seen[:2]), [])
    assert len(set(first)) == 8 and seen[0][1] == 0
    assert seen[2][1] == 1 and len(set(seen[0][0] + seen[1][0] + seen[2][0][:2])) == 10   # epoch 0 = a permutation
    assert D.PromptOrder(10, 4, seed=3)(1) == seen[1]


def test_onpolicy_lr_from_sft_meta():
    from bonsai.optim import scale_tc
    base = resolve_tc({}, 640)
    sft = scale_tc(base, 0.15)
    tc, _ = D.onpolicy_tc({"kind": "sft", "optim": sft, "base_optim": base, "lr_frac": 0.15, "lr_scale": 1.0}, 640,
                          D.DEFAULTS)
    assert tc["optimizer"] == "muon"
    for g, v in tc["groups"].items():
        assert v == pytest.approx(0.5 * 0.15 * base["groups"][g])
    tc2, _ = D.onpolicy_tc({"kind": "sft", "optim": sft, "lr_frac": 0.15}, 640, {**D.DEFAULTS, "optimizer": "adamw"})
    assert tc2["optimizer"] == "adamw" and tc2["groups"]["decay"] == pytest.approx(0.5 * 0.15 * 1.2e-3)
    tc3, note = D.onpolicy_tc({"optim": base, "lr_scale": 0.7}, 640, D.DEFAULTS)    # a base model, no SFT
    assert tc3["groups"]["muon"] == pytest.approx(0.5 * 0.15 * 0.7 * base["groups"]["muon"]) and "not an SFT" in note
    assert D.forwarded_base_optim({"kind": "sft", "optim": sft, "base_optim": base}) == base
    assert D.forwarded_base_optim({"optim": base}) == base and D.forwarded_base_optim({"kind": "sft", "optim": sft}) is None


def test_stop_rules():
    cfg = D.DEFAULTS
    base = {"step": 0, "val_kl": 1.0, "gsm8k": 0.20, "format_errors": 0, "format_n": 20}
    assert D.health(base, base, cfg) == (True, "")
    assert not D.health({**base, "gsm8k": 0.16}, base, cfg)[0]
    assert D.health({**base, "gsm8k": 0.18}, base, cfg)[0]
    assert D.health({**base, "format_errors": 1}, base, cfg)[0]
    ok, why = D.health({**base, "format_errors": 2}, base, cfg)
    assert not ok and "format" in why
    assert D.health({"step": 25}, base, cfg)[0]                      # missing monitors never stop the run
    best = {"step": 25, "val_kl": 0.8}
    assert D.stop_reason({"step": 50, "healthy": True, "why": ""}, best, cfg) is None
    assert "no val-KL improvement" in D.stop_reason({"step": 75, "healthy": True, "why": ""}, best, cfg)
    assert D.stop_reason({"step": 50, "healthy": False, "why": "GSM8K"}, best, cfg) == "GSM8K"
    assert D.is_better({"healthy": True, "val_kl": 0.7}, best, cfg)
    assert not D.is_better({"healthy": True, "val_kl": 0.9}, best, cfg)
    assert not D.is_better({"healthy": False, "val_kl": 0.1}, best, cfg)


# ------------------------------------------------------------------------------------------ the loop


def tiny_meta(model, tok):
    tc = resolve_tc({"optimizer": "adamw", "adamw_lr": 0.01}, model.cfg.d_model)
    return {"kind": "sft", "optim": tc, "base_optim": tc, "lr_frac": 1.0, "lr_scale": 1.0,
            "model_config": model.cfg.to_dict(), "vocab_sha": tok.vmap.sha()}


def test_three_steps_with_fake_teacher_reduce_kl(tok, tmp_path):
    from bonsai.checkpoint import load_meta
    rng = np.random.default_rng(5)
    bias = np.zeros(tok.vocab_size)
    bias[rng.choice(tok.vocab_size, 20, replace=False)] = 9.0     # a peaked teacher the student can move towards
    teacher = BigramTeacher(tok.vmap, scale=0.3, bias=bias, seed=2)
    model = random_model(tok)          # the default init's zero residual branches need a few steps to get going
    meta = tiny_meta(model, tok)
    cfg = {**D.DEFAULTS, "steps": 3, "batch": 4, "eval_every": 3, "ckpt_every": 3, "gsm8k_n": 0, "format_n": 0,
           "think_budget": 6, "max_answer_tokens": 10, "sample_batch": 8, "warmup": 0, "lr_frac": 1.0, "run": "t"}
    prompts, held = records(16), records(24)[16:]
    calls = []
    trainer = D.OnPolicy(cfg, model, tok, teacher, prompts, held, meta, tmp_path / "ckpt", tmp_path / "logs",
                         src="tiny", executor=lambda c: calls.append(c) or "1")
    summary = trainer.run()
    ev = trainer.state["evals"]
    assert [e["step"] for e in ev] == [0, 3]
    assert ev[1]["val_kl"] < 0.9 * ev[0]["val_kl"], ev     # 0.52 x here; 0.41-0.80 x over 12 seeds / LRs
    lines = [json.loads(x) for x in (tmp_path / "logs" / "onpolicy.jsonl").read_text().splitlines()]
    train = [x for x in lines if x["kind"] == "train"]
    assert [x["step"] for x in train] == [1, 2, 3] and all(x["updated"] and x["tokens"] > 0 for x in train)
    assert train[0]["mask"]["scored"] == train[0]["tokens"] <= train[0]["mask"]["sampled"]
    final = tmp_path / "ckpt" / "onpolicy_final.safetensors"
    m = load_meta(final)
    assert m["kind"] == "onpolicy" and m["step"] == 3 and m["vocab_sha"] == tok.vmap.sha()
    assert m["base_optim"] == meta["base_optim"] and m["lr_scale"] == 1.0 and m["optim"]["groups"]["decay"] == 0.01
    assert summary["val_kl"] == ev[1]["val_kl"] and (tmp_path / "ckpt" / "onpolicy" / "step_000003").exists()
    # resume: one more step continues at step 3 with the restored history and optimizer
    model2 = random_model(tok, seed=9)
    t2 = D.OnPolicy({**cfg, "steps": 4}, model2, tok, teacher, prompts, held, meta, tmp_path / "ckpt",
                    tmp_path / "logs", executor=lambda c: "1")
    t2.run(resume="auto")
    assert [e["step"] for e in t2.state["evals"]] == [0, 3, 4] and t2.step == 4
    lines = [json.loads(x) for x in (tmp_path / "logs" / "onpolicy.jsonl").read_text().splitlines()]
    assert [x["step"] for x in lines if x["kind"] == "train"] == [1, 2, 3, 4]


# ------------------------------------------------------------------------------------------ monitors


def test_monitors_use_heldout_math_and_dev_prompts(tok, tmp_path, monkeypatch):
    """The GSM8K-style monitor gets the held-out math items (not the GSM8K test set), the format check the dev
    prompts at the SFT answer cap; a reply cut by the cap is not a format error; a step-0 monitor failure stops."""
    from bonsai.tasks import evals
    calls = {}

    def fake_gsm(model, tok, n=0, items=None, batch=32, **kw):
        calls["gsm"] = {"n": n, "items": items, **kw}
        return {"accuracy": 0.5, "n": len(items), "tool_use_frac": 1.0, "finished_frac": 1.0}

    def fake_ms(model, tok, n=0, split="test", max_answer_tokens=200, batch=34, **kw):
        calls["ms"] = {"n": n, "split": split, "max_answer_tokens": max_answer_tokens}
        return {"accuracy": 0.7, "n": n, "finished_frac": 0.9,
                "failures": {"not finished (length)": 2, "empty think block": 1}}

    monkeypatch.setattr(evals, "eval_gsm8k_tool", fake_gsm)
    monkeypatch.setattr(evals, "eval_mode_switch", fake_ms)
    model = random_model(tok)
    recs = records(40) + [{"id": "x", "kind": "math", "answer": None, "messages": [{"role": "user", "content": "?"}]}]
    cfg = {**D.DEFAULTS, "gsm8k_n": 5, "format_n": 10, "run": "t"}
    tr = D.OnPolicy(cfg, model, tok, BigramTeacher(tok.vmap), recs[:8], [], tiny_meta(model, tok), tmp_path / "c",
                    tmp_path / "l", monitor=recs)
    out = tr.monitor_evals()
    qs = {r["messages"][0]["content"] for r in recs if r["kind"] == "math" and r["answer"]}
    got = calls["gsm"]["items"]
    assert len(got) == 5 and all(it["question"] in qs and it["answer"] == 3.0 for it in got)
    assert calls["ms"] == {"n": 10, "split": "dev", "max_answer_tokens": 400}
    assert out["format_errors"] == 1 and out["format_truncated"] == 2 and out["gsm8k"] == 0.5
    assert D.math_items(recs, 5) == got                                          # a fixed subset
    monkeypatch.setattr(evals, "eval_mode_switch", lambda *a, **k: 1 / 0)
    with pytest.raises(SystemExit, match="step-0 monitor failed"):
        D.OnPolicy(cfg, model, tok, BigramTeacher(tok.vmap), recs[:8], [], tiny_meta(model, tok), tmp_path / "c2",
                   tmp_path / "l2", monitor=recs).evaluate()


@pytest.mark.skipif(not (DATA_ROOT / "data/prompts_heldout.jsonl").exists(), reason="no held-out prompts")
def test_monitor_items_from_heldout_file_are_not_gsm8k_test():
    items = D.math_items(D.load_records("data/prompts_heldout.jsonl"))
    assert len(items) >= 100 and all(isinstance(it["answer"], float) and it["question"] for it in items)
    if (DATA_ROOT / "raw/gsm8k/main/test-00000-of-00001.parquet").exists():
        from bonsai.tasks.evals import load_gsm8k
        test = {" ".join(g["question"].split()) for g in load_gsm8k("test")}
        assert not [it for it in items if " ".join(it["question"].split()) in test]


# ------------------------------------------------------------------------------------------ resume, guard, signals


class GsmScript(D.OnPolicy):
    """The GSM8K monitor replaced by a script of accuracies, one per eval."""
    script = []

    def monitor_evals(self):
        return {"gsm8k": self.script.pop(0)} if self.script else {}


def loop_cfg(**kw):
    return {**D.DEFAULTS, "steps": 6, "batch": 4, "eval_every": 2, "ckpt_every": 2, "keep": 3, "gsm8k_n": 0,
            "format_n": 0, "think_budget": 6, "max_answer_tokens": 10, "sample_batch": 8, "warmup": 0, "lr_frac": 1.0,
            "run": "t", **kw}


def trainer(tok, tmp_path, cfg, script=(), cls=GsmScript, teacher=None, seed=0):
    model = random_model(tok, seed=seed)
    t = cls(cfg, model, tok, teacher or BigramTeacher(tok.vmap, scale=0.3, seed=2), records(16), [],
            tiny_meta(model, tok), tmp_path / "ckpt", tmp_path / "logs", executor=lambda c: "1")
    t.script = list(script)
    return t


def train_steps(tmp_path):
    return [json.loads(x) for x in (tmp_path / "logs" / "onpolicy.jsonl").read_text().splitlines()]


def test_resume_after_health_stop_goes_back_to_best(tok, tmp_path):
    """Plan 16 (`--resume auto --lr-frac 0.25`): the stop is cleared, training restarts at the best healthy
    checkpoint with the new LR, later evals are dropped and later checkpoints moved out of rotation."""
    s1 = trainer(tok, tmp_path, loop_cfg(), [0.3, 0.3, 0.1]).run()     # step 4 collapses (no val KL: best = 2)
    assert s1["stop_reason"].startswith("GSM8K-style monitor dropped") and s1["steps_run"] == 4 and s1["step"] == 2
    t2 = trainer(tok, tmp_path, loop_cfg(lr_frac=0.5), [0.3, 0.3], seed=9)
    s2 = t2.run(resume="auto")
    lines = train_steps(tmp_path)
    res = [x for x in lines if x["kind"] == "resume"]
    assert len(res) == 1 and res[0]["step"] == 2 and res[0]["dropped_evals"] == [4] and res[0]["old_stop"]
    assert [x["step"] for x in lines if x["kind"] == "train"] == [1, 2, 3, 4, 3, 4, 5, 6]
    assert [e["step"] for e in t2.state["evals"]] == [0, 2, 4, 6] and all(e["healthy"] for e in t2.state["evals"])
    assert s2["steps_run"] == 6 and s2["stop_reason"] == "6 steps done" and s2["step"] == 6
    ck = tmp_path / "ckpt" / "onpolicy"
    assert sorted(d.name for d in ck.glob("step_*")) == ["step_000002", "step_000004", "step_000006"]
    moved = list(ck.glob("abandoned_*/step_000004"))
    assert len(moved) == 1 and json.loads((moved[0] / "meta.json").read_text())["state"]["stop"]
    from bonsai.checkpoint import load_meta
    assert load_meta(ck / "best.safetensors")["step"] == 6


class AtResume(GsmScript):
    """Records the weights and state the first training step starts from."""

    def train_step(self):
        if not hasattr(self, "at_resume"):
            self.at_resume = ({k: np.array(v) for k, v in tree_flatten(self.model.parameters())}, self.step,
                              [e["step"] for e in self.state["evals"]], self.state["stop"], self.lr_mult(self.step),
                              sorted(d.name for d in self.ckpt_dir.glob("step_*")))
        return super().train_step()


def test_resume_to_step_0_uses_best_weights_and_a_fresh_optimizer(tok, tmp_path):
    t1 = trainer(tok, tmp_path, loop_cfg(warmup=2), [0.3, 0.1])         # step 2 collapses: best = the start model
    start = {k: np.array(v) for k, v in tree_flatten(t1.model.parameters())}
    assert t1.run()["step"] == 0
    t2 = trainer(tok, tmp_path, loop_cfg(steps=2, warmup=2), [0.3], cls=AtResume, seed=9)
    t2.run(resume="auto")
    weights, step, evals, stop, lr, dirs = t2.at_resume
    assert step == 0 and evals == [0] and stop is None and lr == 0.5 and dirs == []
    for k, v in weights.items():
        assert np.allclose(v, start[k], atol=2e-2, rtol=1e-2), k                     # bf16 copy of the start
    res = [x for x in train_steps(tmp_path) if x["kind"] == "resume"]
    assert res[0]["fresh_optimizer"] and res[0]["dropped_evals"] == [2] and res[0]["from"].endswith("best.safetensors")
    assert [x["step"] for x in train_steps(tmp_path) if x["kind"] == "train"] == [1, 2, 1, 2]


def test_resume_latest_keeps_the_newest_checkpoint(tok, tmp_path):
    trainer(tok, tmp_path, loop_cfg(), [0.3, 0.3, 0.1]).run()
    t2 = trainer(tok, tmp_path, loop_cfg(), [0.3], seed=9)
    assert t2.resume("latest") and t2.step == 4 and t2.state["stop"] is None and t2.state["best"]["step"] == 2


def test_fresh_run_refuses_an_earlier_attempt(tok, tmp_path):
    trainer(tok, tmp_path, loop_cfg(steps=2, eval_every=10)).run()
    with pytest.raises(SystemExit, match="earlier attempt"):
        trainer(tok, tmp_path, loop_cfg(steps=2, eval_every=10)).run()
    trainer(tok, tmp_path, loop_cfg(steps=2, eval_every=10)).run(overwrite=True)
    ck = tmp_path / "ckpt"
    old = list(ck.glob("onpolicy.*"))
    assert len(old) == 1 and (old[0] / "step_000002").exists() and (old[0] / "onpolicy_final.safetensors").exists()
    assert (ck / "onpolicy" / "step_000002").exists() and (ck / "onpolicy_final.safetensors").exists()
    assert [x["step"] for x in train_steps(tmp_path) if x["kind"] == "train"] == [1, 2]   # the new attempt only
    assert len(list((tmp_path / "logs").glob("onpolicy.*.jsonl"))) == 1


class HupTeacher(BigramTeacher):
    """Sends this process SIGHUP (a closed terminal) the first time it scores."""
    sent = False

    def iter_subset_logprobs(self, teacher_ids, vmap, chunk=8):
        if not self.sent:
            self.sent = True
            os.kill(os.getpid(), signal.SIGHUP)
        yield from super().iter_subset_logprobs(teacher_ids, vmap, chunk)


def test_sighup_saves_a_checkpoint_and_stops(tok, tmp_path):
    caught = []
    prev = signal.signal(signal.SIGHUP, lambda *a: caught.append(a))     # the test's safety net, never reached
    try:
        out = trainer(tok, tmp_path, loop_cfg(), teacher=HupTeacher(tok.vmap)).run()
    finally:
        restored = signal.getsignal(signal.SIGHUP)
        signal.signal(signal.SIGHUP, prev)
    assert out == {"stopped": "signal", "step": 1} and not caught and restored is not prev
    assert (tmp_path / "ckpt" / "onpolicy" / "step_000001").exists()
