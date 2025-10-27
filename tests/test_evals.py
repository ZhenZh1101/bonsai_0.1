"""Every eval task runs end to end on a tiny random model with n=4 (plan section 15); CORE's three task types;
the mode-switch checker on scripted conversations; the scoring helpers against a plain full forward."""

import importlib
import sys
import types

import mlx.core as mx
import numpy as np
import pytest

from bonsai import chat as C
from bonsai import engine as E
from bonsai.common import RAW_DIR
from bonsai.model import build_model
from bonsai.tasks import evals as V
from bonsai.tokenizer import QwenTokenizer, Tokenizer, VocabMap

SMALL = {"batch": 4, "max_answer_tokens": 12, "think_budget": 6}


@pytest.fixture(scope="module")
def tok():
    qwen = QwenTokenizer()
    return Tokenizer(qwen, VocabMap.for_tests(qwen))


@pytest.fixture(scope="module")
def model(tok):
    mx.random.seed(0)
    return build_model("tiny", vocab_size=tok.vocab_size)


def need(*paths):
    missing = [p for p in paths if not (RAW_DIR / p).exists()]
    if missing:
        pytest.skip(f"eval data missing: {missing}")


def check(res, n):
    assert res["n"] == n and 0.0 <= res["accuracy"] <= 1.0 and isinstance(res["samples"], list)


def test_scoring_in_fp32(tok):
    """MC letter logits and span log-probs come from an fp32 head: no bf16 rounding (ties resolved towards "A"),
    also on a bf16-cast inference model; cols= gives the same numbers as the full row."""
    mx.random.seed(2)
    m = E.cast_for_inference(build_model("tiny", vocab_size=tok.vocab_size))
    seqs = [tok.encode("Question: x\nAnswer:", bos=True), tok.encode("Answer:", bos=True)]
    cols = [tok.encode(" " + L)[0] for L in "ABCDE"]
    full, part = E.final_logits(m, seqs), E.final_logits(m, seqs, cols=cols)
    assert full.dtype == np.float32 and part.shape == (2, 5)
    np.testing.assert_allclose(part, full[:, cols], rtol=1e-5, atol=1e-5)
    low = full.view(np.uint32) & 0xFFFF     # bf16 values have their low 16 bits zero
    assert (low != 0).mean() > 0.9
    lp = E.span_logprobs(m, seqs, [(1, len(seqs[0])), (1, len(seqs[1]))])[0][0]
    assert ((lp.view(np.uint32) & 0xFFFF) != 0).any()


def test_scoring_helpers_match_full_forward(model, tok):
    seqs = [tok.encode("The cat sat on the mat.", bos=True), tok.encode("Two plus two is four", bos=True)]
    spans = [(3, len(seqs[0])), (1, len(seqs[1]))]
    got = E.span_logprobs(model, seqs, spans)
    lg = E.final_logits(model, seqs)
    for i, s in enumerate(seqs):
        full = model(mx.array([s]))[0]
        logp = np.array(full - mx.logsumexp(full, axis=-1, keepdims=True))
        a, b = spans[i]
        ref = logp[np.arange(a - 1, b - 1), s[a:b]]
        np.testing.assert_allclose(got[i][0], ref, atol=2e-2)
        np.testing.assert_allclose(lg[i], np.array(full[-1]), atol=2e-2)


@pytest.mark.parametrize("task", ["arc_easy", "arc_challenge", "mmlu"])
def test_mc(model, tok, task):
    need("arc/ARC-Easy/test-00000-of-00001.parquet", "mmlu/all/test-00000-of-00001.parquet")
    res = V.TASKS[task](model, tok, n=4, batch=4)
    check(res, 4)
    assert sum(res["predicted_letters"].values()) == 4
    assert 0 <= res["always_most_predicted_accuracy"] <= 1 and res["always_letter_accuracy"]


def test_hellaswag(model, tok):
    need("hellaswag/data/validation-00000-of-00001.parquet", "hellaswag/data/train-00000-of-00001.parquet")
    res = V.eval_hellaswag(model, tok, n=4, shots=2)
    check(res, 4)
    assert res["acc_norm"] == res["accuracy"]


def test_gsm8k_custom_items(model, tok):
    """items= replaces the GSM8K test set (checkpoint selection on held-out items, not on the test set)."""
    items = [{"question": "Tom has 3 apples and buys 4 more. How many apples does he have?", "answer": "7"},
             {"question": "A box holds 12 eggs. How many eggs are in 2 boxes?", "answer": 24},
             {"question": "Sam had $10 and spent $4. How much is left?", "answer": "Sam has 10 - 4 = 6\n#### 6"}]
    res = V.eval_gsm8k_tool(model, tok, n=2, items=items, **SMALL)
    check(res, 2)
    assert {s["prompt"] for s in res["samples"]} <= {it["question"] for it in items}
    assert {"finished_frac", "tool_use_frac"} <= set(res)
    assert [V.gold_number(it["answer"]) for it in items] == [7.0, 24.0, 6.0] and V.gold_number("$1,234") == 1234
    with pytest.raises(ValueError):
        V.eval_gsm8k_tool(model, tok, items=[{"question": "Why?", "answer": "because"}], **SMALL)


@pytest.mark.parametrize("task", ["gsm8k_tool", "gsm8k_think"])
def test_gsm8k(model, tok, task):
    need("gsm8k/main/test-00000-of-00001.parquet")
    res = V.TASKS[task](model, tok, n=4, **SMALL)
    check(res, 4)
    assert len(res["samples"]) == 4 and all("completion" in s for s in res["samples"])


def fake_fld(monkeypatch):
    items = [{"user": f"Facts:\nfact1: A is B.\n\nHypothesis: A is {x}.\n\nAnswer with \"Answer: PROVED\".",
              "answer": f"Answer: {lab}", "label": lab, "depth": d}
             for x, lab, d in [("B", "PROVED", 2), ("C", "UNKNOWN", 3), ("not B", "DISPROVED", 4), ("D", "UNKNOWN", 2),
                               ("E", "UNKNOWN", 4)]]
    mod = types.ModuleType("bonsai.formal.fld")
    mod.load_fld_test = lambda depths=(2, 3, 4), **kw: [it for it in items if it["depth"] in depths]
    try:
        pkg = importlib.import_module("bonsai.formal")
    except ImportError:
        pkg = types.ModuleType("bonsai.formal")
        monkeypatch.setitem(sys.modules, "bonsai.formal", pkg)
    monkeypatch.setitem(sys.modules, "bonsai.formal.fld", mod)
    monkeypatch.setattr(pkg, "fld", mod, raising=False)


def test_fld_with_fake_module(model, tok, monkeypatch):
    fake_fld(monkeypatch)
    res = V.eval_fld(model, tok, n=4, **SMALL)
    check(res, 4)
    assert sum(v["n"] for v in res["by_depth"].values()) == 4 and sum(res["gold_labels"].values()) == 4


def test_mode_switch_runs(model, tok):
    res = V.eval_mode_switch(model, tok, n=6, **SMALL)
    check(res, 6)
    assert {m: v["n"] for m, v in res["by_mode"].items()} == {"think": 2, "no_think": 2, "prefilled": 2}
    assert len(V.mode_switch_items()) == 100
    assert [m for _, m in V.mode_switch_items()].count("think") == 34
    assert {"finished_frac", "truncated", "strict_accuracy", "failures"} <= set(res) and res["split"] == "eval"
    assert res["strict_accuracy"] <= res["accuracy"]
    assert sum(res["failures"].values()) == round((1 - res["accuracy"]) * res["n"])


def test_mode_switch_dev_split(model, tok):
    """The dev prompts (checkpoint gating) are disjoint from the frozen reported ones and run the same way."""
    dev, ev = V.mode_switch_prompts("dev"), V.mode_switch_prompts("eval")
    assert len(dev) >= 24 and len(set(dev)) == len(dev) and not {p.lower() for p in dev} & {p.lower() for p in ev}
    assert len(V.mode_switch_items(split="dev")) == 3 * len(dev)
    assert {p for p, _ in V.mode_switch_items(20, split="dev")} <= set(dev)
    res = V.eval_mode_switch(model, tok, n=3, split="dev", **SMALL)
    check(res, 3)
    assert res["split"] == "dev" and res["samples"][0]["prompt"].endswith(dev[0])


def test_mode_check_scripted(tok):
    """The checker passes the right formats and fails the wrong ones (scripted model, exact template turns)."""
    end = tok.special("<|im_end|>")

    def result(user, reasoning, answer, enable_thinking=True):
        msgs = [{"role": "user", "content": user}]
        full = msgs + [{"role": "assistant", "content": answer, "reasoning_content": reasoning}]
        target = [tok.bos_id] + tok.from_teacher(C.render(tok.qwen, full)[0]).tolist()
        stub = E.ScriptedModel(E.scripted_policy([target], end), tok.vocab_size, tok.bos_id)
        return E.chat(stub, tok, msgs, enable_thinking=enable_thinking)

    q = "What is gravity?"
    think = result(C.with_soft_switch(q, "think"), "Gravity pulls masses.", "A force.")
    empty = result(C.with_soft_switch(q, "no_think"), "", "A force.")
    assert V.mode_check(tok, think, "think") == (True, "ok")
    assert V.mode_check(tok, empty, "no_think") == (True, "ok")
    assert not V.mode_check(tok, think, "no_think")[0] and not V.mode_check(tok, empty, "think")[0]
    pre = result(q, "", "A force.", enable_thinking=False)
    assert V.mode_check(tok, pre, "prefilled") == (True, "ok")
    blank = result(C.with_soft_switch(q, "no_think"), "", "")
    assert V.mode_check(tok, blank, "no_think") == (False, "empty answer")


def test_mode_check_length_apart(tok):
    """A reply cut by the answer cap that obeys the mode passes (length is not a mode failure); one cut before any
    answer fails with the TRUNCATED key; <|endoftext|> instead of <|im_end|> fails."""
    end = tok.special("<|im_end|>")
    long = " ".join(["Gravity is the force that pulls masses together."] * 6)

    def run(user, reasoning, answer, stop=end, **kw):
        msgs = [{"role": "user", "content": user}]
        full = msgs + [{"role": "assistant", "content": answer, "reasoning_content": reasoning}]
        target = [tok.bos_id] + tok.from_teacher(C.render(tok.qwen, full)[0]).tolist()
        target = target[:-2] + [stop] if stop != end else target
        stub = E.ScriptedModel(E.scripted_policy([target], end), tok.vocab_size, tok.bos_id)
        return E.chat(stub, tok, msgs, **kw)

    q = "What is gravity?"
    for mode, reasoning in (("think", "Masses attract."), ("no_think", "")):
        r = run(C.with_soft_switch(q, mode), reasoning, long, max_answer_tokens=10)
        assert not r.finished and r.stop_reason == "length" and r.text
        assert V.mode_check(tok, r, mode) == (True, "ok")
        assert not V.mode_check(tok, r, "no_think" if mode == "think" else "think")[0]   # the mode still counts
    r = run(C.with_soft_switch(q, "think"), long, "A force.", max_len=60)
    assert r.stop_reason == "length" and not r.text and V.mode_check(tok, r, "think") == (False, V.TRUNCATED)
    r = run(C.with_soft_switch(q, "no_think"), "", "A force.", stop=tok.bos_id)
    assert r.stop_reason == "eos" and V.mode_check(tok, r, "no_think") == (False, "not finished (eos)")


def test_core_three_task_types(model, tok):
    need("eval_bundle/core.yaml")
    from bonsai.core_eval import evaluate_core
    res = evaluate_core(model, tok, max_per_task=2, tasks_filter=["copa", "winograd", "lambada_openai"])
    assert set(res["tasks"]) == {"copa", "winograd", "lambada_openai"}
    assert all(t["n"] == 2 for t in res["tasks"].values()) and -2 <= res["core"] <= 1
    res = V.TASKS["core"](model, tok, n=1, tasks_filter=["bigbench_qa_wikidata"])
    assert res["n"] == 1


def test_core_crop_keeps_bos(tok):
    from bonsai.core_eval import _crop
    seq = [tok.bos_id] + list(range(10, 30))
    s, span = _crop(seq, (15, 21), 8)
    assert len(s) == 8 and s[0] == tok.bos_id and s[span[0]:span[1]] == seq[15:21][-(span[1] - span[0]):]
