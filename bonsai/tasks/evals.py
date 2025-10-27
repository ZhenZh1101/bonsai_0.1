"""Benchmarks of plan section 15. Test data is local (raw/, also used for decontamination); prompts in student ids
with BOS first, chat prompts rendered with the Qwen3 template (bonsai.engine).

  hellaswag        base model: 10-shot completion log-likelihood, acc_norm (per-character), as pagoda / CORE
  arc_easy, arc_challenge, mmlu
                   chat format, non-thinking (enable_thinking=False prefills the empty think block), then
                   "Answer:" and the fp32 logits of " A".." E" (the S2 mc_chat format); with the predicted-letter
                   histogram and the always-same-letter baselines (as pagoda)
  gsm8k_tool       non-thinking + the Python tool (tools=chat.TOOLS, at most 2 calls), greedy; the number after
                   the last "Answer:" compared numerically with the gold answer. items=[{"question", "answer"}]
                   replaces the GSM8K test set (checkpoint selection must not see the test set)
  gsm8k_think      thinking mode (budget 128, then chat.THINK_STOP is forced), no tools, greedy
  fld              FLD.v2 test, 2-4 hops (bonsai.formal.fld), thinking mode, "Answer: PROVED|DISPROVED|UNKNOWN"
  mode_switch      prompts of prompts/fixed_prompts.json under /think, /no_think and enable_thinking=False, one
                   third each; split "eval" = the frozen reported set, "dev" = disjoint prompts for gating
                   checkpoints (on-policy format check). Pass = the think block is non-empty / empty / not
                   reopened after the prefill, and the answer is non-empty, has no think tags and did not end
                   with <|endoftext|>. Length is scored apart: a reply cut by the answer cap (400 tokens, the
                   SFT cap) that obeys the mode passes; finished_frac, truncated and strict_accuracy (pass and
                   ended with <|im_end|>) report it
  core             CORE (bonsai.core_eval), base model

Every task: TASKS[name](model, tok, n=0, log=None, **kw) -> dict with "accuracy", "n", "samples" (+ task
extras); n > 0 evaluates a fixed random subset (seed 0, the same items at every stage).
"""

import json
import random
import re
import time

import numpy as np
import pyarrow.parquet as pq

from bonsai import chat as C
from bonsai.common import PROMPT_DIR, RAW_DIR
from bonsai.engine import chat_batch, chat_prompt, final_logits, span_logprobs
from bonsai.sandbox import extract_answer, numbers_equal, parse_number

N_SAMPLES = 8


def _subset(items, n, seed=0):
    if n and len(items) > n:
        items = random.Random(seed).sample(items, n)
    return items


def _progress(log, name, done, total, t0, extra=""):
    if log:
        log(f"  {name} {done}/{total} {extra}({time.time() - t0:.0f}s)")


def _mc_format():
    try:
        from bonsai.data.render import LETTERS, mc_prompt
    except ImportError:  # same format as pagoda's render.py
        LETTERS = "ABCDE"

        def mc_prompt(question, choices):
            lines = [f"Question: {question.strip()}"]
            return "\n".join(lines + [f"{LETTERS[i]}. {str(c).strip()}" for i, c in enumerate(choices)])
    return LETTERS, mc_prompt


# ------------------------------------------------------------- multiple choice (chat, non-thinking)


def load_mc(task):
    if task in ("arc_easy", "arc_challenge"):
        sub = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
        rows = pq.read_table(RAW_DIR / f"arc/{sub}/test-00000-of-00001.parquet").to_pylist()
        return [{"question": r["question"], "choices": r["choices"]["text"],
                 "answer": r["choices"]["label"].index(r["answerKey"])} for r in rows
                if r["answerKey"] in r["choices"]["label"]]
    if task == "mmlu":
        rows = pq.read_table(RAW_DIR / "mmlu/all/test-00000-of-00001.parquet").to_pylist()
        return [{"question": r["question"], "choices": r["choices"], "answer": int(r["answer"]),
                 "subject": r["subject"]} for r in rows]
    raise ValueError(task)


def eval_mc(model, tok, task, n=0, batch=16, log=None, **kw):
    items = _subset(load_mc(task), n)
    letters, mc_prompt = _mc_format()
    letter_ids = []
    for L in letters:
        ids = tok.encode(" " + L)
        assert len(ids) == 1, f"' {L}' is not one token"
        letter_ids.append(ids[0])
    tail = tok.encode("Answer:")
    correct, hist, samples = 0, {}, []
    max_len = model.cfg.seq_len
    t0 = time.time()
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        seqs = []
        for it in chunk:
            ids, _ = chat_prompt(tok, [{"role": "user", "content": mc_prompt(it["question"], it["choices"])}],
                                 enable_thinking=False)
            ids = ids + tail
            seqs.append(ids if len(ids) <= max_len else ids[:1] + ids[len(ids) - max_len + 1:])   # keep BOS
        lg = final_logits(model, seqs, cols=letter_ids)   # fp32: no bf16 ties resolved towards "A"
        for j, it in enumerate(chunk):
            pred = int(np.argmax(lg[j, :len(it["choices"])]))
            correct += pred == it["answer"]
            hist[letters[pred]] = hist.get(letters[pred], 0) + 1
            if len(samples) < N_SAMPLES:
                samples.append({"question": it["question"], "choices": it["choices"], "gold": letters[it["answer"]],
                                "pred": letters[pred]})
        if (i // batch) % 50 == 49:
            _progress(log, task, i + len(chunk), len(items), t0, f"acc {correct / (i + len(chunk)):.3f} ")
    m = max(len(items), 1)
    # a model that always answers one letter: tells skill apart from a letter prior
    always = {L: sum(it["answer"] == k for it in items) / m for k, L in enumerate(letters)
              if any(len(it["choices"]) > k for it in items)}
    top = max(hist, key=hist.get) if hist else letters[0]
    return {"accuracy": correct / m, "n": len(items), "predicted_letters": hist,
            "always_most_predicted_accuracy": always.get(top, 0.0), "always_letter_accuracy": always,
            "best_single_letter_accuracy": max(always.values()) if always else 0.0, "samples": samples}


# ------------------------------------------------------------- hellaswag (base model, completion scoring)


def _hs_pre(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def load_hellaswag(split="validation"):
    rows = pq.read_table(RAW_DIR / f"hellaswag/data/{split}-00000-of-00001.parquet").to_pylist()
    out = []
    for r in rows:
        ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
        out.append({"query": _hs_pre(r["activity_label"] + ": " + ctx),
                    "endings": [_hs_pre(e) for e in r["endings"]], "label": int(r["label"]) if r["label"] else -1})
    return out


def eval_hellaswag(model, tok, n=0, shots=10, batch=8, log=None, **kw):
    items = _subset(load_hellaswag("validation"), n)
    train = load_hellaswag("train") if shots else []
    max_len = model.cfg.seq_len
    correct, samples = 0, []
    t0 = time.time()
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        seqs, spans = [], []
        for idx, it in enumerate(chunk, start=i):
            ex = random.Random(1234 + idx).sample(train, shots) if shots else []
            cont = [tok.encode(" " + e) for e in it["endings"]]
            longest = max(len(c) for c in cont)
            for k in range(len(ex), -1, -1):  # drop shots until the longest ending fits
                prefix = "".join(f"{e['query']} {e['endings'][e['label']]}\n\n" for e in ex[:k])
                ctx = tok.encode(prefix + it["query"], bos=True)
                if len(ctx) + longest <= max_len + 1:
                    break
            if len(ctx) + longest > max_len + 1:  # crop the query from the left, keep BOS
                ctx = ctx[:1] + ctx[len(ctx) - (max_len - longest):]
            for c in cont:
                seqs.append(ctx + c)
                spans.append((len(ctx), len(ctx) + len(c)))
        res = span_logprobs(model, seqs, spans)
        k = 0
        for it in chunk:
            m = len(it["endings"])
            scores = [float(res[k + j][0].sum()) / max(len(it["endings"][j]), 1) for j in range(m)]  # acc_norm
            pred = int(np.argmax(scores))
            correct += pred == it["label"]
            if len(samples) < N_SAMPLES:
                samples.append({"query": it["query"], "gold": it["endings"][it["label"]],
                                "pred": it["endings"][pred], "correct": pred == it["label"]})
            k += m
        if (i // batch) % 100 == 99:
            _progress(log, "hellaswag", i + len(chunk), len(items), t0, f"acc_norm {correct / (i + len(chunk)):.3f} ")
    acc = correct / max(len(items), 1)
    return {"accuracy": acc, "acc_norm": acc, "n": len(items), "shots": shots, "samples": samples}


# ------------------------------------------------------------- generation tasks


def _chat_eval(model, tok, name, items, golds, judge, log, batch, **chat_kw):
    """Run chat_batch over items in chunks; judge(result, gold) -> (correct, pred)."""
    results, t0 = [], time.time()
    correct = 0
    for i in range(0, len(items), batch):
        rs = chat_batch(model, tok, items[i:i + batch], **chat_kw)
        for r, g in zip(rs, golds[i:i + batch]):
            ok, pred = judge(r, g)
            correct += ok
            results.append((r, g, ok, pred))
        _progress(log, name, len(results), len(items), t0, f"acc {correct / len(results):.3f} ")
    return results


def _gen_summary(results, prompts):
    n = max(len(results), 1)
    samples = [{"prompt": p, "gold": g, "pred": pred, "correct": bool(ok), "completion": r.text,
                "reasoning": r.reasoning, "tool_calls": r.tool_calls, "stop_reason": r.stop_reason}
               for (r, g, ok, pred), p in list(zip(results, prompts))[:N_SAMPLES]]
    return {"accuracy": sum(ok for _, _, ok, _ in results) / n, "n": len(results),
            "finished_frac": sum(r.finished for r, *_ in results) / n,
            "tool_use_frac": sum(r.n_tool_calls > 0 for r, *_ in results) / n,
            "mean_tool_calls": sum(r.n_tool_calls for r, *_ in results) / n,
            "hit_budget_frac": sum(r.hit_budget for r, *_ in results) / n,
            "mean_think_tokens": sum(r.think_tokens for r, *_ in results) / n,
            "samples": samples}


def load_gsm8k(split="test"):
    rows = pq.read_table(RAW_DIR / f"gsm8k/main/{split}-00000-of-00001.parquet").to_pylist()
    return [{"question": r["question"], "gold": parse_number(r["answer"].split("####")[-1])} for r in rows]


def gold_number(answer):
    """A gold answer as a float: a number, "1,234", "$5", or a GSM8K solution ending in "#### x"."""
    if isinstance(answer, (int, float)) and not isinstance(answer, bool):
        return float(answer)
    s = str(answer)
    if "####" in s:
        s = s.split("####")[-1]
    v = parse_number(s)
    return v if v is not None else extract_answer(s)


def _math_items(items):
    """GSM8K test set, or the caller's [{"question": str, "answer": str | number}] (e.g. held-out train items)."""
    if items is None:
        return load_gsm8k()
    out = [{"question": it["question"], "gold": gold_number(it["answer"])} for it in items]
    bad = sum(it["gold"] is None for it in out)
    if bad:
        raise ValueError(f"{bad} of {len(out)} items have no numeric answer")
    return out


def _judge_number(r, gold):
    pred = extract_answer(r.text)
    return numbers_equal(pred, gold), pred


def eval_gsm8k_tool(model, tok, n=0, items=None, batch=32, log=None, max_answer_tokens=300, **kw):
    items = _subset(_math_items(items), n)
    msgs = [[{"role": "user", "content": it["question"]}] for it in items]
    res = _chat_eval(model, tok, "gsm8k_tool", msgs, [it["gold"] for it in items], _judge_number, log, batch,
                     enable_thinking=False, tools=C.TOOLS, max_tool_calls=2, max_answer_tokens=max_answer_tokens)
    return _gen_summary(res, [it["question"] for it in items])


def eval_gsm8k_think(model, tok, n=0, items=None, batch=32, log=None, max_answer_tokens=300,
                     think_budget=C.THINK_BUDGET, **kw):
    items = _subset(_math_items(items), n)
    msgs = [[{"role": "user", "content": it["question"]}] for it in items]
    res = _chat_eval(model, tok, "gsm8k_think", msgs, [it["gold"] for it in items], _judge_number, log, batch,
                     enable_thinking=True, tools=None, think_budget=think_budget, max_answer_tokens=max_answer_tokens)
    return _gen_summary(res, [it["question"] for it in items])


_FLD_RE = re.compile(r"Answer:\s*\**\s*(PROVED|DISPROVED|UNKNOWN)\b", re.I)


def _fld_label(text):
    hits = _FLD_RE.findall(text or "")
    return hits[-1].upper() if hits else None


def eval_fld(model, tok, n=0, batch=32, log=None, depths=(2, 3, 4), max_answer_tokens=64,
             think_budget=C.THINK_BUDGET, **kw):
    from bonsai.formal import fld as F
    items = _subset(F.load_fld_test(depths=tuple(depths)), n)
    users = [it["user"] if "user" in it else F.fld_prompt(it) for it in items]
    golds = [it.get("label") or _fld_label(it["answer"] if "answer" in it else F.fld_answer(it)) for it in items]
    parse = getattr(F, "parse_answer", _fld_label)

    def judge(r, gold):
        pred = parse(r.text)
        return pred == gold, pred

    res = _chat_eval(model, tok, "fld", [[{"role": "user", "content": u}] for u in users], golds, judge, log, batch,
                     enable_thinking=True, tools=None, think_budget=think_budget, max_answer_tokens=max_answer_tokens)
    out = _gen_summary(res, users)
    by_depth = {}
    for it, (_, _, ok, _) in zip(items, res):
        d = by_depth.setdefault(str(it.get("depth")), [0, 0])
        d[0] += ok
        d[1] += 1
    out["by_depth"] = {k: {"accuracy": a / b, "n": b} for k, (a, b) in sorted(by_depth.items())}
    out["gold_labels"] = {L: golds.count(L) for L in sorted(set(golds), key=str)}
    return out


# ------------------------------------------------------------- mode switching

MODES = ("think", "no_think", "prefilled")
SPLITS = {"eval": "mode_switch_eval", "dev": "mode_switch_dev"}


def mode_switch_prompts(split="eval"):
    return json.loads((PROMPT_DIR / "fixed_prompts.json").read_text())[SPLITS[split]]


def mode_switch_items(n=0, split="eval"):
    """(prompt, mode) pairs, the modes cycling over the prompts of `split`: "eval" = the frozen reported prompts, 100
    items by default (plan 15); "dev" = disjoint prompts of the same style for checkpoint gating, every prompt in
    every mode by default."""
    prompts = mode_switch_prompts(split)
    total = n or (100 if split == "eval" else len(MODES) * len(prompts))
    return [(prompts[(i // len(MODES)) % len(prompts)], MODES[i % len(MODES)]) for i in range(total)]


TRUNCATED = "not finished (length)"   # the one failure caused by the length cap (no answer before it)


def mode_check(tok, r, mode):
    """(pass, why) for one ChatResult of the mode-switch test: the mode and the format, not the length. A reply cut
    by the answer cap passes if what it produced obeys the mode (reported apart: finished_frac / truncated); one cut
    before any answer fails with TRUNCATED, so a caller can leave it out; <|endoftext|> instead of <|im_end|>
    fails ("not finished (eos)")."""
    last = next(m for m in reversed(r.messages) if m["role"] == "assistant")
    reasoning, content = last.get("reasoning_content"), last.get("content", "")
    gen = [t for t, s in zip(r.ids, r.sampled) if s]
    if not r.finished and r.stop_reason != "length":
        return False, f"not finished ({r.stop_reason})"
    if not content.strip():
        return (False, TRUNCATED) if r.stop_reason == "length" else (False, "empty answer")
    if "<think>" in content or "</think>" in content:
        return False, "think tag in the answer"
    if mode == "think" and not (reasoning or "").strip():
        return False, "think block missing" if reasoning is None else "empty think block"
    if mode == "no_think" and (reasoning is None or reasoning.strip()):
        return False, "think block missing" if reasoning is None else "non-empty think block"
    if mode == "prefilled" and tok.special("<think>") in gen:
        return False, "reopened <think> after the prefill"
    return True, "ok"


def eval_mode_switch(model, tok, n=0, split="eval", batch=34, log=None, max_answer_tokens=400,
                     think_budget=C.THINK_BUDGET, **kw):
    """Mode compliance of split "eval" (reported) or "dev" (checkpoint gating). Extras: by_mode, failures {why:
    count} (the failed items; TRUNCATED is the only length-caused key), truncated (replies cut by the cap, passed
    or not), strict_accuracy (passed and ended with <|im_end|>), finished_frac."""
    pairs = mode_switch_items(n, split)
    items = []
    for p, mode in pairs:
        if mode == "prefilled":
            items.append({"messages": [{"role": "user", "content": p}], "enable_thinking": False})
        else:
            items.append({"messages": [{"role": "user", "content": C.with_soft_switch(p, mode)}]})
    golds = [m for _, m in pairs]

    def judge(r, mode):
        ok, why = mode_check(tok, r, mode)
        return ok, why

    res = _chat_eval(model, tok, "mode_switch", items, golds, judge, log, batch, think_budget=think_budget,
                     max_answer_tokens=max_answer_tokens)
    out = _gen_summary(res, [f"[{m}] {p}" for p, m in pairs])
    out["split"] = split
    out["by_mode"] = {}
    for mode in MODES:
        oks = [ok for (_, g, ok, _) in res if g == mode]
        out["by_mode"][mode] = {"accuracy": sum(oks) / max(len(oks), 1), "n": len(oks)}
    fails = {}
    for _, _, ok, why in res:
        if not ok:
            fails[why] = fails.get(why, 0) + 1
    out["failures"] = fails
    out["truncated"] = sum(r.stop_reason == "length" for r, *_ in res)
    out["strict_accuracy"] = sum(ok and r.finished for r, _, ok, _ in res) / max(len(res), 1)
    return out


# ------------------------------------------------------------- registry


def eval_core(model, tok, n=0, log=None, tasks_filter=None, **kw):
    from bonsai.core_eval import evaluate_core
    res = evaluate_core(model, tok, max_per_task=n or 500, tasks_filter=tasks_filter, log=log)
    return {"accuracy": res["core"], "core": res["core"], "n": sum(t["n"] for t in res["tasks"].values()),
            "tasks": res["tasks"], "samples": []}


TASKS = {
    "hellaswag": eval_hellaswag,
    "arc_easy": lambda m, t, **kw: eval_mc(m, t, "arc_easy", **kw),
    "arc_challenge": lambda m, t, **kw: eval_mc(m, t, "arc_challenge", **kw),
    "mmlu": lambda m, t, **kw: eval_mc(m, t, "mmlu", **kw),
    "gsm8k_tool": eval_gsm8k_tool,
    "gsm8k_think": eval_gsm8k_think,
    "fld": eval_fld,
    "mode_switch": eval_mode_switch,
    "core": eval_core,
}
CHAT_TASKS = ["arc_easy", "arc_challenge", "mmlu", "gsm8k_tool", "gsm8k_think", "fld", "mode_switch"]
BASE_TASKS = ["core", "hellaswag"]

# plan section 15: (3-day Bonsai-0.1-mini, 5-day Bonsai-0.1) expected ranges, in %
EXPECTED = {
    "hellaswag": ((27, 29), (28, 31)), "arc_easy": ((36, 46), (40, 50)), "arc_challenge": ((24, 28), (25, 30)),
    "mmlu": ((25, 28), (26, 29)), "gsm8k_tool": ((15, 28), (20, 35)), "gsm8k_think": ((3, 8), (5, 12)),
    "fld": ((55, 75), (60, 80)), "mode_switch": ((100, 100), (100, 100)),
}
