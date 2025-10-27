"""FLD.v2 (hitachi-nlp/FLD.v2, configs default + star) as chat fields, plain-text derivations and the FLD eval
(plan sections 9, 10, 11, 15).

A record -> three fields (DESIGN 1.1 `fld_chat`):
  user       fld_prompt: the facts (fact1..factN), the hypothesis and a one-line question with the answer format
  reasoning  fld_reasoning: the gold proof as short English steps for the think block ("fact3 and fact7 give int1:
             ..."; FLD's assumption steps kept), <= 128 teacher tokens, else None (the sample is dropped from chat;
             gen_formal puts it into the plain-text pool instead). UNKNOWN: the partial proof FLD gives (if any)
             and a closing sentence.
  answer     fld_answer: "Answer: PROVED" | "Answer: DISPROVED" | "Answer: UNKNOWN"
logic.py renders its synthetic deductions through the same format_* functions, so both look alike.

Eval (plan 15: FLD test, 2-4 hops, thinking mode): `load_fld_test(depths)` = the test splits of both configs,
PROVED / DISPROVED rows with proof depth in `depths` and UNKNOWN rows whose original tree depth is in `depths`;
`parse_answer` reads the label after the last "Answer:".
"""

import random
import re

import pyarrow.parquet as pq

from bonsai.chat import THINK_BUDGET
from bonsai.common import RAW_DIR

FLD_DIR = RAW_DIR / "fld_v2"
CONFIGS = {"default": "data", "star": "star"}
LABELS = ("PROVED", "DISPROVED", "UNKNOWN")
QUESTION = ("Based only on these facts, is the hypothesis proved, disproved, or unknown? Reason step by step, then "
            "reply with \"Answer: PROVED\", \"Answer: DISPROVED\" or \"Answer: UNKNOWN\".")
UNKNOWN_END = "Nothing here proves or disproves the hypothesis, so it is unknown."
COLUMNS = ["context", "hypothesis", "proofs", "proof_label", "depth", "original_tree_depth"]
_SENT = re.compile(r"(?:^|\s)sent(\d+): ")
_QWEN = None
_ANSWER = re.compile(r"Answer:\s*\**\s*(PROVED|DISPROVED|UNKNOWN)\b", re.I)


def load_fld(split="train", configs=tuple(CONFIGS)):
    """Rows of FLD.v2 (`split` in train / validation / test) with "config" and a stable "id"."""
    out = []
    for cfg in configs:
        path = next((FLD_DIR / CONFIGS[cfg]).glob(f"{split}-*.parquet"))
        for i, r in enumerate(pq.read_table(path, columns=COLUMNS).to_pylist()):
            out.append({**r, "config": cfg, "id": f"fld-{cfg}-{split}-{i}"})
    return out


def facts(rec):
    """["everything is a Podilymbus ...", ...]: FLD numbers them sent1..sentN in order (true for all 80k rows)."""
    parts = _SENT.split(" " + rec["context"])
    return [parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)]


def format_prompt(fact_list, hypothesis):
    lines = "\n".join(f"fact{i + 1}: {f}" for i, f in enumerate(fact_list))
    return f"Facts:\n{lines}\n\nHypothesis: {hypothesis}\n\n{QUESTION}"


def format_answer(label):
    return f"Answer: {label}"


def format_step(refs, label=None, text=None, final=None):
    """One proof step: refs -> "intN: text", or the closing step (final = PROVED / DISPROVED)."""
    src, s = " and ".join(refs), "s" if len(refs) == 1 else ""
    if final == "PROVED":
        return f"{src} prove{s} the hypothesis."
    if final == "DISPROVED":
        return f"{src} disprove{s} the hypothesis."
    if not refs:
        return f"{label}: {text}"
    return f"{src} give{s} {label}: {text}"


def format_reasoning(steps, label):
    return "\n".join(steps + ([UNKNOWN_END] if label == "UNKNOWN" else []))


def _ref(r):
    r = r.strip().strip("[]")
    return "fact" + r[4:] if r.startswith("sent") else r


def proof_steps(rec):
    """English steps of the first gold proof (empty for UNKNOWN rows without a partial proof)."""
    if not rec["proofs"]:
        return []
    steps = []
    for step in rec["proofs"][0].split(";"):
        step = step.strip()
        if not step:
            continue
        lhs, rhs = (s.strip() for s in step.split("->", 1))
        refs = [] if lhs == "void" else [_ref(r) for r in lhs.split("&")]
        if rhs == "hypothesis":
            steps.append(format_step(refs, final="DISPROVED" if rec["proof_label"] == "DISPROVED" else "PROVED"))
        else:
            label, text = (s.strip() for s in rhs.split(":", 1))
            steps.append(format_step(refs, label, text))
    return steps


def fld_prompt(rec):
    return format_prompt(facts(rec), rec["hypothesis"])


def qwen_tokenizer(qwen=None):
    global _QWEN
    if qwen is not None:
        return qwen
    if _QWEN is None:
        from bonsai.tokenizer import QwenTokenizer
        _QWEN = QwenTokenizer()
    return _QWEN


def within_budget(text, qwen=None, budget=THINK_BUDGET):
    return budget is None or len(qwen_tokenizer(qwen).encode(text)) <= budget


def fld_reasoning(rec, qwen=None, budget=THINK_BUDGET):
    """The proof for the think block, or None when it is longer than `budget` teacher tokens (None: no limit)."""
    text = format_reasoning(proof_steps(rec), rec["proof_label"])
    return text if within_budget(text, qwen, budget) else None


def fld_answer(rec):
    return format_answer(rec["proof_label"])


def fld_depth(rec):
    return rec["depth"] if rec["proof_label"] != "UNKNOWN" else rec["original_tree_depth"]


def fld_text(rec):
    """Plain-text derivation for S1 synth_reasoning (no token limit)."""
    return text_doc(facts(rec), rec["hypothesis"], proof_steps(rec), rec["proof_label"])


def text_doc(fact_list, hypothesis, steps, label):
    lines = "\n".join(f"fact{i + 1}: {f}" for i, f in enumerate(fact_list))
    verdict = {"PROVED": "proved", "DISPROVED": "disproved", "UNKNOWN": "unknown"}[label]
    return (f"Facts:\n{lines}\nHypothesis: {hypothesis}\nProof:\n{format_reasoning(steps, label)}\n"
            f"Conclusion: the hypothesis is {verdict}.")


def chat_record(rec, qwen):
    reasoning = fld_reasoning(rec, qwen)
    if reasoning is None:
        return None
    return {"user": fld_prompt(rec), "reasoning": reasoning, "answer": fld_answer(rec),
            "label": rec["proof_label"], "depth": fld_depth(rec), "config": rec["config"]}


def load_fld_test(depths=(2, 3, 4), configs=tuple(CONFIGS), n=None, seed=0):
    """FLD test items for the eval: {"id", "user", "answer", "label", "depth", "config", "reasoning"}."""
    items = []
    for rec in load_fld("test", configs):
        if fld_depth(rec) not in depths:
            continue
        items.append({"id": rec["id"], "user": fld_prompt(rec), "answer": fld_answer(rec),
                      "label": rec["proof_label"], "depth": fld_depth(rec), "config": rec["config"],
                      "reasoning": fld_reasoning(rec, budget=None)})
    if n is not None and n < len(items):
        items = random.Random(seed).sample(items, n)
    return items


def parse_answer(text):
    """The label after the last "Answer:" in a response, or None."""
    hits = _ANSWER.findall(text or "")
    return hits[-1].upper() if hits else None
