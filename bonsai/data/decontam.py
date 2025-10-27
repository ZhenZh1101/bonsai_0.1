"""13-gram decontamination against every benchmark we report (plan sections 10, 11, 15; ported from pagoda).

Eval texts: GSM8K test, ARC-Easy/Challenge test, HellaSwag val, MMLU test, HumanEval, FLD.v2 test (default
and star). Text is lower-cased and split into [a-z0-9]+ words; any shared 13-word window drops the whole
document (web/text sources) or the sample (QA/chat sources, prompts, teacher outputs).

The n-gram set is built in the parent and inherited by forked workers, so the (per-process randomised) str
hash is identical everywhere.
"""

import re

import pyarrow.parquet as pq

from bonsai.common import RAW_DIR

N = 13
_WORD = re.compile(r"[a-z0-9]+")
_NGRAMS = None


def words(text):
    return _WORD.findall(text.lower())


def ngram_hashes(text, n=N):
    w = words(text)
    return {hash(tuple(w[i:i + n])) for i in range(len(w) - n + 1)}


def eval_texts():
    """(benchmark, text) for every eval item."""
    out = []
    t = pq.read_table(RAW_DIR / "gsm8k/main/test-00000-of-00001.parquet").to_pylist()
    out += [("gsm8k", r["question"] + " " + r["answer"]) for r in t]
    for sub in ("ARC-Easy", "ARC-Challenge"):
        t = pq.read_table(RAW_DIR / f"arc/{sub}/test-00000-of-00001.parquet").to_pylist()
        out += [(sub.lower(), r["question"] + " " + " ".join(r["choices"]["text"])) for r in t]
    t = pq.read_table(RAW_DIR / "hellaswag/data/validation-00000-of-00001.parquet").to_pylist()
    out += [("hellaswag", r["ctx"] + " " + " ".join(r["endings"])) for r in t]
    t = pq.read_table(RAW_DIR / "mmlu/all/test-00000-of-00001.parquet").to_pylist()
    out += [("mmlu", r["question"] + " " + " ".join(r["choices"])) for r in t]
    t = pq.read_table(RAW_DIR / "humaneval/openai_humaneval/test-00000-of-00001.parquet").to_pylist()
    out += [("humaneval", r["prompt"] + "\n" + r["canonical_solution"]) for r in t]
    for p in sorted((RAW_DIR / "fld_v2").glob("*/test-*.parquet")):
        t = pq.read_table(p, columns=["context", "hypothesis"]).to_pylist()
        out += [("fld", r["context"] + " " + r["hypothesis"]) for r in t]
    return out


def build_index():
    global _NGRAMS
    if _NGRAMS is None:
        s = set()
        for _, text in eval_texts():
            s |= ngram_hashes(text)
        _NGRAMS = s
    return _NGRAMS


def is_contaminated(text, index=None):
    index = index if index is not None else build_index()
    w = words(text)
    for i in range(len(w) - N + 1):
        if hash(tuple(w[i:i + N])) in index:
            return True
    return False
