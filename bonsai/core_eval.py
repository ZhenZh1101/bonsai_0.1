"""CORE metric (DCLM / nanochat eval bundle) for base checkpoints, plan section 15; ported from pagoda.

22 in-context tasks from raw/eval_bundle/core.yaml, three task types:
  multiple_choice  same context, different continuations -> lowest mean loss over the choice tokens
  schema           different contexts, same continuation  -> lowest mean loss over the continuation
  language_modeling                                       -> greedy decode must reproduce the continuation
Accuracy is centered by each task's random baseline; CORE = mean centered accuracy. Same few-shot sampling and
prompt templates as nanochat. Every prompt starts with BOS = tok.bos_id (<|endoftext|>, the student's document
start); a prompt longer than the context is cropped from the left behind the BOS. The head only runs on the
positions that are scored (engine.span_logprobs), so the 65k+ vocabulary costs no (B, T, V) logits, and
the requests of several items share one forward (short zero-shot items are otherwise all overhead).
"""

import csv
import json
import random

import numpy as np

from bonsai.common import RAW_DIR, load_yaml
from bonsai.engine import span_logprobs

BUNDLE = RAW_DIR / "eval_bundle"


def _fewshot(data, idx, k):
    if k == 0:
        return []
    rng = random.Random(1234 + idx)
    pool = [i for i in range(len(data)) if i != idx]
    return [data[i] for i in rng.sample(pool, k)]


def render(item, task, shots):
    d = task.get("continuation_delimiter", " ")
    t = task["icl_task_type"]
    if t == "multiple_choice":
        prefix = "".join(f"{e['query']}{d}{e['choices'][e['gold']]}\n\n" for e in shots)
        return [f"{prefix}{item['query']}{d}{c}" for c in item["choices"]]
    if t == "schema":
        prefix = "".join(f"{e['context_options'][e['gold']]}{d}{e['continuation']}\n\n" for e in shots)
        return [f"{prefix}{c}{d}{item['continuation']}" for c in item["context_options"]]
    if t == "language_modeling":
        prefix = "".join(f"{e['context'].strip()}{d}{e['continuation']}\n\n" for e in shots)
        without = f"{prefix}{item['context'].strip()}{d}"
        return [without.strip(), without + item["continuation"]]
    raise ValueError(t)


def _common_prefix(seqs):
    n = min(len(s) for s in seqs)
    for i in range(n):
        if any(s[i] != seqs[0][i] for s in seqs):
            return i
    return n


def _common_suffix(seqs):
    n = min(len(s) for s in seqs)
    for i in range(1, n + 1):
        if any(s[-i] != seqs[0][-i] for s in seqs):
            return i - 1
    return n


def _crop(seq, span, max_len):
    """Keep BOS + the last max_len - 1 tokens; shift the span with them."""
    c = len(seq) - max_len
    if c <= 0:
        return seq, span
    return seq[:1] + seq[1 + c:], (max(span[0] - c, 1), span[1] - c)


def _requests(tok, item, task, shots, max_len):
    """(sequences, spans) to score for one item; spans are (start, end) token ranges, possibly empty."""
    seqs = [tok.encode(p, bos=True) for p in render(item, task, shots)]
    t = task["icl_task_type"]
    if t == "language_modeling":
        ctx, full = seqs
        return [_crop(full, (len(ctx), len(full)), max_len)]
    if t == "multiple_choice":
        p = _common_prefix(seqs)
        spans = [(max(p, 1), len(s)) for s in seqs]
    else:  # schema
        q = _common_suffix(seqs)
        spans = [(len(s) - q, len(s)) for s in seqs]
    return [_crop(s, sp, max_len) for s, sp in zip(seqs, spans)]


def _decide(task, item, scored):
    """scored: (log-probs, greedy predictions, targets) per request, None for an empty span."""
    if task["icl_task_type"] == "language_modeling":
        r = scored[0]
        return r is not None and bool(np.all(r[1] == r[2]))
    losses = [float("inf") if r is None else -float(r[0].mean()) for r in scored]
    return int(np.argmin(losses)) == item["gold"]


def eval_items(model, tok, items, task, shots, max_len, max_tokens=16384, max_scored=2048):
    """Correct (bool) per item. Requests of several items share one forward: at most `max_tokens` padded tokens
    and `max_scored` scored positions (the (positions, vocab) logits) per forward."""
    reqs = [_requests(tok, it, task, sh, max_len) for it, sh in zip(items, shots)]
    flat = [(i, j, seq, span) for i, rs in enumerate(reqs) for j, (seq, span) in enumerate(rs) if span[1] > span[0]]
    scored = [[None] * len(rs) for rs in reqs]
    k = 0
    while k < len(flat):
        n, longest, pos = 1, len(flat[k][2]), flat[k][3][1] - flat[k][3][0]
        while k + n < len(flat):
            seq, (a, e) = flat[k + n][2], flat[k + n][3]
            if max(longest, len(seq)) * (n + 1) > max_tokens or pos + e - a > max_scored:
                break
            longest, pos, n = max(longest, len(seq)), pos + e - a, n + 1
        part = flat[k:k + n]
        for (i, j, _, _), res in zip(part, span_logprobs(model, [f[2] for f in part], [f[3] for f in part])):
            scored[i][j] = res
        k += n
    return [_decide(task, it, sc) for it, sc in zip(items, scored)]


def eval_item(model, tok, item, task, shots, max_len):
    return eval_items(model, tok, [item], task, [shots], max_len)[0]


def load_tasks():
    cfg = load_yaml(BUNDLE / "core.yaml")
    baselines = {}
    with open(BUNDLE / "eval_meta_data.csv") as f:
        for row in csv.DictReader(f):
            baselines[row["Eval Task"]] = float(row["Random baseline"])
    return cfg["icl_tasks"], baselines


def evaluate_core(model, tok, max_per_task=500, max_len=None, tasks_filter=None, log=None):
    """{"core": mean centered accuracy, "tasks": {label: {accuracy, centered, n}}}; max_per_task 0 = all items."""
    max_len = max_len or model.cfg.seq_len
    tasks, baselines = load_tasks()
    results = {}
    for task in tasks:
        label = task["label"]
        if tasks_filter and label not in tasks_filter:
            continue
        with open(BUNDLE / "eval_data" / task["dataset_uri"]) as f:
            data = [json.loads(line) for line in f if line.strip()]
        k = task["num_fewshot"][0]
        order = list(range(len(data)))
        random.Random(1337).shuffle(order)
        if max_per_task and max_per_task > 0:
            order = order[:max_per_task]
        correct = sum(eval_items(model, tok, [data[i] for i in order], task,
                                 [_fewshot(data, i, k) for i in order], max_len))
        acc = correct / max(len(order), 1)
        rb = baselines.get(label, 0.0) / 100
        results[label] = {"accuracy": acc, "centered": (acc - rb) / (1 - rb), "n": len(order)}
        if log:
            log(f"  core {label}: {acc:.3f} (n {len(order)})")
    core = float(np.mean([r["centered"] for r in results.values()])) if results else 0.0
    return {"core": core, "tasks": results}
