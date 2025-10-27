"""Filter, dedup and decontaminate Bonsai's new raw sources into raw/filtered/<pool>.jsonl (plan sections 10, 17;
DESIGN 1.1). pagoda's pools are symlinked in already filtered and are left alone.

  nemotron_cc      raw/nemotron_cc/<kind>.jsonl (4 synthetic kinds x ~240M chars, scripts.download --only
                   nemotron) -> {"text", "kind"}: exact-duplicate removal (md5 of the stripped text), >= 200
                   chars (~50 tokens; the exact 50-token rule applies at render time), English check (ASCII
                   letters >= 97%, English function words >= 8% of the words), 13-gram decontamination
  smoltalk2_think  SmolTalk2 *_think subsets (scripts.download --only smoltalk2_think) -> {"messages", "subset",
                   "turns", "reasoning_cut"}: conversations with a system prompt (custom_instructions) dropped,
                   English only (also no mixed-language replies), every reply <= 512 teacher tokens, the longest
                   prefix of whole exchanges that fits one 1024-token row (900 teacher tokens), history turns plain,
                   the final assistant turn with its reasoning cut to 128 teacher tokens by chat.budget_thinking
                   (+ THINK_STOP). A multi-turn conversation also yields its first exchange alone (that turn's
                   thinking is otherwise lost to the template's history rule). Qwen / Alibaba mentions dropped
                   (plan 18); 13-gram decontamination; dedup on the user turns

Both pools are fully shuffled before writing (a prefix is a uniform sample, which the cursor-based stage builds
rely on). Counts go to raw/filtered/stats_data.json (DESIGN 1.1: one stats file per producer).
"""

import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import time
from collections import Counter

from bonsai.common import FILTERED_DIR, RAW_DIR, log, read_json, write_json
from bonsai.data import decontam
from bonsai.data.sources import HF_FILES, NEMOTRON_KINDS

STATS_FILE = FILTERED_DIR / "stats_data.json"
MIN_CHARS = 200
MAX_REPLY_TOKENS = 512
ROW_TOKENS = 900             # teacher-token budget of a conversation: one 1024-token row after the subset split
# SmolTalk2 *_think subsets used (sources.HF_FILES["smoltalk2_think"]) -> per-subset cap on a user turn
SMOLTALK2_THINK = {
    "smoltalk_everyday_convs_reasoning_Qwen3_32B_think": 512,
    "multi_turn_reasoning_if_think": 512,
    "table_gpt_Qwen3_32B_think": 384,              # keeps table tasks a minority of the pool
}

_EN_WORDS = frozenset(
    "the of and to in is that for it as with was on be by this are or at from an which have not but they their his "
    "her has had were can will you we been more one all also its there what when who would about into than other "
    "some these them only may if such each most how many your our my she should could".split())
# frequent function words of Spanish / French / German / Portuguese / Italian (rare as English words)
_FOREIGN_WORDS = frozenset(
    "de la que el en los las del por una con une est les des et der und das ist nicht não para com um uma il che di "
    "sono pas sur dans ein eine zu mit als auf je ne".split())
_WORD = re.compile(r"[a-z]+")
_QWEN = re.compile(r"\b(qwen|alibaba|tongyi)\b|通义", re.I)


def english_ok(text, min_ascii=0.97, min_ratio=0.08, min_words=20, max_foreign=0.03):
    """ASCII letters >= min_ascii of all letters; for >= min_words words also English function words >= min_ratio
    and Romance / German function words < max_foreign (catches mixed-language replies)."""
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isascii() for c in letters) / len(letters) < min_ascii:
        return False
    words = _WORD.findall(text.lower())
    if len(words) < min_words:
        return True
    return sum(w in _EN_WORDS for w in words) / len(words) >= min_ratio and \
        sum(w in _FOREIGN_WORDS for w in words) / len(words) < max_foreign


def norm_key(text):
    return " ".join(text.lower().split())


# ----------------------------------------------------------------- decontamination workers
def _init_worker():
    decontam.build_index()


def _decontam_chunk(recs):
    kept, n_bad = [], 0
    for r in recs:
        if decontam.is_contaminated(r.pop("_text")):
            n_bad += 1
        else:
            kept.append(r)
    return kept, n_bad


def write_decontaminated(pool, name, recs, stats, seed):
    """Shuffle `recs` (each with a '_text' field), drop contaminated ones, write raw/filtered/<name>.jsonl."""
    random.Random(seed).shuffle(recs)
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    path = FILTERED_DIR / f"{name}.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    n = 0
    with open(tmp, "w") as f:
        chunks = [recs[i:i + 2000] for i in range(0, len(recs), 2000)]
        for kept, n_bad in pool.imap(_decontam_chunk, chunks):
            stats["decontam_checked"] += len(kept) + n_bad
            stats["contaminated"] += n_bad
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    os.replace(tmp, path)
    return n


# ----------------------------------------------------------------- nemotron_cc
def prep_nemotron(pool, stats):
    seen, recs = set(), []
    for kind in NEMOTRON_KINDS:
        src = RAW_DIR / "nemotron_cc" / f"{kind}.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"{src} missing: run `python -m scripts.download --only nemotron`")
        with open(src) as f:
            for line in f:
                text = json.loads(line)["text"].strip()
                stats["in"] += 1
                stats[f"in_{kind}"] += 1
                h = hashlib.md5(text.encode()).digest()
                if h in seen:
                    stats["exact_dup"] += 1
                    continue
                seen.add(h)
                if len(text) < MIN_CHARS:
                    stats["short"] += 1
                    continue
                if not english_ok(text):
                    stats["non_english"] += 1
                    continue
                recs.append({"text": text, "kind": kind, "_text": text})
    stats["chars"] = sum(len(r["text"]) for r in recs)
    n = write_decontaminated(pool, "nemotron_cc", recs, stats, seed=21)
    return {"nemotron_cc": n}


# ----------------------------------------------------------------- smoltalk2_think
def budget_thinking_batch(qwen, texts, budget=None):
    """chat.budget_thinking for many texts with one batched encode: [(text, was_cut)]."""
    from bonsai.chat import THINK_BUDGET, THINK_STOP
    budget = budget or THINK_BUDGET
    out = []
    for text, ids in zip(texts, qwen.encode_batch(texts)):
        out.append((text, False) if len(ids) <= budget else (qwen.decode(ids[:budget]).rstrip() + THINK_STOP, True))
    return out


def _exchanges(msgs):
    """[(user, reasoning|None, answer)] of a strictly alternating user/assistant conversation, or None."""
    from bonsai.chat import split_reasoning
    if len(msgs) < 2 or len(msgs) % 2:
        return None
    out = []
    for i in range(0, len(msgs), 2):
        u, a = msgs[i], msgs[i + 1]
        if u["role"] != "user" or a["role"] != "assistant" or not u["content"].strip():
            return None
        reasoning, answer = split_reasoning({"content": a["content"]})
        if not answer.strip():
            return None
        out.append((u["content"].strip(), reasoning, answer.strip()))
    return out


def prep_smoltalk2_think(pool, stats):
    import pyarrow.parquet as pq

    from bonsai.tokenizer import QwenTokenizer

    qwen = QwenTokenizer()
    files = HF_FILES["smoltalk2_think"][2]
    recs, seen = [], set()
    for rel in files:
        path = RAW_DIR / "smoltalk2_think" / rel
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `python -m scripts.download --only smoltalk2_think`")
        subset = re.sub(r"-\d{5}-of-\d{5}\.parquet$", "", path.name)
        max_user = SMOLTALK2_THINK[subset]
        sst = Counter()
        cands = []
        for r in pq.read_table(path, columns=["messages", "chat_template_kwargs"]).to_pylist():
            sst["in"] += 1
            msgs = r["messages"]
            if (r.get("chat_template_kwargs") or {}).get("custom_instructions") or \
                    any(m["role"] == "system" for m in msgs):
                sst["system_prompt"] += 1
                continue
            ex = _exchanges(msgs)
            if ex is None:
                sst["malformed"] += 1
                continue
            if not english_ok("\n".join(u + "\n" + a for u, _, a in ex)):
                sst["non_english"] += 1
                continue
            if any(_QWEN.search(a + " " + (rz or "")) for _, rz, a in ex):
                sst["qwen_mention"] += 1
                continue
            cands.append(ex)
        # token lengths of every user turn, answer and reasoning (one batched encode per file)
        flat = [x for ex in cands for u, rz, a in ex for x in (u, a, rz or "")]
        lens = iter(len(ids) for ids in qwen.encode_batch(flat))
        for ex in cands:
            L = [(next(lens), next(lens), next(lens)) for _ in ex]
            keep, used = 0, 0
            for k, (lu, la, lr) in enumerate(L):
                if lu > max_user or la > MAX_REPLY_TOKENS:
                    break
                if used + lu + la + min(lr, 150) + 12 > ROW_TOKENS:   # the last turn's thinking is <= ~150
                    break
                used += lu + la + 10
                keep = k + 1
            while keep and not (ex[keep - 1][1] or "").strip():   # the final turn must carry thinking
                keep -= 1
            if keep == 0:
                sst["too_long_or_no_reasoning"] += 1
                continue
            # the whole conversation (thinking on its last turn), plus its first exchange alone so the first
            # turn's thinking is used too (its text is then seen twice: once as history)
            for part in [ex[:keep]] + ([ex[:1]] if keep >= 2 and (ex[0][1] or "").strip() else []):
                key = norm_key(" || ".join(u for u, _, _ in part))
                if key in seen:
                    sst["dup_prompt"] += 1
                    continue
                seen.add(key)
                sst["kept"] += 1
                sst[f"turns_{len(part)}"] += 1
                recs.append((subset, part))
        stats.update({f"{subset}.{k}": v for k, v in sst.items()})
        log(f"  smoltalk2_think {subset}: {dict(sst)}")
    cut = budget_thinking_batch(qwen, [ex[-1][1].strip("\n") for _, ex in recs])
    out = []
    for (subset, ex), (reasoning, was_cut) in zip(recs, cut):
        msgs = []
        for u, _, a in ex[:-1]:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        msgs += [{"role": "user", "content": ex[-1][0]},
                 {"role": "assistant", "content": ex[-1][2], "reasoning_content": reasoning}]
        stats["reasoning_cut"] += int(was_cut)
        out.append({"messages": msgs, "subset": subset, "turns": len(ex), "reasoning_cut": was_cut,
                    "_text": "\n".join(m["content"] for m in msgs) + "\n" + reasoning})
    stats["in"] = sum(v for k, v in stats.items() if k.endswith(".in"))
    n = write_decontaminated(pool, "smoltalk2_think", out, stats, seed=22)
    return {"smoltalk2_think": n}


PREPS = {"nemotron_cc": prep_nemotron, "smoltalk2_think": prep_smoltalk2_think}


def run(only=None, workers=None):
    all_stats = read_json(STATS_FILE) if STATS_FILE.exists() else {}
    with mp.get_context("spawn").Pool(workers or max(1, (os.cpu_count() or 4) - 2), initializer=_init_worker) as pool:
        for name, fn in PREPS.items():
            if only and name not in only:
                continue
            t0 = time.time()
            log(f"prepare {name} ...")
            counter = Counter()
            written = fn(pool, counter)
            entry = {"pools": written, **dict(counter)}
            if counter.get("in"):
                entry["contaminated_frac"] = round(counter.get("contaminated", 0) / max(counter["decontam_checked"], 1), 6)
                entry["kept_frac"] = round(sum(written.values()) / counter["in"], 6)
            entry["seconds"] = round(time.time() - t0)
            all_stats[name] = entry
            write_json(STATS_FILE, all_stats)
            log(f"prepare {name}: {entry}")
    return all_stats
