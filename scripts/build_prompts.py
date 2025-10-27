"""Prompt pools for distillation (plan sections 11, 12, 15, 18) -> data/prompts.jsonl, prompts_onpolicy.jsonl,
prompts_heldout.jsonl (+ data/prompts_stats.json).

    python -m scripts.build_prompts                       # configs/offpolicy.yaml, ~100k + 40k + 500 prompts
    python -m scripts.build_prompts --config configs/smoke/offpolicy.yaml

Shares (plan 11, configs/offpolicy.yaml `prompts.sources`): SmolTalk user turns 55% (raw/filtered/sft_chat,
reserved by prompt hash and never in S1/S2; 15% keep their earlier turns as history), explanation-type QA 7.5%
(first SmolTalk turns starting with why / how / what / explain / describe / define) + ARC train 2.5%, GSM8K train
5% + TinyGSM 12.5% + OpenMathInstruct-2 12.5% (math with integer answers; OMI-2 only without LaTeX in the
question and with a short reference solution, i.e. its GSM8K-style part: the competition-style rest mostly fails
the 400-token answer cap and is beyond a 100M student), tool-call demos 5% (TinyGSM with the python tool on). TinyGSM and OMI-2 prompts come from the END of their pools, far beyond what S1 + S2 read.

Overlap with S1 / S2 (DESIGN 1.2: S1 -> S2 -> prompt pool consume shared pools without overlap), checked against
tok/pool_cursors.json when it exists: every source's prompt lines must lie at or after its pool's cursor, unless the
source says `allow_overlap` (same question, new supervision; recorded in the stats). Source options:
`after_cursor` reads only the lines after the cursor; `exclude_seen: [pools]` drops a candidate whose first user
turn / question (normalised) is also a record S1 / S2 read from those pools (SmolTalk duplicates across subsets);
`prefer_unseen` takes candidates whose question S1 / S2 never read first, then lines after the cursor, then the
rest, so held-out prompts (taken first) are unseen. Per source the stats give the prompts below the cursor and
those whose question S1 / S2 read, per split.

Every candidate is dropped if any user turn is an identity question (bonsai.identity.is_identity_prompt, plan 18:
identity is taught by scripted conversations only, and Qwen teachers would answer "I am Qwen"), if it is 13-gram
contaminated (bonsai.data.decontam), non-English, or longer than `max_prompt_tokens` teacher tokens. Each source's
shuffled candidates are split in order into held-out (teacher-KL eval, plan 15), off-policy and on-policy prompts
(plan 12: a fresh resample; a source too small for both reuses off-policy prompts, marked "reused").

Record: {"id", "source", "kind": chat | explain | mc | math, "messages", "answer" (math: number string, mc:
letter, else null), "tools"}.
"""

import argparse
import hashlib
import random
import re
import time
from collections import Counter
from functools import lru_cache
from bonsai.common import CONFIG_DIR, DATA_ROOT, FILTERED_DIR, LOG_DIR, TOK_DIR, load_yaml, log, read_json, \
    write_json, write_jsonl

EXPLAIN_RE = re.compile(r"^(why|how|what|explain|describe|define)\b", re.I)
LETTERS = "ABCDE"
_IDENTITY_FALLBACK = re.compile(
    r"\b(who|what) (are|r) (you|u)\b|\byour name\b|\bwho (made|created|built|trained|developed) (you|u)\b|"
    r"\bare (you|u) (an? )?(ai|bot|human|robot|chatgpt|gpt|qwen|claude|gemini|llama)\b|\bintroduce yourself\b|"
    r"\btell me about yourself\b|\bwhat (model|llm|language model) are you\b|你是谁|你叫什么", re.I)


def mc_prompt(question, choices):
    """The multiple-choice user turn, exactly S2's mc_chat format (bonsai.data.render.mc_prompt)."""
    lines = [f"Question: {question.strip()}"]
    lines += [f"{LETTERS[i]}. {str(c).strip()}" for i, c in enumerate(choices)]
    return "\n".join(lines)


@lru_cache(maxsize=1)
def identity_check():
    try:
        from bonsai.identity import is_identity_prompt
        return is_identity_prompt, "bonsai.identity.is_identity_prompt"
    except ImportError:
        return (lambda t: bool(_IDENTITY_FALLBACK.search(t))), "local regex (bonsai.identity missing)"


def is_identity(text):
    return identity_check()[0](text)


def count_lines(path):
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            n += chunk.count(b"\n")
    return n


def read_pool(pool, start=0, stop=None):
    """(line number, record) of raw/filtered/<pool>.jsonl for lines start <= i < stop."""
    import json
    with open(FILTERED_DIR / f"{pool}.jsonl") as f:
        for i, line in enumerate(f):
            if stop is not None and i >= stop:
                break
            if i >= start and line.strip():
                yield i, json.loads(line)


def stable_frac(key, salt):
    return int(hashlib.md5((salt + "\x00" + key).encode()).hexdigest()[:8], 16) / 2 ** 32


def record_key(rec):
    """Hash of a pool record's first user turn (chat records) or question, whitespace / case normalised."""
    if "messages" in rec:
        t = next((m.get("content") for m in rec["messages"] if m.get("role") == "user"), "")
    else:
        t = rec.get("question", "")
    return hashlib.md5(" ".join(str(t).split()).lower().encode()).hexdigest()


_SEEN = {}


def consumed_keys(pools, cursors):
    """record_key of every record S1 + S2 read (the lines below the cursor) of `pools`."""
    keys = set()
    for p in pools:
        cur = int((cursors or {}).get(p, 0))
        k = (str(FILTERED_DIR / p), cur)
        if k not in _SEEN:
            _SEEN[k] = {record_key(r) for _, r in read_pool(p, 0, cur)} if cur else set()
        keys |= _SEEN[k]
    return keys


def make_prompt(spec, rec, line):
    """(messages, answer) for one pool record under a source spec, or (None, reason)."""
    from bonsai.sandbox import extract_answer, format_number, parse_number
    pool = spec["pool"]
    for field, values in (spec.get("match") or {}).items():
        if rec.get(field) not in values:
            return None, "no_match"
    if "messages" in rec:
        msgs = [{"role": m["role"], "content": m["content"]} for m in rec["messages"]]
        if not msgs or msgs[0]["role"] != "user" or not all(isinstance(m["content"], str) for m in msgs):
            return None, "malformed"
        if "explain" in spec and bool(EXPLAIN_RE.match(msgs[0]["content"].strip())) != spec["explain"]:
            return None, "no_match"
        users = [i for i, m in enumerate(msgs) if m["role"] == "user"]
        k = 0
        if not spec.get("first_turn_only") and len(users) > 1 and \
                stable_frac(f"{pool}:{line}", "multi") < spec.get("multi_turn_frac", 0.0):
            k = users[1 + int(stable_frac(f"{pool}:{line}", "turn") * (len(users) - 1))]
        msgs = msgs[:k + 1]
        if any(m["role"] not in ("user", "assistant") for m in msgs) or msgs[-1]["role"] != "user":
            return None, "malformed"
        return msgs, None
    q = str(rec.get("question", "")).strip()
    if not q:
        return None, "malformed"
    if spec.get("no_latex") and ("$" in q or "\\" in q):
        return None, "latex"
    if spec.get("max_ref_chars") and len(rec.get("cot") or rec.get("script") or "") > spec["max_ref_chars"]:
        return None, "long_reference"
    if pool == "mc":
        if not isinstance(rec.get("answer"), int) or not 0 <= rec["answer"] < len(rec["choices"]) <= 5:
            return None, "malformed"
        return [{"role": "user", "content": mc_prompt(q, rec["choices"])}], LETTERS[rec["answer"]]
    ans = parse_number(rec.get("answer")) if rec.get("answer") is not None else extract_answer(rec.get("cot", ""))
    if ans is None:
        return None, "no_answer"
    if spec.get("integer_answer") and abs(ans - round(ans)) > 1e-9:
        return None, "non_integer_answer"
    return [{"role": "user", "content": q}], format_number(ans)


def build_pools(cfg, log_file=None):
    from bonsai.data.decontam import build_index, is_contaminated
    from bonsai.teacher import non_english
    from bonsai.tokenizer import QwenTokenizer

    t0 = time.time()
    pc = cfg["prompts"]
    rng = random.Random(int(pc.get("seed", 0)))
    totals = {"off": int(pc["total"]), "on": int(pc["onpolicy"]), "held": int(pc["heldout"])}
    specs = pc["sources"]
    need = {name: {k: max(1 if k == "held" else 0, round(float(s["share"]) * v)) for k, v in totals.items()}
            for name, s in specs.items()}
    qwen = QwenTokenizer()
    build_index()
    cursors_file = TOK_DIR / pc.get("cursor_file", "pool_cursors.json")
    cursors = read_json(cursors_file) if cursors_file.exists() else None
    stats = {"identity_check": identity_check()[1], "cursor_check": str(cursors_file) if cursors is not None else
             f"skipped: {cursors_file} does not exist yet (run the S1/S2 builds first to verify)", "sources": {}}
    out = {"off": [], "on": [], "held": []}
    by_pool = {}
    for name, s in specs.items():
        by_pool.setdefault(s["pool"], []).append(name)
    for pool, names in by_pool.items():
        n_lines = count_lines(FILTERED_DIR / f"{pool}.jsonl")
        cur = int(cursors.get(pool, 0)) if cursors is not None else 0
        start = 0
        if any(specs[n].get("from_end") for n in names):
            want = sum(sum(need[n].values()) for n in names)
            head = max(float(specs[n].get("tail_headroom", pc.get("tail_headroom", 3.0))) for n in names)
            start = max(0, n_lines - int(want * head))
            if cursors is not None:
                if any(specs[n].get("allow_overlap") for n in names):  # same question, different supervision
                    stats.setdefault("overlap_allowed", {})[pool] = {"prompt_lines_from": start, "s1_s2_cursor": cur}
                else:
                    assert start >= cur, f"{pool}: prompt lines from {start} overlap S1/S2 (cursor {cur})"
        if all(specs[n].get("after_cursor") for n in names):
            start = max(start, cur)
        recs = list(read_pool(pool, start))
        rng.shuffle(recs)
        taken = set()
        for name in names:
            s, n = specs[name], need[name]
            want = sum(n.values())
            cnt = Counter()
            got = []
            batch = []
            excl = consumed_keys(s.get("exclude_seen") or (), cursors)
            own = consumed_keys([pool], cursors) | excl if s.get("prefer_unseen") or excl else set()
            cands = recs
            if s.get("after_cursor"):
                cands = [(line, rec) for line, rec in recs if line >= cur]
            if s.get("prefer_unseen"):           # unseen questions, then lines after the cursor, then the rest
                cands = sorted(cands, key=lambda x: (x[0] < cur) + (record_key(x[1]) in own))
            seen = set()                         # lines of taken prompts whose question S1 / S2 read

            def flush():
                lens = qwen.encode_batch(["\n".join(m["content"] for m in b[1]) for b in batch])
                for (line, msgs, ans, sub), ln in zip(batch, lens):
                    text = "\n".join(m["content"] for m in msgs)
                    if len(ln) > int(pc.get("max_prompt_tokens", 512)):
                        cnt["too_long"] += 1
                    elif any(is_identity(m["content"]) for m in msgs if m["role"] == "user"):
                        cnt["identity_prompt"] += 1
                    elif non_english(text):
                        cnt["non_english"] += 1
                    elif is_contaminated(text):
                        cnt["contaminated"] += 1
                    elif len(got) < want:
                        got.append({"id": f"{name}-{line}", "source": pool + (f"/{sub}" if sub else ""),
                                    "kind": s["kind"], "messages": msgs, "answer": ans, "tools": bool(s.get("tools"))})
                        taken.add(line)
                batch.clear()

            for line, rec in cands:
                if len(got) >= want:
                    break
                if line in taken:
                    continue
                msgs, ans = make_prompt(s, rec, line)
                cnt["read"] += 1
                if msgs is None:
                    cnt[ans] += 1
                    continue
                if excl and record_key(rec) in excl:
                    cnt["seen_in_s1_s2"] += 1
                    continue
                if own and record_key(rec) in own:
                    seen.add(line)
                batch.append((line, msgs, ans, rec.get("subset") or rec.get("source")))
                if len(batch) >= 2000:
                    flush()
            if batch:
                flush()
            held, off, on = got[:n["held"]], got[n["held"]:n["held"] + n["off"]], got[n["held"] + n["off"]:]
            short = n["on"] - len(on)
            if short > 0 and off:
                on += [{**r, "reused": True} for r in random.Random(f"{name}-reuse").sample(off, min(short, len(off)))]
            out["held"] += held
            out["off"] += off
            out["on"] += on
            lines = {k: [int(r["id"].rsplit("-", 1)[1]) for r in v] for k, v in (("held", held), ("off", off), ("on", on))}
            overlap = {"cursor": cur if cursors is not None else None,
                       "below_cursor": {k: sum(x < cur for x in v) for k, v in lines.items()},
                       "seen_question": {k: sum(x in seen for x in v) for k, v in lines.items()} if own else None}
            stats["sources"][name] = {"pool": pool, "lines_from": start, "pool_lines": n_lines, "wanted": n,
                                      "got": {"held": len(held), "off": len(off), "on": len(on)},
                                      "on_reused": max(0, short) if off else 0, "s1_s2_overlap": overlap, **dict(cnt)}
            log(f"  {name:8s} {pool:12s} held {len(held):5,} off {len(off):6,} on {len(on):6,} "
                f"(wanted {n['off']:,} / {n['on']:,}) {dict(cnt)}; below the S1/S2 cursor {overlap['below_cursor']}"
                + (f", question read by S1/S2 {overlap['seen_question']}" if own else ""), log_file)
            if cursors is not None and not s.get("allow_overlap") and sum(overlap["below_cursor"].values()):
                raise SystemExit(f"{name}: {overlap['below_cursor']} prompts from {pool} lines below the S1/S2 cursor "
                                 f"{cur}: take them after it (from_end / after_cursor) or set allow_overlap")
    paths = {"off": pc["out"], "on": pc["onpolicy_out"], "held": pc["heldout_out"]}
    for k, recs in out.items():
        rng.shuffle(recs)
        write_jsonl(DATA_ROOT / paths[k], recs)
        stats[f"{k}_total"] = len(recs)
        stats[f"{k}_by_kind"] = dict(Counter(r["kind"] + ("+tools" if r["tools"] else "") for r in recs))
    stats["seconds"] = round(time.time() - t0, 1)
    write_json(DATA_ROOT / pc.get("stats_out", "data/prompts_stats.json"), stats)
    log(f"prompts: off-policy {len(out['off']):,} -> {paths['off']}, on-policy {len(out['on']):,} -> "
        f"{paths['on']}, held-out {len(out['held']):,} -> {paths['held']} ({stats['seconds']:.0f}s)", log_file)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_DIR / "offpolicy.yaml"))
    args = ap.parse_args()
    build_pools(load_yaml(args.config), log_file=LOG_DIR / "build_prompts.log")


if __name__ == "__main__":
    main()
