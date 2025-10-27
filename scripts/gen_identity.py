"""Build the Bonsai-0.1 identity data (identity/persona.md, plan section 18).

    python -m scripts.gen_identity check identity/parts/qwen_name_01.jsonl  # validate generated files
    python -m scripts.gen_identity merge                    # parts/*.jsonl -> conversations.jsonl
    python -m scripts.gen_identity llm --per-batch 10       # more conversations from a local model
    python -m scripts.gen_identity pretrain                 # pretrain_bank.yaml -> pretrain.jsonl
    python -m scripts.gen_identity pools                    # -> raw/filtered/identity_{chat,text}.jsonl

The conversations were ported from Pagoda-0.1 (names changed, every Pagoda fact rewritten to Bonsai's; the
`origin` / `edit` fields of each part record say where it came from) plus new ones about the Qwen relation
(parts/qwen_*.jsonl). More can be generated into identity/parts/ by a local model behind an OpenAI-compatible
server (`mlx_lm.server --model mlx-community/Qwen3-8B-4bit`); only conversations passing every check are kept.
Merging checks every record against persona.yaml (says / never says the name per category, no claim to be
another model, no wrong or stale fact, not a near-duplicate of the held-out eval prompts), drops exact
duplicates and caps each opening user turn at a few copies.

conversations.jsonl and pretrain.jsonl keep the template fields ({vocab_size}, {params_m}, {params_mini_m});
`pools` fills them from persona.yaml (or --vocab-size), re-checks, 13-gram decontaminates, shuffles with a fixed
seed (conversations.jsonl is in part-file order, i.e. by category, and the S2 builder reads a pool as a prefix
from its cursor, so a prefix must be a uniform sample) and writes the pools the S2 / SFT builders read, with
counts in raw/filtered/stats_identity.json.

Part record format, one JSON object per line:
    {"category": "denial", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
Categories: name denial creator capabilities multiturn in_task persona edge.
"""

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter

from bonsai.common import FILTERED_DIR, load_yaml, log, write_json, write_jsonl
from bonsai.identity import (CATEGORIES, CONVERSATIONS, IDENTITY_DIR, PRETRAIN_TEXTS, conversation_problems,
                             fact_errors, fill, fill_messages, held_out_prompts, near_duplicate, norm_words,
                             other_model_claims, persona, template_fields)

PARTS_DIR = IDENTITY_DIR / "parts"
MAX_SAME_OPENING = 4     # copies of one exact opening user turn (with different replies) kept
PRETRAIN_TOKENS = 1.1e6  # S2 asks for 0.002 x 4.2e8 = 0.84M; a little more so it never repeats


def read_records(path):
    recs = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as e:
                recs.append({"_bad_json": f"line {n}: {e}"})
    return recs


def cmd_check(args):
    total_bad = 0
    for path in args.files:
        recs = read_records(path)
        cats, bad = Counter(), []
        for i, r in enumerate(recs):
            probs = [r["_bad_json"]] if "_bad_json" in r else conversation_problems(r)
            cats[r.get("category")] += 1
            if probs:
                bad.append((i + 1, probs))
        total_bad += len(bad)
        print(f"{path}: {len(recs)} records, {len(bad)} with problems, by category {dict(cats)}")
        for line_no, probs in bad[:args.show]:
            print(f"  record {line_no}: {'; '.join(probs)}")
    sys.exit(1 if total_bad else 0)


def merge_parts(files):
    """Checked, de-duplicated conversations from part files (template fields kept) + a report."""
    seen, openings = set(), Counter()
    kept, drops, by_cat, by_file, edits = [], Counter(), Counter(), {}, Counter()
    for path in files:
        n_in = n_kept = 0
        for r in read_records(path):
            n_in += 1
            if "_bad_json" in r:
                drops["bad_json"] += 1
                continue
            probs = conversation_problems(r)
            if probs:
                for p in probs:
                    drops[p.split(":")[0]] += 1
                continue
            key = tuple(" ".join(norm_words(m["content"])) or m["content"].strip() for m in r["messages"])
            if key in seen:
                drops["exact duplicate"] += 1
                continue
            if openings[key[0]] >= MAX_SAME_OPENING:
                drops["opening turn over cap"] += 1
                continue
            seen.add(key)
            openings[key[0]] += 1
            kept.append({"category": r["category"], "messages": [{"role": m["role"], "content": m["content"].strip()}
                                                                 for m in r["messages"]],
                         "source": path.stem})
            by_cat[r["category"]] += 1
            edits[r.get("edit", "unknown")] += 1
            n_kept += 1
        by_file[path.name] = {"in": n_in, "kept": n_kept}
    for i, r in enumerate(kept):
        r["id"] = f"identity_{i:05d}"
    turns = Counter(len(r["messages"]) // 2 for r in kept)
    qwen = sum(bool(re.search(r"qwen|alibaba|tongyi|通义|千问|阿里", " ".join(m["content"] for m in r["messages"]
                                                                       if m["role"] == "user"), re.I)) for r in kept)
    report = {"conversations": len(kept), "by_category": dict(by_cat), "turns": dict(sorted(turns.items())),
              "edits": dict(edits), "qwen_conversations": qwen, "dropped": dict(drops), "files": by_file}
    return kept, report


def cmd_merge(args):
    files = [f for f in sorted(PARTS_DIR.glob("*.jsonl")) if not f.name.startswith("pretrain")]
    kept, report = merge_parts(files)
    CONVERSATIONS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    write_json(IDENTITY_DIR / "report.json", report)
    log(f"merged {len(files)} files -> {CONVERSATIONS}: {len(kept)} conversations {report['by_category']} "
        f"({report['qwen_conversations']} about Qwen / Alibaba); edits {report['edits']}; dropped {report['dropped']}")


# ------------------------------------------------- generation with a local model

LLM_STYLES = ["formal", "casual", "terse", "rambling", "curious child", "software developer", "student",
              "teacher", "skeptic", "non-native speaker", "busy parent", "journalist", "retiree", "teenager"]


def llm_prompt(cat, spec, n, rng):
    styles = ", ".join(rng.sample(LLM_STYLES, 3))
    page = fill((IDENTITY_DIR / "persona.md").read_text())
    return (f"{spec['rules']}\n\n--- persona page ---\n{page}\n--- end ---\n\n"
            f"Category: {cat}\nRule: {spec['categories'][cat]['rule']}\n{spec['categories'][cat]['brief']}\n"
            f"In this batch, lean towards users who are: {styles}.\n"
            f"Write {n} new conversations of category \"{cat}\" as JSON Lines, one object per line, and nothing else. "
            f"/no_think")


def chat_completion(base_url, model, prompt, temperature, timeout=600):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": temperature, "max_tokens": 6000}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def parse_jsonl_reply(text, cat):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    out = []
    for line in text.splitlines():
        line = line.strip().strip(",")
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("messages"), list):
            out.append({"category": cat, "messages": rec["messages"], "origin": "llm", "edit": "new"})
    return out


def cmd_llm(args):
    """Ask a local OpenAI-compatible server for more conversations; only valid ones are written."""
    spec = load_yaml(IDENTITY_DIR / "generation_specs.yaml")
    cats = args.categories or list(spec["categories"])
    rng = random.Random(args.seed)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    for cat in cats:
        out_path = PARTS_DIR / f"llm_{cat}_{args.seed:03d}.jsonl"
        kept, stats = [], Counter()
        for b in range(args.batches):
            t0 = time.time()
            try:
                reply = chat_completion(args.base_url, args.model, llm_prompt(cat, spec, args.per_batch, rng),
                                        args.temperature)
            except OSError as e:
                log(f"  {cat} batch {b}: request failed: {e}")
                stats["request_failed"] += 1
                continue
            recs = parse_jsonl_reply(reply, cat)
            stats["parsed"] += len(recs)
            for r in recs:
                if conversation_problems(r):
                    stats["rejected"] += 1
                else:
                    kept.append(r)
            log(f"  {cat} batch {b + 1}/{args.batches}: {len(recs)} parsed, {len(kept)} kept so far, "
                f"{time.time() - t0:.0f}s")
        out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
        log(f"{cat}: {len(kept)} conversations -> {out_path} {dict(stats)}")
    log("now run: python -m scripts.gen_identity merge")


# -------------------------------------------------- pretraining texts (source 3)

SUBJECTS = {"S": (["Bonsai-0.1", "Bonsai", "The model", "It"], [0.25, 0.3, 0.2, 0.25]),
            "s": (["Bonsai-0.1", "Bonsai", "the model", "it"], [0.25, 0.3, 0.2, 0.25])}
QA_MARKERS = [("Q: ", "A: "), ("Question: ", "Answer: "), ("Q. ", "A. ")]
CHAT_SPEAKERS = [("User", "Bonsai"), ("Human", "Bonsai-0.1"), ("Me", "Bonsai"), ("Student", "Bonsai"),
                 ("Visitor", "Bonsai-0.1")]
INTERVIEWERS = ["Interviewer", "Q", "Host", "Reporter"]


def fill_subject(sentence, rng):
    def sub(m):
        names, weights = SUBJECTS[m.group(1)]
        return rng.choices(names, weights)[0]
    return re.sub(r"\{([Ss])\}", sub, sentence)


def compose_doc(bank, genre, rng):
    """One short document: a genre frame around a seeded choice of facts, each in a random paraphrase."""
    order = list(bank["facts"])
    lo, hi = genre.get("n_facts", [4, 8])
    required = [k for k in genre.get("required", []) if k in bank["facts"]]
    others = [k for k in order if k not in required]
    keys = required + rng.sample(others, max(0, min(len(others), rng.randint(lo, hi) - len(required))))
    keys.sort(key=order.index)
    for i in range(len(keys) - 1):  # mostly logical order, with some local variation
        if rng.random() < 0.2:
            keys[i], keys[i + 1] = keys[i + 1], keys[i]
    person, layout = genre.get("person", "third"), genre.get("layout", "prose")
    facts = [bank["facts"][k] for k in keys]
    parts = []
    if genre.get("titles") and rng.random() < 0.7:
        parts.append(rng.choice(genre["titles"]))
    if genre.get("openers"):
        parts.append(fill_subject(rng.choice(genre["openers"]), rng))
    if layout in ("faq", "chat", "interview"):
        if layout == "faq":
            q_mark, a_mark = rng.choice(QA_MARKERS)
        elif layout == "chat":
            q_mark, a_mark = (f"{who}: " for who in rng.choice(CHAT_SPEAKERS))
        else:
            q_mark, a_mark = f"{rng.choice(INTERVIEWERS)}: ", f"{rng.choice(['Bonsai-0.1', 'Bonsai'])}: "
        sep = "\n\n" if layout == "faq" else "\n"
        parts.append(sep.join(f"{q_mark}{rng.choice(f['questions'])}\n{a_mark}{rng.choice(f['first'])}"
                              for f in facts))
    else:
        sents = [fill_subject(rng.choice(f[person]), rng) for f in facts]
        if layout == "bullets":
            parts.append("\n".join(f"- {s}" for s in sents))
        else:
            paras, i = [], 0
            while i < len(sents):
                k = rng.randint(2, 4)
                paras.append(" ".join(sents[i:i + k]))
                i += k
            parts.extend(paras)
    if genre.get("closers") and rng.random() < 0.3:
        parts.append(fill_subject(rng.choice(genre["closers"]), rng))
    return "\n\n".join(parts)


def compose_pretrain(bank, n_tokens, count, seed=7, min_tokens=60, fields=None):
    """Documents (template fields kept) until `count(filled text)` tokens add up to n_tokens."""
    rng = random.Random(seed)
    fields = fields or template_fields()
    genres = bank["genres"]
    docs, seen, tokens, stats = [], set(), 0, Counter()
    # FAQ / chat / interview documents quote these questions verbatim: keep the eval prompts unseen
    held = held_out_prompts()
    for f in bank["facts"].values():
        keep = [q for q in f["questions"] if not any(near_duplicate(q, h) for h in held)]
        stats["bank_questions_held_out"] += len(f["questions"]) - len(keep)
        f["questions"] = keep or ["Can you tell me about yourself?"]
    while tokens < n_tokens:
        stats["composed"] += 1
        if stats["composed"] > 50 * (len(docs) + 100):
            raise SystemExit("bank too small: almost every composed document is a duplicate")
        genre = rng.choice(genres)
        text = compose_doc(bank, genre, rng)
        if text in seen:
            stats["duplicate"] += 1
            continue
        filled = fill(text, fields)
        if other_model_claims(filled) or fact_errors(filled, fields):
            stats["rejected"] += 1
            continue
        n = count(filled) + 1  # + BOS
        if n < min_tokens:
            stats["too_short"] += 1
            continue
        seen.add(text)
        docs.append({"id": f"identity_text_{len(docs):05d}", "genre": genre["name"], "text": text})
        tokens += n
    return docs, tokens, stats


def cmd_pretrain(args):
    from bonsai.tokenizer import QwenTokenizer, Tokenizer
    try:
        tok = Tokenizer.load()   # student ids, as in the shards
    except FileNotFoundError:
        tok = QwenTokenizer()
    bank = load_yaml(IDENTITY_DIR / "pretrain_bank.yaml")
    docs, tokens, stats = compose_pretrain(bank, args.tokens, lambda t: len(tok.encode(t)), args.seed, args.min_tokens)
    PRETRAIN_TEXTS.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs))
    by_genre = Counter(d["genre"] for d in docs)
    log(f"{len(docs)} documents, {tokens / 1e6:.2f}M tokens (mean {tokens / len(docs):.0f}) -> {PRETRAIN_TEXTS}; "
        f"{dict(stats)}; genres {dict(by_genre)}")


# ------------------------------------------------------------- pools (S2 / SFT)


def build_pools(conversations, texts, fields, contaminated=None):
    """(identity_chat records, identity_text records, stats): template fields filled, re-checked, decontaminated,
    shuffled with a fixed seed. Records: {"messages", "category", "id"} (plain user/assistant turns) and
    {"text", "genre", "id"}."""
    contaminated = contaminated or (lambda text: False)
    st = {k: Counter({"in": 0, "kept": 0, "invalid": 0, "contaminated": 0}) for k in ("identity_chat", "identity_text")}
    chats, docs = [], []
    for r in conversations:
        c = st["identity_chat"]
        c["in"] += 1
        msgs = fill_messages([{"role": m["role"], "content": m["content"]} for m in r["messages"]], fields)
        if conversation_problems({"category": r["category"], "messages": msgs}, fields):
            c["invalid"] += 1
            continue
        if contaminated("\n".join(m["content"] for m in msgs)):
            c["contaminated"] += 1
            continue
        chats.append({"messages": msgs, "category": r["category"], "id": r["id"]})
        c["kept"] += 1
        c[f"category_{r['category']}"] += 1
    for r in texts:
        c = st["identity_text"]
        c["in"] += 1
        text = fill(r["text"], fields)
        if other_model_claims(text) or fact_errors(text, fields):
            c["invalid"] += 1
            continue
        if contaminated(text):
            c["contaminated"] += 1
            continue
        docs.append({"text": text, "genre": r["genre"], "id": r["id"]})
        c["kept"] += 1
    random.Random("identity_chat").shuffle(chats)   # a prefix of a pool is a uniform sample (S2 reads one)
    random.Random("identity_text").shuffle(docs)
    return chats, docs, {k: dict(v) for k, v in st.items()}


def cmd_pools(args):
    from bonsai.identity import load_conversations
    fields = template_fields(args.vocab_size)
    try:
        from bonsai.tokenizer import Tokenizer
        tok = Tokenizer.load()
        if tok.vocab_size != int(fields["vocab_size"].replace(",", "")):
            log(f"WARNING: the identity data says vocab_size {fields['vocab_size']} but tokenizer/vocab_subset.npz "
                f"has {tok.vocab_size:,}: pass --vocab-size {tok.vocab_size} or update persona.yaml (facts.vocab_size)")
    except (FileNotFoundError, ImportError) as e:
        tok = None
        log(f"no student tokenizer ({e}); token counts skipped")
    contaminated = None
    if not args.no_decontam:
        from bonsai.data.decontam import build_index, is_contaminated
        index = build_index()
        contaminated = lambda text: is_contaminated(text, index)  # noqa: E731
    convs = load_conversations(fill_fields=False)
    texts = [json.loads(l) for l in PRETRAIN_TEXTS.read_text().splitlines() if l.strip()]
    chats, docs, stats = build_pools(convs, texts, fields, contaminated)
    out = args.out_dir
    n_chat = write_jsonl(out / "identity_chat.jsonl", chats)
    n_text = write_jsonl(out / "identity_text.jsonl", docs)
    if tok is not None:
        from bonsai.chat import render_batch
        stats["identity_text"]["tokens"] = sum(len(x) + 1 for x in tok.encode_batch([d["text"] for d in docs]))
        stats["identity_chat"]["teacher_tokens"] = sum(len(ids) + 1 for ids, _ in
                                                       render_batch(tok.qwen, [c["messages"] for c in chats]))
    stats["template_fields"] = fields
    stats["decontaminated"] = not args.no_decontam
    stats_path = out / "stats_identity.json"
    write_json(stats_path, stats)
    log(f"identity_chat: {n_chat} conversations, identity_text: {n_text} documents -> {out} "
        f"(fields {fields}); stats -> {stats_path}: {stats}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="validate generated conversation files")
    c.add_argument("files", nargs="+")
    c.add_argument("--show", type=int, default=30, help="problem records to print per file")
    sub.add_parser("merge", help="identity/parts/*.jsonl -> identity/conversations.jsonl + report.json")
    g = sub.add_parser("llm", help="generate conversations with a local OpenAI-compatible server")
    g.add_argument("--base-url", default="http://localhost:8080/v1",
                   help="mlx_lm.server default; Ollama is http://localhost:11434/v1")
    g.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    g.add_argument("--categories", nargs="*", choices=list(CATEGORIES))
    g.add_argument("--batches", type=int, default=10, help="requests per category")
    g.add_argument("--per-batch", type=int, default=10, help="conversations asked for per request")
    g.add_argument("--temperature", type=float, default=0.9)
    g.add_argument("--seed", type=int, default=0, help="also names the output files, so reruns with new seeds add data")
    p = sub.add_parser("pretrain", help="identity/pretrain_bank.yaml -> identity/pretrain.jsonl")
    p.add_argument("--tokens", type=float, default=PRETRAIN_TOKENS)
    p.add_argument("--min-tokens", type=int, default=60, help="the text renderer drops docs under 50 tokens")
    p.add_argument("--seed", type=int, default=7)
    o = sub.add_parser("pools", help="-> raw/filtered/identity_chat.jsonl, identity_text.jsonl, stats_identity.json")
    o.add_argument("--vocab-size", type=int, default=None,
                   help=f"fill {{vocab_size}} with this instead of persona.yaml ({persona()['facts']['vocab_size']})")
    o.add_argument("--out-dir", type=lambda s: __import__("pathlib").Path(s), default=FILTERED_DIR)
    o.add_argument("--no-decontam", action="store_true", help="skip the 13-gram check (tests / no eval data)")
    args = ap.parse_args()
    cmds = {"check": cmd_check, "merge": cmd_merge, "llm": cmd_llm, "pretrain": cmd_pretrain, "pools": cmd_pools}
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()
