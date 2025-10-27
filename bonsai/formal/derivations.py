"""Formal derivations for S0 (plan section 9) and the rc_short chat pool for S2 (plan section 10).

S0 doc format (Cheng et al.): "premises: p1 ; p2 ; ... ; derived: s1 ; ... ; s(k-1) ; goal: g ; next: sk", one
doc for each of up to `per_trace` (2) randomly sampled steps k of a trace ("derived: none" for k = 1); sampling
steps instead of taking all of them spreads the S0 budget over more distinct problems. The loss covers only " sk": `target_start` is the
character right after "next:". At most 512 student tokens incl. BOS: a long doc first replaces its oldest derived
steps by "...", then drops the premises the trace never cites, else it is dropped. ASCII only.

Sources:
  ProcPile (reasoning-core/procedural-pile, 2 x 250k rows, 55 tasks). Kept, the tasks whose metadata.cot is a step
  trace over formal objects:
    logic_derivation, logic_qa, multistep_nli, multistep_evidence_retrieval, defeasible_nli: rule applications
        over numbered premises ("6. [P13,4,5] george is an ancestor of elena"; the cot's premise numbers were
        checked against the premise list: they match on every sampled row)
    rewrite_system (one rewritten term per step), equation_system (row operations, then back substitution),
    arithmetics (one operation per step)
  plus two cot-less tasks whose trace is reconstructed here and checked against the gold answer:
    lambda_reduction    normal-order beta reduction, capture-avoiding; kept if the normal form is alpha-equal
                        to the answer (<= 60 steps)
    parsing_derivation  leftmost derivation, the sentential forms of the gold rule sequence; kept if the last
                        form is the string
  Not used: tasks without a step trace (the other 45), function_manipulation (its cot hides the operations),
  graph_pathfinding (one-line cot).
  FOL-Traces (fol-traces/fol-traces; our slice is absorption, modus tollens and a distribution law, 100k rows
  each): one step per row. Unicode -> ASCII (~ & | -> <-> forall exists); premises = the conjuncts of the left
  side, goal = the law, next = the right side.
  logic.fol_trace: synthetic multi-step FOL proofs with quantifiers.
reasoning_core (the plan's `pip install reasoning_core`) is not installed: it pulls in 41 packages (duckdb,
pandas, sympy, z3, nltk, udocker, ...) and ProcPile + these generators already exceed the S0 budget.

rc_short (DESIGN 1.1): reasoning-core/symbolic-reasoning-env rows and ProcPile rows with a cot of <= 192 teacher
tokens -> {"user": prompt, "reasoning": cot, "answer": "Answer: <answer>", "task", "source", "cut"}; cots over
128 tokens are cut to 128 with budget_thinking (THINK_STOP appended, "cut": true, as DESIGN 0.2 prescribes for
long thinking). ProcPile rows are split by a stable prompt hash: 3/4 of the short-cot rows of the S0 tasks go to
rc_short, the rest and every long-cot row to S0, so S0 and S2 never share a problem; short-cot rows of the other
tasks all go to rc_short.
"""

import json
import re

import pyarrow.parquet as pq

from bonsai.chat import THINK_BUDGET, budget_thinking
from bonsai.common import RAW_DIR
from bonsai.formal import logic, rng_for, stable_bucket

PROCPILE = sorted((RAW_DIR / "procedural_pile" / "data").glob("train-*.parquet"))
SYMBOLIC_ENV = sorted((RAW_DIR / "symbolic_reasoning_env" / "data").glob("train-*.parquet"))
FOL_TRACES = sorted((RAW_DIR / "fol_traces").glob("**/*.jsonl"))
LOGIC_TASKS = ("logic_derivation", "logic_qa", "multistep_nli", "multistep_evidence_retrieval", "defeasible_nli")
S0_TASKS = LOGIC_TASKS + ("rewrite_system", "equation_system", "arithmetics", "lambda_reduction",
                          "parsing_derivation")
MAX_TOKENS = 512          # incl. BOS
MAX_CHAT_TOKENS = 1000    # rendered rc_short conversation, < S2's 1024
MAX_ANSWER_TOKENS = 100
CUT_MAX = THINK_BUDGET * 3 // 2   # rc_short: cots up to 192 teacher tokens are cut to 128 with budget_thinking
RC_SHARE = 3                      # of 4 hash buckets: ProcPile short-cot rows of the S0 tasks that go to rc_short
_STEP = re.compile(r"^\s*(\d+)\.\s*\[P(\d+)((?:;\d+)*)\]\s*(.*)$")


# ------------------------------------------------------------------ doc format


def derivation_doc(premises, derived, goal, nxt):
    head = (f"premises: {' ; '.join(premises)} ; derived: {' ; '.join(derived) if derived else 'none'} ; "
            f"goal: {goal} ; next:")
    return {"text": f"{head} {nxt}", "target_start": len(head)}


def _clean(item):
    return " ".join(item.replace(" ; ", ", ").split())


def trace_docs(tok, premises, steps, goal, rng, per_trace, cited=None, source=""):
    """Docs for up to `per_trace` sampled steps of one trace; `cited` = indexes of premises the trace uses (the
    others may be dropped to fit). Returns (docs, n_too_long)."""
    premises, steps, goal = [_clean(p) for p in premises], [_clean(s) for s in steps], _clean(goal)
    picks = sorted(rng.sample(range(len(steps)), min(per_trace, len(steps))))
    docs, too_long = [], 0
    for k in picks:
        prem, keep = premises, k
        while True:
            derived = steps[:k] if keep == k else ["..."] + steps[k - keep:k]
            d = derivation_doc(prem, derived, goal, steps[k])
            n = len(tok.encode(d["text"])) + 1
            if n <= MAX_TOKENS:
                break
            if keep > 1:
                keep = keep // 2 if n > 2 * MAX_TOKENS else keep - 1
            elif cited is not None and prem is premises:
                prem = [p for i, p in enumerate(premises) if i in cited]
                keep = k
            else:
                d = None
                break
        if d is None or not d["text"].isascii():
            too_long += d is None
            continue
        d["source"] = source
        docs.append(d)
    return docs, too_long


# ------------------------------------------------------------------ ProcPile traces


def _logic_traces(row, meta):
    premises = meta["premise"]
    goal = meta.get("target") or meta.get("hypothesis") or meta.get("question") or ""
    out = []
    for block in (meta.get("cot") or "").strip().split("\n\n"):
        steps, cited = [], set()
        for line in block.splitlines():
            m = _STEP.match(line)
            if not m:
                steps = []
                break
            cited.add(int(m.group(2)))
            refs = m.group(3).replace(";", ",")
            steps.append(f"{m.group(1)}. [P{m.group(2)}{refs}] {m.group(4).strip()}")
        if steps:
            out.append(([f"P{i} {p}" for i, p in enumerate(premises)], steps, goal, cited))
    return out


def _rewrite_traces(row, meta):
    rules = [r[2:].strip() for r in meta["rules"].splitlines() if r.startswith("- ")]
    steps = [l[3:].strip() for l in (meta.get("cot") or "").splitlines() if l.startswith("-> ")]
    if not rules or not steps:
        return []
    return [(rules, steps, f"normal form of {meta['term']}", None)]


def _equation_traces(row, meta):
    steps = [l.strip() for l in (meta.get("cot") or "").splitlines()
             if l.strip() and not re.match(r"^\d+\. (Forward|Backward):", l.strip())]
    eqs = meta["equations"]
    if not steps or not isinstance(eqs, list):
        return []
    return [(eqs, steps, f"solve for {meta['query_variable']}", None)]


def _arith_traces(row, meta):
    steps = [l.strip() for l in (meta.get("cot") or "").splitlines() if l.strip()]
    expr = row["prompt"].split("\n", 1)[0].strip()
    if not steps or not expr.startswith("Evaluate "):
        return []
    return [([expr[len("Evaluate "):].rstrip(".")], steps, "evaluate", None)]


def _lambda_traces(row, meta):
    try:
        term = parse_lambda(meta["term"])
        nf = parse_lambda(meta["normal_form"])
    except (ValueError, IndexError, RecursionError):
        return []
    steps = []
    for _ in range(60):
        nxt = beta_step(term, used=_names(term))
        if nxt is None:
            break
        term = nxt
        steps.append(show_lambda(term))
        if len(steps[-1]) > 700:
            return []
    if not steps or beta_step(term, used=_names(term)) is not None or de_bruijn(term) != de_bruijn(nf):
        return []
    return [([meta["term"]], steps, "beta normal form", None)]


_BRACKETS = str.maketrans({"⟨": "(<", "⟩": ">)", "⟦": "([", "⟧": "])", "⟪": "(<<", "⟫": ">>)"})


def _parsing_traces(row, meta):
    prompt = row["prompt"].translate(_BRACKETS)   # a consistent renaming of terminals keeps the derivation valid
    m = re.search(r"\(START\)\n(\S+)\n\n\(GRAMMAR\)\n(.*?)\n\n\(STRING\)\n(.*?)\n", prompt, re.S)
    if not m:
        return []
    start, grammar, string = m.group(1), m.group(2), m.group(3).split()
    rules = {}
    for line in grammar.splitlines():
        r = re.match(r"^(R\d+): (\S+) (::=|->) ?(.*)$", line)
        if not r:
            return []
        rules[r.group(1)] = (r.group(2), r.group(3), re.findall(r"'[^']*'|\S+", r.group(4)))
    form, steps = [start], []
    for label in row["answer"].split():
        if label not in rules:
            return []
        lhs, _, rhs = rules[label]
        i = next((j for j, s in enumerate(form) if not s.startswith("'")), None)
        if i is None or form[i] != lhs:
            return []
        form = form[:i] + rhs + form[i + 1:]
        steps.append(f"{label} {' '.join(form)}")
    if [s[1:-1] for s in form] != string or any(not s.startswith("'") for s in form):
        return []
    premises = [f"{k} {lhs} {arrow} {' '.join(rhs) or chr(39) * 2}" for k, (lhs, arrow, rhs) in rules.items()]
    return [(premises, steps, f"derive {' '.join(string)} from {start}", None)]


TRACERS = {**{t: _logic_traces for t in LOGIC_TASKS}, "rewrite_system": _rewrite_traces,
           "equation_system": _equation_traces, "arithmetics": _arith_traces, "lambda_reduction": _lambda_traces,
           "parsing_derivation": _parsing_traces}


def procpile_chunk(path, start, stop, tok, qwen, per_trace=2):
    """S0 docs and rc_short records from ProcPile rows [start, stop). Returns (docs, rc, counts)."""
    rows = pq.read_table(path, columns=["task", "prompt", "answer", "metadata"]).slice(start, stop - start).to_pylist()
    docs, rc, cnt = [], [], {}
    for i, row in enumerate(rows):
        meta = json.loads(row["metadata"])
        cot = meta.get("cot") or ""
        short = bool(cot) and len(qwen.encode(cot)) <= CUT_MAX
        if short and (row["task"] not in TRACERS or stable_bucket(row["prompt"], 4) < RC_SHARE):
            rec = rc_record(row, cot, "procedural_pile", qwen)
            if rec:
                rc.append(rec)
            continue
        if row["task"] not in TRACERS:
            continue
        traces = TRACERS[row["task"]](row, meta)
        cnt[f"{row['task']}_rows"] = cnt.get(f"{row['task']}_rows", 0) + 1
        cnt[f"{row['task']}_traced"] = cnt.get(f"{row['task']}_traced", 0) + bool(traces)
        rng = rng_for("derivations", path.name, start + i)
        for premises, steps, goal, cited in traces:
            got, too_long = trace_docs(tok, premises, steps, goal, rng, per_trace, cited, row["task"])
            docs += got
            cnt["too_long"] = cnt.get("too_long", 0) + too_long
    return docs, rc, cnt


# ------------------------------------------------------------------ lambda calculus


def parse_lambda(s):
    """`\\x.body` abstractions (extending right), left-associative application, parentheses."""
    toks = re.findall(r"\\|\.|\(|\)|[A-Za-z_][A-Za-z0-9_]*", s)
    pos = 0

    def term():
        nonlocal pos
        items = []
        while pos < len(toks) and toks[pos] != ")":
            if toks[pos] == "\\":
                name = toks[pos + 1]
                if toks[pos + 2] != ".":
                    raise ValueError("bad abstraction")
                pos += 3
                items.append(("lam", name, term()))
            elif toks[pos] == "(":
                pos += 1
                items.append(term())
                if toks[pos] != ")":
                    raise ValueError("unbalanced")
                pos += 1
            else:
                items.append(("var", toks[pos]))
                pos += 1
        if not items:
            raise ValueError("empty term")
        t = items[0]
        for a in items[1:]:
            t = ("app", t, a)
        return t

    t = term()
    if pos != len(toks):
        raise ValueError("trailing input")
    return t


def show_lambda(t):
    if t[0] == "var":
        return t[1]
    if t[0] == "lam":
        return f"(\\{t[1]}.{show_lambda(t[2])})"
    return f"({show_lambda(t[1])} {show_lambda(t[2])})"


def _names(t, acc=None):
    acc = set() if acc is None else acc
    if t[0] == "var":
        acc.add(t[1])
    elif t[0] == "lam":
        acc.add(t[1])
        _names(t[2], acc)
    else:
        _names(t[1], acc)
        _names(t[2], acc)
    return acc


def _free(t):
    if t[0] == "var":
        return {t[1]}
    if t[0] == "lam":
        return _free(t[2]) - {t[1]}
    return _free(t[1]) | _free(t[2])


def _subst(t, x, n, fv, used):
    if t[0] == "var":
        return n if t[1] == x else t
    if t[0] == "app":
        return ("app", _subst(t[1], x, n, fv, used), _subst(t[2], x, n, fv, used))
    y, body = t[1], t[2]
    if y == x:
        return t
    if y in fv and x in _free(body):
        z = next(f"v{i}" for i in range(len(used) + 10) if f"v{i}" not in used)
        used.add(z)
        body, y = _subst(body, y, ("var", z), {z}, used), z
    return ("lam", y, _subst(body, x, n, fv, used))


def beta_step(t, used):
    """One normal-order (leftmost-outermost) beta step, or None for a normal form."""
    if t[0] == "app":
        f, a = t[1], t[2]
        if f[0] == "lam":
            return _subst(f[2], f[1], a, _free(a), used)
        s = beta_step(f, used)
        if s is not None:
            return ("app", s, a)
        s = beta_step(a, used)
        return None if s is None else ("app", f, s)
    if t[0] == "lam":
        s = beta_step(t[2], used)
        return None if s is None else ("lam", t[1], s)
    return None


def de_bruijn(t, env=()):
    if t[0] == "var":
        return ("b", env.index(t[1])) if t[1] in env else ("f", t[1])
    if t[0] == "lam":
        return ("l", de_bruijn(t[2], (t[1],) + env))
    return ("a", de_bruijn(t[1], env), de_bruijn(t[2], env))


# ------------------------------------------------------------------ FOL-Traces

FOL_ASCII = {"∨": " | ", "∧": " & ", "¬": "~", "→": " -> ", "⟶": " -> ", "⟹": " -> ", "⇒": " -> ", "⇔": " <-> ",
             "↔": " <-> ", "∀": "forall ", "∃": "exists ", "⊕": " xor "}
FOL_LAWS = {"absorption": "simplify by absorption", "modus_tollens": "conclude by modus tollens",
            "modus_ponens": "conclude by modus ponens", "de_morgan": "apply De Morgan", "c": "distribute | over &"}


def to_ascii_logic(s):
    for u, a in FOL_ASCII.items():
        s = s.replace(u, a)
    s = re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", s))
    return " ".join(s.split())


def _split_top(s, op):
    depth = 0
    for i in range(len(s)):
        c = s[i]
        depth += (c == "(") - (c == ")")
        if depth == 0 and s.startswith(op, i):
            return s[:i].strip(), s[i + len(op):].strip()
    return None


def _strip_parens(s):
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        for i, c in enumerate(s):
            depth += (c == "(") - (c == ")")
            if depth == 0 and i < len(s) - 1:
                return s
        s = s[1:-1].strip()
    return s


def _balanced(s):
    depth = 0
    for c in s:
        depth += (c == "(") - (c == ")")
        if depth < 0:
            return False
    return depth == 0


def fol_traces_doc(rec):
    """(premises, steps, goal) of one FOL-Traces row, or None (malformed). The distribution rows (fol_type c7)
    lack the closing parentheses at the end of their right side; they are appended."""
    s = to_ascii_logic(rec["raw_data"].replace("<|endoftext|>", " "))
    parts = _split_top(s, " <-> ") or _split_top(s, " -> ")
    if not parts or not s.isascii():
        return None
    lhs, rhs = parts
    missing = rhs.count("(") - rhs.count(")")
    if missing > 0:
        rhs += ")" * missing
    if not (_balanced(lhs) and _balanced(rhs)):
        return None
    premises, rest = [], _strip_parens(lhs)
    while (cut := _split_top(rest, " & ")) is not None:
        premises.append(_strip_parens(cut[0]))
        rest = cut[1]
    premises.append(_strip_parens(rest))
    law = re.match(r"[a-z_]+?(?=_\d|\d|$)", rec["fol_type"])
    goal = FOL_LAWS.get(law.group(0) if law else "", "equivalent form")
    return [f"[{i + 1}] {p}" for i, p in enumerate(premises)], [_strip_parens(rhs)], goal


def fol_traces_chunk(path, start, stop, tok, keep=100):
    """S0 docs from FOL-Traces lines [start, stop); `keep` = percent of rows kept (stable hash)."""
    docs, cnt = [], {"rows": 0, "kept": 0}
    with open(path) as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= stop:
                break
            rec = json.loads(line)
            cnt["rows"] += 1
            if stable_bucket(rec["raw_data"]) >= keep:
                continue
            t = fol_traces_doc(rec)
            if t is None:
                continue
            got, _ = trace_docs(tok, t[0], t[1], t[2], rng_for("fol_traces", path.name, i), 1, source="fol_traces")
            docs += got
            cnt["kept"] += len(got)
    return docs, cnt


def logic_chunk(chunk, n, tok, per_trace=2):
    """S0 docs from n synthetic FOL proofs."""
    docs = []
    for i in range(n):
        premises, steps, goal = logic.fol_trace(chunk, i)
        got, _ = trace_docs(tok, premises, steps, goal, rng_for("logic_docs", chunk, i), per_trace, source="logic_fol")
        docs += got
    return docs


# ------------------------------------------------------------------ rc_short

_SCRIPT_OK = re.compile(r"^[\x00-\x7f\u00a0-\u024f\u0370-\u03ff\u2000-\u2bff]*$")   # Latin, Greek, math, arrows


def rc_record(row, cot, source, qwen):
    """{"user", "reasoning", "answer", "task", "source", "cut"} or None (long, empty or non-English). A cot of
    129-192 teacher tokens is cut to 128 with Qwen3's early-exit sentence (DESIGN 0.2), "cut" = True."""
    answer, cot = (row["answer"] or "").strip(), cot.strip()
    if not answer or not cot or len(qwen.encode(answer)) > MAX_ANSWER_TOKENS:
        return None
    if not _SCRIPT_OK.match(row["prompt"] + cot + answer):
        return None
    if len(qwen.encode(cot)) > CUT_MAX or len(qwen.encode(row["prompt"])) > MAX_CHAT_TOKENS - 300:
        return None
    cot, cut = budget_thinking(qwen, cot)
    return {"user": row["prompt"].strip(), "reasoning": cot, "answer": f"Answer: {answer}",
            "task": row["task"], "source": source, "cut": cut}


def symbolic_env_chunk(path, start, stop, qwen):
    rows = pq.read_table(path, columns=["task", "prompt", "answer", "metadata"]).slice(start, stop - start).to_pylist()
    out = []
    for row in rows:
        cot = json.loads(row["metadata"]).get("cot") or ""
        if cot:
            rec = rc_record(row, cot, "symbolic_reasoning_env", qwen)
            if rec:
                out.append(rec)
    return out
