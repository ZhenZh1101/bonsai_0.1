"""Synthetic multi-hop deductions (plan section 10: "FLD.v2 + synthetic natural-language logic, 2-4 hops").

One engine, two renderings:
  english  FLD-style problems for S2 (fld_chat, config "logic") and S1 plain text: facts and rules in plain
           English, a hypothesis, label PROVED / DISPROVED / UNKNOWN, proof steps in bonsai.formal.fld's format
  fol      ASCII first-order logic (forall / exists / ~ / & / ->) for the S0 derivation docs (derivations.py)

A theory has ground facts (signed unary literals A(c), binary relations R(c, d)) and rules:
  r1  forall x (A(x) -> [~]B(x))
  r2  forall x ((A(x) & [~]B(x)) -> [~]C(x))
  r3  forall x forall y ((R(x, y) & A(y)) -> [~]B(x))
  r4  [~]A(c) -> [~]B(d)                       (a ground rule)
Generation: a backbone of `depth` rule applications ending in the target literal, then distractor facts and
rules (some with the target's predicate in the head). The label comes from forward chaining to the closure
(literal match, open world): PROVED if the hypothesis is derived, DISPROVED if its negation is, UNKNOWN if
neither (made by deleting one backbone fact). FLD's labels follow classical logic, so the label is also checked
classically: the theory grounded into clauses goes through a small DPLL test (`classical_label`), and a draw is
rejected when the theory is unsatisfiable or the classical label differs (an UNKNOWN hypothesis that follows by
contraposition / modus tollens or by cases). Inconsistent theories, shortcuts that change the depth and wrong
labels are rejected and regenerated, so every label and proof is checked. The proof is the derivation tree of
the decisive literal (minimal depth).
"""

import itertools

from bonsai.formal import fld, rng_for

ADJ = ("kind red big small tall short quiet loud happy sad young old rich poor smart brave calm cold warm soft "
       "hard round green blue white black heavy light fast slow clean dirty strong weak wet dry gentle shy proud "
       "rough smooth bright dark sharp sweet sour fresh wild busy lazy polite rude honest curious careful clever "
       "funny nervous famous friendly hungry lucky noisy sleepy tired angry empty golden wooden purple yellow "
       "orange gray striped spotted tiny huge thin wide deep ancient modern fancy plain cheap rare simple frozen "
       "rusty shiny fuzzy quick steady silent sticky juicy salty spicy hollow fierce loyal humble eager").split()
NOUN = ("a painter|a doctor|a farmer|a teacher|a dancer|a singer|a poet|a pilot|a baker|a sailor|a hunter|"
        "a student|an artist|an engineer|a nurse|a judge|a writer|a chef|a soldier|a gardener|a mammal|a bird|"
        "a reptile|a fish|a planet|a metal|a tool|a toy|a fruit|a vegetable").split("|")
VERB = [("sings", "sing"), ("dances", "dance"), ("swims", "swim"), ("runs", "run"), ("sleeps", "sleep"),
        ("laughs", "laugh"), ("glows", "glow"), ("flies", "fly"), ("reads", "read"), ("cooks", "cook"),
        ("travels", "travel"), ("whistles", "whistle"), ("smiles", "smile"), ("works", "work"), ("jumps", "jump"),
        ("shines", "shine"), ("grows", "grow"), ("melts", "melt"), ("floats", "float"), ("rings", "ring")]
REL = [("likes", "like"), ("visits", "visit"), ("trusts", "trust"), ("helps", "help"), ("chases", "chase"),
       ("knows", "know"), ("admires", "admire"), ("follows", "follow"), ("needs", "need"), ("sees", "see"),
       ("teaches", "teach"), ("feeds", "feed")]
NAMES = ("Alice Bob Carol Dave Erin Frank Grace Heidi Ivan Judy Oscar Nina Olivia Peggy Rupert Sybil Trent "
         "Victor Walter Anna Ben Chloe Dan Ella Finn Gina Hugo Iris Jack Kate Liam Mia Noah Owen Paul Quinn Rosa "
         "Sam Tina Uma Vera Will Zoe").split()
THINGS = ("cat dog bear mouse lion rabbit tiger squirrel eagle cow robot lamp box tree river stone king queen "
          "wizard knight owl fox horse frog whale shark bee ant duck goat turtle dragon giant clock train ship "
          "statue garden castle").split()
KINDS = ("r1", "r2", "r3", "r4")


class Theory:
    """Literals are (pred, args, positive); preds index self.preds, args are entity indices. Rule argument -1 is
    the variable x, -2 is y."""

    def __init__(self, n_entities):
        self.E = n_entities
        self.facts, self.rules = [], []   # rules: (kind, n_vars, body literals, head literal)

    @staticmethod
    def inst(lit, b):
        p, args, pos = lit
        return p, tuple(b[-a - 1] if a < 0 else a for a in args), pos

    def closure(self):
        known = {f: (0, None, ()) for f in self.facts}
        changed = True
        while changed:
            changed = False
            for ri, (_, nv, body, head) in enumerate(self.rules):
                for b in itertools.product(range(self.E), repeat=nv):
                    lits = [self.inst(l, b) for l in body]
                    if all(l in known for l in lits):
                        h = self.inst(head, b)
                        d = 1 + max(known[l][0] for l in lits)
                        if h not in known or known[h][0] > d:
                            known[h] = (d, ri, tuple(lits))
                            changed = True
        return known


def neg(lit):
    return lit[0], lit[1], not lit[2]


def ground(th):
    """The theory as ground clauses (lists of literals): a fact is a unit clause, a rule `body -> head` is
    ~b1 | ~b2 | head for every assignment of its variables."""
    out = [[f] for f in th.facts]
    for _, nv, body, head in th.rules:
        for b in itertools.product(range(th.E), repeat=nv):
            out.append(list(dict.fromkeys([neg(Theory.inst(l, b)) for l in body] + [Theory.inst(head, b)])))
    return out


def satisfiable(clauses):
    """DPLL (unit propagation, then branching) over ground signed literals; a theory has a few hundred clauses."""
    def solve(cls, val):
        while True:
            rest, units = [], []
            for c in cls:
                free = []
                for l in c:
                    v = val.get(l[:2])
                    if v is None:
                        free.append(l)
                    elif v == l[2]:
                        break
                else:
                    if not free:
                        return False
                    if len(free) == 1:
                        units.append(free[0])
                    rest.append(free)
            if not units:
                break
            for l in units:
                if val.setdefault(l[:2], l[2]) != l[2]:
                    return False
            cls = rest
        if not rest:
            return True
        lit = rest[0][0]
        return any(solve(rest, {**val, lit[:2]: b}) for b in (lit[2], not lit[2]))

    return solve(clauses, {})


def classical_label(th, hyp):
    """PROVED / DISPROVED / UNKNOWN of `hyp` under classical logic, None if the theory is inconsistent."""
    cls = ground(th)
    if not satisfiable(cls):
        return None
    if not satisfiable(cls + [[neg(hyp)]]):
        return "PROVED"
    return "DISPROVED" if not satisfiable(cls + [[hyp]]) else "UNKNOWN"


def _backbone(rng, th, depth, n_unary):
    """Chain of `depth` rule applications; returns (target literal, base facts used)."""
    fresh = iter(rng.sample(range(n_unary), n_unary))
    e = rng.randrange(th.E)
    cur = (next(fresh), (e,), rng.random() < 0.8)
    th.facts.append(cur)
    base = [cur]
    for _ in range(depth):
        kind = rng.choices(KINDS, weights=(3, 3, 2, 2))[0]
        p_prev, (e,), s_prev = cur
        head_pred, s = next(fresh), rng.random() < 0.7
        if kind == "r1":
            th.rules.append(("r1", 1, [(p_prev, (-1,), s_prev)], (head_pred, (-1,), s)))
        elif kind == "r2":
            extra = (next(fresh), (e,), rng.random() < 0.7)
            th.facts.append(extra)
            base.append(extra)
            body = [(p_prev, (-1,), s_prev), (extra[0], (-1,), extra[2])]
            th.rules.append(("r2", 1, body if rng.random() < 0.5 else body[::-1], (head_pred, (-1,), s)))
        elif kind == "r3":
            e2 = rng.choice([i for i in range(th.E) if i != e])
            rel = (n_unary + rng.randrange(2), (e2, e), True)
            th.facts.append(rel)
            base.append(rel)
            th.rules.append(("r3", 2, [(rel[0], (-1, -2), True), (p_prev, (-2,), s_prev)], (head_pred, (-1,), s)))
            e = e2
        else:
            e2 = rng.randrange(th.E)
            th.rules.append(("r4", 0, [(p_prev, (e,), s_prev)], (head_pred, (e2,), s)))
            e = e2
        cur = (head_pred, (e,), s)
    return cur, base


def _distractors(rng, th, n_unary, target):
    for _ in range(rng.randint(3, 8)):
        th.facts.append((rng.randrange(n_unary), (rng.randrange(th.E),), rng.random() < 0.7))
    if rng.random() < 0.5:
        th.facts.append((n_unary + rng.randrange(2), tuple(rng.sample(range(th.E), 2)), True))
    for _ in range(rng.randint(3, 7)):
        kind = rng.choices(KINDS, weights=(3, 3, 2, 2))[0]
        a, b, c = rng.sample(range(n_unary), 3)
        sa, sb, sc = (rng.random() < 0.7 for _ in range(3))
        if rng.random() < 0.35 and target[0] not in (a, b):   # a near miss: concludes the target's predicate
            c, sc = target[0], rng.random() < 0.5
        if kind == "r1":
            th.rules.append(("r1", 1, [(a, (-1,), sa)], (c, (-1,), sc)))
        elif kind == "r2":
            th.rules.append(("r2", 1, [(a, (-1,), sa), (b, (-1,), sb)], (c, (-1,), sc)))
        elif kind == "r3":
            th.rules.append(("r3", 2, [(n_unary + rng.randrange(2), (-1, -2), True), (a, (-2,), sa)], (c, (-1,), sc)))
        else:
            th.rules.append(("r4", 0, [(a, (rng.randrange(th.E),), sa)], (c, (rng.randrange(th.E),), sc)))


def make_problem(rng, depth=None, label=None, max_tries=200):
    """A checked problem: {"theory", "known", "hyp", "label", "depth", "decisive", "partial"}."""
    for _ in range(max_tries):
        d = depth or rng.choice((1, 2, 2, 3, 3, 4, 4, 5))
        lab = label or rng.choice(("PROVED", "DISPROVED", "UNKNOWN"))
        n_unary = 2 * d + 10
        th = Theory(rng.randint(3, 5))
        target, base = _backbone(rng, th, d, n_unary)
        _distractors(rng, th, n_unary, target)
        th.facts = list(dict.fromkeys(th.facts))
        if lab == "UNKNOWN":
            th.facts.remove(rng.choice(base))
        known = th.closure()
        if any(neg(l) in known for l in known):
            continue
        if lab == "UNKNOWN":
            if target in known or neg(target) in known:
                continue
            hyp = target if rng.random() < 0.5 else neg(target)
            if classical_label(th, hyp) != "UNKNOWN":
                continue
            derived = [l for l in known if known[l][0] > 0 and l[0] < n_unary]
            partial = max(derived, key=lambda l: known[l][0]) if derived and rng.random() < 0.7 else None
            return {"theory": th, "known": known, "hyp": hyp, "label": lab, "depth": d, "decisive": None,
                    "partial": partial, "n_unary": n_unary}
        if target not in known or known[target][0] != d:
            continue
        hyp = target if lab == "PROVED" else neg(target)
        if classical_label(th, hyp) != lab:
            continue
        return {"theory": th, "known": known, "hyp": hyp, "label": lab, "depth": d, "decisive": target,
                "partial": None, "n_unary": n_unary}
    raise RuntimeError("could not generate a logic problem")


def proof(known, lit):
    """[(literal, rule index, body literals)] in dependency order, ending with `lit` (derived literals only)."""
    out, seen = [], set()

    def visit(l):
        if l in seen or known[l][0] == 0:
            return
        seen.add(l)
        for b in known[l][2]:
            visit(b)
        out.append((l, known[l][1], known[l][2]))

    visit(lit)
    return out


# ------------------------------------------------------------------ English


class Lexicon:
    """Random surface words for one problem: predicate i -> (pos phrase, neg phrase, fol symbol)."""

    def __init__(self, rng, n_unary, n_entities, abstract=False):
        words = [("adj", w) for w in rng.sample(ADJ, min(len(ADJ), n_unary))]
        mix = [("noun", w) for w in rng.sample(NOUN, 6)] + [("verb", v) for v in rng.sample(VERB, 6)]
        for i in rng.sample(range(n_unary), min(n_unary // 3, len(mix))):
            words[i] = mix.pop()
        self.unary = []
        for kind, w in words:
            if kind == "adj":
                self.unary.append((f"is {w}", f"is not {w}", w.capitalize()))
            elif kind == "noun":
                self.unary.append((f"is {w}", f"is not {w}", w.split()[1].capitalize()))
            else:
                self.unary.append((w[0], f"does not {w[1]}", w[0].capitalize()))
        self.rel = rng.sample(REL, 2)
        if rng.random() < 0.5:
            self.ent = rng.sample(NAMES, n_entities)
            self.ent_fol = [e.lower() for e in self.ent]
        else:
            things = rng.sample(THINGS, n_entities)
            self.ent = [f"the {t}" for t in things]
            self.ent_fol = things
        if abstract:
            self.fol_pred = [f"P{i}" for i in rng.sample(range(1, 60), n_unary)] + ["R1", "R2"]
            self.ent_fol = list("abcdefgh")[:n_entities]
        else:
            self.fol_pred = [u[2] for u in self.unary] + [r[0].capitalize() for r in self.rel]
        self.n_unary = n_unary

    def vp(self, lit, subj=None):
        p, args, pos = lit
        if p >= self.n_unary:
            r = self.rel[p - self.n_unary]
            return f"{r[0] if pos else 'does not ' + r[1]} {self.ent[args[1]]}"
        return self.unary[p][0 if pos else 1]


def _cap(s):
    return s[0].upper() + s[1:]


def english_literal(lx, lit):
    p, args, pos = lit
    return _cap(f"{lx.ent[args[0]]} {lx.vp(lit)}") + "."


def english_rule(rng, lx, rule):
    kind, _, body, head = rule
    h = lx.unary[head[0]][0 if head[2] else 1]
    if kind == "r4":
        (b,) = body
        cond, concl = f"{lx.ent[b[1][0]]} {lx.vp(b)}", f"{lx.ent[head[1][0]]} {h}"
        return _cap(rng.choice([f"if {cond} then {concl}.", f"{concl} if {cond}."]))
    if kind == "r3":
        r, a = body
        rel = lx.rel[r[0] - lx.n_unary][0]
        av = lx.unary[a[0]][0 if a[2] else 1]
        return rng.choice([f"If something {rel} something that {av} then it {h}.",
                           f"Everything that {rel} something that {av} {h}."])
    conds = [lx.unary[l[0]][0 if l[2] else 1] for l in body]
    cond = " and ".join(conds)
    return rng.choice([f"If something {cond} then it {h}.", f"Everything that {cond} {h}.",
                       f"Anything that {cond} {h}.", f"Something {h} if it {cond}."])


def english(rng, prob):
    """{"facts", "hypothesis", "steps", "label", "depth"} with FLD-style numbered references."""
    th, known = prob["theory"], prob["known"]
    lx = Lexicon(rng, prob["n_unary"], th.E)
    items = [("f", f) for f in th.facts] + [("r", i) for i in range(len(th.rules))]
    rng.shuffle(items)
    ref = {it: f"fact{i + 1}" for i, it in enumerate(items)}
    texts = [english_literal(lx, it[1]) if it[0] == "f" else english_rule(rng, lx, th.rules[it[1]]) for it in items]
    lit_ref, steps = {f: ref[("f", f)] for f in th.facts}, []
    goal = prob["decisive"] or prob["partial"]
    chain = proof(known, goal) if goal is not None else []
    for k, (lit, ri, body) in enumerate(chain):
        refs = [ref[("r", ri)]] + [lit_ref[b] for b in body]
        if k == len(chain) - 1 and prob["decisive"] is not None:
            steps.append(fld.format_step(refs, final=prob["label"]))
        else:
            lit_ref[lit] = f"int{len(lit_ref) - len(th.facts) + 1}"
            steps.append(fld.format_step(refs, lit_ref[lit], english_literal(lx, lit)))
    return {"facts": texts, "hypothesis": english_literal(lx, prob["hyp"]), "steps": steps,
            "label": prob["label"], "depth": prob["depth"]}


def chat_record(*key, qwen=None):
    """A fld_chat record (config "logic"), or None if the proof is over the think budget."""
    rng = rng_for("logic", "chat", *key)
    e = english(rng, make_problem(rng, depth=rng.choice((2, 2, 3, 3, 4, 4, 1, 5))))
    reasoning = fld.format_reasoning(e["steps"], e["label"])
    if not fld.within_budget(reasoning, qwen):
        return None
    return {"user": fld.format_prompt(e["facts"], e["hypothesis"]), "reasoning": reasoning,
            "answer": fld.format_answer(e["label"]), "label": e["label"], "depth": e["depth"], "config": "logic"}


def text_record(*key):
    rng = rng_for("logic", "text", *key)
    e = english(rng, make_problem(rng, depth=rng.randint(2, 6)))
    return {"text": fld.text_doc(e["facts"], e["hypothesis"], e["steps"], e["label"]), "kind": "logic_text"}


# ------------------------------------------------------------------ ASCII first-order logic


def fol_literal(lx, lit, var=None):
    p, args, pos = lit
    names = [var.get(a, "?") if a < 0 else lx.ent_fol[a] for a in args] if var else [lx.ent_fol[a] for a in args]
    return ("" if pos else "~") + f"{lx.fol_pred[p]}({','.join(names)})"


def fol_rule(lx, rule):
    kind, nv, body, head = rule
    var = {-1: "x", -2: "y"}
    lits = [fol_literal(lx, l, var if nv else None) for l in body]
    h = fol_literal(lx, head, var if nv else None)
    cond = lits[0] if len(lits) == 1 else "(" + " & ".join(lits) + ")"
    if nv == 0:
        return f"{cond} -> {h}"
    q = "forall x forall y " if nv == 2 else "forall x "
    return f"{q}({cond} -> {h})"


def fol_trace(*key):
    """(premises, steps, goal) of an S0 derivation: numbered premises "[i] formula", steps "[j] formula by i, k"."""
    rng = rng_for("logic", "fol", *key)
    prob = make_problem(rng, depth=rng.choice((2, 3, 3, 4, 4, 5, 6)), label=rng.choice(("PROVED", "DISPROVED")))
    th, known = prob["theory"], prob["known"]
    lx = Lexicon(rng, prob["n_unary"], th.E, abstract=rng.random() < 0.4)
    items = [("f", f) for f in th.facts] + [("r", i) for i in range(len(th.rules))]
    rng.shuffle(items)
    num = {it: i + 1 for i, it in enumerate(items)}
    premises = [f"[{num[it]}] " + (fol_literal(lx, it[1]) if it[0] == "f" else fol_rule(lx, th.rules[it[1]]))
                for it in items]
    lit_num, steps, n = {f: num[("f", f)] for f in th.facts}, [], len(items)
    for lit, ri, body in proof(known, prob["decisive"]):
        n += 1
        lit_num[lit] = n
        refs = ", ".join(str(r) for r in [num[("r", ri)]] + [lit_num[b] for b in body])
        steps.append(f"[{n}] {fol_literal(lx, lit)} by {refs}")
    goal = fol_literal(lx, prob["decisive"])
    p, args, pos = prob["decisive"]
    if rng.random() < 0.3:   # existential generalization of the last step
        goal = f"exists x {'' if pos else '~'}{lx.fol_pred[p]}(x)"
        steps.append(f"[{n + 1}] {goal} by {n}")
    return premises, steps, goal
