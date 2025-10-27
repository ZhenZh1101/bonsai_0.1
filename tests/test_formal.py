"""Tests for bonsai.formal (S0 formal languages, derivations, synthetic reasoning, FLD helpers)."""

import random
import re

import numpy as np
import pytest

from bonsai.common import RAW_DIR, TOKENIZER_DIR
from bonsai.formal import arith, derivations, dyck, fld, igsm, logic, mp_struct
from bonsai.tokenizer import QwenTokenizer, Tokenizer

HAVE_VOCAB = (TOKENIZER_DIR / "vocab_subset.npz").exists()
HAVE_FLD = (RAW_DIR / "fld_v2" / "data").exists()


@pytest.fixture(scope="module")
def qwen():
    return QwenTokenizer()


@pytest.fixture(scope="module")
def tok():
    if not HAVE_VOCAB:
        pytest.skip("tokenizer/vocab_subset.npz missing")
    return Tokenizer.load()


# ------------------------------------------------------------------ Dyck


@pytest.mark.parametrize("lang", ["shuffle", "dyck"])
def test_dyck_grammar_and_one_token_per_symbol(lang, qwen):
    for i in range(20):
        text = dyck.doc(lang, "test", i)["text"]
        seq = dyck.parse(text, lang)
        assert len(seq) == dyck.SEQ_SYMBOLS and text.isascii()
        assert dyck.is_valid(seq, lang)
        assert len(qwen.encode(text)) == len(seq)        # every symbol is one Qwen token in context
    depth = max_depth = 0
    for s in seq:
        k = dyck.K_SHUFFLE if lang == "shuffle" else dyck.K_DYCK
        depth += 1 if s < k else -1
        max_depth = max(max_depth, depth)
        assert depth >= 0
    assert max_depth <= dyck.LANGS[lang][2]


def test_dyck_validity_checker_rejects_bad_sequences():
    k = dyck.K_DYCK
    assert dyck.is_valid([0, 1, k + 1, k], "dyck")
    assert not dyck.is_valid([0, 1, k, k + 1], "dyck")          # crossing closers are not k-Dyck
    ks = dyck.K_SHUFFLE
    assert dyck.is_valid([0, 1, ks, ks + 1], "shuffle")        # ... but they are shuffle Dyck
    assert not dyck.is_valid([0, ks + 1], "shuffle")
    assert not dyck.is_valid([0] * (dyck.SHUFFLE_MAX_DEPTH + 1), "shuffle")


def test_dyck_symbols_single_student_tokens(tok):
    for lang in ("shuffle", "dyck"):
        ids = dyck.symbol_ids(tok, lang)
        assert len(set(ids.tolist())) == len(ids)
        text = dyck.doc(lang, "student", 0)["text"]
        assert len(tok.encode(text)) == dyck.SEQ_SYMBOLS


class _Oracle:
    """Fake model: puts the logit mass on a valid next symbol (the matching closer when one is open)."""

    def __init__(self, tok, vocab):
        self.vocab = vocab
        self.maps = {lang: dyck.symbol_ids(tok, lang) for lang in ("dyck", "shuffle")}

    def __call__(self, x):
        import mlx.core as mx
        x = np.array(x)
        out = np.zeros(x.shape + (self.vocab,), dtype=np.float32)
        for lang, ids in self.maps.items():
            if not np.isin(x[:, 1:], ids).all():
                continue
            back = {int(t): i for i, t in enumerate(ids)}
            k = len(ids) // 2
            for r in range(x.shape[0]):
                seq = [back[int(t)] for t in x[r, 1:]]
                for t, ok in enumerate(dyck.valid_next(seq + [0], lang)):
                    closers = sorted(s for s in ok if s >= k)
                    out[r, t, ids[closers[0] if closers else 0]] = 10.0
                    for s in closers:
                        out[r, t, ids[s]] = max(out[r, t, ids[s]], 5.0)
        return mx.array(out)


def test_dyck_accuracy_oracle_and_tiny_model(tok):
    res = dyck.dyck_accuracy(_Oracle(tok, tok.vocab_size), tok, n=3, length=64)
    assert res["kdyck_close_acc"] == 1.0 and res["kdyck_valid_rate"] == 1.0
    assert res["shuffle_valid_rate"] == 1.0 and res["shuffle_close_acc"] == 1.0
    from bonsai.model import build_model
    model = build_model("tiny", vocab_size=tok.vocab_size, seq_len=128)
    res = dyck.dyck_accuracy(model, tok, n=2, length=64, batch=2)
    for key in ("kdyck_close_acc", "kdyck_valid_rate", "shuffle_valid_rate", "shuffle_close_acc"):
        assert 0.0 <= res[key] <= 1.0
    assert res["kdyck_close_positions"] > 0


# ------------------------------------------------------------------ MP-STRUCT


def test_mp_struct_grammar_and_tokens(qwen):
    seen = set()
    for i in range(30):
        text = mp_struct.doc("test", i)["text"]
        words = text.split()
        assert len(words) == mp_struct.SEQ_SYMBOLS and text.isascii()
        assert set(words) <= set(mp_struct.WORDS)
        assert len(qwen.encode(text)) == len(words)
        trees, _ = mp_struct.parse_tree(words)
        assert trees and all(mp_struct.check_sentence(t) for t in trees)
        seen |= set(words)
    assert {"MOV", "GAP", "WH", "Q", "SG", "PL", "sg", "pl"} <= seen
    good = mp_struct.to_text(mp_struct.sentence(random.Random(3))).split()
    t = next(i for i, w in enumerate(good) if w in ("sg", "pl"))
    bad = good[:t] + [{"sg": "pl", "pl": "sg"}[good[t]]] + good[t + 1:]     # T no longer agrees
    assert mp_struct.check_sentence(mp_struct.parse_tree(good)[0][0])
    with pytest.raises(ValueError, match="agree"):
        mp_struct.check_sentence(mp_struct.parse_tree(bad)[0][0])


# ------------------------------------------------------------------ derivations


def test_derivation_doc_format():
    d = derivations.derivation_doc(["[1] P(a)", "[2] forall x (P(x) -> Q(x))"], [], "Q(a)", "[3] Q(a) by 2, 1")
    assert d["text"].startswith("premises: [1] P(a) ; [2] forall x")
    assert " ; derived: none ; goal: Q(a) ; next:" in d["text"]
    assert d["text"][:d["target_start"]].endswith("next:")
    assert d["text"][d["target_start"]:] == " [3] Q(a) by 2, 1"


def test_trace_docs_target_boundary_and_length(tok):
    premises = [f"P{i} item number {i} is red." for i in range(80)]
    steps = [f"{i}. [P{i},{max(i - 1, 1)}] item {i} is blue" for i in range(1, 60)]
    docs, _ = derivations.trace_docs(tok, premises, steps, "goal text", random.Random(0), 5, cited={1, 2, 3})
    assert docs
    for d in docs:
        ids, offs = tok.qwen.encode_offsets(d["text"])
        assert len(ids) + 1 <= derivations.MAX_TOKENS
        assert any(a == d["target_start"] for a, _ in offs)      # the target starts on a token boundary
        assert d["text"][d["target_start"]:].lstrip() in steps


def test_lambda_reduction():
    t = derivations.parse_lambda(r"((\x.(\y.x)) y)")              # capture: y must be renamed
    nf = derivations.beta_step(t, derivations._names(t))
    assert derivations.de_bruijn(nf) == derivations.de_bruijn(derivations.parse_lambda(r"(\z.y)"))
    assert derivations.show_lambda(derivations.parse_lambda(r"\x.x y")) == r"(\x.(x y))"
    row = {"prompt": "", "answer": ""}
    meta = {"term": r"((\v0.(v0 v0)) ((\v1.v1) a))", "normal_form": "(a a)"}
    (premises, steps, goal, _), = derivations._lambda_traces(row, meta)
    assert steps[-1] == "(a a)" and goal == "beta normal form"
    assert derivations._lambda_traces(row, {**meta, "normal_form": "(a b)"}) == []


def test_parsing_and_logic_traces():
    prompt = ("(START)\nstart\n\n(GRAMMAR)\nR0: start -> seq\nR1: seq -> \nR2: seq -> expr seq\n"
              "R3: expr -> '(' seq ')'\n\n(STRING)\n( )\n\n(QUESTION)\nThe answer is ...")
    (prem, steps, goal, _), = derivations._parsing_traces({"prompt": prompt, "answer": "R0 R2 R3 R1 R1"}, {})
    assert steps[-1] == "R1 '(' ')'" and "R1 seq -> ''" in prem
    assert derivations._parsing_traces({"prompt": prompt, "answer": "R0 R1"}, {}) == []
    meta = {"premise": ["a is red.", "If x is red then x is big."], "target": "a is big.",
            "cot": "1. [P0] a is red\n2. [P1;1] a is big"}
    (prem, steps, goal, cited), = derivations._logic_traces({}, meta)
    assert prem[1] == "P1 If x is red then x is big." and steps[1] == "2. [P1,1] a is big" and cited == {0, 1}


def test_fol_traces_ascii():
    rec = {"raw_data": "<|endoftext|> ((A(x) → B(x)) ∧ ¬B(x)) → ¬A(x) <|endoftext|>", "fol_type": "modus_tollens"}
    premises, steps, goal = derivations.fol_traces_doc(rec)
    assert premises == ["[1] A(x) -> B(x)", "[2] ~B(x)"] and steps == ["~A(x)"]
    assert goal == "conclude by modus tollens"
    rec = {"raw_data": "(P(x) ∨ (P(x) ∧ Q(y))) ⇔ P(x)", "fol_type": "absorption_1_reversed"}
    premises, steps, goal = derivations.fol_traces_doc(rec)
    assert premises == ["[1] P(x) | (P(x) & Q(y))"] and steps == ["P(x)"] and "absorption" in goal
    assert derivations.to_ascii_logic("∀x ∃y (R(x, y) ↔ S(y))") == "forall x exists y (R(x, y) <-> S(y))"
    rec = {"raw_data": "((A(x) ∨ (B(x) ∧ C(x)))) ↔ (((A(x) ∨ B(x)) ∧ ((A(x) ∨ C(x))", "fol_type": "c7"}
    premises, steps, goal = derivations.fol_traces_doc(rec)      # the source drops the final ")" of c7 rows
    assert premises == ["[1] A(x) | (B(x) & C(x))"] and steps == ["(A(x) | B(x)) & ((A(x) | C(x)))"]
    assert derivations.fol_traces_doc({"raw_data": "(P(x) ∨ (P(x) ∧ Q x))) ⇔ P(x)", "fol_type": "absorption"}) is None


# ------------------------------------------------------------------ logic engine


def test_logic_labels_are_checked():
    for i in range(60):
        rng = random.Random(i)
        p = logic.make_problem(rng)
        known = p["theory"].closure()
        if p["label"] == "PROVED":
            assert p["hyp"] in known and logic.neg(p["hyp"]) not in known
        elif p["label"] == "DISPROVED":
            assert logic.neg(p["hyp"]) in known and p["hyp"] not in known
        else:
            assert p["hyp"] not in known and logic.neg(p["hyp"]) not in known
        if p["decisive"] is not None:
            chain = logic.proof(known, p["decisive"])
            assert chain[-1][0] == p["decisive"] and len(chain) >= p["depth"]
        assert logic.classical_label(p["theory"], p["hyp"]) == p["label"]   # classical logic agrees


def test_logic_classical_check():
    import itertools
    rng = random.Random(0)
    atoms = [(p, (0,)) for p in range(5)]
    for _ in range(300):   # DPLL == brute force on random small clause sets
        cls = [[(*rng.choice(atoms), rng.random() < 0.5) for _ in range(rng.randint(1, 3))]
               for _ in range(rng.randint(1, 12))]
        brute = any(all(any(dict(zip(atoms, vals))[l[:2]] == l[2] for l in c) for c in cls)
                    for vals in itertools.product((False, True), repeat=len(atoms)))
        assert logic.satisfiable(cls) == brute
    th = logic.Theory(1)   # "Everything that is big flies. The cat does not fly." -> "The cat is big." DISPROVED
    th.facts = [(1, (0,), False)]
    th.rules = [("r1", 1, [(0, (-1,), True)], (1, (-1,), True))]
    assert (0, (0,), True) not in th.closure() and logic.classical_label(th, (0, (0,), True)) == "DISPROVED"
    th.rules.append(("r1", 1, [(0, (-1,), False)], (1, (-1,), True)))   # big or not, it flies: inconsistent
    assert not any(logic.neg(l) in th.closure() for l in th.closure()) and logic.classical_label(th, (0, (0,), True)) is None


def test_logic_renderings(qwen):
    for i in range(20):
        r = logic.chat_record("test", i, qwen=qwen)
        if r is None:
            continue
        assert r["answer"] == f"Answer: {r['label']}" and r["config"] == "logic"
        assert r["user"].startswith("Facts:\nfact1: ") and fld.QUESTION in r["user"]
        assert len(qwen.encode(r["reasoning"])) <= 128
        if r["label"] != "UNKNOWN":
            assert r["reasoning"].splitlines()[-1].endswith("the hypothesis.")
        refs = set(re.findall(r"fact(\d+)", r["reasoning"]))
        assert all(f"fact{n}:" in r["user"] for n in refs)
    premises, steps, goal = logic.fol_trace("test", 1)
    assert all(s.isascii() for s in premises + steps + [goal])
    assert any("forall x" in p for p in premises)
    assert re.match(r"^\[\d+\] \S+ by \d+(, \d+)*$", steps[0])


# ------------------------------------------------------------------ arith / iGSM


def test_arith_scratchpads():
    minus = 0
    for i in range(300):
        q, lines, v = arith.expression(random.Random(i))
        assert eval(q[len("Evaluate "):-1]) == v
        minus += " - " in q
        for line in lines[:-1]:
            step, cur = line.split(", so: ")
            assert eval(cur) == v and eval(step.split(" = ")[0]) == int(step.split(" = ")[1])
            assert int(step.split(" = ")[1]) >= 0 and not re.search(r"(^|[ (])-\d", cur)   # no negative numbers
        q, lines, v = arith.addition(random.Random(i))
        assert eval(q[len("Compute "):-1]) == v
        q, lines, v = arith.multiplication(random.Random(i))
        assert eval(q[len("Compute "):-1]) == v
        q, lines, v = arith.division(random.Random(i))
        a, b = map(int, re.findall(r"\d+", q))
        assert v == (f"{a // b}" + (f" remainder {a % b}" if a % b else ""))
    assert minus > 100                                     # subtraction (and a - (b + c)) is generated
    d = arith.doc("test", 0)
    assert d["kind"] == "arith" and d["text"].isascii() and "Answer:" in d["text"]


def test_igsm_solutions():
    for i in range(200):
        text = igsm.doc("test", i)["text"]
        assert text.isascii() and text.startswith("Problem: ")
        for line in text.splitlines():
            if "Define " in line:
                parts = [p.strip() for p in line.split("; ", 1)[1].rstrip(".").split("=")]
                vals = [eval(p) for p in parts[2:]] or [eval(parts[1])]
                assert len(set(vals)) == 1
        answers = re.findall(r"Answer: (\d+)", text)
        assert answers and all(int(a) > 0 for a in answers)


# ------------------------------------------------------------------ FLD


@pytest.mark.skipif(not HAVE_FLD, reason="raw/fld_v2 missing")
def test_fld_helpers_on_real_rows(qwen):
    rows = fld.load_fld("validation", ("default",))[:200]
    labels = set()
    for rec in rows:
        user = fld.fld_prompt(rec)
        assert user.startswith("Facts:\nfact1: ") and f"Hypothesis: {rec['hypothesis']}" in user
        assert user.count("\nfact") == len(fld.facts(rec)) and len(fld.facts(rec)) == len(re.findall(r"sent\d+: ", rec["context"]))
        r = fld.fld_reasoning(rec, qwen)
        assert r is None or len(qwen.encode(r)) <= 128
        assert fld.fld_answer(rec) == f"Answer: {rec['proof_label']}"
        assert fld.parse_answer("blah\n" + fld.fld_answer(rec)) == rec["proof_label"]
        labels.add(rec["proof_label"])
        full = fld.fld_reasoning(rec, budget=None)
        assert r is None or r == full
        if rec["proof_label"] == "UNKNOWN":
            assert full.endswith(fld.UNKNOWN_END)
        elif rec["proofs"]:
            assert full.splitlines()[-1].endswith("the hypothesis.")
    assert labels == set(fld.LABELS)
    test = fld.load_fld_test(depths=(2, 3, 4), n=50)
    assert len(test) == 50 and all(t["depth"] in (2, 3, 4) for t in test)
    assert {t["config"] for t in fld.load_fld_test(depths=(4,))} == {"default", "star"}
    assert fld.parse_answer("Answer: proved") == "PROVED" and fld.parse_answer("no answer") is None


# ------------------------------------------------------------------ gen_formal worker


def test_gen_formal_worker(tok):
    from scripts import gen_formal
    idx, out, cnt = gen_formal.work((7, "shuffle", "t", 3))
    recs, counts = out["s0_shuffle_dyck"]
    assert idx == 7 and len(recs) == 3 and counts == [dyck.SEQ_SYMBOLS + 1] * 3
    _, out, _ = gen_formal.work((0, "logic_chat", "t", 5))
    recs, counts = out["fld_chat"]
    assert all(c > len(tok.encode(r["user"])) for r, c in zip(recs, counts))
