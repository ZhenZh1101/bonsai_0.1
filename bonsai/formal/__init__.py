"""Formal-language and synthetic-reasoning generators (plan sections 9, 10, 15).

    dyck         k-Shuffle Dyck / k-Dyck (S0) + `dyck_accuracy`, the S0 -> S1 check of plan section 14
    mp_struct    MP-STRUCT (S0), ported from osekilab/LAD-PPT (MIT)
    derivations  formal derivations `premises: ... ; derived: ... ; goal: ... ; next: ...` (S0) from ProcPile,
                 FOL-Traces and the logic engine; the rc_short chat pool (S2) from Reasoning Core data
    logic        synthetic 2-4 hop deductions: natural-language FLD-style problems (S2) and ASCII FOL proofs (S0)
    fld          FLD.v2 -> chat fields, plain-text derivations, the FLD eval set
    arith, igsm  multi-step arithmetic scratchpads and iGSM-style dependency problems (S1 synth_reasoning)

`scripts/gen_formal.py` writes the pools (DESIGN.md 1.1). S0 text is ASCII and every Dyck / MP-STRUCT symbol is
one space-prefixed Qwen3 token, so a sequence of n symbols is exactly n tokens.
"""

import hashlib
import random


def spaced(symbols):
    """Symbols -> text; each symbol carries its leading space, so it is one token in context."""
    return "".join(" " + s for s in symbols)


def stable_bucket(text, n=100):
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % n


def rng_for(*key):
    """A random.Random seeded by a string key (deterministic across processes and runs)."""
    return random.Random("/".join(map(str, key)))
