"""MP-STRUCT for S0 (plan section 9): Minimalist-Program derivations (Merge + Agree + Move) as bracketed trees.

Ported from osekilab/LAD-PPT, src/mp_struct.py (Mita, Someya, Yoshida, Oseki, "Language Acquisition Device in
Large Language Models", ACL 2026), with the defaults of its `generate` command: strip_lexical, agree_match_ratio
1.0, num_prior 0.5, p_wh_goal 0.2, p_c_wh 0.2, T has EPP, merge + agree + move on. Each sentence is one CP:
subject and object DPs with a number feature, T agrees with the subject (Agree) and the subject moves to Spec-TP
(EPP), a +wh C attracts a [-wh] object or subject to Spec-CP (Move) leaving a trace; the VP orders its two
arguments at random. Sentences are concatenated and cut at 512 symbols, like the original's max_length.
The original's wh-movement never picks Spec-TP (its condition is unsatisfiable); kept as is.

Changes: the derivation is built with plain tuples; symbols are mapped to single-token ASCII words (SYMBOLS,
every one a single Qwen3 token in the student subset); a random.Random is passed in.

LAD-PPT license (MIT):
    Copyright (c) 2026 OsekiLab
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without
    limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
    Software, and to permit persons to whom the Software is furnished to do so, subject to the following
    conditions: The above copyright notice and this permission notice shall be included in all copies or
    substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
    TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
"""

from bonsai.formal import rng_for, spaced

SEQ_SYMBOLS = 512
# LAD-PPT symbol -> single-token word (uNum on T is lower case: an uninterpretable feature valued by Agree)
SYMBOLS = {
    "[": "[", "]": "]", "CP": "CP", "TP": "TP", "VP": "VP", "C": "C", "C(+wh)": "Q", "MOV[+wh]": "MOV",
    "TR[DP]": "TR", "TR[-wh]": "GAP", "[-wh]": "WH", "DP[Num:sg]": "SG", "DP[Num:pl]": "PL",
    "T(+EPP,uNum:sg)": "sg", "T(+EPP,uNum:pl)": "pl", "V": "V", "D": "D", "N": "N",
}
WORDS = {w: s for s, w in SYMBOLS.items()}


class _DP:
    __slots__ = ("num", "wh", "trace")

    def __init__(self, num, wh, trace=False):
        self.num, self.wh, self.trace = num, wh, trace


def _lin_dp(dp):
    if dp.trace:
        return ["TR[-wh]"] if dp.wh else ["TR[DP]"]
    return ["[", f"DP[Num:{dp.num}]", "[", "D", "]", "[", "N", "]", "]"] + (["[-wh]"] if dp.wh else [])


def sentence(rng, agree_match=True, num_prior=0.5, p_wh_goal=0.2, p_c_wh=0.2):
    """One CP derivation as LAD-PPT symbols."""
    subj_num = "sg" if rng.random() < num_prior else "pl"
    obj_num = subj_num if agree_match else ("pl" if subj_num == "sg" else "sg")
    subj = _DP(subj_num, rng.random() < p_wh_goal)
    obj = _DP(obj_num, rng.random() < p_wh_goal)
    t_num = subj.num                                       # Agree: T probes, the closest goal is the subject
    spec_tp = _DP(subj.num, subj.wh)                       # EPP: the subject moves to Spec-TP ...
    subj.trace = True                                      # ... leaving a trace in the VP
    c_wh = rng.random() < p_c_wh
    spec_cp = None
    if c_wh:                                               # wh-movement of the first [-wh] goal
        goal = obj if obj.wh else (subj if subj.wh else None)
        if goal is not None:
            spec_cp = _DP(goal.num, False)
            goal.trace = True
    first, second = (obj, subj) if rng.random() < 0.5 else (subj, obj)
    vp = ["[", "VP", "V", "["] + _lin_dp(first) + ["]", "["] + _lin_dp(second) + ["]", "]"]
    tp = ["[", "TP", "["] + _lin_dp(spec_tp) + ["]", "[", f"T(+EPP,uNum:{t_num})", "]", "["] + vp + ["]", "]"]
    spec = ["MOV[+wh]", "["] + _lin_dp(spec_cp) + ["]"] if spec_cp else []
    return ["[", "CP"] + spec + ["[", "C(+wh)" if c_wh else "C", "]", "["] + tp + ["]", "]"]


def sequence(rng, length=SEQ_SYMBOLS, agree_match_ratio=1.0):
    agree_match = rng.random() < agree_match_ratio
    out = []
    while len(out) < length:
        out += sentence(rng, agree_match)
    return out[:length]


def to_text(symbols):
    return spaced(SYMBOLS[s] for s in symbols)


def doc(*key):
    return {"text": to_text(sequence(rng_for("train", "mp_struct", *key)))}


def parse_tree(words):
    """Nested lists from " [" / " ]" words; returns (complete top-level trees, unfinished tail)."""
    stack, trees = [[]], []
    for w in words:
        if w == "[":
            stack.append([])
        elif w == "]":
            if len(stack) == 1:
                raise ValueError("unbalanced ]")
            node = stack.pop()
            stack[-1].append(node)
            if len(stack) == 1:
                trees.append(stack[0].pop())
        else:
            stack[-1].append(w)
    return trees, len(stack) > 1


def check_sentence(tree):
    """Structural check of one complete CP tree in word form; raises ValueError on a violation."""
    if not tree or tree[0] != "CP":
        raise ValueError("not a CP")
    rest = tree[1:]
    if rest[0] == "MOV":
        rest = rest[2:]
    if len(rest) != 2 or rest[0] not in (["C"], ["Q"]):
        raise ValueError("bad C")
    tp = rest[1][0]
    if tp[0] != "TP" or len(tp) != 4:
        raise ValueError("bad TP")
    spec, t_head = tp[1][0], tp[2]
    if spec[0] not in ("SG", "PL") or t_head not in (["sg"], ["pl"]):
        raise ValueError("bad Spec-TP / T")
    if spec[0].lower() != t_head[0]:
        raise ValueError("T does not agree with its subject")
    vp = tp[3][0]
    if vp[:2] != ["VP", "V"] or len(vp) != 4:
        raise ValueError("bad VP")
    return True
