"""Multi-step arithmetic scratchpads for S1 synth_reasoning (plan section 10: "多步算术 scratchpad").

A document is a worksheet of 3-8 worked problems; every intermediate result is written out and checked by
construction (Python integers / fractions). Kinds: column addition with carries, column subtraction with
borrows, long multiplication (partial products), long division (bring down, remainder), expressions with
precedence and parentheses (one operation per line), fractions (common denominator, simplification),
percentages, one-variable linear equations. Plain ASCII.
"""

from fractions import Fraction
from math import gcd

from bonsai.formal import rng_for

PLACES = ["ones", "tens", "hundreds", "thousands", "ten-thousands", "hundred-thousands", "millions"]
HEADERS = ["Worked examples:", "Practice problems with solutions.", "Arithmetic, step by step.",
           "Show your work:", "Solutions:", "Exercises and worked answers."]


def _num(rng, lo_digits, hi_digits):
    d = rng.randint(lo_digits, hi_digits)
    return rng.randint(10 ** (d - 1), 10 ** d - 1)


def addition(rng):
    nums = [_num(rng, 2, 6) for _ in range(rng.choice((2, 2, 3)))]
    width = max(len(str(n)) for n in nums)
    lines, carry, digits = [], 0, []
    for p in range(width):
        col = [n // 10 ** p % 10 for n in nums if n >= 10 ** p]
        s = sum(col) + carry
        terms = " + ".join(map(str, col)) + (f" + {carry}" if carry else "")
        if len(col) == 1 and not carry:
            lines.append(f"{PLACES[p]}: {s}, write {s}")
        else:
            lines.append(f"{PLACES[p]}: {terms} = {s}, write {s % 10}, carry {s // 10}")
        digits.append(s % 10)
        carry = s // 10
    total = sum(nums)
    if carry:
        lines.append(f"the final carry {carry} becomes the leading digit")
    q = " + ".join(map(str, nums))
    return f"Compute {q}.", lines + [f"So {q} = {total}."], total


def subtraction(rng):
    a, b = sorted((_num(rng, 3, 6), _num(rng, 2, 6)), reverse=True)
    lines, borrow = [], 0
    for p in range(len(str(a))):
        da, db = a // 10 ** p % 10 - borrow, b // 10 ** p % 10
        if da < db:
            lines.append(f"{PLACES[p]}: {da} < {db}, borrow: {da + 10} - {db} = {da + 10 - db}")
            borrow = 1
        else:
            lines.append(f"{PLACES[p]}: {da} - {db} = {da - db}")
            borrow = 0
    return f"Compute {a} - {b}.", lines + [f"So {a} - {b} = {a - b}."], a - b


def multiplication(rng):
    a, b = _num(rng, 2, 4), _num(rng, 2, 3)
    lines, parts = [], []
    for p, d in enumerate(reversed(str(b))):
        if d == "0":
            continue
        part = a * int(d) * 10 ** p
        lines.append(f"{a} * {int(d) * 10 ** p} = {part}")
        parts.append(part)
    if len(parts) > 1:
        lines.append(" + ".join(map(str, parts)) + f" = {a * b}")
    return f"Compute {a} * {b}.", lines + [f"So {a} * {b} = {a * b}."], a * b


def division(rng):
    b = rng.randint(2, 40)
    q = _num(rng, 2, 4)
    a = b * q + (rng.randrange(b) if rng.random() < 0.4 else 0)
    s, lines, cur, i = str(a), [], 0, 0
    while i < len(s) and cur * 10 + int(s[i]) < b:
        cur = cur * 10 + int(s[i])
        i += 1
    first = True
    while i < len(s):
        cur = cur * 10 + int(s[i])
        d = cur // b
        lead = "" if first else f"bring down {s[i]}: "
        lines.append(f"{lead}{cur} / {b} = {d}, {d} * {b} = {d * b}, {cur} - {d * b} = {cur - d * b}")
        cur -= d * b
        first = False
        i += 1
    r = a % b
    res = f"{a // b}" + (f" remainder {r}" if r else "")
    return f"Compute {a} / {b}.", lines + [f"So {a} / {b} = {res}."], res


def _expr(rng, depth):
    if depth == 0 or rng.random() < 0.3:
        v = rng.randint(1, 20)
        return str(v), v, 0
    op = rng.choice("+-*")
    ls, lv, lp = _expr(rng, depth - 1)
    rs, rv, rp = _expr(rng, depth - 1)
    prec = 1 if op in "+-" else 2
    if lp and lp < prec:
        ls = f"({ls})"
    if rp and (rp < prec or (op == "-" and rp == 1)):
        rs = f"({rs})"
    v = {"+": lv + rv, "-": lv - rv, "*": lv * rv}[op]
    return f"{ls} {op} {rs}", v, prec


def _reduce_once(expr):
    """Evaluate the first operation of `expr` in standard order: (new expression string, step line, its value)."""
    toks = expr.replace("(", " ( ").replace(")", " ) ").split()

    def find(lo, hi):
        depth, start = 0, None
        for i in range(lo, hi):
            if toks[i] == "(":
                if depth == 0:
                    start = i
                depth += 1
            elif toks[i] == ")":
                depth -= 1
                if depth == 0:
                    return find(start + 1, i)
        for ops in ("*", "+-"):
            for i in range(lo, hi):
                if toks[i] in ops and len(toks[i]) == 1:
                    return i
        return None

    i = find(0, len(toks))
    a, op, b = int(toks[i - 1]), toks[i], int(toks[i + 1])
    v = {"+": a + b, "-": a - b, "*": a * b}[op]
    new = toks[:i - 1] + [str(v)] + toks[i + 2:]
    out = []
    j = 0
    while j < len(new):   # drop parentheses around a single number
        if new[j] == "(" and j + 2 < len(new) and new[j + 2] == ")":
            out.append(new[j + 1])
            j += 3
        else:
            out.append(new[j])
            j += 1
    s = " ".join(out).replace("( ", "(").replace(" )", ")")
    return s, f"{a} {op} {b} = {v}", v


def expression(rng):
    while True:   # every intermediate result stays >= 0 (no negative numbers, so no "a - -b")
        s, v, _ = _expr(rng, rng.randint(2, 3))
        if not (0 <= v < 10 ** 5 and s.count(" ") >= 4):
            continue
        lines, cur, ok = [], s, True
        while ok and any(f" {o} " in cur for o in "+-*"):
            cur, step, x = _reduce_once(cur)
            ok = x >= 0
            lines.append(f"{step}, so: {cur}")
        if ok:
            return f"Evaluate {s}.", lines + [f"So {s} = {v}."], v


def fractions(rng):
    a, b, c = rng.randint(1, 9), rng.randint(2, 12), rng.randint(1, 9)
    d = rng.choice([x for x in range(2, 13) if x != b])
    op = rng.choice("+-")
    if op == "-" and Fraction(a, b) < Fraction(c, d):
        a, b, c, d = c, d, a, b
    m = b * d // gcd(b, d)
    x, y = a * (m // b), c * (m // d)
    n = x + y if op == "+" else x - y
    lines = [f"the common denominator of {b} and {d} is {m}",
             f"{a}/{b} = {x}/{m} and {c}/{d} = {y}/{m}",
             f"{x}/{m} {op} {y}/{m} = {n}/{m}"]
    f = Fraction(n, m)
    if f.denominator != m and n:
        g = gcd(n, m)
        lines.append(f"divide top and bottom by {g}: {n}/{m} = {f.numerator}/{f.denominator}")
    res = str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
    return f"Compute {a}/{b} {op} {c}/{d}.", lines + [f"So {a}/{b} {op} {c}/{d} = {res}."], res


def percent(rng):
    p = rng.choice((5, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 80, 90))
    base = rng.randint(2, 60) * 20
    prod = base * p
    lines = [f"{p}% of {base} = {base} * {p} / 100", f"{base} * {p} = {prod}", f"{prod} / 100 = {prod // 100}"]
    if prod % 100:
        return percent(rng)
    return f"What is {p}% of {base}?", lines + [f"So {p}% of {base} is {prod // 100}."], prod // 100


def linear(rng):
    a, x, b = rng.randint(2, 12), rng.randint(-10, 20), rng.randint(1, 50)
    c = a * x + b
    lines = [f"subtract {b} from both sides: {a}x = {c} - {b} = {c - b}",
             f"divide both sides by {a}: x = {c - b} / {a} = {x}",
             f"check: {a} * {x} + {b} = {a * x} + {b} = {c}"]
    return f"Solve {a}x + {b} = {c}.", lines + [f"So x = {x}."], x


KINDS = [addition, subtraction, multiplication, division, expression, fractions, percent, linear]


def doc(*key):
    rng = rng_for("arith", *key)
    kinds = [rng.choice(KINDS)] * rng.randint(3, 6) if rng.random() < 0.4 else \
        [rng.choice(KINDS) for _ in range(rng.randint(3, 8))]
    parts = [rng.choice(HEADERS)]
    for i, kind in enumerate(kinds):
        q, lines, ans = kind(rng)
        parts.append(f"Problem {i + 1}. {q}\n" + "\n".join(lines) + f"\nAnswer: {ans}")
    return {"text": "\n\n".join(parts), "kind": "arith"}
