"""iGSM-style dependency word problems with step-by-step solutions for S1 synth_reasoning (plan section 10).

After Ye et al., "Physics of Language Models: Part 2.1" (iGSM): a hierarchy of categories (e.g. school ->
classroom -> backpack -> item); an instance parameter is "the number of <child> in each <parent>" for an edge
of the hierarchy. Each parameter is either given as a number or defined from earlier parameters (k more /
fewer, k times, sum, difference), so the parameters form a DAG; distractor parameters are stated but unused.
The question asks for one parameter, or for a total (the number of grandchildren in a parent, summed over the
children). The solution defines exactly the needed parameters in dependency order, one variable per line, with
the arithmetic written out. Plain integers (iGSM's mod-23 arithmetic is dropped), every value kept < 1000.
"""

from bonsai.formal import rng_for

# levels: [instance names (proper nouns)], then (singular, plural) category nouns per level
WORLDS = [
    (["Riverview High", "Oakwood Middle School", "Maple Elementary", "Cedar Academy", "Lincoln High"],
     [("art room", "art rooms"), ("science lab", "science labs"), ("music room", "music rooms"),
      ("computer lab", "computer labs"), ("library", "libraries"), ("gym", "gyms")],
     [("backpack", "backpacks"), ("locker", "lockers"), ("cabinet", "cabinets"), ("drawer", "drawers")],
     [("pencil", "pencils"), ("notebook", "notebooks"), ("ruler", "rulers"), ("eraser", "erasers"),
      ("marker", "markers"), ("crayon", "crayons")]),
    (["Sunny Zoo", "Green Valley Zoo", "River Park Zoo", "Hilltop Zoo"],
     [("reptile house", "reptile houses"), ("bird aviary", "bird aviaries"), ("monkey enclosure", "monkey enclosures"),
      ("aquarium", "aquariums"), ("lion den", "lion dens")],
     [("cage", "cages"), ("tank", "tanks"), ("nest box", "nest boxes"), ("feeding station", "feeding stations")],
     [("lizard", "lizards"), ("parrot", "parrots"), ("snake", "snakes"), ("turtle", "turtles"), ("frog", "frogs"),
      ("finch", "finches")]),
    (["Northtown", "Lakeside", "Brookfield", "Fairview", "Pine Hill"],
     [("bakery", "bakeries"), ("bookstore", "bookstores"), ("grocery store", "grocery stores"),
      ("toy shop", "toy shops"), ("pharmacy", "pharmacies")],
     [("shelf", "shelves"), ("rack", "racks"), ("display case", "display cases"), ("crate", "crates")],
     [("loaf of bread", "loaves of bread"), ("book", "books"), ("apple", "apples"), ("teddy bear", "teddy bears"),
      ("bottle", "bottles"), ("puzzle", "puzzles")]),
    (["Sunrise Farm", "Willow Farm", "Red Barn Farm", "Hilltop Ranch"],
     [("barn", "barns"), ("chicken coop", "chicken coops"), ("greenhouse", "greenhouses"), ("stable", "stables"),
      ("orchard", "orchards")],
     [("pen", "pens"), ("row", "rows"), ("stall", "stalls"), ("bin", "bins")],
     [("cow", "cows"), ("hen", "hens"), ("tomato plant", "tomato plants"), ("horse", "horses"),
      ("apple tree", "apple trees"), ("goat", "goats")]),
    (["Harbor Mall", "Central Station", "Westfield Plaza", "Union Market"],
     [("food court", "food courts"), ("parking garage", "parking garages"), ("clothing store", "clothing stores"),
      ("electronics store", "electronics stores")],
     [("aisle", "aisles"), ("floor", "floors"), ("counter", "counters"), ("section", "sections")],
     [("table", "tables"), ("car", "cars"), ("jacket", "jackets"), ("phone", "phones"), ("lamp", "lamps")]),
]
LETTERS = "abcdefghjkmnpqrstuvwxyz"


def _problem(rng):
    world = rng.choice(WORLDS)
    n_levels = rng.choice((3, 3, 4))
    levels = [[(n, n) for n in rng.sample(world[0], rng.randint(1, 2))]]
    for lv in world[1:n_levels]:
        levels.append(rng.sample(lv, rng.randint(2, 3)))
    edges = []
    for i in range(len(levels) - 1):
        for child in levels[i + 1]:
            for parent in rng.sample(levels[i], rng.randint(1, min(2, len(levels[i])))):
                edges.append((parent, child))
    rng.shuffle(edges)
    # definitions in topological order: (kind, k, deps)
    defs, values = {}, {}
    for j, e in enumerate(edges):
        earlier = edges[:j]
        r = rng.random()
        if not earlier or r < 0.3:
            defs[e], values[e] = ("given", rng.randint(2, 12), ()), None
        elif r < 0.55:
            d = rng.choice(earlier)
            defs[e] = rng.choice([("more", rng.randint(1, 9), (d,)), ("times", rng.randint(2, 4), (d,)),
                                  ("fewer", rng.randint(1, 5), (d,))])
        else:
            a, b = rng.sample(earlier, 2) if len(earlier) > 1 else (earlier[0], earlier[0])
            if a == b:
                defs[e] = ("more", rng.randint(1, 9), (a,))
            else:
                defs[e] = rng.choice([("sum", 0, (a, b)), ("diff", 0, (a, b)), ("sum_times", rng.randint(2, 3), (a, b))])
    for e in edges:
        kind, k, deps = defs[e]
        v = [values[d] for d in deps]
        values[e] = {"given": lambda: k, "more": lambda: v[0] + k, "times": lambda: v[0] * k,
                     "fewer": lambda: v[0] - k, "sum": lambda: v[0] + v[1], "diff": lambda: v[0] - v[1],
                     "sum_times": lambda: k * (v[0] + v[1])}[kind]()
        if not 0 < values[e] < 1000:
            return None
    return levels, edges, defs, values


def _name(e):
    (parent, _), (_, child_pl) = e
    return f"the number of {child_pl} in each {parent}"


def _sentence(e, defs):
    kind, k, deps = defs[e]
    (parent, _), (_, child_pl) = e
    if kind == "given":
        return f"Each {parent} has {k} {child_pl}."
    p = [_name(d) for d in deps] + [""]
    rhs = {"more": "{k} more than {0}", "times": "{k} times {0}", "fewer": "{k} fewer than {0}",
           "sum": "the sum of {0} and {1}", "diff": "the difference between {0} and {1}",
           "sum_times": "{k} times the sum of {0} and {1}"}[kind].format(*p, k=k)
    return f"{_name(e)[0].upper()}{_name(e)[1:]} equals {rhs}."


def _needed(targets, defs):
    out, seen = [], set()

    def visit(e):
        if e in seen:
            return
        seen.add(e)
        for d in defs[e][2]:
            visit(d)
        out.append(e)

    for t in targets:
        visit(t)
    return out


def problem(rng):
    """(question text, solution lines, answer) or None when the draw is degenerate."""
    got = _problem(rng)
    if got is None:
        return None
    levels, edges, defs, values = got
    depth = {}
    for e in edges:
        depth[e] = 1 + max((depth[d] for d in defs[e][2]), default=0)
    totals = [(a, c, [(a, b) for b in levels[i + 1] if (a, b) in defs and (b, c) in defs])
              for i in range(len(levels) - 2) for a in levels[i] for c in levels[i + 2]]
    totals = [t for t in totals if t[2]]
    if totals and rng.random() < 0.35:
        a, c, via = rng.choice(totals)
        targets = [x for b_edge in via for x in (b_edge, (b_edge[1], c))]
        answer = sum(values[ab] * values[(ab[1], c)] for ab in via)
        question = f"How many {c[1]} are there in total in each {a[0]}?"
    else:
        need = min(max(depth.values()), rng.randint(2, 4))
        target = rng.choice([e for e in edges if depth[e] >= need])
        targets, answer = [target], values[target]
        question = f"What is {_name(target)}?"
    if answer >= 10 ** 5:
        return None
    sentences = [_sentence(e, defs) for e in edges]
    rng.shuffle(sentences)
    var = dict(zip(_needed(targets, defs), rng.sample(LETTERS, len(LETTERS))))
    lines = []
    for e, x in var.items():
        kind, k, deps = defs[e]
        v = [var[d] for d in deps]
        n = [values[d] for d in deps]
        expr = {"given": f"{k}", "more": f"{v[0] if v else ''} + {k} = {n[0] if n else ''} + {k} = {values[e]}",
                "times": f"{k} * {v[0] if v else ''} = {k} * {n[0] if n else ''} = {values[e]}",
                "fewer": f"{v[0] if v else ''} - {k} = {n[0] if n else ''} - {k} = {values[e]}",
                "sum": f"{' + '.join(v)} = {' + '.join(map(str, n))} = {values[e]}",
                "diff": f"{' - '.join(v)} = {' - '.join(map(str, n))} = {values[e]}",
                "sum_times": f"{k} * ({' + '.join(v)}) = {k} * {sum(n)} = {values[e]}"}[kind]
        lines.append(f"Define {_name(e)} as {x}; {x} = {expr}.")
    if len(targets) > 1:
        (a, _), c = targets[0], targets[1][1]
        pairs = [(var[targets[i]], var[targets[i + 1]], values[targets[i]] * values[targets[i + 1]])
                 for i in range(0, len(targets), 2)]
        lines.append(f"Each {a[0]} has " + " + ".join(f"{p} * {q}" for p, q, _ in pairs)
                     + (f" = {' + '.join(str(r) for _, _, r in pairs)}" if len(pairs) > 1 else "")
                     + f" = {answer} {c[1]} in total.")
    return " ".join(sentences) + " " + question, lines, answer


def doc(*key):
    rng = rng_for("igsm", *key)
    parts, n = [], rng.randint(1, 3)
    while len(parts) < n:
        got = problem(rng)
        if got is None:
            continue
        q, lines, ans = got
        parts.append(f"Problem: {q}\nSolution: " + "\n".join(lines) + f"\nAnswer: {ans}")
    return {"text": "\n\n".join(parts), "kind": "igsm"}
