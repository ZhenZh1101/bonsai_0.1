"""k-Shuffle Dyck and k-Dyck for S0 (plan section 9) and the Dyck check at the S0 -> S1 boundary (plan 14).

Re-implemented from the description in Hu et al., "Between Circuits and Chomsky: Pre-pretraining on Formal
Languages Imparts Linguistic Biases" (ACL 2025) and michahu/pre-pretraining (no license, nothing copied):

  k-Shuffle Dyck  k bracket types that may close in any order (a closer is valid when its type is open). At
                  depth 0 open, at max_depth (8) close, otherwise open with p_open 0.5 (uniform type) or close
                  a uniformly chosen open type. One document = one 512-symbol sequence, cut at the length (the
                  tail may stay unbalanced, as in the paper).
  k-Dyck          properly nested brackets: Dyck words (one opening bracket, then open with p 0.5 below
                  max_depth 16, else close the top; the word ends when the stack is empty) concatenated up to
                  512 symbols, the last word cut.

Representation (ASCII, each symbol one Qwen3 token in context and in the student subset, space separated):
  shuffle  k = 26, the whole single-token letter alphabet: open " a".." z", close " A".." Z"
  k-Dyck   k = 20 bracket-like pairs ( ) [ ] { } < > (( )) {{ }} << >> ((( ))) {{{ }}} ([ ]) ({ }) (* *) /* */
           <-- --> (" ") [" "] (' ') [' '] {' '} (: :) -- the mirror pairs that are single tokens in the
           65,536-id subset, minus ones with another meaning (-> <- => </ />) used elsewhere
A document is 512 symbols, so BOS + document is exactly one S0 row (seq_len 512 + 1).

`dyck_accuracy` scores greedy argmax on held-out sequences (a seed stream never used for training data):
  kdyck_close_acc     positions whose true next symbol is a closer: argmax over the k closers == the match
  kdyck_valid_rate    every position: full-vocabulary argmax is a valid continuation of the k-Dyck prefix
  shuffle_valid_rate  every position: full-vocabulary argmax is a valid continuation of the shuffle prefix
  shuffle_close_acc   closing positions: argmax over the closers closes some open type
"""

import numpy as np

from bonsai.formal import rng_for, spaced

SEQ_SYMBOLS = 512
SHUFFLE_OPEN = [chr(c) for c in range(ord("a"), ord("z") + 1)]
SHUFFLE_CLOSE = [c.upper() for c in SHUFFLE_OPEN]
DYCK_PAIRS = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">"), ("((", "))"), ("{{", "}}"), ("<<", ">>"),
              ("(((", ")))"), ("{{{", "}}}"), ("([", "])"), ("({", "})"), ("(*", "*)"), ("/*", "*/"),
              ("<--", "-->"), ('("', '")'), ('["', '"]'), ("('", "')"), ("['", "']"), ("{'", "'}"), ("(:", ":)")]
K_SHUFFLE, K_DYCK = len(SHUFFLE_OPEN), len(DYCK_PAIRS)
SHUFFLE_MAX_DEPTH, DYCK_MAX_DEPTH = 8, 16
LANGS = {  # name -> (open symbols, close symbols, max depth)
    "shuffle": (SHUFFLE_OPEN, SHUFFLE_CLOSE, SHUFFLE_MAX_DEPTH),
    "dyck": ([a for a, _ in DYCK_PAIRS], [b for _, b in DYCK_PAIRS], DYCK_MAX_DEPTH),
}


def shuffle_dyck(rng, k=K_SHUFFLE, length=SEQ_SYMBOLS, p_open=0.5, max_depth=SHUFFLE_MAX_DEPTH):
    """Symbol ids: i < k opens type i, k + i closes it."""
    seq, counts, depth = [], [0] * k, 0
    while len(seq) < length:
        if depth == 0 or (depth < max_depth and rng.random() < p_open):
            t = rng.randrange(k)
            counts[t] += 1
            depth += 1
            seq.append(t)
        else:
            t = rng.choice([i for i, c in enumerate(counts) if c])
            counts[t] -= 1
            depth -= 1
            seq.append(k + t)
    return seq


def dyck_word(rng, k=K_DYCK, p_open=0.5, max_depth=DYCK_MAX_DEPTH, limit=SEQ_SYMBOLS):
    stack = [rng.randrange(k)]
    word = [stack[0]]
    while stack and len(word) < limit:
        if len(stack) < max_depth and rng.random() < p_open:
            stack.append(rng.randrange(k))
            word.append(stack[-1])
        else:
            word.append(k + stack.pop())
    return word


def dyck(rng, k=K_DYCK, length=SEQ_SYMBOLS, p_open=0.5, max_depth=DYCK_MAX_DEPTH):
    seq = []
    while len(seq) < length:
        seq += dyck_word(rng, k, p_open, max_depth)
    return seq[:length]


def to_text(seq, lang):
    opens, closes, _ = LANGS[lang]
    k = len(opens)
    return spaced(opens[s] if s < k else closes[s - k] for s in seq)


def parse(text, lang):
    opens, closes, _ = LANGS[lang]
    index = {s: i for i, s in enumerate(opens + closes)}
    return [index[s] for s in text.split()]


def valid_next(prefix, lang):
    """Per position t: (set of valid symbol ids at t given prefix[:t]) for t in 0..len(prefix)-1."""
    opens, _, max_depth = LANGS[lang]
    k = len(opens)
    out, counts, stack = [], [0] * k, []
    for s in prefix:
        ok = set(range(k)) if len(stack) < max_depth else set()
        if lang == "dyck":
            if stack:
                ok.add(k + stack[-1])
        else:
            ok |= {k + i for i, c in enumerate(counts) if c}
        out.append(ok)
        if s < k:
            stack.append(s)
            counts[s] += 1
        else:
            t = s - k
            if lang == "dyck":
                if stack and stack[-1] == t:
                    stack.pop()
            elif counts[t]:
                counts[t] -= 1
                stack.pop()
    return out


def is_valid(seq, lang):
    """True if every symbol of seq is a valid continuation of its prefix (the tail may stay open)."""
    return all(s in ok for s, ok in zip(seq, valid_next(seq, lang)))


def doc(lang, *key):
    rng = rng_for("train", lang, *key)
    seq = shuffle_dyck(rng) if lang == "shuffle" else dyck(rng)
    return {"text": to_text(seq, lang)}


def symbol_ids(tok, lang):
    """Student ids of the open and close symbols (each must be exactly one token)."""
    opens, closes, _ = LANGS[lang]
    ids = []
    for s in opens + closes:
        got = tok.encode(" " + s)
        if len(got) != 1:
            raise ValueError(f"{lang} symbol {s!r} is {len(got)} student tokens")
        ids.append(got[0])
    return np.array(ids, dtype=np.int64)


def dyck_accuracy(model, tok, n=256, seed=0, batch=4, length=SEQ_SYMBOLS):
    """Greedy next-symbol accuracy on n held-out k-Dyck and n k-Shuffle-Dyck sequences (BOS first)."""
    import mlx.core as mx

    out = {"n": n, "k_dyck": K_DYCK, "k_shuffle": K_SHUFFLE}
    for lang in ("dyck", "shuffle"):
        rng = rng_for("eval", lang, seed)
        seqs = [dyck(rng, length=length) if lang == "dyck" else shuffle_dyck(rng, length=length) for _ in range(n)]
        ids = symbol_ids(tok, lang)
        k = len(ids) // 2
        lookup = {int(t): i for i, t in enumerate(ids)}
        valid = close_ok = n_pos = n_close = 0
        for b in range(0, n, batch):
            chunk = seqs[b:b + batch]
            x = np.array([[tok.bos_id] + [int(ids[s]) for s in q[:-1]] for q in chunk], dtype=np.int64)
            logits = model(mx.array(x))
            full = np.array(mx.argmax(logits, axis=-1))
            closer = np.array(mx.argmax(mx.take(logits, mx.array(ids[k:]), axis=-1), axis=-1))
            for r, q in enumerate(chunk):
                for t, ok in enumerate(valid_next(q, lang)):
                    n_pos += 1
                    valid += lookup.get(int(full[r, t]), -1) in ok
                    if q[t] >= k:
                        n_close += 1
                        close_ok += (k + int(closer[r, t]) == q[t]) if lang == "dyck" else (k + int(closer[r, t]) in ok)
        name = "kdyck" if lang == "dyck" else "shuffle"
        out[f"{name}_valid_rate"] = valid / max(n_pos, 1)
        out[f"{name}_close_acc"] = close_ok / max(n_close, 1)
        out[f"{name}_close_positions"] = n_close
    return out
