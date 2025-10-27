"""Generation for any Bonsai model with the student tokenizer (plan sections 6 and 13).

  generate          one sequence with a KV cache (plain completion: fixed-prompt samples, base-model probes)
  generate_batch    left-padded batched decoding with per-row stops and an optional thinking budget
  chat / chat_batch Qwen3 chat with the thinking budget and the Python tool loop -> ChatResult
                    (evals, scripts/chat.py, on-policy distillation)
  span_logprobs, final_logits   scoring helpers for the evals (head only on the positions that are needed, in
                    fp32: bf16 logits of ~20 are 0.125 apart, so close MC letters would tie and argmax pick "A")
  ScriptedModel     deterministic stand-in model for tests (next token = policy(history))
  cast_for_inference  bf16 weight storage for eval-only processes (~30% faster decoding, same numbers)

Batching: prompts are left-padded to one length. The SDPA mask (boolean, True = attend) is causal and hides
padding keys; every query also sees its own key, so padding positions stay finite (a NaN in a masked value row
still poisons the attention product). RoPE uses the cache offset, the same for all rows: left padding shifts all
real tokens of a row by the same amount and attention only sees relative positions, so a padded row computes what
it would alone. Finished rows are fed padding until the batch is compacted (when at most half is still running).

Thinking budget: inside an open <think> block a row may sample `think_budget` tokens. generate_batch then stops
the row; chat_batch forces chat.think_stop(tools) + "\\n</think>\\n\\n" (fed one token per step, flagged `forced`;
the "..., using the python tool." variant on rows that offer the tool, as the teacher's data has it) and the row
continues with its answer. enable_thinking=False prefills "<think>\\n\\n</think>\\n\\n" as part of the prompt (the
template does it). Soft switches (/think, /no_think) are plain user text.

Tool loop (plan 13): when tools are given and an assistant turn ends with <|im_end|> and holds a <tool_call>, the
call runs in bonsai.sandbox (run_tool_call) and the exact template continuation is appended:
"\\n" <|im_start|> "user\\n" <tool_response> "\\n" output "\\n" </tool_response> <|im_end|> "\\n" <|im_start|>
"assistant\\n" (+ the prefilled empty think block when enable_thinking=False), with real control tokens and the
output encoded as ordinary text, built with bonsai.chat's pieces so the ids equal chat.render of the whole
conversation (non-strict rule: turns after the last user query keep their think block). Then every such row is
re-prefilled and continues. At most `max_tool_calls` calls; a turn that asks for more ends the conversation.

ChatResult.ids = BOS + the rendered prompt + everything sampled or appended (student ids, authoritative);
`sampled` / `forced` / `tool_output` / `oov` are bool arrays over ids (`oov`: produced by an out-of-subset split,
prompt and forced text only); `messages` is a readable reconstruction (reasoning_content, tool_calls, tool turns).
The presence penalty counts the tokens the model sampled in the current assistant turn (like a fresh request to
an OpenAI-style server after each tool result, and like bonsai.teacher), never the prompt or forced text.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from bonsai import chat as C
from bonsai.model import make_caches
from bonsai.sandbox import run_tool_call

PREFILL_CHUNK = 256
SAMPLED, FORCED = 1, 2


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.0   # 0 = greedy
    top_p: float = 1.0
    top_k: int = 0             # 0 = off
    presence_penalty: float = 0.0


GREEDY = Sampling()
THINK_SAMPLING = Sampling(0.6, 0.95, 20, 1.5)      # Qwen3's recommendation, thinking mode
NO_THINK_SAMPLING = Sampling(0.7, 0.8, 20, 1.5)    # non-thinking mode
PRESETS = {"greedy": GREEDY, "think": THINK_SAMPLING, "no_think": NO_THINK_SAMPLING}


def as_sampling(s):
    if s is None:
        return GREEDY
    if isinstance(s, Sampling):
        return s
    if isinstance(s, str):
        return PRESETS[s]
    return Sampling(**s)


def default_stops(tok):
    return {tok.special("<|im_end|>"), tok.bos_id}


# ---------------------------------------------------------------- forward helpers


def cast_for_inference(model):
    """Store the weights in the compute dtype (bf16): ~30% faster decoding, same numbers (every matmul casts its
    fp32 master weight to the compute dtype anyway). Only for models that are not being trained."""
    from mlx.utils import tree_map
    model.update(tree_map(lambda p: p.astype(model.compute_dtype), model.parameters()))
    mx.eval(model.parameters())
    return model


def _head(model):
    return model.embed_tokens.weight if model.cfg.tie_embeddings else model.lm_head.weight


def last_logits(model, ids, caches=None, mask=None):
    """fp32 logits (B, V) of the last position; the head runs on that position only. A model object may supply
    its own `last_logits(ids, caches, mask)` (test stubs)."""
    fn = getattr(model, "last_logits", None)
    if fn is not None:
        return fn(ids, caches, mask)
    x = model.hidden(ids, caches, mask)[:, -1, :]
    return (x @ _head(model).astype(x.dtype).T).astype(mx.float32)


class ScriptedModel:
    """Deterministic stand-in for a Bonsai model in tests: the next token is `policy(history)`, history = the
    row's ids so far without its left padding (pad_id = tok.bos_id, as the engine pads). The history lives in the
    KV cache, so it stays aligned when the batch is compacted."""

    def __init__(self, policy, vocab_size, pad_id, seq_len=4096):
        from types import SimpleNamespace
        self.cfg = SimpleNamespace(n_layer=1, seq_len=seq_len, vocab_size=vocab_size, tie_embeddings=True)
        self.policy, self.pad = policy, pad_id
        self.calls = 0

    def last_logits(self, ids, caches, mask=None):
        self.calls += 1
        x = ids.astype(mx.float32)[:, None, :, None]
        keys, _ = caches[0].update(x, x)
        out = np.full((ids.shape[0], self.cfg.vocab_size), -1e4, dtype=np.float32)
        for b, row in enumerate(np.array(keys[:, 0, :, 0]).astype(np.int64).tolist()):
            i = 0
            while i + 1 < len(row) and row[i] == self.pad and row[i + 1] == self.pad:
                i += 1
            out[b, int(self.policy(row[i:]))] = 0.0
        return mx.array(out)


def scripted_policy(targets, end_id):
    """Policy that continues whichever target sequence the history is a prefix of (else emits end_id)."""
    targets = [list(map(int, t)) for t in targets]

    def policy(hist):
        n = len(hist)
        for t in targets:
            if len(t) > n and t[:n] == hist:
                return t[n]
        return end_id
    return policy


def _right_pad(seqs):
    T = max(len(s) for s in seqs)
    arr = np.zeros((len(seqs), T), dtype=np.int32)
    for i, s in enumerate(seqs):
        arr[i, :len(s)] = s
    return arr


def final_logits(model, seqs, cols=None):
    """fp32 logits (B, V) predicting the token after each sequence (right-padded batch, one forward); `cols`: only
    those vocabulary columns, (B, len(cols)). The head runs in fp32 (no bf16 rounding ties)."""
    arr = _right_pad(seqs)
    x = model.hidden(mx.array(arr))
    last = mx.array([len(s) - 1 for s in seqs])
    h = x[mx.arange(len(seqs)), last].astype(mx.float32)
    W = _head(model) if cols is None else _head(model)[mx.array(list(cols))]
    out = h @ W.astype(mx.float32).T
    mx.eval(out)
    return np.array(out)


def span_logprobs(model, seqs, spans):
    """Per row (log-probs, greedy predictions, targets) of the tokens seqs[i][s:e] for span (s, e), s >= 1.

    One right-padded forward (causal, so padding never affects earlier positions); the head runs only on the
    positions that predict a span token, in fp32."""
    arr = _right_pad(seqs)
    x = model.hidden(mx.array(arr[:, :-1]))
    rows, pos = [], []
    for i, (s, e) in enumerate(spans):
        rows.extend([i] * (e - s))
        pos.extend(range(s - 1, e - 1))
    rows_a, pos_a = np.array(rows, dtype=np.int32), np.array(pos, dtype=np.int32)
    h = x[mx.array(rows_a), mx.array(pos_a)].astype(mx.float32)
    logits = h @ _head(model).astype(mx.float32).T
    tgt = mx.array(arr[rows_a, pos_a + 1])
    lp = mx.take_along_axis(logits, tgt[:, None], axis=-1)[:, 0] - mx.logsumexp(logits, axis=-1)
    pred = mx.argmax(logits, axis=-1)
    mx.eval(lp, pred)
    lp, pred, tgt = np.array(lp), np.array(pred), np.array(tgt)
    out, k = [], 0
    for s, e in spans:
        out.append((lp[k:k + e - s], pred[k:k + e - s], tgt[k:k + e - s]))
        k += e - s
    return out


def _prefill_mask(valid, start, end):
    """(B, 1, end-start, end): causal, padding keys hidden, every query sees itself."""
    q = mx.arange(start, end)[:, None]
    k = mx.arange(end)[None, :]
    m = (k <= q)[None] & mx.array(valid[:, None, :end]) | (k == q)[None]
    return m[:, None]


# ---------------------------------------------------------------- sampling


def _sample(logits, temp, top_p, top_k, kmax, key):
    """Per-row temperature / top-k / top-p sampling; rows with temperature 0 take the argmax."""
    greedy = mx.argmax(logits, axis=-1)
    V = logits.shape[-1]
    if kmax and kmax < V:
        idx = mx.argpartition(-logits, kmax - 1, axis=-1)[:, :kmax]
        vals = mx.take_along_axis(logits, idx, axis=-1)
    else:
        idx, vals = mx.broadcast_to(mx.arange(V)[None], logits.shape), logits
    order = mx.argsort(-vals, axis=-1)
    vals = mx.take_along_axis(vals, order, axis=-1) / mx.where(temp > 0, temp, 1.0)[:, None]
    idx = mx.take_along_axis(idx, order, axis=-1)
    rank = mx.arange(vals.shape[-1])[None]
    vals = mx.where(rank < mx.where(top_k > 0, top_k, V)[:, None], vals, -mx.inf)
    probs = mx.softmax(vals, axis=-1)
    vals = mx.where(mx.cumsum(probs, axis=-1) - probs < top_p[:, None], vals, -mx.inf)
    j = mx.random.categorical(vals, key=key)
    return mx.where(temp > 0, mx.take_along_axis(idx, j[:, None], axis=-1)[:, 0], greedy)


# ---------------------------------------------------------------- core decode loop


@dataclass
class _Row:
    out: list = field(default_factory=list)
    kind: list = field(default_factory=list)
    oov: list = field(default_factory=list)
    length: int = 0
    n_sampled: int = 0
    in_think: bool = False
    think_n: int = 0          # sampled tokens in the current open block
    think_total: int = 0
    answer_n: int = 0
    hit_budget: bool = False
    queue: list = field(default_factory=list)   # (id, oov) still to force
    reason: str = None


def _think_state(ids, think_ids):
    if think_ids is None:
        return False
    o, c = think_ids
    for t in reversed(ids):
        if t == o:
            return True
        if t == c:
            return False
    return False


def _decode(model, seqs, samplings, stop, max_new, *, think_ids=None, budget=None, force=None, max_answer=None,
            max_len=None, pad_id=0, seed=None):
    """Decode a batch of full sequences until every row stops. Returns the _Row of each sequence.

    `force`: (ids, oov flags) forced when a row hits the thinking budget, or a list with one per sequence (None:
    the row stops instead)."""
    B = len(seqs)
    forces = force if isinstance(force, list) else [force] * B
    max_len = max_len or model.cfg.seq_len
    rows = [_Row(length=len(s), in_think=_think_state(s, think_ids)) for s in seqs]
    for r in rows:
        if r.length >= max_len:
            r.reason = "length"
    Lp = max(len(s) for s in seqs)
    arr = np.full((B, Lp), pad_id, dtype=np.int32)
    valid = np.zeros((B, Lp), dtype=bool)
    for i, s in enumerate(seqs):
        arr[i, Lp - len(s):] = s
        valid[i, Lp - len(s):] = True
    caches = make_caches(model)
    padded = not valid.all()
    logits = None
    for a in range(0, Lp, PREFILL_CHUNK):
        e = min(a + PREFILL_CHUNK, Lp)
        logits = last_logits(model, mx.array(arr[:, a:e]), caches, _prefill_mask(valid, a, e) if padded else None)
        mx.eval(logits)

    sp = [as_sampling(s) for s in samplings]
    temp = mx.array([s.temperature for s in sp], dtype=mx.float32)
    top_p = mx.array([s.top_p for s in sp], dtype=mx.float32)
    top_k = mx.array([s.top_k for s in sp], dtype=mx.int32)
    pp = mx.array([s.presence_penalty for s in sp], dtype=mx.float32)
    greedy_only = all(s.temperature <= 0 for s in sp)
    use_pp = any(s.presence_penalty for s in sp)
    hot = [s for s in sp if s.temperature > 0]
    kmax = max(s.top_k for s in hot) if hot and all(s.top_k > 0 for s in hot) else 0
    key = mx.random.key(seed) if seed is not None else None
    seen_arr = mx.zeros((B, logits.shape[-1]), dtype=mx.float32) if use_pp else None   # sampled so far
    stop = set(int(t) for t in (stop or ()))
    o_id, c_id = think_ids if think_ids is not None else (None, None)
    pos = list(range(B))   # batch slot -> row

    while True:
        live = [j for j, i in enumerate(pos) if rows[i].reason is None]
        if not live:
            break
        if len(live) <= len(pos) // 2 and len(pos) >= 4:
            sel = mx.array(live)
            for c in caches:
                c.keys, c.values = c.keys[sel], c.values[sel]
            logits, temp, top_p, top_k, pp = logits[sel], temp[sel], top_p[sel], top_k[sel], pp[sel]
            if seen_arr is not None:
                seen_arr = seen_arr[sel]
            valid = valid[live]
            pos = [pos[j] for j in live]
            live = list(range(len(pos)))
        lg = logits - pp[:, None] * seen_arr if use_pp else logits
        if greedy_only:
            nxt = mx.argmax(lg, axis=-1)
        else:
            sub = None
            if key is not None:
                key, sub = mx.random.split(key)
            nxt = _sample(lg, temp, top_p, top_k, kmax, sub)
        toks = np.array(nxt)
        feed = np.full(len(pos), pad_id, dtype=np.int32)
        fed = np.zeros(len(pos), dtype=bool)
        new_seen = []
        for j in live:
            r = rows[pos[j]]
            if r.queue:
                (t, f), kind = r.queue.pop(0), FORCED
            else:
                t, f, kind = int(toks[j]), False, SAMPLED
            r.out.append(t)
            r.kind.append(kind)
            r.oov.append(f)
            r.length += 1
            if t == o_id:
                r.in_think, r.think_n = True, 0
            elif t == c_id:
                r.in_think = False
            if kind == SAMPLED:
                r.n_sampled += 1
                new_seen.append((j, t))
                if t in stop:
                    r.reason = "stop"
                elif t != o_id and t != c_id:
                    if r.in_think:
                        r.think_n += 1
                        r.think_total += 1
                        if budget and r.think_n >= budget:
                            if forces[pos[j]]:
                                r.queue, r.hit_budget = list(zip(*forces[pos[j]])), True
                            else:
                                r.reason, r.hit_budget = "budget", True
                    else:
                        r.answer_n += 1
                        if max_answer and r.answer_n >= max_answer:
                            r.reason = "length"
            if r.reason is None and (r.n_sampled >= max_new or r.length >= max_len):
                r.reason = "length"
            if r.reason is None:
                feed[j], fed[j] = t, True
        if not fed.any():
            break
        valid = np.concatenate([valid, fed[:, None]], axis=1)
        mask = None
        if not valid.all():
            m = valid.copy()
            m[:, -1] = True
            mask = mx.array(m)[:, None, None, :]
        logits = last_logits(model, mx.array(feed)[:, None], caches, mask)
        if use_pp and new_seen:
            seen_arr[mx.array([p[0] for p in new_seen]), mx.array([p[1] for p in new_seen])] = 1.0
        mx.eval(logits)
    return rows


# ---------------------------------------------------------------- public generation API


def generate_batch(model, prompts, max_new_tokens=256, sampling=None, stop_ids=None, think_budget=None, *, tok=None,
                   think_ids=None, seed=None, max_len=None, max_batch=None, return_info=False):
    """Generated ids per prompt (student ids; a row that hit a stop id ends with it).

    `sampling`: Sampling | dict | preset name | None (greedy), or a list with one per prompt. `stop_ids` defaults
    to <|im_end|> and <|endoftext|> when `tok` is given. With `think_budget`, a row also stops after sampling that
    many tokens inside an open <think> block (think ids from `tok` or `think_ids=(open, close)`). `max_len`
    (default the model's seq_len) bounds prompt + generation per row. return_info adds per-row
    {"reason": stop|budget|length, "think_tokens", "hit_budget"}."""
    if stop_ids is None:
        stop_ids = default_stops(tok) if tok is not None else ()
    if think_budget and think_ids is None:
        if tok is None:
            raise ValueError("think_budget needs tok or think_ids")
        think_ids = (tok.special("<think>"), tok.special("</think>"))
    samplings = sampling if isinstance(sampling, list) else [sampling] * len(prompts)
    pad = tok.bos_id if tok is not None else 0
    step = max_batch or len(prompts) or 1
    outs, infos = [], []
    for k in range(0, len(prompts), step):
        rows = _decode(model, [list(map(int, p)) for p in prompts[k:k + step]], samplings[k:k + step], stop_ids,
                       max_new_tokens, think_ids=think_ids, budget=think_budget, max_len=max_len, pad_id=pad,
                       seed=None if seed is None else seed + k)
        outs.extend(r.out for r in rows)
        infos.extend({"reason": r.reason, "think_tokens": r.think_total, "hit_budget": r.hit_budget} for r in rows)
    return (outs, infos) if return_info else outs


def generate(model, tok, prompt_ids, max_new_tokens=256, temperature=0.0, top_p=1.0, top_k=0, presence_penalty=0.0,
             stop_ids=None, seed=None):
    """One sequence (KV cache). Returns the generated ids; the stop id that ended it, if any, is the last one.
    stop_ids=None: <|im_end|> and <|endoftext|>; () = run to max_new_tokens."""
    s = Sampling(temperature, top_p, top_k, presence_penalty)
    return generate_batch(model, [prompt_ids], max_new_tokens, s, stop_ids, tok=tok, seed=seed)[0]


# ---------------------------------------------------------------- chat


@dataclass
class ChatResult:
    ids: list                  # BOS + prompt + generated / appended, student ids
    prompt_len: int
    sampled: np.ndarray        # bool per position of ids: chosen by the model
    forced: np.ndarray         # budget stop text, tool continuations
    tool_output: np.ndarray    # inside <tool_response> ... </tool_response> (subset of forced)
    oov: np.ndarray            # from an out-of-subset split (prompt / forced text only)
    text: str                  # content of the last assistant turn (after its think block, without <|im_end|>)
    messages: list             # input messages + generated assistant turns and tool turns
    n_tool_calls: int = 0
    think_tokens: int = 0      # tokens sampled inside think blocks (all turns)
    hit_budget: bool = False
    finished: bool = False     # the last turn ended with <|im_end|>
    tool_calls: list = field(default_factory=list)   # [{"name", "arguments", "output"}]
    stop_reason: str = ""      # stop | tool_limit | eos | length

    @property
    def completion_ids(self):
        return self.ids[self.prompt_len:]

    @property
    def reasoning(self):
        """Reasoning of the last assistant turn (None if it had no think block)."""
        last = next((m for m in reversed(self.messages) if m["role"] == "assistant"), None)
        return None if last is None else last.get("reasoning_content")


def _student(tok, tids, flags):
    """Teacher ids + per-token flags -> student ids, flags expanded over out-of-subset splits, oov flags."""
    sids, oov = tok.from_teacher(tids, return_oov=True)
    t2s = tok.vmap.t2s
    lens = [1 if t2s[t] >= 0 else len(tok.vmap.split(t)) for t in tids]
    return [int(i) for i in sids], np.repeat(np.asarray(flags, dtype=bool), lens).tolist(), oov.tolist()


def _from_pieces(tok, P):
    segs = C._segments(P.items)
    enc = [tok.qwen.encode_offsets(s) for kind, s, _ in segs if kind == "text"]
    return _student(tok, *C._assemble(tok.qwen, segs, enc))


class _Templates:
    """Fixed forced sequences for one tokenizer: the budget stop text (think_stop[tools offered]: (ids, oov)) and
    the empty think block."""

    def __init__(self, tok):
        self.think_stop = {}
        for tools in (False, True):
            P = C._Pieces()
            P.text(C.think_stop(tools))
            P.tmpl("\n</think>\n\n")
            ids, _, oov = _from_pieces(tok, P)
            self.think_stop[tools] = (ids, oov)
        P = C._Pieces()
        P.tmpl("<think>\n\n</think>\n\n")
        self.prefill, _, _ = _from_pieces(tok, P)
        self.im_end = tok.special("<|im_end|>")
        self.think = (tok.special("<think>"), tok.special("</think>"))


def tool_continuation(tok, outputs, enable_thinking=True):
    """(ids, tool_output flags, oov) appended after an assistant turn's <|im_end|> that made tool calls."""
    P = C._Pieces()
    P.tmpl("\n<|im_start|>user")
    for out in outputs:
        P.tmpl("\n<tool_response>")
        P.text("\n" + out + "\n", True)
        P.ctrl("</tool_response>")
    P.tmpl("<|im_end|>\n<|im_start|>assistant\n")
    if enable_thinking is False:
        P.tmpl("<think>\n\n</think>\n\n")
    return _from_pieces(tok, P)


def chat_prompt(tok, messages, tools=None, enable_thinking=True):
    """(student ids, oov flags) of BOS + the rendered conversation + the generation prompt."""
    tids, _ = C.render(tok.qwen, messages, tools=tools, add_generation_prompt=True, enable_thinking=enable_thinking)
    ids, _, oov = _student(tok, tids, [False] * len(tids))
    return [tok.bos_id] + ids, [False] + oov


def parse_turn(tok, ids):
    """(reasoning or None, content) of one assistant turn's ids (after "assistant\\n", without <|im_end|>)."""
    o, c = tok.special("<think>"), tok.special("</think>")
    ids = list(ids)
    if o in ids:
        a = ids.index(o)
        if c in ids[a:]:
            b = a + ids[a:].index(c)
            return tok.decode(ids[a + 1:b]).strip("\n"), tok.decode(ids[b + 1:]).lstrip("\n")
        return tok.decode(ids[a + 1:]).strip("\n"), ""
    return None, tok.decode(ids)


class _Conv:
    def __init__(self, tok, T, messages, prompt, oov, enable_thinking, tools, sampling):
        self.tok, self.T = tok, T
        self.messages = [dict(m) for m in messages]
        self.ids = list(prompt)
        self.prompt_len = len(prompt)
        self.sampled = [False] * len(prompt)
        self.forced = [False] * len(prompt)
        self.tool = [False] * len(prompt)
        self.oov = list(oov)
        self.enable_thinking, self.tools, self.sampling = enable_thinking, tools, sampling
        self.turn_start = len(prompt) - (len(T.prefill) if enable_thinking is False else 0)
        self.state, self.text, self.reason = "gen", "", ""
        self.n_calls = self.think_tokens = 0
        self.hit_budget = self.finished = False
        self.calls, self.pending = [], []

    def absorb(self, row, max_tool_calls):
        for t, k in zip(row.out, row.kind):
            self.ids.append(t)
            self.sampled.append(k == SAMPLED)
            self.forced.append(k == FORCED)
            self.tool.append(False)
        self.oov += row.oov
        self.think_tokens += row.think_total
        self.hit_budget |= row.hit_budget
        ended = row.reason == "stop" and self.ids[-1] == self.T.im_end
        end = len(self.ids) - (1 if row.reason == "stop" else 0)   # without <|im_end|> / <|endoftext|>
        reasoning, content = parse_turn(self.tok, self.ids[self.turn_start:end])
        msg = {"role": "assistant", "content": content}
        if reasoning is not None:
            msg["reasoning_content"] = reasoning
        if self.enable_thinking is False:
            msg["think_prefilled"] = True
        self.text = content
        calls = C.parse_tool_calls(content) if (ended and self.tools) else []
        if calls:
            pre = content.split("<tool_call>")[0]
            msg["content"] = pre[:-1] if pre.endswith("\n") else pre
            msg["tool_calls"] = calls
        self.messages.append(msg)
        if calls and self.n_calls < max_tool_calls:
            self.pending, self.state = calls[:max_tool_calls - self.n_calls], "tool"
            return
        self.state = "done"
        self.finished = ended
        self.reason = ("tool_limit" if calls else "stop") if ended else ("eos" if row.reason == "stop" else "length")

    def add_outputs(self, outputs, max_len):
        for call, out in zip(self.pending, outputs):
            self.messages.append({"role": "tool", "content": out})
            self.calls.append({"name": call.get("name"), "arguments": call.get("arguments"), "output": out})
        self.n_calls += len(outputs)
        self.pending = []
        ids, tool, oov = tool_continuation(self.tok, outputs, self.enable_thinking)
        if len(self.ids) + len(ids) >= max_len:
            self.state, self.finished, self.reason = "done", False, "length"
            return
        self.ids += ids
        self.sampled += [False] * len(ids)
        self.forced += [True] * len(ids)
        self.tool += tool
        self.oov += oov
        self.turn_start = len(self.ids) - (len(self.T.prefill) if self.enable_thinking is False else 0)
        self.state = "gen"

    def result(self):
        return ChatResult(ids=self.ids, prompt_len=self.prompt_len, sampled=np.array(self.sampled, dtype=bool),
                          forced=np.array(self.forced, dtype=bool), tool_output=np.array(self.tool, dtype=bool),
                          oov=np.array(self.oov, dtype=bool), text=self.text, messages=self.messages,
                          n_tool_calls=self.n_calls, think_tokens=self.think_tokens, hit_budget=self.hit_budget,
                          finished=self.finished, tool_calls=self.calls, stop_reason=self.reason)


def _run_calls(executor, jobs):
    if len(jobs) <= 1:
        return [executor(c) for c in jobs]
    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as ex:
        return list(ex.map(executor, jobs))


def chat_batch(model, tok, items, enable_thinking=True, tools=None, max_tool_calls=2, think_budget=C.THINK_BUDGET,
               max_answer_tokens=300, sampling=None, *, seed=None, max_batch=64, max_len=None,
               executor=run_tool_call):
    """One ChatResult per item. An item is a message list or a dict {"messages", and optionally "enable_thinking",
    "tools" (a list, or True = chat.TOOLS), "sampling"} overriding the call's defaults (other keys are ignored).
    `sampling` as in generate_batch, or "qwen": Qwen3's preset of each row's effective mode (soft switches
    included). Rounds: decode every running row until it stops (budget stops are forced inline); run the tool calls
    of rows whose turn ended with one (in parallel); re-prefill those rows and continue. `max_answer_tokens` bounds
    the non-thinking tokens of each assistant turn; `max_len` (default the model's seq_len) the whole sequence."""
    T = _Templates(tok)
    max_len = max_len or model.cfg.seq_len
    specs = []
    for it in items:
        d = it if isinstance(it, dict) else {"messages": it}
        et = d.get("enable_thinking", enable_thinking)
        tl = d["tools"] if "tools" in d else tools
        tl = C.TOOLS if tl is True else (tl or None)
        s = d.get("sampling", sampling)
        if s == "qwen":
            s = PRESETS[C.effective_mode(d["messages"], et)]
        specs.append((d["messages"], et, tl, as_sampling(s)))
    rendered = C.render_batch(tok.qwen, [{"messages": m, "tools": tl, "add_generation_prompt": True,
                                          "enable_thinking": et} for m, et, tl, _ in specs])
    convs = []
    for (m, et, tl, s), (tids, _) in zip(specs, rendered):
        ids, _, oov = _student(tok, tids, [False] * len(tids))
        convs.append(_Conv(tok, T, m, [tok.bos_id] + ids, [False] + oov, et, tl, s))
    stops = default_stops(tok)
    max_new = (think_budget or max_answer_tokens) + max_answer_tokens + 4
    rnd = 0
    while True:
        todo = [c for c in convs if c.state == "gen"]
        if not todo:
            break
        for k in range(0, len(todo), max_batch):
            part = todo[k:k + max_batch]
            rows = _decode(model, [c.ids for c in part], [c.sampling for c in part], stops, max_new,
                           think_ids=T.think, budget=think_budget, force=[T.think_stop[bool(c.tools)] for c in part],
                           max_answer=max_answer_tokens, max_len=max_len, pad_id=tok.bos_id,
                           seed=None if seed is None else seed + 7919 * rnd + k)
            for c, r in zip(part, rows):
                c.absorb(r, max_tool_calls)
        waiting = [c for c in todo if c.state == "tool"]
        outputs = _run_calls(executor, [call for c in waiting for call in c.pending])
        k = 0
        for c in waiting:
            n = len(c.pending)
            c.add_outputs(outputs[k:k + n], max_len)
            k += n
        rnd += 1
    return [c.result() for c in convs]


def chat(model, tok, messages, enable_thinking=True, tools=None, max_tool_calls=2, think_budget=C.THINK_BUDGET,
         max_answer_tokens=300, sampling=None, **kw):
    return chat_batch(model, tok, [messages], enable_thinking, tools, max_tool_calls, think_budget,
                      max_answer_tokens, sampling, **kw)[0]
