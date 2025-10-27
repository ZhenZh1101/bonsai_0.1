"""Qwen3 teachers: batched generation for off-policy distillation and log-probs over the student's vocabulary
subset S for on-policy distillation (plan sections 11, 12, 16).

`Teacher(name)` loads teachers/<name> (Qwen3-0.6B / Qwen3-1.7B, bf16 as released) with mlx-lm 0.32 and refuses
to start unless the teacher's tokenizer files hash exactly like tokenizer/manifest.json (plan section 6).

Scoring (on-policy, plan 12): `subset_logprobs(teacher_ids (B, T), vmap)` -> fp32 (B, T, N) log-softmax over S,
row t = the teacher's distribution for the token at t + 1. It equals `logits[..., s2t]` followed by a logsumexp
over those N columns (`subset_log_softmax`, the plan's code block), computed as hidden @ W[s2t].T with the
gathered head rows in fp32, so the 151,936-wide logits are never materialised; probability mass outside S is
dropped and the rest renormalised. Batches are right-padded (causal attention: padding never changes earlier
rows), processed `chunk` rows at a time; `iter_subset_logprobs` yields the chunks for callers that reduce them
on the fly (B x T x N x 4 bytes is 27 GB at B 256, T 400, N 65,536).

Generation (off-policy, plan 11): an own batched decode loop on mlx-lm's Qwen3 module and its BatchKVCache
(left padding, per-row RoPE offsets), not mlx-lm's BatchGenerator, because
  * Qwen3's recommended settings are vLLM semantics: presence penalty on the tokens generated so far in the
    turn, then temperature -> top-k -> top-p; mlx-lm's samplers apply top-p before the temperature and its
    presence penalty looks at the last 20 tokens including the prompt (`sample_next` does it the vLLM way,
    vectorised with per-row temperature / top-p / penalty);
  * thinking-budget stops and tool turns inject tokens into a running row: queued forced tokens are fed one per
    decode step in place of the row's sample, so a row never leaves the batch to be re-prefilled.
Continuous batching: finished rows leave the batch (BatchKVCache.filter, `refill` of them at a time), and the
next prompts are prefilled in groups of `prefill_batch` and merged in (BatchKVCache.extend). Prompts that share
their system turn (the teacher's system prompt, the tools block) get its KV cache copied in and prefill only the
rest (prefix caching: ~2/3 of the prompt tokens).

`chat_batch(jobs)` drives one conversation per row (Qwen3 template, plan sections 6 and 13):
  * thinking mode: the generation prompt is prefilled with "<think>\\n"; after `budget` (128) thinking tokens
    without "</think>", chat.think_stop(job.tools) + "\\n</think>\\n\\n" is forced and the teacher answers from
    what it has (Qwen3's thinking-budget early exit). Jobs that offer the python tool get THINK_STOP_TOOL ("...,
    using the python tool."; Qwen's plain sentence makes a budget-cut teacher answer without calling the tool),
    in every turn, as the student's engine does. Non-thinking: the empty block "<think>\\n\\n</think>\\n\\n" is
    prefilled (enable_thinking=False; also used for /no_think prompts, whose response is that same block).
  * tools: a turn that ends with python tool calls gets them executed by `bonsai.sandbox.run_tool_call`; the
    outputs are appended as the template's tool turn (`<tool_response>` inside a user turn) plus a new assistant
    header, and the teacher continues; at most `max_calls` calls per conversation. The new turn gets the empty
    think block in non-thinking mode; in thinking mode nothing is prefilled: the teacher may open "<think>" (the
    budget applies again) or answer at once, stored as an empty think block (reasoning_content ""). Qwen3-1.7B
    often opens "<think>" there and ends the turn inside it ("...Answer: 26<|im_end|>"): the text from the last
    "Answer:" on becomes the reply (`_salvage_open_think`).
  * a turn ends at <|im_end|> (or <|endoftext|>); an answer over `max_answer` (400) tokens stops the row.

`build_job` / `offpolicy_sample` / `sample_problems` / `generate_offpolicy` turn prompt records into stored
samples and apply the plan's filters (plan 11, 16, 18): verifiable answers correct, English only, no repetition,
answer <= 400 tokens, well-formed think blocks, think-answer consistency (math), no Qwen / Alibaba / Tongyi
self-claims, 13-gram decontamination of the outputs.
"""

import hashlib
import json
import re
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from bonsai import chat
from bonsai.common import TEACHER_DIR, TOKENIZER_DIR, iter_jsonl, log, write_json
from bonsai.tokenizer import CONTROL_IDS, TOKENIZER_FILES, QwenTokenizer, sha256, tokenizer_manifest

TEACHERS = ("Qwen3-0.6B", "Qwen3-1.7B")
IM_END, END_OF_TEXT = CONTROL_IDS["<|im_end|>"], CONTROL_IDS["<|endoftext|>"]
THINK_OPEN, THINK_CLOSE = CONTROL_IDS["<think>"], CONTROL_IDS["</think>"]
EOS = (IM_END, END_OF_TEXT)
MAX_ANSWER = 400          # answer tokens per turn (plan 11 filter: longer answers are dropped)
MAX_TOOL_CALLS = 2        # plan 13
THINK_PREFILL = "<think>\n"
NO_THINK_PREFILL = "<think>\n\n</think>\n\n"
_CTRL_RE = re.compile("(" + "|".join(re.escape(t) for t in CONTROL_IDS) + ")")


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.5


THINK_SAMPLING = Sampling(0.6, 0.95, 20, 1.5)       # Qwen3 model card, thinking mode (plan 11)
NO_THINK_SAMPLING = Sampling(0.7, 0.8, 20, 1.5)     # non-thinking mode
GREEDY = Sampling(0.0, 1.0, 1, 0.0)                 # top_k 1: skips the full-vocabulary sort


def sampling_for(mode, presence_penalty=None):
    s = THINK_SAMPLING if mode == "think" else NO_THINK_SAMPLING
    if presence_penalty is not None:
        s = Sampling(s.temperature, s.top_p, s.top_k, presence_penalty)
    return s


def resolve_name(name):
    """"Qwen/Qwen3-1.7B" or "Qwen3-1.7B" -> "Qwen3-1.7B"."""
    return str(name).split("/")[-1]


def check_tokenizer(directory):
    """The teacher's tokenizer files must hash like the student copy (tokenizer/manifest.json, plan section 6)."""
    want = tokenizer_manifest()["sha256"]
    d = Path(directory)
    bad = {f: sha256(d / f) for f in TOKENIZER_FILES if sha256(d / f) != want[f]}
    if bad:
        raise ValueError(f"{d}: tokenizer files differ from tokenizer/manifest.json: {sorted(bad)}")


def template_ids(qwen, *parts):
    """Teacher ids of template pieces: a str is template text (its control-token literals become control
    tokens), a 1-tuple is content (always ordinary text). Adjacent text is encoded as one run, as chat.render does,
    so the ids equal the rendered conversation."""
    ids, buf = [], []

    def flush():
        if buf:
            ids.extend(qwen.encode("".join(buf)))
            buf.clear()

    for p in parts:
        if isinstance(p, tuple):
            buf.append(p[0])
            continue
        for chunk in _CTRL_RE.split(p):
            if not chunk:
                continue
            if _CTRL_RE.fullmatch(chunk):
                flush()
                ids.append(CONTROL_IDS[chunk])
            else:
                buf.append(chunk)
    flush()
    return ids


def subset_log_softmax(logits, s2t):
    """Reference: log-softmax of full-vocab logits restricted to S (plan 12 code block), fp32."""
    sub = logits[..., mx.array(np.asarray(s2t, dtype=np.int32))].astype(mx.float32)
    return sub - mx.logsumexp(sub, axis=-1, keepdims=True)


def topk_indices(z, k, block=128):
    """Indices of the k largest entries per row (any order). The k blocks with the largest maxima contain the
    k largest entries, so one cheap block-max pass + two small partitions replace an argpartition over the
    whole 151,936-wide row (0.6 ms instead of 8 ms at B 64)."""
    B, V = z.shape
    nb = -(-V // block)
    if nb <= k:
        return mx.argpartition(-z, kth=k - 1, axis=-1)[:, :k]
    if nb * block != V:
        z = mx.concatenate([z, mx.full((B, nb * block - V), -mx.inf, dtype=z.dtype)], axis=1)
    zb = z.reshape(B, nb, block)
    blocks = mx.argpartition(-zb.max(axis=-1), kth=k - 1, axis=-1)[:, :k]
    cand = mx.take_along_axis(zb, blocks[:, :, None], axis=1).reshape(B, k * block)
    cand_idx = (blocks[:, :, None] * block + mx.arange(block)[None, None]).reshape(B, k * block)
    sel = mx.argpartition(-cand, kth=k - 1, axis=-1)[:, :k]
    return mx.take_along_axis(cand_idx, sel, axis=-1)


def sample_next(logits, temperature, top_p, top_k=20, presence=None, seen=None):
    """One token per row. logits (B, V); temperature / top_p / presence (B,) arrays; seen (B, V) 1.0 where the
    token was already generated in this turn. vLLM order: presence penalty, temperature, top-k, top-p; rows with
    temperature 0 take the argmax (after the penalty)."""
    logits = logits.astype(mx.float32)
    if presence is not None and seen is not None:
        logits = logits - presence[:, None] * seen
    greedy = mx.argmax(logits, axis=-1)
    z = logits / mx.maximum(temperature, 1e-5)[:, None]
    if 0 < top_k < z.shape[-1]:
        idx = topk_indices(z, top_k)
        vals = mx.take_along_axis(z, idx, axis=-1)
    else:
        idx = mx.broadcast_to(mx.arange(z.shape[-1])[None], z.shape)
        vals = z
    order = mx.argsort(-vals, axis=-1)
    vals, idx = mx.take_along_axis(vals, order, axis=-1), mx.take_along_axis(idx, order, axis=-1)
    probs = mx.softmax(vals, axis=-1)
    keep = (mx.cumsum(probs, axis=-1) - probs) < top_p[:, None]      # the top token always survives
    choice = mx.random.categorical(mx.where(keep, vals, -mx.inf), axis=-1)
    tok = mx.take_along_axis(idx, choice[:, None], axis=-1)[:, 0]
    return mx.where(temperature > 0, tok, greedy).astype(mx.int32)


# ------------------------------------------------------------------------------------------- decode rows


class _Row:
    """One sequence of the decode loop: ids so far, queued forced tokens, done flag. `push` gets every token
    appended to the row (sampled or forced) and decides what comes next."""

    def __init__(self, ids, sampling):
        self.ids = list(ids)
        self.prompt_len = len(self.ids)
        self.sampling = sampling
        self.queue = deque()
        self.kinds = []          # per generated position: 1 sampled, 2 forced
        self.done = False
        self.finish = None
        self.reset_seen = False  # set when a new assistant turn starts (presence penalty is per turn)
        self.index = None        # the caller's position of this row
        self.prefix_len = 0      # leading ids shared with other rows (system turn): prefilled once, see _prefill

    def push(self, tok, forced):
        raise NotImplementedError

    def _finish(self, reason):
        self.done, self.finish = True, reason
        self.queue.clear()


class _PlainRow(_Row):
    def __init__(self, ids, sampling, max_tokens, stop):
        super().__init__(ids, sampling)
        self.max_tokens, self.stop = max_tokens, stop

    def push(self, tok, forced):
        self.ids.append(tok)
        self.kinds.append(2 if forced else 1)
        if tok in self.stop:
            self._finish("stop")
        elif len(self.kinds) >= self.max_tokens:
            self._finish("length")


@dataclass
class ChatJob:
    """One conversation to continue: `messages` ends with a user turn (a system turn, if any, comes first)."""
    messages: list
    mode: str = "think"              # "think" | "no_think"
    tools: bool = False
    sampling: Sampling = None
    key: object = None


@dataclass
class ChatOutput:
    key: object
    ids: list                        # teacher ids: prompt (+ think prefill) + everything appended
    prompt_len: int
    kinds: list                      # per appended position: 1 sampled, 2 forced (budget stop, tool turn, prefill)
    turns: list                      # per assistant turn: think, think_tokens, hit_budget, text, answer_tokens,
    finish: str                      #   calls, outputs
    n_tool_calls: int = 0

    @property
    def gen_tokens(self):
        return len(self.kinds)


class _ChatRow(_Row):
    def __init__(self, qwen, job, budget, max_answer, max_calls, max_total, run_tool):
        prompt, _ = chat.render(qwen, job.messages, tools=chat.TOOLS if job.tools else None,
                                add_generation_prompt=True)
        self.think_mode = job.mode == "think"
        self.prefill = template_ids(qwen, THINK_PREFILL if self.think_mode else NO_THINK_PREFILL)
        super().__init__(prompt + self.prefill, job.sampling or sampling_for(job.mode))
        head = [m for m in job.messages[:1] if m["role"] == "system"]
        if head or job.tools:
            sys_ids, _ = chat.render(qwen, head, tools=chat.TOOLS if job.tools else None)
            if prompt[:len(sys_ids)] == sys_ids:
                self.prefix_len = len(sys_ids)
        self.q, self.job, self.run_tool = qwen, job, run_tool
        self.budget, self.max_answer, self.max_calls, self.max_total = budget, max_answer, max_calls, max_total
        self.stop_text = chat.think_stop(job.tools)
        self.stop_ids = template_ids(qwen, (self.stop_text,), "\n</think>\n\n")
        self.turns, self.n_calls = [], 0
        self._new_turn()

    def _new_turn(self, prefilled=True):
        self.phase = ("think" if prefilled else "open") if self.think_mode else "answer"
        self.think, self.answer, self.hit = [], [], False

    def _turn(self):
        t = {"think": None, "think_tokens": len(self.think), "hit_budget": self.hit,
             "text": self.q.decode(self.answer), "answer_tokens": len(self.answer), "calls": [], "outputs": []}
        if self.think_mode:
            t["think"] = self.q.decode(self.think).strip() + (self.stop_text if self.hit else "")
        self.turns.append(t)
        return t

    def _salvage_open_think(self):
        """The turn ended inside its think block. After a tool result Qwen3-1.7B often writes "<think>\n...
        Answer: 26<|im_end|>" without closing the block: the text from the last "Answer:" on becomes the reply,
        the rest the reasoning. A first turn that never closes its think block is malformed."""
        text = self.q.decode(self.think)
        if not self.turns or "Answer:" not in text:
            self._turn()
            return self._finish("eos_in_think")
        k = text.rindex("Answer:")
        self.think, self.answer = self.q.encode(text[:k].strip()), self.q.encode(text[k:].strip())
        self._turn()["salvaged"] = True
        self._finish("stop")

    def push(self, tok, forced):
        self.ids.append(tok)
        self.kinds.append(2 if forced else 1)
        if forced or self.done:
            return
        if len(self.ids) >= self.max_total:
            self._turn()
            return self._finish("too_long")
        if self.phase == "open":                # thinking turn after a tool result: the teacher may skip thinking
            self.phase = "think" if tok == THINK_OPEN else "answer"
            if tok == THINK_OPEN:
                return
        if self.phase == "think":
            if tok == THINK_CLOSE:
                self.phase = "answer"
            elif tok in EOS:
                self._salvage_open_think()
            else:
                self.think.append(tok)
                if len(self.think) >= self.budget:
                    self.hit, self.phase = True, "answer"
                    self.queue.extend(self.stop_ids)
            return
        if tok not in EOS:
            self.answer.append(tok)
            if len(self.answer) > self.max_answer:
                self._turn()
                self._finish("answer_too_long")
            return
        t = self._turn()
        n_open = t["text"].count("<tool_call>")
        if not (self.job.tools and n_open):
            return self._finish("stop")
        calls = chat.parse_tool_calls(t["text"])
        if len(calls) != n_open:
            return self._finish("malformed_tool_call")
        if self.n_calls + len(calls) > self.max_calls:
            return self._finish("too_many_calls")
        t["calls"] = calls
        t["outputs"] = [self.run_tool(c) for c in calls]
        self.n_calls += len(calls)
        parts = ["\n<|im_start|>user"]
        for out in t["outputs"]:
            parts += ["\n<tool_response>\n", (out,), "\n</tool_response>"]
        parts.append("<|im_end|>\n<|im_start|>assistant\n")
        # non-thinking: the empty block again (enable_thinking=False); thinking: no "<think>" prefill, because
        # after a tool result Qwen3 often answers at once (with the prefill it writes the answer inside the block)
        self.queue.extend(template_ids(self.q, *parts) + ([] if self.think_mode else self.prefill))
        self.reset_seen = True
        self._new_turn(prefilled=not self.think_mode)


def _run_tool(call):
    from bonsai.sandbox import run_tool_call
    return run_tool_call(call)


# ------------------------------------------------------------------------------------------------ teacher


class Teacher:
    """A Qwen3 teacher: `generate_batch`, `chat_batch` (generation) and `subset_logprobs` (scoring).

    `model` / `qwen` may be given directly (tests: a tiny random mlx-lm Qwen3); otherwise teachers/<name> is
    loaded and its tokenizer hashes checked."""

    def __init__(self, name="Qwen3-0.6B", directory=None, model=None, qwen=None):
        self.name = resolve_name(name)
        if model is None:
            from mlx_lm.utils import load_model
            d = Path(directory) if directory else TEACHER_DIR / self.name
            if not (d / "config.json").exists():
                raise FileNotFoundError(f"{d} missing: run `python -m scripts.download --only teachers` first")
            check_tokenizer(d)
            model, _ = load_model(d)
        self.model = model
        self.qwen = qwen or QwenTokenizer(TOKENIZER_DIR)
        self.n_layers = len(model.model.layers)
        self.vocab_size = model.args.vocab_size
        self._subset_heads = {}
        self._prefixes = {}
        self.stats = {}

    # --- forward pieces ---------------------------------------------------------------------------
    def hidden(self, ids, cache=None):
        return self.model.model(ids, cache)

    def head(self, h):
        m = self.model
        return m.model.embed_tokens.as_linear(h) if m.args.tie_word_embeddings else m.lm_head(h)

    def head_weight(self):
        m = self.model
        layer = m.model.embed_tokens if m.args.tie_word_embeddings else m.lm_head
        return None if hasattr(layer, "scales") else layer.weight   # quantized heads: gather the logits instead

    def logits(self, ids):
        """Full-vocab fp32 logits (B, T, 151936) of right-padded teacher ids."""
        return self.head(self.hidden(_as_ids(ids))).astype(mx.float32)

    # --- scoring over S (on-policy distillation, plan 12) -----------------------------------------
    def _subset_head(self, vmap):
        key = vmap.sha()
        if key not in self._subset_heads:
            w = self.head_weight()
            sub = None if w is None else w[mx.array(vmap.s2t)].astype(mx.float32)
            if sub is not None:
                mx.eval(sub)
            self._subset_heads = {key: (sub, mx.array(vmap.s2t))}
        return self._subset_heads[key]

    def iter_subset_logprobs(self, teacher_ids, vmap, chunk=8):
        """Yields (start, logp) with logp fp32 (b, T, N) for rows start..start+b of right-padded teacher ids."""
        ids = _as_ids(teacher_ids)
        sub, s2t = self._subset_head(vmap)
        for a in range(0, ids.shape[0], chunk):
            h = self.hidden(ids[a:a + chunk])
            z = h.astype(mx.float32) @ sub.T if sub is not None else self.head(h)[..., s2t].astype(mx.float32)
            lp = z - mx.logsumexp(z, axis=-1, keepdims=True)
            mx.eval(lp)
            yield a, lp

    def subset_logprobs(self, teacher_ids, vmap, chunk=8):
        """fp32 (B, T, N) log-softmax over S; row t predicts the token at t + 1 (see the module docstring)."""
        parts = [lp for _, lp in self.iter_subset_logprobs(teacher_ids, vmap, chunk)]
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)

    # --- batched decoding -------------------------------------------------------------------------
    def _prefix_kv(self, ids):
        """Per-layer (keys, values) of a shared prompt prefix (the system turn), computed once."""
        key = tuple(ids)
        if key not in self._prefixes:
            from mlx_lm.models.cache import KVCache
            c = [KVCache() for _ in range(self.n_layers)]
            self.hidden(mx.array([list(ids)], dtype=mx.int32), c)
            kv = [(layer.keys[:, :, :len(ids)], layer.values[:, :, :len(ids)]) for layer in c]
            mx.eval(kv)
            if len(self._prefixes) >= 16:
                self._prefixes.clear()
            self._prefixes[key] = kv
        return self._prefixes[key]

    def _prefill(self, seqs, prefix_len=0, step=1024):
        """BatchKVCache + last-position logits of a group of prompts. Without a prefix the prompts are
        left-padded. With `prefix_len` (all seqs start with the same prefix_len ids) the cached prefix is copied
        into every row and only the suffixes run, right-padded; finalize() then rolls the padding to the left,
        as mlx-lm's own batched prompt processing does."""
        from mlx_lm.models.cache import BatchKVCache
        g = len(seqs)
        sufs = [list(s[prefix_len:]) for s in seqs]
        L = max(len(x) for x in sufs)
        if prefix_len and L <= step:
            pads = [L - len(x) for x in sufs]
            cache = []
            for k, v in self._prefix_kv(seqs[0][:prefix_len]):
                c = BatchKVCache([0] * g)
                c.keys, c.values = mx.repeat(k, g, axis=0), mx.repeat(v, g, axis=0)
                c.offset, c._idx = mx.array([prefix_len] * g), prefix_len
                c.prepare(right_padding=pads)
                cache.append(c)
            x = mx.array([x + [0] * p for x, p in zip(sufs, pads)], dtype=mx.int32)
            h = self.hidden(x, cache)
            for c in cache:
                c.finalize()
            mx.eval([c.state for c in cache])
            last = h[mx.arange(g), mx.array([len(x) - 1 for x in sufs])]
            return cache, self.head(last).astype(mx.float32)
        L = max(len(s) for s in seqs)
        pads = [L - len(s) for s in seqs]
        x = mx.array([[0] * p + list(s) for p, s in zip(pads, seqs)], dtype=mx.int32)
        cache = [BatchKVCache(pads) for _ in range(self.n_layers)]
        for a in range(0, L, step):
            h = self.hidden(x[:, a:a + step], cache)
            mx.eval([c.state for c in cache])
        return cache, self.head(h[:, -1, :]).astype(mx.float32)

    def _step(self, cache, tokens):
        """One decode step: (B,) next input ids -> (B, V) logits; `cache` is updated in place."""
        return self.head(self.hidden(mx.array(tokens, dtype=mx.int32)[:, None], cache)[:, -1, :])

    def _decode(self, rows, batch=64, prefill_batch=16, refill=None, progress_every=0, on_done=None,
                prefix_cache=True):
        """Runs every row to completion (continuous batching, see the module docstring); `on_done(row)` is called
        as soon as a row finishes. A finished row stays in the batch (fed a dummy token, its samples ignored)
        until `refill` rows are finished, because every BatchKVCache.filter copies the whole cache (twice: the
        next update reallocates); compacting per finished row cost half the decode time at batch 64."""
        top_ks = {r.sampling.top_k for r in rows}
        if len(top_ks) > 1:
            raise ValueError(f"one top_k per call, got {sorted(top_ks)}")
        top_k = top_ks.pop() if top_ks else 0
        V = self.vocab_size
        refill = refill or max(1, batch // 4)
        pending, act = deque(rows), []
        cache = x = seen = params = None
        st = {"rows": len(rows), "prefill_tokens": 0, "gen_tokens": 0, "steps": 0, "prefill_s": 0.0, "decode_s": 0.0}
        t_log = time.time()

        def row_params(rs):
            return tuple(mx.array([getattr(r.sampling, k) for r in rs], dtype=mx.float32)
                         for k in ("temperature", "top_p", "presence_penalty"))

        def advance(rs, toks, prev, seen_rows):
            nxt, marks, n_live = [], [], 0
            for i, (r, t) in enumerate(zip(rs, toks)):
                if r.done:                      # finished earlier: keeps its slot until the next compaction
                    nxt.append(prev[i])
                    continue
                if r.queue:
                    t = r.queue.popleft()
                    r.push(t, True)
                else:
                    r.push(t, False)
                    marks.append((i, t))
                if r.reset_seen:
                    r.reset_seen = False
                    seen_rows[i] = 0.0
                    marks = [m for m in marks if m[0] != i]
                nxt.append(t)
                n_live += 1
                if r.done and on_done is not None:
                    on_done(r)
            if marks:
                seen_rows[mx.array([m[0] for m in marks]), mx.array([m[1] for m in marks])] = 1.0
            return nxt, seen_rows, n_live

        while pending or act:
            t0 = time.time()
            n_done = sum(r.done for r in act)
            if n_done and (n_done == len(act) or n_done >= (refill if pending else max(1, len(act) // 4))):
                keep = [i for i, r in enumerate(act) if not r.done]
                if keep:
                    for layer in cache:
                        layer.filter(keep)
                    act, x, seen = [act[i] for i in keep], [x[i] for i in keep], seen[mx.array(keep)]
                    params = row_params(act)
                else:
                    act, cache, x, seen, params = [], None, None, None, None
            if pending and batch - len(act) >= min(refill, len(pending)):
                new = [pending.popleft() for _ in range(min(batch - len(act), len(pending)))]
                groups = {}
                for r in new:                   # rows sharing a system turn prefill it once
                    groups.setdefault(tuple(r.ids[:r.prefix_len]) if prefix_cache else (), []).append(r)
                chunks = [(len(k), rs[g:g + prefill_batch]) for k, rs in groups.items()
                          for g in range(0, len(rs), prefill_batch)]
                parts = []
                for plen, grp in chunks:
                    st["prefill_tokens"] += sum(len(r.ids) - plen for r in grp)   # computed (not cached)
                    c, logits = self._prefill([r.ids for r in grp], plen)
                    p = row_params(grp)
                    toks = sample_next(logits, p[0], p[1], top_k).tolist()
                    nxt, s_new, n_live = advance(grp, toks, toks, mx.zeros((len(grp), V), dtype=mx.float32))
                    if n_live:
                        parts.append((c, nxt, s_new, grp))
                if parts:                       # merge the new groups first: one extend of the big cache per refill
                    c, nxt, s_new, grp = parts[0]
                    for c2, nxt2, s2, grp2 in parts[1:]:
                        for a_, b_ in zip(c, c2):
                            a_.extend(b_)
                        nxt, s_new, grp = nxt + nxt2, mx.concatenate([s_new, s2]), grp + grp2
                    if cache is None:
                        cache, x, seen, act = c, nxt, s_new, grp
                    else:
                        for a_, b_ in zip(cache, c):
                            a_.extend(b_)
                        x, seen, act = x + nxt, mx.concatenate([seen, s_new]), act + grp
                params = row_params(act) if act else None
            st["prefill_s"] += time.time() - t0
            if not act or all(r.done for r in act):
                continue
            t0 = time.time()
            logits = self._step(cache, x)
            toks = sample_next(logits, params[0], params[1], top_k, params[2], seen).tolist()
            x, seen, n_live = advance(act, toks, x, seen)
            st["gen_tokens"] += n_live
            st["steps"] += 1
            st["decode_s"] += time.time() - t0
            if st["steps"] % 256 == 0:
                mx.clear_cache()
            if progress_every and time.time() - t_log > progress_every:
                t_log = time.time()
                el = st["prefill_s"] + st["decode_s"]
                log(f"  teacher {self.name}: {st['gen_tokens']:,} tokens, {st['gen_tokens'] / max(el, 1e-9):.0f} tok/s, "
                    f"{len(act)} active, {len(pending)} pending")
        st["tok_per_s"] = st["gen_tokens"] / max(st["prefill_s"] + st["decode_s"], 1e-9)
        st["peak_gb"] = mx.get_peak_memory() / 1e9
        self.stats = st
        return rows

    def warmup(self, batch=64):
        """A few decode steps at the production batch size (the first steps of a process compile Metal kernels
        and are 2-3x slower; throughput numbers are taken after this)."""
        self.generate_batch([self.qwen.encode("Hello, how are you today?")] * batch, THINK_SAMPLING, 8, stop=(),
                            batch=batch)

    def generate_batch(self, prompts, sampling=GREEDY, max_tokens=256, stop=EOS, batch=64, prefill_batch=16):
        """Plain continuations of teacher-id prompts. `sampling`: one Sampling or one per prompt. Returns the
        generated ids per prompt; a row that ended on a stop token has it as its last id."""
        samp = sampling if isinstance(sampling, (list, tuple)) else [sampling] * len(prompts)
        rows = [_PlainRow(p, s, max_tokens, set(stop or ())) for p, s in zip(prompts, samp)]
        self._decode(rows, batch, prefill_batch)
        return [r.ids[r.prompt_len:] for r in rows]

    def chat_batch(self, jobs, budget=chat.THINK_BUDGET, max_answer=MAX_ANSWER, max_calls=MAX_TOOL_CALLS,
                   max_total=2048, batch=64, prefill_batch=16, run_tool=None, progress_every=0, on_output=None,
                   prefix_cache=True):
        """Generates the next assistant turn(s) of every ChatJob (thinking budget, tool turns; see the module
        docstring). Returns ChatOutputs in job order; `on_output(index, output)` is called as each one finishes
        (stream them to disk); `self.stats` holds the throughput of the call."""
        rows = [_ChatRow(self.qwen, j, budget, max_answer, max_calls, max_total, run_tool or _run_tool)
                for j in jobs]
        outs = [None] * len(rows)

        def done(r):
            k = r.index
            outs[k] = ChatOutput(jobs[k].key, r.ids, r.prompt_len - len(r.prefill), [2] * len(r.prefill) + r.kinds,
                                 r.turns, r.finish, r.n_calls)
            if on_output is not None:
                on_output(k, outs[k])

        for k, r in enumerate(rows):
            r.index = k
        order = sorted(range(len(rows)), key=lambda i: -len(rows[i].ids))   # similar lengths share a prefill
        self._decode([rows[i] for i in order], batch, prefill_batch, progress_every=progress_every, on_done=done,
                     prefix_cache=prefix_cache)
        return outs


def _as_ids(ids):
    return ids if isinstance(ids, mx.array) else mx.array(np.asarray(ids, dtype=np.int32))


# ------------------------------------------------------------------------------- off-policy samples


def stable_frac(key, salt=""):
    return int(hashlib.md5((salt + "\x00" + str(key)).encode()).hexdigest()[:8], 16) / 2 ** 32


def choose_signal(key, mode, think_soft_frac=0.3, no_think_soft_frac=0.5):
    """How the sample asks for its mode: "soft" (/think or /no_think on the last user turn), "prefill"
    (enable_thinking=False: the empty think block is part of the prompt) or "default" (no switch = thinking)."""
    u = stable_frac(key, "signal")
    if mode == "think":
        return "soft" if u < think_soft_frac else "default"
    return "soft" if u < no_think_soft_frac else "prefill"


def build_job(rec, mode, system_prompts, signal, sampling=None):
    """ChatJob for a prompt record {"id", "kind", "messages", "tools", ...}: the teacher-only English system
    prompt for its kind, the soft switch on the last user turn when signal == "soft"."""
    msgs = [dict(m) for m in rec["messages"]]
    if signal == "soft":
        msgs[-1]["content"] = chat.with_soft_switch(msgs[-1]["content"], mode)
    kind = "math_tool" if rec.get("tools") else rec.get("kind", "chat")
    sp = system_prompts.get(kind) or system_prompts.get("chat")
    return ChatJob([{"role": "system", "content": sp}] + msgs if sp else msgs, mode, bool(rec.get("tools")),
                   sampling, f"{rec['id']}:{mode}")


def generated_messages(out, mode, signal):
    """The assistant / tool turns of a ChatOutput as Qwen-format messages (reasoning_content, tool_calls)."""
    msgs = []
    for t in out.turns:
        content = t["text"].split("<tool_call>")[0].strip() if t["calls"] else t["text"].strip()
        m = {"role": "assistant", "content": content, "reasoning_content": t["think"] if mode == "think" else ""}
        if t["calls"]:
            m["tool_calls"] = t["calls"]
        if signal == "prefill":
            m["think_prefilled"] = True
        msgs.append(m)
        msgs += [{"role": "tool", "content": o} for o in t["outputs"]]
    return msgs


def offpolicy_sample(rec, job, out, mode, signal, teacher_name):
    """The stored record: the conversation without the teacher's system prompt + generation stats."""
    prompt_msgs = [m for m in job.messages if m["role"] != "system"]
    return {"id": job.key, "prompt_id": rec["id"], "source": rec.get("source"), "kind": rec.get("kind"),
            "mode": mode, "signal": signal, "tools": bool(rec.get("tools")), "answer": rec.get("answer"),
            "messages": prompt_msgs + generated_messages(out, mode, signal), "teacher": teacher_name,
            "gen": {"finish": out.finish, "gen_tokens": out.gen_tokens, "prompt_tokens": out.prompt_len,
                    "think_tokens": [t["think_tokens"] for t in out.turns],
                    "hit_budget": [t["hit_budget"] for t in out.turns],
                    "answer_tokens": [t["answer_tokens"] for t in out.turns], "n_tool_calls": out.n_tool_calls}}


# --- filters (plan 11, 16, 18) ---------------------------------------------------------------------

_CJK = re.compile(r"[\u1100-\u11ff\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]")
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
_QWEN_WORD = re.compile(r"qwen|tongyi|qianwen|通义|千问", re.I)
_QWEN_FALLBACK = re.compile(r"\b(?:i am|i'm|i’m|my name is)\s+(?:\w+\s+){0,3}?(?:qwen|tongyi)|"
                            r"\b(?:created|developed|made|built|trained) by (?:the )?(?:alibaba|qwen team|tongyi)", re.I)
_LETTER_ANSWER = re.compile(r"Answer:\s*\(?([A-E])\b")
LETTERS = "ABCDE"


def non_english(text, max_foreign=0.03):
    """Any CJK / Hangul / Kana / full-width character, or more than `max_foreign` of the letters non-ASCII."""
    if _CJK.search(text):
        return True
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(ord(c) > 127 for c in letters) / len(letters) > max_foreign


def repetitive(text, n=4, min_words=40, min_distinct=0.5, max_line_repeats=3):
    """Degenerate repetition: few distinct word 4-grams, or one substantial line repeated."""
    w = text.split()
    if len(w) >= min_words:
        grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
        if len(set(grams)) / len(grams) < min_distinct:
            return True
    lines = Counter(l.strip() for l in text.splitlines() if len(l.strip()) >= 20)
    return bool(lines) and max(lines.values()) > max_line_repeats


def numbers_in(text):
    from bonsai.sandbox import parse_number
    return [v for v in (parse_number(s) for s in _NUM.findall(text)) if v is not None]


def answers_match(pred, gold):
    """Numeric answer check: equal, or equal at 2 decimals when the reference is not an integer."""
    from bonsai.sandbox import numbers_equal, parse_number
    gold = parse_number(gold) if isinstance(gold, str) else gold
    if pred is None or gold is None:
        return False
    if numbers_equal(pred, gold):
        return True
    return abs(gold - round(gold)) > 1e-9 and round(pred, 2) == round(gold, 2)


def qwen_claims(text):
    """Qwen / Alibaba / Tongyi self-claims (bonsai.identity.qwen_self_claims; a local regex if it is missing),
    claims to be another model (identity.other_model_claims) and any mention of the Qwen name."""
    hits = []
    try:
        from bonsai.identity import other_model_claims, qwen_self_claims
        hits += list(qwen_self_claims(text)) + list(other_model_claims(text))
    except ImportError:
        hits += [m.group(0) for m in _QWEN_FALLBACK.finditer(text)]
    hits += [m.group(0) for m in _QWEN_WORD.finditer(text)]
    return hits


def sample_problems(sample, contaminated=None, think_answer="strict", max_answer=MAX_ANSWER):
    """Every filter rule a stored off-policy sample breaks (empty = keep), plan 11 / 16 / 18.

    Uses only the stored record (messages, mode, kind, tools, answer, gen stats), so it can be re-run offline
    (gen_offpolicy --refilter). `contaminated`: text -> bool (default: bonsai.data.decontam.is_contaminated).
    `think_answer` (math, thinking, no tool): "strict" = the final number must occur in the thinking (among its
    last 3 numbers when the thinking ended by itself, anywhere in it when the budget cut it); "contradiction" =
    only a thinking that ended by itself is checked (a budget-cut one may not have reached the answer yet);
    "off"."""
    from bonsai.sandbox import extract_answer, numbers_equal
    probs = []
    gen = sample.get("gen", {})
    finish = gen.get("finish", "stop")
    if finish != "stop":
        probs.append(finish)                      # answer_too_long, eos_in_think, too_long, malformed/too many calls
    msgs = sample["messages"]
    lq = chat.last_query_index(msgs)
    turns = [m for m in msgs[lq + 1:] if m["role"] == "assistant"]
    tools_out = [m["content"] for m in msgs[lq + 1:] if m["role"] == "tool"]
    if not turns:
        return probs + ["no_response"]
    final = turns[-1]
    texts = [t.get("reasoning_content") or "" for t in turns] + [t.get("content") or "" for t in turns]
    allt = "\n".join(texts)
    think_mode = sample["mode"] == "think"
    for k, t in enumerate(turns):
        rc, content = t.get("reasoning_content"), t.get("content") or ""
        if think_mode and k == 0 and not (rc or "").strip():  # later (post-tool) turns may skip thinking
            probs.append("missing_think")
        if not think_mode and (rc or "").strip():
            probs.append("think_in_no_think")
        if any(tag in (rc or "") + content for tag in ("<think>", "</think>", "<|im_start|>", "<|im_end|>",
                                                      "<tool_response>", "<|endoftext|>")):
            probs.append("malformed_think")
        if "<tool_call>" in content or "</tool_call>" in content:
            probs.append("malformed_tool_call")
    if final.get("tool_calls"):
        probs.append("ends_with_tool_call")
    if not (final.get("content") or "").strip():
        probs.append("empty_answer")
    if sample.get("tools") and not tools_out:
        probs.append("no_tool_call")
    if not sample.get("tools") and (tools_out or any(t.get("tool_calls") for t in turns)):
        probs.append("unexpected_tool_call")
    if max(gen.get("answer_tokens") or [0]) > max_answer:
        probs.append("answer_too_long")
    if non_english(allt):
        probs.append("non_english")
    if any(repetitive(s) for s in texts if s):
        probs.append("repetition")
    kind, gold = sample.get("kind"), sample.get("answer")
    if kind == "math" and gold is not None:
        pred = extract_answer(final.get("content") or "")
        if pred is None:
            probs.append("no_answer")
        elif not answers_match(pred, gold):
            probs.append("wrong_answer")
        elif sample.get("tools"):
            if not any(numbers_equal(pred, v) for v in numbers_in(tools_out[-1] if tools_out else "")):
                probs.append("tool_answer_mismatch")
        elif think_mode and think_answer != "off":
            nums = numbers_in(final.get("reasoning_content") or "")
            cut = (gen.get("hit_budget") or [False])[-1]
            if not (cut and think_answer == "contradiction") and \
                    not any(numbers_equal(pred, v) for v in (nums if cut else nums[-3:])):
                probs.append("think_answer_mismatch")
    elif kind == "mc" and gold is not None:
        m = _LETTER_ANSWER.search(final.get("content") or "")
        if m is None:
            probs.append("no_answer")
        elif m.group(1) != str(gold).strip().upper():
            probs.append("wrong_answer")
    if qwen_claims(allt):
        probs.append("qwen_claim")
    if contaminated is None:
        from bonsai.data.decontam import is_contaminated as contaminated
    if contaminated(allt):
        probs.append("contaminated")
    return list(dict.fromkeys(probs))


# --- the off-policy generation run (scripts/gen_offpolicy.py) --------------------------------------


def shard_of(prompt_id, n):
    return int(hashlib.md5(str(prompt_id).encode()).hexdigest()[:8], 16) % n


def iter_samples(path, log_file=None):
    """Records of a kept_ / rejected_ file. A hard kill (OOM, power loss) can leave a partly written last line
    without its newline: it is skipped (logged); its sample is not done, so it is generated again."""
    with open(path, "rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            try:
                yield json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if raw.endswith(b"\n"):
                    raise
                log(f"{path}: skipped a truncated last line ({len(raw):,} bytes, killed mid-write)", log_file)


def repair_tail(path, log_file=None):
    """Cuts a JSONL file back to its last complete line before appending to it (otherwise the next record would
    be glued onto a partial one). Returns the number of bytes dropped."""
    path = Path(path)
    if not path.exists():
        return 0
    with open(path, "rb+") as f:
        size = pos = f.seek(0, 2)
        keep = 0
        while pos > 0:
            step = min(1 << 16, pos)
            f.seek(pos - step)
            k = f.read(step).rfind(b"\n")
            if k >= 0:
                keep = pos - step + k + 1
                break
            pos -= step
        if keep < size:
            f.truncate(keep)
    if keep < size:
        log(f"{path}: dropped a truncated last line ({size - keep:,} bytes, killed mid-write); its sample is "
            f"generated again", log_file)
    return size - keep


def done_ids(out_dir, log_file=None):
    ids = set()
    for p in sorted(Path(out_dir).glob("kept_*.jsonl")) + sorted(Path(out_dir).glob("rejected_*.jsonl")):
        ids.update(r["id"] for r in iter_samples(p, log_file))
    return ids


def summarize(out_dir):
    """stats.json from every kept / rejected file and the per-run throughput lines (runs.jsonl)."""
    out_dir = Path(out_dir)
    st = {"kept": 0, "rejected": 0, "reasons": Counter(), "first_reason": Counter(), "kept_by": Counter(),
          "jobs_by": Counter(), "kept_tokens": 0, "budget_hits": 0, "think_turns": 0, "tool_calls": 0}
    for p in sorted(out_dir.glob("kept_*.jsonl")):
        for r in iter_samples(p):
            st["kept"] += 1
            key = f"{r['kind']}{'+tools' if r['tools'] else ''}/{r['mode']}"
            st["kept_by"][key] += 1
            st["jobs_by"][key] += 1
            g = r.get("gen", {})
            st["kept_tokens"] += g.get("prompt_tokens", 0) + g.get("gen_tokens", 0)
            if r["mode"] == "think":
                st["think_turns"] += len(g.get("hit_budget", []))
                st["budget_hits"] += sum(g.get("hit_budget", []))
            st["tool_calls"] += g.get("n_tool_calls", 0)
    for p in sorted(out_dir.glob("rejected_*.jsonl")):
        for r in iter_samples(p):
            st["rejected"] += 1
            st["jobs_by"][f"{r.get('kind')}{'+tools' if r.get('tools') else ''}/{r.get('mode')}"] += 1
            st["reasons"].update(r["reasons"])
            st["first_reason"][r["reasons"][0]] += 1
    runs = list(iter_samples(out_dir / "runs.jsonl")) if (out_dir / "runs.jsonl").exists() else []
    n = st["kept"] + st["rejected"]
    st.update({"jobs": n, "kept_frac": round(st["kept"] / max(n, 1), 4),
               "budget_hit_frac": round(st["budget_hits"] / max(st["think_turns"], 1), 4),
               "keep_rate_by": {k: round(st["kept_by"][k] / v, 3) for k, v in st["jobs_by"].items()},
               "runs": runs})
    st = {k: dict(v.most_common()) if isinstance(v, Counter) else v for k, v in st.items()}
    write_json(out_dir / "stats.json", st)
    return st


def filter_args(cfg):
    f = (cfg or {}).get("filters", {})
    return {"think_answer": f.get("think_answer", "strict"),
            "max_answer": (cfg or {}).get("generation", {}).get("max_answer_tokens", MAX_ANSWER)}


def rejected_record(sample, reasons):
    return {"id": sample["id"], "kind": sample.get("kind"), "tools": sample.get("tools"), "mode": sample.get("mode"),
            "reasons": reasons, "sample": sample}


def refilter(out_dir, cfg=None, contaminated=None, log_file=None):
    """Re-applies sample_problems (current config) to every stored sample and rewrites kept_* / rejected_*."""
    from bonsai.common import write_jsonl
    out_dir = Path(out_dir)
    if contaminated is None:
        from bonsai.data.decontam import is_contaminated as contaminated
    moved = Counter()
    for kp in sorted(out_dir.glob("kept_*.jsonl")):
        rp = out_dir / kp.name.replace("kept_", "rejected_", 1)
        old_rej = list(iter_samples(rp, log_file)) if rp.exists() else []
        kept, rej = [], [r for r in old_rej if "sample" not in r]
        old_kept = list(iter_samples(kp, log_file))
        for s in old_kept + [r["sample"] for r in old_rej if "sample" in r]:
            probs = sample_problems(s, contaminated, **filter_args(cfg))
            if probs:
                rej.append(rejected_record(s, probs))
            else:
                kept.append(s)
        moved["kept_before"] += len(old_kept)
        moved["kept_after"] += len(kept)
        write_jsonl(kp, kept)
        write_jsonl(rp, rej)
    log(f"refilter {out_dir}: kept {moved['kept_before']:,} -> {moved['kept_after']:,}", log_file)
    return summarize(out_dir)


def drop_rejected(out_dir, reasons):
    """Removes rejected records with any of `reasons` so the next run generates them again (plan 16: "重新生成该批",
    e.g. non_english after a system-prompt fix). Returns the number removed."""
    from bonsai.common import write_jsonl
    n = 0
    for p in sorted(Path(out_dir).glob("rejected_*.jsonl")):
        recs = list(iter_samples(p))
        keep = [r for r in recs if not set(r["reasons"]) & set(reasons)]
        n += len(recs) - len(keep)
        write_jsonl(p, keep)
    return n


def generate_offpolicy(teacher, prompts, out_dir, modes=("think", "no_think"), cfg=None, shard=(0, 1), limit=0,
                       batch=64, chunk=8192, budget=chat.THINK_BUDGET, contaminated=None, log_file=None, retry=()):
    """Generate, filter and append samples for every (prompt, mode) of this shard not done yet (resumable by
    sample id "<prompt id>:<mode>"; a line cut by a hard kill is dropped first, `repair_tail`). Each sample is
    filtered and appended as soon as its row finishes: kept ->
    kept_s<i>of<n>.jsonl, rejected -> rejected_s<i>of<n>.jsonl (id, reasons, the full sample); `chunk` jobs share
    one continuous batch (only its end drains the batch). One throughput line per run -> runs.jsonl, totals ->
    stats.json. `retry`: rejection reasons whose samples are generated again."""
    cfg = cfg or {}
    gen_cfg = cfg.get("generation", {})
    sys_prompts = cfg.get("system_prompts", {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    i, n = shard
    tag = f"s{i}of{n}"
    for name in (f"kept_{tag}.jsonl", f"rejected_{tag}.jsonl"):     # the files this process appends to
        repair_tail(out_dir / name, log_file)
    if retry:
        log(f"retrying {drop_rejected(out_dir, retry):,} rejected samples ({', '.join(retry)})", log_file)
    done = done_ids(out_dir, log_file)
    jobs = []
    for rec in prompts:
        if n > 1 and shard_of(rec["id"], n) != i:
            continue
        for mode in modes:
            if f"{rec['id']}:{mode}" in done:
                continue
            signal = choose_signal(f"{rec['id']}:{mode}", mode, gen_cfg.get("think_soft_frac", 0.3),
                                   gen_cfg.get("no_think_soft_frac", 0.5))
            samp = sampling_for(mode, gen_cfg.get("presence_penalty", 1.5))
            jobs.append((rec, mode, signal, build_job(rec, mode, sys_prompts, signal, samp)))
        if limit and len(jobs) >= limit * len(modes):
            break
    log(f"{teacher.name}: {len(jobs):,} jobs to generate for shard {i}/{n} ({len(done):,} already done)", log_file)
    if jobs:
        teacher.warmup(min(batch, len(jobs)))
    if contaminated is None:
        from bonsai.data.decontam import build_index, is_contaminated
        build_index()
        contaminated = is_contaminated
    run = {"teacher": teacher.name, "shard": tag, "jobs": 0, "kept": 0, "gen_tokens": 0, "prefill_tokens": 0,
           "seconds": 0.0, "decode_s": 0.0, "prefill_s": 0.0, "batch": batch, "started": time.strftime("%F %T")}
    t_run = time.time()
    fargs = filter_args(cfg)
    with open(out_dir / f"kept_{tag}.jsonl", "a") as fk, open(out_dir / f"rejected_{tag}.jsonl", "a") as fr:
        for c in range(0, len(jobs), chunk):
            part = jobs[c:c + chunk]
            kept = [0]

            def write(k, out, part=part, kept=kept):
                rec, mode, signal, job = part[k]
                s = offpolicy_sample(rec, job, out, mode, signal, teacher.name)
                probs = sample_problems(s, contaminated, **fargs)
                if probs:
                    fr.write(json.dumps(rejected_record(s, probs), ensure_ascii=False) + "\n")
                else:
                    fk.write(json.dumps(s, ensure_ascii=False) + "\n")
                    kept[0] += 1
                fk.flush()
                fr.flush()

            teacher.chat_batch([j[3] for j in part], budget=budget,
                               max_answer=gen_cfg.get("max_answer_tokens", MAX_ANSWER),
                               max_calls=gen_cfg.get("max_tool_calls", MAX_TOOL_CALLS),
                               max_total=gen_cfg.get("max_total_tokens", 2048), batch=batch,
                               prefill_batch=gen_cfg.get("prefill_batch", 16), progress_every=120, on_output=write)
            kept = kept[0]
            st = teacher.stats
            for k in ("gen_tokens", "prefill_tokens", "decode_s", "prefill_s"):
                run[k] += st[k]
            run["jobs"] += len(part)
            run["kept"] += kept
            run["seconds"] = time.time() - t_run
            run["tok_per_s"] = run["gen_tokens"] / max(run["decode_s"] + run["prefill_s"], 1e-9)
            run["peak_gb"] = max(run.get("peak_gb", 0.0), st["peak_gb"])
            log(f"  {c + len(part):,}/{len(jobs):,} jobs | kept {kept}/{len(part)} | {st['tok_per_s']:.0f} tok/s "
                f"(chunk) | {run['tok_per_s']:.0f} tok/s (run) | peak {st['peak_gb']:.1f} GB", log_file)
    run["seconds"] = time.time() - t_run
    if run["jobs"]:
        run["gen_tokens_per_job"] = run["gen_tokens"] / run["jobs"]
        run["prefill_tokens_per_job"] = run["prefill_tokens"] / run["jobs"]
        repair_tail(out_dir / "runs.jsonl", log_file)
        with open(out_dir / "runs.jsonl", "a") as f:
            f.write(json.dumps(run) + "\n")
    summarize(out_dir)
    return run
