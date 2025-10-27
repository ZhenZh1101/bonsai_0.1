"""Record -> student ids for every document format of S0, S1 and S2 (plan sections 6, 9, 10; DESIGN 0.1, 0.2).

  text / code      BOS + text                                  (pretraining documents)
  tinygsm_text     BOS + question + ```python``` + output + Answer, as plain text (S1)
  s0_text          BOS + ASCII formal text, full loss          (S0: Dyck, shuffle Dyck, MP-STRUCT)
  s0_derivation    BOS + 'premises: ... ; next: sk', loss only from char `target_start` on (S0)
  chat             SmolTalk-style messages; `mode` no_think (empty think block) | think | mixed
  tool_chat        TinyGSM / GSM8K / OMI-1 as the Qwen3 python-tool protocol, non-thinking
  cot_think_chat   OMI-2 / GSM8K CoT as thinking (<= 128 teacher tokens; longer ones dropped, or with
                   `overlong: cut` cut by chat.budget_thinking up to `max_cot_tokens`) + 'Answer: x'
  think_chat       {user, reasoning, answer} pools (FLD, Reasoning Core) and smoltalk2_think messages
  mc_chat          'Question: ...\\nA. ...' -> 'Answer: B', non-thinking

Every document starts with BOS (`<|endoftext|>`). ChatML conversations are rendered with `chat.render_batch`
(teacher ids, control tokens only from the template) and mapped with the vocabulary subset; out-of-subset tokens
are split losslessly and the mask is repeated over the pieces. `soft_switch_frac` (S2: 0.3) appends the soft
switch matching the sample's mode (/think or /no_think) to the last user turn of a stable-hashed share.
`identity_reasoning` gives identity_chat samples in thinking mode bonsai.identity.short_reasoning, chosen with
the record's category (single-turn records and in_task) and the reply.

Loss (pretraining stages): every token of a document, ChatML ones included, except the tools system block of a
`tool_chat` sample ("<|im_start|>system\\n# Tools ...<|im_end|>\\n", 172 tokens, half of a TinyGSM doc): the
evals and the engine put it in the prompt, the model never writes it, so it is context with mask 0 (BOS 1).
The off-policy SFT sets get their assistant-only masks from bonsai.sft, not from here.

`Renderer.batch(records)` returns, aligned with the records, None (rejected by a length rule) or
(ids, mask) with ids in the shard dtype and mask a uint8 array (1 = training target) or None (full loss);
with stats=True each item also carries (teacher_tokens, out_of_subset_teacher_tokens).
"""

import hashlib

import numpy as np

from bonsai import chat
from bonsai.data.convert import split_cot, tool_messages, tool_ok

LETTERS = "ABCDE"
CHAT_FORMATS = {"chat", "tool_chat", "cot_think_chat", "think_chat", "mc_chat"}
TEXT_FORMATS = {"text", "code", "tinygsm_text", "s0_text", "s0_derivation"}
FORMATS = CHAT_FORMATS | TEXT_FORMATS


def stable_frac(key, salt):
    """Deterministic uniform [0, 1) from a string (same in every process, unlike hash())."""
    return int(hashlib.md5((salt + "\x00" + key).encode()).hexdigest()[:8], 16) / 2 ** 32


def mc_prompt(question, choices):
    lines = [f"Question: {question.strip()}"]
    lines += [f"{LETTERS[i]}. {str(c).strip()}" for i, c in enumerate(choices)]
    return "\n".join(lines)


def tinygsm_text(rec):
    return (f"{rec['question']}\n\n```python\n{rec['script']}\n```\n"
            f"Output: {rec['output']}\nAnswer: {rec['answer']}")


def record_key(rec):
    """Text the stable hashes are keyed on: the first user turn / question / text."""
    if "messages" in rec:
        return next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
    return str(rec.get("user") or rec.get("question") or rec.get("text", ""))[:2000]


def to_student(vmap, teacher_ids, mask=None, dtype=np.int32):
    """(student ids, mask expanded over split pieces or None, number of out-of-subset teacher tokens)."""
    t = np.asarray(teacher_ids, dtype=np.int64)
    s = vmap.t2s[t]
    bad = s < 0
    n_oov = int(bad.sum())
    m = None if mask is None else np.asarray(mask, dtype=np.uint8)
    if n_oov:
        reps = np.ones(len(t), dtype=np.int64)
        reps[bad] = [len(vmap.split(x)) for x in t[bad]]
        s = vmap.to_student(t)
        if m is not None:
            m = np.repeat(m, reps)
    return s.astype(dtype), m, n_oov


class Renderer:
    def __init__(self, tok, fmt, spec=None):
        if fmt not in FORMATS:
            raise ValueError(f"unknown format {fmt}")
        spec = dict(spec or {})
        self.tok, self.fmt, self.spec = tok, fmt, spec
        self.dtype = tok.vmap.shard_dtype
        self.min_tokens = spec.get("min_tokens", 50 if fmt == "text" else 1)
        self.max_tokens = spec.get("max_tokens")              # drop docs longer than this (code: 8192)
        self.max_reply = spec.get("max_reply_tokens", 512)    # chat: per assistant reply
        self.budget = spec.get("max_think_tokens", chat.THINK_BUDGET)
        self.mode = spec.get("mode", "think" if fmt == "think_chat" else "no_think")
        self.switch_frac = float(spec.get("soft_switch_frac", 0.0))
        self.identity_reasoning = bool(spec.get("identity_reasoning", False))
        self.overlong = spec.get("overlong", "drop")          # cot_think_chat: CoT > budget -> drop | cut
        self.max_cot = spec.get("max_cot_tokens", 256) if self.overlong == "cut" else self.budget
        # reasoning that went through budget_thinking ends with THINK_STOP: allow for it
        self.cut_budget = self.budget + len(tok.qwen.encode(chat.think_stop())) + 2
        if self.mode not in ("no_think", "think", "mixed"):
            raise ValueError(f"unknown mode {self.mode}")

    # ------------------------------------------------------------------ public
    def batch(self, records, stats=False):
        if self.fmt in CHAT_FORMATS:
            out = self._chat_batch(records)
        elif self.fmt == "s0_derivation":
            out = self._derivation_batch(records)
        else:
            out = self._text_batch(records)
        return out if stats else [None if o is None else o[:2] for o in out]

    def one(self, rec, stats=False):
        return self.batch([rec], stats)[0]

    # ------------------------------------------------------------------ text
    def _finish(self, teacher_ids, mask=None, bos_mask=0):
        tok = self.tok
        s, m, n_oov = to_student(tok.vmap, teacher_ids, mask, self.dtype)
        ids = np.concatenate([np.array([tok.bos_id], dtype=self.dtype), s])
        if m is not None:
            m = np.concatenate([np.array([bos_mask], dtype=np.uint8), m])
        return ids, m, len(teacher_ids), n_oov

    def _text_batch(self, records):
        if self.fmt == "tinygsm_text":
            texts = [tinygsm_text(r) for r in records]
        else:
            texts = [r["text"] for r in records]
        out = []
        for ids in self.tok.qwen.encode_batch(texts):
            if len(ids) < self.min_tokens or (self.max_tokens and len(ids) > self.max_tokens):
                out.append(None)
            elif self.fmt == "s0_text":  # full loss; BOS 1 too, so the loader sees an all-ones mask
                out.append(self._finish(ids, np.ones(len(ids), dtype=np.uint8), bos_mask=1))
            else:
                out.append(self._finish(ids))
        return out

    def _derivation_batch(self, records):
        encs = self.tok.qwen.plain.encode_batch([r["text"] for r in records], add_special_tokens=False)
        out = []
        for rec, e in zip(records, encs):
            ids = e.ids
            if len(ids) < self.min_tokens or (self.max_tokens and len(ids) > self.max_tokens):
                out.append(None)
                continue
            start = int(rec["target_start"])
            ends = np.array([b for _, b in e.offsets], dtype=np.int64)
            mask = (ends > start).astype(np.uint8)
            out.append(None if not mask.any() else self._finish(ids, mask))
        return out

    # ------------------------------------------------------------------ chat
    def _pick_mode(self, rec, has_reasoning):
        mode = self.mode
        if mode == "mixed":
            mode = "think" if stable_frac(record_key(rec), "mode") < 0.5 else "no_think"
        if mode == "think" and not has_reasoning:
            return None if self.mode == "think" else "no_think"
        return mode

    def _switch(self, rec):
        return self.switch_frac > 0 and stable_frac(record_key(rec), "switch") < self.switch_frac

    def _finalize_messages(self, rec, msgs, mode, reasoning):
        """Last assistant turn gets its think block (reasoning or empty); soft switch on the last user turn."""
        msgs = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in msgs]
        msgs[-1]["reasoning_content"] = reasoning if mode == "think" else ""
        lq = chat.last_query_index(msgs)
        user = msgs[lq]["content"]
        if self._switch(rec) or any(chat._SWITCH_RE.search(m["content"]) for m in msgs if m["role"] == "user"):
            msgs[lq] = {"role": "user", "content": chat.with_soft_switch(user, mode)}
        return msgs

    def conversation(self, rec):
        """(conv dict for chat.render_batch, [(text, max_tokens) checks]) or None."""
        fmt = self.fmt
        if fmt == "tool_chat":
            if not tool_ok(rec):
                return None
            mode = "no_think"
            msgs = tool_messages(rec, mode=mode if self._switch(rec) else None)
            return {"messages": msgs, "tools": chat.TOOLS}, []
        if fmt == "mc_chat":
            msgs = [{"role": "user", "content": mc_prompt(rec["question"], rec["choices"])},
                    {"role": "assistant", "content": f"Answer: {LETTERS[int(rec['answer'])]}"}]
            return {"messages": self._finalize_messages(rec, msgs, "no_think", "")}, []
        if fmt == "cot_think_chat":
            return self._cot_conversations([rec])[0]
        if fmt == "think_chat" and "messages" not in rec:
            msgs = [{"role": "user", "content": rec["user"].strip()},
                    {"role": "assistant", "content": rec["answer"].strip()}]
            reasoning = rec["reasoning"].strip()
            mode = self._pick_mode(rec, bool(reasoning))
            if mode is None:
                return None
            checks = [(msgs[-1]["content"], self.max_reply)] + ([(reasoning, self.cut_budget)] if mode == "think" else [])
            return {"messages": self._finalize_messages(rec, msgs, mode, reasoning)}, checks
        # chat / think_chat over a messages list (SmolTalk, smoltalk2_think, identity_chat)
        msgs = [m for m in rec["messages"] if m["role"] in ("user", "assistant", "system")]
        if not msgs or msgs[-1]["role"] != "assistant" or not any(m["role"] == "user" for m in msgs):
            return None
        reasoning = msgs[-1].get("reasoning_content") or ""
        if not reasoning and self.identity_reasoning and self.mode != "no_think":
            try:
                from bonsai.identity import short_reasoning
                users = [m["content"] for m in msgs if m["role"] == "user"]
                cat = rec.get("category")   # it describes the first user turn (a later one may be a follow-up)
                reasoning = short_reasoning(users[-1], key=record_key(rec), reply=msgs[-1]["content"],
                                            category=cat if len(users) == 1 or cat == "in_task" else None)
            except ImportError:
                reasoning = ""
        mode = self._pick_mode(rec, bool(reasoning.strip()))
        if mode is None:
            return None
        checks = [(m["content"], self.max_reply) for m in msgs if m["role"] == "assistant"]
        if mode == "think":
            checks.append((reasoning, self.cut_budget))
        return {"messages": self._finalize_messages(rec, msgs, mode, reasoning.strip())}, checks

    def _cot_conversations(self, records):
        """CoT -> thinking; > budget tokens: dropped, or (overlong: cut) cut by budget_thinking up to max_cot."""
        qwen = self.tok.qwen
        parts = [split_cot(r["cot"]) for r in records]
        encs = qwen.encode_batch([p[0] if p else "" for p in parts])
        out = []
        for rec, p, ids in zip(records, parts, encs):
            if p is None or len(ids) > self.max_cot:
                out.append(None)
                continue
            steps, ans = p
            if len(ids) > self.budget:   # same text as chat.budget_thinking(qwen, steps)
                steps = qwen.decode(ids[:self.budget]).rstrip() + chat.think_stop()
            msgs = [{"role": "user", "content": rec["question"].strip()},
                    {"role": "assistant", "content": f"Answer: {ans}"}]
            out.append(({"messages": self._finalize_messages(rec, msgs, "think", steps)}, []))
        return out

    def conversations(self, records):
        """The conversation dicts ({"messages", "tools"}) the ChatML formats render, before length checks."""
        convs = self._cot_conversations(records) if self.fmt == "cot_think_chat" else \
            [self.conversation(r) for r in records]
        return [None if c is None else c[0] for c in convs]

    def _chat_batch(self, records):
        convs = self._cot_conversations(records) if self.fmt == "cot_think_chat" else \
            [self.conversation(r) for r in records]
        check_texts = [t for c in convs if c is not None for t, _ in c[1]]
        lens = iter(len(ids) for ids in self.tok.qwen.encode_batch(check_texts)) if check_texts else iter(())
        keep = []
        for c in convs:
            if c is None:
                keep.append(None)
                continue
            got = [next(lens) for _ in c[1]]   # consume every check before testing (keeps the iterator aligned)
            ok = all(n <= limit for n, (_, limit) in zip(got, c[1]))
            keep.append(c[0] if ok else None)
        todo = [c for c in keep if c is not None]
        rendered = iter(chat.render_batch(self.tok.qwen, todo)) if todo else iter(())
        out = []
        for c in keep:
            if c is None:
                out.append(None)
                continue
            ids, _ = next(rendered)   # the assistant-only SFT mask is not used: pretraining trains on everything
            if len(ids) < self.min_tokens or (self.max_tokens and len(ids) > self.max_tokens):
                out.append(None)
            elif self.fmt == "tool_chat":   # the tools system block, through its "<|im_end|>\n", is context
                end = ids.index(self.tok.qwen.control_ids["<|im_end|>"]) + 2
                out.append(self._finish(ids, [0] * end + [1] * (len(ids) - end), bos_mask=1))
            else:
                out.append(self._finish(ids))
        return out


def split_long(ids, cap, bos_id, mask=None, min_tail=16):
    """Split a doc longer than `cap` (= seq_len + 1) into BOS-prefixed chunks that each fit a row.

    Returns [(ids, mask)]; a continuation chunk's BOS gets the mask value of the doc's own BOS."""
    if len(ids) <= cap:
        return [(ids, mask)]
    chunks = [(ids[:cap], None if mask is None else mask[:cap])]
    step = cap - 1
    bos = np.array([bos_id], dtype=ids.dtype)
    for i in range(cap, len(ids), step):
        piece = ids[i:i + step]
        if len(piece) < min_tail:
            continue
        if mask is None:
            chunks.append((np.concatenate([bos, piece]), None))
        else:
            mp = mask[i:i + step]
            if mp.any():
                chunks.append((np.concatenate([bos, piece]), np.concatenate([mask[:1], mp])))
    return chunks
