"""Qwen3 chat template, thinking modes and the Python tool protocol (plan sections 6 and 13).

Messages are Qwen / OpenAI style dicts:

    {"role": "system" | "user" | "assistant" | "tool", "content": str}
    assistant extras:  "reasoning_content": str      thinking text ("" = the empty think block)
                       "tool_calls": [{"name": "python", "arguments": {"code": "..."}}]
                       "think_prefilled": True        (training only) the empty think block came from
                                                      enable_thinking=False, so it is not supervised

`render_text` produces exactly the string of Qwen3's Jinja template (tokenizer_config.json; tests/test_chat.py
renders the template itself and compares), with one documented deviation: an assistant turn after the last
user query that has `reasoning_content` set (even "") keeps its think block. The template drops an *empty*
block from intermediate tool-calling turns, but a generation loop that continues the KV cache (the teacher in
off-policy generation, the student in on-policy sampling and evaluation) produces it, and it is the format of
the plan's section 13 example. `strict=True` gives the template's behaviour exactly.

`render` turns messages into teacher ids + a loss mask. Control tokens come only from the template's own
constants; content (user text, replies, reasoning, code, tool output, system prompts) is encoded with the
control tokens removed from the tokenizer, so a literal "<|im_start|>" or "<think>" inside content stays ordinary
text. For content without such literals the ids equal Qwen's own `encode(render_text(...))` (acceptance check 3).

Mask = 1 on the supervised assistant turns, from the token after "<|im_start|>assistant\\n" through
"<|im_end|>" (not the newline after it). Which turns: `train_on="last"` (default) = the turns after the last
user query (the response incl. its tool-calling steps); `"all"` = every assistant turn; `"none"`. Tool results
are never supervised.
"""

import json
import re

THINK_BUDGET = 128   # thinking tokens at inference and in training data (plan section 6)
# Qwen3's thinking-budget early exit: appended to the thinking when the budget is hit, then </think> is forced
THINK_STOP = ("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking "
              "directly now.")
# Variant when the python tool is offered: Qwen's sentence alone makes a budget-cut teacher answer without the tool
# (plan 11 tool demos); the student's engine uses the same rule, so training and inference agree
THINK_STOP_TOOL = THINK_STOP[:-1] + ", using the python tool."
SOFT_THINK, SOFT_NO_THINK = "/think", "/no_think"


def think_stop(tools=False):
    """The text forced when the thinking budget is hit (then "\n</think>\n\n")."""
    return THINK_STOP_TOOL if tools else THINK_STOP

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Run a short Python 3 program and return what it prints; as in a notebook, the value of a "
                       "final bare expression is shown too. No network or files, 2 second limit. Use print() to show "
                       "the result.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "The Python code to run."}},
            "required": ["code"],
        },
    },
}
TOOLS = [PYTHON_TOOL]

_CTRL_RE = re.compile(r"(<\|im_start\|>|<\|im_end\|>|<think>|</think>|<tool_call>|</tool_call>|<tool_response>|"
                      r"</tool_response>|<\|endoftext\|>)")
_TOOLS_HEAD = ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
               "You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
_TOOLS_TAIL = ("\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
               "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
               "<args-json-object>}\n</tool_call><|im_end|>\n")


def tojson(x):
    """transformers' `tojson` filter for chat templates (ensure_ascii=False, default separators)."""
    return json.dumps(x, ensure_ascii=False)


def is_tool_response(content):
    return isinstance(content, str) and content.startswith("<tool_response>") and content.endswith("</tool_response>")


def last_query_index(messages):
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m["role"] == "user" and isinstance(m.get("content"), str) and not is_tool_response(m["content"]):
            return i
    return len(messages) - 1


def split_reasoning(message):
    """(reasoning or None, content) of an assistant message, derived the way the template does."""
    content = message.get("content") if isinstance(message.get("content"), str) else ""
    rc = message.get("reasoning_content")
    if isinstance(rc, str):
        return rc, content
    if "</think>" in content:
        reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
        return reasoning, content.split("</think>")[-1].lstrip("\n")
    return None, content


class _Pieces:
    """Typed pieces: ("ctrl", token, sup) from the template, ("text", s, sup) for template text or content."""

    def __init__(self):
        self.items = []

    def tmpl(self, s, sup=False):
        """Template constant: its control-token literals are real control tokens."""
        for chunk in _CTRL_RE.split(s):
            if not chunk:
                continue
            self.items.append(("ctrl" if _CTRL_RE.fullmatch(chunk) else "text", chunk, bool(sup)))

    def ctrl(self, token, sup=False):
        self.items.append(("ctrl", token, bool(sup)))

    def text(self, s, sup=False):
        """Content: always ordinary text."""
        if s:
            self.items.append(("text", s, bool(sup)))


def pieces(messages, tools=None, add_generation_prompt=False, enable_thinking=True, train_on="last", strict=False):
    """The rendered template as typed pieces; "".join(p[1] for p in pieces(...)) is the template string."""
    P = _Pieces()
    msgs = list(messages)
    n = len(msgs)
    if tools:
        P.tmpl("<|im_start|>system\n")
        if msgs and msgs[0]["role"] == "system":
            P.text(msgs[0]["content"])
            P.tmpl("\n\n")
        P.tmpl(_TOOLS_HEAD)
        for t in tools:
            P.tmpl("\n")
            P.text(tojson(t))
        P.tmpl(_TOOLS_TAIL)
    elif msgs and msgs[0]["role"] == "system":
        P.tmpl("<|im_start|>system\n")
        P.text(msgs[0]["content"])
        P.tmpl("<|im_end|>\n")
    lq = last_query_index(msgs)
    for i, m in enumerate(msgs):
        role = m["role"]
        content = m["content"] if isinstance(m.get("content"), str) else ""
        if role == "user" or (role == "system" and i > 0):
            P.tmpl("<|im_start|>" + role + "\n")
            P.text(content)
            P.tmpl("<|im_end|>\n")
        elif role == "system":
            continue  # the leading system message was rendered above
        elif role == "assistant":
            sup = train_on == "all" or (train_on == "last" and i > lq)
            reasoning, content = split_reasoning(m)
            P.tmpl("<|im_start|>assistant\n")
            raw = content  # the template tests the unstripped content before the first tool call
            if i > lq and (i == n - 1 or reasoning or (not strict and isinstance(m.get("reasoning_content"), str))):
                think_sup = sup and not m.get("think_prefilled")
                P.tmpl("<think>\n", think_sup)
                P.text((reasoning or "").strip("\n"), think_sup)
                P.tmpl("\n</think>\n\n", think_sup)
                content = content.lstrip("\n")
            P.text(content, sup)
            for k, call in enumerate(m.get("tool_calls") or []):
                if (k == 0 and raw) or k > 0:
                    P.tmpl("\n", sup)
                call = call.get("function", call)
                args = call["arguments"] if isinstance(call["arguments"], str) else tojson(call["arguments"])
                P.tmpl('<tool_call>\n{"name": "', sup)
                P.text(call["name"], sup)
                P.tmpl('", "arguments": ', sup)
                P.text(args, sup)
                P.tmpl("}\n</tool_call>", sup)
            P.tmpl("<|im_end|>", sup)
            P.tmpl("\n")
        elif role == "tool":
            if i == 0 or msgs[i - 1]["role"] != "tool":
                P.tmpl("<|im_start|>user")
            P.tmpl("\n<tool_response>\n")
            P.text(content)
            P.tmpl("\n</tool_response>")
            if i == n - 1 or msgs[i + 1]["role"] != "tool":
                P.tmpl("<|im_end|>\n")
        else:
            raise ValueError(f"unknown role {role!r}")
    if add_generation_prompt:
        P.tmpl("<|im_start|>assistant\n")
        if enable_thinking is False:
            P.tmpl("<think>\n\n</think>\n\n")
    return P.items


def render_text(messages, tools=None, add_generation_prompt=False, enable_thinking=True, strict=False):
    return "".join(p[1] for p in pieces(messages, tools, add_generation_prompt, enable_thinking, strict=strict))


def _segments(items):
    """Group pieces into control tokens and maximal text runs: [("ctrl", tok, sup) | ("text", s, char_flags)]."""
    out, buf, flags = [], [], []
    for kind, s, sup in items:
        if kind == "ctrl":
            if buf:
                out.append(("text", "".join(buf), flags))
                buf, flags = [], []
            out.append(("ctrl", s, sup))
        else:
            buf.append(s)
            flags.extend([sup] * len(s))
    if buf:
        out.append(("text", "".join(buf), flags))
    return out


def _assemble(qwen, segs, encoded):
    ids, mask = [], []
    k = 0
    for kind, s, f in segs:
        if kind == "ctrl":
            ids.append(qwen.control_ids[s])
            mask.append(int(f))
            continue
        tids, offs = encoded[k]
        k += 1
        for tid, (a, b) in zip(tids, offs):
            ids.append(tid)
            mask.append(int(any(f[a:max(b, a + 1)])))
    return ids, mask


def render(qwen, messages, tools=None, add_generation_prompt=False, enable_thinking=True, train_on="last",
           strict=False):
    """(teacher ids, mask) of one conversation."""
    segs = _segments(pieces(messages, tools, add_generation_prompt, enable_thinking, train_on, strict))
    encoded = [qwen.encode_offsets(s) for kind, s, _ in segs if kind == "text"]
    return _assemble(qwen, segs, encoded)


def render_batch(qwen, convs, train_on="last", strict=False):
    """Many conversations at once (one batched tokenizer call). `convs` items are message lists or
    dicts {"messages": [...], "tools": [...] | None, "add_generation_prompt": bool, "enable_thinking": bool}."""
    all_segs, texts = [], []
    for c in convs:
        if isinstance(c, dict):
            items = pieces(c["messages"], c.get("tools"), c.get("add_generation_prompt", False),
                           c.get("enable_thinking", True), c.get("train_on", train_on), strict)
        else:
            items = pieces(c, None, False, True, train_on, strict)
        segs = _segments(items)
        all_segs.append(segs)
        texts.extend(s for kind, s, _ in segs if kind == "text")
    encs = qwen.plain.encode_batch(texts, add_special_tokens=False) if texts else []
    encoded = [(e.ids, e.offsets) for e in encs]
    out, k = [], 0
    for segs in all_segs:
        m = sum(1 for kind, _, _ in segs if kind == "text")
        out.append(_assemble(qwen, segs, encoded[k:k + m]))
        k += m
    return out


# ---------------------------------------------------------------- thinking modes


def with_soft_switch(content, mode):
    """Append Qwen3's soft switch to a user turn: mode in {"think", "no_think", None}."""
    if mode is None:
        return content
    flag = SOFT_THINK if mode == "think" else SOFT_NO_THINK
    return content.rstrip() + " " + flag


def effective_mode(messages, enable_thinking=True):
    """What the template + soft switches ask for: "think" or "no_think" (the last switch in a user or system
    turn wins; enable_thinking=False always means no thinking)."""
    if enable_thinking is False:
        return "no_think"
    mode = "think"
    for m in messages:
        if m["role"] in ("user", "system") and isinstance(m.get("content"), str):
            for hit in _SWITCH_RE.finditer(m["content"]):
                mode = "no_think" if hit.group(1) else "think"
    return mode


_SWITCH_RE = re.compile(r"/(no_)?think(?![\w-])")


def budget_thinking(qwen, reasoning, budget=THINK_BUDGET):
    """Cut a reasoning text to `budget` teacher tokens; a cut one ends with Qwen3's early-exit sentence."""
    ids = qwen.encode(reasoning)
    if len(ids) <= budget:
        return reasoning, False
    return qwen.decode(ids[:budget]).rstrip() + THINK_STOP, True


# ---------------------------------------------------------------- tool calls


_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_tool_calls(text):
    """[{"name": ..., "arguments": {...}}] from generated assistant text; malformed calls are skipped."""
    calls = []
    for m in _CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"code": args}
            calls.append({"name": obj["name"], "arguments": args if isinstance(args, dict) else {}})
    return calls


def python_call(code):
    return {"name": "python", "arguments": {"code": code}}


def tool_trajectory(question, code, output, answer_text, reasoning="", final_reasoning="", mode=None,
                    system=None):
    """Messages of a one-call tool-use sample (plan section 13): user -> call -> tool result -> answer.

    `mode` adds a soft switch to the question. Both assistant turns carry their (possibly empty) think block."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": with_soft_switch(question, mode)})
    msgs.append({"role": "assistant", "content": "", "reasoning_content": reasoning,
                 "tool_calls": [python_call(code)]})
    msgs.append({"role": "tool", "content": output})
    msgs.append({"role": "assistant", "content": answer_text, "reasoning_content": final_reasoning})
    return msgs
