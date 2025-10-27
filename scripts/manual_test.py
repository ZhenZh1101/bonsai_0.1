"""The frozen manual tests of plan section 15 -> a markdown sheet to grade by hand.

    python -m scripts.manual_test --run bonsai-0.1 --source final
    python -m scripts.manual_test --run bonsai-0.1 --source sft --mode think     # chat prompts in thinking mode

prompts/manual_tests.json: pagoda's 20 (writing, instructions, qa, math) + 5 mode-switch + 5 multi-turn tool-call
conversations. Writing / instructions / QA run in --mode no_think (default: enable_thinking=False) or think; math
and tool conversations run non-thinking with the python tool. Sampling: Qwen3's preset for the mode, seeded.
Writes logs/<run>/manual_tests_<source>.md: every prompt, thinking, tool calls with outputs, the reply and a
result column (math / tool answers and mode-switch formats are pre-checked automatically, the rest is blank).
Judging (plan 15): writing 4/5, instructions 3/5, QA 2/5, math 2-3/5 are normal at this scale; mode switching
below 5/5 -> check the template and special tokens; tool calls below 3/5 -> the protocol format and the sandbox
continuation.
"""

import argparse
import json
from pathlib import Path

from scripts.chat_eval import add_model_args, load_from_args


def cell(s):
    return (s or "").strip().replace("|", "\\|").replace("\n", "<br>")


def conv_text(messages):
    parts = []
    for m in messages:
        if m["role"] == "tool":
            parts.append(f"*tool:* {m['content']}")
        elif m.get("tool_calls"):
            parts.append("*assistant:* " + " ".join(f"`{c['arguments']['code']}`" for c in m["tool_calls"]))
        else:
            parts.append(f"*{m['role']}:* {m['content']}")
    return "<br>".join(cell(p) for p in parts)


def reply_text(r):
    """Thinking, tool calls / outputs and the answer of everything generated after the last user turn."""
    lu = max(i for i, m in enumerate(r.messages) if m["role"] == "user")
    out = []
    for m in r.messages[lu + 1:]:
        if m["role"] == "tool":
            out.append(f"[output] {m['content']}")
            continue
        if (m.get("reasoning_content") or "").strip():
            out.append(f"[thinking] {m['reasoning_content'].strip()}")
        for c in m.get("tool_calls") or []:
            out.append(f"[python] {c.get('arguments', {}).get('code', '')}")
        if m.get("content", "").strip():
            out.append(m["content"].strip())
    if not r.finished:
        out.append(f"[stopped: {r.stop_reason}]")
    return cell("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap, default_source="final")
    ap.add_argument("--mode", choices=("no_think", "think"), default="no_think",
                    help="writing / instructions / qa: enable_thinking=False or thinking mode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--out", default="", help="default logs/<run>/manual_tests_<source>.md")
    args = ap.parse_args()

    from bonsai import chat as C
    from bonsai import engine as E
    from bonsai.common import LOG_DIR, PROMPT_DIR
    from bonsai.sandbox import extract_answer, numbers_equal
    from bonsai.tasks.evals import mode_check

    tests = json.loads((PROMPT_DIR / "manual_tests.json").read_text())
    model, tok, meta, label = load_from_args(args)

    def run(messages, enable_thinking, tools, seed):
        s = "greedy" if args.greedy else E.PRESETS[C.effective_mode(messages, enable_thinking)]
        return E.chat(model, tok, messages, enable_thinking=enable_thinking, tools=tools,
                      max_answer_tokens=args.max_tokens, sampling=s, seed=seed)

    sampling = "greedy" if args.greedy else "Qwen3 presets (T 0.6 / 0.7, top-p 0.95 / 0.8, top-k 20, presence 1.5)"
    lines = [f"# Manual tests: {meta.get('path')}", "", f"chat mode {args.mode}; sampling {sampling}; seed {args.seed}",
             ""]
    n = args.seed
    for cat in ("writing", "instructions", "qa", "math"):
        lines += [f"## {cat}", "", "| # | result | prompt | reply |", "|---|---|---|---|"]
        for i, item in enumerate(tests[cat]):
            prompt = item["prompt"] if isinstance(item, dict) else item
            msgs = [{"role": "user", "content": prompt}]
            if cat == "math":
                r = run(msgs, False, C.TOOLS, n)
            else:
                r = run(msgs, args.mode == "think", None, n)
            n += 1
            verdict = ""
            if cat == "math" and isinstance(item, dict) and item.get("answer") is not None:
                verdict = "auto PASS" if numbers_equal(extract_answer(r.text), item["answer"]) else "auto FAIL"
            lines.append(f"| {i + 1} | {verdict} | {cell(prompt)} | {reply_text(r)} |")
        lines += ["", f"**{cat}: __ / {len(tests[cat])}**", ""]

    lines += ["## mode switching", "", "| # | expect | auto | conversation | reply |", "|---|---|---|---|---|"]
    for i, item in enumerate(tests["mode_switch"]):
        et = item.get("enable_thinking", True)
        r = run(item["messages"], et, None, n)
        n += 1
        ok, why = mode_check(tok, r, item["expect"])   # the mode, not the length: a cut reply can pass
        auto = ("PASS" + ("" if r.finished else f" (cut: {r.stop_reason})")) if ok else "FAIL: " + why
        lines.append(f"| {i + 1} | {item['expect']} | {auto} | "
                     f"{conv_text(item['messages'])} | {reply_text(r)} |")
    lines += ["", f"**mode switching: __ / {len(tests['mode_switch'])}** (expected 5/5)", ""]

    lines += ["## multi-turn tool calls", "", "| # | answer | auto | conversation | reply |", "|---|---|---|---|---|"]
    for i, item in enumerate(tests["tool_multi_turn"]):
        r = run(item["messages"], False, C.TOOLS, n)
        n += 1
        ok = numbers_equal(extract_answer(r.text), item["answer"])
        auto = ("PASS" if ok else "FAIL") + f" ({r.n_tool_calls} call{'s' if r.n_tool_calls != 1 else ''})"
        lines.append(f"| {i + 1} | {item['answer']} | {auto} | {conv_text(item['messages'])} | {reply_text(r)} |")
    lines += ["", f"**tool calls: __ / {len(tests['tool_multi_turn'])}** (expected >= 3/5)", ""]

    out = Path(args.out) if args.out else LOG_DIR / args.run / f"manual_tests_{label}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
