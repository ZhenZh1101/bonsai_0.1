"""Talk to a trained Bonsai model (Qwen3 template, thinking budget, Python tool in the sandbox).

    python -m scripts.chat --run bonsai-0.1 --source final                      # interactive
    python -m scripts.chat --run bonsai-0.1 --prompt "Why is the sky blue?" --no-think
    python -m scripts.chat --run bonsai-0.1 --prompt "A book costs $12 and a pen $3. Price of 2 books and 4 pens?"
    python -m scripts.chat --run bonsai-0.1 --source base --complete "The capital of France is"

Thinking is on by default (Qwen3's default; the budget is 128 tokens, then chat.think_stop() is forced, the
"..., using the python tool." variant when the tool is offered); --no-think sets
enable_thinking=False (the empty think block is prefilled). /think and /no_think typed in a message are Qwen3's
soft switches (plain text, the last one wins). --tools auto (default) offers the python tool when a message looks
like arithmetic / a word problem; on / off force it. --system none (default: Bonsai is trained without a system
prompt) | bonsai (identity.system_prompt()) | any other text. Sampling defaults to Qwen3's preset for the mode
(thinking T 0.6 / top-p 0.95, non-thinking T 0.7 / top-p 0.8, top-k 20, presence penalty 1.5); --greedy for
argmax. Interactive commands: /clear, /exit; history keeps the last --max-turns user turns.
"""

import argparse
import re

from scripts.chat_eval import add_model_args, load_from_args

_MATH = re.compile(r"\d.*(?:[-+*/^%=x×÷]|\b(?:how many|how much|total|sum|product|percent|average|cost|price|"
                   r"each|per|times|left|remain|calculate|compute|solve)\b)", re.I | re.S)


def looks_like_math(text):
    return bool(_MATH.search(text))


def sampling_for(args, messages, enable_thinking):
    from bonsai import chat as C
    from bonsai import engine as E
    if args.greedy:
        return E.GREEDY
    base = E.PRESETS[C.effective_mode(messages, enable_thinking)]
    return E.Sampling(base.temperature if args.temperature is None else args.temperature,
                      base.top_p if args.top_p is None else args.top_p,
                      base.top_k if args.top_k is None else args.top_k,
                      base.presence_penalty if args.presence_penalty is None else args.presence_penalty)


def show(r, show_thinking=True):
    """Print the assistant side of a ChatResult (thinking, tool calls with their outputs, answer)."""
    for m in r.messages[next(i for i in range(len(r.messages) - 1, -1, -1)
                             if r.messages[i]["role"] == "user") + 1:]:
        if m["role"] == "assistant":
            if show_thinking and (m.get("reasoning_content") or "").strip():
                print(f"  [thinking] {m['reasoning_content'].strip()}")
            for call in m.get("tool_calls") or []:
                print(f"  [python]\n{str(call.get('arguments', {}).get('code', call)).strip()}")
        elif m["role"] == "tool":
            print(f"  [output] {m['content']}")
    flags = [] if r.finished else [f"stopped: {r.stop_reason}"]
    if r.hit_budget:
        flags.append("thinking budget hit")
    print(f"\nassistant> {r.text.strip()}" + (f"   ({', '.join(flags)})" if flags else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap, default_source="final")
    ap.add_argument("--prompt", default="", help="one message, print the reply and exit")
    ap.add_argument("--complete", default="", help="plain completion (base model)")
    think = ap.add_mutually_exclusive_group()
    think.add_argument("--think", dest="thinking", action="store_true", default=True)
    think.add_argument("--no-think", dest="thinking", action="store_false")
    ap.add_argument("--tools", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--system", default="none", help="none | bonsai | <system prompt text>")
    ap.add_argument("--think-budget", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=512, help="answer tokens per assistant turn")
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--presence-penalty", type=float, default=None)
    ap.add_argument("--hide-thinking", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    from bonsai import chat as C
    from bonsai import engine as E
    model, tok, meta, label = load_from_args(args)
    print(f"[loaded {meta.get('path')}]")

    if args.complete:
        pick = lambda v, d: d if v is None else v
        s = E.GREEDY if args.greedy else E.Sampling(pick(args.temperature, 0.7), pick(args.top_p, 0.9),
                                                     pick(args.top_k, 0), pick(args.presence_penalty, 0.0))
        ids = E.generate(model, tok, tok.encode(args.complete, bos=True), args.max_tokens, s.temperature, s.top_p,
                         s.top_k, s.presence_penalty, stop_ids={tok.bos_id}, seed=args.seed)
        print(args.complete + tok.decode([i for i in ids if i != tok.bos_id]))
        return

    system = {"none": "", "bonsai": None}.get(args.system, args.system)
    if system is None:
        from bonsai.identity import system_prompt
        system = system_prompt()

    def reply(history):
        users = [i for i, m in enumerate(history) if m["role"] == "user"]
        msgs = history[users[-args.max_turns]:] if len(users) > args.max_turns else list(history)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
        last = history[users[-1]]["content"]
        use_tools = args.tools == "on" or (args.tools == "auto" and looks_like_math(last))
        r = E.chat(model, tok, msgs, enable_thinking=args.thinking, tools=C.TOOLS if use_tools else None,
                   think_budget=args.think_budget, max_answer_tokens=args.max_tokens,
                   sampling=sampling_for(args, msgs, args.thinking), seed=args.seed)
        show(r, not args.hide_thinking)
        return history + r.messages[len(msgs):]

    if args.prompt:
        reply([{"role": "user", "content": args.prompt}])
        return
    history = []
    print("type /exit to quit, /clear to reset the conversation; /think and /no_think are soft switches")
    while True:
        try:
            msg = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg in ("/exit", "/quit", "exit", "quit"):
            break
        if msg == "/clear":
            history = []
            continue
        if not msg:
            continue
        history = reply(history + [{"role": "user", "content": msg}])


if __name__ == "__main__":
    main()
