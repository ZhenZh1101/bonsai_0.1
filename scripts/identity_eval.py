"""Identity acceptance (plan section 18): 20 held-out identity prompts (5 "Are you Qwen?"-type) + 20 unrelated tasks
(identity/eval.json), with and without the Bonsai system prompt, in non-thinking and thinking mode.

    python -m scripts.identity_eval --run bonsai-0.1 --source final                 # the release
    python -m scripts.identity_eval --run bonsai-0.1 --source onpolicy --system none --modes no_think

PASS, for every (system prompt, mode) setting tested:
  * all 20 identity replies say the name ("Bonsai"), including the 5 Qwen-type ones
  * the 20 unrelated replies say it 0 times in total
  * no reply (or its thinking) claims to be another model or to be made by another company
Fact errors are reported, not part of PASS. Greedy by default. Writes logs/<run>/identity_eval_<source>.{json,md};
exits 1 on FAIL.
"""

import argparse
import sys
from pathlib import Path

from bonsai.common import LOG_DIR, log, write_json
from bonsai.identity import (eval_qwen_type, eval_set, fact_errors, name_count, other_model_claims, says_name,
                             system_prompt)

SETTINGS = {"none": False, "bonsai": True}
MODES = {"no_think": False, "think": True}   # mode -> enable_thinking


def build_items(ev, use_system, enable_thinking):
    items = []
    for split in ("identity", "unrelated"):
        for i, msgs in enumerate(ev[split]):
            if use_system:
                msgs = [{"role": "system", "content": system_prompt()}] + msgs
            items.append({"split": split, "index": i, "messages": msgs, "enable_thinking": enable_thinking})
    return items


def score(items, results, qwen_type=()):
    """Rows + summary of one setting; `results` are engine ChatResults (or anything with .text / .reasoning)."""
    rows = {"identity": [], "unrelated": []}
    for it, res in zip(items, results):
        text, reasoning = res.text, getattr(res, "reasoning", None) or ""
        msgs = it["messages"]
        rows[it["split"]].append({
            "prompt": msgs[-1]["content"], "turns": sum(m["role"] == "user" for m in msgs),
            "qwen_type": it["split"] == "identity" and it["index"] in qwen_type,
            "reply": text, "reasoning": reasoning, "says_name": says_name(text), "name_count": name_count(text),
            "other_model_claims": other_model_claims(text) + other_model_claims(reasoning),
            "fact_errors": fact_errors(text)})
    ident, unrel = rows["identity"], rows["unrelated"]
    summary = {
        "identity_says_name": sum(r["says_name"] for r in ident), "identity_total": len(ident),
        "qwen_says_name": sum(r["says_name"] for r in ident if r["qwen_type"]),
        "qwen_total": sum(r["qwen_type"] for r in ident),
        "unrelated_name_mentions": sum(r["name_count"] for r in unrel), "unrelated_total": len(unrel),
        "other_model_claims": sum(len(r["other_model_claims"]) for r in ident + unrel),
        "fact_errors": sum(len(r["fact_errors"]) for r in ident + unrel),  # reported, not part of PASS
    }
    summary["pass"] = (summary["identity_says_name"] == len(ident) and summary["unrelated_name_mentions"] == 0
                       and summary["other_model_claims"] == 0)
    return {"summary": summary, **rows}


def run_eval(model, tok, settings, modes, chat_batch=None, max_tokens=256, think_budget=128, sampling=None, seed=0,
             ev=None):
    """{"<system>/<mode>": scored setting} for the requested settings and modes."""
    if chat_batch is None:
        from bonsai.engine import chat_batch
    ev, qt = ev or eval_set(), set(eval_qwen_type())
    out = {}
    for setting in settings:
        for mode in modes:
            items = build_items(ev, SETTINGS[setting], MODES[mode])
            results = chat_batch(model, tok, [{"messages": it["messages"], "enable_thinking": it["enable_thinking"]}
                                              for it in items],
                                 think_budget=think_budget, max_answer_tokens=max_tokens, sampling=sampling, seed=seed)
            out[f"{setting}/{mode}"] = score(items, results, qt)
    return out


def _cell(s):
    return s.replace("\n", "<br>").replace("|", "\\|")


def to_markdown(path, results, args):
    lines = [f"# Identity eval — {path}", "",
             f"temperature {args.temperature}, top-p {args.top_p}, seed {args.seed}, "
             f"max {args.max_tokens} answer tokens", ""]
    for setting, res in results.items():
        s = res["summary"]
        lines += [f"## system prompt / mode: {setting} — {'PASS' if s['pass'] else 'FAIL'}", "",
                  f"identity says name {s['identity_says_name']}/{s['identity_total']} "
                  f"(Qwen-type {s['qwen_says_name']}/{s['qwen_total']}), "
                  f"unrelated name mentions {s['unrelated_name_mentions']} (want 0), "
                  f"other-model claims {s['other_model_claims']} (want 0), fact errors {s['fact_errors']}", ""]
        for split in ("identity", "unrelated"):
            lines += [f"### {split}", "", "| # | ok | prompt | reply |", "|---|---|---|---|"]
            for i, r in enumerate(res[split]):
                ok = r["says_name"] if split == "identity" else r["name_count"] == 0
                ok = ok and not r["other_model_claims"]
                prompt = (f"(turn {r['turns']}) " if r["turns"] > 1 else "") + ("[Qwen] " if r["qwen_type"] else "")
                reply = (f"<think>{r['reasoning']}</think> " if r["reasoning"] else "") + r["reply"]
                lines.append(f"| {i + 1} | {'yes' if ok else 'NO'} | {_cell(prompt + r['prompt'])} | {_cell(reply)} |")
            lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="bonsai-0.1")
    ap.add_argument("--source", default="final", help="base | latest | sft | onpolicy | final | a checkpoint path")
    ap.add_argument("--system", default="both", choices=["both", "none", "bonsai"])
    ap.add_argument("--modes", default="no_think,think", help="comma list of no_think, think")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=256, help="answer tokens per reply (thinking has its own 128)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from bonsai.load import load_model
    model, tok, meta = load_model(args.run, args.source)
    settings = {"both": ["none", "bonsai"], "none": ["none"], "bonsai": ["bonsai"]}[args.system]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad:
        sys.exit(f"unknown mode(s) {bad}: use {list(MODES)}")
    sampling = None if args.temperature == 0 else {"temperature": args.temperature, "top_p": args.top_p}
    results = run_eval(model, tok, settings, modes, max_tokens=args.max_tokens, sampling=sampling, seed=args.seed)
    for setting, res in results.items():
        log(f"{setting}: {res['summary']}")
    tag = args.source if args.source in ("base", "latest", "sft", "onpolicy", "final") else Path(args.source).stem
    out = LOG_DIR / args.run / f"identity_eval_{tag}.json"
    write_json(out, {"checkpoint": meta.get("path"), "settings": results})
    out.with_suffix(".md").write_text(to_markdown(meta.get("path"), results, args))
    ok = all(r["summary"]["pass"] for r in results.values())
    log(f"identity acceptance: {'PASS' if ok else 'FAIL'}  ({out.with_suffix('.md')})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
