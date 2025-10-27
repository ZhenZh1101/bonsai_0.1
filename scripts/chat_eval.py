"""Benchmarks of plan section 15 for one model stage.

    python -m scripts.chat_eval --run bonsai-0.1 --source base                 # CORE, HellaSwag + the chat tasks
    python -m scripts.chat_eval --run bonsai-0.1 --source sft                  # HellaSwag + the chat tasks
    python -m scripts.chat_eval --run bonsai-0.1 --source onpolicy --tasks gsm8k_tool,mode_switch
    python -m scripts.chat_eval ... --limit 200                                # quick look (same subset every time)
    python -m scripts.chat_eval ... --task-limits mmlu=2000,gsm8k_tool=400     # per-task caps
    python -m scripts.chat_eval --weights path/x.safetensors --run tmp         # any weights file with meta
    python -m scripts.chat_eval --preset tiny --run smoke --limit 4            # random tiny model (smoke test)

Tasks: hellaswag, arc_easy, arc_challenge, mmlu, gsm8k_tool, gsm8k_think, fld, mode_switch, core; aliases chat
(the 7 chat tasks), base (core, hellaswag), all. Default: all for --source base, else hellaswag + chat.
Results: logs/<run>/eval_<source>.json + .md (plan-15 expectations alongside). The JSON is rewritten after every
task and merged with earlier results of the same checkpoint, so a separate invocation never loses what was
already measured. Run the three stages (base, sft, onpolicy / final) to fill the table of plan section 15.
"""

import argparse
import json
import os
import time
from pathlib import Path

from bonsai.common import LOG_DIR, log, write_json

STAGES = ("base", "latest", "sft", "onpolicy", "final")


def add_model_args(ap, default_source="sft"):
    ap.add_argument("--run", default="bonsai-0.1")
    ap.add_argument("--source", default=default_source, help="base | latest | sft | onpolicy | final | <path>")
    ap.add_argument("--weights", default="", help="fallback: a weights file / checkpoint (meta.model_config, or "
                                                  "--preset when it has none)")
    ap.add_argument("--preset", default="", help="model preset for --weights without meta; alone: random init")


def load_from_args(args):
    """(model, tok, meta, label) with bf16 weights for inference (engine.cast_for_inference)."""
    from bonsai.engine import cast_for_inference
    model, tok, meta, label = _load(args)
    return cast_for_inference(model), tok, meta, label


def _load(args):
    """bonsai.load.load_model(run, source), or the --weights / --preset fallback."""
    import mlx.core as mx
    if not args.weights and not args.preset:
        from bonsai.load import load_model
        model, tok, meta = load_model(args.run, args.source)
        label = args.source if args.source in STAGES else Path(args.source).stem
        return model, tok, meta, label
    from bonsai.tokenizer import get_tokenizer
    tok = get_tokenizer()
    if args.weights and not args.preset:
        from bonsai.load import load_model
        model, tok, meta = load_model(None, args.weights, tok=tok)
        return model, tok, meta, Path(args.weights).stem
    from bonsai.model import build_model
    model = build_model(args.preset, vocab_size=tok.vocab_size)
    if args.weights:
        model.load_weights(args.weights)
        mx.eval(model.parameters())
        return model, tok, {"path": args.weights, "preset": args.preset}, Path(args.weights).stem
    return model, tok, {"path": f"random:{args.preset}", "preset": args.preset}, f"random_{args.preset}"


def checkpoint_id(meta):
    """Same path != same model: mtime, size and step of the weights."""
    p = Path(meta.get("path", ""))
    if p.is_dir():
        files = sorted(p.glob("*.safetensors")) or sorted(p.iterdir())
        p = files[-1] if files else p
    if not p.exists():
        return str(p)
    st = os.stat(p)
    return f"{st.st_mtime_ns}:{st.st_size}:{meta.get('step')}"


def task_list(spec, source):
    from bonsai.tasks.evals import BASE_TASKS, CHAT_TASKS, TASKS
    if not spec:
        spec = "all" if source == "base" else "hellaswag,chat"
    alias = {"all": BASE_TASKS + CHAT_TASKS, "chat": CHAT_TASKS, "base": BASE_TASKS}
    out = []
    for name in [t.strip() for t in spec.split(",") if t.strip()]:
        for t in alias.get(name, [name]):
            if t not in TASKS:
                raise SystemExit(f"unknown task {t!r} (known: {', '.join(TASKS)})")
            if t not in out:
                out.append(t)
    return out


def _pct(x):
    return f"{100 * x:.1f}"


def markdown(results, run):
    from bonsai.tasks.evals import EXPECTED
    five = "mini" not in run
    lines = [f"# Eval: {run} / {results['source']}", "", f"checkpoint `{results['checkpoint']}`", "",
             "| task | accuracy % | n | baseline / notes | plan 15 (3-day) | plan 15 (5-day) | s |",
             "|---|---|---|---|---|---|---|"]
    for name, r in results["tasks"].items():
        notes = ""
        if "always_letter_accuracy" in r:
            notes = (f"best single letter {_pct(r['best_single_letter_accuracy'])}, always most predicted "
                     f"{_pct(r['always_most_predicted_accuracy'])}, predicted {r['predicted_letters']}")
        elif name == "core":
            notes = f"CORE (centered) over {len(r['tasks'])} tasks"
        elif "by_mode" in r:
            notes = ", ".join(f"{m} {_pct(v['accuracy'])}" for m, v in r["by_mode"].items())
            if r.get("failures"):
                notes += f"; failures {r['failures']}"
            if "strict_accuracy" in r:   # length is scored apart from the mode
                notes += f"; cut by the answer cap {r['truncated']}, pass + finished {_pct(r['strict_accuracy'])}"
        elif "by_depth" in r:
            notes = ", ".join(f"depth {d}: {_pct(v['accuracy'])} (n {v['n']})" for d, v in r["by_depth"].items())
        if "finished_frac" in r:
            notes += ("; " if notes else "") + (f"finished {_pct(r['finished_frac'])}, tool use "
                                                 f"{_pct(r['tool_use_frac'])}, budget hit {_pct(r['hit_budget_frac'])}")
        exp = EXPECTED.get(name)
        e3, e5 = (f"{a}-{b}" for a, b in exp) if exp else ("", "")
        mark = " **" if exp and five and not (exp[1][0] <= 100 * r["accuracy"] <= exp[1][1]) else ""
        lines.append(f"| {name} | {_pct(r['accuracy'])}{mark} | {r['n']} | {notes} | {e3} | {e5} | "
                     f"{r.get('seconds', '')} |")
    lines += ["", "** outside the plan's expected range for this build.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--tasks", default="", help="comma list or alias (chat, base, all)")
    ap.add_argument("--limit", type=int, default=0, help="examples per task (0 = all; core: 500 per CORE task)")
    ap.add_argument("--task-limits", default="", help="per-task caps, e.g. mmlu=2000,gsm8k_tool=400")
    ap.add_argument("--core-tasks", default="", help="only these CORE tasks")
    ap.add_argument("--batch", type=int, default=0, help="chat batch size (0 = task default)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    limits = {k: int(v) for k, v in (kv.split("=") for kv in args.task_limits.split(",") if kv)}

    from bonsai.tasks.evals import TASKS
    model, tok, meta, label = load_from_args(args)
    log(f"evaluating {meta.get('path')} ({label})")
    out = Path(args.out or LOG_DIR / args.run / f"eval_{label}.json")
    ident = checkpoint_id(meta)
    results = {"checkpoint": str(meta.get("path")), "checkpoint_id": ident, "run": args.run, "source": label,
               "tasks": {}}
    if out.exists():
        prev = json.loads(out.read_text())
        if prev.get("checkpoint") == results["checkpoint"] and prev.get("checkpoint_id") == ident:
            results["tasks"] = prev.get("tasks", {})   # same model: keep earlier task results
    for name in task_list(args.tasks, args.source):
        t0 = time.time()
        n = limits.get(name, args.limit)
        kw = {"n": n, "log": log}
        if args.batch:
            kw["batch"] = args.batch
        if name == "core" and args.core_tasks:
            kw["tasks_filter"] = [t for t in args.core_tasks.split(",") if t]
        try:
            res = TASKS[name](model, tok, **kw)
        except (FileNotFoundError, ImportError) as e:   # missing test data or a module not built yet
            log(f"{name}: skipped ({type(e).__name__}: {e})")
            continue
        res["seconds"] = round(time.time() - t0, 1)
        res["limit"] = n
        results["tasks"][name] = res
        log(f"{name}: accuracy {res['accuracy']:.4f} (n {res['n']}, {res['seconds']}s)")
        write_json(out, results)
        out.with_suffix(".md").write_text(markdown(results, args.run))
    log(f"wrote {out} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
