"""End-to-end smoke test of the whole pipeline on the tiny model (plan sections 14 and 17: "冒烟测试").

    python -m scripts.smoke_test                # ~10-15 min, needs tok/smoke/* (scripts.build_shards --smoke)
    python -m scripts.smoke_test --skip-teacher # without the Qwen3-0.6B stages (on-policy, teacher KL)

Every stage runs through its real CLI, exactly as the run scripts call it:

  1. pretraining S0 -> S1 -> transition -> S2 on tok/smoke, uninterrupted (run bonsai-smoke-ref) and with a
     SIGINT in the middle + `--resume auto` (run bonsai-smoke); the per-step losses must match exactly
  2. scripts.ckpt_check on the resumed run
  3. off-policy SFT on tok/smoke/offpolicy (2 epochs)
  4. on-policy distillation, 3 steps with the real Qwen3-0.6B teacher (configs/smoke/onpolicy.yaml)
  5. identity mini-SFT on tok/smoke/identity_sft
  6. evals (every plan-15 task, 4 items each), teacher KL, identity acceptance, manual-test sheet, HF export

Scores are meaningless on a tiny model; the test checks that every stage runs and hands the next one what it
needs. Report: logs/bonsai-smoke/smoke_report.json. Uses ckpt/bonsai-smoke*/ and logs/bonsai-smoke*/ (removed
at the start).
"""

import argparse
import json
import shutil
import signal
import subprocess
import sys
import time

from bonsai.common import CKPT_DIR, CODE_ROOT, DATA_ROOT, LOG_DIR, log, write_json

RUN, REF = "bonsai-smoke", "bonsai-smoke-ref"
PY = [sys.executable, "-m"]
TRAIN = ["scripts.train", "--model", "tiny", "--s0", "smoke/s0", "--s1", "smoke/s1", "--s2", "smoke/s2",
         "--val-dir", "smoke/val", "--val-s0-dir", "smoke/val_s0", "--tokens", str(8192 * 60), "--batch-tokens",
         "8192", "--micro-batch", "4", "--warmup", "10", "--transition-steps", "6", "--eval-every", "20",
         "--eval-tokens", "16384", "--sample-every", "30", "--core-max-per-task", "2", "--log-every", "5",
         "--first-save", "10", "--save-every", "10"]


def run(args, report, name, ok_codes=(0,), timeout=1800):
    t0 = time.time()
    log(f"== {name}: {' '.join(args)}")
    p = subprocess.run(PY + args, cwd=CODE_ROOT, capture_output=True, text=True, timeout=timeout)
    report[name] = {"returncode": p.returncode, "seconds": round(time.time() - t0, 1),
                    "ok": p.returncode in ok_codes, "tail": (p.stdout + p.stderr).strip().splitlines()[-6:]}
    log(f"   {name}: exit {p.returncode} in {report[name]['seconds']}s")
    if p.returncode not in ok_codes:
        print(p.stdout[-4000:], p.stderr[-4000:], sep="\n")
        raise SystemExit(f"smoke test failed at {name}")
    return p


def step_losses(run_name):
    out = {}
    for line in open(LOG_DIR / run_name / "metrics.jsonl"):
        r = json.loads(line)
        if r.get("kind") == "train":   # step_loss: the loss of that step (loss is a window average)
            out[r["step"]] = r["step_loss"]
    return out


def interrupted_run(report):
    """Start training, SIGINT it once it is past step 25, then resume."""
    t0 = time.time()
    p = subprocess.Popen(PY + TRAIN + ["--run", RUN], cwd=CODE_ROOT, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    log_file = LOG_DIR / RUN / "train.log"
    while p.poll() is None:
        if log_file.exists() and "step     25/" in log_file.read_text():
            p.send_signal(signal.SIGINT)
            break
        time.sleep(0.2)
    code = p.wait(timeout=600)
    report["pretrain_interrupt"] = {"returncode": code, "seconds": round(time.time() - t0, 1), "ok": code in (0, 3)}
    log(f"   interrupted run exit {code} (3 = stopped by signal, checkpoint saved)")
    run(TRAIN + ["--run", RUN, "--resume", "auto"], report, "pretrain_resume")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-teacher", action="store_true")
    args = ap.parse_args()
    for d in (CKPT_DIR / RUN, CKPT_DIR / REF, LOG_DIR / RUN, LOG_DIR / REF, DATA_ROOT / "export" / RUN):
        shutil.rmtree(d, ignore_errors=True)
    report, t0 = {}, time.time()

    run(TRAIN + ["--run", REF], report, "pretrain_reference")
    interrupted_run(report)
    ref, got = step_losses(REF), step_losses(RUN)
    common = sorted(set(ref) & set(got))
    diff = max(abs(ref[s] - got[s]) for s in common) if common else None
    report["resume_matches"] = {"steps_compared": len(common), "max_abs_diff": diff,
                                "ok": bool(common) and diff < 1e-4 and max(common) == max(ref)}
    log(f"   resume vs uninterrupted: {len(common)} steps, max |diff| {diff}")
    if not report["resume_matches"]["ok"]:
        raise SystemExit("resumed losses differ from the uninterrupted run")
    if not (CKPT_DIR / RUN / "base.safetensors").exists():
        raise SystemExit("pretraining did not write base.safetensors")

    ckpts = sorted(p for p in (CKPT_DIR / RUN).glob("step_*") if p.is_dir())
    run(["scripts.ckpt_check", "--run", RUN, "--steps", "3", "--checkpoint", str(ckpts[-2])], report, "ckpt_check")
    run(["scripts.sft", "--run", RUN, "--data", "tok/smoke/offpolicy", "--epochs", "2", "--batch-tokens", "8192",
         "--micro-batch", "4", "--warmup", "5", "--eval-every", "0"], report, "sft")
    if args.skip_teacher:
        shutil.copy(CKPT_DIR / RUN / "sft_final.safetensors", CKPT_DIR / RUN / "onpolicy_final.safetensors")
    else:
        run(["scripts.distill_onpolicy", "--config", "configs/smoke/onpolicy.yaml", "--run", RUN, "--steps", "3",
             "--set", "eval_every=3", "--set", "ckpt_every=3"], report, "onpolicy")
    run(["scripts.identity_sft", "--run", RUN, "--data", "tok/smoke/identity_sft", "--steps", "10",
         "--batch-tokens", "8192", "--micro-batch", "4", "--warmup", "2"], report, "identity_sft")

    run(["scripts.chat_eval", "--run", RUN, "--source", "final", "--tasks", "all", "--limit", "4",
         "--core-tasks", "copa,piqa"], report, "chat_eval")
    if not args.skip_teacher:
        run(["scripts.teacher_kl", "--run", RUN, "--source", "final", "--prompts",
             "data/smoke/prompts_heldout.jsonl", "--n", "6"], report, "teacher_kl")
    run(["scripts.identity_eval", "--run", RUN, "--source", "final", "--system", "none", "--modes", "no_think",
         "--max-tokens", "48"], report, "identity_eval", ok_codes=(0, 1))   # a tiny model FAILs; plumbing only
    run(["scripts.manual_test", "--run", RUN, "--source", "final", "--max-tokens", "32"], report, "manual_test")
    run(["scripts.export_hf", "--run", RUN, "--source", "final", "--calib-tokens", "4096"], report, "export_hf")

    report["total_seconds"] = round(time.time() - t0, 1)
    report["result"] = "PASS" if all(v.get("ok", True) for v in report.values() if isinstance(v, dict)) else "FAIL"
    write_json(LOG_DIR / RUN / "smoke_report.json", report)
    log(f"smoke test {report['result']} in {report['total_seconds'] / 60:.1f} min -> {LOG_DIR / RUN / 'smoke_report.json'}")


if __name__ == "__main__":
    main()
