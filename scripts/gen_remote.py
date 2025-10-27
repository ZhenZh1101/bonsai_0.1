"""Helpers for running the off-policy teacher generation on a second Mac (plan sections 3, 11; runs/remote_gen/).

    python -m scripts.gen_remote plan       # settings for this machine -> shell lines + data/remote_plan.json
    python -m scripts.gen_remote selftest   # judge data/selftest (kept samples, speed) and project the full run
    python -m scripts.gen_remote status     # progress of the shard, speed, ETA, whether the job is alive
    python -m scripts.gen_remote launch     # start the full generation detached (caffeinate; survives logout)
    python -m scripts.gen_remote stop       # stop it (resumable: ./run_gen.sh continues where it stopped)

`plan` picks the teacher and batch from the machine's memory (env TEACHER / BATCH / MEM_GB / SHARD override):
Qwen3-1.7B (the plan's teacher) in bf16 from 16 GB; on an 8 GB Mac (the M2 Mac mini) the same model quantized to
8 bits (Qwen3-1.7B-8bit, scripts.quantize_teacher: 1.9 GB instead of 3.8 GB of weights, close to bf16), since bf16
would leave room for a batch of ~8 and read 3.8 GB per decode step. Qwen3-0.6B (TEACHER=Qwen3-0.6B) is the plan's
fallback. The teachers have the same KV size per token (28 layers x 8 KV heads x 128), about 112 KB, so the batch
is sized from the memory left after the weights.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from bonsai.common import DATA_ROOT, LOG_DIR, iter_jsonl, write_json

PLAN_FILE = DATA_ROOT / "data" / "remote_plan.json"
TEACHER_GB = {"Qwen3-0.6B": 1.4, "Qwen3-1.7B": 3.8, "Qwen3-1.7B-8bit": 1.9}   # weights in memory


def sysctl(name):
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def machine():
    ram = int(sysctl("hw.memsize") or 0) / 1024 ** 3
    st = os.statvfs(DATA_ROOT)
    return {"chip": sysctl("machdep.cpu.brand_string") or "unknown", "ram_gb": round(ram, 1),
            "free_disk_gb": round(st.f_bavail * st.f_frsize / 1024 ** 3, 1), "host": os.uname().nodename}


def choose(m):
    ram = m["ram_gb"]
    teacher = os.environ.get("TEACHER") or ("Qwen3-1.7B" if ram >= 15 else "Qwen3-1.7B-8bit")
    teacher = teacher.split("/")[-1]
    if teacher not in TEACHER_GB:
        raise SystemExit(f"TEACHER={teacher}: use one of {sorted(TEACHER_GB)}")
    # MLX may use ~65% of unified memory before macOS starts swapping; keep the rest for the system
    mem = float(os.environ.get("MEM_GB") or max(3.0, round(ram * 0.65 - 0.5, 1)))
    kv_gb = max(0.5, mem - TEACHER_GB[teacher] - 1.0)
    # a row holds up to ~1,100 tokens (system prompt + prompt + 128 thinking + <= 400 answer + tool turn)
    batch = int(os.environ.get("BATCH") or max(4, min(64, int(kv_gb * 1024 ** 3 / (112 * 1024 * 1100)) // 4 * 4)))
    shard = os.environ.get("SHARD") or "0/1"
    return {"teacher": teacher, "batch": batch, "mem_gb": mem, "shard": shard}


def shard_jobs(shard, modes=("think", "no_think")):
    from bonsai.teacher import shard_of
    i, n = (int(x) for x in shard.split("/"))
    ids = [r["id"] for r in iter_jsonl(DATA_ROOT / "data" / "prompts.jsonl")]
    return sum(1 for pid in ids if n == 1 or shard_of(pid, n) == i) * len(modes)


def count_lines(path):
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        return sum(1 for line in f if line.endswith(b"\n"))


def runs(out):
    p = out / "runs.jsonl"
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def cmd_plan(args):
    m = machine()
    c = choose(m)
    total = shard_jobs(c["shard"])
    write_json(PLAN_FILE, {**m, **c, "jobs": total, "time": time.strftime("%F %T")})
    for k, v in (("TEACHER", c["teacher"]), ("BATCH", c["batch"]), ("MEM_GB", c["mem_gb"]), ("SHARD", c["shard"]),
                 ("RAM_GB", m["ram_gb"]), ("FREE_GB", m["free_disk_gb"]), ("JOBS", total)):
        print(f"{k}={v}")


def cmd_selftest(args):
    out = DATA_ROOT / "data" / "selftest"
    st = json.loads((out / "stats.json").read_text())
    rs = runs(out)
    gen = sum(r.get("gen_tokens", 0) for r in rs)
    secs = sum(r.get("seconds", 0) for r in rs) or 1.0
    jobs = st["jobs"]
    plan = json.loads(PLAN_FILE.read_text()) if PLAN_FILE.exists() else {}
    # the self-test is one full batch, so its time per job is close to the real run's (tails of long answers aside)
    hours = plan.get("jobs", 0) * secs / max(jobs, 1) / 3600
    print(f"self-test: {jobs} jobs, kept {st['kept']} ({st['kept_frac']:.0%}), {gen:,} generated tokens in "
          f"{secs:.0f} s = {gen / secs:,.0f} tok/s; rejection reasons {st.get('first_reason', {})}")
    if plan:
        print(f"full shard {plan['shard']}: {plan['jobs']:,} jobs with {plan['teacher']} at batch {plan['batch']}: "
              f"~{hours:.0f} h at this pace")
    if st["kept"] == 0:
        raise SystemExit("self-test kept no sample: check logs/gen_offpolicy.log before starting the real run")


def cmd_status(args):
    plan = json.loads(PLAN_FILE.read_text()) if PLAN_FILE.exists() else None
    if plan is None:
        raise SystemExit("no data/remote_plan.json: run ./run_gen.sh first")
    i, n = plan["shard"].split("/")
    tag = f"s{i}of{n}"
    out = DATA_ROOT / "data" / "offpolicy"
    kept, rej = count_lines(out / f"kept_{tag}.jsonl"), count_lines(out / f"rejected_{tag}.jsonl")
    done, total = kept + rej, plan["jobs"]
    pid_file = LOG_DIR / "gen.pid"
    alive = alive_pid() is not None
    print(f"{plan['host']} ({plan['chip']}, {plan['ram_gb']} GB): {plan['teacher']}, batch {plan['batch']}, "
          f"shard {plan['shard']}")
    print(f"progress: {done:,} / {total:,} jobs ({done / max(total, 1):.1%}), kept {kept:,} "
          f"({kept / max(done, 1):.0%}), job {'RUNNING' if alive else 'not running'}")
    # pace from the log: jobs finished since this process started (kept + rejected lines over wall time)
    started = None
    if pid_file.exists():
        parts = pid_file.read_text().split()
        if len(parts) >= 3:
            started = (float(parts[1]), int(parts[2]))
    if alive and started and done > started[1]:
        rate = (done - started[1]) / max(time.time() - started[0], 1)
        eta = (total - done) / rate
        print(f"pace: {rate * 3600:,.0f} jobs/h -> done in ~{eta / 3600:.1f} h "
              f"(around {time.strftime('%m-%d %H:%M', time.localtime(time.time() + eta))})")
    elif done >= total:
        print("all jobs done: results are packed in results/ (or run ./pack_results.sh)")
    for r in runs(out)[-3:]:
        print(f"  run {r.get('started')}: {r.get('jobs', 0):,} jobs, kept {r.get('kept', 0):,}, "
              f"{r.get('tok_per_s', 0):,.0f} tok/s")


def alive_pid():
    pid_file = LOG_DIR / "gen.pid"
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text().split()[0])
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def cmd_launch(args):
    """gen_offpolicy for the whole shard, then pack_results.sh; own session, so closing the terminal is fine."""
    if alive_pid():
        raise SystemExit("generation is already running (./status.sh)")
    plan = json.loads(PLAN_FILE.read_text())
    i, n = plan["shard"].split("/")
    out = DATA_ROOT / "data" / "offpolicy"
    done = count_lines(out / f"kept_s{i}of{n}.jsonl") + count_lines(out / f"rejected_s{i}of{n}.jsonl")
    gen = (f"caffeinate -dims .venv/bin/python -m scripts.gen_offpolicy --teacher Qwen/{plan['teacher']} "
           f"--prompts data/prompts.jsonl --modes think,no_think --budget 128 --shard {plan['shard']} "
           f"--batch {plan['batch']} --memory-limit-gb {plan['mem_gb']}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "gen_offpolicy.stdout.log", "a") as log_f:
        p = subprocess.Popen(["bash", "-c", f"{gen} && bash pack_results.sh"], cwd=DATA_ROOT, stdout=log_f,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    (LOG_DIR / "gen.pid").write_text(f"{p.pid} {time.time():.0f} {done}\n")
    print(f"started (pid {p.pid}): {gen}")


def cmd_stop(args):
    import signal
    pid = alive_pid()
    if pid is None:
        print("not running")
        return
    os.killpg(pid, signal.SIGTERM)   # the whole process group: bash, caffeinate, python
    print(f"stopped process group {pid}; ./run_gen.sh resumes where it stopped")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "selftest", "status", "launch", "stop"])
    args = ap.parse_args()
    {"plan": cmd_plan, "selftest": cmd_selftest, "status": cmd_status, "launch": cmd_launch,
     "stop": cmd_stop}[args.cmd](args)


if __name__ == "__main__":
    main()
