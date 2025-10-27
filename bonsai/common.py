"""Paths, logging and small utilities shared by every Bonsai script.

Data layout (plan section 3), rooted at $BONSAI_HOME (defaults to the repo directory):

    raw/        downloads; raw/filtered/<pool>.jsonl after filtering (pagoda pools are symlinked in)
    tok/        token shards in *student* ids: s0/ s1/ s2/ val/ val_s0/ offpolicy/ identity_sft/
    tokenizer/  Qwen3 tokenizer files (pinned commit + sha256) and the vocab subset (vocab_subset.npz)
    teachers/   Qwen3-0.6B and Qwen3-1.7B (mlx-lm loads them from here)
    data/       prompt pools and the off-policy distillation set (teacher generations)
    ckpt/       rolling full checkpoints + bf16 milestones, ckpt/bonsai-0.1/ ...
    logs/       training logs, eval results, samples, logs/bonsai-0.1/ ...
    sandbox/    scratch dirs for the Python executor
"""

import json
import os
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("BONSAI_HOME", CODE_ROOT)).expanduser().resolve()

RAW_DIR = DATA_ROOT / "raw"
FILTERED_DIR = RAW_DIR / "filtered"
TOK_DIR = DATA_ROOT / "tok"
TOKENIZER_DIR = DATA_ROOT / "tokenizer"
TEACHER_DIR = DATA_ROOT / "teachers"
DATA_DIR = DATA_ROOT / "data"
CKPT_DIR = DATA_ROOT / "ckpt"
LOG_DIR = DATA_ROOT / "logs"
SANDBOX_DIR = DATA_ROOT / "sandbox"
CONFIG_DIR = CODE_ROOT / "configs"
PROMPT_DIR = CODE_ROOT / "prompts"
IDENTITY_DIR = CODE_ROOT / "identity"

# the pagoda project whose downloads and filtered pools Bonsai reuses (symlinked, never copied)
PAGODA_ROOT = Path(os.environ.get("PAGODA_HOME", CODE_ROOT.parent / "pagoda")).expanduser().resolve()


def ensure_dirs():
    for d in (RAW_DIR, FILTERED_DIR, TOK_DIR, TOKENIZER_DIR, TEACHER_DIR, DATA_DIR, CKPT_DIR, LOG_DIR, SANDBOX_DIR):
        d.mkdir(parents=True, exist_ok=True)


def log(msg, file=None):
    """Timestamped print that is also appended to `file` when given."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    if file is not None:  # the file first: stdout may be a pipe whose reader (tee) is already gone
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        with open(file, "a") as f:
            f.write(line + "\n")
    try:
        print(line, flush=True)
    except (BrokenPipeError, OSError):
        pass


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def iter_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def human(n):
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{n:.0f}"
        n /= 1000
    return f"{n:.1f}P"


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)
