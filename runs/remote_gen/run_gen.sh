#!/usr/bin/env bash
# Bonsai-0.1 off-policy teacher generation on a second Mac (plan sections 3, 11). One command:
#   ./run_gen.sh
# 1. checks the machine (Apple Silicon), installs uv if missing, builds the Python environment from uv.lock
# 2. picks the teacher and batch from the memory (16 GB+: Qwen3-1.7B bf16; 8 GB: Qwen3-1.7B-8bit, shipped in the
#    bundle; override with TEACHER=Qwen3-0.6B BATCH=16 MEM_GB=5 SHARD=0/1 ./run_gen.sh); a teacher that is not here
#    is downloaded at the pinned commit (an -8bit one: the bf16 model, quantized on this machine)
# 3. self-test: one full batch (BATCH/2 prompts x 2 modes) into data/selftest: speed, keep rate, projected time; once
# 4. starts the full run detached under caffeinate (closing the terminal is fine), then packs results/
# Re-running is always safe: it shows the progress if the job runs, else resumes where it stopped.
# Other commands: ./status.sh  ./stop.sh  ./pack_results.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs data results
exec > >(tee -a logs/run_gen.log) 2>&1
say() { echo "[$(date '+%F %T')] $*"; }
die() { say "ERROR: $*"; exit 1; }

[ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] || die "needs a Mac with Apple Silicon (MLX)"
[ "$(sw_vers -productVersion | cut -d. -f1)" -ge 14 ] || say "warning: macOS $(sw_vers -productVersion); MLX wheels need macOS 14 or newer"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  say "installing uv (Python package manager, https://docs.astral.sh/uv/)"
  if command -v brew >/dev/null 2>&1; then brew install uv; else curl -LsSf https://astral.sh/uv/install.sh | sh; fi
  command -v uv >/dev/null 2>&1 || die "uv not found after installing; open a new terminal and run ./run_gen.sh again"
fi
say "Python environment (uv.lock)"
uv sync --frozen --no-dev --quiet
PY=.venv/bin/python

eval "$($PY -m scripts.gen_remote plan)"
say "machine: ${RAM_GB} GB RAM, ${FREE_GB} GB free disk -> teacher $TEACHER, batch $BATCH, MLX memory limit ${MEM_GB} GB, shard $SHARD: $JOBS jobs"
[ "${FREE_GB%.*}" -ge 8 ] || die "need at least 8 GB free disk (teacher + outputs)"
if $PY -c "import sys; from scripts.gen_remote import alive_pid; sys.exit(0 if alive_pid() else 1)"; then
  say "the generation is already running"; $PY -m scripts.gen_remote status; exit 0
fi

if [ ! -f "teachers/$TEACHER/config.json" ]; then
  case "$TEACHER" in
    *-8bit)
      BASE="${TEACHER%-8bit}"
      say "$TEACHER is not in the bundle: downloading $BASE and quantizing it here (needs ~6 GB of memory once)"
      $PY -m scripts.download --only teachers --teachers "$BASE"
      $PY -m scripts.quantize_teacher --teacher "$BASE" --check 0
      rm -rf "teachers/$BASE" ;;
    *)
      say "downloading $TEACHER (pinned commit, tokenizer hashes checked)"
      $PY -m scripts.download --only teachers --teachers "$TEACHER" ;;
  esac
fi
$PY -c "from bonsai.teacher import check_tokenizer; check_tokenizer('teachers/$TEACHER')" || die "teachers/$TEACHER: tokenizer differs from tokenizer/manifest.json"

if [ ! -f data/selftest/PASSED ]; then
  rm -rf data/selftest
  SELFTEST_N="${SELFTEST_N:-$(( BATCH / 2 > 8 ? BATCH / 2 : 8 ))}"   # one full batch: its speed is the real run's
  say "self-test: $SELFTEST_N prompts x 2 modes (a few minutes)"
  $PY -m scripts.gen_offpolicy --teacher "Qwen/$TEACHER" --prompts data/prompts.jsonl --modes think,no_think \
    --budget 128 --shard 0/1 --limit "$SELFTEST_N" --batch "$BATCH" --memory-limit-gb "$MEM_GB" --out data/selftest
  $PY -m scripts.gen_remote selftest
  touch data/selftest/PASSED
fi
[ "${SELFTEST_ONLY:-0}" = "1" ] && { say "SELFTEST_ONLY=1: not starting the full run"; exit 0; }

$PY -m scripts.gen_remote launch
say "running in the background. Progress: ./status.sh   log: tail -f logs/gen_offpolicy.log   stop: ./stop.sh"
say "when it finishes, the results are packed into results/*.tar.gz: copy that file to the MacBook"
