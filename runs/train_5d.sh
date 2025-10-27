#!/usr/bin/env bash
# 5-day run: Bonsai-0.1, 14 x 640 (121.1M; 79.2M non-embedding), 2.0B tokens (decision of 2026-10-26; the plan
# said 1.41B). The S0 ablation said drop (logs/s0_ablation.json), so: S1 1.487B -> S2 513M, 15,259 steps of 2^17
# tokens. Day-0 benchmark (logs/bench.json): 14 x 640 runs at 6,799 tok/s with micro-batch 16, so pretraining takes
# ~82 h of step time, ~85-96 h with evals / checkpoints / thermals (the benchmark's 0.85 factor gives 96 h).
# The 3-day run (train_3d.sh) is Bonsai-0.1-mini, 12 x 576, 1.0B tokens (~37 h at 8,987 tok/s).
#   * MODEL / TOKENS: the above; MICRO: the fastest micro-batch logs/bench.json measured for MODEL (else 16).
#     USE_BENCH=1 takes the benchmark's recommendation instead. The shards hold S1 1.52B / S2 525M (built for
#     2.0B without S0); a bigger budget makes them repeat, which scripts.train refuses (EXTRA=--allow-repeat, or
#     raise configs/s*.yaml total_tokens and rebuild).
#   * S0 keep / drop (plan section 9) comes from logs/s0_ablation.json (runs/s0_ablation.sh): keep_s0 false adds
#     --s0-frac 0. Without the file the run refuses to start: S0 is decided before step 0, it cannot be undone
#     later. Override with S0=keep or S0=drop.
#   * Run it with the GPU to itself (teacher generation on the same GPU would slow it down).
# Starts inside a detached tmux session under caffeinate. Re-running resumes from the newest checkpoint
# (exact data position of every phase, LR continues, no re-warmup).
#   attach:  tmux attach -t bonsai      detach: Ctrl-b d
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${RUN:-bonsai-0.1}"
BUDGET="${BUDGET:-96h}"
EXTRA="${EXTRA:-}"
DEF_MODEL=bonsai-0.1; DEF_TOKENS=2.0e9
[ "$BUDGET" = "50h" ] && { DEF_MODEL=bonsai-0.1-mini; DEF_TOKENS=1.0e9; }
if [ "${USE_BENCH:-0}" = "1" ]; then
  read -r B_MODEL B_MICRO B_TOKENS < <(uv run python - "$BUDGET" <<'PY'
import json, sys
r = json.load(open("logs/bench.json"))["summary"][sys.argv[1]]["recommended"]
print(r["config"], r["micro_batch"], r["tokens_flag"])
PY
)
  MODEL="${MODEL:-$B_MODEL}"; MICRO="${MICRO:-$B_MICRO}"; TOKENS="${TOKENS:-$B_TOKENS}"
fi
MODEL="${MODEL:-$DEF_MODEL}"
TOKENS="${TOKENS:-$DEF_TOKENS}"
if [ -z "${MICRO:-}" ]; then   # fastest measured micro-batch for this model (same gradients, only speed differs)
  MICRO=$(uv run python - "$MODEL" <<'PY' 2>/dev/null || echo 16
import json, sys
rs = [r for r in json.load(open("logs/bench.json"))["results"] if r.get("config") == sys.argv[1] and r.get("tok_per_s")]
print(max(rs, key=lambda r: r["tok_per_s"])["micro_batch"] if rs else 16)
PY
)
fi

S0="${S0:-}"
if [ -z "$S0" ]; then
  if [ -f logs/s0_ablation.json ]; then
    S0=$(uv run python -c "import json; print('keep' if json.load(open('logs/s0_ablation.json'))['keep_s0'] else 'drop')")
    echo "S0 decision from logs/s0_ablation.json: $S0"
  elif [ -n "$(ls -d ckpt/$RUN/step_* 2>/dev/null)" ]; then
    S0=resume   # an existing run keeps the plan it was started with
  else
    echo "logs/s0_ablation.json missing: run runs/s0_ablation.sh first (plan section 9), or set S0=keep / S0=drop"; exit 1
  fi
fi
[ "$S0" = "drop" ] && EXTRA="--s0-frac 0 $EXTRA"

CMD="caffeinate -dims uv run python -m scripts.train --model $MODEL --micro-batch $MICRO --tokens $TOKENS --run $RUN \
  --resume auto $EXTRA 2>&1 | tee -i -a logs/$RUN.stdout.log"
mkdir -p logs
if tmux has-session -t bonsai 2>/dev/null; then
  echo "tmux session 'bonsai' already exists: tmux attach -t bonsai"; exit 1
fi
tmux new-session -d -s bonsai "$CMD; echo; echo 'training exited'; read"
echo "started in tmux session 'bonsai' (run=$RUN model=$MODEL micro=$MICRO tokens=$TOKENS S0=$S0). Follow: tail -f logs/$RUN/train.log"
