#!/usr/bin/env bash
# Day 5 (plan sections 11, 12, 14, 15, 18), after pretraining wrote ckpt/<run>/base.safetensors:
#   base evals -> off-policy SFT -> evals -> on-policy distillation -> evals -> identity mini-SFT -> evals,
#   teacher KL at every stage, identity acceptance, manual-test sheet, HF export.
# Plan 15 asks for three evaluations (base, after off-policy, after on-policy); the identity fix gets a fourth
# column (final). Needs data/offpolicy (runs/gen_offpolicy.sh); tok/offpolicy is (re)built when it is older.
# Times on the M4 Pro: SFT ~5 h (2 epochs x ~45M tokens), on-policy ~3 h, full evals ~50 min per stage
# (QUICK=1 caps every eval at 500 items, ~10 min).
# Restartable: every stage is skipped when its output is newer than its input, and the long ones resume
# (--resume auto), so after a crash or a closed terminal just run the same command again. Runs in tmux
# session 'bonsai-post' (NO_TMUX=1 to stay in the foreground). FORCE=1 skips the input checks.
# Dry run on the smoke checkpoints: RUN=bonsai-smoke FORCE=1 NO_TMUX=1 SFT_DATA=tok/smoke/offpolicy \
#   ID_DATA=tok/smoke/identity_sft OP_STEPS=2 OP_BATCH=8 ID_STEPS=5 EVAL_LIMIT=4 KL_N=4 bash runs/after_pretrain.sh
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="${RUN:-bonsai-0.1}"
if [ -z "${TMUX:-}" ] && [ "${NO_TMUX:-0}" != "1" ]; then
  tmux has-session -t bonsai-post 2>/dev/null && { echo "tmux session 'bonsai-post' exists: tmux attach -t bonsai-post"; exit 1; }
  mkdir -p logs
  tmux new-session -d -s bonsai-post "RUN=$RUN QUICK=${QUICK:-0} FORCE=${FORCE:-0} NO_TMUX=1 bash runs/after_pretrain.sh \
    2>&1 | tee -i -a logs/$RUN.after_pretrain.log; echo; echo 'after_pretrain exited'; read"
  echo "started in tmux session 'bonsai-post' (run=$RUN). Follow: tail -f logs/$RUN.after_pretrain.log"; exit 0
fi
PY="caffeinate -dims uv run python -m"
LIMIT=""; [ "${QUICK:-0}" = "1" ] && LIMIT="--limit 500"; [ -n "${EVAL_LIMIT:-}" ] && LIMIT="--limit $EVAL_LIMIT"
SFT_DATA="${SFT_DATA:-tok/offpolicy}"; ID_DATA="${ID_DATA:-tok/identity_sft}"
KL=""; [ -n "${KL_N:-}" ] && KL="--n $KL_N"
C="ckpt/$RUN"; L="logs/$RUN"
fail=0
newer() { [ -e "$1" ] && [ -e "$2" ] && [ "$1" -nt "$2" ]; }   # newer OUT IN: OUT exists and is newer than IN
step() { echo; echo "=== $(date '+%F %T') $*"; }

# inputs: a finished pretraining run and the teacher set
if [ "${FORCE:-0}" != "1" ]; then
  [ -f "$L/pretrain_done.json" ] && [ -f "$C/base.safetensors" ] || {
    echo "$C/base.safetensors / $L/pretrain_done.json missing: pretraining has not finished (FORCE=1 to post-train the newest checkpoint anyway)"; exit 1; }
  ls data/offpolicy/kept_*.jsonl >/dev/null 2>&1 || { echo "data/offpolicy/kept_*.jsonl missing: run runs/gen_offpolicy.sh first"; exit 1; }
fi
stale=0
[ -f "$SFT_DATA/train_manifest.json" ] || stale=1
for f in data/offpolicy/kept_*.jsonl; do [ -e "$f" ] && [ "$f" -nt "$SFT_DATA/train_manifest.json" ] && stale=1; done
if [ "$stale" = "1" ] && [ "$SFT_DATA" = "tok/offpolicy" ]; then
  step "build tok/offpolicy (data/offpolicy changed)"
  uv run python -c "import json; s=json.load(open('data/offpolicy/stats.json')); print('teacher samples kept:', s.get('kept'), 'by kind/mode:', s.get('kept_by'))" || true
  $PY scripts.build_sft || { echo "build_sft FAILED"; exit 1; }
fi
BASE="$C/base.safetensors"

evals() {  # evals SOURCE WEIGHTS
  newer "$L/eval_$1.json" "$2" || { step "eval $1"; $PY scripts.chat_eval --run "$RUN" --source "$1" $LIMIT || { echo "eval $1 FAILED"; fail=1; }; }
  newer "$L/teacher_kl_$1.json" "$2" || { step "teacher KL $1"; $PY scripts.teacher_kl --run "$RUN" --source "$1" $KL || { echo "teacher KL $1 FAILED"; fail=1; }; }
}

evals base "$BASE"
if ! { newer "$C/sft_final.safetensors" "$BASE" && newer "$C/sft_final.safetensors" "$SFT_DATA/train_manifest.json"; }; then
  step "off-policy SFT"; $PY scripts.sft --run "$RUN" --data "$SFT_DATA" --epochs 2 --resume auto || { echo "SFT FAILED"; exit 1; }
fi
evals sft "$C/sft_final.safetensors"
if ! newer "$L/onpolicy_summary.json" "$C/sft_final.safetensors"; then
  step "on-policy distillation"
  $PY scripts.distill_onpolicy --run "$RUN" --teacher Qwen/Qwen3-0.6B --steps "${OP_STEPS:-150}" --batch "${OP_BATCH:-256}" --resume auto || { echo "on-policy FAILED"; exit 1; }
fi
evals onpolicy "$C/onpolicy_final.safetensors"
if ! newer "$C/final.safetensors" "$C/onpolicy_final.safetensors"; then
  step "identity mini-SFT"; $PY scripts.identity_sft --run "$RUN" --data "$ID_DATA" --steps "${ID_STEPS:-150}" --resume auto || { echo "identity SFT FAILED"; exit 1; }
fi
evals final "$C/final.safetensors"
step "identity acceptance"
$PY scripts.identity_eval --run "$RUN" --source final || echo "identity acceptance FAILED (see $L/identity_eval_final.md)"
step "manual-test sheet"; $PY scripts.manual_test --run "$RUN" --source final || { echo "manual test sheet FAILED"; fail=1; }
step "HF export"; $PY scripts.export_hf --run "$RUN" --source final || { echo "HF export FAILED"; fail=1; }
echo "results: $L/eval_{base,sft,onpolicy,final}.md, teacher_kl.json, identity_eval_final.md, manual_tests_final.md, export/$RUN/"
exit $fail
