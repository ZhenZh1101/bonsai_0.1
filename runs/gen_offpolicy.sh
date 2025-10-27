#!/usr/bin/env bash
# Off-policy distillation set (plan sections 11, 14): Qwen3 teachers answer data/prompts.jsonl in both modes.
# Measured on the M4 Pro: Qwen3-1.7B ~16 h for 100k prompts x 2 modes (the plan estimated 5-6 h), Qwen3-0.6B
# ~6.5 h. Default: the 1.7B takes shard 0 of 2 here; run shard 1 with the 0.6B on the M2 Mac mini or later:
#   SHARD=1/2 TEACHER=Qwen/Qwen3-0.6B runs/gen_offpolicy.sh        (or SHARD=0/1 for everything with one teacher)
# On a second Mac instead (one command there): python -m scripts.make_gen_bundle -> copy dist/bonsai-gen-*.tar.gz,
# `tar xzf ... && cd bonsai-gen && ./run_gen.sh`, then bring results/*.tar.gz back with scripts.merge_offpolicy
# (runs/remote_gen/README.md).
# Not on the same GPU as pretraining (it would slow the benchmarked run). Resumable: re-running skips finished
# prompt ids. When every shard you want is done: uv run python -m scripts.build_sft (after_pretrain.sh rebuilds
# tok/offpolicy automatically when data/offpolicy changed).
set -euo pipefail
cd "$(dirname "$0")/.."
TEACHER="${TEACHER:-Qwen/Qwen3-1.7B}"
SHARD="${SHARD:-0/2}"
SESSION="${SESSION:-bonsai-gen}"
CMD="caffeinate -dims uv run python -m scripts.gen_offpolicy --teacher $TEACHER --prompts data/prompts.jsonl \
  --modes think,no_think --budget 128 --shard $SHARD 2>&1 | tee -i -a logs/gen_offpolicy.stdout.log"
mkdir -p logs
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "tmux session '$SESSION' exists"; exit 1; fi
tmux new-session -d -s "$SESSION" "$CMD; echo; echo 'generation exited'; read"
echo "started in tmux session '$SESSION' ($TEACHER, shard $SHARD). Progress: data/offpolicy/stats.json"
