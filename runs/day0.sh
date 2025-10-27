#!/usr/bin/env bash
# Day 0 on the MacBook (plan section 14), before training:
#   unit tests -> data acceptance (plan section 6) -> end-to-end smoke test -> throughput benchmark (~1 h GPU)
# Then, in this order (plan sections 9, 11, 14):
#   1. runs/s0_ablation.sh   S0 keep / drop (2-4 h on the M4 Pro; train_5d.sh refuses to start without its verdict)
#   2. runs/gen_offpolicy.sh teacher generation: ~16 h with Qwen3-1.7B alone (the plan assumed 5-6 h). It only has
#      to be done by day 5, but it must not share the GPU with training: run it before training, on the M2 Mac
#      mini (SHARD=1/2 TEACHER=Qwen/Qwen3-0.6B), or after pretraining.
#   3. runs/train_5d.sh (or train_3d.sh) with the GPU to itself: the plan's model and tokens (decided after this
#      benchmark), the fastest measured micro-batch; USE_BENCH=1 for the benchmark's recommendation instead.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="caffeinate -dims uv run python -m"
uv run python -m pytest -q
$PY scripts.verify_data --max-oov 0.004        # measured 0.29% on the full shards at N = 65,536 (user's choice)
$PY scripts.smoke_test
$PY scripts.bench                              # -> logs/bench.json
uv run python -c "
import json
s = json.load(open('logs/bench.json'))['summary']
for b in ('50h', '96h'):
    r = s[b]['recommended']
    print(b, '->', r['command'] if r else 'no recommendation', '|', r.get('decided_by') if r else '', '|', r.get('note', '') if r else '')"
