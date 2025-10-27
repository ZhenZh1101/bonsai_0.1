#!/usr/bin/env bash
# S0 keep-or-drop check (plan section 9): Bonsai-0.1-probe (12 x 256), 100M tokens with S0 (5%) vs without, same
# val set, the main run's schedule scaled to the probe's length; S0 stays only if it is >= 0.02 nat better.
# 1-2 h per arm on the M4 Pro (more on the M2). Result: logs/s0_ablation.json, read by runs/train_5d.sh.
# Resumable (re-run the same command).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
caffeinate -dims uv run python -m scripts.s0_ablation "$@" 2>&1 | tee -a logs/s0_ablation.stdout.log
