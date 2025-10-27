#!/usr/bin/env bash
# 3-day run: Bonsai-0.1-mini, 12 x 576 (89.5M; 51.8M non-embedding), plan section 4: 1.0B tokens
# (S0 50M -> S1 700M -> S2 250M), ~37 h at the measured 8,987 tok/s; USE_BENCH=1 takes summary["50h"]["recommended"];
# same S0 rule, tmux and resume behaviour as train_5d.sh.
set -euo pipefail
cd "$(dirname "$0")"
RUN="${RUN:-bonsai-0.1-mini}" BUDGET=50h ./train_5d.sh
