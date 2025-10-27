#!/usr/bin/env bash
# Pack data/offpolicy (kept / rejected samples, runs, stats) into results/*.tar.gz for the MacBook. Runs by itself
# at the end of the generation; run it by hand any time to take partial results.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
TAG=$(.venv/bin/python -c "import json; p=json.load(open('data/remote_plan.json')); i,n=p['shard'].split('/'); print(f\"{p['teacher']}-s{i}of{n}\")")
HOST=$(scutil --get LocalHostName 2>/dev/null || hostname -s)
OUT="results/bonsai-offpolicy-$HOST-$TAG-$(date +%Y%m%d-%H%M).tar.gz"
.venv/bin/python -c "from bonsai.teacher import summarize; summarize('data/offpolicy')"
COPYFILE_DISABLE=1 tar czf "$OUT" BUNDLE.json data/remote_plan.json data/offpolicy logs/gen_offpolicy.log
echo "packed $(du -h "$OUT" | cut -f1) -> $OUT"
echo "on the MacBook: python -m scripts.merge_offpolicy $OUT"
