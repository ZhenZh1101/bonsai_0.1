#!/usr/bin/env bash
# Progress, speed and ETA of the generation; the last lines of its log.
cd "$(dirname "$0")"
.venv/bin/python -m scripts.gen_remote status
echo; tail -n 3 logs/gen_offpolicy.log 2>/dev/null | cut -c1-200
ls -lh results/*.tar.gz 2>/dev/null
