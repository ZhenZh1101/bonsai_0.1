#!/usr/bin/env bash
# Stop the generation (nothing is lost: ./run_gen.sh resumes where it stopped).
cd "$(dirname "$0")"
.venv/bin/python -m scripts.gen_remote stop
