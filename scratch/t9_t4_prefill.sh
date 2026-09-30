#!/usr/bin/env bash
# PROTOCOL7 T9/T10 secondary T4: pre-fill the end-activity / in-distribution trace cache (results/p6_endact.json) for
# the 34 competent T9/T10 models (measurement only; not read before the T4 layer mapping is registered).
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
python -m latent.protocol6 measure $(cat scratch/t9_causal_runs.txt) --device ${1:-cuda:2} > logs/t9_t4_prefill.log 2>&1
echo "$(date +%T) T4_PREFILL_DONE rc $?" >> logs/t9_t4_prefill.log
