#!/usr/bin/env bash
# PROTOCOL7 amendment 10: literal (first-block) frontier_causal2 measurement of the competent T9/T10 models.
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
python -m latent.frontier_causal2 $(cat scratch/t9_causal_runs.txt) --out results/p7s_causal2_t9.json --device ${1:-cuda:2} > logs/t9_causal.log 2>&1
echo "$(date +%T) T9_CAUSAL_DONE rc $?" >> logs/t9_causal.log
