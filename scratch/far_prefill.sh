#!/usr/bin/env bash
# Pre-fill the PROTOCOL7 T9/T10 P7 cache (results/far_sweep_layers_s2033.json; scratch/far_sweep_layers.py skips runs
# already measured) on an idle card. Output goes to a log that is not read before scoring.
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
python scratch/far_sweep_layers.py $(cat scratch/far_prefill_runs.txt) --out results/far_sweep_layers_s2033.json --device ${1:-cuda:2} > logs/far_prefill.log 2>&1
echo "$(date +%T) FAR_PREFILL_DONE rc $?" >> logs/far_prefill.log
