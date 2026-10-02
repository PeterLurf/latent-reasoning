#!/usr/bin/env bash
# GSM8K_REPRO amendment 3: budget curves. bash scratch/gsm_budget.sh <device> <coconut|codi>
cd "$(dirname "$0")/.."
dev=$1; which=$2
run() { [ -f bench/results/gsm8k__$2.json ] && return; python -u -m bench.run --task gsm8k --model "$1" --tag $2 --device $dev > logs/bench_gsm8k_$2.log 2>&1; grep -q WROTE logs/bench_gsm8k_$2.log && echo "$(date +%T) $2 done" >> logs/gsm_budget.log || echo "$(date +%T) $2 FAILED" >> logs/gsm_budget.log; }
if [ $which = coconut ]; then for K in 4 6 8 12 16; do run "coconut_gsm:prosqa/ckpt_ext/hf/simcot_coconut/checkpoint_28?K=$K" simcot_coconut_K$K; done
else for L in 2 4 8 12; do run "codi:codi?L=$L" codi_L$L; run "codi:simcot_codi?L=$L" simcot_codi_L$L; done; fi
echo "$(date +%T) BUDGET_DONE $which" >> logs/gsm_budget.log
