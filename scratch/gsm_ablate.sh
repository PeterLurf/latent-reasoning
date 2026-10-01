#!/usr/bin/env bash
# GSM8K_REPRO amendment 2: latent-step ablations, sequential on one local GPU.  bash scratch/gsm_ablate.sh <device>
cd "$(dirname "$0")/.."
dev=$1
run() { python -u -m bench.run --task gsm8k --model "$1" --tag $2 --device $dev > logs/bench_gsm8k_$2.log 2>&1; grep -q WROTE logs/bench_gsm8k_$2.log && echo "$(date +%T) $2 done" >> logs/gsm_ablate.log || echo "$(date +%T) $2 FAILED" >> logs/gsm_ablate.log; }
for L in 0 1; do run "codi:codi?L=$L" codi_L$L; run "codi:simcot_codi?L=$L" simcot_codi_L$L; done
for K in 0 1 2; do run "coconut_gsm:prosqa/ckpt_ext/hf/simcot_coconut/checkpoint_28?K=$K" simcot_coconut_K$K; done
echo "$(date +%T) ABLATE_DONE" >> logs/gsm_ablate.log
