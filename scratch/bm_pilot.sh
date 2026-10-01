#!/usr/bin/env bash
# THEORY §60 development pilot (pre-panel): BM-oc lambda_order in {0.1, 1}, seed 0 (not a panel seed), dev set only.
# Also the throughput pilot that fixes steps/batch. bash scratch/bm_pilot.sh <lam> <device>
cd "$(dirname "$0")/.."
python -u bm/run.py train --pool bm/data_pool --out bm/runs/pilot_oc_lam$1 --lam $1 --seed 0 --steps 10000 --bs 128 --device $2 > logs/bm_pilot_lam$1.log 2>&1
python -u bm/run.py eval --pool bm/data_pool --run bm/runs/pilot_oc_lam$1 --sets dev --device $2 >> logs/bm_pilot_lam$1.log 2>&1
echo "$(date +%T) PILOT_DONE lam $1 rc $?" >> logs/bm_pilot_lam$1.log
