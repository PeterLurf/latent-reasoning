#!/usr/bin/env bash
# Theory 2026-10-11 (§189): after chain_cpu_1011n.sh (5 lines in logs/cpu_chain_1011n.done), the 187-3 per-item rows on all n 9..12 test items for
# s185_rms_s1 -> eval_s187_deepaux_items_n9_12.json. CPU, Idle, 3 threads.   nohup bash scratch/chain_cpu_1011n2.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
until [ "$(grep -c rc logs/cpu_chain_1011n.done 2>/dev/null || echo 0)" -ge 5 ]; do sleep 60; done
OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 nice -n 19 python -u -m boxes.halt173 --run boxes/runs_fsq/s185_rms_s1 --deepaux --deepaux_items_only --items_n 9,10,11,12 --items_tmax 14 --items_file eval_s187_deepaux_items_n9_12.json --bs 125 --device cpu > logs/cpu_items_n9_12_rms_s1.log 2>&1
echo "$(date +%T) items_n9_12_rms_s1 rc $?" >> logs/cpu_chain_1011n.done
