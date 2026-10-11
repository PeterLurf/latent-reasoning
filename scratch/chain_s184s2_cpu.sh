#!/usr/bin/env bash
# Theory 2026-10-11: after the CPU chain (187-4 self-depth, 187-5 on s184 s1 / NOFADE: logs/cpu_posoffset.done has 2 lines), on s184_reweight_s2:
# (1) 187-3 per-item rows on ALL n 9..12 test items -> eval_s187_deepaux_items_n9_12.json; (2) 187-5 posoffset -> eval_s187_posoffset.json. CPU, Idle, 3 threads.
#   nohup bash scratch/chain_s184s2_cpu.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
until [ "$(grep -c rc logs/cpu_posoffset.done 2>/dev/null || echo 0)" -ge 2 ]; do sleep 60; done
R=boxes/runs_fsq/s184_reweight_s2
OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 nice -n 19 python -u -m boxes.halt173 --run "$R" --deepaux --deepaux_items_only --items_n 9,10,11,12 --items_tmax 14 --items_file eval_s187_deepaux_items_n9_12.json --bs 125 --device cpu > logs/cpu_items_n9_12_s184s2.log 2>&1
echo "$(date +%T) items rc $?" >> logs/cpu_s184s2.done
OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 nice -n 19 python -u -m boxes.halt173 --run "$R" --posoffset --bs 125 --device cpu > logs/cpu_posoffset_s184s2.log 2>&1
echo "$(date +%T) posoffset rc $?" >> logs/cpu_s184s2.done
