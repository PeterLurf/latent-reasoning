#!/usr/bin/env bash
# Theory 2026-10-11: after chain_s184s2_cpu.sh (logs/cpu_s184s2.done has 2 lines), the 187-4 selfdepth read on s184_reweight_s2 (n 9 / 10) on CPU,
# Idle, 3 threads -> s184_reweight_s2/eval_s187_selfdepth_pre.json; a line in logs/cpu_s184s2_selfdepth.done.
#   nohup bash scratch/chain_s184s2_selfdepth.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
until [ "$(grep -c rc logs/cpu_s184s2.done 2>/dev/null || echo 0)" -ge 2 ]; do sleep 60; done
OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 nice -n 19 python -u -m boxes.halt173 --run boxes/runs_fsq/s184_reweight_s2 --selfdepth --bs 125 --device cpu > logs/cpu_selfdepth_s184s2.log 2>&1
echo "$(date +%T) selfdepth s2 rc $?" >> logs/cpu_s184s2_selfdepth.done
