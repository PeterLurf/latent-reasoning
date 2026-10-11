#!/usr/bin/env bash
# note 187-5 (Theory 2026-10-11): after the 187-4 self-depth CPU run (logs/cpu_selfdepth.done), the position-offset eval on CPU, Idle, 3 threads:
# s184_reweight_s1 then s182_nofade_s1 -> <run>/eval_s187_posoffset.json; one line per run in logs/cpu_posoffset.done.
#   nohup bash scratch/chain_posoffset_cpu.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
until [ -s logs/cpu_selfdepth.done ]; do sleep 60; done
for r in s184_reweight_s1 s182_nofade_s1; do
  OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 nice -n 19 python -u -m boxes.halt173 --run "boxes/runs_fsq/$r" --posoffset --bs 125 --device cpu > "logs/cpu_posoffset_$r.log" 2>&1
  echo "$(date +%T) $r rc $?" >> logs/cpu_posoffset.done
done
