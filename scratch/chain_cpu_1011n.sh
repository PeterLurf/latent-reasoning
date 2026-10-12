#!/usr/bin/env bash
# CPU eval chain (2026-10-11 night; Theory 187-5a / 187-3 / 187-4), Idle, 3 threads, one job at a time; replaces chain_posoffset_cpu.sh /
# chain_s184s2_cpu.sh / chain_s184s2_selfdepth.sh (stopped: the all-shift 187-5 design was amended to KEEP-SINK before any output was read).
#   1. 187-5a keep0 posoffset on s184_reweight_s1, then s182_nofade_s1  -> eval_s187_posoffset_keep0.json
#   2. s184_reweight_s2: 187-3 rows on all n 9..12 -> eval_s187_deepaux_items_n9_12.json; 187-5a keep0 posoffset; 187-4 selfdepth
#   nohup bash scratch/chain_cpu_1011n.sh > /dev/null 2>&1 &      progress: logs/cpu_chain_1011n.done
cd "$(dirname "$0")/.."
run() { local tag=$1; shift; OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 nice -n 19 python -u -m boxes.halt173 "$@" --bs 125 --device cpu > "logs/cpu_$tag.log" 2>&1; echo "$(date +%T) $tag rc $?" >> logs/cpu_chain_1011n.done; }
run keep0_s184s1 --run boxes/runs_fsq/s184_reweight_s1 --posoffset_keep0
run keep0_nofade --run boxes/runs_fsq/s182_nofade_s1 --posoffset_keep0
run items_n9_12_s184s2 --run boxes/runs_fsq/s184_reweight_s2 --deepaux --deepaux_items_only --items_n 9,10,11,12 --items_tmax 14 --items_file eval_s187_deepaux_items_n9_12.json
run keep0_s184s2 --run boxes/runs_fsq/s184_reweight_s2 --posoffset_keep0
run selfdepth_s184s2 --run boxes/runs_fsq/s184_reweight_s2 --selfdepth
