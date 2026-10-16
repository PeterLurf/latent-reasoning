#!/usr/bin/env bash
# VB-2G Stage 0 chain v4 (Theory 13:4x): note 212-11 d3diag first (N2 4060, N3 3070, N1 2070 as the wpe reference), then the 211-15 A descriptives (a15).
#   nohup bash scratch/s211_stage0_chain4.sh >> logs/s211_stage0.out 2>&1 &
cd "$(dirname "$0")/.."
cell() {
  local c=$1 d=$2 R=boxes/runs_fsq/s211_vb_$1
  for p in d3diag a15; do
    python -m boxes.vbank_diag --run $R --part $p --device $d > logs/s211_${p}_$c.log 2>&1; echo "$(date '+%F %T') $c $p rc $? $(tail -n 1 logs/s211_${p}_$c.log | cut -c1-100)"
  done
}
cell pa_s1 cuda:0 &
cell pamct_s1 cuda:1 &
cell carry_s1 cuda:2 &
wait; echo "$(date '+%F %T') stage0 chain4 done"
