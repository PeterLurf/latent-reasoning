#!/usr/bin/env bash
# VB-2G Stage 0 chain v3 (Theory 13:0x: GATHER-SPLICE, note 212-6, ahead of the remaining parts) for the PA cells: wait for the cell's running CONTENT-HALT
# (pid) to exit, then gsplice -> pf -> x16 -> foreign on the same card. N1 (no gather) keeps chain2's order.  args: run dev halt_pid (one cell per call)
cd "$(dirname "$0")/.."
c=$1; d=$2; hp=$3; R=boxes/runs_fsq/s211_vb_$c
while kill -0 $hp 2>/dev/null; do sleep 30; done; echo "$(date '+%F %T') $c halt finished (pid $hp): $(tail -n 1 logs/s211_halt_$c.log | cut -c1-100)"
for p in gsplice pf x16 foreign; do
  python -m boxes.vbank_diag --run $R --part $p --device $d > logs/s211_${p}_$c.log 2>&1; echo "$(date '+%F %T') $c $p rc $? $(tail -n 1 logs/s211_${p}_$c.log | cut -c1-100)"
done
