#!/usr/bin/env bash
# note 213-37 CHAIN-LABEL-FRESH ($0): vbank_diag part chainlabel_fresh, D3-T on the 3070 (cuda:1) and D3-F on the 4060 (cuda:0) in parallel (no bitwise gate;
# the card is recorded per file), fp32, eval_bs 32 -> <cell>/eval_chainlabel_fresh.json.   nohup bash scratch/s213_chainfresh.sh >> logs/s213_chainfresh.out 2>&1 &
cd "$(dirname "$0")/.."
one() { python -m boxes.vbank_diag --run boxes/runs_fsq/$1 --part chainlabel_fresh --device $2 --eval_bs 32 > logs/s213_chainfresh_$1.log 2>&1; rc=$?
  echo "$(date '+%F %T') $1 chainlabel_fresh ($2) rc $rc $(tail -n 1 logs/s213_chainfresh_$1.log | cut -c1-140)"; }
one s213_d3t_s1 cuda:1 & one s213_d3f_s1 cuda:0 & wait
echo "$(date '+%F %T') §213 chainlabel_fresh done"
