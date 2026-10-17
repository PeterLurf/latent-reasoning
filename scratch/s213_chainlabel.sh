#!/usr/bin/env bash
# note 213-33 / 213-34 CHAIN-LABEL ($0): vbank_diag part chainlabel on D3-T then D3-F, the 3070 (cuda:1), fp32, eval_bs 32 -> <cell>/eval_chainlabel.json.
#   nohup bash scratch/s213_chainlabel.sh >> logs/s213_chainlabel.out 2>&1 &
cd "$(dirname "$0")/.."
for c in s213_d3t_s1 s213_d3f_s1; do
  python -m boxes.vbank_diag --run boxes/runs_fsq/$c --part chainlabel --device cuda:1 --eval_bs 32 > logs/s213_chainlabel_$c.log 2>&1; rc=$?
  echo "$(date '+%F %T') $c chainlabel rc $rc $(tail -n 1 logs/s213_chainlabel_$c.log | cut -c1-140)"
done
echo "$(date '+%F %T') §213 chainlabel done"
