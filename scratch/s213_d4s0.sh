#!/usr/bin/env bash
# D4_v1 §7 Stage 0 on D3-T ($0; Audit OK + diff review needed before starting): S0-3 agerevert then S0-2 carrier on DEV (default cuda:1 = the 3070, D3-T's
# eval_vb card -> bitwise intact gates), fp32, eval_bs 32 -> s213_d3t_s1/eval_agerevert.json, eval_carrier.json; gate: intact acc_n2 == eval_vb acc_n2.
#   nohup bash scratch/s213_d4s0.sh >> logs/s213_d4s0.out 2>&1 &
cd "$(dirname "$0")/.."
DEV=${DEV:-cuda:1}; R=boxes/runs_fsq/s213_d3t_s1
for P in agerevert carrier; do
  python -m boxes.vbank_diag --run $R --part $P --device $DEV --eval_bs 32 > logs/s213_${P}_s213_d3t_s1.log 2>&1; rc=$?
  echo "$(date '+%F %T') $P rc $rc $(tail -n 1 logs/s213_${P}_s213_d3t_s1.log | cut -c1-140)"
  python -c "
import json; R = '$R'; h = json.load(open(R + '/eval_$P.json'))['intact']['acc_n2']; v = json.load(open(R + '/eval_vb.json'))['acc_n2']
print('GATE $P intact == eval_vb acc_n2 (n %s..%s):' % (min(h, key=int), max(h, key=int)), all(h[n] == v[n] for n in h))" 2>&1 | sed "s/^/$(date '+%F %T') /"
done
echo "$(date '+%F %T') d4 S0-2/S0-3 done"
