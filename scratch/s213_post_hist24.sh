#!/usr/bin/env bash
# note 213-20 HIST-AT-K24 ($0; Audit OK needed before starting): $1 = "now" -> N1 (s211_vb_carry_s1, the 2070) and N2 (s211_vb_pa_s1, the 4060), each on its
# eval_vb card, in parallel; otherwise the §213 cells on DEV (default cuda:1 = the 3070) after scratch/s213_post_stepbyt.sh's "§213 stepbyt done" line.
# Output <cell>/eval_hist24.json; gate: intact acc_n2 / acc_k24 == eval_vb.json acc_n2 / acc_k['24'] (n 3..8), bitwise.
#   nohup bash scratch/s213_post_hist24.sh now >> logs/s213_post_hist24.out 2>&1 &     nohup bash scratch/s213_post_hist24.sh >> logs/s213_post_hist24.out 2>&1 &
cd "$(dirname "$0")/.."
run() {   # cell device
  python -m boxes.vbank_diag --run boxes/runs_fsq/$1 --part hist24 --device $2 --eval_bs 32 > logs/s213_hist24_$1.log 2>&1; rc=$?
  echo "$(date '+%F %T') $1 hist24 rc $rc $(tail -n 1 logs/s213_hist24_$1.log | cut -c1-140)"
  python -c "
import json; R = 'boxes/runs_fsq/$1'; h = json.load(open(R + '/eval_hist24.json'))['intact']; v = json.load(open(R + '/eval_vb.json'))
print('GATE hist24 intact == eval_vb (acc_n2, acc_k24; n 3..8):', all(h['acc_n2'][n] == v['acc_n2'][n] and h['acc_k24'][n] == v['acc_k']['24'][n] for n in h['acc_n2']))" 2>&1 | sed "s/^/$(date '+%F %T') $1 /"
}
if [ "${1:-}" = "now" ]; then run s211_vb_carry_s1 cuda:2 & run s211_vb_pa_s1 cuda:0 & wait; exit 0; fi
until grep -q "§213 stepbyt done" logs/s213_post_stepbyt.out 2>/dev/null; do sleep 300; done
for r in s213_d3t_s1 s213_d3f_s1; do run $r ${DEV:-cuda:1}; done
echo "$(date '+%F %T') §213 hist24 done"
