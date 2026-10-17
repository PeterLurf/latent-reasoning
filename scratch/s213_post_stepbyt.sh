#!/usr/bin/env bash
# note 213-18 STEP-BY-T ($0; Audit OK needed before starting): $1 = "now" -> N1 (s211_vb_carry_s1) on DEV1 (default cuda:2 = the 2070) immediately;
# otherwise the §213 cells on DEV (default cuda:1 = the 3070) after scratch/s213_post_halt.sh's "§213 vbhalt done" line (no co-tenant). Output
# <cell>/eval_stepbyt.json; gate: per-n sums over t == the cell's committed eval_lfk.json step block (events, strict_exact, voiddrop_exact).
#   nohup bash scratch/s213_post_stepbyt.sh now >> logs/s213_post_stepbyt.out 2>&1 &     nohup bash scratch/s213_post_stepbyt.sh >> logs/s213_post_stepbyt.out 2>&1 &
cd "$(dirname "$0")/.."
run() {   # cell device
  python -m boxes.vbank_diag --run boxes/runs_fsq/$1 --part stepbyt --device $2 --eval_bs 32 > logs/s213_stepbyt_$1.log 2>&1; rc=$?
  echo "$(date '+%F %T') $1 stepbyt rc $rc $(tail -n 1 logs/s213_stepbyt_$1.log | cut -c1-140)"
  python -c "
import json; R = 'boxes/runs_fsq/$1'; s = json.load(open(R + '/eval_stepbyt.json'))['step']; l = json.load(open(R + '/eval_lfk.json'))['step']
print('GATE stepbyt sums == eval_lfk step:', all(s[n] == {k: l[n][k] for k in s[n]} for n in l) and set(s) == set(l))" 2>&1 | sed "s/^/$(date '+%F %T') $1 /"
}
if [ "${1:-}" = "now" ]; then run s211_vb_carry_s1 ${DEV1:-cuda:2}; exit 0; fi
until grep -q "§213 vbhalt done" logs/s213_post_halt.out 2>/dev/null; do sleep 300; done
for r in s213_d3t_s1 s213_d3f_s1; do run $r ${DEV:-cuda:1}; done
echo "$(date '+%F %T') §213 stepbyt done"
