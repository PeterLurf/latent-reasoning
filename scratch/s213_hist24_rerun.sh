#!/usr/bin/env bash
# 213-21 / 213-22: hist24 re-run on N1 (2070) / N2 (4060) with dup_newest + dup_foreign; the earlier arms must reproduce ae8d67b6 bitwise.
cd "$(dirname "$0")/.."
one() {
  python -m boxes.vbank_diag --run boxes/runs_fsq/$1 --part hist24 --device $2 --eval_bs 32 > logs/s213_hist24b_$1.log 2>&1; rc=$?
  echo "$(date '+%F %T') $1 hist24 (dup) rc $rc $(tail -n 1 logs/s213_hist24b_$1.log | cut -c1-120)"
  python -c "
import json; c = '$1'; new = json.load(open('boxes/runs_fsq/%s/eval_hist24.json' % c)); old = json.load(open('scratch/_tmp/hist24_ae8d_%s.json' % c)); v = json.load(open('boxes/runs_fsq/%s/eval_vb.json' % c))
arms = [a for a in old if a not in ('def', 'omitted')]
print('GATE earlier arms == ae8d67b6:', all(new[a] == old[a] for a in arms), arms, '| intact == eval_vb:', all(new['intact']['acc_n2'][n] == v['acc_n2'][n] and new['intact']['acc_k24'][n] == v['acc_k']['24'][n] for n in new['intact']['acc_n2']))" 2>&1 | sed "s/^/$(date '+%F %T') $1 /"
}
one s211_vb_carry_s1 cuda:2 & one s211_vb_pa_s1 cuda:0 & wait
