#!/usr/bin/env bash
# re-run S0-3 agerevert (arm key "mct" was overwritten by main()'s run-flag "mct"; now "age_mct") on the 3070 after s213_d4s0.sh's done-line (no co-tenant).
cd "$(dirname "$0")/.."
until grep -q "d4 S0-2/S0-3 done" logs/s213_d4s0.out; do sleep 30; done
R=boxes/runs_fsq/s213_d3t_s1
python -m boxes.vbank_diag --run $R --part agerevert --device cuda:1 --eval_bs 32 > logs/s213_agerevert_s213_d3t_s1.log 2>&1; rc=$?
echo "$(date '+%F %T') agerevert (re-run) rc $rc $(tail -n 1 logs/s213_agerevert_s213_d3t_s1.log | cut -c1-140)"
python -c "
import json; R = '$R'; d = json.load(open(R + '/eval_agerevert.json')); v = json.load(open(R + '/eval_vb.json'))['acc_n2']; o = json.load(open(R + '/eval_agerevert.json.void'))
print('GATE agerevert intact == eval_vb acc_n2:', all(d['intact']['acc_n2'][n] == v[n] for n in d['intact']['acc_n2']), '| revert == first run:', d['revert'] == o['revert'], '| age_mct present:', isinstance(d.get('age_mct'), dict))" 2>&1 | sed "s/^/$(date '+%F %T') /"
