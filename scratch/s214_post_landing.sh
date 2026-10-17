#!/usr/bin/env bash
# §214 post-landing evals (D4_v1 §6 at 10187c1b): local fp32, eval_bs 32, the 3070 (cuda:1; D3-T's comparators' card class), the §170 test set. After
# b309's drain (ALL_DRAINED + model.pt + log.jsonl) AND the §215 chain's last line (one job per card; no co-tenants): eval_vb -> steppos -> lfk -> rs ->
# d3hist -> vbhalt -> hist24 (DUP arms; G-P15) -> pause (eval_pause.json, the only pi-applying file), then the d3hist gate (intact == eval_vb acc_n2).
# rc is captured before any $(date). Scoring is Theory's (theory/score_s214.py).
#   nohup bash scratch/s214_post_landing.sh >> logs/s214_post_landing.out 2>&1 &
cd "$(dirname "$0")/.."
DEV=${DEV:-cuda:1}; n=309; r=s214_pausea_s1
say() { echo "$(date '+%F %T') $*"; }
diag() { for P in "$@"; do python -m boxes.vbank_diag --run boxes/runs_fsq/$r --part $P --device $DEV --eval_bs 32 > logs/s214_${P}_$r.log 2>&1; rc=$?
  say "b$n $r $P rc $rc $(tail -n 1 logs/s214_${P}_$r.log | cut -c1-160)"; done; }
until grep -q "ALL_DRAINED" logs/auto_drain_gen_b$n.out 2>/dev/null && [ -s boxes/runs_fsq/$r/model.pt ] && [ -s boxes/runs_fsq/$r/log.jsonl ]; do sleep 120; done
say "b$n $r drained"
until grep -q "§215 post-landing evals done" logs/s215_post_landing.out 2>/dev/null; do sleep 120; done
say "b$n $r -> evals on $DEV (the §215 chain is done)"
python -m boxes.vbank_eval --run boxes/runs_fsq/$r --device $DEV --eval_bs 32 > logs/s214_eval_vb_$r.log 2>&1; rc=$?
say "b$n $r eval_vb rc $rc $(tail -n 1 logs/s214_eval_vb_$r.log | cut -c1-160)"
diag steppos lfk rs d3hist
python -m boxes.vbank_halt --run boxes/runs_fsq/$r --device $DEV --eval_bs 32 > logs/s214_vbhalt_$r.log 2>&1; rc=$?
say "b$n $r vbhalt rc $rc $(tail -n 1 logs/s214_vbhalt_$r.log | cut -c1-160)"
diag hist24 pause
python -c "
import json; R = 'boxes/runs_fsq/$r'; v = json.load(open(R + '/eval_vb.json'))['acc_n2']; h = json.load(open(R + '/eval_d3hist.json'))['intact']['acc_n2']
print('GATE d3hist intact == eval_vb acc_n2 (n 1..12):', all(h[k] == v[k] for k in h))" 2>&1 | sed "s/^/$(date '+%F %T') b$n $r /"
say "§214 post-landing evals done"
