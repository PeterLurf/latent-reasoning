#!/usr/bin/env bash
# §215 post-landing evals (SELFDEPTH_v1 §4.1 at 5ded8703): local fp32, eval_bs 32, the 3070 only (cuda:1), cells in landing order b310 FT-SELF -> b311
# FT-CTRL. Per cell, after its drain (ALL_DRAINED + model.pt + log.jsonl): eval_vb (acc, probe, vswap, stall) -> steppos -> lfk -> rs -> d3hist ->
# hist24 (DUP arms) -> vbhalt -> chainlabel -> chainlabel_fresh -> selfdepth (FT-CTRL reads FT-SELF's final pool_cursor), then the d3hist gate
# (intact == eval_vb acc_n2). rc is captured before any $(date) (memory: rc after $(date) is date's). Scoring is Theory's (theory/score_s215.py).
#   nohup bash scratch/s215_post_landing.sh >> logs/s215_post_landing.out 2>&1 &
cd "$(dirname "$0")/.."
DEV=${DEV:-cuda:1}
say() { echo "$(date '+%F %T') $*"; }
gate() { python -c "
import json; R = 'boxes/runs_fsq/$1'; v = json.load(open(R + '/eval_vb.json'))['acc_n2']; h = json.load(open(R + '/eval_d3hist.json'))['intact']['acc_n2']
print('GATE d3hist intact == eval_vb acc_n2 (n 1..12):', all(h[k] == v[k] for k in h))" 2>&1 | sed "s/^/$(date '+%F %T') b$n $1 /"; }
diag() { for P in "$@"; do python -m boxes.vbank_diag --run boxes/runs_fsq/$r --part $P --device $DEV --eval_bs 32 > logs/s215_${P}_$r.log 2>&1; rc=$?
  say "b$n $r $P rc $rc $(tail -n 1 logs/s215_${P}_$r.log | cut -c1-160)"; done; }
for cell in 310:s215_ftself_s1 311:s215_ftctrl_s1; do
  n=${cell%%:*}; r=${cell#*:}
  until grep -q "ALL_DRAINED" logs/auto_drain_gen_b$n.out 2>/dev/null && [ -s boxes/runs_fsq/$r/model.pt ] && [ -s boxes/runs_fsq/$r/log.jsonl ]; do sleep 120; done
  say "b$n $r drained -> evals on $DEV"
  python -m boxes.vbank_eval --run boxes/runs_fsq/$r --device $DEV --eval_bs 32 > logs/s215_eval_vb_$r.log 2>&1; rc=$?
  say "b$n $r eval_vb rc $rc $(tail -n 1 logs/s215_eval_vb_$r.log | cut -c1-160)"
  diag steppos lfk rs d3hist hist24
  python -m boxes.vbank_halt --run boxes/runs_fsq/$r --device $DEV --eval_bs 32 > logs/s215_vbhalt_$r.log 2>&1; rc=$?
  say "b$n $r vbhalt rc $rc $(tail -n 1 logs/s215_vbhalt_$r.log | cut -c1-160)"
  diag chainlabel chainlabel_fresh selfdepth
  gate $r
done
say "§215 post-landing evals done"
