#!/usr/bin/env bash
# §216 R1-s2 (s216_r1_s2) remaining round-1 parts after the FT-DIVERGED stop (Theory 12:5x 10-18): the descriptive record (the cell reads FAILURE either way).
# Waits for the running selfdepth s2 r1 (pid $1), then on the 3070: steppos, lfk, rs, d3hist, hist24, vbhalt, chainlabel, chainlabel_fresh, d3hist gate.
#   nohup bash scratch/s216_r1s2_rest.sh <pid> >> logs/s216_s2_chain.out 2>&1 &
cd "$(dirname "$0")/.."
DEV=${DEV:-cuda:1}; n=314; r=s216_r1_s2; say() { echo "$(date '+%F %T') $*"; }
while powershell -NoProfile -Command "if (Get-Process -Id $1 -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 30; done
say "b$n $r selfdepth done: $(tail -n 1 logs/s216_selfdepth_$r.log | cut -c1-140)"
diag() { for P in "$@"; do python -m boxes.vbank_diag --run boxes/runs_fsq/$r --part $P --device $DEV --eval_bs 32 > logs/s216_${P}_$r.log 2>&1; rc=$?
  say "b$n $r $P rc $rc $(tail -n 1 logs/s216_${P}_$r.log | cut -c1-160)"; done; }
diag steppos lfk rs d3hist hist24
python -m boxes.vbank_halt --run boxes/runs_fsq/$r --device $DEV --eval_bs 32 > logs/s216_vbhalt_$r.log 2>&1; rc=$?; say "b$n $r vbhalt rc $rc $(tail -n 1 logs/s216_vbhalt_$r.log | cut -c1-160)"
diag chainlabel chainlabel_fresh
python -c "
import json; R = 'boxes/runs_fsq/$r'; v = json.load(open(R + '/eval_vb.json'))['acc_n2']; h = json.load(open(R + '/eval_d3hist.json'))['intact']['acc_n2']
print('GATE d3hist intact == eval_vb acc_n2 (n 1..12):', all(h[k] == v[k] for k in h))" 2>&1 | sed "s/^/$(date '+%F %T') b$n $r /"
say "§216 R1-s2 remaining parts done"
