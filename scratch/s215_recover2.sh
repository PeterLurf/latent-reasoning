#!/usr/bin/env bash
# §215 post-landing, recovery 2 after the 3070 was lost ~22:56 10-17 (vbhalt hung at STALL n 5; partial log + json kept as .void_gpulost); see also
# logs/s215_rs_s215_ftctrl_s1.log.void_reboot). b310 is complete (668cd6e8); b311 has eval_vb / steppos / lfk. Same contract (3070 cuda:1, fp32, eval_bs 32):
# b311 rs -> d3hist -> hist24 -> vbhalt -> chainlabel -> chainlabel_fresh -> selfdepth -> d3hist gate, then the done-line scratch/s214_post_landing.sh waits for.
#   nohup bash scratch/s215_recover.sh >> logs/s215_post_landing.out 2>&1 &
cd "$(dirname "$0")/.."
DEV=${DEV:-cuda:1}; n=311; r=s215_ftctrl_s1
say() { echo "$(date '+%F %T') $*"; }
diag() { for P in "$@"; do python -m boxes.vbank_diag --run boxes/runs_fsq/$r --part $P --device $DEV --eval_bs 32 > logs/s215_${P}_$r.log 2>&1; rc=$?
  say "b$n $r $P rc $rc $(tail -n 1 logs/s215_${P}_$r.log | cut -c1-160)"; done; }
say "b$n $r recovery 2 after the 3070 loss (vbhalt voided) -> remaining parts on $DEV"
# rs, d3hist, hist24 done before the GPU loss
python -m boxes.vbank_halt --run boxes/runs_fsq/$r --device $DEV --eval_bs 32 > logs/s215_vbhalt_$r.log 2>&1; rc=$?
say "b$n $r vbhalt rc $rc $(tail -n 1 logs/s215_vbhalt_$r.log | cut -c1-160)"
diag chainlabel chainlabel_fresh selfdepth
python -c "
import json; R = 'boxes/runs_fsq/$r'; v = json.load(open(R + '/eval_vb.json'))['acc_n2']; h = json.load(open(R + '/eval_d3hist.json'))['intact']['acc_n2']
print('GATE d3hist intact == eval_vb acc_n2 (n 1..12):', all(h[k] == v[k] for k in h))" 2>&1 | sed "s/^/$(date '+%F %T') b$n $r /"
say "§215 post-landing evals done"
