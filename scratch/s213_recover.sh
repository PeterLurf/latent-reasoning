#!/usr/bin/env bash
# §213 post-landing RECOVERY after the 23:37 10-16 local reboot (all four chain scripts died; D3-F rs was killed ~4 min in, log kept as .void).
# One card (cuda:1 = the 3070), fp32, eval_bs 32, sequential, rc captured before the date substitution:
#   D3-F rs -> d3hist -> d3hist gate -> "§213 post-landing evals done" (logs/s213_post_landing.out)
#   vbhalt D3-T, D3-F -> "§213 vbhalt done" (logs/s213_post_halt.out)
#   stepbyt D3-T, D3-F + sum gate vs eval_lfk -> "§213 stepbyt done" (logs/s213_post_stepbyt.out)
#   hist24 D3-T, D3-F (incl. dup_newest / dup_foreign) + intact gate -> "§213 hist24 done" (logs/s213_post_hist24.out)
#   nohup bash scratch/s213_recover.sh > logs/s213_recover.out 2>&1 &
cd "$(dirname "$0")/.."
DEV=cuda:1; ts() { date '+%F %T'; }
diag() { python -m boxes.vbank_diag --run boxes/runs_fsq/$1 --part $2 --device $DEV --eval_bs 32 > logs/s213_${2}_$1.log 2>&1; rc=$?; echo "$(ts) $1 $2 rc $rc $(tail -n 1 logs/s213_${2}_$1.log | cut -c1-140)"; }
r=s213_d3f_s1
diag $r rs >> logs/s213_post_landing.out; diag $r d3hist >> logs/s213_post_landing.out
python -c "
import json; R = 'boxes/runs_fsq/$r'; v = json.load(open(R + '/eval_vb.json'))['acc_n2']; h = json.load(open(R + '/eval_d3hist.json'))['intact']['acc_n2']
print('GATE d3hist intact == eval_vb acc_n2 (n 1..12):', all(h[k] == v[k] for k in h))" 2>&1 | sed "s/^/$(ts) b308 $r /" >> logs/s213_post_landing.out
echo "$(ts) §213 post-landing evals done" >> logs/s213_post_landing.out
for c in s213_d3t_s1 s213_d3f_s1; do
  python -m boxes.vbank_halt --run boxes/runs_fsq/$c --device $DEV --eval_bs 32 > logs/s213_vbhalt_$c.log 2>&1; rc=$?; echo "$(ts) $c vbhalt rc $rc $(tail -n 1 logs/s213_vbhalt_$c.log | cut -c1-140)" >> logs/s213_post_halt.out
done; echo "$(ts) §213 vbhalt done" >> logs/s213_post_halt.out
for c in s213_d3t_s1 s213_d3f_s1; do
  diag $c stepbyt >> logs/s213_post_stepbyt.out
  python -c "
import json; R = 'boxes/runs_fsq/$c'; s = json.load(open(R + '/eval_stepbyt.json'))['step']; l = json.load(open(R + '/eval_lfk.json'))['step']
print('GATE stepbyt sums == eval_lfk step:', all(s[n] == {k: l[n][k] for k in s[n]} for n in l) and set(s) == set(l))" 2>&1 | sed "s/^/$(ts) $c /" >> logs/s213_post_stepbyt.out
done; echo "$(ts) §213 stepbyt done" >> logs/s213_post_stepbyt.out
for c in s213_d3t_s1 s213_d3f_s1; do
  diag $c hist24 >> logs/s213_post_hist24.out
  python -c "
import json; R = 'boxes/runs_fsq/$c'; h = json.load(open(R + '/eval_hist24.json'))['intact']; v = json.load(open(R + '/eval_vb.json'))
print('GATE hist24 intact == eval_vb (acc_n2, acc_k24; n 3..8):', all(h['acc_n2'][n] == v['acc_n2'][n] and h['acc_k24'][n] == v['acc_k']['24'][n] for n in h['acc_n2']))" 2>&1 | sed "s/^/$(ts) $c /" >> logs/s213_post_hist24.out
done; echo "$(ts) §213 hist24 done" >> logs/s213_post_hist24.out
echo "$(ts) §213 recovery chain done"
