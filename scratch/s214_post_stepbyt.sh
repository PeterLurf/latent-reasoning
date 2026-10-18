#!/usr/bin/env bash
# §214 eval_stepbyt.json (D4_v1 §6 file list at 10187c1b; Theory 22:4x 10-17: missing from scratch/s214_post_landing.sh c8074432; descriptive, not gated
# by score_s214). Runs on the 3070 (cuda:1) after that chain's "§214 post-landing evals done" line (no co-tenant); the gate as scratch/s213_post_stepbyt.sh:
# per-n sums over t == the cell's eval_lfk.json step block (events, strict_exact, voiddrop_exact).
#   nohup bash scratch/s214_post_stepbyt.sh >> logs/s214_post_landing.out 2>&1 &
cd "$(dirname "$0")/.."
DEV=${DEV:-cuda:1}; r=s214_pausea_s1
until grep -q "§214 post-landing evals done" logs/s214_post_landing.out 2>/dev/null; do sleep 300; done
python -m boxes.vbank_diag --run boxes/runs_fsq/$r --part stepbyt --device $DEV --eval_bs 32 > logs/s214_stepbyt_$r.log 2>&1; rc=$?
echo "$(date '+%F %T') b309 $r stepbyt rc $rc $(tail -n 1 logs/s214_stepbyt_$r.log | cut -c1-140)"
python -c "
import json; R = 'boxes/runs_fsq/$r'; s = json.load(open(R + '/eval_stepbyt.json'))['step']; l = json.load(open(R + '/eval_lfk.json'))['step']
print('GATE stepbyt sums == eval_lfk step:', all(s[n] == {k: l[n][k] for k in s[n]} for n in l) and set(s) == set(l))" 2>&1 | sed "s/^/$(date '+%F %T') b309 $r /"
echo "$(date '+%F %T') §214 stepbyt done"
