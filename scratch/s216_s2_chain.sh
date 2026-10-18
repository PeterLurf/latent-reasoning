#!/usr/bin/env bash
# §216 s2 lineage, after b314 s216_r1_s2 lands AND scratch/s216_post_landing.sh's "§216 b313 evals done" (one job on the 3070 at a time), fp32, cuda:1.
# Critical path to s216_r2_s2 first (SD2_v1 §2.4 round-2 condition needs R1-s2's eval_vb + selfdepth; the s2 round-2 labelling uses R1-s2's model):
#   R1-s2 eval_vb -> selfdepth --lineage s2 --round 1 -> GATE-2-s2 / GATE-1'-s2 items + scoring (B1 --lineage s2 --round 2 --stage gates; B2; fallback =
#   s216_d3t_s2) -> the r2-s2 pool (B1 --stage pool) -> the rest of R1-s2's round-1 panel (steppos, lfk, rs, d3hist, hist24, vbhalt, chainlabel,
#   chainlabel_fresh) -> the d3hist gate. R1-s2's eval_deep (V3, required) runs only after s216_r2_s2 has launched (scratch/s216_r1s2_deep.sh).
#   nohup bash scratch/s216_s2_chain.sh >> logs/s216_s2_chain.out 2>&1 &
cd "$(dirname "$0")/.."
DEV=${DEV:-cuda:1}; n=314; r=s216_r1_s2; say() { echo "$(date '+%F %T') $*"; }
diag() { for P in "$@"; do python -m boxes.vbank_diag --run boxes/runs_fsq/$r --part $P --device $DEV --eval_bs 32 ${XA:-} > logs/s216_${P}_$r.log 2>&1; rc=$?
  say "b$n $r $P rc $rc $(tail -n 1 logs/s216_${P}_$r.log | cut -c1-160)"; done; }
until grep -q "ALL_DRAINED" logs/auto_drain_gen_b$n.out 2>/dev/null && [ -s boxes/runs_fsq/$r/model.pt ] && [ -s boxes/runs_fsq/$r/log.jsonl ]; do sleep 120; done
say "b$n $r drained"
until grep -q "§216 b313 evals done" logs/s216_post_landing.out 2>/dev/null; do sleep 120; done
say "b$n $r -> evals on $DEV (b313's panel is done)"
python -m boxes.vbank_eval --run boxes/runs_fsq/$r --device $DEV --eval_bs 32 > logs/s216_eval_vb_$r.log 2>&1; rc=$?; say "b$n $r eval_vb rc $rc $(tail -n 1 logs/s216_eval_vb_$r.log | cut -c1-160)"
XA="--lineage s2 --round 1" diag selfdepth
python scratch/s216_build_pool.py --lineage s2 --round 2 --stage gates --device $DEV > logs/s216_s2r2_gates.log 2>&1; rc=$?; say "B1 s2 r2 gates rc $rc $(tail -n 1 logs/s216_s2r2_gates.log | cut -c1-200)"
python scratch/s216_gate2.py --lineage s2 --round 2 --device $DEV > logs/s216_s2r2_gate2.log 2>&1; rc=$?; say "B2 s2 r2 rc $rc $(tail -n 1 logs/s216_s2r2_gate2.log | cut -c1-300)"
python scratch/s216_build_pool.py --lineage s2 --round 2 --stage pool --device $DEV > logs/s216_s2r2_pool.log 2>&1; rc=$?; say "B1 s2 r2 pool rc $rc $(tail -n 1 logs/s216_s2r2_pool.log | cut -c1-200)"
diag steppos lfk rs d3hist hist24
python -m boxes.vbank_halt --run boxes/runs_fsq/$r --device $DEV --eval_bs 32 > logs/s216_vbhalt_$r.log 2>&1; rc=$?; say "b$n $r vbhalt rc $rc $(tail -n 1 logs/s216_vbhalt_$r.log | cut -c1-160)"
diag chainlabel chainlabel_fresh
python -c "
import json; R = 'boxes/runs_fsq/$r'; v = json.load(open(R + '/eval_vb.json'))['acc_n2']; h = json.load(open(R + '/eval_d3hist.json'))['intact']['acc_n2']
print('GATE d3hist intact == eval_vb acc_n2 (n 1..12):', all(h[k] == v[k] for k in h))" 2>&1 | sed "s/^/$(date '+%F %T') b$n $r /"
say "§216 s2 chain done"
