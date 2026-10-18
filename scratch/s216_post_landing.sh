#!/usr/bin/env bash
# §216 post-landing on the 3070 (cuda:1), fp32, one job at a time (SD2_v1 §4.1 / §2.4 / §8 at e9ade2ec). The s2 lineage is the long pole, so b312 first:
#  (A) b312 s216_d3t_s2 after its drain: eval_vb (all parts; `acc` registered for COMPETENT-BASE) -> chainlabel_fresh -> hist24 -> COMPETENT-BASE
#      (acc_n2 >= 0.99 and acc_k[24] >= 0.97 at every n 1..8; a completed run below the floor = "s2 base not competent", stop the s2 lineage) ->
#      GATE-1-s2 items + scoring (B1 --lineage s2 --round 1 --stage gates; B2) -> the r1-s2 pool (B1 --stage pool; exit 3 = no passing labeller = x2 FAIL).
#  (B) b313 s216_r2_s1 after its drain: eval_vb -> deep (bs 16, G-S16) -> steppos -> lfk -> rs -> d3hist -> hist24 (E4) -> vbhalt -> chainlabel ->
#      chainlabel_fresh (E3, n 9..24) -> selfdepth --lineage s1 --round 2 (E2) -> the d3hist gate.
# rc captured before any $(date); done-lines "§216 b312 base + s2 round-1 labelling done" / "§216 b313 evals done".
#   nohup bash scratch/s216_post_landing.sh >> logs/s216_post_landing.out 2>&1 &
cd "$(dirname "$0")/.."
DEV=${DEV:-cuda:1}; say() { echo "$(date '+%F %T') $*"; }
diag() { for P in "$@"; do python -m boxes.vbank_diag --run boxes/runs_fsq/$r --part $P --device $DEV --eval_bs ${BS:-32} ${XA:-} > logs/s216_${P}_$r.log 2>&1; rc=$?
  say "b$n $r $P rc $rc $(tail -n 1 logs/s216_${P}_$r.log | cut -c1-160)"; done; }
vb() { python -m boxes.vbank_eval --run boxes/runs_fsq/$r --device $DEV --eval_bs 32 > logs/s216_eval_vb_$r.log 2>&1; rc=$?; say "b$n $r eval_vb rc $rc $(tail -n 1 logs/s216_eval_vb_$r.log | cut -c1-160)"; }
drained() { until grep -q "ALL_DRAINED" logs/auto_drain_gen_b$1.out 2>/dev/null && [ -s boxes/runs_fsq/$2/model.pt ] && [ -s boxes/runs_fsq/$2/log.jsonl ]; do sleep 120; done; }
# ---- (A) the s2 base
n=312; r=s216_d3t_s2; drained $n $r; say "b$n $r drained -> evals on $DEV"
vb; diag chainlabel_fresh hist24
python -c "
import json, sys; v = json.load(open('boxes/runs_fsq/$r/eval_vb.json')); L = [json.loads(l) for l in open('boxes/runs_fsq/$r/log.jsonl')]; last = max(x.get('step', 0) for x in L)
a2 = {n: v['acc_n2'][str(n)] for n in range(1, 9)}; a24 = {n: v['acc_k']['24'][str(n)] for n in range(1, 9)}
ok = last >= 48000 and all(a2[n] >= 0.99 and a24[n] >= 0.97 for n in a2); o = {'last_step': last, 'acc_n2': a2, 'acc_k24': a24, 'COMPETENT_BASE': ok}
json.dump(o, open('scratch/gate_s216/competent_base_s2.json', 'w', newline='\n'), indent=1); print('COMPETENT-BASE', o); sys.exit(0 if ok else 4)"; rc=$?
say "b$n COMPETENT-BASE rc $rc (0 = competent)"
if [ $rc = 0 ]; then
  python scratch/s216_build_pool.py --lineage s2 --round 1 --stage gates --device $DEV > logs/s216_s2r1_gates.log 2>&1; rc=$?; say "B1 s2 r1 gates rc $rc $(tail -n 1 logs/s216_s2r1_gates.log | cut -c1-160)"
  python scratch/s216_gate2.py --lineage s2 --round 1 --device $DEV > logs/s216_s2r1_gate.log 2>&1; rc=$?; say "B2 s2 r1 rc $rc $(tail -n 1 logs/s216_s2r1_gate.log | cut -c1-200)"
  python scratch/s216_build_pool.py --lineage s2 --round 1 --stage pool --device $DEV > logs/s216_s2r1_pool.log 2>&1; rc=$?; say "B1 s2 r1 pool rc $rc $(tail -n 1 logs/s216_s2r1_pool.log | cut -c1-160)"
else
  say "s2 base not competent (or incomplete): s2 lineage stops here - tell Theory / Audit"
fi
say "§216 b312 base + s2 round-1 labelling done"
# ---- (B) the s1 round-2 cell
n=313; r=s216_r2_s1; drained $n $r; say "b$n $r drained -> evals on $DEV"
vb; BS=16 diag deep; diag steppos lfk rs d3hist hist24
python -m boxes.vbank_halt --run boxes/runs_fsq/$r --device $DEV --eval_bs 32 > logs/s216_vbhalt_$r.log 2>&1; rc=$?; say "b$n $r vbhalt rc $rc $(tail -n 1 logs/s216_vbhalt_$r.log | cut -c1-160)"
diag chainlabel chainlabel_fresh; XA="--lineage s1 --round 2" diag selfdepth
python -c "
import json; R = 'boxes/runs_fsq/$r'; v = json.load(open(R + '/eval_vb.json'))['acc_n2']; h = json.load(open(R + '/eval_d3hist.json'))['intact']['acc_n2']
print('GATE d3hist intact == eval_vb acc_n2 (n 1..12):', all(h[k] == v[k] for k in h))" 2>&1 | sed "s/^/$(date '+%F %T') b$n $r /"
say "§216 b313 evals done"
