#!/bin/bash
# 4060 (cuda:0) queue c (10-21 06:1x; §218 DIRECT landed 06:07): s218_after_diag_4060b.sh's wrapper was stopped by pid while its reference
# k3antixor x10 python ran (kept running). This waits for that python's WROTE / Traceback, then:
#  (1) reference K2-anti x1.5 / 2.0 / 3.0 (the 218-22 (ii) m-rule input);
#  (2) §218 DIRECT-SYM registered evals on the Stage A eval card (LANG_v1 §4: RTX 4060, fp32): langvb_eval --parts panel,direct,contamination,
#      eval_bs direct:32 (G-EVMEM) -> boxes/runs_fsq/lvb_direct_s1/eval_direct.json (no probe part: DIRECT has none, so 218-18 does not gate it);
#  (3) KL-DRIFT s1 trio (216-10 / 216-12); (4) descriptive reference arms (k3anti / k3antixor x1; k3sym / k3symxor x1 / x10).
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s218_ref_chain.out; K=logs/s216_kl_base.out
until grep -qE "^WROTE k3_anti_xor_ref|Traceback" logs/s218_ref_antixor_x10.log 2>/dev/null; do sleep 60; done; sleep 20
echo "$(date '+%F %T') ref antixor_x10 ended ($(grep -c '^WROTE' logs/s218_ref_antixor_x10.log) WROTE line; started by s218_after_diag_4060b.sh)" >> $L
M="--device cuda:0 --model_run scratch/gate_s218/g_l1c_model"
run() { local tag=$1; shift; python -B scratch/s218_gl5_known.py $M "$@" > logs/s218_ref_$tag.log 2>&1; local rc=$?; echo "$(date '+%F %T') ref $tag rc $rc" >> $L; }
run k2anti --arms k2anti --k2anti_mults 1.5,2.0,3.0 --tag _ref
python -B -m boxes.langvb_eval --run boxes/runs_fsq/lvb_direct_s1 --device cuda:0 --parts panel,direct,contamination --eval_bs direct:32 > logs/s218_direct_eval.log 2>&1; rc=$?
echo "$(date '+%F %T') §218 DIRECT eval_direct (4060) rc $rc $(ls boxes/runs_fsq/lvb_direct_s1/eval_direct.json 2>/dev/null)" >> $L
python -B scratch/s216_kl_base.py --device cuda:0 --runs s215_ftctrl_s1,s215_ftself_s1,s216_r2_s1 > logs/s216_kl_base.log 2>&1; rc=$?
echo "$(date '+%F %T') s216 kl_base (4060, s1) rc $rc" >> $K
run anti_x1 --arms k3anti,k3antixor --k3_mults 1 --tag _ref_x1
run sym --arms k3sym,k3symxor --k3_mults 1,10 --tag _ref
echo "$(date '+%F %T') 4060 queue c done" >> $L
