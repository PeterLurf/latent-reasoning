#!/bin/bash
# 4060 (cuda:0) queue d (Audit 10-21 06:2x): s218_4060_queue_c.sh's wrapper stopped by pid while its reference K2-anti python ran (kept running).
# Waits for that python, then: (1) §218 DIRECT-SYM eval_direct (panel, direct, contamination; eval_bs direct:32 per G-EVMEM; Stage A card);
# (2) KL-DRIFT s1 trio (216-10 / 216-12). A1's non-probe evals at landing are in scratch/s218_a1_post_landing.sh (waits for this queue + the 4060 free).
# The descriptive reference arms (anti x1, sym x1 / x10) are DROPPED from this queue (Audit: never ahead of A1; only if Theory still wants them).
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s218_ref_chain.out; K=logs/s216_kl_base.out
until grep -qE "^WROTE k2_anti_ref|Traceback" logs/s218_ref_k2anti.log 2>/dev/null; do sleep 60; done; sleep 20
echo "$(date '+%F %T') ref k2anti ended ($(grep -c '^WROTE' logs/s218_ref_k2anti.log) WROTE line; started by s218_4060_queue_c.sh)" >> $L
python -B -m boxes.langvb_eval --run boxes/runs_fsq/lvb_direct_s1 --device cuda:0 --parts panel,direct,contamination --eval_bs direct:32 > logs/s218_direct_eval.log 2>&1; rc=$?
echo "$(date '+%F %T') §218 DIRECT eval_direct (4060) rc $rc $(ls boxes/runs_fsq/lvb_direct_s1/eval_direct.json 2>/dev/null)" >> $L
python -B scratch/s216_kl_base.py --device cuda:0 --runs s215_ftctrl_s1,s215_ftself_s1,s216_r2_s1 > logs/s216_kl_base.log 2>&1; rc=$?
echo "$(date '+%F %T') s216 kl_base (4060, s1) rc $rc" >> $K
echo "$(date '+%F %T') 4060 queue d done" >> $L
