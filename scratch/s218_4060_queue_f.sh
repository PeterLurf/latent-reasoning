#!/bin/bash
# 4060 (cuda:0) queue f (Note 218-25 revised feasibility, Audit-approved 10-21; replaces queue e, stopped by pid while DIRECT's eval_direct python ran):
# waits for that python to leave the 4060 (UUID GPU-be1bdfb5; two checks 60 s apart), then (1) feas2: S1 (the registered read path: cexor x1.5 / x10
# at t 0 / 1, anti-lin x1.5 t 0, cexor-NOPLANT t 1, cexor x10 t 1 shuffled) and S2 (PROBE2 with the antisymmetric head: cexor x1.5, anti-lin x1.5,
# PROBE(0) randomised order, cexor x1.5 shuffled), each 2,000 steps, held-out every 250 -> g_l5_known/feas2_*.json; (2) KL-DRIFT s1 trio.
# Prints queue d's done-line for scratch/s218_a1_post_landing.sh.
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s218_ref_chain.out; K=logs/s216_kl_base.out
busy() { nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader | grep "GPU-be1bdfb5" | grep -qi python; }
while busy; do sleep 60; done; sleep 60; while busy; do sleep 60; done
echo "$(date '+%F %T') §218 DIRECT eval_direct python ended (started by queue e): $(ls boxes/runs_fsq/lvb_direct_s1/eval_direct.json 2>/dev/null); log tail: $(tail -n 1 logs/s218_direct_eval.log | cut -c1-120)" >> $L
python -B scratch/s218_gl5_known.py --device cuda:0 --arms feas2 > logs/s218_feas2.log 2>&1; rc=$?
echo "$(date '+%F %T') 218-25 feas2 S1 + S2 (untrained, 4060) rc $rc" >> $L
python -B scratch/s216_kl_base.py --device cuda:0 --runs s215_ftctrl_s1,s215_ftself_s1,s216_r2_s1 > logs/s216_kl_base.log 2>&1; rc=$?
echo "$(date '+%F %T') s216 kl_base (4060, s1) rc $rc" >> $K
echo "$(date '+%F %T') 4060 queue d done (queue f)" >> $L
