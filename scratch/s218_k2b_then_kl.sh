#!/bin/bash
# 4060 (cuda:0) queue after the 218-17 K1-K3 run ends (K3 written or a crash): (1) K2 ORACLE at eps x1.75 (Theory 10-20, the planned Note 218-18 plant,
# same oracle) -> scratch/gate_s218/g_l5_known/k2_oracle_eps175.json; (2) Note 216-10 KL-DRIFT (Audit-approved) -> <run>/eval_kl_base.json. One job at a time.
# (replaces scratch/s216_kl_after_k3.sh, stopped by pid while still waiting - it had started no job)
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
until grep -qE "WROTE k3_probe_learn|Traceback" logs/s218_gl5_known.log; do sleep 60; done
sleep 30
python -B scratch/s218_gl5_known.py --device cuda:0 --arms k2 --k2_mults 1.75 --k2_name k2_oracle_eps175 > logs/s218_gl5_known_k2b.log 2>&1; rc=$?
echo "$(date '+%F %T') 218-17 k2 eps1.75 rc $rc" >> logs/s216_kl_base.out
python -B scratch/s216_kl_base.py --device cuda:0 > logs/s216_kl_base.log 2>&1; rc=$?
echo "$(date '+%F %T') s216 kl_base rc $rc" >> logs/s216_kl_base.out
