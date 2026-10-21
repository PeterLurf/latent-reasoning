#!/bin/bash
# 4060 (cuda:0) queue after the 218-17 K3 run ends (k3 written or a crash), Theory / Audit 10-21 (Note 218-18 waits on it):
# (1) OVERFIT-32 (sym + antisym, eps x10, param identity) -> item 6 K3 antisymmetric (x1, x10) -> K2-anti (x1 / 1.5 / 2) -> K3-DIAG items 1-5,
#     one invocation, each JSON saved as it completes (g_l5_known/overfit32, k3_anti, k2_anti, k3_diag);
# (2) K2 x1.75 / x2.0 (symmetric, the 218-18 m rule as first written); (3) KL-DRIFT s1 trio (216-10 / 216-12). One job at a time.
# (replaces scratch/s218_k2bc_then_kl.sh, stopped by pid while still waiting - it had started no job)
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s216_kl_base.out
until grep -qE "WROTE k3_probe_learn|Traceback" logs/s218_gl5_known.log; do sleep 60; done
sleep 30
python -B scratch/s218_gl5_known.py --device cuda:0 --arms overfit32,k3anti,k2anti,k3diag > logs/s218_gl5_diag.log 2>&1; rc=$?
echo "$(date '+%F %T') 218-17 diag (overfit32, k3anti, k2anti, k3diag) rc $rc" >> $L
for m in 1.75:eps175 2.0:eps200; do
  python -B scratch/s218_gl5_known.py --device cuda:0 --arms k2 --k2_mults ${m%%:*} --k2_name k2_oracle_${m##*:} > logs/s218_gl5_known_k2_${m##*:}.log 2>&1; rc=$?
  echo "$(date '+%F %T') 218-17 k2 ${m##*:} rc $rc" >> $L
done
python -B scratch/s216_kl_base.py --device cuda:0 --runs s215_ftctrl_s1,s215_ftself_s1,s216_r2_s1 > logs/s216_kl_base.log 2>&1; rc=$?
echo "$(date '+%F %T') s216 kl_base (4060, s1) rc $rc" >> $L
