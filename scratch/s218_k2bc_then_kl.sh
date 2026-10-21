#!/bin/bash
# 4060 (cuda:0) queue after the 218-17 K1-K3 run ends (K3 written or a crash): (1) K2 ORACLE at eps x1.75 and (2) x2.0 (Theory 10-20, the Note 218-18
# plant rule: m = 1.75 if K2(x1.75) >= 0.985, else 2.0 if K2(x2.0) >= 0.985) -> g_l5_known/k2_oracle_eps175.json / k2_oracle_eps200.json; (3) KL-DRIFT
# (216-10) on the s1 trio + D3-T s1 (Note 216-12: R1-s2 + D3-T s2 run on the 2070, scratch/s216_kl_2070.sh). One job at a time.
# (replaces scratch/s218_k2b_then_kl.sh, stopped by pid while still waiting - it had started no job)
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
until grep -qE "WROTE k3_probe_learn|Traceback" logs/s218_gl5_known.log; do sleep 60; done
sleep 30
for m in 1.75:eps175 2.0:eps200; do
  python -B scratch/s218_gl5_known.py --device cuda:0 --arms k2 --k2_mults ${m%%:*} --k2_name k2_oracle_${m##*:} > logs/s218_gl5_known_k2_${m##*:}.log 2>&1; rc=$?
  echo "$(date '+%F %T') 218-17 k2 ${m##*:} rc $rc" >> logs/s216_kl_base.out
done
python -B scratch/s216_kl_base.py --device cuda:0 --runs s215_ftctrl_s1,s215_ftself_s1,s216_r2_s1 > logs/s216_kl_base.log 2>&1; rc=$?
echo "$(date '+%F %T') s216 kl_base (4060, s1) rc $rc" >> logs/s216_kl_base.out
