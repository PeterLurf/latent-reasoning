#!/bin/bash
# 4060 (cuda:0) queue after the 218-17 K3 run ends (k3 written or a crash), Theory / Audit 10-21 (Note 218-18 waits on it):
# (1) one invocation, each JSON saved as it completes: overfit32 (sym + antisym, param identity) -> k3anti (lin, x1 / x10) -> k3antixor (x1 / x10)
#     -> k2anti (x1 / 1.5 / 2) -> k2antixor (x1 / 1.5 / 2) -> k3diag (items 1-5);
# (2) symmetric K2 x1.75 / x2.0 ONLY IF k3anti failed to learn (Theory: else superseded / moot). Rule fixed here before any k3anti output:
#     'failed' = the max held-out accuracy over k3anti's x1 and x10 curves < 0.60;
# (3) KL-DRIFT s1 trio (216-10 / 216-12). One job at a time.
# (replaces scratch/s218_diag_then_kl.sh, stopped by pid while still waiting - it had started no job)
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s216_kl_base.out
until grep -qE "WROTE k3_probe_learn|Traceback" logs/s218_gl5_known.log; do sleep 60; done
sleep 30
python -B scratch/s218_gl5_known.py --device cuda:0 --arms overfit32,k3anti,k3antixor,k2anti,k2antixor,k3diag > logs/s218_gl5_diag.log 2>&1; rc=$?
echo "$(date '+%F %T') 218-17 diag (overfit32, k3anti, k3antixor, k2anti, k2antixor, k3diag) rc $rc" >> $L
fail=$(python -c "
import json
try:
    d = json.load(open('scratch/gate_s218/g_l5_known/k3_anti.json'))
    mx = max(a for v in d['by_mult'].values() for _, a in v['curve'])
    print('1' if mx < 0.60 else '0')
except Exception as e:
    print('1')")
echo "$(date '+%F %T') 218-17 k3anti failed-to-learn = $fail (rule: max held-out < 0.60)" >> $L
if [ "$fail" = "1" ]; then
  for m in 1.75:eps175 2.0:eps200; do
    python -B scratch/s218_gl5_known.py --device cuda:0 --arms k2 --k2_mults ${m%%:*} --k2_name k2_oracle_${m##*:} > logs/s218_gl5_known_k2_${m##*:}.log 2>&1; rc=$?
    echo "$(date '+%F %T') 218-17 k2 ${m##*:} rc $rc" >> $L
  done
fi
python -B scratch/s216_kl_base.py --device cuda:0 --runs s215_ftctrl_s1,s215_ftself_s1,s216_r2_s1 > logs/s216_kl_base.log 2>&1; rc=$?
echo "$(date '+%F %T') s216 kl_base (4060, s1) rc $rc" >> $L
