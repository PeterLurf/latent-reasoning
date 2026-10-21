#!/bin/bash
# 4060 (cuda:0) after the running untrained diag process (overfit32, k3anti, k3antixor, k2anti, k2antixor, k3diag; launched by s218_diag2_then_kl.sh,
# whose bash wrapper was stopped by pid 10-21 01:0x so this queue could insert the ×3.0 K2-anti; the python job was left running):
# (1) Note 218-22 m-rule input: untrained K2-anti-lin at x3.0 (-> g_l5_known/k2_anti_x3.json); (2) KL-DRIFT s1 trio (216-10 / 216-12).
# The symmetric K2 x1.75 / x2.0 is dropped: 218-22 supersedes the symmetric m rule.
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s216_kl_base.out
until grep -qE "^WROTE k3_diag|Traceback" logs/s218_gl5_diag.log; do sleep 60; done
sleep 20
echo "$(date '+%F %T') 218-17 diag process ended (k3_diag written or a traceback; see logs/s218_gl5_diag.log)" >> $L
python -B scratch/s218_gl5_known.py --device cuda:0 --arms k2anti --k2anti_mults 3.0 --tag _x3 > logs/s218_gl5_k2anti_x3.log 2>&1; rc=$?
echo "$(date '+%F %T') 218-17 k2anti x3 (untrained, 4060) rc $rc" >> $L
python -B scratch/s216_kl_base.py --device cuda:0 --runs s215_ftctrl_s1,s215_ftself_s1,s216_r2_s1 > logs/s216_kl_base.log 2>&1; rc=$?
echo "$(date '+%F %T') s216 kl_base (4060, s1) rc $rc" >> $L
