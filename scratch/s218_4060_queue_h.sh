#!/bin/bash
# 4060 (cuda:0) queue h (Audit 10-21): queue g's wrapper stopped by pid while its feas2_s1 python ran (kept running). After that python leaves the 4060:
# (1) §218 DIRECT contamination re-run (langvb_eval one-liner ddf914da, Audit G-D PASS): same card (cuda:0), --parts contamination, with the before /
#     after identity check (scratch/s218_direct_contam_check.py); (2) KL-DRIFT s1; prints queue d's done-line for A1's post-landing chain.
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s218_ref_chain.out; K=logs/s216_kl_base.out
busy() { nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader | grep "GPU-be1bdfb5" | grep -qi python; }
while busy; do sleep 60; done; sleep 60; while busy; do sleep 60; done
echo "$(date '+%F %T') 218-25 feas2 S1 python ended (started by queue g; $(grep -c '^WROTE' logs/s218_feas2_s1.log) WROTE lines)" >> $L
mkdir -p scratch/_tmp/s218_contam && cp boxes/runs_fsq/lvb_direct_s1/eval_direct.json scratch/_tmp/s218_contam/eval_direct.before.json
python -B -m boxes.langvb_eval --run boxes/runs_fsq/lvb_direct_s1 --device cuda:0 --parts contamination > logs/s218_direct_contam.log 2>&1; rc=$?
chk=$(python -I scratch/s218_direct_contam_check.py scratch/_tmp/s218_contam/eval_direct.before.json boxes/runs_fsq/lvb_direct_s1/eval_direct.json); crc=$?
echo "$(date '+%F %T') §218 DIRECT contamination re-run (4060) rc $rc; identity check rc $crc: $chk" >> $L
python -B scratch/s216_kl_base.py --device cuda:0 --runs s215_ftctrl_s1,s215_ftself_s1,s216_r2_s1 > logs/s216_kl_base.log 2>&1; rc=$?
echo "$(date '+%F %T') s216 kl_base (4060, s1) rc $rc" >> $K
echo "$(date '+%F %T') 4060 queue d done (queue h)" >> $L
