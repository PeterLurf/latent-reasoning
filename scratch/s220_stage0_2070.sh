#!/bin/bash
# §220 stage 0 on the 2070 (cuda:2), Audit G-D PASS 10-20 (e5ea9325): G-M first (synthetic ids, longest length at K 10, 256 forced steps), then CODI20
# -> boxes/runs_fsq/gsm_codi_20ep_s1/eval_gcs.json. One card per model. Launched from a clean scratch/s220_gcs.py (meta.script_git_status empty).
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s220_stage0.out
python -B scratch/s220_gcs.py --model codi20 --device cuda:2 --gate_m --out_dir scratch/_tmp/s220/gm_codi20 > logs/s220_gm_codi20.log 2>&1; rc=$?
echo "$(date '+%F %T') s220 G-M codi20 (2070) rc $rc" >> $L
[ $rc -ne 0 ] && exit 1
python -B scratch/s220_gcs.py --model codi20 --device cuda:2 > logs/s220_codi20.log 2>&1; rc=$?
echo "$(date '+%F %T') s220 codi20 (2070) rc $rc $(ls boxes/runs_fsq/gsm_codi_20ep_s1/eval_gcs.json 2>/dev/null)" >> $L
