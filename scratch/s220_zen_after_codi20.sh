#!/bin/bash
# §220 stage 0 zen-E/CODI-gpt2 on the 2070 (cuda:2) after CODI20 finishes there (Note 220-2: zen format gate = bs-1 INTACT within +-0.025 of 0.4246,
# in score_s220). G-M first, then -> scratch/s220/zen_codi_gpt2/eval_gcs.json. One card per model.
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s220_stage0.out
until grep -q "s220 codi20 (2070) rc" $L; do sleep 60; done
python -B scratch/s220_gcs.py --model zen --device cuda:2 --gate_m --out_dir scratch/_tmp/s220/gm_zen > logs/s220_gm_zen.log 2>&1; rc=$?
echo "$(date '+%F %T') s220 G-M zen (2070) rc $rc" >> $L
[ $rc -ne 0 ] && exit 1
python -B scratch/s220_gcs.py --model zen --device cuda:2 > logs/s220_zen.log 2>&1; rc=$?
echo "$(date '+%F %T') s220 zen (2070) rc $rc $(ls scratch/s220/zen_codi_gpt2/eval_gcs.json 2>/dev/null)" >> $L
