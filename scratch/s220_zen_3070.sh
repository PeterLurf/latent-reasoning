#!/bin/bash
# §220 stage 0 zen-E/CODI-gpt2 on the 3070 (cuda:1) - Theory 10-21: 'after CODI20' was a reading order only; zen's card = the 3070. The §219 eval chain has
# priority on the 3070: if T1 lands first, zen yields (eval-only; partials kept as .void, restart). G-M first, then -> scratch/s220/zen_codi_gpt2/eval_gcs.json.
# (replaces scratch/s220_zen_after_codi20.sh, stopped by pid while still waiting - it had started no job)
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s220_stage0.out
python -B scratch/s220_gcs.py --model zen --device cuda:1 --gate_m --out_dir scratch/_tmp/s220/gm_zen > logs/s220_gm_zen.log 2>&1; rc=$?
echo "$(date '+%F %T') s220 G-M zen (3070) rc $rc" >> $L
[ $rc -ne 0 ] && exit 1
python -B scratch/s220_gcs.py --model zen --device cuda:1 > logs/s220_zen.log 2>&1; rc=$?
echo "$(date '+%F %T') s220 zen (3070) rc $rc $(ls scratch/s220/zen_codi_gpt2/eval_gcs.json 2>/dev/null)" >> $L
