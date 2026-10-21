#!/bin/bash
# 218-17 / 218-22 DESCRIPTIVE (Theory 10-21; feeds only a possible route-(iii) Note; reads no routing input): K3-antixor on the UNTRAINED LangSLVA on the
# idle 2070 (cuda:2, fp32), registered recipe except lr: steps 2000, bs 64, batches rng 22501, z rng 22551 / 22552, held-out every 250;
# lr in {3e-4, 1e-3} x eps {x10, x1.5}; then, if time allows, lr 1e-4 for 10,000 steps at x10 (the registered full budget).
# -> scratch/gate_s218/g_l5_known/k3antixor_lr_<lr>_x<m>_2070.json. START ONLY AFTER AUDIT'S OK.
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s218_ref_chain.out
for lr in 3e-4 1e-3; do for m in 10 1.5; do
  python -B scratch/s218_gl5_known.py --device cuda:2 --arms k3antixor --k3_mults $m --probe_lr $lr --out_name k3antixor_lr_${lr}_x${m}_2070 > logs/s218_antixor_lr_${lr}_x${m}.log 2>&1; rc=$?
  echo "$(date '+%F %T') 218-17 k3antixor lr $lr x$m (untrained, 2070) rc $rc" >> $L
done; done
python -B scratch/s218_gl5_known.py --device cuda:2 --arms k3antixor --k3_mults 10 --probe_lr 1e-4 --k3_steps 10000 --out_name k3antixor_lr_1e-4_x10_10k_2070 > logs/s218_antixor_lr_1e-4_x10_10k.log 2>&1; rc=$?
echo "$(date '+%F %T') 218-17 k3antixor lr 1e-4 x10 10k steps (untrained, 2070) rc $rc" >> $L
