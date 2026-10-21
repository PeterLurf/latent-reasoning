#!/bin/bash
# Note 216-12: KL-DRIFT for R1-s2 and its base D3-T s2 on the 2070 (cuda:2), fp32, eval_bs 32, own n 3..8 gate vs eval_vb (+-0.005); card in meta.
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -B scratch/s216_kl_base.py --device cuda:2 --runs s216_r1_s2 > logs/s216_kl_base_2070.log 2>&1; rc=$?
echo "$(date '+%F %T') s216 kl_base (2070, s2) rc $rc" >> logs/s216_kl_base.out
