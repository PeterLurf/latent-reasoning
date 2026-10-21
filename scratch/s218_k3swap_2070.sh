#!/bin/bash
# 218-17 (b) NAME-PRIOR-FREE k3swap (descriptive; Theory 10-21) on the untrained LangSLVA, on the 2070 (cuda:2) after §220 CODI20 frees it; card recorded.
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
until grep -q "s220 codi20 (2070) rc" logs/s220_stage0.out; do sleep 60; done
python -B scratch/s218_gl5_known.py --device cuda:2 --arms k3swap --tag _2070 > logs/s218_k3swap_2070.log 2>&1; rc=$?
echo "$(date '+%F %T') 218-17 k3swap (untrained, 2070) rc $rc" >> logs/s218_ref_chain.out
