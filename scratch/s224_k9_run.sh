#!/bin/bash
# §224 K9 take 2 (scratch/s224_k9.py a then b) on the 3070; done-lines in logs/s222_chain.out.
cd /d/Latent; L=logs/s222_chain.out; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "$(date '+%F %T') s224 K9a start (3070, n 1..4)" >> $L
python -B scratch/s224_k9.py a > logs/s224_k9a.log 2>&1; rc=$?; echo "$(date '+%F %T') s224 K9a rc $rc: $(grep 'K9 a pass' logs/s224_k9a.log | tail -1)" >> $L
echo "$(date '+%F %T') s224 K9b start (3070, n 7..8, max_split_size_mb:128)" >> $L
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python -B scratch/s224_k9.py b > logs/s224_k9b.log 2>&1; rc=$?; echo "$(date '+%F %T') s224 K9b rc $rc: $(grep 'K9 b pass' logs/s224_k9b.log | tail -1)" >> $L
