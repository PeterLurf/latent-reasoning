#!/bin/bash
# 4060 queue (10-22; the 3070 held after the 19:24 crash): §224 K9 take 3 (old vs new part_khorizon, a then b), then §219 T2 eval_textself GPU parts
# (descriptive; Theory 10-22: on the 4060, card recorded; T1's ran on the 3070 -> cross-card). Done-lines in logs/s222_chain.out / logs/s219_post_landing.out.
cd /d/Latent; L=logs/s222_chain.out
for p in a b; do
  echo "$(date '+%F %T') s224 K9$p take 3 start (4060, old vs new)" >> $L
  python -B scratch/s224_k9.py $p > logs/s224_k9${p}_t3.log 2>&1; rc=$?
  echo "$(date '+%F %T') s224 K9$p take 3 rc $rc: $(grep "K9 $p pass" logs/s224_k9${p}_t3.log | tail -1)" >> $L
done
echo "$(date '+%F %T') s219_post_t2 textself GPU parts start (4060; Theory 10-22)" >> logs/s219_post_landing.out
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m boxes.text_eval --run boxes/runs_fsq/s219_t2_s1 --part textself --round 2 --parts gpu --device cuda:0 > logs/s219_t2_post_textself_gpu.log 2>&1; rc=$?
echo "$(date '+%F %T') s219_post_t2 textself_gpu rc $rc (4060; descriptive)" >> logs/s219_post_landing.out
