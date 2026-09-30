#!/usr/bin/env bash
# PROTOCOL10 amendment 1 (exploratory): bash scratch/p10v_worker.sh <device>. Jobs claimed by lock dirs; ctl_s2 left out (not yet released).
cd "$(dirname "$0")/../prosqa"
for kp in 5 10; do for m in p8b1_ckpt_s1 p8b1_ctl_s1 p8b1_ckpt_s2 p8b1_ctl_s3 p8b1_ckpt_s3; do
  mkdir res_p10/lock_${m}_k${kp} 2>/dev/null || continue
  python -u segchain_p10.py --ckpt ckpt/${m}_checkpoint_48 --data p8data/p8_test.json --out res_p10/${m}_k${kp}_s8.json --Kp $kp --S 8 --device $1 >> ../logs/p10v_${m}_k${kp}.log 2>&1
  echo "$(date +%T) ${m} k$kp rc $? ($1)" >> ../logs/p10v.log
done; done
echo "$(date +%T) WORKER_DONE $1" >> ../logs/p10v.log
