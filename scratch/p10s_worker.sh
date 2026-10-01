#!/usr/bin/env bash
# PROTOCOL10 amendment 2 (exploratory, d3-6): bash scratch/p10s_worker.sh <device>. Jobs claimed by lock dirs.
cd "$(dirname "$0")/../prosqa"
for kp in 5 10; do for m in p8b1_ckpt_s1 p8b1_ctl_s1 p8b1_ckpt_s2 p8b1_ctl_s3 p8b1_ckpt_s3; do
  mkdir res_p10/lock_${m}_k${kp}_d36 2>/dev/null || continue
  python -u segchain_p10.py --ckpt ckpt/${m}_checkpoint_48 --data p8data/p8_test.json --out res_p10/${m}_k${kp}_s8_d3-6.json --Kp $kp --S 8 --dmin 3 --dmax 6 --exclude_cal --device $1 >> ../logs/p10s_${m}_k${kp}.log 2>&1
  echo "$(date +%T) ${m} k$kp rc $? ($1)" >> ../logs/p10s.log
done; done
echo "$(date +%T) WORKER_DONE $1" >> ../logs/p10s.log
