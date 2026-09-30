#!/usr/bin/env bash
# Exploratory: hop sweep at answer depths 11-14 (beyond training) on the step-1 models; blind like p8b_hopsweep.sh.
# bash scratch/p8b_hopsweep_deep.sh <device>; models claimed by lock dirs; ctl_s2 left for later (not yet drained).
cd "$(dirname "$0")/../prosqa"
for m in ckpt_s1 ctl_s1 ckpt_s2 ctl_s3 ckpt_s3; do
  mkdir res_p8b/lock_deep_$m 2>/dev/null || continue
  python -u hopsweep_p8b.py --ckpt ckpt/p8b1_${m}_checkpoint_48 --data p8data/p8_test.json --out res_p8b/hopsweep_deep_$m.json --n_per_d 25 --dmin 11 --dmax 14 --skip_probe --device $1 >> ../logs/p8b_hopsweep_deep.log 2>&1
  echo "$(date +%T) $m rc $? ($1)" >> ../logs/p8b_hopsweep_deep.log
done
echo "$(date +%T) WORKER_DONE $1" >> ../logs/p8b_hopsweep_deep.log
