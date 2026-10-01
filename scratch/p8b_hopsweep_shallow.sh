#!/usr/bin/env bash
# Exploratory diagnostic (PROTOCOL8B_STEP1.md): hop sweep at depths 3-6. bash scratch/p8b_hopsweep_shallow.sh <device>; lock dirs.
cd "$(dirname "$0")/../prosqa"
for m in ctl_s1 ckpt_s1 ctl_s3 ckpt_s2; do
  mkdir res_p8b/lock_shallow_$m 2>/dev/null || continue
  python -u hopsweep_p8b.py --ckpt ckpt/p8b1_${m}_checkpoint_48 --data p8data/p8_test.json --out res_p8b/hopsweep_shallow_$m.json --n_per_d 40 --dmin 3 --dmax 6 --skip_probe --device $1 >> ../logs/p8b_hopsweep_shallow.log 2>&1
  echo "$(date +%T) $m rc $? ($1)" >> ../logs/p8b_hopsweep_shallow.log
done
echo "$(date +%T) WORKER_DONE $1" >> ../logs/p8b_hopsweep_shallow.log
