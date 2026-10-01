#!/usr/bin/env bash
# PROTOCOL11 worker: bash scratch/p11_worker.sh <device>; models claimed by lock dirs; log logs/p11.log.
cd "$(dirname "$0")/../prosqa"
mkdir -p res_p11
for m in p8b1_ckpt_s1 p8b1_ctl_s1 p8b1_ckpt_s2 p8b1_ctl_s2 p8b1_ckpt_s3 p8b1_ctl_s3 p8_coconut; do
  mkdir res_p11/lock_$m 2>/dev/null || continue
  python -u bidir_p11.py --ckpt ckpt/${m}_checkpoint_48 --data p8data/p8_test.json --out res_p11/$m.json --device $1 >> ../logs/p11_$m.log 2>&1
  echo "$(date +%T) $m rc $? ($1)" >> ../logs/p11.log
done
echo "$(date +%T) WORKER_DONE $1" >> ../logs/p11.log
