#!/usr/bin/env bash
# Chained per card after its PROTOCOL11 worker: cue test (PROTOCOL8B_STEP1.md), then PROTOCOL11 amendment 1. Lock dirs.
# bash scratch/after_p11.sh <device>
cd "$(dirname "$0")/../prosqa"
until grep -q "WORKER_DONE $1" ../logs/p11.log 2>/dev/null; do sleep 60; done
mkdir -p res_p11/sets
for m in ctl_s1 ckpt_s1 ctl_s2 ckpt_s2 ctl_s3 ckpt_s3; do
  mkdir res_p8b/lock_cue_$m 2>/dev/null || continue
  python -u cue_p8b.py --ckpt ckpt/p8b1_${m}_checkpoint_48 --data p8data/p8_test.json --out res_p8b/cue_$m.json --device $1 >> ../logs/cue_$m.log 2>&1
  echo "$(date +%T) cue $m rc $? ($1)" >> ../logs/after_p11.log
done
for m in ckpt_s1 ctl_s1 ckpt_s2 ctl_s2 ckpt_s3 ctl_s3; do
  mkdir res_p11/sets/lock_$m 2>/dev/null || continue
  python -u bidir_p11.py --ckpt ckpt/p8b1_${m}_checkpoint_48 --data p8data/p8_test.json --out res_p11/sets/p8b1_$m.json --save_sets --device $1 >> ../logs/p11a_$m.log 2>&1
  echo "$(date +%T) p11a $m rc $? ($1)" >> ../logs/after_p11.log
done
echo "$(date +%T) WORKER_DONE $1" >> ../logs/after_p11.log
