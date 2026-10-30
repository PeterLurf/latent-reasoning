#!/usr/bin/env bash
# §250 FRAME-DIV Stage 0 ($0 local; FD_v1 §3; Audit G-D on 61945c34): eval_frames.json on lh_fh_s1 then lh_fh_s2, the 4060 (cuda:0), fp32, eval_bs 250
# for both models (scratch/s250_frames.py). Unread - Theory scores (Stage 1 decision). Shares the local card lock.
#   nohup bash scratch/s250_run.sh > logs/s250_run.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
L=scratch/_tmp/s240_land.lock
until mkdir $L 2>/dev/null; do sleep 30; done
trap 'rmdir $L' EXIT
for r in lh_fh_s1 lh_fh_s2; do
  echo "$(date '+%F %T') start $r"; python -B scratch/s250_frames.py --run $r --eval_bs 250 --device cuda:0; rc=$?
  echo "$(date '+%F %T') rc $rc $r"; [ $rc = 0 ] || exit 3
done
echo "$(date '+%F %T') S250 DONE"
