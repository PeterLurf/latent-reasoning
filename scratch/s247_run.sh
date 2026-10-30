#!/usr/bin/env bash
# §247 READ-HALT ($0 local; RH_v1 + Note 247-1; Audit G-D b34056b5): eval_c7.json + eval_halt.json on lh_fh_s1 then lh_fh_s2, the 4060 (cuda:0), fp32, eval_bs 250
# for both parts and models (scratch/s247_rh.py). Unread - Theory scores (theory/score_rh.py). Shares the local card lock.
#   nohup bash scratch/s247_run.sh > logs/s247_run.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
L=scratch/_tmp/s240_land.lock
until mkdir $L 2>/dev/null; do sleep 30; done
trap 'rmdir $L' EXIT
for r in lh_fh_s1 lh_fh_s2; do
  echo "$(date '+%F %T') start $r"; python -B scratch/s247_rh.py --run $r --eval_bs 250 --device cuda:0; rc=$?
  echo "$(date '+%F %T') rc $rc $r"; [ $rc = 0 ] || exit 3
done
echo "$(date '+%F %T') S247 DONE"
