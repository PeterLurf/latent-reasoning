#!/usr/bin/env bash
# §242 NORM-COUNTER ($0 local; NC_v1 + Note 242-1): eval_normctr.json on lh_fh_s1 then lh_fh_s2, the 4060 (cuda:0), fp32, eval_bs 250 for every arm and model
# (scratch/s242_normctr.py; writes eval_normctr.json only). Unread - Theory scores (theory/score_nc.py). Shares the local card lock.
#   nohup bash scratch/s242_run.sh > logs/s242_run.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
L=scratch/_tmp/s240_land.lock
until mkdir $L 2>/dev/null; do sleep 30; done
trap 'rmdir $L' EXIT
for r in lh_fh_s1 lh_fh_s2; do
  echo "$(date '+%F %T') start $r"; python -B scratch/s242_normctr.py --run $r --eval_bs 250 --device cuda:0; rc=$?
  echo "$(date '+%F %T') rc $rc $r"; [ $rc = 0 ] || exit 3
done
echo "$(date '+%F %T') S242 DONE"
