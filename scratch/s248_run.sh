#!/usr/bin/env bash
# §248 HALT-HEAD ($0 local; HH_v1 + Note 248-1; Audit G-D on 6161aed7): eval_halthead.json on lh_fh_s1 then lh_fh_s2, the 4060 (cuda:0), fp32, eval_bs 250
# for both models (scratch/s248_hh.py). Unread - Theory scores (theory/score_rh.py --hh). Shares the local card lock.
#   nohup bash scratch/s248_run.sh > logs/s248_run.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
L=scratch/_tmp/s240_land.lock
until mkdir $L 2>/dev/null; do sleep 30; done
trap 'rmdir $L' EXIT
for r in lh_fh_s1 lh_fh_s2; do
  echo "$(date '+%F %T') start $r"; python -B scratch/s248_hh.py --run $r --eval_bs 250 --device cuda:0; rc=$?
  echo "$(date '+%F %T') rc $rc $r"; [ $rc = 0 ] || exit 3
done
echo "$(date '+%F %T') S248 DONE"
