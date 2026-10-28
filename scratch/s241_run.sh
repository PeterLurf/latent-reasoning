#!/usr/bin/env bash
# §241 LEN-STEP ($0 local; LS_v1 + Note 241-1): eval_lenstep.json on lh_fh_s1 then lh_fh_s2, the 4060 (cuda:0), fp32, eval_bs 250 for every cell and model
# (scratch/s241_lenstep.py; writes eval_lenstep.json only). Unread - Theory scores (theory/score_ls.py). Shares s240_land's card lock.
#   nohup bash scratch/s241_run.sh > logs/s241_run.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
L=scratch/_tmp/s240_land.lock
until mkdir $L 2>/dev/null; do sleep 30; done
trap 'rmdir $L' EXIT
for r in lh_fh_s1 lh_fh_s2; do
  echo "$(date '+%F %T') start $r"; python -B scratch/s241_lenstep.py --run $r --eval_bs 250 --device cuda:0; rc=$?
  echo "$(date '+%F %T') rc $rc $r"; [ $rc = 0 ] || exit 3
done
echo "$(date '+%F %T') S241 DONE"
