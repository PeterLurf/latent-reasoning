#!/usr/bin/env bash
# §249 FRAME-TRANSFER ($0 local; FT_v1 + Note 249-1; Audit G-D on a755253b): eval_transfer.json on lh_fh_s1 / lh_fh_s2 (LAT, eval_bs 250) and df_pcd4_s1 (CD-AN,
# eval_bs 100: the direct trunk on the cued text OOMs the 8 GB 4060 at 250), the 4060 (cuda:0), fp32, one eval_bs per model (scratch/s249_ft.py). Unread - Theory scores
# (theory/score_ft.py). Shares the local card lock.   nohup bash scratch/s249_run.sh > logs/s249_run.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
L=scratch/_tmp/s240_land.lock
until mkdir $L 2>/dev/null; do sleep 30; done
trap 'rmdir $L' EXIT
for rb in lh_fh_s1:250 lh_fh_s2:250 df_pcd4_s1:100; do
  r=${rb%%:*}; bs=${rb##*:}
  echo "$(date '+%F %T') start $r (bs $bs)"; python -B scratch/s249_ft.py --run $r --eval_bs $bs --device cuda:0; rc=$?
  echo "$(date '+%F %T') rc $rc $r"; [ $rc = 0 ] || exit 3
done
echo "$(date '+%F %T') S249 DONE"
