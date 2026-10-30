#!/usr/bin/env bash
# §251 ANSWER-EXTEND local refs (AE_v1 §2 / §3.4; $0, the 4060 = cuda:0, fp32), serialised on the shared card lock. Writes the output dirs only. Unread.
#   CTRL_re : rcb_plaw4_s1 / s2 under the CURRENT eval module (eval sha drift bd515f -> 21ae80) into rcb_plaw4_s<n>_re (model.pt hard link, args / train_summary copied,
#             rcb's own eval parts; eval_bs 100: bs 250 OOMed the 8 GB 4060 in acc, kept .void).
#   step 0  : aw_plaw2_s1 REG step-0 eval at set_d_max 4 (s236_reg_eval --law 0) into aw_plaw2_s1_reg_s251 (acc, ws, swap_kd, lp5h, cue_only, cue_wrong, lp5; eval_bs 100).
#   nohup bash scratch/s251_refs.sh >> logs/s251_refs.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # allocator only (no numeric change)
L=scratch/_tmp/s240_land.lock
until mkdir $L 2>/dev/null; do sleep 30; done
trap 'rmdir $L' EXIT
EV=acc,lp5,mean_s,cue_only,cue_wrong,swap_k,swap_1,probe,ws,lp5h,swap_kd
for s in 1 2; do
  src=boxes/runs_fsq/rcb_plaw4_s$s; d=boxes/runs_fsq/rcb_plaw4_s${s}_re; mkdir -p $d
  cp $src/args.json $src/train_summary.json $d/; [ -f $d/model.pt ] || ln $src/model.pt $d/model.pt
  echo "$(date '+%F %T') start re rcb_plaw4_s$s"; python -B -m boxes.inject_aw_eval --run $d --parts $EV --eval_bs 100 --device cuda:0; rc=$?
  echo "$(date '+%F %T') rc $rc re rcb_plaw4_s$s"; [ $rc = 0 ] || exit 3; rm -f $d/model.pt
done
echo "$(date '+%F %T') start step0 aw_plaw2_s1"; python -B scratch/s236_reg_eval.py --law 0 --run aw_plaw2_s1 --set_d_max 4 --parts acc,ws,swap_kd,lp5h,cue_only,cue_wrong,lp5 --eval_bs 100 --device cuda:0 --out boxes/runs_fsq/aw_plaw2_s1_reg_s251; rc=$?
echo "$(date '+%F %T') rc $rc step0 aw_plaw2_s1"; [ $rc = 0 ] || exit 3; rm -f boxes/runs_fsq/aw_plaw2_s1_reg_s251/model.pt
echo "$(date '+%F %T') S251 REFS DONE"
