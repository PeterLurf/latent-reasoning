#!/usr/bin/env bash
# §254 Note 254-1 $0 local work on the 4060 (cuda:0), serialised on the card lock; each step re-run once on a crash (Theory: the 10-31 box faults).
#  (a) STEP-0: inject_aw_eval --parts test_perm on a COPY of boxes/runs_fsq/p2pc_s1 (args.json + train_summary.json + model.pt hard link; the module rewrites eval.json, so the
#      host eval.json is never touched) -> eval_test_perm.json copied into boxes/runs_fsq/p2pc_s1/ (eval_bs 250, fp32; d 1-2 for this d_max 2 run).
#  (b) PILOT pilot_cur_s1: rcb_plaw4_s1's flags (gate_s251.RCB = gen_q_b354; its own --init_trunk, overwritten by the strict --init_model load) with --init_model
#      boxes/runs_fsq/p2pc_s1/model.pt (sha fa90fc3a751c5342) + --perm_p 0.5, seed 1, the 16k schedule, --stop_at 2000, ckpt 1k / 2k, log 100; micro 8 fixed (the §252
#      pilot's measured 4.8 GiB at micro 8 on this card, same flags family), eval_bs 250; logs ws_readout_perm / ws_readout_chain in the 1k / 2k bins.
#   nohup bash scratch/s254_local.sh >> logs/s254_local.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
L=scratch/_tmp/s240_land.lock
until mkdir $L 2>/dev/null; do sleep 30; done
trap 'rmdir $L' EXIT
d=scratch/_tmp/s254/p2pc_step0; rm -rf $d; mkdir -p $d
cp boxes/runs_fsq/p2pc_s1/args.json boxes/runs_fsq/p2pc_s1/train_summary.json $d/ && ln boxes/runs_fsq/p2pc_s1/model.pt $d/model.pt
for try in 1 2; do
  echo "$(date '+%F %T') step0 try $try"; python -X faulthandler -B -m boxes.inject_aw_eval --run $d --parts test_perm --eval_bs 250 --device cuda:0 > $d/eval_stdout_$try.log 2>&1; rc=$?
  echo "$(date '+%F %T') step0 rc $rc"; [ $rc = 0 ] && break
done
[ $rc = 0 ] || { echo "STEP0 FAILED twice"; exit 3; }
cp $d/eval_test_perm.json boxes/runs_fsq/p2pc_s1/eval_test_perm.json; rm -f $d/model.pt; echo "$(date '+%F %T') STEP0 DONE"
RCB="--arm inject --think_k 8 --aux_step 1 --aux_fade 48000,48000 --aux_reweight --fast_add --cache_lower --d_max 4 --steps 16000 --cue step --write_src answer --init_trunk boxes/runs_fsq/s233_askname_s1/model.pt --init_sha c00f7ec0bd7977fe --write_mix_start 0 --write_mix_end 0 --mul_s_p 0"
PF="--init_model boxes/runs_fsq/p2pc_s1/model.pt --init_model_sha fa90fc3a751c5342 --perm_p 0.5"
M=""; PK=""                                                                 # Note 254-2 (2): --mem_check at the d 4 shape first, then pin micro (<= 6.8 GiB)
for mi in 8 16; do
  mc=scratch/_tmp/s254/memck_$mi; rm -rf $mc; mkdir -p $mc
  python -B -m boxes.inject_aw $RCB $PF --micro $mi --eval_bs 50 --mem_check --device cuda:0 --out $mc > $mc/stdout.log 2>&1
  pk=$(python -c "import json; print(json.load(open('$mc/mem_check.json'))['peak_gib_train'])" 2>/dev/null); echo "$(date '+%F %T') mem_check micro $mi train peak ${pk:-none} GiB"
  if [ -n "$pk" ] && python -c "import sys; sys.exit(0 if float('$pk') <= 6.8 else 1)"; then M=$mi; PK=$pk; break; fi
done
[ -n "$M" ] || { echo "PILOT STOP: no micro fits 6.8 GiB"; exit 3; }
for try in 1 2; do
  out=boxes/runs_fsq/pilot_cur_s1; [ $try = 2 ] && [ -d $out ] && mv $out ${out}.void_crash1
  echo "$(date '+%F %T') pilot try $try (micro $M)"; python -X faulthandler -B -m boxes.inject_aw $RCB $PF --micro $M --eval_bs 250 --stop_at 2000 --ckpt_every 1000 --log_every 100 --device cuda:0 --out $out; rc=$?
  echo "$(date '+%F %T') pilot rc $rc"; [ $rc = 9 ] && break
done
[ $rc = 9 ] || { echo "PILOT FAILED twice"; exit 3; }
echo "{\"micro\": $M, \"mem_check_peak_gib_train\": $PK, \"budget_gib\": 6.8, \"gpu\": \"$(python -c 'import torch; print(torch.cuda.get_device_name(0))')\"}" > boxes/runs_fsq/pilot_cur_s1/micro.json; echo "$(date '+%F %T') S254 LOCAL DONE"
