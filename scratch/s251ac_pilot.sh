#!/usr/bin/env bash
# Note 251-10 ANSWER-AT-CUE $0 local PILOT (predictive only, not scored): the §244 recipe exactly (ao_s1's queue line gen_q_b366, run flags; askname init, mul_s_p 1,
# d_max 2, 12k cosine schedule), seed 1, --stop_at 2000 (ckpt at 1k / 2k, exit 9, no eval), both cells on ONE card (the 4060 = cuda:0), serialised on the card lock:
#   pac_acue_s1 : + --cue_ans 1;   pac_ctrl_s1 : no flag.
# micro: the smallest of 2 / 4 / 8 / 16 whose --mem_check train peak (the P-ACUE flags, the heavier cell) <= 6.8 GiB, then the SAME --micro for both cells (recorded in
# scratch/s251ac_pilot/micro.json with the GPU name). Theory reads log.jsonl ws_readout at 1k / 2k (the rule is Note 251-10's).
#   nohup bash scratch/s251ac_pilot.sh >> logs/s251ac_pilot.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
L=scratch/_tmp/s240_land.lock
until mkdir $L 2>/dev/null; do sleep 30; done
trap 'rmdir $L' EXIT
AO="--arm inject --think_k 8 --aux_step 1 --aux_fade 0,0 --aux_reweight --fast_add --cache_lower --d_max 2 --steps 12000 --cue step --write_src answer --init_trunk boxes/runs_fsq/s233_askname_s1/model.pt --init_sha c00f7ec0bd7977fe --write_mix_start 0 --write_mix_end 0"
mkdir -p scratch/s251ac_pilot; M=""
for mi in 2 4 8 16; do
  d=scratch/_tmp/s251ac/memck_$mi; rm -rf $d; mkdir -p $d
  python -B -m boxes.inject_aw $AO --cue_ans 1 --micro $mi --eval_bs 50 --mem_check --device cuda:0 --out $d > $d/stdout.log 2>&1
  pk=$(python -c "import json; print(json.load(open('$d/mem_check.json'))['peak_gib_train'])" 2>/dev/null); echo "$(date '+%F %T') mem_check micro $mi train peak ${pk:-none} GiB"
  if [ -n "$pk" ] && python -c "import sys; sys.exit(0 if float('$pk') <= 6.8 else 1)"; then M=$mi; break; fi
done
[ -n "$M" ] || { echo "$(date '+%F %T') PILOT STOP: no micro fits 6.8 GiB"; exit 3; }
gpu=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
echo "{\"micro\": $M, \"gpu\": \"$gpu\", \"budget_gib\": 6.8}" > scratch/s251ac_pilot/micro.json; echo "$(date '+%F %T') micro $M on $gpu"
for c in acue ctrl; do
  ex=""; [ $c = acue ] && ex="--cue_ans 1"
  echo "$(date '+%F %T') start pac_${c}_s1"; python -B -m boxes.inject_aw $AO $ex --micro $M --stop_at 2000 --ckpt_every 1000 --log_every 100 --device cuda:0 --out boxes/runs_fsq/pac_${c}_s1; rc=$?
  echo "$(date '+%F %T') rc $rc pac_${c}_s1"; [ $rc = 9 ] || { echo "$(date '+%F %T') PILOT STOP (rc $rc)"; exit 3; }
done
echo "$(date '+%F %T') PILOT DONE"
