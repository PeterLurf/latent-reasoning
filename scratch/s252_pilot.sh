#!/usr/bin/env bash
# §252 UNSCAFFOLDED $0 local PILOT (NS_v1 §3; Note 252-1): rcb_plaw4_s1's recipe (gate_s251.RCB = gen_q_b354's flags) warm from rcb_plaw4_s1/model.pt, seed 1, the 16k
# schedule, --stop_at 2000 (ckpt at 1k / 2k, exit 9, no eval), both cells on ONE card (the 4060 = cuda:0), serialised on the card lock:
#   pilot_np_s1 : --perm_p 0.5;   pilot_gp_s1 : --perm_p 0.5 --cue generic.
# micro: per arm the smallest of 4 / 8 / 16 / 32 whose --mem_check train peak <= 6.8 GiB; both cells run at the LARGER of the two (recorded with the GPU name in
# scratch/s252_pilot/micro.json). Theory reads log.jsonl ws_readout_perm["2"]["1"] (the step-1000 / step-2000 bin records; theory/score_ns_pilot.py).
#   nohup bash scratch/s252_pilot.sh >> logs/s252_pilot.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
L=scratch/_tmp/s240_land.lock
until mkdir $L 2>/dev/null; do sleep 30; done
trap 'rmdir $L' EXIT
RCB="--arm inject --think_k 8 --aux_step 1 --aux_fade 48000,48000 --aux_reweight --fast_add --cache_lower --d_max 4 --steps 16000 --cue step --write_src answer --init_trunk boxes/runs_fsq/s233_askname_s1/model.pt --init_sha c00f7ec0bd7977fe --write_mix_start 0 --write_mix_end 0 --mul_s_p 0 --init_model boxes/runs_fsq/rcb_plaw4_s1/model.pt --init_model_sha 67775e2492044a8a"
mkdir -p scratch/s252_pilot; M=0
for arm in np gp; do
  ex="--perm_p 0.5"; [ $arm = gp ] && ex="--perm_p 0.5 --cue generic"; Ma=""
  for mi in 4 8 16 32; do
    d=scratch/_tmp/s252/memck_${arm}_$mi; rm -rf $d; mkdir -p $d
    python -B -m boxes.inject_aw $RCB $ex --micro $mi --eval_bs 50 --mem_check --device cuda:0 --out $d > $d/stdout.log 2>&1
    pk=$(python -c "import json; print(json.load(open('$d/mem_check.json'))['peak_gib_train'])" 2>/dev/null); echo "$(date '+%F %T') mem_check $arm micro $mi train peak ${pk:-none} GiB"
    if [ -n "$pk" ] && python -c "import sys; sys.exit(0 if float('$pk') <= 6.8 else 1)"; then Ma=$mi; break; fi
  done
  [ -n "$Ma" ] || { echo "$(date '+%F %T') PILOT STOP: no micro fits 6.8 GiB ($arm)"; exit 3; }
  [ "$Ma" -gt "$M" ] && M=$Ma
done
gpu=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
echo "{\"micro\": $M, \"gpu\": \"$gpu\", \"budget_gib\": 6.8}" > scratch/s252_pilot/micro.json; echo "$(date '+%F %T') micro $M on $gpu"
for arm in np gp; do
  ex="--perm_p 0.5"; [ $arm = gp ] && ex="--perm_p 0.5 --cue generic"
  echo "$(date '+%F %T') start pilot_${arm}_s1"; python -B -m boxes.inject_aw $RCB $ex --micro $M --stop_at 2000 --ckpt_every 1000 --log_every 100 --device cuda:0 --out boxes/runs_fsq/pilot_${arm}_s1; rc=$?
  echo "$(date '+%F %T') rc $rc pilot_${arm}_s1"; [ $rc = 9 ] || { echo "$(date '+%F %T') PILOT STOP (rc $rc)"; exit 3; }
done
echo "$(date '+%F %T') PILOT DONE"
