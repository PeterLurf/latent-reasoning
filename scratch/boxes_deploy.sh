#!/usr/bin/env bash
# Deploy §66 panel hosts in parallel: bash scratch/boxes_deploy.sh <h> ... (queue scratch/boxes_q_W<h>.txt; >= 24 GB GPUs,
# drivers 550-570); each started host gets its drain watcher (auto_drain_boxes.sh).
cd "$(dirname "$0")/.."
V="C:/Users/peter/.local/vastai/vast.sh"
mapfile -t OFF < <(bash "$V" search offers 'gpu_name in [RTX_3090,RTX_A5000,RTX_4090,RTX_3090_Ti,RTX_A6000] num_gpus=1 reliability>0.97 cuda_vers>=12.4 dph<=0.40 disk_space>=40 rentable=true' -o dph --raw 2>/dev/null | python -c "
import json,sys; d=json.load(sys.stdin)
for o in d:
    if str(o.get('driver_version')).split('.')[0] in ('550','555','560','565','570'): print(o['id'])" | head -25)
echo "$(date +%T) boxes offers: ${OFF[*]}" >> logs/boxes_deploy.log
one() {
  h=$1; o=$2
  [ -n "$o" ] || { echo "$(date +%T) boxes host W$h: NO OFFER" >> logs/boxes_deploy.log; return; }
  out=$(bash scratch/vdeploy_toy.sh $o boxes-W$h scratch/boxes_q_W$h.txt boxes_W$h scratch/boxes_vast_job.sh 2>&1)
  echo "$(date +%T) boxes host W$h offer $o: $(echo "$out" | grep -E 'INSTANCE|NVIDIA|STARTED|BAD HOST|FAILED' | tr '\n' ' ')" >> logs/boxes_deploy.log
  if echo "$out" | grep -q STARTED; then read -r _ id hh pp <<<"$(echo "$out" | grep '^INSTANCE')"
    nohup bash scratch/auto_drain_boxes.sh "$hh $pp boxes_W$h $id boxes" > logs/auto_drain_boxes_W$h.out 2>&1 &
  else id=$(echo "$out" | grep '^INSTANCE' | awk '{print $2}'); [ -n "$id" ] && bash scratch/vdestroy.sh $id >> logs/boxes_deploy.log 2>&1
  fi
}
k=0; for h in "$@"; do one $h "${OFF[$k]}" & k=$((k+1)); sleep 3; done
wait
echo "$(date +%T) boxes DEPLOY_DONE $*" >> logs/boxes_deploy.log
