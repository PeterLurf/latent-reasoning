#!/usr/bin/env bash
# Deploy §65 Vast hosts: bash scratch/st65_deploy.sh <h> ... (queue scratch/st65_q_<h>.txt with parent files .files)
cd "$(dirname "$0")/.."
V="C:/Users/peter/.local/vastai/vast.sh"
mapfile -t OFF < <(bash "$V" search offers 'gpu_name in [RTX_3090,RTX_A5000,RTX_4090,RTX_A4000,RTX_3080_Ti,RTX_4080,RTX_4070,RTX_A6000] num_gpus=1 reliability>0.95 cuda_vers>=12.4 dph<=0.40 disk_space>=40 rentable=true' -o dph --raw 2>/dev/null | python -c "
import json,sys; d=json.load(sys.stdin)
for o in d:
    if str(o.get('driver_version')).split('.')[0] not in ('535',): print(o['id'])" | grep -v -x -F "${SKIP_OFFER:-none}" | head -20)   # SKIP_OFFER: an offer known bad
one() {
  h=$1; o=$2; [ -n "$o" ] || { echo "$(date +%T) st65 host $h: NO OFFER" >> logs/tj_deploy.log; return; }
  out=$(EXTRA_FILES="$(cat scratch/st65_q_$h.files)" bash scratch/vdeploy_toy.sh $o st65-$h scratch/st65_q_$h.txt st65_$h ${WORKER:-scratch/tjob_vast2.sh} 2>&1)
  echo "$(date +%T) st65 host $h offer $o: $(echo "$out" | grep -E 'INSTANCE|NVIDIA|STARTED|BAD HOST|FAILED' | tr '\n' ' ')" >> logs/tj_deploy.log
  if echo "$out" | grep -q STARTED; then read -r _ id hh pp <<<"$(echo "$out" | grep '^INSTANCE')"
    nohup bash scratch/auto_drain_tj.sh "$hh $pp st65_$h $id tj" > logs/auto_drain_st65_$h.out 2>&1 &
  else id=$(echo "$out" | grep '^INSTANCE' | awk '{print $2}'); [ -n "$id" ] && bash scratch/vdestroy.sh $id >> logs/tj_deploy.log 2>&1; fi
}
k=0; for h in "$@"; do one $h "${OFF[$k]}" & k=$((k+1)); sleep 3; done; wait
echo "$(date +%T) st65 DEPLOY_DONE $*" >> logs/tj_deploy.log
