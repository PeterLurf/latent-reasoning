#!/usr/bin/env bash
# Deploy generic theory-run queues to Vast in parallel: bash scratch/tj_deploy.sh <queue prefix, e.g. st> <host index> ...
# Queue file scratch/<prefix>_q_<h>.txt, label <prefix>-<h>, tag <prefix>_<h>; spawns a drain watcher per started host.
cd "$(dirname "$0")/.."
pre=$1; shift
V="C:/Users/peter/.local/vastai/vast.sh"
mapfile -t OFF < <(bash "$V" search offers 'gpu_name in [RTX_3090,RTX_A4000,RTX_A5000,RTX_3080_Ti,RTX_4070,RTX_3080,RTX_4080,RTX_4090] num_gpus=1 reliability>0.98 cuda_vers>=12.4 dph<=0.30 disk_space>=40 verified=true rentable=true' -o dph --raw 2>/dev/null | python -c "
import json,sys; d=json.load(sys.stdin)
for o in d:
    if str(o.get('driver_version')).split('.')[0] not in ('580','590','595','535'): print(o['id'])" | head -20)
echo "$(date +%T) $pre offers: ${OFF[*]}" >> logs/tj_deploy.log
one() {
  h=$1; o=$2
  [ -n "$o" ] || { echo "$(date +%T) $pre host $h: NO OFFER" >> logs/tj_deploy.log; return; }
  out=$(bash scratch/vdeploy_toy.sh $o $pre-$h scratch/${pre}_q_$h.txt ${pre}_$h scratch/tjob_vast.sh 2>&1)
  echo "$(date +%T) $pre host $h offer $o: $(echo "$out" | grep -E 'INSTANCE|NVIDIA|STARTED|BAD HOST|FAILED' | tr '\n' ' ')" >> logs/tj_deploy.log
  if echo "$out" | grep -q STARTED; then read -r _ id hh pp <<<"$(echo "$out" | grep '^INSTANCE')"
    nohup bash scratch/auto_drain_tj.sh "$hh $pp ${pre}_$h $id tj" > logs/auto_drain_${pre}_$h.out 2>&1 &
  else id=$(echo "$out" | grep '^INSTANCE' | awk '{print $2}'); [ -n "$id" ] && bash scratch/vdestroy.sh $id >> logs/tj_deploy.log 2>&1
  fi
}
k=0; for h in "$@"; do one $h "${OFF[$k]}" & k=$((k+1)); sleep 3; done
wait
echo "$(date +%T) $pre DEPLOY_DONE $*" >> logs/tj_deploy.log
