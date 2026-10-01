#!/usr/bin/env bash
# Deploy the 10 §63 F4 hosts in parallel; each spawns its drain watcher (logs/auto_drain_f4_<h>.out) after STARTED.
cd "$(dirname "$0")/.."
V="C:/Users/peter/.local/vastai/vast.sh"
mapfile -t OFF < <(bash "$V" search offers 'gpu_name in [RTX_3090,RTX_A4000,RTX_A5000,RTX_3080_Ti,RTX_4070,RTX_3080,RTX_4080] num_gpus=1 reliability>0.99 cuda_vers>=12.4 dph<=0.22 disk_space>=40 verified=true rentable=true' -o dph --raw 2>/dev/null | python -c "
import json,sys; d=json.load(sys.stdin)
for o in d:
    if str(o.get('driver_version')).split('.')[0] not in ('580','590','595','535'): print(o['id'])" | head -14)
one() {
  h=$1; o=$2
  files=""; for md in $(grep -v '^#' scratch/f4_q_$h.txt | awk '{print $1}'); do for f in config.json arch.json model.pt log.jsonl; do [ -f $md/$f ] && files="$files $md/$f"; done; done
  out=$(EXTRA_FILES="$files" bash scratch/vdeploy_toy.sh $o f4-$h scratch/f4_q_$h.txt f4_$h scratch/f4_vast_job.sh 2>&1)
  echo "$(date +%T) host $h offer $o: $(echo "$out" | grep -E 'INSTANCE|NVIDIA|STARTED|BAD HOST|FAILED' | tr '\n' ' ')" >> logs/f4_deploy.log
  if echo "$out" | grep -q STARTED; then read -r _ id hh pp <<<"$(echo "$out" | grep '^INSTANCE')"
    nohup bash scratch/auto_drain_f4.sh "$hh $pp f4_$h $id f4" > logs/auto_drain_f4_$h.out 2>&1 &
  fi
}
for h in 0 1 2 3 4 5 6 7 8 9; do one $h ${OFF[$h]} & sleep 3; done
wait
echo "$(date +%T) DEPLOY_ALL_DONE" >> logs/f4_deploy.log
