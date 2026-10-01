#!/usr/bin/env bash
# Deploy bench batch (2) hosts in parallel: bash scratch/bench_deploy.sh <h> ... (queue scratch/bench_q2_<h>.txt, files .files)
cd "$(dirname "$0")/.."
V="C:/Users/peter/.local/vastai/vast.sh"
mapfile -t OFF < <(bash "$V" search offers 'gpu_name in [RTX_3090,RTX_A4000,RTX_A5000,RTX_3080_Ti,RTX_4090,RTX_4080,RTX_3080,RTX_4070,RTX_4070_Ti,RTX_3060,RTX_4060_Ti,RTX_A2000,RTX_2080_Ti,Tesla_V100,RTX_A6000,L4] num_gpus=1 reliability>0.97 cuda_vers>=12.4 dph<=0.35 disk_space>=40 rentable=true gpu_ram>=15' -o dph --raw 2>/dev/null | python -c "
import json,sys; d=json.load(sys.stdin)
for o in d:
    if str(o.get('driver_version')).split('.')[0] not in ('535',): print(o['id'])" | head -20)
echo "$(date +%T) bench offers: ${OFF[*]}" >> logs/bench_deploy.log
one() {
  h=$1; o=$2
  [ -n "$o" ] || { echo "$(date +%T) bench host $h: NO OFFER" >> logs/bench_deploy.log; return; }
  out=$(EXTRA_FILES="$(cat scratch/bench_q2_$h.files)" bash scratch/vdeploy_toy.sh $o bench-$h scratch/bench_q2_$h.txt bench_$h scratch/bench_vast_job.sh 2>&1)
  echo "$(date +%T) bench host $h offer $o: $(echo "$out" | grep -E 'INSTANCE|NVIDIA|UPLOAD|STARTED|BAD HOST|FAILED' | tr '\n' ' ')" >> logs/bench_deploy.log
  if echo "$out" | grep -q STARTED; then read -r _ id hh pp <<<"$(echo "$out" | grep '^INSTANCE')"
    nohup bash scratch/auto_drain_bench.sh "$hh $pp bench_$h $id bench" > logs/auto_drain_bench_$h.out 2>&1 &
  else id=$(echo "$out" | grep '^INSTANCE' | awk '{print $2}'); [ -n "$id" ] && bash scratch/vdestroy.sh $id >> logs/bench_deploy.log 2>&1
  fi
}
k=0; for h in "$@"; do one $h "${OFF[$k]}" & k=$((k+1)); sleep 3; done
wait
echo "$(date +%T) bench DEPLOY_DONE $*" >> logs/bench_deploy.log
