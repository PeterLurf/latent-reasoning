#!/usr/bin/env bash
# Deploy PROTOCOL8-NA jobs to Vast one after another (uploads are bandwidth-bound, ~2.25 MB/s), each with its own checkpoint;
# after each STARTED, spawn a drain watcher for that host (log logs/auto_drain_na_<job>.out). Hosts register in live_hosts.txt.
#   bash scratch/na_deploy_chain.sh <job> ...
cd "$(dirname "$0")/.."
V="C:/Users/peter/.local/vastai/vast.sh"
for j in "$@"; do
  case $j in cot_*) ck=prosqa/ckpt/p8c_${j}_checkpoint_30;; *) ck=prosqa/ckpt/p8c_${j}_checkpoint_48;; esac
  for attempt in 1 2 3; do
    O=$(bash "$V" search offers 'gpu_name in [RTX_3090,RTX_A4000,RTX_A5000,RTX_3080_Ti,RTX_4070] num_gpus=1 reliability>0.99 cuda_vers>=12.4 dph<=0.22 disk_space>=40 verified=true rentable=true inet_down>=500' -o dph --raw 2>/dev/null | python -c "
import json,sys; d=json.load(sys.stdin)
ok=[o for o in d if str(o.get('driver_version')).split('.')[0] not in ('580','590','595','535')]
print(ok[0]['id'] if ok else '')")
    [ -n "$O" ] || { sleep 30; continue; }
    out=$(EXTRA_FILES="$ck" bash scratch/vdeploy_toy.sh $O na-$j scratch/p8na_q_$j.txt na_$j scratch/p8na_vast_job.sh 2>&1)
    echo "$(date +%T) $j offer $O: $(echo "$out" | grep -E 'INSTANCE|UPLOAD|STARTED|BAD HOST|FAILED' | tr '\n' ' ')" >> logs/na_deploy_chain.log
    if echo "$out" | grep -q STARTED; then
      read -r _ id h p <<<"$(echo "$out" | grep '^INSTANCE')"
      (nohup bash scratch/auto_drain_na.sh "$h $p na_$j $id na" > logs/auto_drain_na_$j.out 2>&1 &)
      break
    fi
    id=$(echo "$out" | grep '^INSTANCE' | awk '{print $2}'); [ -n "$id" ] && bash scratch/vdestroy.sh $id >> logs/na_deploy_chain.log 2>&1
  done
done
echo "$(date +%T) CHAIN_DONE" >> logs/na_deploy_chain.log
