#!/usr/bin/env bash
# Deploy the remaining PROTOCOL8 latent-arm seeds as good >= 22 GB offers appear (every 10 min, until all are placed
# or 18:00). Each config is deployed at most once; a BAD HOST is destroyed (vdestroy.sh) and its config re-queued.
#   bash scratch/p8c_fill.sh p8c_coco_s3 p8c_at_s3 ...
cd "$(dirname "$0")/.."
V=C:/Users/peter/.local/vastai/vast.sh; export PYTHONIOENCODING=utf-8
S=C:/Users/peter/AppData/Local/Temp/claude/D--/6af9c751-a423-5604-9170-0c177c13d67f/scratchpad
todo=("$@"); used=""
while [ ${#todo[@]} -gt 0 ] && [ "$(date +%H%M)" -lt 1800 ]; do
  mapfile -t offers < <(bash $V search offers 'num_gpus=1 gpu_ram>=22 reliability>0.97 dph<=0.34 disk_space>=40 rentable=true inet_down>=50 compute_cap>=800' -o dph --raw --limit 300 2>/dev/null | python -c "
import json,sys
for o in json.load(sys.stdin):
  if str(o.get('driver_version','')).startswith(('580','590','595','535')) or any(b in o['gpu_name'] for b in ('PRO','5070','5080','5090','5060')): continue
  print(o['id'])")
  for o in "${offers[@]}"; do
    [ ${#todo[@]} -eq 0 ] && break
    echo " $used " | grep -q " $o " && continue
    used="$used $o"; cfg=${todo[0]}
    bash prosqa/vdeploy.sh $o ${cfg//_/-} pod_p8c.sh $cfg > $S/dep_fill_$cfg.out 2>&1
    if grep -q STARTED $S/dep_fill_$cfg.out; then
      echo "$(date +%T) placed $cfg on offer $o: $(grep -o 'INSTANCE [0-9]* [0-9.]* [0-9]*' $S/dep_fill_$cfg.out)"; todo=("${todo[@]:1}")
    else
      id=$(grep -o 'INSTANCE [0-9]*' $S/dep_fill_$cfg.out | awk '{print $2}')
      [ -n "$id" ] && bash scratch/vdestroy.sh $id >/dev/null && echo "$id vast | $cfg failed deploy ($(tail -1 $S/dep_fill_$cfg.out)), DESTROYED_VERIFIED $(date '+%F %H:%M')" >> scratch/pods.txt
      echo "$(date +%T) $cfg failed on offer $o, retrying later"
    fi
  done
  [ ${#todo[@]} -gt 0 ] && sleep 600
done
echo "$(date +%T) FILL_DONE remaining: ${todo[*]}"
