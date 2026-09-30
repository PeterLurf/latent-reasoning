#!/usr/bin/env bash
# GPU idle watcher (user request 2026-09-29 16:15: "check gpus every few mins"). Every 5 min, samples utilisation 3x
# (5 s apart) on each Vast host and on each local card. A GPU whose max sample is < 15 % on 2 consecutive polls
# (~10 min idle) or a host unreachable on 3 polls makes the script print the reason and exit (waking the session).
#   bash scratch/gpu_watch.sh "<host> <port> <label>" ...     (local cards are always checked)
declare -A idle fails
PODS=("$@")
while :; do
  for hp in "${PODS[@]}"; do
    read -r h p l <<<"$hp"
    grep -qx "$h:$p" logs/drained_hosts.txt 2>/dev/null && continue      # drained by scratch/auto_drain.sh
    u=$(timeout 40 ssh -o ConnectTimeout=15 -i ~/.ssh/runpod_ed25519 -p $p root@$h "for i in 1 2 3; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; sleep 5; done | sort -n | tail -1" 2>/dev/null | tr -dc '0-9')
    if [ -z "$u" ]; then fails[$l]=$(( ${fails[$l]:-0} + 1 )); [ ${fails[$l]} -ge 3 ] && { echo "$(date +%T) $l UNREACHABLE ($h:$p)"; exit 0; }; continue; fi
    fails[$l]=0
    if [ "$u" -lt 15 ]; then idle[$l]=$(( ${idle[$l]:-0} + 1 )); [ ${idle[$l]} -ge 2 ] && { echo "$(date +%T) $l IDLE (max util $u%) ($h:$p)"; exit 0; }; else idle[$l]=0; fi
  done
  mapfile -t loc < <(for i in 1 2 3; do nvidia-smi --query-gpu=name,utilization.gpu --format=csv,noheader,nounits; sleep 5; done)
  for name in "RTX 2070" "RTX 4060" "RTX 3070"; do
    u=$(printf '%s\n' "${loc[@]}" | grep "$name" | awk -F', ' '{print $2}' | sort -n | tail -1)
    k="local_${name// /}"
    if [ -n "$u" ] && [ "$u" -lt 15 ]; then idle[$k]=$(( ${idle[$k]:-0} + 1 )); [ ${idle[$k]} -ge 2 ] && { echo "$(date +%T) $k IDLE (max util $u%)"; exit 0; }; else idle[$k]=0; fi
  done
  sleep 300
done
