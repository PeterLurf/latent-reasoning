#!/usr/bin/env bash
# GPU idle watcher, registry version (Audit 2026-10-01: new hosts must be covered from their create step). Every 5 min it
# re-reads scratch/live_hosts.txt ("<host> <port> <label>", appended by the deploy scripts at create) plus any hosts given
# as arguments, skips hosts in logs/drained_hosts.txt and hosts whose job has finished (QUEUE_DONE / ANALYSIS_DONE: they
# await their drain), samples GPU utilisation 3x on each remaining host and on each local card, and prints + exits on a GPU
# < 15 % on 2 consecutive polls or a host unreachable on 3 polls.
#   [GPU_WATCH_SKIP="RTX 2070,RTX 3070"] bash scratch/gpu_watch2.sh ["<host> <port> <label>" ...]
cd "$(dirname "$0")/.."
declare -A idle fails
while :; do
  mapfile -t PODS < <( { printf '%s\n' "$@"; grep -v '^#' scratch/live_hosts.txt 2>/dev/null; } | awk 'NF>=3 && !seen[$1":"$2]++')
  for hp in "${PODS[@]}"; do
    read -r h p l <<<"$hp"
    grep -qx "$h:$p" logs/drained_hosts.txt 2>/dev/null && continue
    out=$(timeout 40 ssh -o ConnectTimeout=15 -i ~/.ssh/runpod_ed25519 -p $p root@$h "cat /workspace/Latent/logs/queue_*.out /workspace/logs/prosqa.out 2>/dev/null | grep -c -E 'QUEUE_DONE|ANALYSIS_DONE'; for i in 1 2 3; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; sleep 5; done | sort -n | tail -1" 2>/dev/null)
    if [ -z "$out" ]; then fails[$l]=$(( ${fails[$l]:-0} + 1 )); [ ${fails[$l]} -eq 1 ] && bash scratch/rebuild_live_hosts.sh > /dev/null 2>&1   # 2026-10-10 (Audit): an instance can change IP mid-run (b237, b254) -> re-read the registry from 'vast show instances'
      [ ${fails[$l]} -ge 3 ] && { echo "$(date +%T) $l UNREACHABLE ($h:$p)"; exit 0; }; continue; fi
    fails[$l]=0
    done_=$(echo "$out" | sed -n 1p | tr -dc '0-9'); u=$(echo "$out" | sed -n 2p | tr -dc '0-9')
    [ "${done_:-0}" -ge 1 ] && { idle[$l]=0; continue; }                  # finished, awaiting its drain watcher
    if [ -n "$u" ] && [ "$u" -lt 15 ]; then idle[$l]=$(( ${idle[$l]:-0} + 1 )); [ ${idle[$l]} -ge 2 ] && { echo "$(date +%T) $l IDLE (max util $u%) ($h:$p)"; exit 0; }; else idle[$l]=0; fi
  done
  mapfile -t loc < <(for i in 1 2 3; do nvidia-smi --query-gpu=name,utilization.gpu --format=csv,noheader,nounits; sleep 5; done)
  for name in "RTX 2070" "RTX 4060" "RTX 3070"; do
    case ",${GPU_WATCH_SKIP:-}," in *",$name,"*) continue;; esac
    u=$(printf '%s\n' "${loc[@]}" | grep "$name" | awk -F', ' '{print $2}' | sort -n | tail -1)
    k="local_${name// /}"
    if [ -n "$u" ] && [ "$u" -lt 15 ]; then idle[$k]=$(( ${idle[$k]:-0} + 1 )); [ ${idle[$k]} -ge 2 ] && { echo "$(date +%T) $k IDLE (max util $u%)"; exit 0; }; else idle[$k]=0; fi
  done
  sleep 300
done
