#!/usr/bin/env bash
# Toy-queue Vast host watcher: exits (waking the session) when a host's queue log shows QUEUE_DONE, a run fails or has no model
# ("rc [1-9]" or FAIL), the log stops growing for 3 h (one run takes ≤ ~2 h), or SSH fails 3 times in a row.
#   bash scratch/t9_watch.sh "<host> <port> <tag>" ...      (tag = queue tag, e.g. t9_vA)
declare -A size stale fails
PODS=("$@")
while :; do
  for hp in "${PODS[@]}"; do
    read -r h p t <<<"$hp"
    out=$(timeout 40 ssh -o ConnectTimeout=15 -i ~/.ssh/runpod_ed25519 -p $p root@$h "cd /workspace/Latent; wc -c < logs/queue_$t.out; grep -c QUEUE_DONE logs/queue_$t.out; grep -E 'rc [1-9]|FAIL|no model' logs/queue_$t.out | head -2" 2>/dev/null)
    if [ -z "$out" ]; then fails[$t]=$(( ${fails[$t]:-0} + 1 )); [ ${fails[$t]} -ge 3 ] && { echo "$(date +%T) $t UNREACHABLE"; exit 0; }; continue; fi
    fails[$t]=0
    sz=$(echo "$out" | sed -n 1p); dn=$(echo "$out" | sed -n 2p); err=$(echo "$out" | sed -n '3,$p')
    [ -n "$err" ] && { echo "$(date +%T) $t ERROR: $err"; exit 0; }
    [ "$dn" -ge 1 ] 2>/dev/null && { echo "$(date +%T) $t QUEUE_DONE $h $p"; exit 0; }
    if [ "$sz" = "${size[$t]}" ]; then stale[$t]=$(( ${stale[$t]:-0} + 1 )); [ ${stale[$t]} -ge 18 ] && { echo "$(date +%T) $t STALLED"; exit 0; }
    else stale[$t]=0; size[$t]=$sz; fi
  done
  sleep 600
done
