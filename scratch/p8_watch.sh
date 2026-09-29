#!/usr/bin/env bash
# PROTOCOL8 Phase 0 pod watcher: exits (waking the session) on ANALYSIS_DONE, an error marker, a log that stops growing for
# 30 min, or 3 failed SSH checks in a row. Usage: bash scratch/p8_watch.sh "<host> <port> <label>" ...
declare -A size stale fails
PODS=("$@")
while :; do
  for hp in "${PODS[@]}"; do
    read -r h p l <<<"$hp"
    out=$(timeout 40 ssh -o ConnectTimeout=15 -i ~/.ssh/runpod_ed25519 -p $p root@$h "wc -c < /workspace/logs/prosqa.out; grep -a -c -E 'ANALYSIS_DONE' /workspace/logs/prosqa.out; grep -a -E 'Traceback|FAIL |NO_FINAL|MASK_TEST_FAILED|TRAIN_EXIT [1-9]' /workspace/logs/prosqa.out | head -3" 2>/dev/null)
    if [ -z "$out" ]; then fails[$l]=$(( ${fails[$l]:-0} + 1 )); [ ${fails[$l]} -ge 3 ] && { echo "$(date +%T) $l UNREACHABLE"; exit 0; }; continue; fi
    fails[$l]=0
    sz=$(echo "$out" | sed -n 1p); done_=$(echo "$out" | sed -n 2p); err=$(echo "$out" | sed -n '3,$p')
    [ -n "$err" ] && { echo "$(date +%T) $l ERROR: $err"; exit 0; }
    [ "$done_" -ge 1 ] 2>/dev/null && { echo "$(date +%T) $l ANALYSIS_DONE"; exit 0; }
    if [ "$sz" = "${size[$l]}" ]; then stale[$l]=$(( ${stale[$l]:-0} + 1 )); [ ${stale[$l]} -ge 3 ] && { echo "$(date +%T) $l STALLED (log unchanged 30 min)"; exit 0; }
    else stale[$l]=0; size[$l]=$sz; fi
  done
  sleep 600
done
