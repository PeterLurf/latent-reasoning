#!/usr/bin/env bash
# Toy-queue Vast watcher with automatic, verified draining.
#   bash scratch/auto_drain.sh "<host> <port> <queue tag> <instance id> <fetch prefix t9|p9>" ...
# Every 10 min, for each host still in the list:
#   - QUEUE_DONE: run scratch/drain_toy.sh. On DRAINED, record host:port in logs/drained_hosts.txt (scratch/gpu_watch.sh
#     skips those), log it in logs/auto_drain.log, and drop the host from the list. Otherwise print the drain output
#     and exit, waking the session.
#   - A failure line ("rc [1-9]", FAIL, "no model") that is new since the watcher started: print it and exit.
#   - Unreachable on 3 polls in a row, or the log unchanged for 3 h: print and exit.
# Exits with ALL_DRAINED when the list is empty.
cd "$(dirname "$0")/.."
declare -A size stale fails base
PODS=("$@")
while :; do
  left=()
  for hp in "${PODS[@]}"; do
    read -r h p t id pre <<<"$hp"
    out=$(timeout 40 ssh -o ConnectTimeout=15 -i ~/.ssh/runpod_ed25519 -p $p root@$h "cd /workspace/Latent; wc -c < logs/queue_$t.out; tail -n 1 logs/queue_$t.out | grep -c QUEUE_DONE; grep -E 'rc [1-9]|FAIL|no model' logs/queue_$t.out | tail -2" 2>/dev/null)
    if [ -z "$out" ]; then
      fails[$t]=$(( ${fails[$t]:-0} + 1 )); [ ${fails[$t]} -ge 3 ] && { echo "$(date +%T) $t UNREACHABLE"; exit 0; }
      left+=("$hp"); continue
    fi
    fails[$t]=0
    sz=$(echo "$out" | sed -n 1p); dn=$(echo "$out" | sed -n 2p); err=$(echo "$out" | sed -n '3,$p')
    [ -z "${base[$t]+x}" ] && base[$t]="$err"
    [ -n "$err" ] && [ "$err" != "${base[$t]}" ] && { echo "$(date +%T) $t NEW FAILURE LINE: $err"; exit 0; }
    if [ "$dn" -ge 1 ] 2>/dev/null; then
      res=$(bash scratch/drain_toy.sh "$h" "$p" "$pre" "$id" "$t" 2>&1)
      if echo "$res" | grep -q "^DRAINED $id"; then
        echo "$h:$p" >> logs/drained_hosts.txt
        echo "$(date '+%F %T') $t $id $(echo "$res" | grep '^verified')" >> logs/auto_drain.log
        continue
      fi
      echo "$(date +%T) $t DRAIN FAILED:"; echo "$res" | tail -5; exit 0
    fi
    if [ "$sz" = "${size[$t]}" ]; then stale[$t]=$(( ${stale[$t]:-0} + 1 )); [ ${stale[$t]} -ge 18 ] && { echo "$(date +%T) $t STALLED"; exit 0; }
    else stale[$t]=0; size[$t]=$sz; fi
    left+=("$hp")
  done
  PODS=("${left[@]}")
  [ ${#PODS[@]} -eq 0 ] && { echo "$(date +%T) ALL_DRAINED"; exit 0; }
  sleep 600
done
