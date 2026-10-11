#!/usr/bin/env bash
# gen8 (2026-10-10, Audit): gen7 + on an empty poll, re-read the instance's public_ipaddr / 22/tcp HostPort from 'vast show instance <id>' and, if it changed
# (b237, b254 changed IP mid-run), switch to the new address (logged ADDRESS CHANGED) before counting the poll as a failure. GEN8_ONESHOT=1: one pass, print the pod list (tests).
# gen7 (2026-10-06): gen6 with the DRAINED match unanchored (a carriage return before it made e31 / d8 / a4 log "DRAIN FAILED" after a good drain)
# gen6 (2026-10-06) = gen5 + boxes/runs_fsq counted as activity + an unreachable host keeps being polled (logged at 3 fails, exit at 18 = 3 h).
# Toy-queue Vast watcher with automatic, verified draining (gen5 = gen4 + a NOT_FINISHED drain result (a python job still running after QUEUE_DONE) keeps polling instead of exiting; gen4 = gen3, but a new failure line is logged and polling continues (no exit), and QUEUE_DONE drains with scratch/drain_gen_partial.sh (pre 'gen'), which fetches the good runs, records failed ones and still destroys; activity = queue log + every run log.jsonl, which training appends every 100 steps; stall window 36 polls = 6 h of no training-log growth).
#   bash scratch/auto_drain.sh "<host> <port> <queue tag> <instance id> <fetch prefix t9|p9|t12|f6>" ...   (t12: drain_t12.sh; f6: drain_f6.sh)
# Every 10 min, for each host still in the list:
#   - QUEUE_DONE: run scratch/drain_toy.sh. On DRAINED, record host:port in logs/drained_hosts.txt (scratch/gpu_watch.sh
#     skips those), log it in logs/auto_drain.log, and drop the host from the list. Otherwise print the drain output
#     and exit, waking the session.
#   - A failure line ("rc [1-9]", FAIL, "no model") that is new since the watcher started: print it and exit, unless it is
#     "no model for <r>" and <r>'s guard log ends "relaunch budget spent" (a genuine stage-1 outcome, logged, not alarmed).
#   - Unreachable on 3 polls in a row, or the log unchanged for 3 h: print and exit.
# Exits with ALL_DRAINED when the list is empty.
cd "$(dirname "$0")/.."
declare -A size stale fails base dfail
PODS=("$@")
while :; do
  left=()
  for hp in "${PODS[@]}"; do
    read -r h p t id pre <<<"$hp"
    out=$(timeout 40 ssh -o ConnectTimeout=15 -i ~/.ssh/runpod_ed25519 -p $p root@$h "cd /workspace/Latent; cat logs/queue_$t.out \$(find c1/runs boxes/runs boxes/runs_s5 boxes/runs_fsq boxes/runs_mult gsmlen/runs -maxdepth 2 -name log.jsonl 2>/dev/null) 2>/dev/null | wc -c; tail -n 1 logs/queue_$t.out | grep -c QUEUE_DONE; grep -E 'rc [1-9]|FAIL|no model' logs/queue_$t.out | tail -2" 2>/dev/null)
    if [ -z "$out" ]; then
      na=$(bash C:/Users/peter/.local/vastai/vast.sh show instance "$id" --raw 2>/dev/null | python -c "import json,sys;d=json.load(sys.stdin);print(str(d.get('public_ipaddr') or '').strip(), d['ports']['22/tcp'][0]['HostPort'])" 2>/dev/null)
      if [ -n "$na" ] && [ "$na" != "$h $p" ]; then
        echo "$(date +%T) $t ADDRESS CHANGED $h:$p -> ${na/ /:} (vast show instance $id)"; read -r h p <<<"$na"; hp="$h $p $t $id $pre"; fails[$t]=0; left+=("$hp"); continue
      fi
      fails[$t]=$(( ${fails[$t]:-0} + 1 )); [ ${fails[$t]} -eq 3 ] && echo "$(date +%T) $t UNREACHABLE (3 polls; still polling)"; [ ${fails[$t]} -ge 18 ] && { echo "$(date +%T) $t UNREACHABLE 3 h: exiting"; exit 0; }
      left+=("$hp"); continue
    fi
    fails[$t]=0
    sz=$(echo "$out" | sed -n 1p); dn=$(echo "$out" | sed -n 2p); err=$(echo "$out" | sed -n '3,$p')
    [ -z "${base[$t]+x}" ] && base[$t]="$err"
    if [ -n "$err" ] && [ "$err" != "${base[$t]}" ]; then
      # a new "no model for <r>" whose guard log says "relaunch budget spent" is a genuine stage-1 outcome: note it, go on
      nm=$(echo "$err" | tail -1 | sed -n 's/^no model for //p')
      gl="/workspace/Latent/logs/guard_$nm.log"
      if [ -n "$nm" ] && timeout 40 ssh -o ConnectTimeout=15 -i ~/.ssh/runpod_ed25519 -p $p root@$h            "tail -n 1 $gl | grep -q 'relaunch budget spent'" 2>/dev/null; then
        echo "$(date '+%F %T') $t stage-1 failure (genuine, relaunch budget spent): $nm" >> logs/auto_drain.log
        base[$t]="$err"
      else
        echo "$(date +%T) $t NEW FAILURE LINE (continuing): $err"; base[$t]="$err"
      fi
    fi
    if [ "$dn" -ge 1 ] 2>/dev/null; then
      if [ "$pre" = t12 ]; then res=$(bash scratch/drain_t12.sh "$h" "$p" "$t" "$id" 2>&1)
      elif [ "$pre" = f6 ]; then res=$(bash scratch/drain_f6.sh "$h" "$p" "$t" "$id" 2>&1)
      elif [ "$pre" = bm ]; then res=$(bash scratch/drain_bm.sh "$h" "$p" "$t" "$id" 2>&1)
      elif [ "$pre" = ail ]; then res=$(bash scratch/drain_ail.sh "$h" "$p" "$t" "$id" 2>&1)
      elif [ "$pre" = na ]; then res=$(bash scratch/drain_p8na.sh "$h" "$p" "$t" "$id" 2>&1)
      elif [ "$pre" = f4 ]; then res=$(bash scratch/drain_f4.sh "$h" "$p" "$t" "$id" 2>&1)
      elif [ "$pre" = c1 ]; then res=$(bash scratch/drain_c1.sh "$h" "$p" "$t" "$id" 2>&1)
      elif [ "$pre" = tj ]; then res=$(bash scratch/drain_tjob.sh "$h" "$p" "$t" "$id" 2>&1)
      elif [ "$pre" = g78 ]; then res=$(bash scratch/drain_g78.sh "$h" "$p" "$t" "$id" 2>&1)
      elif [ "$pre" = gen ]; then res=$(bash scratch/drain_gen_partial.sh "$h" "$p" "$t" "$id" 2>&1)
      else res=$(bash scratch/drain_toy.sh "$h" "$p" "$pre" "$id" "$t" 2>&1); fi
      if echo "$res" | grep -q "DRAINED $id"; then
        echo "$h:$p" >> logs/drained_hosts.txt
        echo "$(date '+%F %T') $t $id $(echo "$res" | grep '^verified')" >> logs/auto_drain.log
        continue
      fi
      if echo "$res" | grep -q "^NOT_FINISHED"; then echo "$(date +%T) $t not finished (a job still running); polling"; left+=("$hp"); continue; fi
      if echo "$res" | grep -q "^FETCH_INCOMPLETE" && [ ${dfail[$t]:-0} -lt 3 ]; then dfail[$t]=$(( ${dfail[$t]:-0} + 1 )); echo "$(date +%T) $t fetch retry ${dfail[$t]}: $(echo "$res" | tail -1)"; left+=("$hp"); continue; fi
      echo "$(date +%T) $t DRAIN FAILED:"; echo "$res" | tail -5; exit 0
    fi
    if [ "$sz" = "${size[$t]}" ]; then stale[$t]=$(( ${stale[$t]:-0} + 1 )); [ ${stale[$t]} -ge 36 ] && { echo "$(date +%T) $t STALLED"; exit 0; }
    else stale[$t]=0; size[$t]=$sz; fi
    left+=("$hp")
  done
  PODS=("${left[@]}")
  [ "${GEN8_ONESHOT:-0}" = 1 ] && { printf 'PODS: %s
' "${PODS[@]}"; exit 0; }
  [ ${#PODS[@]} -eq 0 ] && { echo "$(date +%T) ALL_DRAINED"; exit 0; }
  sleep 600
done
