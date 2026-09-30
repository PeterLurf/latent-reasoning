#!/usr/bin/env bash
# Parallel theory-host watcher (replaces watch_hosts2.sh, which polled serially and fell behind at ~50 hosts).
#   bash theory/watch_hosts3.sh <de_already_reported 0|1>
# Every ~3 min, checks all live hosts in theory/hosts_theory.txt in parallel (skips "<tag> <n>" lines in hosts_done.txt):
#   * QUEUE_DONE  -> in the background: verified fetch (vfetch_arch.sh, per-host lock), destroy -y, confirm the id is absent
#                    from a successful `show instances`, log to theory/vast_log.txt, append "<tag> <D/E count>" to hosts_done
#                    (a failed fetch or unconfirmed destroy is logged and the host is retried next cycle)
#   * GPU <= 5 % on 3 samples, 2 consecutive polls, queue not done -> exit IDLE <tag>
#   * unreachable 3 consecutive polls -> exit UNREACHABLE <tag>
#   * all 56 Pilot D/E runs evaluated (first time) -> exit PILOT_DE_DONE
cd "$(dirname "$0")/.."
de_done=$1
V="C:/Users/peter/.local/vastai/vast.sh"
KEY=~/.ssh/runpod_ed25519
S=/tmp/wh3; mkdir -p $S; touch theory/hosts_done.txt
declare -A idle unreach
drain() {   # tag h p id n
  local tag=$1 h=$2 p=$3 id=$4 n=$5
  mkdir $S/drain_$tag.lock 2>/dev/null || return 0
  r=$(timeout 1800 bash theory/vfetch_arch.sh $h $p 2>&1 | tail -1)
  if [[ "$r" == fetched,\ sha\ ok* ]] || [[ "$r" == "NOTHING TO FETCH" ]]; then
    echo "$id $tag - - $h:$p fetched $(date '+%Y-%m-%d %H:%M') ($r)" >> theory/vast_log.txt
    d=$(timeout 120 bash "$V" destroy instance $id -y 2>&1 | tail -1)
    for try in 1 2 3 4 5 6; do
      sleep 20
      lst=$(timeout 60 bash "$V" show instances --raw 2>/dev/null | python -c "import json,sys
try: print('OK', ' '.join(str(i['id']) for i in json.load(sys.stdin)))
except Exception: print('ERR')")
      if [[ "$lst" == OK* ]] && [[ " $lst " != *" $id "* ]]; then
        echo "$id $tag - - $h:$p destroyed $(date '+%Y-%m-%d %H:%M') after queue end + verified fetch, confirmed absent ($d)" >> theory/vast_log.txt
        grep -q "^$tag " theory/hosts_done.txt || echo "$tag ${n:-0}" >> theory/hosts_done.txt
        rmdir $S/drain_$tag.lock; return 0
      fi
    done
    echo "$id $tag - - $h:$p DESTROY NOT CONFIRMED $(date '+%Y-%m-%d %H:%M') ($d); will retry" >> theory/vast_log.txt
  else
    echo "$id $tag - - $h:$p FETCH FAILED $(date '+%Y-%m-%d %H:%M') ($r); will retry" >> theory/vast_log.txt
  fi
  rmdir $S/drain_$tag.lock
}
while :; do
  rm -f $S/poll_*
  live=0
  while read -r h p tag id; do
    [ -z "$h" ] && continue; case "$h" in \#*) continue;; esac
    grep -q "^$tag " theory/hosts_done.txt && continue
    live=$((live + 1))
    ( timeout 60 ssh -n -o ConnectTimeout=15 -i $KEY -p $p root@$h \
      "grep -c 'evaluated p[de][38_]' /workspace/Latent/logs/queue_$tag.out; grep -c QUEUE_DONE /workspace/Latent/logs/queue_$tag.out; for i in 1 2 3; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; sleep 5; done | sort -n | tail -1" \
      2>/dev/null | grep -v "Welcome\|Have fun" | tr '\n' ' ' > $S/poll_$tag ) &
  done < theory/hosts_theory.txt
  wait
  [ "$live" -eq 0 ] && { echo ALL_DONE; date; exit 0; }
  total=$(awk '{s+=$2} END {print s+0}' theory/hosts_done.txt)
  while read -r h p tag id; do
    [ -z "$h" ] && continue; case "$h" in \#*) continue;; esac
    grep -q "^$tag " theory/hosts_done.txt && continue
    read -r n qd util < $S/poll_$tag 2>/dev/null
    if [ -z "$util" ]; then
      unreach[$tag]=$(( ${unreach[$tag]:-0} + 1 ))
      [ ${unreach[$tag]} -ge 3 ] && { echo "UNREACHABLE $tag $h $p $id"; date; exit 0; }
      continue
    fi
    unreach[$tag]=0
    total=$((total + ${n:-0}))
    if [ "${qd:-0}" -ge 1 ]; then
      drain $tag $h $p $id ${n:-0} &
      continue
    fi
    if [ "${util:-0}" -le 5 ]; then
      idle[$tag]=$(( ${idle[$tag]:-0} + 1 ))
      [ ${idle[$tag]} -ge 2 ] && { echo "IDLE $tag $h $p $id util=$util"; date; exit 0; }
    else
      idle[$tag]=0
    fi
  done < theory/hosts_theory.txt
  if [ "$de_done" = 0 ] && [ "$total" -ge 56 ]; then echo "PILOT_DE_DONE ($total)"; date; exit 0; fi
  sleep 150
done
