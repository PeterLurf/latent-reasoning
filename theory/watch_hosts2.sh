#!/usr/bin/env bash
# Theory-host watcher (every 5 min), per the user's instruction to catch idle GPUs overnight.
#   bash theory/watch_hosts2.sh <de_already_reported 0|1>
# Per host in theory/hosts_theory.txt (skipping hosts already destroyed: theory/hosts_done.txt lines "<tag> <D/E count>"):
#   * queue finished (QUEUE_DONE in its log): fetch (theory/vfetch_arch.sh, checksum-verified) and, only if the fetch
#     succeeded, destroy the instance; log both in theory/vast_log.txt; continue watching.
#   * GPU utilisation <= 5 % in all 3 samples on 2 consecutive polls (> 10 min) with the queue not finished: exit IDLE.
#   * unreachable, or Vast status not "running", on 3 consecutive polls (> 15 min): exit UNREACHABLE.
#   * all 56 Pilot D/E runs evaluated (first time): exit PILOT_DE_DONE.
# Exits ALL_DONE when every host is destroyed.
cd "$(dirname "$0")/.."
de_done=$1
V="C:/Users/peter/AppData/Local/Temp/claude/D--/6af9c751-a423-5604-9170-0c177c13d67f/scratchpad/vast.sh"
KEY=~/.ssh/runpod_ed25519
touch theory/hosts_done.txt
declare -A idle unreach
log() { echo "$1" >> theory/vast_log.txt; echo "$1"; }
while :; do
  status=$(timeout 60 bash "$V" show instances --raw 2>/dev/null | python -c "import json,sys
try:
  [print(i['id'], i.get('actual_status')) for i in json.load(sys.stdin)]
except Exception: pass")
  total=0; live=0
  while read -r h p tag id; do
    [ -z "$h" ] && continue; case "$h" in \#*) continue;; esac
    grep -q "^$tag " theory/hosts_done.txt && continue
    live=$((live + 1))
    out=$(timeout 60 ssh -n -o ConnectTimeout=15 -i $KEY -p $p root@$h \
      "grep -c 'evaluated p[de][38_]' /workspace/Latent/logs/queue_$tag.out; grep -c QUEUE_DONE /workspace/Latent/logs/queue_$tag.out; for i in 1 2 3; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; sleep 5; done | sort -n | tail -1" \
      2>/dev/null | grep -v "Welcome\|Have fun" | tr '\n' ' ')
    read -r n qd util <<<"$out"
    st=$(echo "$status" | awk -v id=$id '$1==id {print $2}')
    if [ -z "$util" ] || { [ -n "$status" ] && [ "$st" != "running" ]; }; then
      unreach[$tag]=$(( ${unreach[$tag]:-0} + 1 ))
      [ ${unreach[$tag]} -ge 3 ] && { echo "UNREACHABLE $tag $h $p $id status=$st"; date; exit 0; }
      continue
    fi
    unreach[$tag]=0
    total=$((total + ${n:-0}))
    if [ "${qd:-0}" -ge 1 ]; then
      NOW=$(date '+%Y-%m-%d %H:%M')
      if r=$(timeout 1800 bash theory/vfetch_arch.sh $h $p 2>&1 | tail -1) && { [[ "$r" == fetched,\ sha\ ok* ]] || [[ "$r" == "NOTHING TO FETCH" ]]; }; then
        log "$id $tag - - $h:$p fetched $NOW ($r)"
        d=$(timeout 120 bash "$V" destroy instance $id -y 2>&1 | tail -1)
        log "$id $tag - - $h:$p destroyed $(date '+%Y-%m-%d %H:%M') after queue end + verified fetch ($d)"
        echo "$tag ${n:-0}" >> theory/hosts_done.txt
      else
        echo "FETCH_FAILED $tag $h $p $id: $r"; date; exit 0
      fi
      continue
    fi
    if [ "${util:-0}" -le 5 ]; then
      idle[$tag]=$(( ${idle[$tag]:-0} + 1 ))
      [ ${idle[$tag]} -ge 2 ] && { echo "IDLE $tag $h $p $id util=$util"; date; exit 0; }
    else
      idle[$tag]=0
    fi
  done < theory/hosts_theory.txt
  total=$((total + $(awk '{s+=$2} END {print s+0}' theory/hosts_done.txt)))
  [ "$live" -eq 0 ] && { echo ALL_DONE; date; exit 0; }
  if [ "$de_done" = 0 ] && [ "$total" -ge 56 ]; then echo "PILOT_DE_DONE ($total)"; date; exit 0; fi
  sleep 180
done
