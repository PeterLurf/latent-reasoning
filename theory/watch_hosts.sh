#!/usr/bin/env bash
# Wait until (a) all 56 Pilot D/E runs are evaluated across the theory hosts, or (b) some host not yet in the handled
# list reaches QUEUE_DONE. Prints the event and exits (the caller fetches/destroys, then restarts the watch).
#   bash theory/watch_hosts.sh <handled-tags-comma-separated> <de_already_reported 0|1>
cd "$(dirname "$0")/.."
handled=",$1,"; de_done=$2
while :; do
  total=0
  while read -r h p tag id; do
    [ -z "$h" ] && continue; case "$h" in \#*) continue;; esac
    out=$(timeout 30 ssh -n -o ConnectTimeout=15 -i ~/.ssh/runpod_ed25519 -p $p root@$h \
      "grep -c 'evaluated p[de][38_]' /workspace/Latent/logs/queue_$tag.out; grep -c QUEUE_DONE /workspace/Latent/logs/queue_$tag.out" 2>/dev/null | grep -v "Welcome\|Have fun" | tr '\n' ' ')
    read -r n qd <<<"$out"
    total=$((total + ${n:-0}))
    if [ "${qd:-0}" -ge 1 ] && [[ "$handled" != *",$tag,"* ]]; then echo "QUEUE_DONE $tag $h $p $id"; date; exit 0; fi
  done < theory/hosts_theory.txt
  if [ "$de_done" = 0 ] && [ "$total" -ge 56 ]; then echo "PILOT_DE_DONE ($total)"; date; exit 0; fi
  sleep 600
done
