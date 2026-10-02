#!/usr/bin/env bash
# Every 60 s: for each live Vast host labelled s71-*, if its queue log ends in QUEUE_DONE and every run line is rc 0,
# run the verified drain (drain_s71.sh: checksummed fetch, then vdestroy) at once; hosts with a failed run are listed in
# logs/fast_drain_s71.log and left for a decision. Exits when no s71 host is left.
cd "$(dirname "$0")/.."
V="C:/Users/peter/.local/vastai/vast.sh"
while :; do
  bash "$V" show instances --raw 2>/dev/null | python -c "
import json,sys
for i in json.load(sys.stdin):
  l=str(i.get('label'))
  if not l.startswith('s71-'): continue
  p=(i.get('ports') or {}).get('22/tcp'); print(i['id'], l, i.get('public_ipaddr'), p[0]['HostPort'] if p else '')" > /tmp/s71_live.txt
  [ -s /tmp/s71_live.txt ] || { echo "$(date +%T) no s71 hosts left" >> logs/fast_drain_s71.log; exit 0; }
  while read -r id lab ip port; do
    [ -n "$port" ] || continue; t=${lab#s71-}
    st=$(timeout 25 ssh -o ConnectTimeout=12 -i ~/.ssh/runpod_ed25519 -p $port root@$ip "f=/workspace/Latent/logs/queue_s71_$t.out; tail -1 \$f | grep -c QUEUE_DONE; grep -c -E 'rc [1-9]' \$f" 2>/dev/null | tr '\n' ' ')
    set -- $st
    if [ "${1:-0}" = 1 ] && [ "${2:-1}" = 0 ]; then r=$(bash scratch/drain_s71.sh $ip $port s71_$t $id 2>&1 | tail -1); echo "$(date +%T) $lab $r" >> logs/fast_drain_s71.log
    elif [ "${1:-0}" = 1 ]; then grep -q "FAILED_RUN $lab" logs/fast_drain_s71.log 2>/dev/null || echo "$(date +%T) FAILED_RUN $lab (left for a decision)" >> logs/fast_drain_s71.log; fi
  done < /tmp/s71_live.txt
  sleep 60
done
