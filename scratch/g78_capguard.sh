#!/usr/bin/env bash
# §78 hard cap (Audit): destroy any g78 host past 2x its projection (b-hosts 19:10, a-hosts 21:40 local), log it.
cd "$(dirname "$0")/.."
V="C:/Users/peter/.local/vastai/vast.sh"
while :; do
  now=$(date +%H%M); left=0
  for x in "53887296 b1 1910" "53887305 b2 1910" "53887312 b3 1910" "53887276 a1 2140" "53887285 a2 2140" "53887291 a3 2140"; do set -- $x
    bash "$V" show instance $1 --raw >/dev/null 2>&1 || continue
    st=$(bash "$V" show instance $1 --raw 2>/dev/null | python -c "import json,sys; print(json.load(sys.stdin).get('actual_status'))" 2>/dev/null)
    [ "$st" = "None" ] || [ -z "$st" ] && continue
    left=1
    if [ "$now" -ge "$3" ]; then r=$(bash scratch/vdestroy.sh $1); echo "$(date +%T) CAP g78-$2 $1: $r" >> logs/g78_capguard.log
      echo "$1 vast | g78-$2 $r $(date '+%Y-%m-%d %H:%M'): hard cap (2x projection, Audit); logs on host lost unless drained" >> scratch/pods.txt; fi
  done
  [ $left = 0 ] && { echo "$(date +%T) no g78 hosts left" >> logs/g78_capguard.log; exit 0; }
  sleep 600
done
