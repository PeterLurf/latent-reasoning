#!/usr/bin/env bash
# Restart gpu_watch with every live cell's host whenever a listed queue STARTs (Audit 01:56: cells must not run uncovered). Hosts come from the
# latest "gen host b<n> ... STARTED" line per queue; queues whose host is no longer live (drained_hosts.txt) are dropped. Also restarts the
# watcher if it exited (after an IDLE / UNREACHABLE alert, which stays in logs/gpu_watch.out). Kills only the watcher pid it started itself.
#   nohup bash scratch/gpuwatch_autoadd.sh 291 295 296 297 298 >> logs/gpuwatch_autoadd.out 2>&1 &
cd "$(dirname "$0")/.."
seen=""; WPID=""
while :; do
  cur=""
  for n in "$@"; do
    l=$(grep "gen host b$n offer .*STARTED" logs/gen_deploy.log | tail -1); [ -z "$l" ] && continue
    h=$(echo "$l" | awk '{for(i=1;i<=NF;i++) if($i=="INSTANCE") print $(i+2)}'); p=$(echo "$l" | awk '{for(i=1;i<=NF;i++) if($i=="INSTANCE") print $(i+3)}')
    grep -qx "$h:$p" logs/drained_hosts.txt 2>/dev/null && continue
    cur="$cur|$h $p b$n"
  done
  alive=0; [ -n "$WPID" ] && kill -0 "$WPID" 2>/dev/null && alive=1
  if [ "$cur" != "$seen" ] || { [ $alive -eq 0 ] && [ -n "$cur" ]; }; then
    [ $alive -eq 1 ] && kill "$WPID"
    IFS='|' read -ra A <<< "${cur#|}"; WPID=""
    if [ ${#A[@]} -gt 0 ]; then
      GPU_WATCH_SKIP="RTX 2070,RTX 3070,RTX 4060" nohup bash scratch/gpu_watch.sh "${A[@]}" >> logs/gpu_watch.out 2>&1 &
      WPID=$!
    fi
    echo "$(date '+%F %T') gpu_watch (pid ${WPID:-none}) restarted with: ${cur#|}"; seen="$cur"
  fi
  sleep 60
done
