#!/usr/bin/env bash
# Host-side idle stop (scratch/idle_stop/DESIGN.md; Audit MED 11-04). Runs ON the Vast host, started by gen_vast_job.sh with setsid nohup.
# Stops (never destroys) its own instance when (1) the queue log ends in QUEUE_DONE older than IDLE_H and no logs/FETCH_ACTIVE was
# touched in the last 45 min, or (2) the job has run longer than HARD_WALL_H (0 = off). Uses the per-instance CONTAINER_API_KEY /
# CONTAINER_ID from /proc/1/environ (ssh shells don't inherit the docker env); the key is never printed.
#   bash scratch/host_idle_stop.sh <queue tag> [IDLE_H] [HARD_WALL_H] [POLL_S]
set -u
cd /workspace/Latent || exit 1
T=$1; IDLE_H=${2:-1.5}; WALL_H=${3:-0}; POLL=${4:-300}
Q=logs/queue_$T.out; L=logs/idle_stop.log; t0=$(date +%s)
envv() { tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n "s/^$1=//p" | head -1; }
CID=$(envv CONTAINER_ID); KEY=$(envv CONTAINER_API_KEY)
case "$CID" in ''|*[!0-9]*) echo "$(date +%T) CONTAINER_ID not numeric; idle stop disabled" >> $L; exit 2 ;; esac   # Audit MED-2: only id = /proc/1/environ
[ -n "$KEY" ] || { echo "$(date +%T) NO_SELF_KEY (CONTAINER_ID ${CID:+set} / CONTAINER_API_KEY ${KEY:+set}); idle stop disabled" >> $L; exit 2; }
echo "$T $IDLE_H $WALL_H $POLL" > logs/idle_stop.args   # the Vast --onstart-cmd relaunches from this after a container restart (Audit LOW b)
if [ -f logs/IDLE_STOP ]; then   # restarted after a self stop (a drain started it): grace so the drain can fetch before any re-stop
  mv logs/IDLE_STOP logs/IDLE_STOP.$(date +%s); echo "$(date +%T) restarted after IDLE_STOP: grace 45 min" >> $L; sleep 2700
fi
echo "$(date +%T) armed: tag $T idle_h $IDLE_H wall_h $WALL_H poll $POLL" >> $L
stop_self() {
  echo "$(date +%T) IDLE_STOP $1" | tee -a $L > logs/IDLE_STOP
  sync
  for i in 1 2 3 4 5; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -X PUT "https://console.vast.ai/api/v0/instances/$CID/" \
      -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d '{"state":"stopped"}')
    echo "$(date +%T) stop request $i http $code" >> $L
    case $code in 2*) sleep 600 ;; *) sleep 60 ;; esac
  done
  exit 3
}
while :; do
  now=$(date +%s)
  if [ "$WALL_H" != 0 ] && python -c "import sys; sys.exit(0 if ($now - $t0) / 3600 > float('$WALL_H') else 1)"; then stop_self "hard wall $WALL_H h"; fi
  if [ -f "$Q" ] && tail -n 1 "$Q" | grep -q QUEUE_DONE; then
    qd=$(stat -c %Y "$Q"); fa=$(stat -c %Y logs/FETCH_ACTIVE 2>/dev/null || echo 0)
    busy=$(pgrep -fc 'scp -f|sftp-server|tar -czf /tmp/f_' 2>/dev/null)   # Audit MED-1: a fetch in flight blocks the stop
    if [ "${busy:-0}" -eq 0 ] && [ $((now - fa)) -gt 2700 ] && python -c "import sys; sys.exit(0 if ($now - $qd) / 3600 > float('$IDLE_H') else 1)"; then
      stop_self "QUEUE_DONE idle > $IDLE_H h, no fetch"
    fi
  fi
  sleep "$POLL"
done
