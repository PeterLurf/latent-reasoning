#!/usr/bin/env bash
# Theory 2026-10-13 (path_indep series): probe running cloud cells on a fixed step cadence through their entry window, read-only, one probe at a time on
# the local 3070. Each spec "run:host:port:first:last:every"; at each target step the loop polls the host's last log step (the step FIELD only - no loss
# is read) and, once step >= target, runs fetch_probe_pathindep.sh (it probes the host's ckpt_last, written every 1000 steps, so the probe's step is
# the latest 1000 multiple). PROBE_SEEN=no: the pipeline reads no training loss before the probe (the probe reads the 1k bin after it is computed).
# Waits for HOLD_FILE (if set) to exist before each probe (a local gate on the same card). Writes logs/probe_cadence.log.
#   bash scratch/probe_cadence.sh <spec> [<spec> ...]
set -u
cd "$(dirname "$0")/.."
LOG=logs/probe_cadence.log; declare -A NEXT LAST EVERY HOST PORT
for sp in "$@"; do
  IFS=: read -r r h p f l e <<<"$sp"; NEXT[$r]=$f; LAST[$r]=$l; EVERY[$r]=$e; HOST[$r]=$h; PORT[$r]=$p
done
echo "$(date '+%F %T') start: $*" >> $LOG
while [ ${#NEXT[@]} -gt 0 ]; do
  for r in "${!NEXT[@]}"; do
    s=$(timeout 40 ssh -o ConnectTimeout=15 -i $HOME/.ssh/runpod_ed25519 -p ${PORT[$r]} root@${HOST[$r]} "tail -n 1 /workspace/Latent/boxes/runs_fsq/$r/log.jsonl | cut -d, -f1 | tr -dc 0-9" 2>/dev/null)
    [ -n "$s" ] || continue
    if [ "$s" -ge "${NEXT[$r]}" ]; then
      while [ -n "${HOLD_FILE:-}" ] && [ ! -f "$HOLD_FILE" ]; do sleep 60; done
      echo "$(date '+%F %T') $r: host step $s >= target ${NEXT[$r]}, probing" >> $LOG
      PROBE_SEEN=no bash scratch/fetch_probe_pathindep.sh ${HOST[$r]} ${PORT[$r]} $r cuda:1 >> $LOG 2>&1
      n=$(( (s / EVERY[$r] + 1) * EVERY[$r] ))
      if [ $n -gt ${LAST[$r]} ]; then unset "NEXT[$r]"; echo "$(date '+%F %T') $r: cadence done" >> $LOG; else NEXT[$r]=$n; fi
    fi
  done
  sleep 180
done
echo "$(date '+%F %T') all done" >> $LOG
