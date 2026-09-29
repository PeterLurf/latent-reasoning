#!/usr/bin/env bash
# Poll cloud pods every 5 min (5 polls): print a line when a run's DONE flag / stage-1 failure count changes,
# and POD_DONE <id> once a pod's post-training evaluation (scratch/pod_after.sh) has finished.
cd "$(dirname "$0")/.."
declare -A prev
for t in $(seq 1 5); do
  while read -r line; do
    key=$(echo "$line" | awk '{print $1, $3}')
    sig=$(echo "$line" | grep -oE 'fails [0-9]+|DONE|not-started' | tr '\n' ' ')
    if [ "${prev[$key]-}" != "$sig" ]; then [ -n "${prev[$key]-}" ] && echo "CHANGE $line"; prev[$key]=$sig; fi
  done < <(bash scratch/pod_status.sh)
  grep -v '^#' scratch/pods.txt | while read id U rest; do
    printf 'test -f /workspace/Latent/logs/EVAL_ALL_DONE && echo "PODDONE_%s"; exit\n' "$id" | timeout 60 ssh -tt -i ~/.ssh/runpod_ed25519 $U@ssh.runpod.io 2>&1 | tr -d '\r' | grep -aoE "PODDONE_$id\$" | sed 's/PODDONE_/POD_DONE /'
  done
  sleep 300
done
