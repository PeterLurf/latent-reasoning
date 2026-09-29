#!/usr/bin/env bash
# One line per cloud run (step, curriculum stage, latest delayed bal-acc, ms/step, stage-1 failures). Reads scratch/pods.txt.
cd "$(dirname "$0")/.."
grep -v '^#' scratch/pods.txt | while read id U card price runs; do
  printf 'cd /workspace/Latent && python3 scratch/run_status.py %s; exit\n' "$runs" \
  | timeout 60 ssh -tt -i ~/.ssh/runpod_ed25519 $U@ssh.runpod.io 2>&1 | tr -d '\r' | grep -aoE 'STAT [a-z].*' | sed "s/^STAT /$id $card: /"
done
