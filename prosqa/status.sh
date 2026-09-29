#!/usr/bin/env bash
# Local helper: status of the ProsQA cloud machine.
#   bash prosqa/status.sh <runpod-ssh-user>        (RunPod proxy)
#   bash prosqa/status.sh vast:<host>:<port>         (direct SSH, e.g. Vast.ai)
if [[ $1 == vast:* ]]; then
  IFS=: read -r _ H P <<<"$1"
  timeout 60 ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -i $HOME/.ssh/runpod_ed25519 -p "$P" root@"$H" \
    'bash /workspace/prosqa/podstat.sh' 2>/dev/null | tr -d '\r' | grep -ao "PODX .*" | sed 's/^PODX //'
else
  printf 'bash /workspace/prosqa/podstat.sh; exit\n' | timeout 60 ssh -tt -o ConnectTimeout=20 -i $HOME/.ssh/runpod_ed25519 $1@ssh.runpod.io 2>&1 \
   | tr -d '\r' | grep -ao "PODX .*" | sed 's/^PODX //'
fi
