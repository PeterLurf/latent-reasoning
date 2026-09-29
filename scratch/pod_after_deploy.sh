#!/usr/bin/env bash
# Local helper: upload the evaluation scripts to a pod and start scratch/pod_after.sh there.
#   bash scratch/pod_after_deploy.sh <ssh-user> <run>[:clock|:variant|:variantclock] ...
set -u
U=$1; shift
S="ssh -tt -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -i $HOME/.ssh/runpod_ed25519 $U@ssh.runpod.io"
cd /d/Latent && tar -czf /tmp/after_$U.tgz --exclude=__pycache__ latent scratch/pod_after.sh scratch/eval_battery.sh scratch/variant_pipeline.py
{ echo 'stty -echo; mkdir -p /workspace/Latent/logs && cd /workspace/Latent'
  echo "base64 -d > after.tgz <<'B64EOF'"; base64 -w 76 /tmp/after_$U.tgz; echo 'B64EOF'
  echo 'tar --no-same-owner -xzf after.tgz && rm -f logs/EVAL_ALL_DONE && echo "UPLOADED_OK"'
  echo "nohup bash scratch/pod_after.sh $* >logs/pod_after.out 2>&1 &"
  echo 'sleep 2; ps -eo args | grep -ac "^bash scratch/pod_after" | sed "s/^/AFTERPROCS /"; exit'; } \
  | timeout 180 $S 2>&1 | tr -d '\r' | grep -aoE 'UPLOADED_OK|AFTERPROCS [0-9]+' | sed "s/^/[$U] /"
