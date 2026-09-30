#!/usr/bin/env bash
# Rent a Vast offer and start a toy-model queue on it (direct SSH):
#   bash scratch/vdeploy_toy.sh <offer id> <label> <queue file> <queue tag> [worker script, default queue_worker2.sh]
# Uploads latent/ + the queue worker, stage-1 guard, eval battery and the queue file to /workspace/Latent, then starts
# scratch/queue_worker2.sh <tag> cuda:0 <queue file>. Prints INSTANCE <id> <host> <port>, the cuInit/clock line, and
# STARTED; prints BAD HOST if cuInit fails (the instance is left for the caller to destroy).
set -u
O=$1; LBL=$2; Q=$3; TAG=$4; W=${5:-scratch/queue_worker2.sh}
EXTRA=""; [ "$W" = scratch/queue_worker3.sh ] && EXTRA="scratch/queue_worker3.sh scratch/p9_frozen"
V="C:/Users/peter/.local/vastai/vast.sh"
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."
ID=$(bash "$V" create instance $O --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel --disk 30 --ssh --direct --label $LBL --raw 2>&1 | grep -o '"new_contract": *[0-9]*' | grep -o '[0-9]*$')
[ -n "$ID" ] || { echo "CREATE FAILED"; exit 1; }
for i in $(seq 1 60); do
  j=$(bash "$V" show instance $ID --raw 2>/dev/null)
  H=$(echo "$j" | python -c "import json,sys;d=json.load(sys.stdin);print(d.get('public_ipaddr','').strip())" 2>/dev/null)
  P=$(echo "$j" | python -c "import json,sys;d=json.load(sys.stdin);print(d['ports']['22/tcp'][0]['HostPort'])" 2>/dev/null)
  [ -n "$H" ] && [ -n "$P" ] && timeout 30 ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -i ~/.ssh/runpod_ed25519 -p $P root@$H true 2>/dev/null && break
  sleep 20
done
echo "INSTANCE $ID $H $P"
S="ssh -o ConnectTimeout=20 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
$S "python -c \"import ctypes;print('CUINIT',ctypes.CDLL('libcuda.so.1').cuInit(0))\"; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader" 2>/dev/null | grep -v -i "welcome\|have fun"
$S "python -c \"import ctypes;exit(ctypes.CDLL('libcuda.so.1').cuInit(0))\"" 2>/dev/null || { echo "BAD HOST"; exit 2; }
tar -czf /tmp/up_$LBL.tgz --exclude=__pycache__ latent scratch/queue_worker2.sh scratch/stage1_guard.py scratch/eval_battery.sh $EXTRA $Q
scp -q -i ~/.ssh/runpod_ed25519 -P $P /tmp/up_$LBL.tgz root@$H:/tmp/up.tgz
$S "mkdir -p /workspace/Latent && cd /workspace/Latent && tar --no-same-owner -xzf /tmp/up.tgz && mkdir -p logs runs results && (setsid nohup bash $W $TAG cuda:0 $Q > /dev/null 2>&1 < /dev/null &) ; sleep 5; pgrep -fa queue_worker | head -1" 2>/dev/null | grep -q queue_worker && echo STARTED
