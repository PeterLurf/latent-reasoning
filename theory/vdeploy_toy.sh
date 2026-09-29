#!/usr/bin/env bash
# Theory stream: rent a Vast offer, check cuInit, upload the toy code, start theory/queue_arch.sh on <queue file>.
#   bash theory/vdeploy_toy.sh <offer id> <label> <queue file (local path)>
# Prints INSTANCE <id> <host> <port>, CUINIT, STARTED; a bad host prints BAD HOST (instance left for you to destroy).
set -u
O=$1; LBL=$2; QF=$3
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
$S "python -c \"import ctypes;print('CUINIT',ctypes.CDLL('libcuda.so.1').cuInit(0))\"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader" 2>/dev/null | grep -v -i "welcome\|have fun"
$S "python -c \"import ctypes;exit(ctypes.CDLL('libcuda.so.1').cuInit(0))\"" 2>/dev/null || { echo "BAD HOST"; exit 2; }
for f in theory/*.sh; do sed -i 's/\r$//' "$f"; done
tar -czf /tmp/up_$LBL.tgz --exclude=__pycache__ latent theory/arch.py theory/train_arch.py theory/eval_arch.py theory/rollout.py theory/queue_arch.sh scratch/stage1_guard.py
scp -q -i ~/.ssh/runpod_ed25519 -P $P /tmp/up_$LBL.tgz root@$H:/workspace/up.tgz
scp -q -i ~/.ssh/runpod_ed25519 -P $P "$QF" root@$H:/workspace/queue.txt
$S "mkdir -p /workspace/Latent && cd /workspace/Latent && tar --no-same-owner -xzf /workspace/up.tgz && mv /workspace/queue.txt theory/queue.txt && sed -i 's/\r$//' theory/queue.txt && (setsid nohup bash theory/queue_arch.sh $LBL cuda:0 theory/queue.txt > logs_q.out 2>&1 < /dev/null &) ; sleep 3; pgrep -fa queue_arch | head -1" 2>/dev/null | grep -q queue_arch && echo STARTED
