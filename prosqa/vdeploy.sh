#!/usr/bin/env bash
# Rent a Vast offer, wait for direct SSH, check cuInit, upload prosqa/ (no data/res/ckpt) and start a pod script.
#   bash prosqa/vdeploy.sh <offer id> <label> <pod script> <config>
# Prints INSTANCE <id> <host> <port>, CUINIT, STARTED; on a bad host prints BAD HOST (instance left for you to destroy).
set -u
O=$1; LBL=$2; PS=$3; CFG=$4
V="C:/Users/peter/.local/vastai/vast.sh"
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."
ID=$(bash "$V" create instance $O --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel --disk 40 --ssh --direct --label $LBL --raw 2>&1 | grep -o '"new_contract": *[0-9]*' | grep -o '[0-9]*$')
[ -n "$ID" ] || { echo "CREATE FAILED"; exit 1; }
echo "$ID vast offer $O created $(date '+%Y-%m-%d %H:%M'): label $LBL, $PS $CFG (logged at create by prosqa/vdeploy.sh)" >> scratch/pods.txt
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
tar -czf /tmp/up_$LBL.tgz --exclude=__pycache__ --exclude="res*" --exclude="ckpt*" --exclude=logs --exclude=data prosqa
scp -q -i ~/.ssh/runpod_ed25519 -P $P /tmp/up_$LBL.tgz root@$H:/workspace/up.tgz
$S "cd /workspace && tar --no-same-owner -xzf up.tgz && mkdir -p logs && (setsid nohup bash /workspace/prosqa/$PS $CFG > /workspace/logs/prosqa.out 2>&1 < /dev/null &) ; sleep 2; pgrep -fa $PS | head -1" 2>/dev/null | grep -q $PS && echo STARTED
