#!/usr/bin/env bash
# THEORY §69 (amendment 69-1): rent one many-core Vast host, CPU only, and run theory/c1_rl_driver.py on it.
#   bash scratch/c1rl_vast.sh <offer id> <label> "<driver args>" [driver, default theory/c1_rl_driver.py]   e.g. "--phase all --P 16"
# Prints INSTANCE <id> <host> <port> and STARTED. CPU-only job: not added to live_hosts.txt (the GPU-idle watcher would
# flag it); Theory polls it with: ssh ... "cat /workspace/Latent/logs/c1rl.out; ls /workspace/Latent/theory/results/c1_rl | wc -l"
set -u
O=$1; LBL=$2; DARGS=$3; DRV=${4:-theory/c1_rl_driver.py}
V="C:/Users/peter/.local/vastai/vast.sh"
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."
ID=$(bash "$V" create instance $O --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel --disk 20 --ssh --direct --label $LBL --raw 2>&1 | grep -o '"new_contract": *[0-9]*' | grep -o '[0-9]*$')
[ -n "$ID" ] || { echo "CREATE FAILED"; exit 1; }
echo "$ID vast offer $O created $(date '+%Y-%m-%d %H:%M'): label $LBL, CPU-only theory job ($DRV $DARGS; logged at create by c1rl_vast.sh)" >> scratch/pods.txt
for i in $(seq 1 60); do
  j=$(bash "$V" show instance $ID --raw 2>/dev/null)
  H=$(echo "$j" | python -c "import json,sys;d=json.load(sys.stdin);print(d.get('public_ipaddr','').strip())" 2>/dev/null)
  P=$(echo "$j" | python -c "import json,sys;d=json.load(sys.stdin);print(d['ports']['22/tcp'][0]['HostPort'])" 2>/dev/null)
  [ -n "$H" ] && [ -n "$P" ] && timeout 30 ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -i ~/.ssh/runpod_ed25519 -p $P root@$H true 2>/dev/null && break
  sleep 20
done
echo "INSTANCE $ID $H $P"
S="ssh -o ConnectTimeout=20 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
$S "nproc; python -c 'import torch;print(torch.__version__)'; lscpu | grep -E 'Model name|^CPU\(s\)|Thread'" 2>/dev/null | grep -v -i "welcome\|have fun"
tar -czf /tmp/up_$LBL.tgz --exclude=__pycache__ theory/*.py
scp -q -i ~/.ssh/runpod_ed25519 -P $P /tmp/up_$LBL.tgz root@$H:/tmp/up.tgz
$S "mkdir -p /workspace/Latent && cd /workspace/Latent && tar --no-same-owner -xzf /tmp/up.tgz && mkdir -p logs theory/results/c1_rl && (setsid nohup python -u $DRV $DARGS > logs/c1rl.out 2>&1 < /dev/null &) ; sleep 5; pgrep -fa driver.py | head -2" 2>/dev/null | grep -q driver && echo STARTED
