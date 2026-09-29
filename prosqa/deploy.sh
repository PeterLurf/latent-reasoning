#!/usr/bin/env bash
# Local helper: cuInit check on a RunPod pod, upload prosqa/ over the SSH proxy, start pod_prosqa.sh (nohup).
#   bash prosqa/deploy.sh <ssh-user> [pod script, default pod_prosqa.sh]
set -u
U=$1; P=${2:-pod_prosqa.sh}
S="ssh -tt -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -i $HOME/.ssh/runpod_ed25519 $U@ssh.runpod.io"
rsh() { timeout "${T:-90}" $S 2>&1 | tr -d '\r'; }
for i in $(seq 1 30); do
  r=$(printf 'python -c "import ctypes;print(\\"CUINIT\\",ctypes.CDLL(\\"libcuda.so.1\\").cuInit(0))"; exit\n' | rsh | grep -aoE 'CUINIT [0-9]+')
  [ -n "$r" ] && break; sleep 15
done
echo "[prosqa] $r"; [ "$r" = "CUINIT 0" ] || { echo "[prosqa] BAD HOST"; exit 2; }
cd /d/Latent && tar -czf /tmp/up_prosqa.tgz --exclude=__pycache__ --exclude="res*" --exclude="ckpt*" --exclude=logs --exclude=data prosqa
{ echo 'stty -echo; mkdir -p /workspace/logs && cd /workspace'
  echo "base64 -d > up.tgz <<'B64EOF'"; base64 -w 76 /tmp/up_prosqa.tgz; echo 'B64EOF'
  echo 'tar --no-same-owner -xzf up.tgz && echo "UPLOADED $(ls prosqa | wc -l)"'
  echo "nohup bash /workspace/prosqa/$P > /workspace/logs/prosqa.out 2>&1 &"
  echo 'sleep 1; exit'; } | T=120 rsh | grep -aoE 'UPLOADED [0-9]+'
