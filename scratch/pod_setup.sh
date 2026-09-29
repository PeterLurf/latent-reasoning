#!/usr/bin/env bash
# Local helper: prepare a RunPod pod over the SSH proxy and start one run.
#   bash scratch/pod_setup.sh <ssh-user> <run-name> [train args...]
# Steps: wait for sshd -> cuInit check (abort on failure) -> upload code -> torch 2.14.0+cu126 -> pod_run.sh.
set -u
U=$1; name=$2; shift 2
S="ssh -tt -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -i $HOME/.ssh/runpod_ed25519 $U@ssh.runpod.io"
rsh() { timeout "${T:-90}" $S 2>&1 | tr -d '\r'; }
for i in $(seq 1 30); do
  r=$(printf 'python -c "import ctypes;print(\\"CUINIT\\",ctypes.CDLL(\\"libcuda.so.1\\").cuInit(0))"; exit\n' | rsh | grep -aoE 'CUINIT [0-9]+')
  [ -n "$r" ] && break; sleep 15
done
echo "[$name] $r"
[ "$r" = "CUINIT 0" ] || { echo "[$name] BAD HOST"; exit 2; }
cd /d/Latent && tar -czf /tmp/up_$name.tgz --exclude=__pycache__ latent scratch/stage1_guard.py scratch/chain.py scratch/pod_run.sh scratch/run_status.py
{ echo 'stty -echo; mkdir -p /workspace/Latent/logs && cd /workspace/Latent'
  echo "base64 -d > up.tgz <<'B64EOF'"; base64 -w 76 /tmp/up_$name.tgz; echo 'B64EOF'
  echo 'tar --no-same-owner -xzf up.tgz && echo "UPLOADED $(ls latent | wc -l)"'
  echo "nohup sh -c 'pip install -q torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126 >logs/setup.log 2>&1; pip install -q numpy==2.5.2 >>logs/setup.log 2>&1; python -c \"import torch;assert torch.cuda.is_available();print(\\\"TORCHOK\\\",torch.__version__)\" >>logs/setup.log 2>&1 && bash scratch/pod_run.sh $name $* >logs/pod_run_$name.out 2>&1' >/dev/null 2>&1 &"
  echo 'sleep 1; exit'; } | T=180 rsh | grep -aoE 'UPLOADED [0-9]+' | sed "s/^/[$name] /"
for i in $(seq 1 40); do
  r=$(printf 'grep -ao "TORCHOK.*" /workspace/Latent/logs/setup.log; tail -2 /workspace/Latent/logs/guard_%s.log 2>/dev/null | grep -ao "launched pid [0-9]*"; exit\n' "$name" | rsh | grep -aoE 'TORCHOK [^ ]+|launched pid [0-9]+' | tr '\n' ' ')
  case "$r" in *launched*) echo "[$name] $r"; exit 0;; esac; sleep 20
done
echo "[$name] TIMEOUT waiting for launch: $r"; exit 1
