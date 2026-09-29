#!/usr/bin/env bash
# On the pod: short status lines, each prefixed PODX (read by prosqa/status.sh).
L=/workspace/logs/prosqa.out
echo "PODX gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) procs $(pgrep -fc 'run.py|analyze.py')"
grep -aoE "ENVOK.*|TRAIN_EXIT.*|ANALYSIS_DONE|FAIL.*|[A-Za-z]*Error.*|Accuracy on validation set.*" $L | tail -5 | sed "s/^/PODX /"
tail -c 600 $L | tr "\r" "\n" | grep -a "Epoch" | tail -1 | cut -c1-160 | sed "s/^/PODX /"
echo "PODX ckpts $(ls /workspace/ckpt/prosqa-coconut* 2>/dev/null | tr '\n' ' ') res $(ls /workspace/prosqa/res 2>/dev/null | tr '\n' ' ')"
