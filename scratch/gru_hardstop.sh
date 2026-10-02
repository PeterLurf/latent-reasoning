#!/usr/bin/env bash
# §71 GRU (descriptive): hard stop at 09:00 local (Audit), unless it finished; fetch its log, then destroy the host.
cd "$(dirname "$0")/.."
until [ "$(date +%H%M)" -ge 0900 ] && [ "$(date +%H)" -lt 12 ]; do sleep 300; done
S="ssh -o ConnectTimeout=15 -i $HOME/.ssh/runpod_ed25519 -p 60171 root@137.175.22.196"
mkdir -p logs/gru_s71; timeout 120 scp -q -i ~/.ssh/runpod_ed25519 -P 60171 "root@137.175.22.196:/workspace/Latent/logs/s71_gru_s1.log" logs/gru_s71/ 2>/dev/null
if timeout 30 $S "test -s /workspace/Latent/boxes/runs_s5/gru_s1/preds_test.npy" 2>/dev/null; then bash scratch/drain_s71.sh 137.175.22.196 60171 s71_gru 53814109 >> logs/gru_s71/stop.log 2>&1
else r=$(bash scratch/vdestroy.sh 53814109); echo "$(date +%T) $r" >> logs/gru_s71/stop.log
  case "$r" in DESTROYED_VERIFIED*) echo "53814109 vast | s71-gru $r $(date '+%Y-%m-%d %H:%M'): hard stop 09:00 (Audit), not converged within budget; log in logs/gru_s71/" >> scratch/pods.txt;; esac; fi
