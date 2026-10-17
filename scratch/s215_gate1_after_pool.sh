#!/bin/bash
# §215: GATE-1 (G-S15) on the 3070 after the pool build (pid passed as $1) exits; skipped unless pool_meta.json landed.
cd /d/Latent
while powershell -NoProfile -Command "if (Get-Process -Id $1 -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 60; done
if [ -f boxes/runs_fsq/s215_pool/pool_meta.json ] && grep -q "^DONE" logs/s215_build_pool.log; then
  python scratch/s215_gate1.py --device cuda:1 > logs/s215_gate1.log 2>&1; rc=$?
  echo "$(date '+%F %T') gate1 rc $rc" >> logs/s215_gate1.done
else
  echo "$(date '+%F %T') pool build did not finish - GATE-1 not run" >> logs/s215_gate1.done
fi
