#!/bin/bash
# §218 local gates on the 3070 (cuda:1) after G-L1b (pid 14168): G-L7(b') on 2,000 items.
cd /d/Latent
while powershell -NoProfile -Command "if (Get-Process -Id 14168 -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 60; done
python -B scratch/gate_s218_lvb.py l7b --device cuda:1 --n 2000 > scratch/gate_s218/g_l7b.log 2>&1; rc=$?
echo "$(date '+%F %T') l7b rc $rc" >> logs/s218_gates.out
