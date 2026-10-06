#!/usr/bin/env bash
# THEORY 128-4 (Theory 2026-10-06, low priority): cert_cover128 (bf16 + fp32, full 2000 x 512) on the remaining certified §128
# MOD / CYC / RDFA runs, on the 4060 (cuda:0, the card mod_s1 / rdfa_s4 used) AFTER §130's 4060 runner (TEST-SAME + s130chk) and the
# mod_s1 pass exit. Waits on Windows pids via PowerShell (MSYS kill -0 cannot see them: the 10:03 rdfa_s4 chain started early).
#   bash scratch/c1_q_certcover128.sh <pid> [<pid> ...]
cd "$(dirname "$0")/.."
alive() { powershell -NoProfile -Command "if (Get-Process -Id $1 -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" > /dev/null 2>&1; }
for p in "$@"; do while alive "$p"; do sleep 120; done; done
echo "$(date +%T) certcover128 queue start (waited on pids $*)" >> logs/c1_local.log
for r in mod_s2 mod_s3 mod_s4 cyc_s1 cyc_s2 cyc_s3 cyc_s4 rdfa_s1 rdfa_s2 rdfa_s3; do
  d=boxes/runs_fsq/s128_$r
  [ -f scratch/certcover128.stop ] && break
  [ -s $d/eval_cert_cover.json ] && continue
  [ -s $d/model.pt ] || { echo "$(date +%T) cert_cover128 s128_$r SKIPPED (no model.pt)" >> logs/c1_local.log; continue; }
  OMP_NUM_THREADS=3 python -u scratch/cert_cover128.py $d cuda:0 > logs/c1_cert_cover_$r.log 2>&1
  echo "$(date +%T) cert_cover128 s128_$r rc $? (cuda:0)" >> logs/c1_local.log
done
echo "$(date +%T) certcover128 queue done" >> logs/c1_local.log
