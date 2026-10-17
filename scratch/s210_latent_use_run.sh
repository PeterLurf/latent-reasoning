#!/usr/bin/env bash
# 210-2 / 210-3 (Audit-approved 73084338): part b then part a of scratch/s210_latent_use.py on the 3070 (cuda:1).
cd "$(dirname "$0")/.."
for P in b a; do
  python scratch/s210_latent_use.py --part $P --device cuda:1 > logs/s210_latent_use_$P.log 2>&1; rc=$?
  echo "$(date '+%F %T') s210 latent_use part $P rc $rc $(tail -n 1 logs/s210_latent_use_$P.log | cut -c1-140)"
done
echo "$(date '+%F %T') s210 latent_use done"
