#!/bin/bash
# Note 216-10 KL-DRIFT on the 4060 (cuda:0) once the 218-17 known-answer run ends (K3 written or a crash) - one job per card.
cd /d/Latent
until grep -qE "WROTE k3_probe_learn|Traceback" logs/s218_gl5_known.log; do sleep 60; done
sleep 30
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B scratch/s216_kl_base.py --device cuda:0 > logs/s216_kl_base.log 2>&1; rc=$?
echo "$(date '+%F %T') s216 kl_base rc $rc" >> logs/s216_kl_base.out
