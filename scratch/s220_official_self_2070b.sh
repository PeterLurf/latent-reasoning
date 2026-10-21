#!/bin/bash
# Note 220-3: CODI20 official-vs-official self-consistency on the 2070 (cuda:2, fp32), RE-RUN after the overwrite-assert fix (f2a3850b); the first attempt
# (s220_official_self_2070.sh) died at start with rc 1, nothing written (logs *.void_assert). The 2070 is free (k3swap done 02:53).
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s220_stage0.out
for sd in 22030 22031; do
  python -B scratch/s220_gcs.py --model codi20 --device cuda:2 --official_shuffle $sd > logs/s220_codi20_shuffle_$sd.log 2>&1; rc=$?
  echo "$(date '+%F %T') s220 codi20 official_shuffle $sd (2070) rc $rc" >> $L
done
python -B scratch/s220_official_self.py --model codi20 > logs/s220_codi20_official_self.log 2>&1; rc=$?
echo "$(date '+%F %T') s220 codi20 official_self rc $rc: $(tail -n 1 logs/s220_codi20_official_self.log)" >> $L
