#!/usr/bin/env bash
# Audit 2026-10-06 23:08: §133 slot hosts (queues via scratch.restart133) skipped gen_vast_job's S5 data step (its grep keys on the module
# names) -> only S5=2048.csv on disk, eval crashes on S5=512.csv. This is that step verbatim (deterministic generator, sha-verified); it
# writes the CSVs atomically (tmp + mv) so a run reaching eval mid-way never reads a partial file.   nohup bash scratch/s5_data_fix.sh &
cd /workspace/Latent; log=logs/s5_data_fix.log
SHA128=6ee3ef95405dc43b094824dff8973753c587fe95a00762d2023bd64a941504f2; SHA512=a0087fe4a7496a47f76d9436e6bb089c8a8869f3f343f71057355e127d0c33da
for try in 1 2 3 4; do pip install -q polars fire pyrootutils "git+https://github.com/jopetty/abstract_algebra" >> $log 2>&1; python -c "import polars, fire, pyrootutils" >> $log 2>&1 && break; sleep 30; done
G=/workspace/s71gen; mkdir -p $G/src $G/data bench/data/s5; cp scratch/s71_gen/generate_data.py $G/src/; touch $G/.project-root
for gtry in 1 2; do
  [ $gtry = 2 ] && pip install -q "polars[rtcompat]" >> $log 2>&1
  ( cd $G && PYTHONPATH=$G python src/generate_data.py --group=S5 --k=128 --samples=100000 --seed=7101 && \
    PYTHONPATH=$G python src/generate_data.py --group=S5 --k=512 --samples=100000 --seed=7102 ) >> $log 2>&1
  s1=$(sha256sum $G/data/S5=128.csv | cut -c1-64); s2=$(sha256sum $G/data/S5=512.csv | cut -c1-64); echo "$(date +%T) data sha $s1 $s2" >> $log
  if [ "$s1" = "$SHA128" ] && [ "$s2" = "$SHA512" ]; then
    cp $G/data/S5=128.csv bench/data/s5/.S5=128.tmp && mv bench/data/s5/.S5=128.tmp bench/data/s5/S5=128.csv
    cp $G/data/S5=512.csv bench/data/s5/.S5=512.tmp && mv bench/data/s5/.S5=512.tmp bench/data/s5/S5=512.csv
    echo "$(date +%T) S5_DATA_OK" >> $log; exit 0
  fi
done
echo "$(date +%T) S5_DATA_FAILED" >> $log; exit 1
