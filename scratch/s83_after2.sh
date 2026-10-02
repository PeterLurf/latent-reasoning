#!/usr/bin/env bash
# §83 on the retrain host, v2 (note 83-3, f263c9e script): primary first for all seeds, extras afterwards.
#   primary: --test_n 2000 --ext_n 0 --sweep_n 0 --fixed 4 16 -> theory/results/s83/wc0k4r_s<seed>.json
#   extras : Theory's full command (ext 500, sweep 500, fixed 4 8 16) -> theory/results/s83/wc0k4r_s<seed>_ext.json
cd /workspace/Latent
L=logs/s83_after.out
for s in 1 2 3; do [ -d boxes/runs_s5/s83_wc0k4_s$s ] && mv boxes/runs_s5/s83_wc0k4_s$s boxes/runs_s5/wc0k4r_s$s; done
for s in 1 2 3; do
  echo "$(date +%T) start primary s$s" >> $L
  python -u theory/s83_restart.py --run boxes/runs_s5/wc0k4r_s$s --k 4 --kmax 16 --test_n 2000 --ext_n 0 --cal_n 20000 --sweep_n 0 --fixed 4 16 --device cuda:0 --out theory/results/s83/wc0k4r_s$s.json > logs/s83_primary_s$s.log 2>&1
  echo "$(date +%T) primary s$s rc $?" >> $L
done
echo "$(date +%T) S83_PRIMARY_DONE" >> $L
for s in 1 2 3; do
  echo "$(date +%T) start extras s$s" >> $L
  python -u theory/s83_restart.py --run boxes/runs_s5/wc0k4r_s$s --k 4 --kmax 16 --test_n 2000 --ext_n 500 --cal_n 20000 --sweep_n 500 --fixed 4 8 16 --device cuda:0 --out theory/results/s83/wc0k4r_s${s}_ext.json > logs/s83_ext_s$s.log 2>&1
  echo "$(date +%T) extras s$s rc $?" >> $L
done
echo "$(date +%T) S83_DONE" >> $L
