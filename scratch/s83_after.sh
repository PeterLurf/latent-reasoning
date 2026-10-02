#!/usr/bin/env bash
# §83 on the retrain host: after the §71-command retrain (QUEUE_DONE), rename runs to wc0k4r_s<seed> (Theory's names)
# and run theory/s83_restart.py per seed (Theory's command). Writes logs/s83_after.out, ends with S83_DONE.
cd /workspace/Latent
L=logs/s83_after.out
until tail -n 1 logs/queue_s83.out 2>/dev/null | grep -q QUEUE_DONE; do sleep 30; done
for s in 1 2 3; do
  [ -d boxes/runs_s5/s83_wc0k4_s$s ] && mv boxes/runs_s5/s83_wc0k4_s$s boxes/runs_s5/wc0k4r_s$s
  echo "$(date +%T) start s83 seed $s" >> $L
  python -u theory/s83_restart.py --run boxes/runs_s5/wc0k4r_s$s --k 4 --kmax 16 --test_n 2000 --ext_n 500 --cal_n 20000 --sweep_n 500 --fixed 4 8 16 --device cuda:0 --out theory/results/s83/wc0k4r_s$s.json > logs/s83_wc0k4r_s$s.log 2>&1
  echo "$(date +%T) s83 seed $s rc $?" >> $L
done
echo "$(date +%T) S83_DONE" >> $L
