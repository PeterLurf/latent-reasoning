#!/usr/bin/env bash
# Audit check (1): the largest 2-concurrent configuration on one 3090 (a 16k §91 WD-SP arm + a §92 run, real batch
# sizes, 400 steps each), peak GPU memory sampled every 2 s -> logs/memtest.out. Non-evidential (scratch outputs).
cd /workspace/Latent
python -u -m c1.p80_p0 loop90 --group S4 --k 2 --n_words 400 --seed 4 --steps 400 --eval_every 200 --section 91 --wd_sp --device cuda:0 --out /tmp/mt91 > logs/memtest_91.log 2>&1 &
p1=$!
python -u -m boxes.latent86 train --k 4 --seed 1 --steps 400 --bs 64 --test_n 20 --ext_n 0 --check_n 500 --log_every 200 --log_n 500 --device cuda:0 --out /tmp/mt92 > logs/memtest_92.log 2>&1 &
p2=$!
peak=0
while kill -0 $p1 2>/dev/null || kill -0 $p2 2>/dev/null; do
  u=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1); m=${u%%,*}; [ "$m" -gt "$peak" ] && peak=$m; sleep 2
done
wait $p1; r1=$?; wait $p2; r2=$?
echo "MEMTEST peak_used_MiB $peak total $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) rc91 $r1 rc92 $r2 eval91 $([ -s /tmp/mt91/eval.json ] && echo ok) eval92 $([ -s /tmp/mt92/eval.json ] && echo ok)" > logs/memtest.out
