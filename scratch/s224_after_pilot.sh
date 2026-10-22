#!/bin/bash
# §224 local gates after the §223 pilots (2 cards at most at a time: crash note 10-21): wait for the 2070 PS pilot's summary line, then K11 on the
# 2070 (cuda:2), then K9 on the 3070 (cuda:1). Done-lines in logs/s222_chain.out.   nohup bash scratch/s224_after_pilot.sh >> logs/s222_chain.out 2>&1 &
cd /d/Latent; L=logs/s222_chain.out
until grep -q "s223 pilot summary rc" $L; do sleep 120; done
echo "$(date '+%F %T') s224 K11 resume test start (2070)" >> $L
bash scratch/s224_resume_test.sh cuda:2; echo "$(date '+%F %T') s224 K11 done: $(tail -1 logs/s224_resume.log)" >> $L
echo "$(date '+%F %T') s224 K9 start (3070)" >> $L
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B scratch/s224_k9.py > logs/s224_k9.log 2>&1; rc=$?
echo "$(date '+%F %T') s224 K9 rc $rc: $(grep 'K9 pass' logs/s224_k9.log | tail -1)" >> $L
