#!/bin/bash
# §224 local gates, take 2 (replaces s224_after_pilot.sh, stopped while still waiting: the 4060 freed first). K11 on the 4060 (cuda:0) now; K9 on the
# 3070 (cuda:1) after the 2070 PS pilot's summary line (2 cards at most at a time: crash note 10-21). Done-lines in logs/s222_chain.out.
cd /d/Latent; L=logs/s222_chain.out
echo "$(date '+%F %T') s224 K11 resume test start (4060)" >> $L
bash scratch/s224_resume_test.sh cuda:0; echo "$(date '+%F %T') s224 K11 done: $(tail -1 logs/s224_resume.log)" >> $L
until grep -q "s223 pilot summary rc" $L; do sleep 120; done
echo "$(date '+%F %T') s224 K9 start (3070)" >> $L
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B scratch/s224_k9.py > logs/s224_k9.log 2>&1; rc=$?
echo "$(date '+%F %T') s224 K9 rc $rc: $(grep 'K9 pass' logs/s224_k9.log | tail -1)" >> $L
