#!/bin/bash
# §219 round 1 on the 3070 (cuda:1, fp32): B1 gate items (CPU) -> GATE-T1 label (primary labeller, incl. LABELLER-FULL + G-T16 EQUIV) -> score (CPU, gold)
# -> T1 pool (B1 asserts eval_gate's window labeller / EQUIV / label settings itself; exits 3 if a window has no passing labeller). One step per line.
cd /d/Latent; export LATENT_MEM_CAP_GB=6.0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
L=logs/s219_gate_t1.out
run() { local tag=$1; shift; "$@" > logs/s219_t1_$tag.log 2>&1; local rc=$?; echo "$(date '+%F %T') s219 $tag rc $rc" >> $L; return $rc; }
run gates python -B scratch/s219_build_pool.py --round 1 --stage gates || exit 1
run label python -B scratch/s219_gate.py --round 1 --stage label --labeller primary --device cuda:1 || exit 1
run score python -B scratch/s219_gate.py --round 1 --stage score || exit 1
run pool python -B scratch/s219_build_pool.py --round 1 --stage pool --device cuda:1 || exit 1
echo "$(date '+%F %T') §219 round-1 gate + pool done" >> $L
