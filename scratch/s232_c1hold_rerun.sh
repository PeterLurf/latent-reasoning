#!/bin/bash
# §232 Stage A c1hold re-run on the b344 host (Audit 10-24 G-D: 56c88f8e allocator fix; plan (1)-(3)). Runs ON THE HOST from /workspace/Latent, sequentially,
# the queue's own c1hold line per staged dir (eval_bs 32, the queue runner's device), reading each dir's eval_vb.json from the queue run already there.
cd /workspace/Latent; L=logs/c1hold_rerun.out
for m in s213_d3t_s1 s215_ftctrl_s1 s230_fhftfix_s1 s216_d3t_s2 s224_ftctrl_s2 s224_fhft_s2; do d=boxes/runs_fsq/${m}_c1
  echo "$(date +%T) start $d" >> $L
  python -u -m boxes.vbank_diag --run $d --part c1hold --eval_bs 32 --device cuda:0 --out_dir $d > logs/c1hold_rerun_$m.log 2>&1; rc=$?
  echo "$(date +%T) done $d rc $rc $( [ -s $d/eval_c1hold.json ] && echo present || echo MISSING)" >> $L
done
echo "$(date +%T) RERUN_DONE" >> $L
