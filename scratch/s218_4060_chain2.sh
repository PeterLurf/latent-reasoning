#!/bin/bash
# §218 on the 4060 (cuda:0): G-EVMEM re-run (probe groups), then the final-code e2e writers (tiny CPU-trained runs, evaluated on the 4060) for G-L17.
cd /d/Latent
G=scratch/gate_s218/g_evmem; mv $G/eval_lvb.json $G/eval_lvb.json.void_probe_lockstep_oom 2>/dev/null; mv $G/run.log $G/run.log.void_probe_lockstep_oom 2>/dev/null
python -B -m boxes.langvb_eval --untrained --cal_n 16 --out_dir $G --device cuda:0 --parts evmem > $G/run.log 2>&1; rc=$?
echo "$(date '+%F %T') evmem2 rc $rc" >> logs/s218_gates.out
E=scratch/_tmp/e2e/boxes/runs_fsq
for R in lvb_sym_s1 lvb_direct_s1; do mv $E/$R/eval_lvb.json $E/$R/eval_lvb.json.void_oldcode 2>/dev/null; mv $E/$R/eval_direct.json $E/$R/eval_direct.json.void_oldcode 2>/dev/null; done
python -B -m boxes.langvb_eval --run $E/lvb_sym_s1 --device cuda:0 --per_D 2 --eval_bs 4 --parts panel,acc,stall,stall_selfcheck,vswap,cf,hist24,causal_deep,probe,fd,contamination --probe_steps 2 --pool_n 64 --fd_items 64 --chunk 4 --probe_group 6 > $E/lvb_sym_s1.eval2.log 2>&1
echo "$(date '+%F %T') e2e2 eval lslva rc $?" >> logs/s218_gates.out
python -B -m boxes.langvb_eval --run $E/lvb_direct_s1 --device cuda:0 --per_D 2 --eval_bs 4 --parts panel,direct,contamination > $E/lvb_direct_s1.eval2.log 2>&1
echo "$(date '+%F %T') e2e2 eval direct rc $?" >> logs/s218_gates.out
