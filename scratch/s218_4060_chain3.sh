#!/bin/bash
# §218 on the 4060: G-EVMEM with probe micro-chunks (pmicro), then the e2e probe / fd parts with the pinned group / pmicro.
cd /d/Latent
G=scratch/gate_s218/g_evmem; mv $G/eval_lvb.json $G/eval_lvb.json.void_probe_group_oom; mv $G/run.log $G/run.log.void_probe_group_oom
python -B -m boxes.langvb_eval --untrained --cal_n 16 --out_dir $G --device cuda:0 --parts evmem > $G/run.log 2>&1; rc=$?
echo "$(date '+%F %T') evmem3 rc $rc" >> logs/s218_gates.out
read -r GR PM <<<"$(python -c "import json; p = json.load(open('boxes/runs_fsq/lvb_gates/gate_evmem.json'))['probe']; print(p.get('group') or 0, p.get('pmicro') or 64)")"
E=scratch/_tmp/e2e/boxes/runs_fsq
python -B -m boxes.langvb_eval --run $E/lvb_sym_s1 --device cuda:0 --per_D 2 --eval_bs 4 --parts probe,fd --probe_steps 2 --pool_n 64 --fd_items 64 --chunk 4 --probe_group $GR --probe_pmicro $PM > $E/lvb_sym_s1.eval3.log 2>&1
echo "$(date '+%F %T') e2e3 eval probe,fd rc $? (group $GR pmicro $PM)" >> logs/s218_gates.out
