#!/bin/bash
# §218 A1 lvb_sym_s1 post-landing on the Stage A eval card (RTX 4060, cuda:0, fp32; LANG_v1 §4): waits for the gen_b316 drain (DRAINED / verified line)
# + model.pt, then for 4060 queue d's done line and no python on the 4060 (UUID GPU-be1bdfb5); runs the NON-PROBE registered parts at G-EVMEM's eval_bs:
# panel, acc (8), stall (32), vswap (16), cf (32), hist24 (8), causal_deep (16), contamination -> boxes/runs_fsq/lvb_sym_s1/eval_lvb.json.
# probe / fd (PLANT-dependent) are HELD (Note 218-18: route (iii); a probe successor needs its own Note + certification first).
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s218_ref_chain.out
until grep -q " gen_b316 " logs/auto_drain.log 2>/dev/null && [ -s boxes/runs_fsq/lvb_sym_s1/model.pt ]; do sleep 120; done
echo "$(date '+%F %T') §218 A1 landed: $(grep ' gen_b316 ' logs/auto_drain.log | tail -1)" >> $L
until grep -q "4060 queue d done" $L && ! nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader | grep "GPU-be1bdfb5" | grep -qi python; do sleep 60; done
python -B -m boxes.langvb_eval --run boxes/runs_fsq/lvb_sym_s1 --device cuda:0 --parts panel,acc,stall,vswap,cf,hist24,causal_deep,contamination --eval_bs acc:8,stall:32,vswap:16,cf:32,hist24:8,causal_deep:16 > logs/s218_a1_eval_nonprobe.log 2>&1; rc=$?
echo "$(date '+%F %T') §218 A1 eval_lvb NON-PROBE parts (4060) rc $rc $(ls boxes/runs_fsq/lvb_sym_s1/eval_lvb.json 2>/dev/null)" >> $L
