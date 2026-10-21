#!/bin/bash
# The 3070 (cuda:1) queue, r2d (10-21 06:3x: R3 stopped by pid, uninformative by Note 218-25; Audit / Theory: §221 next). Was r2c (Audit 10-21 05:5x, priority: §221 is the C1-on-language test): wait for the running sweep_R2_xor python (started by r2b,
# whose bash wrapper was stopped by pid while it ran) -> R3 lin + xor x1.5 -> §221 zen official fp32 + s221_gfid zen -> §219 round 2 (gates / label /
# fallback / pool) -> T1 eval_textself GPU parts -> §221 official bf16 (descriptive). Replaces s219_3070_queue_r2b.sh and s221_gfid_3070.sh (stopped by pid).
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s219_3070_queue.out
run() { local tag=$1; shift; "$@" > logs/s219_q_$tag.log 2>&1; local rc=$?; echo "$(date '+%F %T') $tag rc $rc" >> $L; return $rc; }
noslots() { python -c "
import json; e = json.load(open('boxes/runs_fsq/s219_gate_t2_s1/eval_gate.json')); print(','.join(w for w, v in e['window_labeller'].items() if v is None))"; }
rs() { local name=$1 arm=$2 lr=$3 steps=$4 m=$5; run $name python -B scratch/s218_gl5_known.py --device cuda:1 --arms $arm --k3_mults $m --probe_lr $lr --k3_steps $steps --out_name $name; }
run s221_zen_official_fp32 env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 third_party/venv_codi/Scripts/python.exe scratch/s221_official_zen.py --device cuda:1 --dtype fp32
run s221_gfid_zen python -B scratch/s221_gfid.py --model zen --device cuda:1
echo "$(date '+%F %T') s221 gfid zen (3070) step ended: $(ls scratch/s220/zen_codi_gpt2/gate_fid.json 2>/dev/null)" >> logs/s221_gfid.out
if run r2_gates python -B scratch/s219_build_pool.py --round 2 --stage gates && run r2_label python -B scratch/s219_gate.py --round 2 --stage label --labeller primary --device cuda:1 \
   && run r2_score python -B scratch/s219_gate.py --round 2 --stage score; then
  fw=$(noslots); echo "$(date '+%F %T') r2 windows without a passing primary labeller: '${fw}'" >> $L
  if [ -n "$fw" ]; then
    run r2_label_fallback python -B scratch/s219_gate.py --round 2 --stage label --labeller fallback --windows "$fw" --device cuda:1 && run r2_score2 python -B scratch/s219_gate.py --round 2 --stage score
    echo "$(date '+%F %T') r2 windows still without a labeller after the fallback: '$(noslots)'" >> $L
  fi
  run r2_pool python -B scratch/s219_build_pool.py --round 2 --stage pool --device cuda:1
  echo "$(date '+%F %T') §219 round-2 gates + pool step ended (see the rc lines; STAGE_BOUNDARY: commit eval_gate.json / pool_meta.json)" >> $L
fi
LATENT_MEM_CAP_GB=6.0 run t1_textself_gpu python -B -m boxes.text_eval --run boxes/runs_fsq/s219_t1_s1 --part textself --round 1 --parts gpu --device cuda:1
run s221_zen_official_bf16 env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 third_party/venv_codi/Scripts/python.exe scratch/s221_official_zen.py --device cuda:1 --dtype bf16 --out_dir scratch/s220/zen_codi_gpt2/official_bf16
echo "$(date '+%F %T') 3070 queue (r2d) done" >> $L
