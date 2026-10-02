#!/usr/bin/env bash
# Rent a Vast offer and start a toy-model queue on it (direct SSH):
#   bash scratch/vdeploy_toy.sh <offer id> <label> <queue file> <queue tag> [worker script, default queue_worker2.sh]
# Uploads latent/ + the queue worker, stage-1 guard, eval battery and the queue file to /workspace/Latent, then starts
# scratch/queue_worker2.sh <tag> cuda:0 <queue file>. Prints INSTANCE <id> <host> <port>, the cuInit/clock line, and
# STARTED; prints BAD HOST if cuInit fails (the instance is left for the caller to destroy).
set -u
O=$1; LBL=$2; Q=$3; TAG=$4; W=${5:-scratch/queue_worker2.sh}
EXTRA=""; [ "$W" = scratch/queue_worker3.sh ] && EXTRA="scratch/queue_worker3.sh scratch/p9_frozen"
[ "$W" = scratch/queue_worker_t12.sh ] && EXTRA="scratch/queue_worker_t12.sh scratch/t12_frozen"
[ "$W" = scratch/queue_worker_f6.sh ] && EXTRA="scratch/queue_worker_f6.sh scratch/f6_frozen"
[ "$W" = scratch/boxes_vast_job.sh ] && EXTRA="scratch/boxes_vast_job.sh boxes/__init__.py boxes/core.py boxes/train.py bench/__init__.py bench/run.py bench/tasks.py bench/models.py bench/report.py bench/literature.json"
[ "$W" = scratch/f1sb_vast_job.sh ] && EXTRA="scratch/f1sb_vast_job.sh prosqa/ail_gpt2.py prosqa/ail_f1s.py prosqa/gpt2c.py prosqa/analyze.py prosqa/eval_p8.py bench/__init__.py bench/run.py bench/tasks.py bench/models.py bench/report.py bench/literature.json bench/data prosqa/p8data/f1s_train.json prosqa/p8data/f1s_test_shallow.json prosqa/p8data/p8_na_test.json prosqa/p8data/p8_test.json"
[ "$W" = scratch/bench_vast_job.sh ] && EXTRA="scratch/bench_vast_job.sh bench/__init__.py bench/run.py bench/tasks.py bench/models.py bench/report.py bench/literature.json bench/data prosqa/gpt2c.py prosqa/analyze.py prosqa/eval_p8.py prosqa/ail_gpt2.py prosqa/p8data/p8_test.json prosqa/p8data/p8_na_test.json"
[ "$W" = scratch/tjob_vast.sh ] && EXTRA="scratch/tjob_vast.sh $(echo theory/*.py)"
[ "$W" = scratch/tjob_vast2.sh ] && EXTRA="scratch/tjob_vast2.sh $(echo theory/*.py)"
[ "$W" = scratch/gsm_time_vast.sh ] && EXTRA="scratch/gsm_time_vast.sh"
[ "$W" = scratch/s71_gpt_job.sh ] && EXTRA="scratch/s71_gpt_job.sh scratch/s71_gen boxes/__init__.py boxes/core.py boxes/s5.py"
[ "$W" = scratch/s71_dp_job.sh ] && EXTRA="scratch/s71_dp_job.sh boxes/s71_dp_eval.py"
[ "$W" = scratch/c1_vast_job.sh ] && EXTRA="scratch/c1_vast_job.sh c1/__init__.py c1/core.py c1/train.py boxes/__init__.py boxes/core.py"
[ "$W" = scratch/gsm_time_vast2.sh ] && EXTRA="scratch/gsm_time_vast2.sh s68/__init__.py s68/ws.py s68/ws_data.py"
[ "$W" = scratch/f4_vast_job.sh ] && EXTRA="scratch/f4_vast_job.sh $(echo theory/*.py)"
[ "$W" = scratch/p8na_vast_job.sh ] && EXTRA="scratch/p8na_vast_job.sh prosqa/eval_na_p8.py prosqa/eval_p8.py prosqa/analyze.py prosqa/gpt2c.py prosqa/p8data/p8_test.json prosqa/p8data/p8_na_test.json"
[ "$W" = scratch/p8na_vast_par.sh ] && EXTRA="scratch/p8na_vast_par.sh prosqa/eval_na_p8.py prosqa/eval_p8.py prosqa/analyze.py prosqa/gpt2c.py prosqa/p8data/p8_test.json prosqa/p8data/p8_na_test.json"
[ "$W" = scratch/f1s_vast_job.sh ] && EXTRA="scratch/f1s_vast_job.sh prosqa/ail_gpt2.py prosqa/ail_f1s.py prosqa/p8data/f1s_train.json prosqa/p8data/f1s_test_shallow.json prosqa/p8data/p8_na_test.json"
[ "$W" = scratch/ail_vast_job.sh ] && EXTRA="scratch/ail_vast_job.sh prosqa/ail_gpt2.py prosqa/p8data/p8_train.json prosqa/p8data/p8_test.json"
[ "$W" = scratch/bm_vast_job.sh ] && EXTRA="scratch/bm_vast_job.sh bm/__init__.py bm/data_b.py bm/model_b.py bm/run_b.py bm/data.py bm/model.py bm/run.py bm/data_pool_b"
V="C:/Users/peter/.local/vastai/vast.sh"
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."
ID=$(bash "$V" create instance $O --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel --disk 30 --ssh --direct --label $LBL --raw 2>&1 | grep -o '"new_contract": *[0-9]*' | grep -o '[0-9]*$')
[ -n "$ID" ] || { echo "CREATE FAILED"; exit 1; }
echo "$ID vast offer $O created $(date '+%Y-%m-%d %H:%M'): label $LBL, queue $Q (logged at create by vdeploy_toy.sh)" >> scratch/pods.txt
for i in $(seq 1 60); do
  j=$(bash "$V" show instance $ID --raw 2>/dev/null)
  H=$(echo "$j" | python -c "import json,sys;d=json.load(sys.stdin);print(d.get('public_ipaddr','').strip())" 2>/dev/null)
  P=$(echo "$j" | python -c "import json,sys;d=json.load(sys.stdin);print(d['ports']['22/tcp'][0]['HostPort'])" 2>/dev/null)
  [ -n "$H" ] && [ -n "$P" ] && timeout 30 ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -i ~/.ssh/runpod_ed25519 -p $P root@$H true 2>/dev/null && break
  sleep 20
done
echo "INSTANCE $ID $H $P"
echo "$H $P $LBL" >> scratch/live_hosts.txt          # watched by gpu_watch2.sh from creation (Audit)
S="ssh -o ConnectTimeout=20 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
$S "python -c \"import ctypes;print('CUINIT',ctypes.CDLL('libcuda.so.1').cuInit(0))\"; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader" 2>/dev/null | grep -v -i "welcome\|have fun"
$S "python -c \"import ctypes;exit(ctypes.CDLL('libcuda.so.1').cuInit(0))\"" 2>/dev/null || { echo "BAD HOST"; exit 2; }
tar -czf /tmp/up_$LBL.tgz --exclude=__pycache__ latent scratch/queue_worker2.sh scratch/stage1_guard.py scratch/eval_battery.sh $EXTRA ${EXTRA_FILES:-} $Q
echo "UPLOAD_BYTES $(stat -c %s /tmp/up_$LBL.tgz) START $(date +%s)"
scp -q -i ~/.ssh/runpod_ed25519 -P $P /tmp/up_$LBL.tgz root@$H:/tmp/up.tgz
echo "UPLOAD_DONE $(date +%s)"
$S "mkdir -p /workspace/Latent && cd /workspace/Latent && tar --no-same-owner -xzf /tmp/up.tgz && mkdir -p logs runs results && (setsid nohup bash $W $TAG cuda:0 $Q > /dev/null 2>&1 < /dev/null &) ; sleep 5; pgrep -fa queue_worker | head -1" 2>/dev/null | grep -q queue_worker && echo STARTED
