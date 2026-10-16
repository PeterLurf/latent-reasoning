#!/usr/bin/env bash
# Rent a Vast offer and start a toy-model queue on it (direct SSH):
#   bash scratch/vdeploy_toy.sh <offer id> <label> <queue file> <queue tag> [worker script, default queue_worker2.sh]
# Uploads latent/ + the queue worker, stage-1 guard, eval battery and the queue file to /workspace/Latent, then starts
# scratch/queue_worker2.sh <tag> cuda:0 <queue file>. Prints INSTANCE <id> <host> <port>, the cuInit/clock line, and
# STARTED; prints BAD HOST if cuInit fails (the instance is left for the caller to destroy).
set -u
T00=$(date +%s)                                                          # 2026-10-12 (b273: 4 bare CREATE FAILEDs): timing + the raw create reply on failure
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
[ "$W" = scratch/s71_gru_job.sh ] && EXTRA="scratch/s71_gru_job.sh scratch/s71_gen boxes/s71_gru_eval.py"
[ "$W" = scratch/s71_dp_job.sh ] && EXTRA="scratch/s71_dp_job.sh boxes/s71_dp_eval.py"
[ "$W" = scratch/c1_vast_job.sh ] && EXTRA="scratch/c1_vast_job.sh c1/__init__.py c1/core.py c1/train.py boxes/__init__.py boxes/core.py"
[ "$W" = scratch/gsm_time_vast2.sh ] && EXTRA="scratch/gsm_time_vast2.sh s68/__init__.py s68/ws.py s68/ws_data.py"
[ "$W" = scratch/f4_vast_job.sh ] && EXTRA="scratch/f4_vast_job.sh $(echo theory/*.py)"
[ "$W" = scratch/p8na_vast_job.sh ] && EXTRA="scratch/p8na_vast_job.sh prosqa/eval_na_p8.py prosqa/eval_p8.py prosqa/analyze.py prosqa/gpt2c.py prosqa/p8data/p8_test.json prosqa/p8data/p8_na_test.json"
[ "$W" = scratch/p8na_vast_par.sh ] && EXTRA="scratch/p8na_vast_par.sh prosqa/eval_na_p8.py prosqa/eval_p8.py prosqa/analyze.py prosqa/gpt2c.py prosqa/p8data/p8_test.json prosqa/p8data/p8_na_test.json"
[ "$W" = scratch/f1s_vast_job.sh ] && EXTRA="scratch/f1s_vast_job.sh prosqa/ail_gpt2.py prosqa/ail_f1s.py prosqa/p8data/f1s_train.json prosqa/p8data/f1s_test_shallow.json prosqa/p8data/p8_na_test.json"
[ "$W" = scratch/ail_vast_job.sh ] && EXTRA="scratch/ail_vast_job.sh prosqa/ail_gpt2.py prosqa/p8data/p8_train.json prosqa/p8data/p8_test.json"
[ "$W" = scratch/gen_vast_job.sh ] && EXTRA="scratch/gen_vast_job.sh c1/__init__.py c1/p80_p0.py c1/p90_split_S4_k2.json boxes/__init__.py boxes/core.py boxes/s5.py boxes/memcap.py boxes/latent86.py scratch/s71_gen bench/data/s5/S5=2048.csv"
[ "$W" = scratch/g78_vast_job.sh ] && EXTRA="scratch/g78_vast_job.sh gsmlen/train78.py gsmlen/d78"
[ "$W" = scratch/bm_vast_job.sh ] && EXTRA="scratch/bm_vast_job.sh bm/__init__.py bm/data_b.py bm/model_b.py bm/run_b.py bm/data.py bm/model.py bm/run.py bm/data_pool_b"
V="C:/Users/peter/.local/vastai/vast.sh"
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."
[ -n "${SMOKE_PLAIN_OK:-}" ] && [ "$SMOKE_PLAIN_OK" = "$(git rev-parse HEAD)-$(git status --porcelain -- 'boxes/*.py' scratch/smoke_plain.py | md5sum | cut -c1-8)" ] || python scratch/smoke_plain.py > /tmp/smoke_plain_$LBL.log 2>&1 || { echo "FAILED smoke_plain (plain-path smoke, Audit 2026-10-06): $(tail -n 2 /tmp/smoke_plain_$LBL.log | tr '\n' ' ')"; exit 4; }   # before creating the instance (a failure costs nothing)
# Fleet cap (Audit 2026-10-07 07:49: concurrent fillers each counted the same free slot -> 17 hosts): the count check and the create are
# serialised across all deploys by a lock dir; at >= 16 live instances the create is refused (the filler retries later).
CAPLOCK=scratch/.vdeploy_cap.lock; for w in $(seq 1 60); do mkdir $CAPLOCK 2>/dev/null && break; sleep 5; done
NLIVE=$(bash "$V" show instances --raw 2>/dev/null | python -c "import json,sys;print(len(json.load(sys.stdin)))" 2>/dev/null)
if [ -z "$NLIVE" ] || [ "$NLIVE" -ge 16 ]; then rmdir $CAPLOCK 2>/dev/null; echo "CREATE FAILED (fleet cap: ${NLIVE:-unknown} live >= 16)"; exit 1; fi
T0C=$(date +%s); CR=$(bash "$V" create instance $O --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel --disk 30 --ssh --direct --label $LBL --raw 2>&1)
ID=$(echo "$CR" | grep -o '"new_contract": *[0-9]*' | grep -o '[0-9]*$')
sleep 3; rmdir $CAPLOCK 2>/dev/null                                      # the new instance is listed before the next deploy counts
[ -n "$ID" ] || { echo "CREATE FAILED (create call $(( $(date +%s) - T0C )) s, $(( T0C - T00 )) s after vdeploy start): $(echo "$CR" | tr '\n' ' ' | sed -E 's/api_key=[^&" ]*/api_key=REDACTED/g; s/[0-9a-fA-F]{40,}/<redacted>/g' | cut -c1-400)"; exit 1; }   # no key in logs
echo "$ID vast offer $O created $(date '+%Y-%m-%d %H:%M'): label $LBL, queue $Q (logged at create by vdeploy_toy.sh)" >> scratch/pods.txt
# STUCK_POLL_BEGIN (Audit 2026-10-09 ~17:16: b240 lost 3 hosts that read cur_state stopped mid image pull) - a poll counts toward STUCK_LOADING only if
# cur_state is stopped AND status_msg shows no image-pull progress (actual_status dropped 18:1x: real stuck hosts read loading); the 60-poll (20-min) loop stays the hard cap.
stuck_poll() { { [ "$1" = "stopped" ] || echo "$3" | grep -qi "retrying in"; } && ! echo "$3" | grep -qiE "pull|download|extract|waiting"; }   # 2026-10-15 (Audit): + cur_state running with an image-pull "Retrying in" loop (b303 56183853 sat 22 min)   # 17:5x: b241 sat loading / stopped / empty msg 20+ min -> actual_status is not a progress signal ($2 kept for the log)
# STUCK_POLL_END
for i in $(seq 1 60); do
  j=$(bash "$V" show instance $ID --raw 2>/dev/null)
  H=$(echo "$j" | python -c "import json,sys;d=json.load(sys.stdin);print(d.get('public_ipaddr','').strip())" 2>/dev/null)
  P=$(echo "$j" | python -c "import json,sys;d=json.load(sys.stdin);print(d['ports']['22/tcp'][0]['HostPort'])" 2>/dev/null)
  CS=$(echo "$j" | python -c "import json,sys;d=json.load(sys.stdin);print(d.get('cur_state') or '')" 2>/dev/null)   # 2026-10-08: hosts stuck 'loading' with cur_state stopped
  AS=$(echo "$j" | python -c "import json,sys;d=json.load(sys.stdin);print(d.get('actual_status') or '')" 2>/dev/null)
  SM=$(echo "$j" | python -c "import json,sys;d=json.load(sys.stdin);print(' '.join((d.get('status_msg') or '').split()))" 2>/dev/null)
  if stuck_poll "$CS" "$AS" "$SM"; then NST=$(( ${NST:-0} + 1 )); else NST=0; fi
  [ "${NST:-0}" -ge 18 ] && { echo "STUCK_LOADING cur_state stopped, no pull progress for 6 min (actual_status $AS)"; echo "INSTANCE $ID ${H:-none} ${P:-none}"; echo "BAD HOST"; exit 2; }   # INSTANCE -> gen_deploy destroys it
  [ -n "$H" ] && [ -n "$P" ] && [ -z "${KR:-}" ] && { ssh-keygen -R "[$H]:$P" > /dev/null 2>&1; KR=1; }   # 2026-10-07 (b156): a recycled ip:port kept an old key; cleared ONCE per new instance (Audit: not per retry)
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
LSUM=$(sha256sum /tmp/up_$LBL.tgz | cut -c1-64)                    # 2026-10-04: e7's upload arrived corrupt (gzip CRC) and was never checked
for try in 1 2 3; do
  scp -q -i ~/.ssh/runpod_ed25519 -P $P /tmp/up_$LBL.tgz root@$H:/tmp/up.tgz
  [ "$($S "sha256sum /tmp/up.tgz" 2>/dev/null | cut -c1-64)" = "$LSUM" ] && break
  echo "UPLOAD_SHA_MISMATCH try $try"; [ $try = 3 ] && { rm -f /tmp/up_$LBL.tgz; echo "FAILED upload"; exit 3; }
done
rm -f /tmp/up_$LBL.tgz   # upload bundle no longer needed (C: filled up 2026-10-03)
echo "UPLOAD_DONE $(date +%s)"
WB=$(basename $W); WP="[${WB:0:1}]${WB:1}"                          # bracket pattern: pgrep must not match this ssh command itself (it did: false STARTED)
$S "mkdir -p /workspace/Latent && cd /workspace/Latent && tar --no-same-owner -xzf /tmp/up.tgz && rm -f /tmp/up.tgz && mkdir -p logs runs results && (IMAGE_TAG=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel setsid nohup bash $W $TAG cuda:0 $Q > /dev/null 2>&1 < /dev/null &) ; sleep 8; [ -d /workspace/Latent/boxes ] && pgrep -f '$WP' > /dev/null && echo JOB_RUNNING" 2>/dev/null | grep -q JOB_RUNNING && echo STARTED
