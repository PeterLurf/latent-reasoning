#!/usr/bin/env bash
# v3 (2026-10-14, §200 \$7 cap): requeue_watch2 + a per-queue offer price cap (scratch/gen_q_b<n>.dph -> fill_generic_px DPH_MAX, default 0.30); fresh start marker.
# v2 (2026-10-10, Audit's hardening): + scratch/gen_q_b<n>.extra (a sidecar listing the queue's EXTRA_120 data files, e.g. ProsQA p8) is passed on the
# requeue (rq256 / rq260 deployed without the p8 data); + heartbeat logs/requeue_watch3.hb; its own start marker (kept if present) and stop file.
# Setup-failure requeue watchdog (Audit 2026-10-07 01:48: b92 / b96 were STARTED, failed a setup gate (rc 9: HOSTCHECK_FAIL /
# DATA_MISSING / DATA_SHA_MISMATCH / PIP_FAILED / GPT2_SHA_MISMATCH) before any training, were drained + destroyed, and the fillers had
# already counted them as deployed -> the queue was silently dropped). Every 5 min: a drained host's fetched queue log
# (logs/vast_<ip>_<port>/queue_gen_b<n>.out) newer than this watchdog's start with an rc 9 setup line -> one fill_generic run for that
# queue (tag rq<n>), once per queue file, with the extra files the queue needs. Never touches running hosts or other fillers.
#   nohup bash scratch/requeue_watch3.sh > logs/requeue_watch3.out 2>&1 &        stop: touch scratch/requeue_watch3.stop
cd "$(dirname "$0")/.."
touch logs/.requeue_watch3_start; done_list=" "                       # v3: always a fresh start marker (failures before this start are not requeued)
log() { echo "$(date '+%F %T') requeue_watch3 $*" >> logs/gen_deploy.log; echo "$(date '+%F %T') $*"; }
log "start"
while [ ! -f scratch/requeue_watch3.stop ]; do
  date +%s > logs/requeue_watch3.hb
  for f in $(find logs -path 'logs/vast_*' -name 'queue_gen_b*.out' -newer logs/.requeue_watch3_start 2>/dev/null); do
    grep -q -E "(HOSTCHECK_FAIL|DATA_MISSING|DATA_SHA_MISMATCH|PIP_FAILED|GPT2_SHA_MISMATCH|HOSTPROBE_SLOW|MODULE_MISSING [a-z0-9_.]+) rc 9" "$f" || continue   # MODULE_MISSING: 2026-10-07 (§138)
    if grep -q "HOSTPROBE_SLOW rc 9" "$f"; then                             # 2026-10-09 (Audit): a slow host is excluded so the requeue lands elsewhere
      ip_=$(echo "$f" | sed -E 's|.*logs/vast_([0-9.]+)_[0-9]+/.*|\1|'); grep -q "^$ip_ " scratch/vast_exclude_ips.txt 2>/dev/null || echo "$ip_ # HOSTPROBE_SLOW $(grep -o 'HOSTPROBE_SLOW rc 9 ([^)]*)' "$f" | head -1) $(date '+%F %T')" >> scratch/vast_exclude_ips.txt
    fi
    grep -q " start boxes" "$f" && continue                                 # something trained: not a setup-only failure
    n=$(basename "$f" .out | sed 's/queue_gen_b//'); key="$f"
    case "$done_list" in *" $key "*) continue;; esac
    done_list="$done_list$key "; q=scratch/gen_q_b$n.txt; [ -f "$q" ] || { log "b$n: no $q, skipped"; continue; }
    ex="scratch/hostcheck.py"
    grep -q "cc_job" "$q" && ex="$ex scratch/cc_job.py scratch/cert_cover128.py"
    grep -q -E "multcol|restart137" "$q" && ex="$ex boxes/multcol.py scratch/restart137.py $(grep -o -E -- '--task [45]x[45]' "$q" | awk '{print "bench/data/mult_"$2"_test.txt bench/data/mult_train/"$2"_train.txt"}' | sort -u | tr '\n' ' ')"
    grep -q "restart133" "$q" && ex="$ex scratch/restart133.py"
    [ -f "scratch/gen_q_b$n.extra" ] && ex="$ex $(tr '\n' ' ' < "scratch/gen_q_b$n.extra")"   # v2: the queue's EXTRA_120 data files
    log "b$n: setup failure in $f ($(grep -o -E "[A-Z_]+( [a-z0-9_.]+)? rc 9" "$f" | head -1)) -> requeue via fill_generic rq$n"
    rm -f scratch/fill_rq$n.stop; EXTRA_120="$ex" DPH_MAX="$(cat scratch/gen_q_b$n.dph 2>/dev/null || echo 0.30)" nohup bash scratch/fill_generic_px.sh rq$n "$n" > logs/fill_rq$n.out 2>&1 &
  done
  sleep 300
done
log "stopped"
