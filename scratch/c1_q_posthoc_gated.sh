#!/usr/bin/env bash
# Local short post-hoc queue under the user's rule (2026-10-06): "only put processes on local cards that wont overfill" - one job per
# card, nothing beside it; card memory (nvidia-smi, the card otherwise idle) sampled every 2 s while each job runs. The FIRST job is
# the memory check under the exact flags: if its peak leaves < 1.5 GiB of the card free, the queue stops (GATE_FAIL) before the rest.
# Later jobs over the limit are logged OVER_LIMIT and the queue stops too. Skips runs whose output exists. Stop file: scratch/<tag>.stop.
#   nohup bash scratch/c1_q_posthoc_gated.sh <tag> <torch device> "<GPU name substring>" <mode: contraction|cert> <run dir> ... &
cd "$(dirname "$0")/.."
tag=$1; dev=$2; gname=$3; mode=$4; shift 4
idx=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | grep "$gname" | head -1 | cut -d, -f1)
total=$(nvidia-smi --id=$idx --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' \r'); limit=$((total - 1536))
echo "$(date +%T) $tag start: $mode on $dev (nvidia-smi $idx '$gname', limit $limit MiB of $total)" >> logs/c1_local.log
first=1
for d in "$@"; do
  [ -f scratch/$tag.stop ] && { echo "$(date +%T) $tag STOPPED (stop file)" >> logs/c1_local.log; break; }
  n=$(basename $d)
  if [ $mode = contraction ]; then out=$d/eval_contraction_fp32.json; F="--contraction_only --fp32"; else out=$d/eval_cert.json; F="--cert_only"; fi
  [ -s $out ] && continue
  [ -s $d/model.pt ] || { echo "$(date +%T) $tag $n SKIPPED (no model.pt)" >> logs/c1_local.log; continue; }
  OMP_NUM_THREADS=3 python -u -m boxes.fsq120 --from_args $d/args.json $F --device $dev --out $d > logs/c1_${n}_${mode}_${tag}.log 2>&1 &
  pid=$!; peak=0
  while kill -0 $pid 2>/dev/null; do                                      # $pid is this shell's own child (MSYS pid): kill -0 is valid here
    u=$(nvidia-smi --id=$idx --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' \r'); [ "$u" -gt "$peak" ] 2>/dev/null && peak=$u; sleep 2
  done
  wait $pid; rc=$?
  echo "$(date +%T) $tag $n $mode rc $rc out $([ -s $out ] && echo yes || echo NO) peak_card_mem ${peak} MiB of $total ($dev)" >> logs/c1_local.log
  if [ "$peak" -gt "$limit" ]; then
    echo "$(date +%T) $tag $([ $first = 1 ] && echo GATE_FAIL || echo OVER_LIMIT): peak $peak MiB > $limit; queue stopped" >> logs/c1_local.log; break
  fi
  first=0
done
echo "$(date +%T) $tag done" >> logs/c1_local.log
