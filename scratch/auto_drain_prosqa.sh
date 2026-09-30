#!/usr/bin/env bash
# ProsQA Vast watcher with automatic, verified draining (PROTOCOL8 Phase 0 and confirmatory hosts).
#   bash scratch/auto_drain_prosqa.sh "<host> <port> <tag> <instance id>" ...
# tag: p8-coconut (Phase 0 Coconut) or p8c_<arm>_s<seed> (arm coco|at|rt|cot).
# Every 10 min per host: an error marker in /workspace/logs/prosqa.out (Traceback, FAIL, NO_FINAL_CKPT, P8_TEST_FAILED,
# MASK_TEST_FAILED, TRAIN_EXIT non-zero) prints and exits (wakes the session). ANALYSIS_DONE runs scratch/drain_prosqa.sh
# with the tag's expected result files and final checkpoint. On DRAINED the host is dropped and logged in
# logs/auto_drain.log and logs/drained_hosts.txt; on any other drain result, it prints and exits. Unreachable 3x: exit.
cd "$(dirname "$0")/.."
declare -A fails
PODS=("$@")
spec() {  # tag -> "<local res dir> <remote ckpt> <local ckpt> <expected res files...>"
  case $1 in
    p8-coconut) echo "prosqa/res_p8_coconut /workspace/ckpt/p8-coconut/checkpoint_48 prosqa/ckpt/p8_coconut_checkpoint_48 coco_test_e48.json coco_test_e44.json coco_ctl_e48.json cot_test_e4.json" ;;
    p8c_cot_s*) a=${1#p8c_}; echo "prosqa/res_p8c/$a /workspace/ckpt/p8c-${a//_/-}/checkpoint_30 prosqa/ckpt/p8c_${a}_checkpoint_30 cot_test_e30.json cot_ctl_e30.json" ;;
    p8c_*) a=${1#p8c_}; echo "prosqa/res_p8c/$a /workspace/ckpt/p8c-${a//_/-}/checkpoint_48 prosqa/ckpt/p8c_${a}_checkpoint_48 coco_test_e48.json coco_ctl_e48.json" ;;
  esac
}
while :; do
  left=()
  for hp in "${PODS[@]}"; do
    read -r h p t id <<<"$hp"
    out=$(timeout 40 ssh -o ConnectTimeout=15 -i ~/.ssh/runpod_ed25519 -p $p root@$h "grep -a -c ANALYSIS_DONE /workspace/logs/prosqa.out; grep -a -E 'Traceback|^FAIL |NO_FINAL_CKPT|P8_TEST_FAILED|MASK_TEST_FAILED|TRAIN_EXIT [1-9]' /workspace/logs/prosqa.out | head -2" 2>/dev/null)
    if [ -z "$out" ]; then
      fails[$t]=$(( ${fails[$t]:-0} + 1 )); [ ${fails[$t]} -ge 3 ] && { echo "$(date +%T) $t UNREACHABLE ($h:$p)"; exit 0; }
      left+=("$hp"); continue
    fi
    fails[$t]=0
    ad=$(echo "$out" | sed -n 1p); err=$(echo "$out" | sed -n '2,$p')
    [ -n "$err" ] && { echo "$(date +%T) $t ERROR: $err"; exit 0; }
    if [ "$ad" -ge 1 ] 2>/dev/null; then
      read -r OUT RC LC files <<<"$(spec $t)"
      res=$(bash scratch/drain_prosqa.sh "$h" "$p" "$id" "$OUT" "$RC" "$LC" $files 2>&1)
      if echo "$res" | grep -q "^DRAINED $id"; then
        echo "$h:$p" >> logs/drained_hosts.txt
        echo "$(date '+%F %T') $t $id drained (res $OUT, ckpt $LC)" >> logs/auto_drain.log
        continue
      fi
      echo "$(date +%T) $t DRAIN FAILED:"; echo "$res" | tail -5; exit 0
    fi
    left+=("$hp")
  done
  PODS=("${left[@]}")
  [ ${#PODS[@]} -eq 0 ] && { echo "$(date +%T) ALL_DRAINED"; exit 0; }
  sleep 600
done
