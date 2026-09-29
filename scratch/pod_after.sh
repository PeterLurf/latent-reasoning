#!/usr/bin/env bash
# On a pod: wait until every training job there has ended, then evaluate its runs.
#   nohup bash scratch/pod_after.sh <run>[:clock|:variant|:variantclock] ... &
# variant = PROTOCOL3 Part II battery (scratch/variant_pipeline.py) first; clock = add clock_freeze (P17).
cd /workspace/Latent
while pgrep -f "^sh -c while pgrep" >/dev/null || pgrep -f "^python scratch/stage1_guard.py" >/dev/null; do sleep 60; done
for it in "$@"; do
  r=${it%%:*}; mode=""; [ "$it" != "$r" ] && mode=${it#*:}
  [ -f "runs/$r/model.pt" ] || { echo "no model $r" >> logs/pod_after.log; continue; }
  case "$mode" in variant*) python scratch/variant_pipeline.py "runs/$r" cuda:0 "$r" >> "logs/eval_$r.log" 2>&1 ;; esac
  case "$mode" in *clock) c=clock ;; *) c="" ;; esac
  bash scratch/eval_battery.sh "$r" cuda:0 $c >> logs/pod_after.log 2>&1
done
touch logs/EVAL_ALL_DONE
