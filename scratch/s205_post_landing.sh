#!/usr/bin/env bash
# §205: run scratch/s205_eval_iit.py on each seed right after its gen8 drain reports ALL_DRAINED (eval.json present), in landing order; one done-file per seed.
#   nohup bash scratch/s205_post_landing.sh >> logs/s205_post_landing.out 2>&1 &
cd /d/Latent
declare -A RUN=( [298]=s205_iit_s3 [297]=s205_iit_s2 [296]=s205_iit_s1 )
left="298 297 296"
while [ -n "$left" ]; do
  nl=""
  for n in $left; do
    r=${RUN[$n]}
    if grep -q "ALL_DRAINED" logs/auto_drain_gen_b$n.out 2>/dev/null && [ -s boxes/runs_fsq/$r/eval.json ]; then
      echo "$(date '+%F %T') b$n $r drained -> eval_iit"; python scratch/s205_eval_iit.py $r --device cuda:1 > logs/s205_eval_iit_$r.log 2>&1
      echo "$(date '+%F %T') b$n $r eval_iit rc $? $(tail -n 1 logs/s205_eval_iit_$r.log | cut -c1-200)"
    else
      nl="$nl $n"
    fi
  done
  left=$(echo $nl); [ -n "$left" ] && sleep 120
done
echo "$(date '+%F %T') all three eval_iit done"
