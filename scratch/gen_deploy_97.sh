#!/usr/bin/env bash
# §97 fleet: bash scratch/gen_deploy_97.sh "<offer> <n>" ... -> hosts gen-x<n> on scratch/gen_q_x<n>.txt (n 1-3 = GEN-4 / GEN-1 / HYB-4 s1-3; n 4-5 = GEN-4 / HYB-4 s4-5, C6), auto-drain gen3.
cd "$(dirname "$0")/.."
one() { o=$1; n=$2
  out=$(EXTRA_FILES="prosqa/ckpt/gpt2_pretrained.bin boxes/gen97.py" bash scratch/vdeploy_toy.sh $o gen-x$n scratch/gen_q_x$n.txt gen_x$n scratch/gen_vast_job.sh 2>&1)
  echo "$(date +%T) gen host x$n offer $o: $(echo "$out" | grep -E 'INSTANCE|NVIDIA|STARTED|BAD HOST|FAILED' | tr '\n' ' ')" >> logs/gen_deploy.log
  if echo "$out" | grep -q STARTED; then read -r _ id hh pp <<<"$(echo "$out" | grep '^INSTANCE')"
    nohup bash scratch/auto_drain_gen3.sh "$hh $pp gen_x$n $id gen" > logs/auto_drain_gen_x$n.out 2>&1 &
  else id=$(echo "$out" | grep '^INSTANCE' | awk '{print $2}'); [ -n "$id" ] && bash scratch/vdestroy.sh $id >> logs/gen_deploy.log 2>&1; fi; }
for x in "$@"; do set -- $x; one $1 $2 & sleep 3; done; wait
echo "$(date +%T) gen X DEPLOY_DONE" >> logs/gen_deploy.log
