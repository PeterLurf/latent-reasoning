#!/usr/bin/env bash
# §101 fleet: bash scratch/gen_deploy_101.sh "<offer> <n>" -> gen-s<n> on scratch/gen_q_s<n>.txt (snap101), auto-drain gen5.
cd "$(dirname "$0")/.."
one() { o=$1; n=$2
  out=$(EXTRA_FILES="prosqa/ckpt/gpt2_pretrained.bin boxes/gen97.py boxes/snap101.py" bash scratch/vdeploy_toy.sh $o gen-s$n scratch/gen_q_s$n.txt gen_s$n scratch/gen_vast_job.sh 2>&1)
  echo "$(date +%T) gen host s$n offer $o: $(echo "$out" | grep -E 'INSTANCE|NVIDIA|STARTED|BAD HOST|FAILED' | tr '\n' ' ')" >> logs/gen_deploy.log
  if echo "$out" | grep -q STARTED; then read -r _ id hh pp <<<"$(echo "$out" | grep '^INSTANCE')"
    nohup bash scratch/auto_drain_gen5.sh "$hh $pp gen_s$n $id gen" > logs/auto_drain_gen_s$n.out 2>&1 &
  else id=$(echo "$out" | grep '^INSTANCE' | awk '{print $2}'); [ -n "$id" ] && bash scratch/vdestroy.sh $id >> logs/gen_deploy.log 2>&1; fi; }
for x in "$@"; do set -- $x; one $1 $2 & sleep 3; done; wait
echo "$(date +%T) gen S DEPLOY_DONE" >> logs/gen_deploy.log
