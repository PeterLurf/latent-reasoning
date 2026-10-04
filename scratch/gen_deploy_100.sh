#!/usr/bin/env bash
# §100 fleet (k = 2): bash scratch/gen_deploy_100.sh "<offer> <n>" ... -> gen-u<n> on scratch/gen_q_u<n>.txt, auto-drain gen4.
cd "$(dirname "$0")/.."
one() { o=$1; n=$2
  out=$(EXTRA_FILES="prosqa/ckpt/gpt2_pretrained.bin boxes/mult98.py bench/data/mult_4x4_test.txt bench/data/mult_5x5_test.txt bench/data/mult_train/4x4_train.txt bench/data/mult_train/5x5_train.txt" bash scratch/vdeploy_toy.sh $o gen-u$n scratch/gen_q_u$n.txt gen_u$n scratch/gen_vast_job.sh 2>&1)
  echo "$(date +%T) gen host u$n offer $o: $(echo "$out" | grep -E 'INSTANCE|NVIDIA|STARTED|BAD HOST|FAILED' | tr '\n' ' ')" >> logs/gen_deploy.log
  if echo "$out" | grep -q STARTED; then read -r _ id hh pp <<<"$(echo "$out" | grep '^INSTANCE')"
    nohup bash scratch/auto_drain_gen4.sh "$hh $pp gen_u$n $id gen" > logs/auto_drain_gen_u$n.out 2>&1 &
  else id=$(echo "$out" | grep '^INSTANCE' | awk '{print $2}'); [ -n "$id" ] && bash scratch/vdestroy.sh $id >> logs/gen_deploy.log 2>&1; fi; }
for x in "$@"; do set -- $x; one $1 $2 & sleep 3; done; wait
echo "$(date +%T) gen U DEPLOY_DONE" >> logs/gen_deploy.log
