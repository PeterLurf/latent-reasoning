#!/usr/bin/env bash
# §98 fleet: bash scratch/gen_deploy_98.sh "<offer> <n>" ... -> hosts gen-y<n> on scratch/gen_q_y<n>.txt, auto-drain gen3 (data: mult train/test files uploaded).
cd "$(dirname "$0")/.."
one() { o=$1; n=$2
  out=$(EXTRA_FILES="prosqa/ckpt/gpt2_pretrained.bin boxes/mult98.py bench/data/mult_4x4_test.txt bench/data/mult_5x5_test.txt bench/data/mult_train/4x4_train.txt bench/data/mult_train/5x5_train.txt" bash scratch/vdeploy_toy.sh $o gen-y$n scratch/gen_q_y$n.txt gen_y$n scratch/gen_vast_job.sh 2>&1)
  echo "$(date +%T) gen host y$n offer $o: $(echo "$out" | grep -E 'INSTANCE|NVIDIA|STARTED|BAD HOST|FAILED' | tr '\n' ' ')" >> logs/gen_deploy.log
  if echo "$out" | grep -q STARTED; then read -r _ id hh pp <<<"$(echo "$out" | grep '^INSTANCE')"
    nohup bash scratch/auto_drain_gen3.sh "$hh $pp gen_y$n $id gen" > logs/auto_drain_gen_y$n.out 2>&1 &
  else id=$(echo "$out" | grep '^INSTANCE' | awk '{print $2}'); [ -n "$id" ] && bash scratch/vdestroy.sh $id >> logs/gen_deploy.log 2>&1; fi; }
for x in "$@"; do set -- $x; one $1 $2 & sleep 3; done; wait
echo "$(date +%T) gen Y DEPLOY_DONE" >> logs/gen_deploy.log
