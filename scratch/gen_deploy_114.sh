#!/usr/bin/env bash
# §114 fleet: bash scratch/gen_deploy_114.sh "<offer> <n>" ... -> gen-l<n> on scratch/gen_q_l<n>.txt (loop114), auto-drain gen5.
cd "$(dirname "$0")/.."
one() { o=$1; n=$2
  out=$(EXTRA_FILES="prosqa/ckpt/gpt2_pretrained.bin boxes/mult98.py boxes/accel99.py boxes/loop114.py boxes/donor100.py bench/data/mult_4x4_test.txt bench/data/mult_train/4x4_train.txt bench/data/mult_train/4x4_valid.txt" bash scratch/vdeploy_toy.sh $o gen-l$n scratch/gen_q_l$n.txt gen_l$n scratch/gen_vast_job.sh 2>&1)
  echo "$(date +%T) gen host l$n offer $o: $(echo "$out" | grep -E 'INSTANCE|NVIDIA|STARTED|BAD HOST|FAILED' | tr '\n' ' ')" >> logs/gen_deploy.log
  if echo "$out" | grep -q STARTED; then read -r _ id hh pp <<<"$(echo "$out" | grep '^INSTANCE')"
    nohup bash scratch/auto_drain_gen5.sh "$hh $pp gen_l$n $id gen" > logs/auto_drain_gen_l$n.out 2>&1 &
  else id=$(echo "$out" | grep '^INSTANCE' | awk '{print $2}'); [ -n "$id" ] && bash scratch/vdestroy.sh $id >> logs/gen_deploy.log 2>&1; fi; }
for x in "$@"; do set -- $x; one $1 $2 & sleep 3; done; wait
echo "$(date +%T) gen L DEPLOY_DONE" >> logs/gen_deploy.log
