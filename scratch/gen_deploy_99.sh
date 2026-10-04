#!/usr/bin/env bash
# §99 fleet: bash scratch/gen_deploy_99.sh "<offer> <n>" ... -> hosts gen-z<n> on scratch/gen_q_z<n>.txt, auto-drain gen4 (partial drains; 4x4 data only).
cd "$(dirname "$0")/.."
one() { o=$1; n=$2
  out=$(EXTRA_FILES="prosqa/ckpt/gpt2_pretrained.bin boxes/mult98.py boxes/accel99.py bench/data/mult_4x4_test.txt bench/data/mult_train/4x4_train.txt bench/data/mult_train/4x4_valid.txt" bash scratch/vdeploy_toy.sh $o gen-z$n scratch/gen_q_z$n.txt gen_z$n scratch/gen_vast_job.sh 2>&1)
  echo "$(date +%T) gen host z$n offer $o: $(echo "$out" | grep -E 'INSTANCE|NVIDIA|STARTED|BAD HOST|FAILED' | tr '\n' ' ')" >> logs/gen_deploy.log
  if echo "$out" | grep -q STARTED; then read -r _ id hh pp <<<"$(echo "$out" | grep '^INSTANCE')"
    nohup bash scratch/auto_drain_gen4.sh "$hh $pp gen_z$n $id gen" > logs/auto_drain_gen_z$n.out 2>&1 &
  else id=$(echo "$out" | grep '^INSTANCE' | awk '{print $2}'); [ -n "$id" ] && bash scratch/vdestroy.sh $id >> logs/gen_deploy.log 2>&1; fi; }
for x in "$@"; do set -- $x; one $1 $2 & sleep 3; done; wait
echo "$(date +%T) gen Z DEPLOY_DONE" >> logs/gen_deploy.log
