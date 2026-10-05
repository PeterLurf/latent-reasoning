#!/usr/bin/env bash
# §116 hosts: bash scratch/gen_deploy_116.sh "<offer> <n>" -> gen-o<n> on scratch/gen_q_o<n>.txt (self110 --opdef / --no_r0), auto-drain gen5.
cd "$(dirname "$0")/.."
one() { o=$1; n=$2
  out=$(EXTRA_FILES="prosqa/ckpt/gpt2_pretrained.bin boxes/gen97.py boxes/self103.py boxes/mk108.py boxes/self110.py" bash scratch/vdeploy_toy.sh $o gen-o$n scratch/gen_q_o$n.txt gen_o$n scratch/gen_vast_job.sh 2>&1)
  echo "$(date +%T) gen host o$n offer $o: $(echo "$out" | grep -E 'INSTANCE|NVIDIA|STARTED|BAD HOST|FAILED' | tr '\n' ' ')" >> logs/gen_deploy.log
  if echo "$out" | grep -q STARTED; then read -r _ id hh pp <<<"$(echo "$out" | grep '^INSTANCE')"
    nohup bash scratch/auto_drain_gen5.sh "$hh $pp gen_o$n $id gen" > logs/auto_drain_gen_o$n.out 2>&1 &
  else id=$(echo "$out" | grep '^INSTANCE' | awk '{print $2}'); [ -n "$id" ] && bash scratch/vdestroy.sh $id >> logs/gen_deploy.log 2>&1; fi; }
for x in "$@"; do set -- $x; one $1 $2 & sleep 3; done; wait
echo "$(date +%T) gen O DEPLOY_DONE" >> logs/gen_deploy.log
