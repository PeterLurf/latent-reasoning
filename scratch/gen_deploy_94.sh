#!/usr/bin/env bash
# §91/§92 fleet: bash scratch/gen_deploy.sh "<offer> <n>" ... -> hosts gen-h<n> on scratch/gen_q_h<n>.txt, auto-drain "gen".
cd "$(dirname "$0")/.."
one() { o=$1; n=$2
  out=$(EXTRA_FILES="prosqa/ckpt/gpt2_pretrained.bin boxes/fade89.py boxes/train.py bench/__init__.py bench/tasks.py" bash scratch/vdeploy_toy.sh $o gen-w$n scratch/gen_q_w$n.txt gen_w$n scratch/gen_vast_job.sh 2>&1)
  echo "$(date +%T) gen host w$n offer $o: $(echo "$out" | grep -E 'INSTANCE|NVIDIA|STARTED|BAD HOST|FAILED' | tr '\n' ' ')" >> logs/gen_deploy.log
  if echo "$out" | grep -q STARTED; then read -r _ id hh pp <<<"$(echo "$out" | grep '^INSTANCE')"
    nohup bash scratch/auto_drain_gen3.sh "$hh $pp gen_w$n $id gen" > logs/auto_drain_gen_w$n.out 2>&1 &
  else id=$(echo "$out" | grep '^INSTANCE' | awk '{print $2}'); [ -n "$id" ] && bash scratch/vdestroy.sh $id >> logs/gen_deploy.log 2>&1; fi; }
for x in "$@"; do set -- $x; one $1 $2 & sleep 3; done; wait
echo "$(date +%T) gen W DEPLOY_DONE" >> logs/gen_deploy.log
