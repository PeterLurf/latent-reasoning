#!/usr/bin/env bash
# §120 fleet: bash scratch/gen_deploy_120.sh "<offer> <n>" -> gen-b<n> on scratch/gen_q_b<n>.txt (fsq120), auto-drain gen5.
cd "$(dirname "$0")/.."
one() { o=$1; n=$2
  out=$(EXTRA_FILES="prosqa/ckpt/gpt2_pretrained.bin boxes/gen97.py boxes/mk108.py boxes/fsq120.py ${EXTRA_120:-}" bash scratch/vdeploy_toy.sh $o gen-b$n scratch/gen_q_b$n.txt gen_b$n scratch/gen_vast_job.sh 2>&1)
  echo "$(date +%T) gen host b$n offer $o: $(echo "$out" | grep -E 'INSTANCE|NVIDIA|STARTED|BAD HOST|FAILED' | tr '\n' ' ')" >> logs/gen_deploy.log
  if echo "$out" | grep -q STARTED; then read -r _ id hh pp <<<"$(echo "$out" | grep '^INSTANCE')"
    nohup bash scratch/auto_drain_gen6.sh "$hh $pp gen_b$n $id gen" > logs/auto_drain_gen_b$n.out 2>&1 &
  else id=$(echo "$out" | grep '^INSTANCE' | awk '{print $2}'); [ -n "$id" ] && bash scratch/vdestroy.sh $id >> logs/gen_deploy.log 2>&1; fi; }
for x in "$@"; do set -- $x; one $1 $2 & sleep 3; done; wait
echo "$(date +%T) gen B DEPLOY_DONE" >> logs/gen_deploy.log
