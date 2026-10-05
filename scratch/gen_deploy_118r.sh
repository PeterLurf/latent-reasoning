#!/usr/bin/env bash
# §118 hosts (+ RESUME_FILES env: extra run files to ship, e.g. a ckpt for a resume)
cd "$(dirname "$0")/.."
one() { o=$1; n=$2
  out=$(EXTRA_FILES="prosqa/ckpt/gpt2_pretrained.bin boxes/gen97.py boxes/mk108.py boxes/snap115.py ${RESUME_FILES}" bash scratch/vdeploy_toy.sh $o gen-a$n scratch/gen_q_a$n.txt gen_a$n scratch/gen_vast_job.sh 2>&1)
  echo "$(date +%T) gen host a$n offer $o: $(echo "$out" | grep -E 'INSTANCE|NVIDIA|STARTED|BAD HOST|FAILED' | tr '\n' ' ')" >> logs/gen_deploy.log
  if echo "$out" | grep -q STARTED; then read -r _ id hh pp <<<"$(echo "$out" | grep '^INSTANCE')"
    nohup bash scratch/auto_drain_gen5.sh "$hh $pp gen_a$n $id gen" > logs/auto_drain_gen_a$n.out 2>&1 &
  else id=$(echo "$out" | grep '^INSTANCE' | awk '{print $2}'); [ -n "$id" ] && bash scratch/vdestroy.sh $id >> logs/gen_deploy.log 2>&1; fi; }
for x in "$@"; do set -- $x; one $1 $2 & sleep 3; done; wait
echo "$(date +%T) gen A DEPLOY_DONE" >> logs/gen_deploy.log
