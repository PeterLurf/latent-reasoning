#!/usr/bin/env bash
# §95 fleet: bash scratch/gen_deploy_95.sh "<offer> <n>" ... -> hosts gen-v<n> on scratch/gen_q_v<n>.txt (n 1-3 = f 0.05 / 0.25 / 0.5), auto-drain gen3.
cd "$(dirname "$0")/.."
one() { o=$1; n=$2
  out=$(EXTRA_FILES="prosqa/ckpt/gpt2_pretrained.bin" bash scratch/vdeploy_toy.sh $o gen-v$n scratch/gen_q_v$n.txt gen_v$n scratch/gen_vast_job.sh 2>&1)
  echo "$(date +%T) gen host v$n offer $o: $(echo "$out" | grep -E 'INSTANCE|NVIDIA|STARTED|BAD HOST|FAILED' | tr '\n' ' ')" >> logs/gen_deploy.log
  if echo "$out" | grep -q STARTED; then read -r _ id hh pp <<<"$(echo "$out" | grep '^INSTANCE')"
    nohup bash scratch/auto_drain_gen3.sh "$hh $pp gen_v$n $id gen" > logs/auto_drain_gen_v$n.out 2>&1 &
  else id=$(echo "$out" | grep '^INSTANCE' | awk '{print $2}'); [ -n "$id" ] && bash scratch/vdestroy.sh $id >> logs/gen_deploy.log 2>&1; fi; }
for x in "$@"; do set -- $x; one $1 $2 & sleep 3; done; wait
echo "$(date +%T) gen V DEPLOY_DONE" >> logs/gen_deploy.log
