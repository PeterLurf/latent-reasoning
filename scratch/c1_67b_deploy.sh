#!/usr/bin/env bash
# Deploy the §67b panel: one host per seed. bash scratch/c1_67b_deploy.sh "<offer> <seed>" ...
cd "$(dirname "$0")/.."
one() { o=$1; s=$2
  out=$(bash scratch/vdeploy_toy.sh $o c1-67b-$s scratch/c1_q_67b_$s.txt c1_67b_$s scratch/c1_vast_job.sh 2>&1)
  echo "$(date +%T) c1 67b host $s offer $o: $(echo "$out" | grep -E 'INSTANCE|NVIDIA|STARTED|BAD HOST|FAILED' | tr '\n' ' ')" >> logs/c1_deploy.log
  if echo "$out" | grep -q STARTED; then read -r _ id hh pp <<<"$(echo "$out" | grep '^INSTANCE')"
    nohup bash scratch/auto_drain_c1.sh "$hh $pp c1_67b_$s $id c1" > logs/auto_drain_c1_67b_$s.out 2>&1 &
  else id=$(echo "$out" | grep '^INSTANCE' | awk '{print $2}'); [ -n "$id" ] && bash scratch/vdestroy.sh $id >> logs/c1_deploy.log 2>&1; fi; }
for x in "$@"; do set -- $x; one $1 $2 & sleep 3; done; wait
echo "$(date +%T) c1 67b DEPLOY_DONE" >> logs/c1_deploy.log
