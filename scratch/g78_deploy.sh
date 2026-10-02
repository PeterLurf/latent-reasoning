#!/usr/bin/env bash
# THEORY §78 fleet: bash scratch/g78_deploy.sh "<offer> <name>" ... ; name a1..a3 (TC + WS, seed n) or b1..b3 (WS-pad).
cd "$(dirname "$0")/.."
one() { o=$1; n=$2
  out=$(bash scratch/vdeploy_toy.sh $o g78-$n scratch/g78_q_$n.txt g78_$n scratch/g78_vast_job.sh 2>&1)
  echo "$(date +%T) g78 host $n offer $o: $(echo "$out" | grep -E 'INSTANCE|NVIDIA|STARTED|BAD HOST|FAILED' | tr '\n' ' ')" >> logs/g78_deploy.log
  if echo "$out" | grep -q STARTED; then read -r _ id hh pp <<<"$(echo "$out" | grep '^INSTANCE')"
    nohup bash scratch/auto_drain_c1.sh "$hh $pp g78_$n $id g78" > logs/auto_drain_g78_$n.out 2>&1 &
  else id=$(echo "$out" | grep '^INSTANCE' | awk '{print $2}'); [ -n "$id" ] && bash scratch/vdestroy.sh $id >> logs/g78_deploy.log 2>&1; fi; }
for x in "$@"; do set -- $x; one $1 $2 & sleep 3; done; wait
echo "$(date +%T) g78 DEPLOY_DONE" >> logs/g78_deploy.log
