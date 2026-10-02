#!/usr/bin/env bash
# Deploy §71 hosts: bash scratch/s71_deploy.sh <worker script> "<offer> <tag> <queue>" ...
cd "$(dirname "$0")/.."
Wk=$1; shift
one() { o=$1; t=$2; q=$3
  out=$(bash scratch/vdeploy_toy.sh $o s71-$t $q s71_$t $Wk 2>&1)
  echo "$(date +%T) s71 host $t offer $o: $(echo "$out" | grep -E 'INSTANCE|NVIDIA|STARTED|BAD HOST|FAILED' | tr '\n' ' ')" >> logs/s71_deploy.log
  if echo "$out" | grep -q STARTED; then read -r _ id hh pp <<<"$(echo "$out" | grep '^INSTANCE')"
    nohup bash scratch/auto_drain_s71.sh "$hh $pp s71_$t $id s71" > logs/auto_drain_s71_$t.out 2>&1 &
  else id=$(echo "$out" | grep '^INSTANCE' | awk '{print $2}'); [ -n "$id" ] && bash scratch/vdestroy.sh $id >> logs/s71_deploy.log 2>&1; fi; }
for x in "$@"; do set -- $x; one $1 $2 $3 & sleep 3; done; wait
echo "$(date +%T) s71 DEPLOY_DONE" >> logs/s71_deploy.log
