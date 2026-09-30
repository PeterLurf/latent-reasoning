#!/usr/bin/env bash
# Pilot M (THEORY §38): deploy one host per queue file, logging each host in the same step it is created.
#   bash theory/deploy_M.sh <idx> <offer id>     (idx 01..20 -> theory/queues/pilotM_<idx>.txt, label theory-M<idx>)
cd "$(dirname "$0")/.."
i=$1; O=$2; Q=theory/queues/pilotM_$i.txt; LBL=theory-M$i
init=$(grep -o 'theory/runs/pl_t8_s[0-9]*/model.pt' $Q | head -1)
out=$(EXTRA="theory/deep_chain.py $init" RUNNER=queue_distill.sh bash theory/vdeploy_toy.sh $O $LBL $Q 2>&1)
read -r _ ID H P <<<"$(echo "$out" | grep '^INSTANCE')"
card=$(echo "$out" | grep -i ', [0-9]*\.' | head -1 | tr -d '\r')
if [ -n "$ID" ]; then
  echo "$ID $LBL offer$O ${H}:${P} created $(date '+%Y-%m-%d %H:%M') (Pilot M, one model x 3 arms; queue $Q; $card)" >> theory/vast_log.txt
  [ -n "$H" ] && echo "$H $P $LBL $ID" >> theory/hosts_theory.txt
fi
st=$(echo "$out" | grep -E 'STARTED|BAD HOST|CREATE FAILED' | head -1)
echo "$LBL $ID $H:$P ${st:-NOT STARTED} | $card"
