#!/usr/bin/env bash
# §119 hosts: bash scratch/gen_deploy_119.sh "<offer> <n>" -> gen-d<n> on scratch/gen_q_d<n>.txt (base119 Coconut / SIM-CoT / CODI), auto-drain gen5.
cd "$(dirname "$0")/.."
for f_ in "$0" scratch/vdeploy_toy.sh scratch/gen_vast_job.sh scratch/auto_drain_gen7.sh scratch/drain_gen_partial.sh; do   # Audit 2026-10-06: syntax gate
  bash -n "$f_" 2>/dev/null || { echo "$(date +%T) DEPLOY_ABORT bash -n failed: $f_" >> logs/gen_deploy.log; echo "DEPLOY_ABORT bash -n $f_"; exit 3; }
done
one() { o=$1; n=$2
  out=$(EXTRA_FILES="${SHIP_GPT2:+prosqa/ckpt/gpt2_pretrained.bin} boxes/gen97.py boxes/base119.py ${RESUME_FILES:-}" bash scratch/vdeploy_toy.sh $o gen-d$n scratch/gen_q_d$n.txt gen_d$n scratch/gen_vast_job.sh 2>&1)
  echo "$(date +%T) gen host d$n offer $o: $(echo "$out" | grep -E 'INSTANCE|NVIDIA|STARTED|BAD HOST|FAILED' | tr '\n' ' ')" >> logs/gen_deploy.log
  echo "$out" | grep -qE 'INSTANCE|FAILED|BAD HOST|STARTED' || echo "$(date +%T) DEPLOY_SILENT_FAIL offer $o: $(echo "$out" | tail -n 3 | tr '\n' ' ')" >> logs/gen_deploy.log   # Audit: a silent vdeploy exit is logged, never blank
  if echo "$out" | grep -q STARTED; then read -r _ id hh pp <<<"$(echo "$out" | grep '^INSTANCE')"
    nohup bash scratch/auto_drain_gen7.sh "$hh $pp gen_d$n $id gen" > logs/auto_drain_gen_d$n.out 2>&1 &
  else id=$(echo "$out" | grep '^INSTANCE' | awk '{print $2}'); [ -n "$id" ] && bash scratch/vdestroy.sh $id >> logs/gen_deploy.log 2>&1; fi; }
for x in "$@"; do set -- $x; one $1 $2 & sleep 3; done; wait
echo "$(date +%T) gen D DEPLOY_DONE" >> logs/gen_deploy.log
