#!/usr/bin/env bash
# After a §98 host's queue is done: re-run (one at a time, eval_bs 100) the eval of every finished run whose eval OOM'd
# beside its siblings (log.jsonl at --steps, model.pt present, no eval.json), with the current boxes/mult98.py
# (--eval_only; log.jsonl / args.json untouched; the OOM log kept as *.eval_oom.void), then drain with
# scratch/drain_gen_partial.sh (fetch the good runs, record any still-failed, destroy).
#   bash scratch/finish_evals.sh <host> <port> <tag> <instance>
set -u
H=$1; P=$2; TAG=$3; ID=$4
cd "$(dirname "$0")/.."
S="ssh -o ConnectTimeout=20 -o ServerAliveInterval=30 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
until $S "tail -n 1 /workspace/Latent/logs/queue_$TAG.out | grep -q QUEUE_DONE" 2>/dev/null; do sleep 300; done
echo "$(date +%T) $TAG QUEUE_DONE; re-running missing evals"
scp -q -i ~/.ssh/runpod_ed25519 -P $P boxes/mult98.py root@$H:/workspace/Latent/boxes/mult98.py
$S "cd /workspace/Latent; for j in \$(grep -h ' evaluated ' logs/queue_$TAG.out | awk '{print \$3}'); do
  [ -s \$j/eval.json ] && continue; [ -s \$j/model.pt ] || { echo \"no model: \$j\"; continue; }
  n=\$(basename \$j); read A T Sd St <<<\$(python -c \"import json;a=json.load(open('\$j/args.json'));print(a['arm'],a['task'],a['seed'],a['steps'])\")
  last=\$(tail -n 1 \$j/log.jsonl | python -c 'import json,sys;print(json.loads(sys.stdin.read())[\"step\"])')
  [ \"\$last\" = \"\$St\" ] || { echo \"unfinished: \$j (\$last/\$St)\"; continue; }
  mv logs/gen_\$n.log logs/gen_\$n.eval_oom.void
  python -u -m boxes.mult98 train --arm \$A --task \$T --seed \$Sd --steps \$St --eval_bs 100 --eval_only --device cuda:0 --out \$j > logs/gen_\$n.log 2>&1
  echo \"\$(date +%T) eval_only \$j rc \$? \$([ -s \$j/eval.json ] && echo ok || echo still-missing)\"
done" 2>/dev/null
bash scratch/drain_gen_partial.sh "$H" "$P" "$TAG" "$ID"
