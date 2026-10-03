#!/usr/bin/env bash
# Drain a generic host: QUEUE_DONE + no live python; per-run checksummed fetch (each run dir with model.pt in its own tar,
# 3 tries) + the queue log; local eval.json check; then vdestroy + pods.txt.  bash scratch/drain_gen.sh <host> <port> <tag> <id>
set -u
H=$1; P=$2; TAG=$3; ID=$4
cd "$(dirname "$0")/.."
S="ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
st=$($S "cd /workspace/Latent || exit 3; L=logs/queue_${TAG}.out; [ -f \$L ] || exit 4
echo QD=\$(tail -n 1 \$L | grep -c QUEUE_DONE); echo PROCS=\$(ps -eo args | grep -a -c -E '^[^ ]*python[^ ]* -u -m ')
for j in \$(grep -h ' evaluated ' \$L | awk '{print \$3}'); do echo RUN \$j; done; echo END_OK" 2>/dev/null)
echo "$st" | grep -q "^END_OK$" || { echo "FETCH_INCOMPLETE $ID (status)"; exit 1; }
[ "$(echo "$st" | sed -n 's/^QD=//p')" = 1 ] && [ "$(echo "$st" | sed -n 's/^PROCS=//p')" = 0 ] || { echo "NOT_FINISHED $ID"; exit 1; }
runs=$(echo "$st" | sed -n 's/^RUN //p'); [ -n "$runs" ] || { echo "FETCH_INCOMPLETE $ID (no runs)"; exit 1; }
mkdir -p logs/vast_${H}_${P}
for j in $runs logs; do n=$(echo $j | tr '/' '_'); ok=0
  for try in 1 2 3; do
    if [ $j = logs ]; then src="logs"; else src="$j logs/gen_$(basename $j).log"; fi
    rs=$($S "cd /workspace/Latent && tar -czf /tmp/f_$n.tgz $src 2>/dev/null; sha256sum /tmp/f_$n.tgz" 2>/dev/null | grep -o '^[0-9a-f]\{64\}')
    scp -q -o ServerAliveInterval=15 -i ~/.ssh/runpod_ed25519 -P $P root@$H:/tmp/f_$n.tgz /tmp/f_${H}_$n.tgz 2>/dev/null
    [ -n "$rs" ] && [ "$rs" = "$(sha256sum /tmp/f_${H}_$n.tgz | cut -c1-64)" ] && { ok=1; break; }; sleep 20
  done
  [ $ok = 1 ] || { echo "FETCH_INCOMPLETE $ID ($j)"; exit 1; }
  tar --no-same-owner -xzf /tmp/f_${H}_$n.tgz -C . --transform "s|^logs/|logs/vast_${H}_${P}/|"
done
for j in $runs; do [ -s $j/eval.json ] || { echo "FETCH_INCOMPLETE $ID missing $j/eval.json"; exit 1; }; done
bash scratch/vdestroy.sh "$ID" | grep -q "^DESTROYED_VERIFIED" || { echo "DESTROY_UNVERIFIED $ID"; exit 1; }
echo "$ID vast | $TAG DESTROYED_VERIFIED $(date '+%Y-%m-%d %H:%M') after verified per-run fetch ($(echo $runs))" >> scratch/pods.txt
echo "DRAINED $ID"
