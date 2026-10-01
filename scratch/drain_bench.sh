#!/usr/bin/env bash
# Drain a bench host: QUEUE_DONE + no live python, checksummed fetch of bench/results/*.json + logs, then vdestroy.
#   bash scratch/drain_bench.sh <host> <port> <queue tag> <instance id>
set -u
H=$1; P=$2; TAG=$3; ID=$4
cd "$(dirname "$0")/.."
S="ssh -o ConnectTimeout=20 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
st=$($S "cd /workspace/Latent || exit 3; L=logs/queue_${TAG}.out; [ -f \$L ] || exit 4
echo QD=\$(tail -n 1 \$L | grep -c QUEUE_DONE); echo PROCS=\$(ps -eo args | grep -a -c -E '^[^ ]*python[^ ]* -u -m bench')
echo NJ=\$(ls bench/results/*.json 2>/dev/null | wc -l); echo NE=\$(grep -c ' evaluated ' \$L); echo END_OK" 2>/dev/null)
echo "$st" | grep -q "^END_OK$" || { echo "FETCH_INCOMPLETE $ID (status)"; exit 1; }
[ "$(echo "$st" | sed -n 's/^QD=//p')" = 1 ] && [ "$(echo "$st" | sed -n 's/^PROCS=//p')" = 0 ] || { echo "NOT_FINISHED $ID"; exit 1; }
ok=0
for try in 1 2 3; do
  rs=$($S "cd /workspace/Latent && tar -czf /tmp/bench.tgz bench/results logs/bench_*.log logs/queue_${TAG}.out && sha256sum /tmp/bench.tgz" 2>/dev/null | grep -o '^[0-9a-f]\{64\}')
  scp -q -i ~/.ssh/runpod_ed25519 -P $P root@$H:/tmp/bench.tgz /tmp/bench_${H}_${P}.tgz 2>/dev/null
  [ -n "$rs" ] && [ "$rs" = "$(sha256sum /tmp/bench_${H}_${P}.tgz | cut -c1-64)" ] && { ok=1; break; }
  sleep 20
done
[ $ok -eq 1 ] || { echo "FETCH_INCOMPLETE $ID (fetch failed 3x)"; exit 1; }
mkdir -p logs/vast_${H}_${P} && tar --no-same-owner -xzf /tmp/bench_${H}_${P}.tgz -C . --transform "s|^logs/|logs/vast_${H}_${P}/|"
bash scratch/vdestroy.sh "$ID" | grep -q "^DESTROYED_VERIFIED" || { echo "DESTROY_UNVERIFIED $ID"; exit 1; }
echo "$ID vast | $TAG DESTROYED_VERIFIED $(date '+%Y-%m-%d %H:%M') after verified fetch (bench: $(echo "$st" | sed -n 's/^NE=//p') jobs)" >> scratch/pods.txt
echo "DRAINED $ID"
