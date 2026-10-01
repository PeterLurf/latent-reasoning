#!/usr/bin/env bash
# Drain a finished §60b host: QUEUE_DONE + no live process, checksummed fetch of bm/runs/<r> (model.pt, log.jsonl,
# args.json, eval*.json) for every run in the queue log plus logs, local verification, then vdestroy.
#   bash scratch/drain_bm.sh <host> <port> <queue tag> <instance id>
set -u
H=$1; P=$2; TAG=$3; ID=$4
cd "$(dirname "$0")/.."
S="ssh -o ConnectTimeout=20 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
st=$($S "cd /workspace/Latent || exit 3; L=logs/queue_${TAG}.out; [ -f \$L ] || exit 4
echo QD=\$(tail -n 1 \$L | grep -c QUEUE_DONE); echo PROCS=\$(ps -eo args | grep -a -c -E '^[^ ]*python[^ ]* .*run_b')
for r in \$(grep -h ' evaluated ' \$L | awk '{print \$3}'); do echo RUN \$r; done; echo END_OK" 2>/dev/null)
echo "$st" | grep -q "^END_OK$" || { echo "FETCH_INCOMPLETE $ID (status)"; exit 1; }
[ "$(echo "$st" | sed -n 's/^QD=//p')" = 1 ] && [ "$(echo "$st" | sed -n 's/^PROCS=//p')" = 0 ] || { echo "NOT_FINISHED $ID"; exit 1; }
runs=$(echo "$st" | sed -n 's/^RUN //p'); [ -n "$runs" ] || { echo "FETCH_INCOMPLETE $ID (no runs)"; exit 1; }
files=""; for r in $runs; do files="$files bm/runs/$r logs/$r.out"; done
ok=0
for try in 1 2 3; do
  rs=$($S "cd /workspace/Latent && tar -czf /tmp/bm.tgz $files logs/queue_${TAG}.out && sha256sum /tmp/bm.tgz" 2>/dev/null | grep -o '^[0-9a-f]\{64\}')
  scp -q -i ~/.ssh/runpod_ed25519 -P $P root@$H:/tmp/bm.tgz /tmp/bm_${H}_${P}.tgz 2>/dev/null
  [ -n "$rs" ] && [ "$rs" = "$(sha256sum /tmp/bm_${H}_${P}.tgz | cut -c1-64)" ] && { ok=1; break; }
  sleep 20
done
[ $ok -eq 1 ] || { echo "FETCH_INCOMPLETE $ID (fetch failed 3x)"; exit 1; }
mkdir -p logs/vast_${H}_${P} && tar --no-same-owner -xzf /tmp/bm_${H}_${P}.tgz -C . --transform "s|^logs/|logs/vast_${H}_${P}/|"
for r in $runs; do for f in model.pt log.jsonl args.json; do [ -s bm/runs/$r/$f ] || { echo "FETCH_INCOMPLETE $ID missing $r/$f"; exit 1; }; done
  ls bm/runs/$r/eval*.json >/dev/null 2>&1 || { echo "FETCH_INCOMPLETE $ID missing $r/eval"; exit 1; }; done
bash scratch/vdestroy.sh "$ID" | grep -q "^DESTROYED_VERIFIED" || { echo "DESTROY_UNVERIFIED $ID"; exit 1; }
echo "$ID vast | $TAG DESTROYED_VERIFIED $(date '+%Y-%m-%d %H:%M') after verified fetch (bm runs: $(echo $runs))" >> scratch/pods.txt
echo "DRAINED $ID"
