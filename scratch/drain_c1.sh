#!/usr/bin/env bash
# Drain a §67b c1 host (model.pt kept for D1b only, for c1/remech.py): QUEUE_DONE + no live python, checksummed fetch of each evaluated <out dir> + its log,
# local check (eval*.json present), then vdestroy.   bash scratch/drain_tjob.sh <host> <port> <queue tag> <instance id>
set -u
H=$1; P=$2; TAG=$3; ID=$4
cd "$(dirname "$0")/.."
S="ssh -o ConnectTimeout=20 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
st=$($S "cd /workspace/Latent || exit 3; L=logs/queue_${TAG}.out; [ -f \$L ] || exit 4
echo QD=\$(tail -n 1 \$L | grep -c QUEUE_DONE); echo PROCS=\$(ps -eo args | grep -a -c -E '^[^ ]*python[^ ]* -u -m c1.train')
for j in \$(grep -h ' evaluated ' \$L | awk '{print \$3}'); do echo RUN \$j; done; echo END_OK" 2>/dev/null)
echo "$st" | grep -q "^END_OK$" || { echo "FETCH_INCOMPLETE $ID (status)"; exit 1; }
[ "$(echo "$st" | sed -n 's/^QD=//p')" = 1 ] && [ "$(echo "$st" | sed -n 's/^PROCS=//p')" = 0 ] || { echo "NOT_FINISHED $ID"; exit 1; }
runs=$(echo "$st" | sed -n 's/^RUN //p'); [ -n "$runs" ] || { echo "FETCH_INCOMPLETE $ID (no runs)"; exit 1; }
files=""; for j in $runs; do files="$files $j logs/c1_$(basename $j).log"; done
ok=0
for try in 1 2 3; do
  rs=$($S "cd /workspace/Latent && tar --exclude='p67b_reset_*/model.pt' --exclude='p67b_fp*/model.pt' --exclude='p72_*/model.pt' --exclude='p72b_*/model.pt' --exclude='p73_*/model.pt' -czf /tmp/tj.tgz $files logs/queue_${TAG}.out && sha256sum /tmp/tj.tgz" 2>/dev/null | grep -o '^[0-9a-f]\{64\}')
  scp -q -i ~/.ssh/runpod_ed25519 -P $P root@$H:/tmp/tj.tgz /tmp/c1_${H}_${P}.tgz 2>/dev/null
  [ -n "$rs" ] && [ "$rs" = "$(sha256sum /tmp/c1_${H}_${P}.tgz | cut -c1-64)" ] && { ok=1; break; }
  sleep 20
done
[ $ok -eq 1 ] || { echo "FETCH_INCOMPLETE $ID (fetch failed 3x)"; exit 1; }
mkdir -p logs/vast_${H}_${P} && tar --no-same-owner -xzf /tmp/c1_${H}_${P}.tgz -C . --transform "s|^logs/|logs/vast_${H}_${P}/|"
for j in $runs; do ls $j/eval*.json >/dev/null 2>&1 || { echo "FETCH_INCOMPLETE $ID missing $j/eval"; exit 1; }; done
bash scratch/vdestroy.sh "$ID" | grep -q "^DESTROYED_VERIFIED" || { echo "DESTROY_UNVERIFIED $ID"; exit 1; }
echo "$ID vast | $TAG DESTROYED_VERIFIED $(date '+%Y-%m-%d %H:%M') after verified fetch ($(echo $runs))" >> scratch/pods.txt
echo "DRAINED $ID"
