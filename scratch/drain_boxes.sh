#!/usr/bin/env bash
# Drain a §66 boxes host: QUEUE_DONE + no live python, checksummed fetch of boxes/runs/<run> (model.pt kept only for
# seed-1 runs, *_s1, per Theory; ~500 MB each otherwise dropped) + logs + bench/results, local check (eval_train.json per run), then vdestroy.
#   bash scratch/drain_boxes.sh <host> <port> <queue tag> <instance id>
set -u
H=$1; P=$2; TAG=$3; ID=$4
cd "$(dirname "$0")/.."
S="ssh -o ConnectTimeout=20 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
st=$($S "cd /workspace/Latent || exit 3; L=logs/queue_${TAG}.out; [ -f \$L ] || exit 4
echo QD=\$(tail -n 1 \$L | grep -c QUEUE_DONE); echo PROCS=\$(ps -eo args | grep -a -c -E '^[^ ]*python[^ ]* -u (boxes/train|-m bench)')
for j in \$(grep -h ' evaluated ' \$L | awk '{print \$3}'); do echo RUN \$j; done; echo END_OK" 2>/dev/null)
echo "$st" | grep -q "^END_OK$" || { echo "FETCH_INCOMPLETE $ID (status)"; exit 1; }
[ "$(echo "$st" | sed -n 's/^QD=//p')" = 1 ] && [ "$(echo "$st" | sed -n 's/^PROCS=//p')" = 0 ] || { echo "NOT_FINISHED $ID"; exit 1; }
runs=$(echo "$st" | sed -n 's/^RUN //p'); [ -n "$runs" ] || { echo "FETCH_INCOMPLETE $ID (no runs)"; exit 1; }
files=""; ex=""; for j in $runs; do files="$files boxes/runs/$j logs/$j.out"; case $j in *_s1) ;; *) ex="$ex --exclude=boxes/runs/$j/model.pt";; esac; done
ok=0
for try in 1 2 3; do
  rs=$($S "cd /workspace/Latent && tar -czf /tmp/bx.tgz $ex $files bench/results logs/bench_*.log logs/queue_${TAG}.out 2>/dev/null; sha256sum /tmp/bx.tgz" 2>/dev/null | grep -o '^[0-9a-f]\{64\}')
  scp -q -i ~/.ssh/runpod_ed25519 -P $P root@$H:/tmp/bx.tgz /tmp/bx_${H}_${P}.tgz 2>/dev/null
  [ -n "$rs" ] && [ "$rs" = "$(sha256sum /tmp/bx_${H}_${P}.tgz | cut -c1-64)" ] && { ok=1; break; }
  sleep 20
done
[ $ok -eq 1 ] || { echo "FETCH_INCOMPLETE $ID (fetch failed 3x)"; exit 1; }
mkdir -p logs/vast_${H}_${P} && tar --no-same-owner -xzf /tmp/bx_${H}_${P}.tgz -C . --transform "s|^logs/|logs/vast_${H}_${P}/|"
for j in $runs; do [ -s boxes/runs/$j/eval_train.json ] || { echo "FETCH_INCOMPLETE $ID missing $j"; exit 1; }; done
bash scratch/vdestroy.sh "$ID" | grep -q "^DESTROYED_VERIFIED" || { echo "DESTROY_UNVERIFIED $ID"; exit 1; }
echo "$ID vast | $TAG DESTROYED_VERIFIED $(date '+%Y-%m-%d %H:%M') after verified fetch (boxes: $(echo $runs); model.pt kept for *_s1 only)" >> scratch/pods.txt
echo "DRAINED $ID"
