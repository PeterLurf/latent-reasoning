#!/usr/bin/env bash
# drain_gen.sh variant (2026-10-03, Audit: a failed run must not leave a 3090 idle): runs WITH eval.json on the host are
# fetched (checksummed, 3 tries) with logs/; runs without one are recorded FAILED (their logs come with logs/, their dirs
# are not fetched: local reruns may reuse the names); then vdestroy + pods.txt.  bash scratch/drain_gen_partial.sh <host> <port> <tag> <id>
set -u
H=$1; P=$2; TAG=$3; ID=$4
cd "$(dirname "$0")/.."
S="ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
st=$($S "cd /workspace/Latent || exit 3; L=logs/queue_${TAG}.out; [ -f \$L ] || exit 4
echo QD=\$(tail -n 1 \$L | grep -c QUEUE_DONE); echo PROCS=\$(ps -eo args | grep -a -c -E '^[^ ]*python[^ ]* -u -m ')
for j in \$(grep -h ' evaluated ' \$L | awk '{print \$3}'); do if [ -s \$j/eval.json ]; then echo RUN \$j; elif [ -s \$j/model.pt ]; then echo BADM \$j; else echo BAD \$j; fi; done; echo END_OK" 2>/dev/null)
echo "$st" | grep -q "^END_OK$" || { echo "FETCH_INCOMPLETE $ID (status)"; exit 1; }
[ "$(echo "$st" | sed -n 's/^QD=//p')" = 1 ] && [ "$(echo "$st" | sed -n 's/^PROCS=//p')" = 0 ] || { echo "NOT_FINISHED $ID"; exit 1; }
runs=$(echo "$st" | sed -n 's/^RUN //p'); bad=$(echo "$st" | sed -n 's/^BADM\{0,1\} //p')
# failed runs WITH a model.pt are fetched too (an eval OOM must not lose a trained model), unless a local dir of that name exists
keep=""; for j in $(echo "$st" | sed -n 's/^BADM //p'); do [ -e "$j" ] || keep="$keep $j"; done
mkdir -p logs/vast_${H}_${P}
for j in $runs $keep logs; do n=$(echo $j | tr '/' '_'); ok=0
  for try in 1 2 3; do
    if [ $j = logs ]; then src="logs"; else src="$j logs/gen_$(basename $j).log"; fi
    rs=$($S "cd /workspace/Latent && tar -czf /tmp/f_$n.tgz $src 2>/dev/null; sha256sum /tmp/f_$n.tgz" 2>/dev/null | grep -o '^[0-9a-f]\{64\}')
    scp -q -o ServerAliveInterval=15 -i ~/.ssh/runpod_ed25519 -P $P root@$H:/tmp/f_$n.tgz /d/drain_tmp/f_${H}_$n.tgz 2>/dev/null
    [ -n "$rs" ] && [ "$rs" = "$(sha256sum /d/drain_tmp/f_${H}_$n.tgz | cut -c1-64)" ] && { ok=1; break; }; sleep 20
  done
  [ $ok = 1 ] || { echo "FETCH_INCOMPLETE $ID ($j)"; exit 1; }
  tar --no-same-owner -xzf /d/drain_tmp/f_${H}_$n.tgz -C . --transform "s|^logs/|logs/vast_${H}_${P}/|" && rm -f /d/drain_tmp/f_${H}_$n.tgz   # extracted and sha-verified: drop the local copy (C: filled up 2026-10-03)
done
for j in $runs; do [ -s $j/eval.json ] || { echo "FETCH_INCOMPLETE $ID missing $j/eval.json"; exit 1; }; done
bash scratch/vdestroy.sh "$ID" | grep -q "^DESTROYED_VERIFIED" || { echo "DESTROY_UNVERIFIED $ID"; exit 1; }
echo "$ID vast | $TAG DESTROYED_VERIFIED $(date '+%Y-%m-%d %H:%M') after verified per-run fetch ($(echo $runs))${bad:+; FAILED (no eval.json, logs fetched): $(echo $bad)}" >> scratch/pods.txt
echo "DRAINED $ID${bad:+ PARTIAL failed: $(echo $bad)}"
