#!/usr/bin/env bash
# drain_gen.sh variant (2026-10-03, Audit: a failed run must not leave a 3090 idle): runs WITH eval.json on the host are
# fetched (checksummed, 3 tries) with logs/; runs without one are recorded FAILED (their logs come with logs/, their dirs
# are not fetched: local reruns may reuse the names; their args.json + log.jsonl ARE kept as <dir>.void_partial); then vdestroy + pods.txt.  bash scratch/drain_gen_partial.sh <host> <port> <tag> <id>
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
# Model-skip rule (Theory + Audit 2026-10-06): scratch/drain_posthoc.py decides KEEP / SKIP per run ON THE HOST (and runs the registered
# on-host post-hoc before a SKIP); a SKIP drops only <run>/model.pt from the tarball and is logged to logs/model_skips.log. Any
# failure to decide -> KEEP.
SC="scp -q -o ConnectTimeout=20 -i $HOME/.ssh/runpod_ed25519 -P $P"; skipset=" "
if $SC scratch/drain_posthoc.py scratch/model_keep_patterns.txt root@$H:/workspace/Latent/scratch/ 2>/dev/null; then
  for j in $runs; do
    dec=$(timeout 900 $S "cd /workspace/Latent && python scratch/drain_posthoc.py $j cuda:0 2>/dev/null | tail -n 1; stat -c %s $j/model.pt 2>/dev/null" 2>/dev/null)
    if echo "$dec" | head -n 1 | grep -q "^SKIP "; then
      skipset="$skipset$j "; echo "$(date '+%F %T') $ID $TAG $j model.pt $(echo "$dec" | sed -n 2p) bytes NOT FETCHED: $(echo "$dec" | head -n 1)" >> logs/model_skips.log
    fi
  done
fi
for j in $runs $keep logs; do n=$(echo $j | tr '/' '_'); ok=0
  for try in 1 2 3; do
    if [ $j = logs ]; then src="logs"; else src="$j logs/gen_$(basename $j).log"; fi
    ex=""; case "$skipset" in *" $j "*) ex="--exclude=$j/model.pt";; esac
    rs=$($S "cd /workspace/Latent && tar -czf /tmp/f_$n.tgz $ex $src 2>/dev/null; sha256sum /tmp/f_$n.tgz" 2>/dev/null | grep -o '^[0-9a-f]\{64\}')
    scp -q -o ServerAliveInterval=15 -i ~/.ssh/runpod_ed25519 -P $P root@$H:/tmp/f_$n.tgz /d/drain_tmp/f_${H}_$n.tgz 2>/dev/null
    [ -n "$rs" ] && [ "$rs" = "$(sha256sum /d/drain_tmp/f_${H}_$n.tgz | cut -c1-64)" ] && { ok=1; break; }; sleep 20
  done
  [ $ok = 1 ] || { echo "FETCH_INCOMPLETE $ID ($j)"; exit 1; }
  tar --no-same-owner -xzf /d/drain_tmp/f_${H}_$n.tgz -C . --transform "s|^logs/|logs/vast_${H}_${P}/|" && rm -f /d/drain_tmp/f_${H}_$n.tgz   # extracted and sha-verified: drop the local copy (C: filled up 2026-10-03)
done
for j in $runs; do [ -s $j/eval.json ] || { echo "FETCH_INCOMPLETE $ID missing $j/eval.json"; exit 1; }; done
# Theory 2026-10-05 (aborted-run rule): failed runs not fetched above keep their partial args.json + log.jsonl as <dir>.void_partial
# (sha-verified; non-fatal - a miss is reported, the drain continues). Their run dir names stay free for local reruns.
vp=""; for j in $(echo "$st" | sed -n 's/^BADM\{0,1\} //p'); do case " $keep " in *" $j "*) continue;; esac
  n=$(echo $j | tr '/' '_')_vp; ok=0
  for try in 1 2 3; do
    rs=$($S "cd /workspace/Latent && tar -czf /tmp/f_$n.tgz --ignore-failed-read $j/args.json $j/log.jsonl 2>/dev/null; sha256sum /tmp/f_$n.tgz" 2>/dev/null | grep -o '^[0-9a-f]\{64\}')
    scp -q -o ServerAliveInterval=15 -i ~/.ssh/runpod_ed25519 -P $P root@$H:/tmp/f_$n.tgz /d/drain_tmp/f_${H}_$n.tgz 2>/dev/null
    [ -n "$rs" ] && [ "$rs" = "$(sha256sum /d/drain_tmp/f_${H}_$n.tgz | cut -c1-64)" ] && { ok=1; break; }; sleep 10
  done
  if [ $ok = 1 ] && tar --no-same-owner -xzf /d/drain_tmp/f_${H}_$n.tgz -C . --transform "s|^$j/|$j.void_partial/|" && [ -d $j.void_partial ]; then vp="$vp $j.void_partial"; else echo "PARTIAL_LOGS_UNFETCHED $j"; fi
  rm -f /d/drain_tmp/f_${H}_$n.tgz
done
bash scratch/vdestroy.sh "$ID" | grep -q "^DESTROYED_VERIFIED" || { echo "DESTROY_UNVERIFIED $ID"; exit 1; }
nsk=$(echo $skipset | wc -w)
echo "$ID vast | $TAG DESTROYED_VERIFIED $(date '+%Y-%m-%d %H:%M') after verified per-run fetch ($(echo $runs))${bad:+; FAILED (no eval.json, logs fetched): $(echo $bad)}${vp:+; partial logs kept:$vp}$( [ "$nsk" -gt 0 ] && echo "; model.pt not fetched (rule, logs/model_skips.log):$skipset")" >> scratch/pods.txt
echo "DRAINED $ID${bad:+ PARTIAL failed: $(echo $bad)}"
