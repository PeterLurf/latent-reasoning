#!/usr/bin/env bash
# §222 copy of scratch/drain_gen_partial_s219.sh (Note 222-1). Changes ONLY:
#   (1) the done marker is scratch/s222_done_marker.sh's s222_mark (a finished group file cert_<model>.g<i>.json / eval_lvb_s222.g<i>.json, A1's
#       eval_lvb_s222_pre.json, G-TIME timing.json, or the ACC cross-card eval_lvb.json), sent to the host for the status and sourced locally for the
#       post-fetch check (s222_check_fetched);
#   (2) no seal rename and no model.pt pass: §222 run dirs hold no model (A1's model.pt is shipped into its own dir, never an "evaluated" out dir);
#   (3) a FINISHED run's ckpt_last.pt is not fetched (retention: the group file holds the result); a partial run (BADM) keeps its full-state
#       ckpt_last.pt + ckpt_fingerprint.json + log.jsonl as <dir>.void_partial (sha-verified), so a crashed group resumes exactly (--resume).
#   Everything else is the original: status / NOT_FINISHED, sha-verified 3-try fetch, destroy only after the fetch.  bash scratch/drain_gen_partial_s222.sh <host> <port> <tag> <id>
set -u
H=$1; P=$2; TAG=$3; ID=$4
cd "$(dirname "$0")/.."
S="ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
. scratch/s222_done_marker.sh || { echo "FETCH_INCOMPLETE $ID (scratch/s222_done_marker.sh)"; exit 1; }
REM="$(cat scratch/s222_done_marker.sh)"$'\n'"cd /workspace/Latent || exit 3; L=logs/queue_${TAG}.out; [ -f \$L ] || exit 4
echo QD=\$(tail -n 1 \$L | grep -c QUEUE_DONE); echo PROCS=\$(ps -eo args | grep -a -c -E '^[^ ]*python[^ ]* -u -m ')
for j in \$(grep -h ' evaluated ' \$L | awk '{print \$3}'); do echo \$(s222_mark \$j) \$j; done; echo END_OK"
st=$($S "$REM" 2>/dev/null)
echo "$st" | grep -q "^END_OK$" || { echo "FETCH_INCOMPLETE $ID (status)"; exit 1; }
[ "$(echo "$st" | sed -n 's/^QD=//p')" = 1 ] && [ "$(echo "$st" | sed -n 's/^PROCS=//p')" = 0 ] || { echo "NOT_FINISHED $ID"; exit 1; }
runs=$(echo "$st" | sed -n 's/^RUN //p'); bad=$(echo "$st" | sed -n 's/^BADM\{0,1\} //p')
mkdir -p logs/vast_${H}_${P}
fetch() {   # fetch <name> <remote tar arguments>: sha-verified, 3 tries, extracted with logs/ -> logs/vast_<host>_<port>/
  local n=$1; shift; local ok=0
  for try in 1 2 3; do
    rs=$($S "cd /workspace/Latent && rm -f /tmp/f_$n.tgz && tar -czf /tmp/f_$n.tgz $* 2>/dev/null; gzip -t /tmp/f_$n.tgz 2>/dev/null && sha256sum /tmp/f_$n.tgz || { df -h / | tail -1 | sed 's/^/ARCHIVE_TRUNCATED /'; }" 2>/dev/null | grep -o '^[0-9a-f]\{64\}')
    scp -q -o ServerAliveInterval=15 -i ~/.ssh/runpod_ed25519 -P $P root@$H:/tmp/f_$n.tgz /d/drain_tmp/f_${H}_$n.tgz 2>/dev/null
    [ -n "$rs" ] && [ "$rs" = "$(sha256sum /d/drain_tmp/f_${H}_$n.tgz | cut -c1-64)" ] && { ok=1; break; }; sleep 20
  done
  [ $ok = 1 ] || return 1
  tar --no-same-owner -xzf /d/drain_tmp/f_${H}_$n.tgz -C . --transform "s|^logs/|logs/vast_${H}_${P}/|" && rm -f /d/drain_tmp/f_${H}_$n.tgz
}
mkdir -p /d/drain_tmp
for j in $runs; do n=$(echo $j | tr '/' '_')
  fetch $n --exclude=$j/ckpt_last.pt --exclude=$j/ckpt_last.pt.tmp $j logs/gen_$(basename $j).log || { echo "FETCH_INCOMPLETE $ID ($j)"; exit 1; }
done
fetch logs logs || { echo "FETCH_INCOMPLETE $ID (logs)"; exit 1; }
for j in $runs; do r_=$(s222_check_fetched $j); [ "$r_" = OK ] || { echo "FETCH_INCOMPLETE $ID $j $r_"; exit 1; }; done
vp=""; for j in $bad; do
  n=$(echo $j | tr '/' '_')_vp; ok=0
  for try in 1 2 3; do
    rs=$($S "$(declare -f s222_resume_ckpt); cd /workspace/Latent && tar -czf /tmp/f_$n.tgz --ignore-failed-read $j/log.jsonl $j/ckpt_fingerprint.json \$(s222_resume_ckpt $j) 2>/dev/null; sha256sum /tmp/f_$n.tgz" 2>/dev/null | grep -o '^[0-9a-f]\{64\}')
    scp -q -o ServerAliveInterval=15 -i ~/.ssh/runpod_ed25519 -P $P root@$H:/tmp/f_$n.tgz /d/drain_tmp/f_${H}_$n.tgz 2>/dev/null
    [ -n "$rs" ] && [ "$rs" = "$(sha256sum /d/drain_tmp/f_${H}_$n.tgz | cut -c1-64)" ] && { ok=1; break; }; sleep 10
  done
  if [ $ok = 1 ] && tar --no-same-owner -xzf /d/drain_tmp/f_${H}_$n.tgz -C . --transform "s|^$j/|$j.void_partial/|" && [ -d $j.void_partial ]; then vp="$vp $j.void_partial"; else echo "PARTIAL_UNFETCHED $j"; fi
  rm -f /d/drain_tmp/f_${H}_$n.tgz
done
bash scratch/vdestroy.sh "$ID" | grep -q "^DESTROYED_VERIFIED" || { echo "DESTROY_UNVERIFIED $ID"; exit 1; }
echo "$ID vast | $TAG DESTROYED_VERIFIED $(date '+%Y-%m-%d %H:%M') after verified per-run fetch ($(echo $runs))${bad:+; FAILED (no §222 done marker): $(echo $bad)}${vp:+; partial ckpt kept:$vp}" >> scratch/pods.txt
echo "DRAINED $ID${bad:+ PARTIAL failed: $(echo $bad)}"
