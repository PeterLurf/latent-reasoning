#!/usr/bin/env bash
# §224 copy of scratch/drain_gen_partial_s223.sh (G-D'd; FH_v1 §6 Stage 1). Changes ONLY:
#   (1) the done marker is scratch/s224_done_marker.sh (a trained cell = model.pt [+ eval_vb.json / eval_khorizon.json]; <run>_re = its two eval files;
#       G-TIME timing.json; K7 mem_check.json; G-FH-EQ <run>_eqA / _eqB = the step-1200 log record);
#   (2) each queue dir is listed once (a cell dir is the out dir of its train and eval lines: sort -u);
#   (3) the tarball excludes ckpt_last.pt / ckpt_prev.pt, and model.pt of a *_re dir (the shipped copy of a local model);
#   (4) a partial run (BADM) keeps ckpt_last.pt + ckpt_fingerprint.json + log.jsonl + args.json + draws_log.json as <dir>.void_partial (vbank --resume);
#       s224_resume_ckpt is expanded ON THE HOST (\$(...); the §219 / §222 / §223 copies expand it locally, where the dir does not exist, so their
#       partial tarballs never carried ckpt_last.pt).
#   Everything else is the §223 original: status / NOT_FINISHED, sha-verified 3-try fetch, destroy only after the fetch.  bash scratch/drain_gen_partial_s224.sh <host> <port> <tag> <id>
set -u
H=$1; P=$2; TAG=$3; ID=$4
cd "$(dirname "$0")/.."
S="ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
. scratch/s224_done_marker.sh || { echo "FETCH_INCOMPLETE $ID (scratch/s224_done_marker.sh)"; exit 1; }
REM="$(cat scratch/s224_done_marker.sh)"$'\n'"cd /workspace/Latent || exit 3; L=logs/queue_${TAG}.out; [ -f \$L ] || exit 4
echo QD=\$(tail -n 1 \$L | grep -c QUEUE_DONE); echo PROCS=\$(ps -eo args | grep -a -c -E '^[^ ]*python[^ ]* -u -m ')
for j in \$(grep -h ' evaluated ' \$L | awk '{print \$3}' | sort -u); do echo \$(s224_mark \$j) \$j; done; echo END_OK"
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
  xm=""; case $j in *_re|*_c1) xm="--exclude=$j/model.pt";; esac
  fetch $n --exclude=$j/ckpt_last.pt --exclude=$j/ckpt_last.pt.tmp --exclude=$j/ckpt_prev.pt $xm $j logs/gen_$(basename $j).log || { echo "FETCH_INCOMPLETE $ID ($j)"; exit 1; }
done
fetch logs logs || { echo "FETCH_INCOMPLETE $ID (logs)"; exit 1; }
for j in $runs; do r_=$(s224_check_fetched $j); [ "$r_" = OK ] || { echo "FETCH_INCOMPLETE $ID $j $r_"; exit 1; }; done
vp=""; for j in $bad; do
  n=$(echo $j | tr '/' '_')_vp; ok=0
  for try in 1 2 3; do
    rs=$($S "$(declare -f s224_resume_ckpt); cd /workspace/Latent && tar -czf /tmp/f_$n.tgz --ignore-failed-read $j/log.jsonl $j/ckpt_fingerprint.json $j/args.json $j/draws_log.json \$(s224_resume_ckpt $j) 2>/dev/null; sha256sum /tmp/f_$n.tgz" 2>/dev/null | grep -o '^[0-9a-f]\{64\}')
    scp -q -o ServerAliveInterval=15 -i ~/.ssh/runpod_ed25519 -P $P root@$H:/tmp/f_$n.tgz /d/drain_tmp/f_${H}_$n.tgz 2>/dev/null
    [ -n "$rs" ] && [ "$rs" = "$(sha256sum /d/drain_tmp/f_${H}_$n.tgz | cut -c1-64)" ] && { ok=1; break; }; sleep 10
  done
  if [ $ok = 1 ] && tar --no-same-owner -xzf /d/drain_tmp/f_${H}_$n.tgz -C . --transform "s|^$j/|$j.void_partial/|" && [ -d $j.void_partial ]; then vp="$vp $j.void_partial"; else echo "PARTIAL_UNFETCHED $j"; fi
  rm -f /d/drain_tmp/f_${H}_$n.tgz
done
bash scratch/vdestroy.sh "$ID" | grep -q "^DESTROYED_VERIFIED" || { echo "DESTROY_UNVERIFIED $ID"; exit 1; }
echo "$ID vast | $TAG DESTROYED_VERIFIED $(date '+%Y-%m-%d %H:%M') after verified per-run fetch ($(echo $runs))${bad:+; FAILED (no §224 done marker): $(echo $bad)}${vp:+; partial ckpt kept:$vp}" >> scratch/pods.txt
echo "DRAINED $ID${bad:+ PARTIAL failed: $(echo $bad)}"
