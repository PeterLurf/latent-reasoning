#!/usr/bin/env bash
# §219 copy of scratch/drain_gen_partial.sh (Note 219-3 item 1; Audit G-T13 item 6). Changes ONLY:
#   (1) the done marker is scratch/s219_done_marker.sh's s219_mark (train_done.json | mem_check_self.json | timing.json), never eval.json; the
#       same file is sent to the host for the status and sourced locally for the post-fetch check (s219_check_fetched: the §219 file list present,
#       no eval*.json at all). No eval*.json is ever required, created or fetched;
#   (2) a trainer-written eval*.json on the host (a SEAL-BREACH) is renamed *.void_unsealed on the host BEFORE the fetch (kept and disclosed, TS_v1
#       §5.2) and reported (SEAL_BREACH lines, pods.txt);
#   (3) no model-skip post-hoc: every §219 model.pt is kept (T1 is T2's labeller, TS_v1 §8 retention), so a RUN cell without a fetched model.pt fails
#       the verification;
#   (4) the partial-run (.void_partial) tar also carries draws_log.json and init_from.json (a --resume needs draws_log.json).
#   Everything else is the original: status / NOT_FINISHED, small files first, 2 model slots, sha-verified 3-try fetch, destroy only after the fetch.
# original header:
# drain_gen.sh variant (2026-10-03, Audit: a failed run must not leave a 3090 idle): runs WITH eval.json on the host are
# fetched (checksummed, 3 tries) with logs/; runs without one are recorded FAILED (their logs come with logs/, their dirs
# are not fetched: local reruns may reuse the names; their args.json + log.jsonl ARE kept as <dir>.void_partial); then vdestroy + pods.txt.  bash scratch/drain_gen_partial.sh <host> <port> <tag> <id>
set -u
H=$1; P=$2; TAG=$3; ID=$4
cd "$(dirname "$0")/.."
S="ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
. scratch/s219_done_marker.sh || { echo "FETCH_INCOMPLETE $ID (scratch/s219_done_marker.sh)"; exit 1; }
REM="$(cat scratch/s219_done_marker.sh)"$'\n'"cd /workspace/Latent || exit 3; L=logs/queue_${TAG}.out; [ -f \$L ] || exit 4
echo QD=\$(tail -n 1 \$L | grep -c QUEUE_DONE); echo PROCS=\$(ps -eo args | grep -a -c -E '^[^ ]*python[^ ]* -u -m ')
QD_=\$(tail -n 1 \$L | grep -c QUEUE_DONE)
for j in \$(grep -h ' evaluated ' \$L | awk '{print \$3}'); do [ \"\$QD_\" = 1 ] && s219_seal_rename \$j; echo \$(s219_mark \$j) \$j; done; echo END_OK"
st=$($S "$REM" 2>/dev/null)
echo "$st" | grep -q "^END_OK$" || { echo "FETCH_INCOMPLETE $ID (status)"; exit 1; }
[ "$(echo "$st" | sed -n 's/^QD=//p')" = 1 ] && [ "$(echo "$st" | sed -n 's/^PROCS=//p')" = 0 ] || { echo "NOT_FINISHED $ID"; exit 1; }
seal=$(echo "$st" | sed -n 's/^SEAL_BREACH //p'); [ -n "$seal" ] && echo "SEAL_BREACH_ON_HOST $ID: $(echo $seal) (renamed *.void_unsealed on the host, fetched; TS_v1 §5.2)"
runs=$(echo "$st" | sed -n 's/^RUN //p'); bad=$(echo "$st" | sed -n 's/^BADM\{0,1\} //p')
# failed runs WITH a model.pt are fetched too (an eval OOM must not lose a trained model), unless a local dir of that name exists
keep=""; for j in $(echo "$st" | sed -n 's/^BADM //p'); do [ -e "$j" ] || keep="$keep $j"; done
mkdir -p logs/vast_${H}_${P}
# Model-skip rule (Theory + Audit 2026-10-06): scratch/drain_posthoc.py decides KEEP / SKIP per run ON THE HOST (and runs the registered
# on-host post-hoc before a SKIP); a SKIP drops only <run>/model.pt from the tarball and is logged to logs/model_skips.log. Any
# failure to decide -> KEEP.
skipset=" "   # §219 change (3): no model-skip post-hoc; every model.pt is fetched (TS_v1 §8 retention: T1's model.pt is T2's labeller)
# Audit 2026-10-06: (1) small files first - pass 1 fetches every run dir WITHOUT model.pt (eval / args / log / preds) plus logs, so
# a finished host's result is safe early; pass 2 fetches model.pt only for kept runs. (2) at most 2 model transfers at once across all
# drains (lock dirs D:/drain_tmp/locks/slot{1,2} holding the drain's pid; a dead holder's slot is reclaimed).
SLOT=""
acquire() { mkdir -p /d/drain_tmp/locks; while :; do for k in 1 2; do
    if mkdir /d/drain_tmp/locks/slot$k 2>/dev/null; then echo $$ > /d/drain_tmp/locks/slot$k/pid; SLOT=$k; return; fi
    op=$(cat /d/drain_tmp/locks/slot$k/pid 2>/dev/null); [ -n "$op" ] && ! kill -0 "$op" 2>/dev/null && rm -rf /d/drain_tmp/locks/slot$k
  done; sleep 30; done; }
release() { [ -n "$SLOT" ] && rm -rf /d/drain_tmp/locks/slot$SLOT; SLOT=""; }
trap release EXIT
fetch() {   # fetch <name> <remote tar arguments>: sha-verified, 3 tries, extracted with logs/ -> logs/vast_<host>_<port>/
  local n=$1; shift; local ok=0
  for try in 1 2 3; do
    rs=$($S "cd /workspace/Latent && rm -f /tmp/f_$n.tgz && tar -czf /tmp/f_$n.tgz $* 2>/dev/null; gzip -t /tmp/f_$n.tgz 2>/dev/null && sha256sum /tmp/f_$n.tgz || { df -h / | tail -1 | sed 's/^/ARCHIVE_TRUNCATED /'; }" 2>/dev/null | grep -o '^[0-9a-f]\{64\}')   # 2026-10-06: gzip -t on the host (b46: a full host disk truncated the archive and its sha still matched)
    scp -q -o ServerAliveInterval=15 -i ~/.ssh/runpod_ed25519 -P $P root@$H:/tmp/f_$n.tgz /d/drain_tmp/f_${H}_$n.tgz 2>/dev/null
    [ -n "$rs" ] && [ "$rs" = "$(sha256sum /d/drain_tmp/f_${H}_$n.tgz | cut -c1-64)" ] && { ok=1; break; }; sleep 20
  done
  [ $ok = 1 ] || return 1
  tar --no-same-owner -xzf /d/drain_tmp/f_${H}_$n.tgz -C . --transform "s|^logs/|logs/vast_${H}_${P}/|" && rm -f /d/drain_tmp/f_${H}_$n.tgz   # extracted and sha-verified: drop the local copy (C: filled up 2026-10-03)
}
for j in $runs $keep; do n=$(echo $j | tr '/' '_'); ck=""   # Audit 2026-10-08: a FINISHED run's ckpt_* (~3 GB) is not fetched (retention deletes it on arrival)
  case " $runs " in *" $j "*) ck="--exclude=$j/ckpt_last.pt --exclude=$j/ckpt_prev.pt";; esac   # resume ckpts only: kept copies (ckpt_s<step>_model.pt) still come
  grep -qxF "$j" scratch/keep_ckpt.txt 2>/dev/null && ck="--exclude=$j/ckpt_prev.pt"   # note 177-6 (2026-10-09): a run listed in scratch/keep_ckpt.txt keeps ckpt_last (a continuation point)
  fetch $n --exclude=$j/model.pt $ck $j logs/gen_$(basename $j).log || { echo "FETCH_INCOMPLETE $ID ($j small files)"; exit 1; }
done
fetch logs logs || { echo "FETCH_INCOMPLETE $ID (logs)"; exit 1; }
for j in $runs $keep; do case "$skipset" in *" $j "*) continue;; esac
  has=$($S "cd /workspace/Latent && { [ -e $j/model.pt ] && echo Y || echo N; }" 2>/dev/null | tail -n 1)
  [ "$has" = N ] && continue; [ "$has" = Y ] || { echo "FETCH_INCOMPLETE $ID ($j model check: no answer)"; exit 1; }
  n=$(echo $j | tr '/' '_')_model; acquire
  fetch $n $j/model.pt || { release; echo "FETCH_INCOMPLETE $ID ($j model.pt)"; exit 1; }
  release
done
for j in $runs; do r_=$(s219_check_fetched $j); [ "$r_" = OK ] || { echo "FETCH_INCOMPLETE $ID $j $r_"; exit 1; }   # §219 change (1)
  [ -s $j/train_done.json ] && { [ -s $j/model.pt ] || { echo "FETCH_INCOMPLETE $ID $j MISSING model.pt"; exit 1; }; }; done        # §219 change (3)
# Theory 2026-10-05 (aborted-run rule): failed runs not fetched above keep their partial args.json + log.jsonl as <dir>.void_partial
# (user 2026-10-06: + ckpt_last.pt (else ckpt_prev.pt: s219_resume_ckpt, Audit G-T13 cond. 2), args_resume_*.json and log.jsonl.pre_resume_* when present, so a crashed run resumes with --resume)
# (sha-verified; non-fatal - a miss is reported, the drain continues). Their run dir names stay free for local reruns.
vp=""; for j in $(echo "$st" | sed -n 's/^BADM\{0,1\} //p'); do case " $keep " in *" $j "*) continue;; esac
  n=$(echo $j | tr '/' '_')_vp; ok=0
  for try in 1 2 3; do
    rs=$($S "$(declare -f s219_resume_ckpt); cd /workspace/Latent && tar -czf /tmp/f_$n.tgz --ignore-failed-read $j/args*.json $j/log.jsonl* $j/draws_log.json $j/init_from.json \$(s219_resume_ckpt $j) 2>/dev/null; sha256sum /tmp/f_$n.tgz" 2>/dev/null | grep -o '^[0-9a-f]\{64\}')
    scp -q -o ServerAliveInterval=15 -i ~/.ssh/runpod_ed25519 -P $P root@$H:/tmp/f_$n.tgz /d/drain_tmp/f_${H}_$n.tgz 2>/dev/null
    [ -n "$rs" ] && [ "$rs" = "$(sha256sum /d/drain_tmp/f_${H}_$n.tgz | cut -c1-64)" ] && { ok=1; break; }; sleep 10
  done
  if [ $ok = 1 ] && tar --no-same-owner -xzf /d/drain_tmp/f_${H}_$n.tgz -C . --transform "s|^$j/|$j.void_partial/|" && [ -d $j.void_partial ]; then vp="$vp $j.void_partial"; else echo "PARTIAL_LOGS_UNFETCHED $j"; fi
  rm -f /d/drain_tmp/f_${H}_$n.tgz
done
bash scratch/vdestroy.sh "$ID" | grep -q "^DESTROYED_VERIFIED" || { echo "DESTROY_UNVERIFIED $ID"; exit 1; }
nsk=$(echo $skipset | wc -w)
echo "$ID vast | $TAG DESTROYED_VERIFIED $(date '+%Y-%m-%d %H:%M') after verified per-run fetch ($(echo $runs))${bad:+; FAILED (no §219 done marker, logs fetched): $(echo $bad)}${seal:+; SEAL_BREACH renamed *.void_unsealed: $(echo $seal)}${vp:+; partial logs kept:$vp}$( [ "$nsk" -gt 0 ] && echo "; model.pt not fetched (rule, logs/model_skips.log):$skipset")" >> scratch/pods.txt
echo "DRAINED $ID${bad:+ PARTIAL failed: $(echo $bad)}"
