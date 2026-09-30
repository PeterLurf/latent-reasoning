#!/usr/bin/env bash
# Fetch evaluated theory-stream runs from a direct-SSH Vast host into theory/runs/ (checksum-verified).
#   bash theory/vfetch_arch.sh <host> <port>
# Per run: model.pt, config.json, arch.json, log.jsonl, eval.json, and step checkpoints every 3000 steps + curriculum
# advances (for training-dynamics analyses). Queue and guard logs go to theory/runs/_logs_<host>/.
set -eu
H=$1; P=$2
cd "$(dirname "$0")/.."
# one fetch per host at a time; different hosts may fetch in parallel (per-host temp files)
L=/tmp/vfetch_${H}_${P}; until mkdir $L.lock 2>/dev/null; do sleep 5; done
trap 'rmdir $L.lock' EXIT
S="ssh -o ConnectTimeout=20 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
HAVE=" $(for d in theory/runs/*/; do [ -f "$d/eval.json" ] && basename "$d"; done | paste -sd" " -) "
$S "HAVE='$HAVE'; "'cd /workspace/Latent && files="" && for d in theory/runs/*/; do r=${d%/}; [ -f $r/eval.json ] || continue; case "$HAVE" in *" $(basename $r) "*) continue;; esac; for f in model.pt config.json arch.json log.jsonl eval.json; do [ -f $r/$f ] && files="$files $r/$f"; done; adv=$(grep -o "\"step\": [0-9]*, \"curriculum_advance_to\"" $r/log.jsonl | grep -o "[0-9]*" | while read s; do printf "%s/ckpts/step_%05d.pt " $r $s; done); for c in $r/ckpts/step_*.pt; do [ -f $c ] || continue; n=$(basename $c .pt); n=${n#step_}; [ $((10#$n % 3000)) -eq 0 ] && files="$files $c"; done; for c in $adv; do [ -f $c ] && files="$files $c"; done; done; tar -czf /tmp/arch.tgz $files $(ls logs/queue_theory-*.out logs/guard_*.log 2>/dev/null) && sha256sum /tmp/arch.tgz' 2>/dev/null | grep -o '^[0-9a-f]\{64\}' > $L.sha || true
[ -s $L.sha ] || { echo "NOTHING TO FETCH"; exit 0; }
scp -q -i ~/.ssh/runpod_ed25519 -P $P root@$H:/tmp/arch.tgz $L.tgz
[ "$(cut -c1-64 $L.sha)" = "$(sha256sum $L.tgz | cut -c1-64)" ] || { echo "CHECKSUM MISMATCH"; exit 2; }
mkdir -p theory/runs/_logs_$H && tar --no-same-owner -xzf $L.tgz -C . --transform "s|^logs/|theory/runs/_logs_$H/|"
echo "fetched, sha ok: $(tar -tzf $L.tgz | grep -c eval.json) runs"
