#!/usr/bin/env bash
# Abort a host whose runs are void (e.g. §98/§99 GEN-M final-slot fork conflict, note 98-3): kill its queue and jobs,
# fetch logs/ and every run's args/log.jsonl/eval.json (no model.pt) checksummed into logs/vast_<h>_<p>/aborted/, then
# vdestroy + pods.txt.   bash scratch/abort_host.sh <host> <port> <tag> <instance> "<reason>"
set -u
H=$1; P=$2; TAG=$3; ID=$4; WHY=$5
cd "$(dirname "$0")/.."
S="ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
D=logs/vast_${H}_${P}/aborted; mkdir -p $D
rs=$($S "pkill -f '[g]en_vast_job.sh'; pkill -f '[p]ython -u -m boxes'; sleep 3; cd /workspace/Latent && tar -czf /tmp/abort.tgz logs \$(ls -d boxes/runs_mult/*/ boxes/runs_s5/*/ 2>/dev/null | while read d; do for f in args.json log.jsonl eval.json; do [ -f \$d\$f ] && echo \$d\$f; done; done) 2>/dev/null; sha256sum /tmp/abort.tgz" 2>/dev/null | grep -o '^[0-9a-f]\{64\}')
scp -q -o ServerAliveInterval=15 -i ~/.ssh/runpod_ed25519 -P $P root@$H:/tmp/abort.tgz /tmp/abort_${H}_${P}.tgz
[ -n "$rs" ] && [ "$rs" = "$(sha256sum /tmp/abort_${H}_${P}.tgz | cut -c1-64)" ] || { echo "ABORT_FETCH_FAILED $ID"; exit 1; }
tar --no-same-owner -xzf /tmp/abort_${H}_${P}.tgz -C $D && rm -f /tmp/abort_${H}_${P}.tgz
bash scratch/vdestroy.sh "$ID" | grep -q "^DESTROYED_VERIFIED" || { echo "DESTROY_UNVERIFIED $ID"; exit 1; }
echo "$ID vast | $TAG DESTROYED_VERIFIED $(date '+%Y-%m-%d %H:%M') ABORTED: $WHY; logs + run args/log.jsonl/eval.json (no models) in $D" >> scratch/pods.txt
echo "ABORTED $ID -> $D"
