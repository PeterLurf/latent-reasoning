#!/usr/bin/env bash
# Copy one finished cloud run back into local runs/ and logs/ (checksum-verified):  bash scratch/pod_fetch.sh <ssh-user> <run>
# Brings runs/<run> (model.pt, config.json, log.jsonl; state.pt is left out, it is only needed to resume),
# the config and log of every <run>_stage1fail_* attempt, runs/<run>.host.txt, the guard/stdout/stderr/eval logs,
# and results/*_<run>.json from the scoring battery.
set -eu
U=$1; r=$2
cd "$(dirname "$0")/.."
[ -e "runs/$r/model.pt" ] && echo "[$r] present locally; refreshing from the pod (the pod is the source for cloud runs)"
out=$(printf 'stty -echo; cd /workspace/Latent && tar -czf /tmp/f.tgz --exclude=state.pt --exclude="*.tmp" runs/%s runs/%s_stage1fail_* runs/%s.host.txt logs/guard_%s.log logs/%s.out logs/%s.err logs/pod_run_%s.out logs/eval_%s.log results/*_%s.json 2>/dev/null; echo "SHA=$(sha256sum /tmp/f.tgz | cut -c1-64)"; echo BEGIN_B64; base64 -w 76 /tmp/f.tgz; echo END_B64; exit\n' $r $r $r $r $r $r $r $r $r \
  | timeout 900 ssh -tt -i ~/.ssh/runpod_ed25519 $U@ssh.runpod.io 2>&1 | tr -d '\r')
sha=$(printf '%s\n' "$out" | grep -aoE 'SHA=[0-9a-f]{64}' | cut -c5-)
printf '%s\n' "$out" | sed -n '/^BEGIN_B64$/,/^END_B64$/p' | sed '1d;$d' | base64 -d > "/tmp/fetch_$r.tgz"
got=$(sha256sum "/tmp/fetch_$r.tgz" | cut -c1-64)
[ -n "$sha" ] && [ "$sha" = "$got" ] || { echo "[$r] CHECKSUM MISMATCH remote=$sha local=$got"; exit 2; }
tar --no-same-owner -xzf "/tmp/fetch_$r.tgz" && echo "[$r] fetched $(du -h /tmp/fetch_$r.tgz | cut -f1), sha ok; $(ls runs/$r)"
