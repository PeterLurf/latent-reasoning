#!/usr/bin/env bash
# Copy the ProsQA pod's analysis results and logs into prosqa/res and prosqa/logs (checksum-verified).
#   bash prosqa/fetch.sh <ssh-user>
set -eu
U=$1
cd "$(dirname "$0")/.."
mkdir -p prosqa/res prosqa/logs
out=$(printf 'stty -echo; cd /workspace && tar -czf /tmp/f.tgz prosqa/res logs/prosqa*.out logs/test_patch.out coconut/args/prosqa_1gpu.yaml 2>/dev/null; echo "SHA=$(sha256sum /tmp/f.tgz | cut -c1-64)"; echo BEGIN_B64; base64 -w 76 /tmp/f.tgz; echo END_B64; exit\n' \
  | timeout 900 ssh -tt -i ~/.ssh/runpod_ed25519 $U@ssh.runpod.io 2>&1 | tr -d '\r')
sha=$(printf '%s\n' "$out" | grep -aoE 'SHA=[0-9a-f]{64}' | cut -c5-)
printf '%s\n' "$out" | sed -n '/^BEGIN_B64$/,/^END_B64$/p' | sed '1d;$d' | base64 -d > /tmp/fetch_prosqa.tgz
got=$(sha256sum /tmp/fetch_prosqa.tgz | cut -c1-64)
[ -n "$sha" ] && [ "$sha" = "$got" ] || { echo "[prosqa] CHECKSUM MISMATCH remote=$sha local=$got"; exit 2; }
rm -rf /tmp/fetch_prosqa && mkdir -p /tmp/fetch_prosqa && tar --no-same-owner -xzf /tmp/fetch_prosqa.tgz -C /tmp/fetch_prosqa
cp -r /tmp/fetch_prosqa/prosqa/res/. prosqa/res/ && cp /tmp/fetch_prosqa/logs/* prosqa/logs/ && cp /tmp/fetch_prosqa/coconut/args/prosqa_1gpu.yaml prosqa/logs/
echo "[prosqa] fetched $(du -h /tmp/fetch_prosqa.tgz | cut -f1), sha ok; res: $(ls prosqa/res | tr '\n' ' ')"
