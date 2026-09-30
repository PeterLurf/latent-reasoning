#!/usr/bin/env bash
# Copy a direct-SSH host's ProsQA results (and optionally the final checkpoint) into prosqa/, checksum-verified.
#   bash prosqa/vfetch.sh <host> <port> <local res dir> [ckpt path on host] [local ckpt path]
set -eu
H=$1; P=$2; OUT=$3
cd "$(dirname "$0")/.."
S="ssh -o ConnectTimeout=20 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
$S 'cd /workspace && tar -czf /tmp/res.tgz prosqa/res logs && sha256sum /tmp/res.tgz' 2>/dev/null | grep -o '^[0-9a-f]\{64\}' > /tmp/res_${H}_${P}.sha
scp -q -i ~/.ssh/runpod_ed25519 -P $P root@$H:/tmp/res.tgz /tmp/res_g_${H}_${P}.tgz
[ -s /tmp/res_${H}_${P}.sha ] && [ "$(cut -c1-64 /tmp/res_${H}_${P}.sha)" = "$(sha256sum /tmp/res_g_${H}_${P}.tgz | cut -c1-64)" ] || { echo "RES CHECKSUM MISMATCH"; exit 2; }
rm -rf /tmp/res_g_${H}_${P} && mkdir -p /tmp/res_g_${H}_${P} "$OUT" && tar --no-same-owner -xzf /tmp/res_g_${H}_${P}.tgz -C /tmp/res_g_${H}_${P}
cp -r /tmp/res_g_${H}_${P}/prosqa/res/. "$OUT"/ && mkdir -p "$OUT/logs" && cp /tmp/res_g_${H}_${P}/logs/* "$OUT/logs/"
echo "res fetched, sha ok: $(ls "$OUT" | tr '\n' ' ')"
if [ $# -ge 5 ]; then
  scp -q -i ~/.ssh/runpod_ed25519 -P $P root@$H:"$4" "$5"
  r=$($S "sha256sum $4" 2>/dev/null | grep -o '^[0-9a-f]\{64\}'); l=$(sha256sum "$5" | cut -c1-64)
  [ "$r" = "$l" ] && echo "ckpt fetched, sha ok" || { echo "CKPT CHECKSUM MISMATCH"; exit 3; }
fi
