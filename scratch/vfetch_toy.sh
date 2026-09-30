#!/usr/bin/env bash
# Fetch finished toy runs (model, config, log; no step checkpoints) and their eval JSON from a direct-SSH Vast host,
# checksum-verified, into runs/ and results/.   bash scratch/vfetch_toy.sh <host> <port>
# Runs without a model.pt (stage-1 failures after the relaunch budget) are skipped; their guard logs come along.
set -eu
H=$1; P=$2; Q=${3:-p30}
cd "$(dirname "$0")/.."
S="ssh -o ConnectTimeout=20 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
$S "cd /workspace/Latent && runs=\$(grep -h evaluated logs/queue_${Q}_*.out | awk '{print \$3}') && [ -n \"\$runs\" ] && files=\$(for r in \$runs; do [ -f runs/\$r/model.pt ] || continue; echo runs/\$r/model.pt runs/\$r/config.json runs/\$r/log.jsonl; ls results/*_\$r.json; done) && tar -czf /tmp/toy.tgz \$files logs/queue_${Q}_*.out logs/guard_*.log && sha256sum /tmp/toy.tgz" 2>/dev/null | grep -o '^[0-9a-f]\{64\}' > /tmp/toy_${H}_${P}.sha
scp -q -i ~/.ssh/runpod_ed25519 -P $P root@$H:/tmp/toy.tgz /tmp/toy_l_${H}_${P}.tgz
[ -s /tmp/toy_${H}_${P}.sha ] && [ "$(cut -c1-64 /tmp/toy_${H}_${P}.sha)" = "$(sha256sum /tmp/toy_l_${H}_${P}.tgz | cut -c1-64)" ] || { echo "CHECKSUM MISMATCH"; exit 2; }
mkdir -p logs/vast_${H}_${P} && tar --no-same-owner -xzf /tmp/toy_l_${H}_${P}.tgz -C . --transform "s|^logs/|logs/vast_${H}_${P}/|"
echo "fetched, sha ok: $(tar -tzf /tmp/toy_l_${H}_${P}.tgz | grep -c model.pt) runs"
