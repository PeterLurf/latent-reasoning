#!/usr/bin/env bash
# §83 fetch v2 (replaces s83_fetch.sh: that host cuts transfers at ~1.2 GB). Waits for S83_DONE (hard stop 20:00, Audit:
# extension within the $1.50 cap at $0.169/h), fetches the small files (s83 JSONs, logs, run dirs WITHOUT model.pt) with a
# checksum, requires the models to be present locally already (fetched in 300 MB checksummed parts:
# boxes/runs_s5/wc0k4r_s{1,2,3}/model.pt), logs the model sha256s in pods.txt, then destroys.
cd "$(dirname "$0")/.."
H=137.175.22.196; P=60155; ID=53891982
S="ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
until $S "tail -n 1 /workspace/Latent/logs/s83_after.out 2>/dev/null" 2>/dev/null | grep -q S83_DONE; do
  [ "$(date +%H%M)" -ge 2000 ] && { echo "HARD STOP 20:00: fetching what exists"; break; }; sleep 120; done
ok=0
for try in 1 2 3; do
  rs=$($S "cd /workspace/Latent && for d in boxes/runs_s5/s83_wc0k4_s*; do [ -d \$d ] && mv \$d boxes/runs_s5/wc0k4r_s\${d##*_s}; done; tar --exclude='*/model.pt' -czf /tmp/s83small.tgz boxes/runs_s5/wc0k4r_s* theory/results/s83 logs && sha256sum /tmp/s83small.tgz" 2>/dev/null | grep -o '^[0-9a-f]\{64\}')
  scp -q -o ServerAliveInterval=15 -i ~/.ssh/runpod_ed25519 -P $P root@$H:/tmp/s83small.tgz /tmp/s83small.tgz 2>/dev/null
  [ -n "$rs" ] && [ "$rs" = "$(sha256sum /tmp/s83small.tgz | cut -c1-64)" ] && { ok=1; break; }; sleep 30
done
[ $ok = 1 ] || { echo "FETCH_FAILED (host kept)"; exit 1; }
mkdir -p logs/vast_s83 && tar --no-same-owner -xzf /tmp/s83small.tgz -C . --transform "s|^logs/|logs/vast_s83/|"
for s in 1 2 3; do [ -s boxes/runs_s5/wc0k4r_s$s/model.pt ] || { echo "MODEL MISSING s$s (host kept)"; exit 1; }; done
ls theory/results/s83/
r=$(bash scratch/vdestroy.sh $ID); echo "$r"
echo "$ID vast | s83-retrain $r $(date '+%Y-%m-%d %H:%M') after verified fetch (theory/results/s83: $(ls theory/results/s83 | tr '\n' ' '); models fetched in parts) model.pt sha256: $(for s in 1 2 3; do echo -n "s$s $(sha256sum boxes/runs_s5/wc0k4r_s$s/model.pt | cut -c1-16) "; done)" >> scratch/pods.txt
echo S83_FETCHED
