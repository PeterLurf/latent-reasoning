#!/usr/bin/env bash
# §83: wait for S83_DONE on the retrain host (cap: 4 h from 13:30 -> 17:30), checksummed fetch of runs (with model.pt),
# s83 JSONs and logs, then destroy and log in pods.txt.
cd "$(dirname "$0")/.."
H=137.175.22.196; P=60155; ID=53891982
S="ssh -o ConnectTimeout=20 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
until $S "tail -n 1 /workspace/Latent/logs/s83_after.out 2>/dev/null" 2>/dev/null | grep -q S83_DONE; do
  [ "$(date +%H%M)" -ge 1730 ] && { echo "CAP reached, fetching what exists"; break; }; sleep 120; done
ok=0
for try in 1 2 3; do
  rs=$($S "cd /workspace/Latent && tar -czf /tmp/s83.tgz boxes/runs_s5/wc0k4r_s* theory/results/s83 logs && sha256sum /tmp/s83.tgz" 2>/dev/null | grep -o '^[0-9a-f]\{64\}')
  scp -q -i ~/.ssh/runpod_ed25519 -P $P root@$H:/tmp/s83.tgz /tmp/s83_local.tgz 2>/dev/null
  [ -n "$rs" ] && [ "$rs" = "$(sha256sum /tmp/s83_local.tgz | cut -c1-64)" ] && { ok=1; break; }; sleep 30
done
[ $ok = 1 ] || { echo "FETCH_FAILED (host kept)"; exit 1; }
mkdir -p logs/vast_s83 && tar --no-same-owner -xzf /tmp/s83_local.tgz -C . --transform "s|^logs/|logs/vast_s83/|"
ls boxes/runs_s5/wc0k4r_s*/model.pt theory/results/s83/*.json
r=$(bash scratch/vdestroy.sh $ID); echo "$r"
echo "$ID vast | s83-retrain $r $(date '+%Y-%m-%d %H:%M') after verified fetch (boxes/runs_s5/wc0k4r_s1-3 with model.pt, theory/results/s83)" >> scratch/pods.txt
echo S83_FETCHED
