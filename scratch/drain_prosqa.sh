#!/usr/bin/env bash
# Drain a finished ProsQA Vast host: fetch res/ + the final checkpoint (checksummed, with retries), verify the expected
# result files are present locally and non-empty, and only then destroy. Never destroys after a failed or incomplete fetch.
#   bash scratch/drain_prosqa.sh <host> <port> <instance id> <local res dir> <ckpt path on host> <local ckpt path> <expected res files...>
set -u
H=$1; P=$2; ID=$3; OUT=$4; RC=$5; LC=$6; shift 6
cd "$(dirname "$0")/.."
ok=0
for try in 1 2 3; do
  bash prosqa/vfetch.sh "$H" "$P" "$OUT" "$RC" "$LC" | tee /tmp/drain_${ID}.log | grep -q "ckpt fetched, sha ok" && { ok=1; break; }
  echo "fetch attempt $try failed: $(tail -1 /tmp/drain_${ID}.log)"; sleep 30
done
[ $ok -eq 1 ] || { echo "FETCH_INCOMPLETE $ID"; exit 1; }
missing=()
for f in "$@"; do [ -s "$OUT/$f" ] || missing+=("$f"); done
[ -s "$LC" ] || missing+=("$LC")
[ ${#missing[@]} -eq 0 ] || { echo "FETCH_INCOMPLETE $ID missing: ${missing[*]}"; exit 1; }
bash scratch/vdestroy.sh "$ID" | grep -q "^DESTROYED_VERIFIED" || { echo "DESTROY_UNVERIFIED $ID"; exit 1; }
echo "$ID vast | DESTROYED_VERIFIED $(date '+%Y-%m-%d %H:%M') after verified fetch ($OUT: $*; ckpt $LC)" >> scratch/pods.txt
echo "DRAINED $ID"
