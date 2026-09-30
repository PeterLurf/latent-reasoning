#!/usr/bin/env bash
# Drain a finished ProsQA Vast host: fetch res/ + the final checkpoint (checksummed, with retries), verify the expected
# result files are present locally and non-empty, and only then destroy. Requires ANALYSIS_DONE on the host and no live
# training/eval process first (NOT_FINISHED otherwise). Never destroys after a failed or incomplete fetch.
#   bash scratch/drain_prosqa.sh <host> <port> <instance id> <local res dir> <ckpt path on host> <local ckpt path> <expected res files...>
set -u
H=$1; P=$2; ID=$3; OUT=$4; RC=$5; LC=$6; shift 6
cd "$(dirname "$0")/.."
S="ssh -o ConnectTimeout=20 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
st=$($S "echo AD=\$(grep -a -c ANALYSIS_DONE /workspace/logs/prosqa.out)
echo PROCS=\$(ps -eo args | grep -a -c -E '^[^ ]*(python|torchrun)[^ ]* .*(run\.py|long_eval|eval_p8|analyze|pod_p8)')
echo END_OK" 2>/dev/null)
echo "$st" | grep -q "^END_OK$" || { echo "FETCH_INCOMPLETE $ID (status call failed)"; exit 1; }
ad=$(echo "$st" | sed -n 's/^AD=//p'); pr=$(echo "$st" | sed -n 's/^PROCS=//p')
[ "$ad" -ge 1 ] 2>/dev/null && [ "$pr" = 0 ] || { echo "NOT_FINISHED $ID (ANALYSIS_DONE: $ad, live processes: $pr)"; exit 1; }
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
