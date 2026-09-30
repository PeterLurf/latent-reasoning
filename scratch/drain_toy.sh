#!/usr/bin/env bash
# Drain a finished toy-queue Vast host: fetch (with retries), verify every run the queue finished is present locally,
# and only then destroy. A failed or incomplete fetch never destroys (Audit lesson 2026-09-29: theory-L01).
#   bash scratch/drain_toy.sh <host> <port> <queue tag prefix, e.g. t9 | p9> <instance id> <queue tag, e.g. t9_vA>
# Verified means, for every run the host's queue log marks "evaluated":
#   - with a model on the host: runs/<r>/model.pt, runs/<r>/config.json, runs/<r>/log.jsonl,
#     results/competence_matched_<r>.json and results/length_gen_<r>.json exist locally and are non-empty;
#   - without a model (stage-1 failure after the relaunch budget): its guard log was fetched.
# Prints DRAINED <id> (destroyed and verified gone), or FETCH_INCOMPLETE / DESTROY_UNVERIFIED with exit 1.
set -u
H=$1; P=$2; Q=$3; ID=$4; TAG=$5
cd "$(dirname "$0")/.."
S="ssh -o ConnectTimeout=20 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
ok=0
for try in 1 2 3; do
  bash scratch/vfetch_toy.sh "$H" "$P" "$Q" && { ok=1; break; }
  echo "fetch attempt $try failed"; sleep 30
done
[ $ok -eq 1 ] || { echo "FETCH_INCOMPLETE $ID (fetch failed 3x)"; exit 1; }
mapfile -t runs < <($S "grep -h ' evaluated ' /workspace/Latent/logs/queue_${TAG}.out | awk '{print \$3}'" 2>/dev/null)
mapfile -t withmodel < <($S "cd /workspace/Latent; for r in \$(grep -h ' evaluated ' logs/queue_${TAG}.out | awk '{print \$3}'); do [ -f runs/\$r/model.pt ] && echo \$r; done" 2>/dev/null)
[ ${#runs[@]} -gt 0 ] || { echo "FETCH_INCOMPLETE $ID (could not list the host's evaluated runs)"; exit 1; }
missing=()
for r in "${runs[@]}"; do
  if printf '%s\n' "${withmodel[@]}" | grep -qx "$r"; then
    for f in runs/$r/model.pt runs/$r/config.json runs/$r/log.jsonl results/competence_matched_$r.json results/length_gen_$r.json; do
      [ -s "$f" ] || missing+=("$f")
    done
  else
    ls logs/vast_${H}_${P}/guard_$r.log >/dev/null 2>&1 || missing+=("guard_$r.log")
  fi
done
[ ${#missing[@]} -eq 0 ] || { echo "FETCH_INCOMPLETE $ID missing: ${missing[*]}"; exit 1; }
echo "verified ${#runs[@]} runs (${#withmodel[@]} with model) from $TAG"
bash scratch/vdestroy.sh "$ID" | grep -q "^DESTROYED_VERIFIED" || { echo "DESTROY_UNVERIFIED $ID"; exit 1; }
echo "$ID vast | $TAG DESTROYED_VERIFIED $(date '+%Y-%m-%d %H:%M') after verified fetch (${#runs[@]} runs, ${#withmodel[@]} with model)" >> scratch/pods.txt
echo "DRAINED $ID"
