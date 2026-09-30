#!/usr/bin/env bash
# Drain a finished toy-queue Vast host: confirm the queue has finished, fetch (checksummed, with retries), verify every
# run is present locally, and only then destroy. Any failed step destroys nothing (Audit lessons 2026-09-29: theory-L01,
# and the review of the first version of this script).
#   bash scratch/drain_toy.sh <host> <port> <queue tag prefix, e.g. t9 | p9> <instance id> <queue tag, e.g. t9_vA>
# 1. One status call, which must end with END_OK. The last line of logs/queue_<tag>.out must be QUEUE_DONE, and no
#    python training or guard process may be alive. Otherwise NOT_FINISHED.
# 2. Fetch with scratch/vfetch_toy.sh (3 tries).
# 3. Every run the queue marks "evaluated" must be one of:
#    - a positively identified no-model run (the queue log has "no model for <r>"), with its guard log fetched;
#    - otherwise, runs/<r>/{model.pt,config.json,log.jsonl} and results/{competence_matched,length_gen}_<r>.json
#      present and non-empty locally.
# Prints DRAINED <id>, or NOT_FINISHED / FETCH_INCOMPLETE / DESTROY_UNVERIFIED with exit 1.
set -u
H=$1; P=$2; Q=$3; ID=$4; TAG=$5
cd "$(dirname "$0")/.."
S="ssh -o ConnectTimeout=20 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
st=$($S "cd /workspace/Latent || exit 3
L=logs/queue_${TAG}.out
[ -f \$L ] || exit 4
echo QD=\$(tail -n 1 \$L | grep -c QUEUE_DONE)
echo PROCS=\$(ps -eo args | grep -a -c -E '^[^ ]*python[^ ]* .*(stage1_guard|latent\.train|train_arch|latent\.)')
for r in \$(grep -h ' evaluated ' \$L | awk '{print \$3}'); do
  echo RUN \$r \$([ -f runs/\$r/model.pt ] && echo 1 || echo 0) \$(grep -c \"no model for \$r\\\$\" \$L)
done
echo END_OK" 2>/dev/null)
echo "$st" | grep -q "^END_OK$" || { echo "FETCH_INCOMPLETE $ID (status call failed)"; exit 1; }
qd=$(echo "$st" | sed -n 's/^QD=//p'); pr=$(echo "$st" | sed -n 's/^PROCS=//p')
[ "$qd" = 1 ] && [ "$pr" = 0 ] || { echo "NOT_FINISHED $ID (QUEUE_DONE last line: $qd, live processes: $pr)"; exit 1; }
mapfile -t rows < <(echo "$st" | grep "^RUN ")
[ ${#rows[@]} -gt 0 ] || { echo "FETCH_INCOMPLETE $ID (no evaluated runs listed)"; exit 1; }
ok=0
for try in 1 2 3; do
  bash scratch/vfetch_toy.sh "$H" "$P" "$Q" && { ok=1; break; }
  echo "fetch attempt $try failed"; sleep 30
done
[ $ok -eq 1 ] || { echo "FETCH_INCOMPLETE $ID (fetch failed 3x)"; exit 1; }
missing=(); nm=0; wm=0
for row in "${rows[@]}"; do
  read -r _ r has nomodel <<<"$row"
  if [ "$has" = 1 ]; then
    wm=$((wm + 1))
    for f in runs/$r/model.pt runs/$r/config.json runs/$r/log.jsonl results/competence_matched_$r.json results/length_gen_$r.json; do
      [ -s "$f" ] || missing+=("$f")
    done
  elif [ "$nomodel" -ge 1 ] 2>/dev/null; then
    nm=$((nm + 1))
    [ -s "logs/vast_${H}_${P}/guard_$r.log" ] || missing+=("guard_$r.log")
  else
    missing+=("$r(no model on host and no 'no model for' line)")
  fi
done
[ ${#missing[@]} -eq 0 ] || { echo "FETCH_INCOMPLETE $ID missing: ${missing[*]}"; exit 1; }
echo "verified ${#rows[@]} runs from $TAG: $wm with model, $nm stage-1 failures"
bash scratch/vdestroy.sh "$ID" | grep -q "^DESTROYED_VERIFIED" || { echo "DESTROY_UNVERIFIED $ID"; exit 1; }
echo "$ID vast | $TAG DESTROYED_VERIFIED $(date '+%Y-%m-%d %H:%M') after verified fetch (${#rows[@]} runs: $wm with model, $nm stage-1 failures)" >> scratch/pods.txt
echo "DRAINED $ID"
