#!/usr/bin/env bash
# Drain a finished T12 host (theory-layout runs in theory/runs/<r>): queue finished + no live process, checksummed fetch
# of every evaluated run's model.pt/config.json/arch.json/log.jsonl/eval.json (plus queue and guard logs), local
# verification, then vdestroy. Never destroys on a failed or incomplete fetch.
#   bash scratch/drain_t12.sh <host> <port> <queue tag> <instance id>
set -u
H=$1; P=$2; TAG=$3; ID=$4
cd "$(dirname "$0")/.."
S="ssh -o ConnectTimeout=20 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
st=$($S "cd /workspace/Latent || exit 3
L=logs/queue_${TAG}.out
[ -f \$L ] || exit 4
echo QD=\$(tail -n 1 \$L | grep -c QUEUE_DONE)
echo PROCS=\$(ps -eo args | grep -a -c -E '^[^ ]*python[^ ]* .*(stage1_guard|train_arch|eval_arch)')
for r in \$(grep -h ' evaluated ' \$L | awk '{print \$3}'); do echo RUN \$r \$([ -f theory/runs/\$r/model.pt ] && echo 1 || echo 0) \$(grep -c \"no model for \$r\\$\" \$L); done
echo END_OK" 2>/dev/null)
echo "$st" | grep -q "^END_OK$" || { echo "FETCH_INCOMPLETE $ID (status call failed)"; exit 1; }
[ "$(echo "$st" | sed -n 's/^QD=//p')" = 1 ] && [ "$(echo "$st" | sed -n 's/^PROCS=//p')" = 0 ] || { echo "NOT_FINISHED $ID"; exit 1; }
mapfile -t rows < <(echo "$st" | grep "^RUN ")
[ ${#rows[@]} -gt 0 ] || { echo "FETCH_INCOMPLETE $ID (no runs)"; exit 1; }
files=""; for row in "${rows[@]}"; do read -r _ r has _ <<<"$row"; [ "$has" = 1 ] && files="$files theory/runs/$r/model.pt theory/runs/$r/config.json theory/runs/$r/arch.json theory/runs/$r/log.jsonl theory/runs/$r/eval.json"; done
ok=0
for try in 1 2 3; do
  rs=$($S "cd /workspace/Latent && tar -czf /tmp/t12.tgz $files logs/queue_${TAG}.out logs/guard_*.log logs/pt12_*.err && sha256sum /tmp/t12.tgz" 2>/dev/null | grep -o '^[0-9a-f]\{64\}')
  scp -q -i ~/.ssh/runpod_ed25519 -P $P root@$H:/tmp/t12.tgz /tmp/t12_${H}_${P}.tgz 2>/dev/null
  [ -n "$rs" ] && [ "$rs" = "$(sha256sum /tmp/t12_${H}_${P}.tgz | cut -c1-64)" ] && { ok=1; break; }
  echo "fetch attempt $try failed"; sleep 30
done
[ $ok -eq 1 ] || { echo "FETCH_INCOMPLETE $ID (fetch failed 3x)"; exit 1; }
mkdir -p logs/vast_${H}_${P} && tar --no-same-owner -xzf /tmp/t12_${H}_${P}.tgz -C . --transform "s|^logs/|logs/vast_${H}_${P}/|"
missing=()
for row in "${rows[@]}"; do read -r _ r has nm <<<"$row"
  if [ "$has" = 1 ]; then for f in model.pt config.json arch.json log.jsonl eval.json; do [ -s theory/runs/$r/$f ] || missing+=("$r/$f"); done
  elif [ "$nm" -ge 1 ] 2>/dev/null; then [ -s logs/vast_${H}_${P}/guard_$r.log ] || missing+=("guard_$r.log")
  else missing+=("$r(no model, no marker)"); fi
done
[ ${#missing[@]} -eq 0 ] || { echo "FETCH_INCOMPLETE $ID missing: ${missing[*]}"; exit 1; }
bash scratch/vdestroy.sh "$ID" | grep -q "^DESTROYED_VERIFIED" || { echo "DESTROY_UNVERIFIED $ID"; exit 1; }
echo "$ID vast | $TAG DESTROYED_VERIFIED $(date '+%Y-%m-%d %H:%M') after verified fetch (${#rows[@]} T12 runs)" >> scratch/pods.txt
echo "DRAINED $ID"
