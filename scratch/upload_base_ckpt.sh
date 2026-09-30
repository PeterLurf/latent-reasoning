#!/usr/bin/env bash
# Upload the Phase 0 checkpoint (+ .sha256) to PROTOCOL8b step-1 hosts as they come up.
#   bash scratch/upload_base_ckpt.sh <deploy-output file> ...
cd "$(dirname "$0")/.."
sha256sum prosqa/ckpt/p8_coconut_checkpoint_48 | sed 's#prosqa/ckpt/##' > /tmp/p8ck.sha256
pending=("$@")
end=$((SECONDS+2400))
while [ ${#pending[@]} -gt 0 ] && [ $SECONDS -lt $end ]; do
  left=()
  for f in "${pending[@]}"; do
    if grep -q STARTED "$f" 2>/dev/null; then
      read -r _ id h p <<<"$(grep -o 'INSTANCE [0-9]* [0-9.]* [0-9]*' "$f")"
      if scp -q -o ConnectTimeout=20 -i ~/.ssh/runpod_ed25519 -P $p prosqa/ckpt/p8_coconut_checkpoint_48 root@$h:/workspace/ckpt/p8_coconut_checkpoint_48 &&
         scp -q -o ConnectTimeout=20 -i ~/.ssh/runpod_ed25519 -P $p /tmp/p8ck.sha256 root@$h:/workspace/ckpt/p8_coconut_checkpoint_48.sha256; then
        echo "$(date +%T) uploaded to $id ($h:$p) from $(basename $f)"
      else left+=("$f"); fi
    elif grep -q -E "BAD HOST|CREATE FAILED" "$f" 2>/dev/null; then
      echo "$(date +%T) $(basename $f): $(grep -o -E 'BAD HOST|CREATE FAILED|INSTANCE [0-9]*' $f | tr '\n' ' ')"
    else left+=("$f"); fi
  done
  pending=("${left[@]}")
  [ ${#pending[@]} -gt 0 ] && sleep 30
done
echo "$(date +%T) UPLOADS_DONE pending: ${pending[*]}"
