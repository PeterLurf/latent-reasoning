#!/usr/bin/env bash
# Destroy a Vast instance and verify it is gone:  bash scratch/vdestroy.sh <instance id>
# Prints DESTROYED_VERIFIED <id> (exit 0), STILL_LIVE <id> (exit 1) or UNKNOWN <id> (exit 2: the instance list could not
# be read, so nothing is claimed). A failed `show instances` is never taken as "gone" (Audit review 2026-09-29).
V="C:/Users/peter/.local/vastai/vast.sh"
[ -f "$V" ] || { echo "UNKNOWN $1 (no CLI at $V)"; exit 2; }
export PYTHONIOENCODING=utf-8
bash "$V" destroy instance "$1" -y > /dev/null 2>&1 || echo y | bash "$V" destroy instance "$1" > /dev/null 2>&1
rc=2
for i in 1 2 3 4 5 6; do
  bash "$V" show instances --raw 2>/dev/null | python -c "
import json, sys
try:
    ids = {d['id'] for d in json.load(sys.stdin)}
except Exception:
    sys.exit(2)
sys.exit(0 if int(sys.argv[1]) in ids else 1)" "$1"
  rc=$?
  [ $rc -eq 1 ] && { echo "DESTROYED_VERIFIED $1"; exit 0; }
  sleep 10
done
[ $rc -eq 0 ] && { echo "STILL_LIVE $1"; exit 1; }
echo "UNKNOWN $1 (instance list unreadable)"; exit 2
