#!/usr/bin/env bash
# Destroy a Vast instance and verify it is gone:  bash scratch/vdestroy.sh <instance id>
# Prints DESTROYED_VERIFIED <id>, or STILL_LIVE <id> with exit 1.
V="C:/Users/peter/AppData/Local/Temp/claude/D--/6af9c751-a423-5604-9170-0c177c13d67f/scratchpad/vast.sh"
export PYTHONIOENCODING=utf-8
bash "$V" destroy instance "$1" -y > /dev/null 2>&1 || echo y | bash "$V" destroy instance "$1" > /dev/null 2>&1
for i in 1 2 3 4 5 6; do
  bash "$V" show instances --raw 2>/dev/null | python -c "import json,sys;ids={d['id'] for d in json.load(sys.stdin)};sys.exit(0 if int(sys.argv[1]) in ids else 1)" "$1" || { echo "DESTROYED_VERIFIED $1"; exit 0; }
  sleep 10
done
echo "STILL_LIVE $1"; exit 1
