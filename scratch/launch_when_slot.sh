#!/usr/bin/env bash
# Launch gen queues one at a time as fleet slots free (2026-10-12, §191): for each n, wait until `vast show instances` lists < MAX live
# (default 5, Theory's fleet cap) and no fill_generic runs, then run fill_generic <tag> <n> in the foreground (it returns after STARTED).
# Gate = the live-instance count from the API, never a log grep (latent-no-loggrep-deploy-gates).
#   nohup bash scratch/launch_when_slot.sh <tag> <n> [<n> ...] > logs/launch_<tag>.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
V="C:/Users/peter/.local/vastai/vast.sh"; MAX=${MAX:-5}; tag=$1; shift
for n in "$@"; do
  while true; do
    nl=$(bash "$V" show instances --raw 2>/dev/null | python -c "import json,sys;print(len(json.load(sys.stdin)))" 2>/dev/null)
    fr=$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='bash.exe'\" | Where-Object { \$_.CommandLine -match 'bash\.exe.? scratch/fill_generic\.sh' }).Count" 2>/dev/null | tr -d '\r')
    if [ -n "$nl" ] && [ "$nl" -lt "$MAX" ] && [ "${fr:-1}" = "0" ]; then break; fi
    sleep 300
  done
  echo "$(date '+%F %T') $tag b$n: slot free ($nl live < $MAX), launching fill_generic"
  bash scratch/fill_generic.sh "${tag}_b$n" "$n" >> logs/fill_${tag}_b$n.out 2>&1
  echo "$(date '+%F %T') $tag b$n: fill_generic returned: $(tail -n 1 logs/fill_${tag}_b$n.out)"
  sleep 120
done
echo "$(date '+%F %T') $tag ALL_LAUNCHED"
