#!/usr/bin/env bash
# Per card: once the card has no python job, one fixed worker (py_local_worker2.sh) on scratch/c1_q_next.txt
# (§88 fix, §90, then §89's remainder). One GPT-2 job per card: a single worker per card, started only when idle.
cd "$(dirname "$0")/.."
for d in cuda:0 cuda:1 cuda:2; do
  ( until [ "$(powershell -NoProfile -c "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*--device $d *' }).Count")" = "0" ]; do sleep 60; done
    bash scratch/py_local_worker2.sh $d scratch/c1_q_next.txt ) &
  sleep 5
done
wait
