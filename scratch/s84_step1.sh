#!/usr/bin/env bash
# §84 Step 1 on cuda:0 once its current §79 job ends (one GPT-2 job per card), then give the card back to §79.
cd "$(dirname "$0")/.."
until [ "$(powershell -NoProfile -c "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*--device cuda:0 *' }).Count")" = "0" ]; do sleep 60; done
for s in 1 2 3; do python gsmlen/oracle84.py predict --seed $s --device cuda:0 > logs/s84_oracle_s$s.log 2>&1; echo "$(date +%T) s84 ORACLE seed $s rc $? (cuda:0)" >> logs/c1_local.log; done
Q=scratch/c1_q_79_gpt.txt
nohup bash scratch/py_local_worker.sh cuda:0 $Q > /dev/null 2>&1 &
echo "$(date +%T) s84 step1 predictions done; the §79 worker restarted on cuda:0" >> logs/c1_local.log
