#!/usr/bin/env bash
# §86 placement (Theory): after §79's GPT-2 queue (and §84 Step 1 / §85). The first card whose §79 worker finishes takes
# the pilot (k 4, seed 1); the panel starts on every free card only after the pilot's eval.json exists.
cd "$(dirname "$0")/.."
free_card() { for d in cuda:0 cuda:1 cuda:2; do
  [ "$(powershell -NoProfile -c "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*--device $d *' -and \$_.CommandLine -notlike '*c1_s85*' -and \$_.CommandLine -notlike '*c1_rl.py*' }).Count")" = "0" ] && \
  ! powershell -NoProfile -c "Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | Where-Object { \$_.CommandLine -like '*local_worker.sh $d *' -or \$_.CommandLine -like '*s84_step1*' }" | grep -q . && { echo $d; return; }; done; }
until [ -z "$(grep -v '^#' scratch/c1_q_79_gpt.txt | while read -r out rest; do [ -s $out/eval.json ] || [ -d c1/runs/locks/$(basename $out) ] || echo x; done)" ] && d=$(free_card) && [ -n "$d" ]; do sleep 120; done
echo "$(date +%T) l86 pilot on $d" >> logs/c1_local.log
bash scratch/py_local_worker.sh $d scratch/l86_pilot.txt
[ -s boxes/runs_s5/l86_k4_s1/eval.json ] || { echo "$(date +%T) l86 PILOT FAILED (no eval.json)" >> logs/c1_local.log; exit 1; }
for d in cuda:0 cuda:1 cuda:2; do nohup bash -c "cd $(pwd); until [ -n \"\$(powershell -NoProfile -c \"(Get-CimInstance Win32_Process -Filter \\\"Name='python.exe'\\\" | Where-Object { \\$_.CommandLine -like '*--device $d *' }).Count\" | grep -x 0)\" ]; do sleep 120; done; bash scratch/py_local_worker.sh $d scratch/l86_q.txt" > /dev/null 2>&1 & done
echo "$(date +%T) l86 panel workers armed" >> logs/c1_local.log
