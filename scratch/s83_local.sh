#!/usr/bin/env bash
# §83 remainder, local (Theory, option (a)): primaries s2, s3 (5794e4c script) on the first card freed after §79 and
# §84 Step 1; extras afterwards. Outputs theory/results/s83/wc0k4r_s<seed>.json and _ext.json.
cd "$(dirname "$0")/.."
d=${1:-cuda:2}
# lowest priority (Audit: s2/s3 cannot change the verdict): wait until the §86 panel worker on this card has finished
until grep -q "WORKER_DONE $d scratch/l86_q.txt" logs/c1_local.log; do sleep 300; done
until [ "$(powershell -NoProfile -c "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*--device $d *' -and \$_.CommandLine -notlike '*c1_rl.py*' -and \$_.CommandLine -notlike '*c1_s85*' }).Count")" = "0" ] && \
      ! powershell -NoProfile -c "Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | Where-Object { \$_.CommandLine -like '*local_worker.sh $d *' -or \$_.CommandLine -like '*s84_step1*' }" | grep -q .; do sleep 120; done
for s in 2 3; do python -u theory/s83_restart.py --run boxes/runs_s5/wc0k4r_s$s --k 4 --kmax 16 --test_n 2000 --ext_n 0 --cal_n 20000 --sweep_n 0 --fixed 4 16 --device $d --out theory/results/s83/wc0k4r_s$s.json > logs/s83_local_primary_s$s.log 2>&1; echo "$(date +%T) s83 primary s$s rc $? ($d)" >> logs/c1_local.log; done
for s in 1 2 3; do python -u theory/s83_restart.py --run boxes/runs_s5/wc0k4r_s$s --k 4 --kmax 16 --test_n 2000 --ext_n 500 --cal_n 20000 --sweep_n 500 --fixed 4 8 16 --device $d --out theory/results/s83/wc0k4r_s${s}_ext.json > logs/s83_local_ext_s$s.log 2>&1; echo "$(date +%T) s83 extras s$s rc $? ($d)" >> logs/c1_local.log; done
