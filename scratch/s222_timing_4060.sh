#!/bin/bash
# §222 cert smoke + timing at the real group shapes on the 4060 (Audit 09:33 asked for a runtime estimate): after m_rule.json exists and the feas2_s1
# python (pid 9100) has left the 4060, run the cert mode on the untrained model with the registered group shapes (7 probes Kmax 1, then 6 probes Kmax 2,
# pmicro 16) for 50 steps each (held-out 64 items every 10 steps; probe0 read at per_D 3) into scratch/_tmp/s222_timing (never gate_s222).
# Queue h re-checks the 4060 60 s after it frees, so it waits for this (~10-15 min) and then proceeds.
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s222_chain.out; O=scratch/_tmp/s222_timing
until [ -s scratch/gate_s222/m_rule.json ] && ! powershell -NoProfile -c "if (Get-Process -Id 9100 -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 10; done
mkdir -p $O && cp scratch/gate_s222/m_rule.json $O/
echo "$(date '+%F %T') cert timing smoke start (4060)" >> $L
python -B scratch/s222_cert.py --mode cert --model untrained --device cuda:0 --pool_n 3000 --hold_n 64 --steps 50 --eval_every 10 --per_D 3 --out $O > logs/s222_cert_timing.log 2>&1; rc=$?
echo "$(date '+%F %T') cert timing smoke rc $rc $(ls $O/cert_untrained.json 2>/dev/null)" >> $L
