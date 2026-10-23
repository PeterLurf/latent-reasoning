#!/bin/bash
# §224 K10 (FH_v1 §6; Theory 10-22: Experiments runs it): docs/research_fixedhold_2026-10-21/fh_check_cpu.py on each new Stage-1 model as it lands
# (model.pt + eval_khorizon.json present), CPU, 2 torch threads (the script's own), Idle priority, one at a time. Outputs stay UNCOMMITTED and UNREAD
# until all five cells have landed (Theory); stdout goes to logs/s224_k10_<run>.log (not read; completion judged by the output file).
#   nohup bash scratch/s224_k10_runner.sh > /dev/null 2>&1 &
cd /d/Latent; L=${K10LOG:-logs/s224_k10.out}; F=docs/research_fixedhold_2026-10-21
RUNS=${RUNS:-"s224_fhft_s1 s224_fhft_s2 s224_rdft_s1 s224_rdft_s2 s224_ftctrl_s2"}   # Stage 2: RUNS="s224_fh_s1 s224_rd_s1 s224_fh_s2 s224_rd_s2"
for r in $RUNS; do
  until [ -s boxes/runs_fsq/$r/model.pt ] && [ -s boxes/runs_fsq/$r/eval_khorizon.json ]; do sleep 600; done
  [ -s $F/fh_check_cpu.$r.json ] && { echo "$(date '+%F %T') $r already done" >> $L; continue; }
  echo "$(date '+%F %T') $r start (CPU, Idle)" >> $L
  powershell -NoProfile -Command "\$env:OMP_NUM_THREADS='2'; \$env:MKL_NUM_THREADS='2'; \$p = Start-Process -FilePath python -ArgumentList '-B','$F/fh_check_cpu.py','boxes/runs_fsq/$r' -PassThru -NoNewWindow -RedirectStandardOutput 'logs/s224_k10_$r.log' -RedirectStandardError 'logs/s224_k10_$r.err'; \$p.PriorityClass = 'Idle'; \$p.WaitForExit(); exit \$p.ExitCode"
  rc=$?; echo "$(date '+%F %T') $r rc $rc output $( [ -s $F/fh_check_cpu.$r.json ] && echo present || echo MISSING)" >> $L
done
echo "$(date '+%F %T') K10 runner done" >> $L
