#!/bin/bash
# REVIEW_HD: Stage-0 check B on the FT cells (CPU, 2 threads, idle priority via psutil in the script)
cd /d/Latent
for r in s224_rdft_s1 s224_rdft_s2 s224_ftctrl_s2 s224_fhft_s2; do
  python -B docs/research_holddepth_2026-10-23/review_scripts/rv_check_cpu_ft.py boxes/runs_fsq/$r > docs/research_holddepth_2026-10-23/review_scripts/rv_$r.log 2>&1
  echo "$r rc $?" >> docs/research_holddepth_2026-10-23/review_scripts/run_ft.done
done
echo ALLDONE >> docs/research_holddepth_2026-10-23/review_scripts/run_ft.done
