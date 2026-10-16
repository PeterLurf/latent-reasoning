#!/usr/bin/env bash
# §213 note 213-17 (Audit-approved, $0, descriptive C5 HALT): after scratch/s213_post_landing.sh prints its final "§213 post-landing evals done" line (so the
# 3070 never has a co-tenant), boxes.vbank_halt per cell on the same card (DEV, default cuda:1), fp32, eval_bs 32 -> <cell>/eval_vbhalt.json.
#   nohup bash scratch/s213_post_halt.sh >> logs/s213_post_landing.out 2>&1 &
cd "$(dirname "$0")/.."
DEV=${DEV:-cuda:1}
until grep -q "§213 post-landing evals done" logs/s213_post_landing.out 2>/dev/null; do sleep 300; done
for r in s213_d3t_s1 s213_d3f_s1; do
  python -m boxes.vbank_halt --run boxes/runs_fsq/$r --device $DEV --eval_bs 32 > logs/s213_vbhalt_$r.log 2>&1
  echo "$(date '+%F %T') $r vbhalt rc $? $(tail -n 1 logs/s213_vbhalt_$r.log | cut -c1-160)"
done
echo "$(date '+%F %T') §213 vbhalt done"
