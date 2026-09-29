#!/usr/bin/env bash
# Every 60 min: fetch finished runs from every live theory host (checksum-verified; skips hosts already destroyed), then
# run the frozen measurement-only analyses on newly fetched runs so scoring is quick when the pilots end:
#   le_pointer (pointer trajectories), transfer_pairs (restoring index; Coconut-type runs), set_traj (set-memory runs),
#   score_pilotDE measurement cache. Low priority, <= 3 threads. Stops when theory/hosts_done.txt lists every host.
#   bash theory/incremental.sh
cd "$(dirname "$0")/.."
export THEORY_THREADS=3 OMP_NUM_THREADS=3
while :; do
  while read -r h p tag id; do
    [ -z "$h" ] && continue; case "$h" in \#*) continue;; esac
    grep -q "^$tag " theory/hosts_done.txt 2>/dev/null && continue
    echo "$(date '+%H:%M') fetch $tag: $(timeout 1800 bash theory/vfetch_arch.sh $h $p < /dev/null 2>&1 | tail -1)"
  done < theory/hosts_theory.txt
  runs() { ls -d theory/runs/$1 2>/dev/null | while read -r d; do [ -f "$d/eval.json" ] && echo "$d"; done; }
  nice -n 19 python theory/le_pointer.py $(runs 'pd_*') $(runs 'pe8_*') $(runs 'pe3_*') $(runs 'pf_*') $(runs 'pi_*') \
      $(runs 'pj_*') $(runs 'ph_*') $(runs 'pc_*') < /dev/null
  nice -n 19 python theory/transfer_pairs.py $(runs 'pd_*') $(runs 'pf_j*') $(runs 'pi_*') $(runs 'pj_*') $(runs 'ph_*') < /dev/null
  nice -n 19 python theory/set_traj.py $(runs 'pg_*') < /dev/null
  nice -n 19 python theory/score_pilotDE.py < /dev/null > /dev/null
  n_live=$(grep -v '^#' theory/hosts_theory.txt | grep -c . ); n_done=$(grep -c . theory/hosts_done.txt 2>/dev/null || echo 0)
  [ "$n_done" -ge "$n_live" ] && { echo "all hosts done $(date '+%H:%M')"; exit 0; }
  sleep 3600
done
