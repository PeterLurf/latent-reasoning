#!/usr/bin/env bash
# Retention sweep 2026-10-05 (Audit+Theory OK): A) delete .pt in *.void* dirs; B) move c1/runs, boxes/runs .pt to
# E:\LatentArchive (same relative path): copy -> sha256 source & dest -> delete D: copy only if equal. Logs never touched.
cd "$(dirname "$0")/.."
LOG=logs/retention_sweep_20261005.log; MAN=/e/LatentArchive/MANIFEST.txt; mkdir -p /e/LatentArchive
grep '^DELETE' scratch/retention_candidates.txt | awk '{print $4}' | while read -r f; do
  [ -f "$f" ] || continue; b=$(stat -c %s "$f"); rm -f "$f" && echo "$(date +%T) DELETED $b $f" >> $LOG; done
grep '^MOVE' scratch/retention_candidates.txt | awk '{print $4}' | while read -r f; do
  [ -f "$f" ] || { echo "$(date +%T) MISSING $f" >> $LOG; continue; }
  d="/e/LatentArchive/$f"; mkdir -p "$(dirname "$d")"; b=$(stat -c %s "$f")
  h1=$(sha256sum "$f" | cut -d' ' -f1); cp "$f" "$d" || { echo "$(date +%T) COPYFAIL $f" >> $LOG; continue; }
  h2=$(sha256sum "$d" | cut -d' ' -f1)
  if [ "$h1" = "$h2" ] && [ "$(stat -c %s "$d")" = "$b" ]; then rm -f "$f"; echo "$f $b $h1" >> $MAN; echo "$(date +%T) MOVED $b $f" >> $LOG
  else rm -f "$d"; echo "$(date +%T) VERIFYFAIL $f (kept on D:)" >> $LOG; fi
done
echo "$(date +%T) SWEEP DONE moved=$(grep -c MOVED $LOG) deleted=$(grep -c DELETED $LOG) fail=$(grep -cE 'FAIL|MISSING' $LOG)" >> $LOG
