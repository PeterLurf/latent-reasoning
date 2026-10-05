#!/usr/bin/env bash
# COPY-ONLY archive (2026-10-05): copy B-list .pt to E:\LatentArchive (same relative path), sha256-verify, write manifest.
# Deletes nothing on D: (the delete step was blocked pending the user's OK; see scratch/retention_candidates.txt).
cd "$(dirname "$0")/.."
LOG=logs/retention_archive_20261005.log; MAN=/e/LatentArchive/MANIFEST.txt; mkdir -p /e/LatentArchive
grep '^MOVE' scratch/retention_candidates.txt | awk '{print $4}' | while read -r f; do
  [ -f "$f" ] || { echo "$(date +%T) MISSING $f" >> $LOG; continue; }
  d="/e/LatentArchive/$f"; mkdir -p "$(dirname "$d")"; b=$(stat -c %s "$f")
  h1=$(sha256sum "$f" | cut -d' ' -f1); cp -n "$f" "$d" || { echo "$(date +%T) COPYFAIL $f" >> $LOG; continue; }
  h2=$(sha256sum "$d" | cut -d' ' -f1)
  if [ "$h1" = "$h2" ]; then echo "$f $b $h1" >> $MAN; echo "$(date +%T) VERIFIED $b $f" >> $LOG; else echo "$(date +%T) VERIFYFAIL $f" >> $LOG; fi
done
echo "$(date +%T) ARCHIVE COPY DONE verified=$(grep -c VERIFIED $LOG) fail=$(grep -cE 'FAIL|MISSING' $LOG)" >> $LOG
