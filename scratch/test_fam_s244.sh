#!/usr/bin/env bash
# Fixture for the filler's fam() (Audit 10-29): a landed cell (projection + realised) counts once (its realised $), a running cell counts its projection,
# other ledger lines (gatehold:) and BAD lines count as written, '#' lines never. Extracts fam() from the filler under test and runs it on temp ledgers (under scratch/_tmp: Windows python cannot open MSYS /tmp paths).
#   bash scratch/test_fam_s244.sh [scratch/s244_fill.sh]
F=${1:-scratch/s244_fill.sh}; T=scratch/_tmp/test_fam_$$; mkdir -p $T; ADMLOG=$T/adm.log; BADLOG=$T/bad.log
eval "$(sed -n '/^fam() {/,/^print(round(s, 4))"; }$/p' "$F")"
type fam > /dev/null 2>&1 || { echo "FAIL: fam() not found in $F"; exit 1; }
printf '%s\n' "# ledger" "366:ao_s1 h:p 1 \$0.2/h sps 0.5 0.54" "realised:b366 # landed 0.48" "367:ao_s2 h:p 2 \$0.2/h sps 0.5 0.64" "gatehold:x 0.10" \
  "# superseded by realised:b365: 365:old h:p 3 \$0.2/h sps 0.5 9.99" > $ADMLOG
printf '%s\n' "# bad" "2026 b367 offer 1 h 10min \$0.2/h 0.05" > $BADLOG
got=$(fam); want=1.27   # 0.48 (b366 realised, not + 0.54) + 0.64 (b367 running) + 0.10 (gatehold) + 0.05 (BAD) = 1.27
rm -rf "$T"
python -c "import sys; sys.exit(0 if abs($got - $want) < 1e-9 else 1)" && { echo "PASS fam() = $got (want $want)"; exit 0; } || { echo "FAIL fam() = $got (want $want)"; exit 1; }
