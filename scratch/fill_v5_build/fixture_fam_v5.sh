#!/usr/bin/env bash
# fam v5 fixture (Audit 11-03): the fam() of a v5-patched filler copy on fake ledgers. Cases:
#   A  VOIDed attempt 1 realised (realised:b900:a1 0.215) + attempt 2 admitted (900: ... 0.36) + BAD 0.121       -> 0.696 (the sum; v4 gave 0.336)
#   B  a landed cell: admitted 901 0.30 + its own realised:b901 0.28                                               -> 0.28 (counted once, v4 behaviour)
#   C  A + B together                                                                                                -> 0.976
set -u
cd "$(dirname "$0")/../.." || exit 2
T=scratch/_tmp/famv5; rm -rf "$T"; mkdir -p "$T"
cp scratch/s267_fill.sh "$T/fill.sh"; python -B scratch/fill_v5_build/mk_fam_v5.py "$T/fill.sh" > /dev/null || { echo "FIXTURE: patch failed"; exit 2; }
body=$(awk '/^fam\(\) \{/,/^print\(round\(s, 4\)\)"; \}$/' "$T/fill.sh")
run() { ADMLOG="$1" BADLOG="$2" bash -c "$body"'
fam'; }
printf '%s\n' "# VOID attempt1 900:x a b c 0.29" "realised:b900:a1 # attempt1 0.215" "900:pilot a b c 0.36" > "$T/adm_A"; printf '%s\n' "b900 bad 0.121" > "$T/bad_A"
printf '%s\n' "901:landed a b c 0.30" "realised:b901 # landed 0.28" > "$T/adm_B"; : > "$T/bad_B"
cat "$T/adm_A" "$T/adm_B" > "$T/adm_C"
ok=1
for c in "A 0.696" "B 0.28" "C 0.976"; do set -- $c; bd="$T/bad_A"; [ "$1" = B ] && bd="$T/bad_B"
  got=$(sed "s#\$ADMLOG#$T/adm_$1#; s#\$BADLOG#$bd#" <<<"$body" | bash -c "$(cat); fam")
  [ "$got" = "$2" ] && echo "case $1 OK ($got)" || { echo "case $1 FAIL: got $got want $2"; ok=0; }; done
[ $ok = 1 ] && echo "FIXTURE fam v5 PASS" || echo "FIXTURE fam v5 FAIL"
