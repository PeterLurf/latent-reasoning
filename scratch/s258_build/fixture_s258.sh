#!/usr/bin/env bash
# s258 template fixture (set -u): the WALLW-count refusal, the ALLC family-cell builder (ledger with voided / realised lines + unadmitted QS cells), the guard-health
# check, and the sNNNfam pid finder (print only, never kills). Fixture files under scratch/_tmp/s258_fx (created here; the caller removes the dir).
set -u
cd "$(dirname "$0")/../.." || exit 2
F=scratch/_tmp/s258_fx; mkdir -p "$F"
printf '%s\n' '# ledger' '391:dis_s1 1.2.3.4:5 111 $0.17/h sps 1.1 1.07' '# VOID attempt1: 392:dis_s2 x 222 $0.2/h sps 1 0.9' '392:dis_s2 6.7.8.9:10 333 $0.16/h sps 1.2 1.08' 'realised:b391 # 111 (x) 1.15' > "$F/adm.log"
printf '%s\n' '# c' 'a' 'b' 'c' 'd' > "$F/q.txt"; printf 'x start: cap\nCHECK ok\n' > "$F/good.out"; printf 'Traceback (most recent call last):\nAssertionError\n' > "$F/bad.out"
ADMLOG="$F/adm.log"; STEPS=16000; FAMC=" 393:dis_s3:1.2:16000 392:dis_s2:1.2:16000"
for WALLW in 0.1,0.1,0.4,4.0 0.1,0.1,4.0; do
  nlq=$(grep -vc '^#' "$F/q.txt"); nw=$(echo "$WALLW" | tr ',' '\n' | grep -c .); echo "WALLW $WALLW: lines $nlq weights $nw -> $([ "$nlq" = "$nw" ] && echo deploy || echo REFUSE)"
done
ALLC=$(awk -v S="$STEPS" '!/^#/ && /^[0-9]+:/ {split($1, a, ":"); print a[1] ":" a[2] ":" $NF ":" S}' "$ADMLOG" | tr '\n' ' ')
for c_ in $FAMC; do grep -q "^${c_%%:*}:" "$ADMLOG" || ALLC="$ALLC $c_"; done; echo "ALLC = [$ALLC]"
for gf in "$F/good.out" "$F/bad.out"; do { grep -q "start:" "$gf" && ! grep -q "Traceback" "$gf"; } && echo "$gf healthy" || echo "$gf ALERT"; done
echo "s257fam pids (print only): $(ps -ef | awk '$7 == "scratch/cost_guard.py" && $8 == "s257fam" {print $2}')"
