#!/usr/bin/env bash
# Template v6 item 4 (Audit 11-03 LAUNCH_POSTMORTEM §3.4, void residue): void attempt <k> of cell b<n> in filler family <fam>, from the D:/Latent root:
#   bash scratch/void_cell.sh <fam e.g. s271> <n> <k>
# (1) kills, by an exact command-line match (never this script, never another cell), the cell's post_admit (scratch/<fam>_post_admit.sh <n>), cell guard
#     (scratch/cost_guard.py <fam>_b<n>), wall guard (scratch/*wall_guard*.py <fam>_wall_b<n>), ckpt mirror (scratch/ckpt_mirror.py <n>:...) and drain watcher
#     (scratch/auto_drain_gen*.sh <host> <port> gen_b<n> ...); then checks none is left;
# (2) rotates their logs to *.attempt<k>; (3) comments the cell's admitted line ('# VOID attempt<k> ...') and retags its untagged realised line(s) realised:b<n>:a<k>
#     in logs/<fam>_admitted.log; (4) copies scratch/gen_q_b<n>.txt to scratch/gen_q_b<n>_attempt<k>.txt. The instance itself is NOT destroyed here (vdestroy.sh by id).
# Exit 0 = voided cleanly; 1 = a process survived the kill.
set -u
FAM=$1; N=$2; K=$3
[ -n "$FAM" ] && [ -n "$N" ] && [ -n "$K" ] || { echo "usage: void_cell.sh <fam> <n> <k>"; exit 2; }
pat="( scratch/${FAM}_post_admit\.sh ${N}( |$))|( scratch/cost_guard\.py ${FAM}_b${N}( |$))|( scratch/[a-z0-9_]*wall_guard[a-z0-9_]*\.py ${FAM}_wall_b${N}( |$))|( scratch/ckpt_mirror\.py ${N}:)|( scratch/auto_drain_gen[a-z0-9_]*\.sh [^ ]+ [^ ]+ gen_b${N}( |$))"
pids() { ps -ef | grep -E -- "$pat" | grep -v -E 'void_cell\.sh|grep' | awk '{print $2}' | grep -v -x -e "$$" -e "$PPID"; }
for p_ in $(pids); do echo "kill $p_: $(ps -ef | awk -v p="$p_" '$2 == p' | cut -c1-160)"; kill "$p_" 2>/dev/null; done
sleep 2; left=$(pids)
for p_ in $left; do kill -9 "$p_" 2>/dev/null; done; sleep 1; left=$(pids)
for f in "logs/${FAM}_post_b${N}.out" "logs/${FAM}_guard_b${N}.out" "logs/${FAM}_wall_b${N}.out" "logs/ckpt_mirror_b${N}.out" "logs/auto_drain_gen_b${N}.out"; do
  [ -e "$f" ] && mv "$f" "$f.attempt$K" && echo "rotated $f"
done
L="logs/${FAM}_admitted.log"
if [ -f "$L" ]; then
  python - "$L" "$N" "$K" <<'EOF'
import re, sys
p, n, k = sys.argv[1], sys.argv[2], sys.argv[3]; out = []; c = r = 0
for l in open(p, encoding="utf-8").read().split("\n"):
    if re.match(r"%s:[^ ]+ " % n, l):
        l = "# VOID attempt%s %s" % (k, l); c += 1
    elif re.match(r"realised:b%s " % n, l):
        l = l.replace("realised:b%s " % n, "realised:b%s:a%s " % (n, k), 1); r += 1
    out.append(l)
open(p, "w", encoding="utf-8", newline="\n").write("\n".join(out)); print("ledger: %d admitted line(s) voided, %d realised line(s) retagged :a%s" % (c, r, k))
EOF
fi
[ -f "scratch/gen_q_b${N}.txt" ] && cp "scratch/gen_q_b${N}.txt" "scratch/gen_q_b${N}_attempt${K}.txt" && echo "queue kept as scratch/gen_q_b${N}_attempt${K}.txt"
if [ -n "$left" ]; then echo "VOID INCOMPLETE: still running $left"; exit 1; fi
echo "VOIDED ${FAM} b${N} attempt ${K}"
