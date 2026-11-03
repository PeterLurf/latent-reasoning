#!/usr/bin/env bash
# v6 item 2 fixture: scratch/queue_dryrun.py on (a) b412 attempt 5's queue (scratch/gen_q_b412.txt as committed at 3b44605a's parent: its gates / train lines
# --blob'd scratch/s259_guarded.py, which no line shipped) -> DRYRUN FAIL naming 's259_guarded.py not shipped'; (b) a queue built at HEAD (default
# scratch/gen_q_b417.txt[.hold]) -> DRYRUN OK. 2026-11-03 run: (a) FAIL x4 (s259_guarded not shipped + a stale inject_aw_eval blob), (b) OK 5 lines / 101 files.
set -u
cd "$(dirname "$0")/../.." || exit 2
G=${1:-scratch/gen_q_b417.txt}; [ -f "$G" ] || G=$G.hold
git show 3b44605a~1:scratch/gen_q_b412.txt > scratch/_tmp/q_b412_a5.txt
a=$(python -B scratch/queue_dryrun.py scratch/_tmp/q_b412_a5.txt 2>&1); ra=$?; b=$(python -B scratch/queue_dryrun.py "$G" 2>&1); rb=$?
ok=1
[ $ra = 1 ] && echo "$a" | grep -q "s259_guarded.py not shipped" || { echo "FAIL (a): $(echo "$a" | tail -n 2)"; ok=0; }
[ $rb = 0 ] || { echo "FAIL (b): $(echo "$b" | tail -n 3)"; ok=0; }
[ $ok = 1 ] && echo "FIXTURE queue_dryrun PASS" || echo "FIXTURE queue_dryrun FAIL"
