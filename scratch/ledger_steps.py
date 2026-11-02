"""s260 template v3 (Audit 11-01): the family guard's cell list from an admitted ledger -> "n:run:proj:steps ..." on stdout. Each admitted (not '#') line's steps:
its 'steps S' field (v3 admit lines), else the cell's OWN run line in scratch/gen_q_b<n>.txt(.norequeue) (out dir == boxes/runs_fsq/<run>): --steps, or --stop_at if smaller.
Unresolvable -> ALERT on stderr, exit 3 (the caller HOLDs; never a guess).
    python scratch/ledger_steps.py <ledger>"""
import os
import re
import sys


def queue_steps(n, run):
    for f in ("scratch/gen_q_b%s.txt" % n, "scratch/gen_q_b%s.txt.norequeue" % n):
        if not os.path.exists(f):
            continue
        for ln in open(f, encoding="utf-8"):
            t = ln.split()
            if not t or t[0].startswith("#") or t[0] != "boxes/runs_fsq/" + run:
                continue
            st = [int(x) for x in re.findall(r"--steps (\d+)", ln)]; sa = [int(x) for x in re.findall(r"--stop_at (\d+)", ln)]
            if st:
                return min([st[-1]] + [x for x in sa if x > 0])
    return None


out = []; bad = []
for ln in open(sys.argv[1], encoding="utf-8"):
    if ln.startswith("#") or not re.match(r"^\d+:", ln):
        continue
    t = ln.split(); n, run = t[0].split(":", 1); st = int(t[t.index("steps") + 1]) if "steps" in t[:-1] else queue_steps(n, run)
    if st is None:
        bad.append(t[0]); continue
    out.append("%s:%s:%s:%d" % (n, run, t[-1], st))
if bad:
    print("ALERT ledger_steps: no steps for %s (no 'steps' field, no run line with --steps in its queue)" % " ".join(bad), file=sys.stderr); sys.exit(3)
print(" ".join(out))
