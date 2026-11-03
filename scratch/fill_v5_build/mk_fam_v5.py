"""Filler template v5 (Audit 11-03, the s269 fam() under-count): fam() keys realised lines by attempt. A VOIDed attempt's spend is logged as
'realised:b<n>:a<k> # ... <$>' and always ADDS to the family; an untagged 'realised:b<n> ... <$>' is the current attempt's landed spend and replaces that cell's admit
projection (Audit 10-29, b369: a landed cell counts once). Applied to a NEW family's filler only (never to a running one):
    python -B scratch/fill_v5_build/mk_fam_v5.py scratch/sNNN_fill.sh        (in place, asserted once; bash -n checked)
Fixture: bash scratch/fill_v5_build/fixture_fam_v5.sh"""
import subprocess
import sys

OLD = """        v = float(l.split()[-1]); m = re.match(r'realised:b(\\d+)', l); c = re.match(r'(\\d+):', l)
        if m: real[m.group(1)] = real.get(m.group(1), 0.0) + v
        elif c: proj[c.group(1)] = v
        else: s += v"""
NEW = """        v = float(l.split()[-1]); m = re.match(r'realised:b(\\d+)(:a\\d+)?', l); c = re.match(r'(\\d+):', l)
        if m and m.group(2): s += v   # v5 (Audit 11-03): a VOIDed attempt's realised spend adds to the family (it never replaces the relaunch's projection)
        elif m: real[m.group(1)] = real.get(m.group(1), 0.0) + v
        elif c: proj[c.group(1)] = v
        else: s += v"""
p = sys.argv[1]; s = open(p, encoding="utf-8").read(); assert s.count(OLD) == 1 and "fam_v5" not in s, ("v5 fam(): the v4 body not found once", p)
s = s.replace(OLD, NEW).replace("# Audit 10-29 (b369 refused twice)", "# fam_v5 + Audit 10-29 (b369 refused twice)", 1)
open(p, "w", encoding="utf-8", newline="\n").write(s); assert subprocess.run(["bash", "-n", p]).returncode == 0, p; print("fam v5 applied:", p)
