"""Audit 11-03 (b412 attempt 3: gate_s268's os.path.isjunction, Python 3.12+, crashed the host gates; hosts run the pytorch image's Python 3.11, the box 3.14, so a
local smoke cannot catch it): every .py a queue would run on the host - boxes/*.py (top level), each scratch.X / boxes.X queue module and each .py path token in the
queue, and every scratch/*.py those import (fixed point) - is (1) parsed with ast feature_version (3, 11) (3.12+ grammar: PEP 695 type params / `type X =`, ...;
best effort: CPython's parser does not reject every newer construct) and (2) scanned for known 3.12+ stdlib names. Exit 1 + the hits, else 0.
    python scratch/check_py311.py scratch/gen_q_b<n>.txt"""
import ast
import glob
import os
import re
import sys

NEW = {("os.path", "isjunction"), ("os.path", "splitroot"), ("itertools", "batched"), ("typing", "override"), ("typing", "TypeAliasType"), ("math", "sumprod"),
       ("pathlib.Path", "walk"), ("datetime", "UTC_NOT"), ("tomllib", "loads_NOT"), ("os", "listdrives"), ("os", "listvolumes"), ("os", "listmounts"),
       ("shutil", "copytree_dirs_exist_ok_NOT"), ("calendar", "Month"), ("calendar", "Day"), ("sys", "monitoring"), ("sys", "activate_stack_trampoline")}
ATTR = {(a, b) for a, b in NEW if not b.endswith("_NOT")}


def dotted(n):
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        d = dotted(n.value); return d + "." + n.attr if d else None
    return None


def scan(p):
    src = open(p, encoding="utf-8").read(); hits = []
    try:
        tree = ast.parse(src, filename=p, feature_version=(3, 11))
    except SyntaxError as e:
        return ["%s:%s 3.12+ grammar (feature_version 3.11): %s" % (p, e.lineno, e.msg)]
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            d = dotted(n)
            for a, b in ATTR:
                if d and (d == a + "." + b or d.endswith("." + a.split(".")[-1] + "." + b)):
                    hits.append("%s:%d %s (Python 3.12+)" % (p, n.lineno, d))
        if isinstance(n, ast.ImportFrom) and n.module:
            for al in n.names:
                if (n.module, al.name) in ATTR:
                    hits.append("%s:%d from %s import %s (Python 3.12+)" % (p, n.lineno, n.module, al.name))
        if type(n).__name__ in ("TypeAlias", "TypeVar", "ParamSpec", "TypeVarTuple"):
            hits.append("%s:%d PEP 695 %s (Python 3.12+)" % (p, n.lineno, type(n).__name__))
    return hits


def files_of(q):
    toks = [t for ln in open(q, encoding="utf-8") if ln.strip() and not ln.startswith("#") for t in ln.split()]
    fs = set(glob.glob("boxes/*.py"))
    for t in toks:
        if "=" in t:                                                        # a '<path>=<sha>' blob / init token does not ship (auto_ship takes plain tokens only)
            continue
        if re.match(r"^(scratch|boxes)\.[A-Za-z0-9_]+$", t):
            fs.add(t.replace(".", "/") + ".py")
        elif t.endswith(".py") and os.path.exists(t):
            fs.add(t)
    for _ in range(5):                                                      # scratch modules imported by the shipped scratch files, to a fixed point
        new = set()
        for f in [f for f in fs if f.startswith("scratch/") and os.path.exists(f)]:
            for m in re.findall(r"^\s*(?:import|from)\s+(?:scratch\.)?([A-Za-z0-9_]+)", open(f, encoding="utf-8").read(), re.M):
                if os.path.exists("scratch/%s.py" % m):
                    new.add("scratch/%s.py" % m)
        if new <= fs:
            break
        fs |= new
    return sorted(f for f in fs if os.path.exists(f))


if __name__ == "__main__":
    fs = files_of(sys.argv[1]); hits = [h for f in fs for h in scan(f)]
    if hits:
        print("PY311 FAIL:", "; ".join(hits)); sys.exit(1)
    print("py311 OK (%d files)" % len(fs))
