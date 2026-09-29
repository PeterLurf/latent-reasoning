"""Compare a re-computed follow-up summary with the stored confirmatory one (host-RAM integrity check)."""
import json
import sys

import numpy as np

run = sys.argv[1] if len(sys.argv) > 1 else "runs/main_s1"
A = json.load(open(f"{run}/followup/fresh/summary.json"))["results"]
B = json.load(open(f"{run}/followup/fresh_integrity/summary.json"))["results"]


def leaves(x, path=()):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, path + (k,))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, path + (i,))
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield path, float(x)


n = worst = 0
bad = []
for W in B:
    for ex in B[W]:
        a = dict(leaves(A[W][ex]))
        for p, v in leaves(B[W][ex]):
            if p not in a:
                continue
            n += 1
            d = abs(v - a[p]) if not (np.isnan(v) and np.isnan(a[p])) else 0.0
            worst = max(worst, d)
            if d > 1e-4:
                bad.append((W, ex) + p + (a[p], v))
print(f"{run}: compared {n} numbers; max |diff| = {worst:.2e}; {len(bad)} differ by > 1e-4")
from collections import Counter

grp = Counter((b[1], b[3] if b[2] == "kinds" else b[2]) for b in bad)
print("differences by (experiment, kind/field):", dict(grp))
det = [b for b in bad if not (b[2] == "kinds" and ("random" in str(b[3]) or "normrand" in str(b[3])))]
print(f"deterministic quantities differing by > 1e-4: {len(det)}")
for b in det[:15]:
    print("  ", b)
maxdet = max([abs(b[-1] - b[-2]) for b in det], default=0.0)
json.dump({"run": run, "compared": n, "deterministic_compared": n - (len(bad) - len(det)),
           "deterministic_over_1e-4": len(det), "random_control_over_1e-4": len(bad) - len(det),
           "max_abs_diff_all": worst, "max_abs_diff_deterministic_flagged": maxdet},
          open("results/integrity.json", "w"), indent=1)
