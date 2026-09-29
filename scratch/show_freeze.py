import json
import sys

R = {}
for f in sys.argv[1:]:
    R.update(json.load(open(f)))
for r, conds in R.items():
    print(r)
    for k, v in conds.items():
        if k.endswith("_K20"):
            print(f"  {k:14s} d3-12 " + " ".join(f"{v['acc'][str(d)]:.2f}" for d in range(3, 13)))
