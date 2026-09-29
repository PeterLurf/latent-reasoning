import json
import re
import sys

import numpy as np

R = json.load(open(sys.argv[1]))
far = lambda d: np.mean([d[str(x)] for x in (7, 8, 9)])
near = lambda d: np.mean([d[str(x)] for x in (3, 4, 5, 6)])
print("baseline:", {k: (round(far(v), 3), round(near(v), 3)) for k, v in R["baseline"].items()})
for Kp in (12, 16):
    print(f"K'={Kp}: far(7-9) for rewind, rows = injection step j, cols = source thought index s = j-1-r")
    tab = {}
    for key, v in R["grid"].items():
        mm = re.match(rf"K{Kp}_j(\d+)_r(\d+)_(\w+)", key)
        if mm:
            j, r, mode = int(mm[1]), int(mm[2]), mm[3]
            tab[(j, j - 1 - r, mode)] = (far(v), near(v))
    srcs = sorted({s for _, s, _ in tab})
    print("   j\\s " + " ".join(f"{s:>6d}" for s in srcs))
    for j in sorted({j for j, _, _ in tab}):
        print(f"  {j:3d}  " + " ".join(f"{tab[(j, s, 'rewind')][0]:6.3f}" if (j, s, "rewind") in tab else "     ." for s in srcs))
    print("  foreign (r=2,4):", {f"j{j}_s{s}": round(v[0], 3) for (j, s, mo), v in tab.items() if mo == "foreign"})
    z = {k: round(far(v), 3) for k, v in R["grid"].items() if k.startswith(f"K{Kp}_") and k.endswith("zero")}
    print("  zero:", z)
    print("  near(3-6) cost at best rewinds:", {f"j{j}_s{s}": round(v[1], 3) for (j, s, mo), v in tab.items() if mo == "rewind" and v[0] > 0.6})
