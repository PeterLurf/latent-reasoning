import json
import sys

runs = sys.argv[1:]
logs = {r: {json.loads(l)["step"]: json.loads(l) for l in open(f"runs/{r}/log.jsonl") if '"loss_upfront"' in l} for r in runs}
print("step  " + "  ".join(f"{r[:22]:>36s}" for r in runs))
for s in (200, 400, 600, 800, 1000, 1400, 2000, 3000, 4000, 6000, 8000):
    cells = []
    for r in runs:
        x = logs[r].get(s)
        cells.append(f"d {x['loss_delayed']:.3f} u {x['loss_upfront']:.3f} c {x['loss_children']:.2f} b{x['max_bdepth']}" if x else " " * 36)
    print(f"{s:5d} " + "  ".join(f"{c:>36s}" for c in cells))
