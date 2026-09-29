import json
import sys

R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/budget_sweep_wd.json"))
for run, o in R.items():
    rows = {(r["variant"], r["K'"]): r for r in o["rows"]}
    print(run)
    for W in ("2", "4", "8"):
        line = []
        for key in (("train", 4), ("train", 6), ("repeat", 8), ("repeat", 10)):
            r = rows[key]["by_width_depth"][W]
            line.append(f"K'={key[1]:2d}: " + " ".join(f"{r[d]:.2f}" for d in ("3", "4", "5", "6")))
        print(f"  W{W} depth 3..6 | " + " | ".join(line))
