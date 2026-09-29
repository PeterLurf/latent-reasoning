"""Tabulate evaluated theory-stream pilot runs by arm (theory/runs/*/eval.json)."""
import json, os, collections
arms = collections.defaultdict(list)
for r in sorted(os.listdir("theory/runs")):
    if not r[:3] in ("pa_", "pb_", "pc_", "pd_", "pe8", "pe3") or "stage1fail" in r:
        continue
    f = f"theory/runs/{r}/eval.json"
    if not os.path.exists(f):
        continue
    e = json.load(open(f)); arm = r.rsplit("_s", 1)[0]
    arms[arm].append((r, e["gate"], e["extrap_K12"], round(e["d7_K12"], 3), round(e["far8_10_K12"], 3), e["final_stage"]))
for a in sorted(arms):
    L = arms[a]; comp = [x for x in L if x[1]]
    print("%-7s runs %2d competent %2d extrapolate %2d | far8-10 of competent: %s" % (a, len(L), len(comp), sum(x[2] for x in comp),
          " ".join("%.2f" % x[4] for x in comp)))
    for x in L:
        print("        %-12s gate %-5s extrap %-5s d7 %.3f far8-10 %.3f stage %s" % x)
