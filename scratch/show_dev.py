import json
import sys

D = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/development_dev_s4.json"))
for s in sorted(D, key=int):
    r = D[s]
    sh = r["cache"]["4"]["shares"]
    route = f"TL {sh['TL']:.2f} Gq {sh['Gq']:.2f}" if sh else "  (clean effect too small)"
    print(f"{int(s):6d}  d2-6 " + " ".join(f"{r['matched'][d]:.2f}" for d in "23456") + f" | W4 {route} | frontier {r.get('frontier_schedule')}")
