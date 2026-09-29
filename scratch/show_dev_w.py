import json
D = json.load(open("results/development_dev_s4.json"))
for s in ("1000", "2000", "3000", "8000", "17500", "20000", "30000"):
    r = D[s]["cache"]
    print(s, " ".join(f"W{W}: TL {r[W]['shares']['TL']:.2f} Gq {r[W]['shares']['Gq']:.2f}" if r[W]["shares"] else f"W{W}: -" for W in ("2", "4", "8")))
