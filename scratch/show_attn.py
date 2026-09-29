import json

X = json.load(open("results/extra.json"))
for run, r in X.items():
    print(run, "k*", r["k_star"], {k: round(v, 4) for k, v in r["energy_W4"].items()})
    for layer in ("layer1", "layer2"):
        row = []
        for W in ("1", "2", "4", "8"):
            ex = sum(r["attention"][layer][L][W]["excess"] for L in ("LAT1", "LAT2", "LAT3")) / 3
            row.append(f"W{W}:{ex:.3f}")
        print("  ", layer, " ".join(row))
