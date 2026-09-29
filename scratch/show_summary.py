import json

S = json.load(open("results/summary.json"))
for s, d in S["seeds"].items():
    e = d["effects"]
    print(s, "k*", d["k_star"])
    for c in ["full|remove|hub", "pattern|remove|hub", "probe|remove|hub", "full|insert|hub", "pattern|insert|hub"]:
        print("  ", c.ljust(20), " ".join(
            f"W{W}:{e[c][str(W)]['targeted'][0]:+.3f}/{e[c][str(W)]['collateral_attached'][0]:.3f}"
            for W in range(1, 9) if str(W) in e[c]))
    print("   signed rem", " ".join(f"{e['full|remove|hub'][str(W)]['collateral_signed'][0]:+.3f}" for W in range(2, 9)))
    print("   signed ins", " ".join(f"{e['full|insert|hub'][str(W)]['collateral_signed'][0]:+.3f}" for W in range(2, 9)))
    print("   branch AUC", " ".join(f"{v:.3f}" for v in d["decode"]["branch_auc_by_width_at_kstar"].values()))
    print("   same_status", " ".join(f"{v[0]:.3f}" for v in d["controls"]["same_status"].values()),
          " identity max", max(v[0] for v in d["controls"]["identity"].values()))
    print("   upfront full tgt", " ".join(f"{v['targeted'][0]:.3f}" for v in d["upfront"]["full"].values()),
          " col", " ".join(f"{v['collateral'][0]:.3f}" if v["collateral"] else "-" for v in d["upfront"]["full"].values()))
    print("   upfront pattern tgt", " ".join(f"{v['targeted'][0]:.3f}" for v in d["upfront"]["pattern"].values()))
    dk = d["depth"]["full|remove|hub"]
    for W in ("2", "4", "8"):
        print(f"   depth W{W}", " ".join(f"d{int(j) + 1}:{v[0]:+.3f}" for j, v in dk[W].items()))
    for key, v in d["downstream"].items():
        print("   trace", key, " ".join(f"k{kp}:{m['recipient_margin'][0]:+.2f}->{m['patched_margin'][0]:+.2f}" for kp, m in v.items()))
for name, c in S["competence"].items():
    b = c[max(c, key=int)]
    print(name, "K'max delayed", round(b["delayed"]["bal_acc"][0], 3), "upfront", round(b["upfront"]["bal_acc"][0], 3),
          {k: round(v, 2) for k, v in b["delayed"]["by_depth"].items()})
