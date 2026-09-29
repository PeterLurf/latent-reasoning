"""Exploratory: does the share of the pointer written locally (MLPs) vs retrieved (attention to earlier latent
positions) during the trained transitions predict extrapolation?"""
import json, numpy as np
S = json.load(open("theory/results/exp4_shares.json")); T = {o["name"]: o for o in json.load(open("theory/results/pointer_taxonomy.json"))}
def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool); p, n = x[y], x[~y]
    return float(((p[:, None] > n[None]) + 0.5 * (p[:, None] == n[None])).mean())
feat = {}
for name, o in S.items():
    K = o["K"]; R = o["rows"]
    def sh(r, gname): return r["groups"].get(gname, 0.0) / max(abs(r["amp"]), 1e-6)
    tr = [r for r in R if r["k"] <= K - 1]                  # outputs h_1..h_{K-1}: fed back in training
    f = {"local_tr": np.mean([sh(r, "local") for r in tr]), "retr_tr": np.mean([sh(r, "retr") for r in tr]),
         "carry_tr": np.mean([sh(r, "carry") for r in tr]),
         "local_K": sh(R[K - 1], "local"), "retr_K": sh(R[K - 1], "retr"),
         "retr_max_tr": max(sh(r, "retr") for r in tr), "local_min_tr": min(sh(r, "local") for r in tr),
         "n_retr_steps": sum(sh(r, "retr") > 0.5 for r in tr)}
    f["extrap"] = T[name]["extrap"]; f["typ"] = T[name]["typ"]; f["P7"] = T[name]["P7"]
    feat[name] = f
y = [f["extrap"] for f in feat.values()]
for k in ("local_tr", "retr_tr", "carry_tr", "local_K", "retr_K", "retr_max_tr", "local_min_tr", "n_retr_steps", "P7"):
    x = [f[k] for f in feat.values()]
    print("%-13s AUC %.3f   mean extrap %.2f  non %.2f" % (k, auc(x, y), np.mean([a for a, b in zip(x, y) if b]), np.mean([a for a, b in zip(x, y) if not b])))
print()
for name, f in sorted(feat.items(), key=lambda kv: (kv[1]["extrap"], kv[1]["retr_max_tr"])):
    print("%-9s %s %-7s P7 %.2f  local_tr %.2f retr_tr %.2f retr_max %.2f nretr %d | step K: local %.2f retr %.2f" % (
        name, "EXT" if f["extrap"] else "   ", f["typ"], f["P7"], f["local_tr"], f["retr_tr"], f["retr_max_tr"], f["n_retr_steps"], f["local_K"], f["retr_K"]))
json.dump(feat, open("theory/results/share_features.json", "w"), indent=1, default=float)
