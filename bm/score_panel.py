"""Score the THEORY §60 / §60b Phase A panels (Gate A per seed, PA-dir) with the identity-map reference line per split.
    python bm/score_panel.py a     # §60 panel: bm/runs/{oc,no}_s{1..4}, pool bm/data_pool
    python bm/score_panel.py b     # §60b panel: bm/runs/{oc,no}_b_s{1..4}, pool bm/data_pool_b
Gate A (BM-oc, every seed): A1 probe >= 0.95 on held-out instances; A2 >= 0.95 on held-out motifs and held-out orders;
A3 entry accuracy >= 0.95 and >= 95 % of tables satisfying the laws. PA-dir: BM-oc beats BM-no on held-out-order probe
AND has lower residual associativity error (assoc_abs), in >= 3/4 seed pairs.
Identity baseline: accuracy of f(a) = a (predict 1 iff j in a) on each split's labels.
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bm.data import PATTERNS  # noqa: E402

v = sys.argv[1] if len(sys.argv) > 1 else "a"
pool = "bm/data_pool" if v == "a" else "bm/data_pool_b"
suf = "" if v == "a" else "_b"
SPL = ["ev_instances", "ev_motifs", "ev_orders", "ev_size"]
ident = {}
for sp in SPL:
    t = pickle.load(open(f"{pool}/{sp}.pkl", "rb"))
    ident[sp] = float(np.mean([float(PATTERNS[x["row"]][x["col"]] == 1) == x["label"] for x in t]))
res, gate, pa = {}, [], []
for s in (1, 2, 3, 4):
    for arm in ("oc", "no"):
        f = f"bm/runs/{arm}{suf}_s{s}/eval.json"
        res[f"{arm}_s{s}"] = json.load(open(f)) if os.path.exists(f) else None
    oc, no = res[f"oc_s{s}"], res[f"no_s{s}"]
    if oc:
        a1 = oc["ev_instances"]["probe"] >= 0.95
        a2 = oc["ev_motifs"]["probe"] >= 0.95 and oc["ev_orders"]["probe"] >= 0.95
        a3 = oc["ev_instances"]["entry"] >= 0.95 and oc["ev_instances"]["laws"] >= 0.95
        gate.append({"seed": s, "A1": a1, "A2": a2, "A3": a3, "all": a1 and a2 and a3})
    if oc and no:
        pa.append(oc["ev_orders"]["probe"] > no["ev_orders"]["probe"] and oc["ev_orders"]["assoc_abs"] < no["ev_orders"]["assoc_abs"])
table = {k: ({sp: {m: round(r[sp][m], 4) for m in ("probe", "entry", "laws")} for sp in SPL} | {
    "assoc_abs": round(r["ev_orders"]["assoc_abs"], 4), "assoc_disagree": round(r["ev_orders"]["assoc_disagree"], 4),
    "binarised_instances": round(r["ev_instances"]["probe_binarised"], 4), "age_audit": r.get("age_audit"),
    "margin_by_depth": r.get("margin_by_depth")} if r else None) for k, r in res.items()}
out = {"panel": v, "identity_baseline": ident, "per_run": table, "gate_A_per_seed": gate,
       "gate_A_passed": bool(gate) and len(gate) == 4 and all(g["all"] for g in gate),
       "PA_dir_pairs": sum(pa), "PA_dir": sum(pa) >= 3}
json.dump(out, open(f"bm/runs/score_panel_{v}.json", "w", newline="\n"), indent=1)
print("identity baseline:", {k: round(x, 3) for k, x in ident.items()})
for k, r in table.items():
    if r:
        print(k, " ".join(f"{sp[3:]} {r[sp]['probe']:.3f}" for sp in SPL), f"| entry {r['ev_instances']['entry']:.3f} laws {r['ev_instances']['laws']:.3f} assoc {r['assoc_abs']:.3f}")
print("Gate A:", gate, "-> passed" if out["gate_A_passed"] else "-> NOT passed")
print("PA-dir pairs", sum(pa), "/", len(pa))
