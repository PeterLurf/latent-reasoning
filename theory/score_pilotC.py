"""Score THEORY §11 Pilot C (slack dose-response) by its pre-registered rules, plus §25 LE1 inputs for K5/K7.

PC1: extrapolating share among competent runs, K5 (pc_k5_*) vs K6 (Pilot A coconut arm pa_c_s700-707) vs K7 (pc_k7_*).
     Supported if share(K5) > share(K6) > share(K7) and share(K5) - share(K7) >= 0.30; not supported if
     share(K5) <= share(K7); arms with < 3 competent runs are unresolved ("K too tight to learn" for K5).
PC2: per competent run, mean retrieval share of the trained transitions (exp4 split; sweep head as in exp11) and the
     number of thoughts before the pointer first reaches depth 4 (ptr_traj of analyse_pilotA, long-chain probe);
     prediction: both increase with K (reported as arm means).
§25 LE1 input: p[K-1] (pointer depth of the last fed-back thought) per competent run.
    python theory/score_pilotC.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "3")))
from analyse_pilotA import ptr_traj  # noqa: E402
from eval_arch import load_arch  # noqa: E402
from exp4_shares import shares  # noqa: E402
from exp7_dev import sweep_head  # noqa: E402
from rollout import probe  # noqa: E402

OUT = "theory/results/score_pilotC.json"


def arm_of(r):
    if r.startswith("pc_k5_"):
        return "K5"
    if r.startswith("pc_k7_"):
        return "K7"
    if r.startswith("pa_c_s") and 700 <= int(r.split("_s")[-1]) <= 707:
        return "K6"
    return None


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    runs = sorted(r for r in os.listdir("theory/runs") if arm_of(r) and os.path.exists(f"theory/runs/{r}/eval.json"))
    g, br, E = probe(512)
    gs, brs, Es = probe(256)
    for r in runs:
        ev = json.load(open(f"theory/runs/{r}/eval.json"))
        row = {"arm": arm_of(r), "gate": ev["gate"], "extrap": ev["extrap_K12"], "d7": ev["d7_K12"], "far": ev["far8_10_K12"],
               "stage": ev["final_stage"]}
        if ev["gate"] and "ptr" not in res["runs"].get(r, {}):
            m, c = load_arch(f"theory/runs/{r}", "cpu")
            K = c["K"]
            with torch.no_grad():
                p, a, _ = ptr_traj(m, K, g, br, E, {})
                head, _ = sweep_head(m, K, gs, brs, Es)
                S = shares(m, K, gs, brs, Es, head, Kp=K + 2)
            retr = [x["groups"].get("retr", 0) / max(abs(x["amp"]), 1e-6) for x in S]
            row.update(K=K, ptr=p, amp=a, pK1=p[K - 1], first_d4=next((i for i, x in enumerate(p) if x >= 4), None),
                       retr_share=float(np.mean(retr[:K])))
        elif r in res["runs"]:
            row.update({k: v for k, v in res["runs"][r].items() if k not in row})
        res["runs"][r] = row
        json.dump(res, open(OUT, "w", newline="\n"), indent=1)
        print(r, {k: row[k] for k in row if k not in ("ptr", "amp")}, "ptr", row.get("ptr"), flush=True)
    arms = {}
    for a in ("K5", "K6", "K7"):
        L = [v for v in res["runs"].values() if v["arm"] == a]
        C = [v for v in L if v["gate"]]
        arms[a] = {"runs": len(L), "competent": len(C), "extrap": sum(v["extrap"] for v in C),
                   "share": (sum(v["extrap"] for v in C) / len(C)) if C else None,
                   "retr_share_mean": float(np.mean([v["retr_share"] for v in C if "retr_share" in v])) if C else None,
                   "first_d4_mean": float(np.mean([v["first_d4"] if v.get("first_d4") is not None else 12 for v in C])) if C else None,
                   "pK1": sorted(v["pK1"] for v in C if "pK1" in v)}
    s = {a: arms[a]["share"] for a in arms}
    unres = [a for a in arms if arms[a]["competent"] < 3]
    if unres:
        v = "unresolved (" + ", ".join(a + (" too tight to learn" if a == "K5" else "") for a in unres) + ": < 3 competent)"
    elif s["K5"] > s["K6"] > s["K7"] and s["K5"] - s["K7"] >= 0.30:
        v = "supported"
    elif s["K5"] <= s["K7"]:
        v = "not supported"
    else:
        v = "inconclusive"
    res["arms"], res["PC1"] = arms, v
    json.dump(res, open(OUT, "w", newline="\n"), indent=1)
    print(json.dumps({"arms": arms, "PC1": v}, indent=1))


if __name__ == "__main__":
    main()
