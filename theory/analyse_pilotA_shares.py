"""Pilot A PA4: retrieval / local pointer shares (exp4 split) for Coconut-arm runs; sweep head from exp7 method."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, "theory"); sys.path.insert(0, ".")
from eval_arch import load_arch
from exp4_shares import shares
from exp7_dev import sweep_head
from rollout import probe
g, br, E = probe(256)
out = {}
for r in sorted(d for d in os.listdir("theory/runs") if d.startswith("pa_c_") and os.path.exists(f"theory/runs/{d}/eval.json")):
    ev = json.load(open(f"theory/runs/{r}/eval.json"))
    m, c = load_arch(f"theory/runs/{r}", "cpu"); K = c["K"]
    with torch.no_grad():
        head, _ = sweep_head(m, K, g, br, E)
        S = shares(m, K, g, br, E, head, Kp=K + 2)
    sh = lambda x, k: x["groups"].get(k, 0) / max(abs(x["amp"]), 1e-6)
    retr_tr = [sh(x, "retr") for x in S[:K - 1]]
    o = {"gate": ev["gate"], "extrap": ev["extrap_K12"], "d7": ev["d7_K12"], "retr_max_tr": max(retr_tr), "retr_K": sh(S[K - 1], "retr"),
         "ptr": [x["ptr"] for x in S]}
    o["retr_type"] = o["retr_max_tr"] >= 0.5 or o["retr_K"] >= 0.5
    out[r] = o
    print("%-10s gate %-5s extrap %-5s d7 %.3f | retr_max_tr %.2f retr_K %.2f -> %s | ptr %s" % (r, o["gate"], o["extrap"], o["d7"], o["retr_max_tr"], o["retr_K"],
          "RETRIEVAL" if o["retr_type"] else "local", " ".join(map(str, o["ptr"]))))
json.dump(out, open("theory/results/pilotA_shares.json", "w"), indent=1)
