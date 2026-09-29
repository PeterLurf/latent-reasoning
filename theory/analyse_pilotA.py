"""Exploratory: pointer trajectories of Pilot A models (Markov models reproduced by the rollout harness with
block_from=1 on the Markov layers). Prints pointer depth / amplitude per thought h_0..h_11 on the long-chain probe."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, "theory"); sys.path.insert(0, ".")
from eval_arch import load_arch
from rollout import probe, rollout

def ptr_traj(m, K, g, br, E, kw, Kp=12):
    T = lambda a: torch.as_tensor(a)
    out = rollout(m, K, T(g.src), T(g.dst), T(g.start), Kp, **kw)
    n = len(g); bi = np.arange(n)[:, None]
    reach = np.concatenate([E["A"][:, 1:], E["B"]], 1); decoy = np.concatenate([E["D"][:, 1:], E["C"]], 1)
    H = out["logit1"][0].shape[1]
    A = np.zeros((Kp, H, 11)); L = np.zeros((Kp, H, 11))
    for k in range(Kp):
        a = out["attn1"][k].numpy(); s = out["logit1"][k].numpy()
        for h in range(H):
            A[k, h] = (a[bi, h, reach] - a[bi, h, decoy]).mean(0); L[k, h] = np.nan_to_num(s[bi, h, reach] - s[bi, h, decoy]).mean(0)
    head = int(np.argmax(A[:K].clip(0).sum((0, 2))))
    Lh = L[:, head]
    return [int(np.argmax(r)) + 1 if r.max() >= 1 else 0 for r in Lh], [float(r.max()) for r in Lh], head

if __name__ == "__main__":
    g, br, E = probe(512)
    res = {}
    for r in sorted(d for d in os.listdir("theory/runs") if d.startswith("pa_") and os.path.exists(f"theory/runs/{d}/eval.json")):
        ev = json.load(open(f"theory/runs/{r}/eval.json")); A = json.load(open(f"theory/runs/{r}/arch.json"))
        m, c = load_arch(f"theory/runs/{r}", "cpu")
        ml = getattr(m, "markov_layers", ())
        kw = {"block_from": 1, "block_layers": ml} if ml else {}
        with torch.no_grad():
            p, a, head = ptr_traj(m, c["K"], g, br, E, kw)
        res[r] = {"arch": A["arch"], "gate": ev["gate"], "extrap": ev["extrap_K12"], "d7": ev["d7_K12"], "far": ev["far8_10_K12"],
                  "stage": ev["final_stage"], "ptr": p, "amp": a, "head": head}
        print("%-12s %-9s gate %-5s ext %-5s d7 %.3f far %.3f | ptr %s | amp %s" % (r, A["arch"], ev["gate"], ev["extrap_K12"], ev["d7_K12"], ev["far8_10_K12"],
              " ".join(map(str, p)), " ".join("%.1f" % x for x in a)), flush=True)
    json.dump(res, open("theory/results/pilotA_ptr.json", "w"), indent=1)
