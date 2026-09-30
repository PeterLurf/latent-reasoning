"""THEORY §43: per-layer sweep strength of the frontier read (exploratory, measurement only).
S[l, h] = sum over steps 1..K and the 11 chain edges of max(0, mean over graphs of attention(reach edge) - attention(its
look-alike decoy edge)), attention = softmax of the recorded logits (rollout(..., record_layers=True)); probe(512).
    python theory/layer_sweep.py runs/t9l4_s522 ... --out theory/results/layer_sweep.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "2")))
from rollout import load, probe, rollout  # noqa: E402

T = lambda a: torch.as_tensor(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="theory/results/layer_sweep.json")
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, br, E = probe(512)
    bi = np.arange(len(g))[:, None]
    reach = np.concatenate([E["A"][:, 1:], E["B"]], 1); decoy = np.concatenate([E["D"][:, 1:], E["C"]], 1)
    for path in a.runs:
        name = path.rstrip("/\\")
        if name in res:
            continue
        m, c = load(path, "cpu")
        K = c["K"]
        with torch.no_grad():
            o = rollout(m, K, T(g.src), T(g.dst), T(g.start), K, record_layers=True)
        S = []
        for L in o["logitL"]:
            H = L[0].shape[1]
            s = np.zeros(H)
            for k in range(K):
                att = torch.softmax(torch.nan_to_num(L[k], nan=-1e9), -1).numpy()       # (n, H, T)
                for h in range(H):
                    ex = (att[bi, h, reach] - att[bi, h, decoy]).mean(0)
                    s[h] += np.clip(ex, 0, None).sum()
            S.append(s.tolist())
        per_layer = [max(x) for x in S]
        top = int(np.argmax(per_layer))
        res[name] = {"K": K, "S": S, "per_layer_max": per_layer, "argmax_layer_1based": top + 1,
                     "top_share": float(per_layer[top] / max(sum(per_layer), 1e-9))}
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        print(name, "per-layer max S", " ".join("%.2f" % x for x in per_layer), "| top layer", top + 1,
              "share %.2f" % res[name]["top_share"], flush=True)


if __name__ == "__main__":
    main()
