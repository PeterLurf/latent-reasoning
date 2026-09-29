"""PROTOCOL7 T8 negative controls. Written 2026-09-28 ~23:57, before either control was computed.
(a) pause-token models (mode "pause"): P7 (independent code, T7-audited) and d7 at K'=12 (latent.length_gen);
(b) placebo P7 on P30 + L competent runs: max over steps of [attention on C1's edge (unreachable distance-7 look-alike)
    - attention on a random edge of another unreachable branch], / in_peak. AUC for extrapolation must lie in [0.35, 0.65].
    PYTHONPATH=. python scratch/p7s_t8.py --device cuda:0
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "scratch")
import p7_indep as PI                                                   # noqa: E402
from latent import length_gen as LG                                     # noqa: E402
from latent.competence import load                                      # noqa: E402
from latent.graphs import NB                                            # noqa: E402
from latent.length_gen import long_chain_graphs                         # noqa: E402


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool)
    pos, neg = x[y], x[~y]
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def placebo(run, device, seed=2033, n=1024, Kp=12):
    m, c = load(run, device)
    K = c["K"]
    g, br = long_chain_graphs(np.random.default_rng(seed), n, [2, 4, 6])
    bi = np.arange(n)
    idx = {ch: PI.edges_into(g, g.nodes[bi, br[ch]]) for ch in "ABCD"}
    rng = np.random.default_rng(7)
    other = np.zeros(n, int)
    for i in range(n):
        cand = [b for b in range(NB) if not g.attached[i, b] and b not in (br["B"][i], br["C"][i], br["D"][i])]
        b = rng.choice(cand)
        other[i] = PI.edges_into(g[i:i + 1] if hasattr(g, "__getitem__") else g, g.nodes[i:i + 1, b])[0, rng.integers(0, 6)] \
            if False else np.argmax(g.dst[i] == g.nodes[i, b, rng.integers(0, 6)])
    A = PI.first_block_attention(m, K, g, Kp, device)
    at = lambda e: A[:, bi, e].mean(1)
    near = np.stack([at(idx["A"][:, j]) - at(idx["D"][:, j]) for j in range(6)], 1)
    in_peak = float(near[:K].max())
    return float((at(idx["C"][:, 0]) - at(other)).max()) / in_peak, in_peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = {"pause": {}, "placebo": {}}
    for r in ("runs/pause_s0", "runs/pause_s1", "runs/pause_s2"):
        p = PI.p7(r, a.device)
        L = LG.evaluate(r, a.device, budgets=(12,))
        row = L["rows"][0]
        out["pause"][r] = {"P7": p["P7"], "in_peak": p["in_peak"], "d7": row["far"]["7"], "d7_ci": row["far_ci"]["7"],
                           "extrap": bool(row["far"]["7"] >= 0.60 and row["far_ci"]["7"][0] > 0.55)}
    lab = {}
    for f in ("results/p30.json", "results/p_l.json"):
        d = json.load(open(f))
        for r in d["primary"]:
            lab[f"runs/{r}"] = d["rows"][r]["extrap"]
    for r in lab:
        v, ip = placebo(r, a.device)
        out["placebo"][r] = {"placebo_P7": v, "in_peak": ip, "extrap": lab[r]}
    y = [out["placebo"][r]["extrap"] for r in lab]; x = [out["placebo"][r]["placebo_P7"] for r in lab]
    au = auc(x, y)
    out["summary"] = {"pause": {r: (round(v["P7"], 3), round(v["d7"], 3), v["extrap"]) for r, v in out["pause"].items()},
                      "placebo_auc": au, "placebo_ok": 0.35 <= au <= 0.65}
    json.dump(out, open("results/p7s_t8.json", "w"), indent=1)
    print(out["summary"])


if __name__ == "__main__":
    main()
