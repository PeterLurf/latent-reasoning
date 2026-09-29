"""PROTOCOL7 T6: P7 multiverse on P30 + L competent runs. Written 2026-09-28 ~23:58, before any specification was computed.
Specifications: probe seed {2033, 7, 11} x probe widths {2/4/6, 2, 6} x layer {1, 2} x floor {0.02, 0.03, 0.05} (54).
Runs below a specification's floor are left out of that specification's AUC. Supported if AUC >= 0.85 in >= 80 %.
Permutation p (10 000 shuffles) for the frozen specification (seed 2033, 2/4/6, layer 1, floor 0.03).
    PYTHONPATH=. python scratch/p7s_t6.py --device cuda:0
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, "scratch")
import far_sweep_layers as FL                                           # noqa: E402
from latent import length_gen                                           # noqa: E402

OUT = "results/p7s_t6.json"
SEEDS, WIDTHS, LAYERS, FLOORS = (2033, 7, 11), ((2, 4, 6), (2,), (6,)), (1, 2), (0.02, 0.03, 0.05)


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool)
    pos, neg = x[y], x[~y]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    lab = {}
    for f in ("results/p30.json", "results/p_l.json"):
        d = json.load(open(f))
        for r in d["primary"]:
            lab[f"runs/{r}"] = bool(d["rows"][r]["extrap"])
    res = json.load(open(OUT)).get("raw", {}) if os.path.exists(OUT) else {}
    orig = FL.long_chain_graphs
    for s, w in itertools.product(SEEDS, WIDTHS):
        key = f"{s}|{'-'.join(map(str, w))}"
        res.setdefault(key, {})
        FL.long_chain_graphs = lambda rng, n, widths, _w=w: orig(rng, n, list(_w))
        for r in lab:
            if r not in res[key]:
                m = FL.measure(r, a.device, seed=s)
                res[key][r] = {li: {"P7": v["P7"], "in_peak": v["in_peak"]} for li, v in m["layers"].items()}
        json.dump({"raw": res}, open(OUT, "w"))
        print("done", key, flush=True)
    FL.long_chain_graphs = orig
    specs = []
    for s, w, li, fl in itertools.product(SEEDS, WIDTHS, LAYERS, FLOORS):
        key = f"{s}|{'-'.join(map(str, w))}"
        rr = [r for r in lab if res[key][r][str(li) if str(li) in res[key][r] else li]["in_peak"] >= fl]
        get = lambda r: res[key][r][str(li) if str(li) in res[key][r] else li]
        x = [get(r)["P7"] for r in rr]; y = [lab[r] for r in rr]
        specs.append({"seed": s, "widths": w, "layer": li, "floor": fl, "n": len(rr), "n_extrap": int(sum(y)), "auc": auc(x, y)})
    ok = [sp for sp in specs if sp["auc"] is not None and sp["auc"] >= 0.85]
    frozen_key = "2033|2-4-6"
    rr = [r for r in lab if res[frozen_key][r]["1" if "1" in res[frozen_key][r] else 1]["in_peak"] >= 0.03]
    x = np.array([res[frozen_key][r]["1" if "1" in res[frozen_key][r] else 1]["P7"] for r in rr]); y = np.array([lab[r] for r in rr])
    a0 = auc(x, y); rng = np.random.default_rng(0)
    perm = np.array([auc(x, rng.permutation(y)) for _ in range(10000)])
    summ = {"n_specs": len(specs), "share_auc_ge_0.85": len(ok) / len(specs), "T6": "supported" if len(ok) / len(specs) >= 0.80 else "not supported",
            "frozen_auc": a0, "perm_p": float((1 + (perm >= a0).sum()) / (1 + len(perm))),
            "auc_min": min(sp["auc"] for sp in specs if sp["auc"] is not None),
            "by_layer": {li: float(np.median([sp["auc"] for sp in specs if sp["layer"] == li and sp["auc"] is not None])) for li in LAYERS},
            "n_undefined": sum(sp["auc"] is None for sp in specs)}
    json.dump({"raw": res, "specs": specs, "summary": summ}, open(OUT, "w"), indent=1)
    print(summ)


if __name__ == "__main__":
    main()
