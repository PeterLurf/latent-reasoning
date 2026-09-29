"""PROTOCOL4 addendum E: fresh-graph confirmations of the delayed sweep (P23 blocking, P24 width, P25 frontier).

    python -m latent.confirm_sweep run <runs...> --device cuda:0      # measurements -> results/confirm_sweep.json
    python -m latent.confirm_sweep score                                # verdicts     -> results/confirm_sweep_verdicts.json

All graphs come from the held-out confirm split (seed 2026 + offset); fork graphs (no family split) use seed 4052.
Definitions are those frozen in addendum E; nothing here is tuned after seeing results.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .competence import load
from .delayed_block import run as block_run
from .frontier_amp import per_graph_attention
from .graphs import E, NB, NPB, sample_graphs, set_shapes
from .interventions import all_queries
from .train import evaluate

OUT, VERD = "results/confirm_sweep.json", "results/confirm_sweep_verdicts.json"
MODELS = ["runs/r3_pc_gated_gpt2", "runs/main_s1", "runs/main_s2", "runs/main_s3", "runs/v1_wide_s0", "runs/v2_deep_s0",
          "runs/v3_rawfb_s0", "runs/budget_k4_s0", "runs/budget_k4_s2", "runs/budget_k8_s0", "runs/nostep_s0",
          "runs/nostep_s2", "runs/nostep_s4", "runs/budget_k8_s1", "runs/nostep_k8_s0", "runs/nostep_k8_s1",
          "runs/randk_s0", "runs/randk_s2", "runs/budget_k10_s1", "runs/budget_k10_s2", "runs/budget_k10_s3",
          "runs/v3_rawfb_s1"]
EXPLORATORY = ["runs/r3_pc_gated_gpt2", "runs/main_s1", "runs/main_s2", "runs/main_s3"]   # P25 replication set
DEPTHS = (1, 2, 3)


def trace(m, K, Kp, W, device, n=600, seed=2026, split="confirm"):
    """Pooled per-edge layer-1 attention by (step, depth) on reachable / decoy edges, plus depth-1 frontier size."""
    g = sample_graphs(np.random.default_rng(seed + W), n, widths=[W], split=split)
    A, dep, att = per_graph_attention(m, K, g, Kp, device)
    reach = [[float((A[k] * (att & (dep == x))).sum() / max((att & (dep == x)).sum(), 1)) for x in range(NPB)] for k in range(Kp)]
    decoy = [[float((A[k] * (~att & (dep == x))).sum() / max((~att & (dep == x)).sum(), 1)) for x in range(NPB)] for k in range(Kp)]
    return {"reachable": reach, "decoy": decoy, "n_depth1": float((att & (dep == 1)).sum(1).mean())}


def clock(tr):
    R, D = np.array(tr["reachable"]), np.array(tr["decoy"])
    ks = [int(np.argmax(R[:, x])) for x in DEPTHS]
    strong = all(R[k, x] >= 1.5 * D[k, x] for k, x in zip(ks, DEPTHS))
    return [k + 1 for k in ks], bool(strong and ks[0] < ks[1] < ks[2])


def blocking(m, K, Kp, device, n=400, seed=2027, split="confirm"):
    """latent/delayed_block.py design on confirm graphs; rows = blocked step, cols = blocked depth 0..4."""
    rng = np.random.default_rng(seed)
    g = sample_graphs(rng, 4 * n, widths=[4], split=split)
    deep = (g.bdepth.max(2) >= 4) & g.attached
    A = np.argmax(np.where(deep, rng.random(deep.shape), -1), 1)
    keep = np.nonzero(deep.any(1))[0][:n]
    g, A = g.subset(keep), A[keep]
    N, bi = len(g), np.arange(len(keep))
    q = all_queries(g)[:, : NB * NPB]
    dA = g.bdepth[bi, A]
    edge_of = np.zeros((N, NPB), int)
    edge_of[:, 0] = g.hub_pos[bi, A]
    for j in range(1, NPB):
        edge_of[:, j] = np.argmax(g.dst == g.nodes[bi, A, j][:, None], 1)
    others = g.attached.copy(); others[bi, A] = False
    base = block_run(m, K, g, q, Kp, device).reshape(N, NB, NPB)
    dn, ct = {}, {}
    for k in range(1, Kp + 1):
        dn[k], ct[k] = [], []
        for x in range(0, 5):
            mask = np.zeros((N, E), bool)
            for j in range(NPB):
                sel = dA[:, j] == x
                mask[bi[sel], edge_of[sel, j]] = True
            d = block_run(m, K, g, q, Kp, device, k, mask).reshape(N, NB, NPB) - base
            dn[k].append(float(np.nanmean(np.where(dA > x, d[bi, A], np.nan))))
            ct[k].append(float(np.nanmean(np.where(others[:, :, None], d, np.nan))))
    return {"n": N, "downstream": dn, "control": ct}


def fork_test(m, K, Kp, ks, device, n=8000, seed=4052):
    set_shapes("fork", 0.5)
    try:
        acc = evaluate(m, K, n=1024, split="confirm", seed=seed + 1, device=device)["delayed"]
        g = sample_graphs(np.random.default_rng(seed), n, widths=[2, 4, 8], split=None)
    finally:
        set_shapes("default")
    acc = acc["bal_acc"] if isinstance(acc, dict) else acc
    A, dep, att = per_graph_attention(m, K, g, Kp, device)
    W = g.width
    kids = np.stack([(g.parent[:, 0] == j).sum(1) for j in range(NPB)], 1)
    fork = kids.max(1) == 2
    out = {"fork_bal_acc": float(acc), "ratio": {}}
    for x, k in zip(DEPTHS, ks):
        msk, dm = att & (dep == x), ~att & (dep == x)
        nn = msk.sum(1)
        ex = (A[k - 1] * msk).sum(1) / np.maximum(nn, 1) - (A[k - 1] * dm).sum(1) / np.maximum(dm.sum(1), 1)
        fj = fork & (nn == 2 * W)
        out["ratio"][x] = {int(w): float(ex[fj & (W == w)].mean() / ex[~fork & (W == w)].mean())
                           for w in (2, 4, 8) if (fj & (W == w)).sum() >= 30}
    return out


def measure(runs, device):
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for r in runs:
        if r in res:
            continue
        m, c = load(r, device)
        K = c["K"]; Kp = max(8, K)
        o = {"K": K, "Kp": Kp, "trace": {W: trace(m, K, Kp, W, device) for W in (2, 4, 8)}}
        o["clock"], o["sweeps"] = clock(o["trace"][4])
        o["clock_by_W"] = {W: clock(o["trace"][W])[0] for W in (2, 4, 8)}
        o["block"] = blocking(m, K, Kp, device)
        o["fork"] = fork_test(m, K, Kp, o["clock"], device)
        res = {**(json.load(open(OUT)) if os.path.exists(OUT) else {}), r: o}
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"{r}: clock {o['clock']} sweeps {o['sweeps']} fork acc {o['fork']['fork_bal_acc']:.3f}", flush=True)


def score():
    res = json.load(open(OUT))
    per = {}
    for r, o in res.items():
        p = {"sweeps": o["sweeps"], "clock": o["clock"]}
        if o["sweeps"]:
            ks, B = o["clock"], o["block"]
            dn = {int(k): v for k, v in B["downstream"].items()}
            ct = {int(k): v for k, v in B["control"].items()}
            diag = all(dn[k][x] <= -0.05 for x, k in zip(DEPTHS, ks))
            off = all(abs(dn[k][x]) <= 0.02 for x, kx in zip(DEPTHS, ks) for k in dn if abs(k - kx) >= 2)
            ctl = all(v >= -0.01 for k in ct for v in ct[k])
            p["P23"] = {"diag": diag, "off": off, "control": ctl, "pass": diag and off and ctl}
            T = o["trace"]; k1 = ks[0] - 1
            ex = lambda W: T[str(W)]["reachable"][k1][1] - T[str(W)]["decoy"][k1][1]
            same = all(o["clock_by_W"][str(W)] == ks for W in (2, 4, 8))
            r_edge = ex(8) / ex(2)
            r_tot = ex(8) * T["8"]["n_depth1"] / (ex(2) * T["2"]["n_depth1"])
            p["P24"] = {"same_clock": same, "edge_W8_W2": r_edge, "total_W8_W2": r_tot,
                        "pass": bool(same and 0.10 <= r_edge <= 0.45 and 0.6 <= r_tot <= 1.6)}
            F = o["fork"]
            if F["fork_bal_acc"] < 0.80:
                p["P25"] = {"scorable": False, "fork_bal_acc": F["fork_bal_acc"]}
            else:
                mr = {x: float(np.mean(list(F["ratio"][str(x)].values()))) if F["ratio"][str(x)] else None for x in (2, 3)}
                p["P25"] = {"scorable": all(v is not None for v in mr.values()), "fork_bal_acc": F["fork_bal_acc"],
                            "mean_ratio": mr, "pass": all(v is not None and v <= 0.70 for v in mr.values())}
        per[r] = p

    def reading(models, key):
        sc = [r for r in models if key in per[r] and per[r][key].get("scorable", True)]
        k = sum(per[r][key]["pass"] for r in sc)
        frac = k / len(sc) if sc else None
        return {"scorable": sc, "n_pass": k, "frac": frac,
                "reading": None if frac is None else "confirmed" if frac >= 0.8 else "fails" if frac < 0.5 else "mixed"}

    V = {"P23": reading(list(per), "P23"), "P24": reading(list(per), "P24"),
         "P25_primary": reading([r for r in per if r not in EXPLORATORY], "P25"),
         "P25_replication_s0_s3": reading([r for r in per if r in EXPLORATORY], "P25"),
         "not_sweeping": [r for r in per if not per[r]["sweeps"]]}
    json.dump({"per_model": per, "verdicts": V}, open(VERD, "w"), indent=1, default=float)
    for k, v in V.items():
        print(k, v if k == "not_sweeping" else {kk: v[kk] for kk in ("n_pass", "frac", "reading")} | {"n": len(v["scorable"])})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.cmd == "run":
        measure(a.runs or MODELS, a.device)
    else:
        score()
