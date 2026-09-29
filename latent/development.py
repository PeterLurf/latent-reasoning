"""Exploratory (PROTOCOL3 addendum D): how the latent and query-time routes develop during training.

For every checkpoint in runs/<run>/ckpts (plus the final model): depth-matched competence and the
cache-route decomposition (thought T, latent cache TL, query-time graph Gq; shares of the clean
counterfactual effect, overall and by node distance) at W = 2, 4, 8 with edit site k = 1.
The same discovery graphs are used for every checkpoint.

    python -m latent.development runs/dev_s4 --device cuda:0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import warnings

import numpy as np
import torch

from . import followup as F
from . import followup_analysis as FA
from .graphs import NB, NPB, sample_graphs, set_shapes
from .interventions import all_queries
from .model import LatentReasoner
from .upfront_trace import trace_delayed


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck["config"]
    m = LatentReasoner(c["d"], c["layers"], c["heads"], k_max=max(c["K"], 8), mode=c["mode"],
                       emb=c.get("emb", "slots"), d_id=c.get("d_id", 64), init=c.get("init", "torch"),
                       feedback=c.get("feedback", "post_ln"), step_emb=not c.get("no_step_emb", False)).to(device)
    set_shapes(c.get("shapes", "default"), c.get("p_fork", 0.5))   # the run's own graph distribution
    m.load_state_dict(ck["model"])
    return m.eval(), c, ck.get("step")


@torch.no_grad()
def matched(m, K, g, device):
    n = len(g)
    q = all_queries(g)[:, : NB * NPB]
    T = lambda a: torch.as_tensor(a, device=device)
    pred = np.concatenate([(m.run(T(g.src[i:i + 256]), T(g.dst[i:i + 256]), T(g.start[i:i + 256]), K,
                                  qnodes=T(q[i:i + 256]))["logit"] > 0).cpu().numpy() for i in range(0, n, 256)])
    pred = pred.reshape(n, NB, NPB)
    att = np.broadcast_to(g.attached[:, :, None], pred.shape)
    return {j + 1: 0.5 * (float(pred[(g.bdepth == j) & att].mean()) + 1 - float(pred[(g.bdepth == j) & ~att].mean()))
            for j in range(NPB)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--k", type=int, default=1)
    a = p.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)      # empty depth strata
    cks = sorted(glob.glob(os.path.join(a.run, "ckpts", "step_*.pt"))) + [os.path.join(a.run, "model.pt")]
    rng = np.random.default_rng(2718)
    g_comp = sample_graphs(rng, 1024, split="discovery")
    gW = {W: sample_graphs(rng, a.n, widths=[W], split="discovery") for W in (2, 4, 8)}
    out_path = os.path.join("results", f"development_{os.path.basename(a.run)}.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for ck in cks:
        m, c, step = load_ckpt(ck, a.device)
        key = str(step)
        if key in res and ck.endswith("step_%05d.pt" % step):
            continue
        A = {"model": m, "K": c["K"], "probes": None, "pca": None, "k_star": a.k, "device": a.device, "mode": c["mode"]}
        row = {"step": step, "matched": matched(m, c["K"], g_comp, a.device), "cache": {}}
        for W, g in gW.items():
            s = FA.summarize_cache(F.cache_records(A, g, a.k, seed=W))
            cl = s["clean_gap"][0]
            row["cache"][W] = {"clean_gap": cl,
                               "shares": {cn: s["conds"][cn]["targeted"][0] / cl for cn in ("T", "TL", "Gq", "L", "G")}
                               if cl > 0.05 else None,
                               "by_depth": s["by_depth"]}
        # addendum-F frontier schedule: depth read by layer 1 at each latent step (W = 4 graphs)
        tr = trace_delayed(m, c["K"], gW[4], c["K"], a.device)
        sched = []
        for k in range(1, c["K"] + 1):
            t = tr[f"L1_k{k}"]
            j = int(np.argmax(t["reachable"]))
            sched.append(j if t["reachable"][j] >= 1.5 * float(np.mean(t["decoy"])) else None)
        row["frontier_schedule"] = sched
        res[key] = row
        sh = {W: row["cache"][W]["shares"] for W in gW}
        print(f"step {step}: matched " + " ".join(f"d{d}:{v:.2f}" for d, v in row["matched"].items()) + " | " +
              " ".join(f"W{W} TL {s['TL']:.2f} Gq {s['Gq']:.2f}" if s else f"W{W} -" for W, s in sh.items()) +
              f" | frontier {sched}", flush=True)
        json.dump(res, open(out_path, "w"), indent=1, default=float)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
