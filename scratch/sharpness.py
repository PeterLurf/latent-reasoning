"""Exploratory (literature follow-up, docs/LITERATURE_NOTES.md): is the closed/open outcome already visible as
sharper layer-1 frontier attention at the full-depth-entry checkpoint? (Zhu et al. 2026: a bounded index-matching
logit keeps the superposition open; a growing one over-commits.)

For every run with checkpoints and a full-depth advance: at the checkpoint saved at that advance and at the final
model, W = 4 confirm graphs (seed 2026 + 4, 600 graphs), steps 1..K:
  * EA (PROTOCOL5 definition)
  * top1: mean over graphs and steps of the largest single-edge attention share among edges
  * ent: mean attention entropy over edges (nats)
  * reach_mass: attention mass on reachable (attached) edges of any depth
  * front_ratio: per-edge reachable/decoy attention ratio at each step's most-attended depth, mean over steps
    PYTHONPATH=. python scratch/sharpness.py runs/dev6_s30 runs/dev6_s35 ... --out results/sharpness.json
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from latent import protocol5_confirm as P5
from latent.confirm_sweep import trace
from latent.development import load_ckpt
from latent.frontier_amp import per_graph_attention
from latent.graphs import sample_graphs


def metrics(m, c, device):
    K = c["K"]
    g = sample_graphs(np.random.default_rng(2026 + 4), 600, widths=[4], split="confirm")
    A, dep, att = per_graph_attention(m, K, g, max(8, K), device)
    A = A[:K]                                                    # (K, N, E)
    p = A / np.maximum(A.sum(-1, keepdims=True), 1e-12)
    top1 = float(p.max(-1).mean())
    ent = float((-(p * np.log(np.maximum(p, 1e-12))).sum(-1)).mean())
    reach = float((A * att[None]).sum(-1).mean())
    tr = trace(m, K, max(8, K), 4, device)
    R, D = np.array(tr["reachable"])[:K], np.array(tr["decoy"])[:K]
    x = (R - D)[:, 1:].argmax(1) + 1
    ratio = float(np.mean([R[k, x[k]] / max(D[k, x[k]], 1e-9) for k in range(K)]))
    return {"EA": P5.end_activity(tr, K), "top1": top1, "ent": ent, "reach_mass": reach, "front_ratio": ratio}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results/sharpness.json")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for run in a.runs:
        if run in res:
            continue
        adv = [json.loads(l) for l in open(f"{run}/log.jsonl") if "curriculum_advance_to" in l]
        full = next((d["step"] for d in adv if d["curriculum_advance_to"] is None), None)
        f = f"{run}/ckpts/step_{full:05d}.pt" if full else None
        if not f or not os.path.exists(f):
            continue
        o = {"full_depth_step": full}
        for tag, path in (("at_full", f), ("final", f"{run}/model.pt")):
            m, c, step = load_ckpt(path, a.device)
            o[tag] = metrics(m, c, a.device)
        res[run] = o
        json.dump(res, open(a.out, "w"), indent=1)
        print(run, full, {k: round(v, 3) for k, v in o["at_full"].items()}, "final EA", round(o["final"]["EA"], 3),
              flush=True)


if __name__ == "__main__":
    main()
