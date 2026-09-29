"""Driver for the follow-up experiments.

    python -m latent.followup_run --run runs/main_s1 --split discovery --device cuda:1
    python -m latent.followup_run --run runs/main_s1 --split fresh --device cuda:1

split=discovery : development graphs (discovery buckets, new sampling seed).
split=fresh     : fresh confirmation graphs from the confirm buckets, excluding every family
                  drawn by any earlier confirm-split sampler in this project (see used_confirm_families).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import followup as F
from . import followup_analysis as FA
from .graphs import sample_graphs


def used_confirm_families():
    """Replay every earlier sampler that touched the confirm split and collect their families."""
    fams = []
    rng = np.random.default_rng(424242)                     # docs/protocol_frozen.json confirmation
    fams.append(sample_graphs(rng, 3000, split="confirm").family)
    for W in range(1, 9):
        fams.append(sample_graphs(rng, 300, widths=[W], split="confirm").family)
    for W in (2, 4, 8):
        fams.append(sample_graphs(rng, 200, widths=[W], split="confirm").family)
    rng = np.random.default_rng(777)                        # competence evaluation
    fams.append(sample_graphs(rng, 4096, split="confirm").family)
    for _ in range(2):                                      # two smoke tests (seed 5)
        rng = np.random.default_rng(5)
        fams.append(sample_graphs(rng, 300, split="confirm").family)
        for W in (2, 8):
            fams.append(sample_graphs(rng, 60, widths=[W], split="confirm").family)
        fams.append(sample_graphs(rng, 50, widths=[4], split="confirm").family)
    rng = np.random.default_rng(0)                          # preflight split audit (train, discovery, confirm order)
    for s in ("train", "discovery", "confirm"):
        g = sample_graphs(rng, 4000, split=s)
        if s == "confirm":
            fams.append(g.family)
    return np.unique(np.concatenate(fams))


def sampler(split, seed):
    rng = np.random.default_rng(seed)
    if split == "discovery":
        return lambda n, widths: sample_graphs(rng, n, widths=widths, split="discovery")
    used = used_confirm_families()

    def fresh(n, widths):
        out = []
        have = 0
        while have < n:
            g = sample_graphs(rng, 2 * n, widths=widths, split="confirm")
            g = g.subset(np.nonzero(~np.isin(g.family, used))[0])
            out.append(g); have += len(g)
        from .graphs import GraphBatch
        g = out[0] if len(out) == 1 else GraphBatch(**{k: np.concatenate([getattr(o, k) for o in out]) for k in GraphBatch.__dataclass_fields__})
        return g.subset(np.arange(n))
    return fresh


EXPERIMENTS = ("dose", "swap", "cont", "cache", "attn", "gain", "jac")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", choices=["discovery", "fresh"], required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--widths", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--k", type=int, default=None, help="edit site (default: the run's frozen k*)")
    p.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    seed = a.seed if a.seed is not None else (31337 if a.split == "discovery" else 271828)
    A = F.load_assets(a.run, a.device)
    k = A["k_star"] if a.k is None else a.k
    recurrent = A["mode"] == "coconut"
    out_dir = os.path.join(a.run, "followup", a.split + (f"_{a.tag}" if a.tag else ""))
    os.makedirs(out_dir, exist_ok=True)
    draw = sampler(a.split, seed)
    summ = {"run": a.run, "split": a.split, "seed": seed, "k": k, "mode": A["mode"], "results": {}}
    t0 = time.time()
    for W in a.widths:
        g = draw(a.n, [W])
        R = summ["results"].setdefault(str(W), {})
        for ex in a.experiments:
            if not recurrent and ex not in ("cache", "attn"):
                continue
            if ex == "dose":
                for direction in ("remove", "insert"):
                    r = F.dose_records(A, g, k, direction=direction, seed=seed + W)
                    R[f"dose_{direction}"] = FA.summarize_dose(r)
            elif ex == "swap":
                R["swap"] = FA.summarize_swap(F.swap_records(A, g, k, seed=seed + W))
            elif ex == "cont":
                for j in (2, 3):
                    R[f"cont_j{j}"] = FA.summarize_continuation(F.continuation_records(A, g, k, j=j, seed=seed + W))
            elif ex == "cache":
                R["cache"] = FA.summarize_cache(F.cache_records(A, g, k, seed=seed + W))
            elif ex == "attn":
                R["attn"] = FA.summarize_attention(F.attention_records(A, g, seed=seed + W))
            elif ex == "gain" and W > 2:
                R["gain"] = FA.summarize_gain(F.gain_restore_records(A, g, k))
            elif ex == "upcache":
                R["upcache"] = FA.summarize_upfront_cache(F.upfront_cache_records(A, g, k, seed=seed + W))
            elif ex == "amp":
                R["amp"] = FA.summarize_amplify(F.amplify_records(A, g, k, seed=seed + W))
            elif ex == "pair":
                R["pair"] = FA.summarize_pair(F.continuation_pair_records(A, g, k, j=2, seed=seed + W))
            elif ex == "gainfit":
                r = F.gain_records(A, g, k)
                R["gainfit"] = {"diff_norm": r["diff_norm"].tolist(), "h_norm": r["h_norm"].tolist()}
            elif ex == "jac":
                rows = FA.jacobian_features(F.jacobian_records(A, g, k))
                R["jac_rows"] = rows
            print(f"[{a.run} {a.split} W={W}] {ex} done {time.time() - t0:.0f}s", flush=True)
        json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=float)
    print("wrote", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
