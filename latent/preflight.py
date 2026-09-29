"""Preflight checks before the follow-up experiments (docs/NEXT_STEPS.md, 'Preflight').

1. Reproduce a published confirmation table from the saved checkpoint and diff it.
2. Audit splits: bucket membership of sampled graphs, family disjointness, split hash.
3. Audit donor widths per condition (flags widths outside training range 1..8).
4. Record retained vs recomputed positions for the thought interface.

    python -m latent.preflight --device cuda:1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from .graphs import E, SPLITS, WMAX, move_branch, sample_graphs


def reproduce(run, device, out_dir):
    from .study import confirm
    confirm(run, "docs/protocol_frozen.json", device, out_dir=out_dir)
    a = json.load(open(os.path.join(run, "confirm", "confirm.json")))["effects"]
    b = json.load(open(os.path.join(out_dir, "confirm.json")))["effects"]
    worst, n = 0.0, 0
    for key in a:
        for m, v in a[key].items():
            if isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
                w = b[key][m]
                if not (np.isnan(v[0]) and np.isnan(w[0])):
                    worst = max(worst, abs(v[0] - w[0])); n += 1
    return {"compared_point_estimates": n, "max_abs_diff": worst, "identical": worst == 0.0}


def split_audit():
    src = open(os.path.join(os.path.dirname(__file__), "graphs.py"), "rb").read()
    rng = np.random.default_rng(0)
    out = {"generator_sha256": hashlib.sha256(src).hexdigest()[:16],
           "splits": {k: [min(v), max(v)] for k, v in SPLITS.items()}}
    fam = {}
    for s in SPLITS:
        g = sample_graphs(rng, 4000, split=s)
        assert np.isin(g.bucket, list(SPLITS[s])).all(), s
        fam[s] = set(g.family.tolist())
    out["family_overlap_train_confirm"] = len(fam["train"] & fam["confirm"])
    out["family_overlap_train_discovery"] = len(fam["train"] & fam["discovery"])
    out["family_overlap_discovery_confirm"] = len(fam["discovery"] & fam["confirm"])
    return out


def donor_audit():
    rows = []
    for W in range(1, WMAX + 1):
        rows.append({"W": W,
                     "remove: recipient/donor width": [W, W - 1],
                     "insert: recipient/donor width": [W - 1, W],
                     "swap (fixed width): recipient/donor width": [W, W],
                     "out_of_training_range": (W - 1) < 1})
    return rows


def positions(K=6):
    start = E
    return {"edges": [0, E - 1], "START": start, "LAT_j": f"{start}+j (j=1..{K})",
            "thought h_k": "final-LN output at position START+k (h_0 = START output)",
            "edit at k retains": f"KV of positions 0..{start}+k (edges, START, LAT_1..LAT_k)",
            "edit at k recomputes": f"LAT_(k+1)..LAT_{K} and every forked query"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:1")
    a = p.parse_args()
    rep = {"reproduction_s0": reproduce("runs/r3_pc_gated_gpt2", a.device, "scratch/repro_s0"),
           "split_audit": split_audit(), "donor_audit": donor_audit(), "interface_positions": positions()}
    os.makedirs("results", exist_ok=True)
    json.dump(rep, open("results/preflight.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "donor_audit"}, indent=1))


if __name__ == "__main__":
    main()
