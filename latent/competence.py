"""Stage A: task competence and whether latent computation is used.

For each checkpoint: balanced accuracy on held-out (confirm-split) families for
both formats, at every inference-time latent budget K' = 0..K, broken down by
query depth and width. Bootstrap CIs resample graphs.

    python -m latent.competence runs/main_s0 runs/pause_s0 runs/k0_s0 --out results/competence.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .data import make_batch
from .graphs import UNREACH, WMAX, sample_graphs, set_shapes
from .model import LatentReasoner


def load(run, device):
    ck = torch.load(os.path.join(run, "model.pt"), map_location=device, weights_only=False)
    c = ck["config"]
    m = LatentReasoner(c["d"], c["layers"], c["heads"], k_max=max(c["K"], 8), mode=c["mode"],
                       emb=c.get("emb", "slots"), d_id=c.get("d_id", 64),
                       init=c.get("init", "torch"), feedback=c.get("feedback", "post_ln"), step_emb=not c.get("no_step_emb", False)).to(device)
    set_shapes(c.get("shapes", "default"), c.get("p_fork", 0.5))   # the run's own graph distribution
    m.load_state_dict(ck["model"])
    return m.eval(), c


def boot_bal_acc(correct, y, gid, n_boot=1000, seed=0):
    """Balanced accuracy with a graph-level bootstrap. correct/y/gid are flat arrays."""
    def bal(c, yy):
        return 0.5 * (c[yy == 1].mean() + c[yy == 0].mean())
    point = bal(correct, y)
    rng = np.random.default_rng(seed)
    G = gid.max() + 1
    # per-graph sums for fast resampling
    s1 = np.bincount(gid, weights=correct * (y == 1), minlength=G)
    n1 = np.bincount(gid, weights=(y == 1), minlength=G)
    s0 = np.bincount(gid, weights=correct * (y == 0), minlength=G)
    n0 = np.bincount(gid, weights=(y == 0), minlength=G)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, G, G)
        bs.append(0.5 * (s1[i].sum() / max(n1[i].sum(), 1) + s0[i].sum() / max(n0[i].sum(), 1)))
    return float(point), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


@torch.no_grad()
def evaluate_run(run, n=4096, Q=16, device="cuda:0", seed=777, split="confirm"):
    m, c = load(run, device)
    K = c["K"]
    rng = np.random.default_rng(seed)
    g = sample_graphs(rng, n, widths=range(1, WMAX + 1), split=split)
    batches = []
    for i in range(0, n, 512):
        sub = g.subset(np.arange(i, min(n, i + 512)))
        batches.append((i, sub, make_batch(rng, sub, Q, device)))
    out = {"run": run, "config": c, "budgets": {}}
    for Kp in range(0, K + 1):
        rows_d, rows_u = [], []
        for i0, sub, b in batches:
            nb = len(sub)
            ld = m.run(b["src"], b["dst"], b["start"], Kp, qnodes=b["q"])["logit"]
            lu = m.run(b["src"], b["dst"], b["start"], Kp, target=b["tgt"])["logit"][:, 0]
            gid = np.arange(i0, i0 + nb)
            rows_d.append(np.stack([np.repeat(gid, Q), np.repeat(sub.width, Q), b["qd"].cpu().numpy().ravel(),
                                    ((ld > 0).float() == b["qy"]).cpu().numpy().ravel(),
                                    b["qy"].cpu().numpy().ravel()], 1))
            rows_u.append(np.stack([gid, sub.width, b["tgtd"].cpu().numpy(),
                                    ((lu > 0).float() == b["ty"]).cpu().numpy(), b["ty"].cpu().numpy()], 1))
        res = {}
        for name, rows in (("delayed", rows_d), ("upfront", rows_u)):
            a = np.concatenate(rows)
            gid, w, d, cor, y = a[:, 0].astype(int), a[:, 1], a[:, 2], a[:, 3], a[:, 4]
            r = {"bal_acc": boot_bal_acc(cor, y, gid)}
            r["by_depth"] = {int(dd): float(cor[d == dd].mean()) for dd in np.unique(d)}
            r["by_width"] = {int(ww): float(0.5 * (cor[(w == ww) & (y == 1)].mean() + cor[(w == ww) & (y == 0)].mean()))
                             for ww in np.unique(w)}
            res[name] = r
        out["budgets"][Kp] = res
        print(f"{run} K'={Kp} delayed {res['delayed']['bal_acc'][0]:.3f} upfront {res['upfront']['bal_acc'][0]:.3f} "
              f"depth(delayed) {{{', '.join(f'{k}:{v:.2f}' for k, v in res['delayed']['by_depth'].items())}}}",
              flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--split", default="confirm")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    res = [evaluate_run(r, a.n, device=a.device, split=a.split) for r in a.runs]
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
