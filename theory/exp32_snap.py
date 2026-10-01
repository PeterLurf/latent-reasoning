"""THEORY §67: is the S3 single loop's failure past its trained length off-manifold drift, or an out-of-range count/history?
Inference-time snapping of latent thoughts to a codebook fitted on in-range thoughts. CPU, no training.

For each gated §64 S3 model:
  codebook : k-means (Lloyd, k-means++ init, 30 iterations, seed 0) on thoughts h_1..h_8 of 4096 training-distribution
             problems (N ~ U{1..8}, K = 8, init random or identity as in training; seed 7000). POOLED over steps (primary,
             step-agnostic) with C = 128 (primary), 256 and 32; per-step codebooks (C = 32) for the secondary age-cycle arm.
  arms     : none; snap (pooled, every s steps; s = 1 primary, 4 secondary); cycle (secondary, step-indexed: at step k snap
             to the codebook of in-range step 1 + (k - 1) mod 8); restart = §64's eval_st.json value (written-state
             restarts every 8 ops).
  test     : N 9-16 and 17-24 at K' = N (seed 1000 + N, 1024 each, as §64), and N 1-8 (the in-range cost of snapping)
  measures : full-state accuracy; the share of answers that are valid arrangements; codebook purity (the share of each
             centroid's training thoughts whose current true arrangement is the centroid's majority; the number of
             distinct majority arrangements)
    python theory/exp32_snap.py --out theory/results/exp32_snap.json [--device cuda:0]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from st_model import NMAX_TRAIN, STReasoner, group_elems, sample, tt  # noqa: E402


def kmeans(X, C, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    cent = [X[rng.integers(len(X))]]
    d2 = ((X - cent[0]) ** 2).sum(1)
    for _ in range(C - 1):
        cent.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((X - cent[-1]) ** 2).sum(1))
    cent = np.stack(cent)
    for _ in range(iters):
        lab = ((X[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
        for c in range(C):
            if (lab == c).any():
                cent[c] = X[lab == c].mean(0)
    return cent, lab


def snapper(cent, dev="cpu"):
    ct = torch.as_tensor(cent, dtype=torch.float32, device=dev)
    return lambda h: ct[torch.cdist(h, ct).argmin(1)]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="theory/results/exp32_snap.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--runs", nargs="*", help="run dirs (default: all gated theory/runs/st/s3_s*)")
    ap.add_argument("--n_test", type=int, default=1024); ap.add_argument("--n_fit", type=int, default=4096)
    ap.add_argument("--Ns", type=int, nargs="*", default=list(range(1, 25)))
    ap.add_argument("--no_gate", action="store_true", help="smoke only: skip the in-range gate")
    a = ap.parse_args(); torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "3"))); dev = a.device
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for path in (a.runs or sorted(glob.glob("theory/runs/st/s3_s*"))):
        name = os.path.basename(path)
        if name in res or not os.path.exists(os.path.join(path, "eval_st.json")):
            continue
        ev = json.load(open(os.path.join(path, "eval_st.json")))
        if not a.no_gate and np.mean([ev["single"][str(k)] for k in range(1, 9)]) < 0.90:
            continue
        ck = torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=False)
        m = STReasoner("S3").to(dev); m.load_state_dict(ck["model"]); m.eval()
        n, G = group_elems("S3")
        # in-range thoughts and their current true arrangement (after min(k, N) ops)
        init, ops, pos, nops, final, _ = sample(np.random.default_rng(7000), "S3", a.n_fit, 1, NMAX_TRAIN)
        H = {}
        m.run(*tt(dev, init, ops, pos, nops), NMAX_TRAIN, patch=lambda k, h: H.__setitem__(k, h.clone()) or h)
        cur = []
        for k in range(1, NMAX_TRAIN + 1):
            st = init.copy()
            for b in range(len(init)):
                s = init[b].copy()
                for t in range(min(k, nops[b])):
                    s = np.array(G[ops[b, t]])[s]
                st[b] = s
            cur.append(st)
        Xp = torch.cat([H[k] for k in range(1, NMAX_TRAIN + 1)]).cpu().numpy()
        Sp = np.concatenate(cur)
        key = np.array([hash(tuple(r)) for r in Sp])
        out = {"codebooks": {}}
        books = {}
        for C in (128, 256, 32):
            cent, lab = kmeans(Xp, C)
            maj, pur = [], []
            for c in range(C):
                ks = key[lab == c]
                if len(ks):
                    v, cnt = np.unique(ks, return_counts=True); maj.append(v[cnt.argmax()]); pur.append(cnt.max() / len(ks))
            out["codebooks"][C] = {"purity_mean": float(np.mean(pur)), "distinct_majority_states": int(len(set(maj)))}
            books[("pool", C)] = snapper(cent, dev)
            if C == 32:                                                     # per-step books (secondary, step-indexed)
                books[("step", C)] = {k: snapper(kmeans(H[k].cpu().numpy(), C)[0], dev) for k in range(1, NMAX_TRAIN + 1)}
        arms = {"snap128_s1": ("pool", 128, 1), "snap128_s4": ("pool", 128, 4), "snap256_s1": ("pool", 256, 1),
                "snap32_s1": ("pool", 32, 1), "cycle32_s1": ("step", 32, 1)}
        for N in a.Ns:
            init, ops, pos, nops, final, _ = sample(np.random.default_rng(1000 + N), "S3", a.n_test, N, N, Nmax=24)
            I, O, P, Nn, Fi = tt(dev, init, ops, pos, nops, final)
            K = max(NMAX_TRAIN, N)
            row = {}
            for arm, (kind, C, s) in [("none", (None, None, None))] + list(arms.items()):
                if kind is None:
                    p = m.run(I, O, P, Nn, K).argmax(-1)
                else:
                    bk = books[(kind, C)]
                    f = (lambda k, h, bk=bk, s=s: bk(h) if k % s == 0 else h) if kind == "pool" else \
                        (lambda k, h, bk=bk, s=s: bk[1 + (k - 1) % NMAX_TRAIN](h) if k % s == 0 else h)
                    p = m.run(I, O, P, Nn, K, patch=f).argmax(-1)
                row[arm] = {"acc": float((p == Fi).all(1).float().mean()),
                            "valid": float(np.mean([len(set(r)) == n for r in p.cpu().numpy()]))}
            row["restart"] = {"acc": ev["restart"][str(N)]}
            out[N] = row
        res[name] = out
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        f = lambda arm, ks: np.mean([out[k][arm]["acc"] for k in ks if k in out]) if any(k in out for k in ks) else float("nan")
        print("%-7s purity128 %.2f | in-range none %.3f snap128 %.3f snap256 %.3f | N9-16 none %.3f snap128 %.3f snap256 %.3f snap128/4 %.3f snap32 %.3f cycle %.3f restart %.3f" % (
            name, out["codebooks"][128]["purity_mean"], f("none", range(1, 9)), f("snap128_s1", range(1, 9)), f("snap256_s1", range(1, 9)),
            f("none", range(9, 17)), f("snap128_s1", range(9, 17)), f("snap256_s1", range(9, 17)), f("snap128_s4", range(9, 17)),
            f("snap32_s1", range(9, 17)), f("cycle32_s1", range(9, 17)), f("restart", range(9, 17))), flush=True)


if __name__ == "__main__":
    main()
