"""THEORY autosnap memo premise ($0, descriptive, CPU): is the Markov carried latent already near-discrete per state, and
does a fixed per-state codebook fitted in range still classify the drifted latents of a no-snap chain?
Models: §111 MK-4-READ s6 and §108 MK-4 s1 (no reader). n chains from random start-state text, G ops, no snaps.
* Fit = generations 2-4 (latent-fed, in range) of the first half of the chains; latents centred by the fit mean.
  Centroids = the mean centred fit latent per true state. Generation 1 (written from text) is scored, not fitted.
* Per generation g = 1..G on the held-out half: nearest-centroid accuracy (Euclidean and cosine, centred) against the
  true state, and the centred cosine to the state's centroid.
* Unsupervised: k-means (M = 240) purity and NMI on the in-range latents (theory/s101_premise.py's kmeans).
Output theory/results/autosnap_premise.json.
    python theory/autosnap_premise.py --threads 3
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "theory"))
from boxes import s5  # noqa: E402
from boxes.gen97 import STTOK  # noqa: E402
from boxes.mk108 import EOT, MK108  # noqa: E402
from s101_premise import kmeans, purity_nmi  # noqa: E402


@torch.no_grad()
def chains(m, S0, O, bs):
    g = m.g; N, G = O.shape; out = np.zeros((N, G, 768), np.float32)
    for s in range(0, N, bs):
        sl = slice(s, s + bs); n = len(S0[sl]); eot = g.embed(torch.full((n, 1), EOT))
        prev = g.embed(torch.tensor(STTOK[S0[sl]]))
        for t in range(G):
            h, _ = g(torch.cat([eot, prev, g.embed(m.optok[torch.tensor(O[sl, t])])], 1), None)
            s_in = h[:, -1]; out[sl, t] = s_in.float().numpy(); prev = s_in[:, None]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=3); ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--G", type=int, default=32); ap.add_argument("--bs", type=int, default=250)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
    except Exception:
        pass
    rng = np.random.default_rng(12001); S0 = rng.integers(0, 120, a.n); O = rng.integers(0, 120, (a.n, a.G))
    Y = np.zeros((a.n, a.G), np.int64); st = S0.copy()
    for t in range(a.G):
        st = s5.COMP[st, O[:, t]]; Y[:, t] = st
    res = {"n": a.n, "G": a.G, "models": {}}
    for name in ("s111_mk4_read_s6", "s108_mk4_s1"):
        m = MK108(s5.load_gpt2("cpu")); sd = torch.load(os.path.join(ROOT, "boxes", "runs_s5", name, "model.pt"), map_location="cpu")
        missing, unexpected = m.load_state_dict(sd, strict=False); assert not missing and not unexpected, (missing, unexpected)
        m.eval(); L = chains(m, S0, O, a.bs)
        h = a.n // 2; Xin = L[:h, 1:4].reshape(-1, 768); yin = Y[:h, 1:4].reshape(-1); mu = Xin.mean(0); Xin = Xin - mu
        C = np.stack([Xin[yin == k].mean(0) for k in range(120)]); Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
        per = {}
        for t in range(a.G):
            X = L[h:, t] - mu; yt = Y[h:, t]; d = ((X[:, None, :] - C[None]) ** 2).sum(-1); Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
            cs = Xn @ Cn.T
            per[t + 1] = {"nc_euclid": float((d.argmin(1) == yt).mean()), "nc_cos": float((cs.argmax(1) == yt).mean()),
                          "cos_to_own_centroid": float(cs[np.arange(len(yt)), yt].mean()),
                          "norm_ratio": float(np.linalg.norm(L[h:, t], axis=1).mean() / np.linalg.norm(L[:h, 1:4], axis=-1).mean())}
        sub = rng.choice(len(Xin), min(len(Xin), 12000), replace=False)
        km = kmeans(torch.tensor(Xin[sub]), 240, 20, rng, restarts=2); pur, nmi = purity_nmi(km, yin[sub])
        res["models"][name] = {"per_generation": per, "kmeans_in_range": {"purity": pur, "nmi": nmi, "M": 240}}
        print(name, "kmeans purity %.3f nmi %.3f" % (pur, nmi), {g: (round(v["nc_cos"], 3), round(v["cos_to_own_centroid"], 3)) for g, v in per.items() if g in (1, 2, 4, 5, 8, 16, 32)}, flush=True)
        json.dump(res, open(os.path.join(ROOT, "theory", "results", "autosnap_premise.json"), "w", newline="\n"), indent=1)
        del m


if __name__ == "__main__":
    main()
