"""THEORY note 101-2 follow-up (descriptive, $0; Audit's suggestion): where does the S5 state cluster?
k-means (M = 240) purity / NMI against the true state after a 4-op chunk, 20k fresh chunks, on:
  raw_std   per-coordinate z-score of the last-slot output hidden (unit variance; no embedding rescale)
  whiten    PCA whitening (top 128 components, each scaled to unit variance)
  drop10    z-score after removing the top-10 principal components (tests a low-variance-subspace account, label-free)
  head      (§92 only) the 5 x 5 head's 25 logits: the label-trained readout subspace (a positive control / ceiling)
Models: §97 HYB-4 s1-3 (last slot of gen97's chain) and §92 k = 4 s1-3 (latent86).
    python theory/s101_manifold_probe.py --device cuda:0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "theory"))
from boxes import gen97 as G  # noqa: E402
from boxes import latent86 as L  # noqa: E402
from boxes import s5  # noqa: E402
from s101_premise import kmeans, purity_nmi  # noqa: E402


def variants(H, head=None):
    out = {}
    Z = (H - H.mean(0)) / H.std(0).clamp_min(1e-6); out["raw_std"] = Z
    U, S, V = torch.pca_lowrank(H - H.mean(0), q=128, center=False)
    P = (H - H.mean(0)) @ V; out["whiten"] = P / P.std(0).clamp_min(1e-6)
    top = V[:, :10]; R = (H - H.mean(0)) - (H - H.mean(0)) @ top @ top.T
    out["drop10"] = R / R.std(0).clamp_min(1e-6)
    if head is not None:
        out["head"] = head
    return out


@torch.no_grad()
def collect_g97(m, S, O, dev, bs=250):
    H = []
    for i in range(0, len(S), bs):
        g = m.g; start = m.start_from(els=S[i:i + bs], dev=dev); ops = torch.tensor(O[i:i + bs], device=dev)
        x = torch.cat([m.start_emb(start), g.embed(m.optok[ops[:, 0]])], 1); past = None
        for j in range(4):
            h, past = g(x, past); hs, past = g(h[:, -1][:, None], past)
            if j < 3:
                x = g.embed(m.optok[ops[:, j + 1]])
        H.append(hs[:, 0].float().cpu())
    return torch.cat(H)


@torch.no_grad()
def collect_l86(m, S, O, dev, bs=250):
    H, Hd = [], []
    for i in range(0, len(S), bs):
        lg, _ = m.chunk(torch.tensor(S[i:i + bs], device=dev), torch.tensor(O[i:i + bs], device=dev))
        H.append(m._slot_out[-1].float().cpu()); Hd.append(lg[:, -1].reshape(lg.shape[0], -1).float().cpu())
    return torch.cat(H), torch.cat(Hd)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args(); dev = a.device; res = {}; torch.set_num_threads(3)
    for fam, path, arm in (("hyb4", "g97_hyb4_s%d", "hyb"), ("l92", "l92_k4_s%d", None)):
        for s in (1, 2, 3):
            rng = np.random.default_rng(10100 + s); S = rng.integers(0, 120, a.n); O = rng.integers(0, 120, (a.n, 4))
            Y = G.true_chain(S, O)[:, -1]
            if arm:
                m = G.Gen97(s5.load_gpt2(dev), arm).to(dev)
            else:
                m = L.LatentRestart(s5.load_gpt2(dev)).to(dev)
            m.load_state_dict(torch.load(os.path.join(ROOT, "boxes", "runs_s5", path % s, "model.pt"), map_location=dev)); m.eval()
            if arm:
                H = collect_g97(m, S, O, dev); head = None
            else:
                H, head = collect_l86(m, S, O, dev)
            key = path % s; res[key] = {}
            for name, X in variants(H, head).items():
                asg = kmeans(X, 240, 30, np.random.default_rng(7 + s), restarts=2)
                pur, nmi = purity_nmi(asg, Y); res[key][name] = {"purity": pur, "nmi": nmi}
            print(key, {k: round(v["purity"], 3) for k, v in res[key].items()}, flush=True)
            del m; torch.cuda.empty_cache() if dev.startswith("cuda") else None
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "s101_manifold_probe.json"), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
