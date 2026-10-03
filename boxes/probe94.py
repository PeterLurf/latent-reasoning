"""THEORY §94 Step 0 (inference only): is the "where" readout a reversal problem? Linear probes on §89's FADE models.
For each model: 4,000 held-out training-distribution problems (N = 8, both question types, generator seed 9400 + seed;
disjoint from §66's training seeds and test seed 6600), run in pure latent (alpha = 1) with the thought inputs collected
at all 40 thought positions. Probes (linear, on per-coordinate standardised latents; Adam lr 1e-2, 300 full-batch
steps; train 3,000 / test 1,000):
  (i)  letter -> box: for each of the 4 letters, its box index (4-way) in the target arrangement;
  (ii) box -> letter: for each box, its letter (4-way) (positive control).
Targets: the final latent (position 40) uses the final arrangement; thought position j (1-based) uses the arrangement
after op ceil(j / 5). Reported: per-direction accuracy (mean over the 4 letters / boxes and all-4-correct) at the final
latent and at the last position of every op block (positions 5, 10, ..., 40).
    python -m boxes.probe94 --run boxes/runs/f89_fade_std_dec_s1 --device cpu --out theory/results/s94_step0/s1.json
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes.core import LET, apply, make_problem  # noqa: E402
from boxes.fade89 import Fade  # noqa: E402
from boxes.train import load_model  # noqa: E402


def probe(X, Y, n_train, steps=300):
    """X (n, D), Y (n, 4) 4-way targets per slot -> (mean slot accuracy, all-4 accuracy) on the test part."""
    mu, sd = X[:n_train].mean(0), X[:n_train].std(0) + 1e-5; Z = (X - mu) / sd
    lin = nn.Linear(Z.shape[1], 16); op = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for _ in range(steps):
        lo = F.cross_entropy(lin(Z[:n_train]).view(-1, 4), Y[:n_train].reshape(-1)); op.zero_grad(); lo.backward(); op.step()
    with torch.no_grad():
        p = lin(Z[n_train:]).view(-1, 4, 4).argmax(-1); c = p == Y[n_train:]
    return float(c.float().mean()), float(c.all(-1).float().mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=4000); ap.add_argument("--threads", type=int, default=3); ap.add_argument("--out", required=True)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    args = json.load(open(os.path.join(a.run, "args.json")))
    f = Fade(load_model(os.path.join(ROOT, "prosqa", "ckpt", "gpt2_pretrained.bin"), a.device), args["arm"])
    f.load_state_dict(torch.load(os.path.join(a.run, "model.pt"), map_location=a.device)); f.to(a.device).eval()
    rng = random.Random(9400 + args["seed"]); probs = [make_problem(rng, 8, rng.choice(["all", "where"])) for _ in range(a.n)]
    lats = []
    with torch.no_grad():
        for i in range(0, a.n, 50):
            b = probs[i:i + 50]
            for qt in ("all", "where"):                                    # batches need one question type
                bb = [p for p in b if p["qtype"] == qt]
                if not bb:
                    continue
                coll = []; f(bb, a.device, 1.0, stage=None, train=False, collect=coll)
                lats.append((bb, torch.stack(coll, 1)))                   # (B, 40, D)
    P = [p for bb, _ in lats for p in bb]; L = torch.cat([x for _, x in lats]).float()
    n_train = int(0.75 * len(P)); res = {"run": a.run, "arm": args["arm"], "seed": args["seed"], "n": len(P), "positions": {}}
    for pos in list(range(5, 41, 5)):
        op_i = (pos + 4) // 5
        arrs = [apply(p["start"], p["ops"][:op_i]) for p in P]
        box2let = torch.tensor([[LET.index(c) for c in arr] for arr in arrs])                 # (n, 4): letter in each box
        let2box = torch.tensor([[arr.index(c) for c in LET] for arr in arrs])                 # (n, 4): box of each letter
        X = L[:, pos - 1]
        res["positions"][str(pos)] = {"letter_to_box": probe(X, let2box, n_train), "box_to_letter": probe(X, box2let, n_train)}
    res["final"] = res["positions"]["40"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w", newline="\n"), indent=1)
    print(a.run, "final letter->box", res["final"]["letter_to_box"], "box->letter", res["final"]["box_to_letter"], flush=True)


if __name__ == "__main__":
    main()
