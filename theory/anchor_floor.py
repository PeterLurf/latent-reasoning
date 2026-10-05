"""THEORY anchor-cosine floor ($0, CPU, descriptive; Audit / claude-13's check after 122-2). Is the anchor's cosine to the
text path state-specific, or near the floor set by GPT-2's anisotropy?
For hand-off j = 2: the latent-path carried latent z (the chain from the start) and the text-path carried latent b (the
same segment fed the gold step-1 / state-1 text instead of the latent). Reported over N problems:
  raw_own      mean cos(z_i, b_i)                 (what the anchor reports)
  raw_cross    mean cos(b_i, b_k), i != k         (the floor between different problems' text codes)
  ctr_own      mean cos(z_i - mean z, b_i - mean b)
  ctr_cross    mean cos(b_i - mean b, b_k - mean b), i != k
  nn_match     share of i whose nearest b_k (centred cosine) is b_i itself (state-specific binding)
Models: §122 a122_textmixanchor_4x4_s1 (4x4 multiplication, accel99 Markov) and §118 s118_anchor_s71 (S5, mk108).
Output theory/results/anchor_floor.json.
    python theory/anchor_floor.py --n 400
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BOXES_NB", "5")
from boxes import s5  # noqa: E402


def stats(Z, B):
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True); Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
    n = len(Z); off = ~np.eye(n, dtype=bool)
    raw_own = float((Z * Bn).sum(1).mean()); raw_cross = float((Bn @ Bn.T)[off].mean())
    Zc = Z - Z.mean(0); Bc = B - B.mean(0); Zc /= np.linalg.norm(Zc, axis=1, keepdims=True); Bc /= np.linalg.norm(Bc, axis=1, keepdims=True)
    ctr_own = float((Zc * Bc).sum(1).mean()); C = Bc @ Bc.T; ctr_cross = float(C[off].mean())
    nn = float(((Zc @ Bc.T).argmax(1) == np.arange(n)).mean())
    return {"raw_own": raw_own, "raw_cross": raw_cross, "ctr_own": ctr_own, "ctr_cross": ctr_cross, "nn_match": nn}


@torch.no_grad()
def mult(n):
    from boxes.accel99 import MultGen99
    from boxes.mult98 import DIGIT_IDS, TOK, load
    m = MultGen99(s5.load_gpt2("cpu"), "genm", std=False); m.markov = True
    m.load_state_dict(torch.load(os.path.join(ROOT, "boxes", "runs_mult", "a122_textmixanchor_4x4_s1", "model.pt"), map_location="cpu")); m.eval()
    D = load(os.path.join(ROOT, "bench", "data", "mult_4x4_test.txt")); idx = np.arange(n); g = m.g; T = torch.tensor
    prefix = T(np.concatenate([DIGIT_IDS[D["A"][idx]], np.full((n, 1), TOK["*"])], 1)); B = D["B"][idx]
    seg = lambda parts: g(torch.cat(parts, 1), None)[0][:, -1]  # noqa: E731
    op = lambda j: g.embed(m.digtok[T(B[:, j])])[:, None]  # noqa: E731
    z1 = seg([g.embed(prefix), op(0)]); z2 = seg([g.embed(prefix), z1[:, None], op(1)])
    b2 = seg([g.embed(prefix), g.embed(T(D["steps"][0][idx])), op(1)])
    return stats(z2.float().numpy(), b2.float().numpy())


@torch.no_grad()
def s5chain(n):
    from boxes.gen97 import STTOK
    from boxes.mk108 import MK108
    from boxes.snap115 import seg
    m = MK108(s5.load_gpt2("cpu")); sd = torch.load(os.path.join(ROOT, "boxes", "runs_s5", "s118_anchor_s71", "model.pt"), map_location="cpu")
    m.load_state_dict(sd, strict=False); m.eval(); g = m.g
    rng = np.random.default_rng(12201); S0 = rng.integers(0, 120, n); O = rng.integers(0, 120, (n, 2)); S1 = s5.COMP[S0, O[:, 0]]
    z1, _ = seg(m, g.embed(torch.tensor(STTOK[S0])), torch.tensor(O[:, 0])); z2, _ = seg(m, z1[:, None], torch.tensor(O[:, 1]))
    b2, _ = seg(m, g.embed(torch.tensor(STTOK[S1])), torch.tensor(O[:, 1]))
    return stats(z2.float().numpy(), b2.float().numpy())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=400); ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    res = {"n": a.n, "mult_a122_s1_j2": mult(a.n), "s5_s118_anchor_s71_j2": s5chain(a.n)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "anchor_floor.json"), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
