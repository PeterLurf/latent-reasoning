"""THEORY note 118-3 (descriptive, $0; Audit's contraction diagnostic): is the Markov hand-off map an attractor around the
state code, and does that separate no-snap success (§111 READ s8) from drift (READ s6 / s7, §108 MK-4)?
Per model (mk108 MK108; READ models load as MK108): N chains from a random start state's text, random ops; the clean
no-snap recurrence (as snap115 (A): [EOT, carried latent, op] at every position, no text after position 0) is run to
position T0 = 4 (in range). There the carried latent z is perturbed: z' = z + eps * ||z|| * u (u a random unit vector;
one u per chain), and clean and perturbed chains are run on the same ops for K = 16 more hand-offs. Reported per eps:
  rho_k     median over chains of ||z'_{T0+k} - z_{T0+k}|| / ||z'_{T0+k-1} - z_{T0+k-1}|| (per-hand-off distance ratio)
  rho_geo   geometric mean of rho_k over k = 1..8
  agree_k   share of chains whose greedy slot decode at T0+k equals the clean chain's decode
  basin     the largest eps with agree_8 >= 0.99 (on the grid)
Prediction (fixed in 118-3 before any run): NOSNAP success <-> contraction (rho_geo < 1 at small eps; s8), drift <->
rho_geo >= 1 or a small basin (s6 / s7; §108).
Output theory/results/contraction_diag.json.
    python theory/contraction_diag.py --device cuda:0 --n 500
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
from boxes.gen97 import STTOK, decode_to_el  # noqa: E402
from boxes.mk108 import MK108  # noqa: E402
from boxes.snap115 import seg  # noqa: E402

RUNS = ["s111_mk4_read_s%d" % s for s in range(6, 11)] + ["s108_mk4_s1", "s108_mk4_s3", "s108_mk4_s5"]
EPS = [0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--runs", nargs="*", default=RUNS); ap.add_argument("--T0", type=int, default=4); ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(); torch.set_num_threads(a.threads); dev = a.device
    rng = np.random.default_rng(11803); S0 = rng.integers(0, 120, a.n); O = rng.integers(0, 120, (a.n, a.T0 + a.K))
    U = rng.standard_normal((a.n, 768)); U /= np.linalg.norm(U, axis=1, keepdims=True); U = torch.tensor(U, dtype=torch.float32, device=dev)
    outp = os.path.join(ROOT, "theory", "results", "contraction_diag.json")
    res = json.load(open(outp)) if os.path.exists(outp) else {}
    for name in a.runs:
        p = os.path.join(ROOT, "boxes", "runs_s5", name, "model.pt")
        if not os.path.exists(p):
            continue
        m = MK108(s5.load_gpt2(dev)).to(dev); sd = torch.load(p, map_location=dev)
        missing, unexpected = m.load_state_dict(sd, strict=False); assert not missing and not unexpected, (missing, unexpected)
        m.eval(); ops = torch.tensor(O, device=dev)
        prev = m.g.embed(torch.tensor(STTOK[S0], device=dev))
        for t in range(a.T0):                                                # clean, in range
            z, _ = seg(m, prev, ops[:, t]); prev = z[:, None]
        z0 = prev[:, 0].float(); clean = [z0]; cdec = []; c = prev
        for k in range(a.K):
            z, d = seg(m, c, ops[:, a.T0 + k]); clean.append(z.float()); cdec.append(decode_to_el(d.cpu().numpy())[0]); c = z[:, None]
        R = {}
        for eps in EPS:
            zp = (z0 + eps * z0.norm(dim=-1, keepdim=True) * U).to(prev.dtype)[:, None]; dist = [(zp[:, 0].float() - z0).norm(dim=-1)]; agree = []
            for k in range(a.K):
                z, d = seg(m, zp, ops[:, a.T0 + k]); zp = z[:, None]
                dist.append((z.float() - clean[k + 1]).norm(dim=-1)); agree.append(float((decode_to_el(d.cpu().numpy())[0] == cdec[k]).mean()))
            D = torch.stack(dist).cpu().numpy()                             # (K + 1, N)
            ratio = D[1:] / np.maximum(D[:-1], 1e-6)
            rho_k = np.median(ratio, axis=1)
            R[str(eps)] = {"rho_k": rho_k.round(4).tolist(), "rho_geo_1_8": float(np.exp(np.log(np.maximum(rho_k[:8], 1e-6)).mean())),
                           "agree_k": [round(x, 4) for x in agree], "dist_rel_k": (np.median(D, axis=1) / float(z0.norm(dim=-1).median())).round(4).tolist()}
        kb = min(7, a.K - 1); basin = max([e for e in EPS if R[str(e)]["agree_k"][kb] >= 0.99], default=0.0)
        res[name] = {"T0": a.T0, "K": a.K, "n": a.n, "eps": R, "basin_eps_agree8": basin}
        print(name, "basin", basin, {e: round(R[str(e)]["rho_geo_1_8"], 3) for e in ("0.01", "0.1", "0.3")}, flush=True)
        json.dump(res, open(outp, "w", newline="\n"), indent=1)
        del m
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
