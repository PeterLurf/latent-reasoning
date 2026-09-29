"""Exploratory exp9: transfer curve of the local latent map (Mechanism 2: analog vs restoring).

For each model: roll out normally to step K-1 on the long-chain probe; take the input thought of step K, h_{K-1};
scale its deviation from the batch mean by s (h_s = c + s (h - c)); run step K with layer-2 retrieval blocked (the
local map only; cache of positions < K unchanged); read the output thought h_K through the sweep head of the next step:
amp_out(s) = membership logit excess at the depth the unscaled output points to. amp_in(s) = the same measure for the
input thought (step K's own layer-1 logits at the input pointer depth).
Summary: local gain g1 = amp_out(1)/amp_in(1); slope d log amp_out / d log amp_in at s = 1 (1 = linear, > 1 = restoring
towards a threshold, < 1 = saturating); saturation ratio amp_out(2)/amp_out(1).
    python theory/exp9_transfer.py --out theory/results/exp9_transfer.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp1_window import members  # noqa: E402
from rollout import load, probe, rollout  # noqa: E402

S = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


@torch.no_grad()
def measure(m, K, g, br, E, head):
    T = lambda a: torch.as_tensor(a)
    src, dst, start = T(g.src), T(g.dst), T(g.start)
    n = len(g); bi = np.arange(n)[:, None]
    reach = np.concatenate([E["A"][:, 1:], E["B"]], 1); decoy = np.concatenate([E["D"][:, 1:], E["C"]], 1)
    mem = lambda s_: np.nan_to_num(s_[bi, head, reach] - s_[bi, head, decoy]).mean(0)       # (11,)
    out = {}
    for s in S:
        def feed(k, h, hist, s=s):
            if k == K:
                c = h.mean(0, keepdim=True)
                return c + s * (h - c)
            return h
        o = rollout(m, K, src, dst, start, K + 1, feed=feed, block_from=K, block_layers=(1,))
        lin = mem(o["logit1"][K - 1].numpy())            # step K's layer-1 logits: reads the (scaled) input thought
        lout = mem(o["logit1"][K].numpy())               # step K+1's logits: reads the output thought h_K
        out[s] = (lin, lout)
    j_in = int(np.argmax(out[1.0][0])); j_out = int(np.argmax(out[1.0][1]))
    a_in = np.array([out[s][0][j_in] for s in S]); a_out = np.array([out[s][1][j_out] for s in S])
    i1 = S.index(1.0)
    lo, hi = S.index(0.75), S.index(1.25)
    slope = float((np.log(max(a_out[hi], 1e-3)) - np.log(max(a_out[lo], 1e-3))) / (np.log(max(a_in[hi], 1e-3)) - np.log(max(a_in[lo], 1e-3))))
    return {"j_in": j_in + 1, "j_out": j_out + 1, "a_in": a_in.tolist(), "a_out": a_out.tolist(),
            "gain": float(a_out[i1] / max(a_in[i1], 1e-3)), "slope": slope, "sat": float(a_out[-1] / max(a_out[i1], 1e-3)),
            "low": float(a_out[0] / max(a_out[i1], 1e-3))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp9_transfer.json")
    a = ap.parse_args()
    rows = members()
    Mj = json.load(open("theory/results/exp2_membership.json"))
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, br, E = probe(256)
    for name in sorted(rows, key=lambda r: (rows[r]["extrap"], rows[r]["P7"])):
        if name in res:
            continue
        m, c = load("runs/" + name, "cpu")
        o = measure(m, c["K"], g, br, E, Mj[name]["heads"][0])
        o["extrap"] = rows[name]["extrap"]
        res[name] = o
        json.dump(res, open(a.out, "w"))
        print("%-9s %s in d%d -> out d%d gain %.2f slope %.2f sat %.2f low %.2f | a_out %s" % (
            name, "EXT" if o["extrap"] else "   ", o["j_in"], o["j_out"], o["gain"], o["slope"], o["sat"], o["low"],
            " ".join("%.1f" % x for x in o["a_out"])), flush=True)


if __name__ == "__main__":
    main()
