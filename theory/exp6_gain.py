"""Exploratory exp6: where does the pointer lose amplitude (Mechanism 2, attenuation)?

Per step k, sweep head (exp2) of layer 1, latent position, graphs averaged, long-chain probes:
  ptr   : pointer depth j* of the thought fed into step k (argmax membership logit) and its logit excess
  front : attention mass on the chain edge(s) out of the pointed node (A/B chain; the frontier read)
  edges : total mass on edge tokens; start: START; lat: earlier latent positions; self
  zlog  : log-sum-exp of logits over non-frontier keys (the competition the frontier logit must beat)
Also per-width pointer logit (W = 2, 4, 6), since beyond distance 6 the frontier is one node for every W.
Interventions (pointer logit at steps K+1..12, sweep head): base; bL1@K (layer 1 cannot see earlier latent positions
from step K on); bL12@K.
    python theory/exp6_gain.py runs/p30_s105 ...
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rollout import load, probe, rollout  # noqa: E402


@torch.no_grad()
def measure(run, head, n=512, Kp=12, **kw):
    m, c = load(run, "cpu")
    K = c["K"]
    g, br, E = probe(n)
    T = lambda a: torch.as_tensor(a)
    out = rollout(m, K, T(g.src), T(g.dst), T(g.start), Kp, **kw)
    P = out["P"]
    bi = np.arange(n)[:, None]
    reach = np.concatenate([E["A"][:, 1:], E["B"]], 1); decoy = np.concatenate([E["D"][:, 1:], E["C"]], 1)
    rows = []
    for k in range(Kp):
        s = out["logit1"][k][:, head].numpy(); a = out["attn1"][k][:, head].numpy()      # (n, T)
        mem = (s[bi, reach] - s[bi, decoy])                                              # (n, 11)
        mm = mem.mean(0); j = int(np.argmax(mm))
        fr = a[bi, reach[:, j:j + 1]][:, 0]
        s_nf = s.copy(); s_nf[bi, reach[:, j:j + 1]] = -np.inf
        s_nf = np.where(np.isnan(s_nf), -np.inf, s_nf)
        zl = np.log(np.exp(s_nf - s_nf.max(1, keepdims=True)).sum(1)) + s_nf.max(1)
        w = g.width
        rows.append({"k": k + 1, "ptr": j + 1, "logit": float(mm[j]),
                     "front": float(fr.mean()), "edges": float(a[:, :P - 1].sum(1).mean()),
                     "start": float(a[:, P - 1].mean()), "lat": float(np.nansum(a[:, P:P + k], 1).mean()) if k else 0.0,
                     "self": float(a[:, P + k].mean()),
                     "front_logit": float(s[bi, reach[:, j:j + 1]][:, 0].mean()), "zlog": float(zl.mean()),
                     "logit_byW": {int(x): float(mem[w == x, j].mean()) for x in (2, 4, 6)}})
    return K, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="theory/results/exp6_gain.json")
    a = ap.parse_args()
    Mj = json.load(open("theory/results/exp2_membership.json"))
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for r in a.runs:
        name = r.rstrip("/").split("/")[-1]
        head = Mj[name]["heads"][0]
        o = {}
        Kn = Mj[name]["K"]
        for tag, kw in (("base", {}), ("bL1@K", {"block_from": Kn, "block_layers": (0,)}),
                        ("bL12@K", {"block_from": Kn, "block_layers": (0, 1)})):
            K, rows = measure(r, head, **kw)
            o[tag] = rows
        res[name] = {"K": K, "head": head, **o}
        json.dump(res, open(a.out, "w"))
        print(f"== {name} K={K} head {head}")
        print(" k ptr logit | front  edges start  lat   self | front_logit zlog | logit W2/W4/W6 | bL1@K logit bL12@K logit")
        for b, x, y in zip(o["base"], o["bL1@K"], o["bL12@K"]):
            print("%2d %2d %5.2f | %.3f %.3f %.3f %.3f %.3f | %6.2f %6.2f | %5.2f %5.2f %5.2f | %5.2f %5.2f" % (
                b["k"], b["ptr"], b["logit"], b["front"], b["edges"], b["start"], b["lat"], b["self"], b["front_logit"],
                b["zlog"], b["logit_byW"][2], b["logit_byW"][4], b["logit_byW"][6], x["logit"], y["logit"]))


if __name__ == "__main__":
    main()
