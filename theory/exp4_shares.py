"""Exploratory exp4: share of each thought's pointer written locally (MLPs at the current step) vs retrieved from
earlier latent positions (attention to lagged latent positions), for every step, all P30 / L competent models.

Uses exp3's exact component split (LayerNorms linearised at the actual input). For step k the pointer depth j* is
the argmax of the total membership logit (sweep head); each component's contribution at j* is recorded.
Groups: local = mlp1 + mlp2; retr = attention (either layer) to earlier latent positions; carry = emb (the input
thought itself, via the residual); edges = attention to edge tokens; other = START, self, biases.
    python theory/exp4_shares.py --out theory/results/exp4_shares.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp3_attrib import attn_parts, ln_lin  # noqa: E402
from rollout import load, probe  # noqa: E402
from latent.model import T_LAT  # noqa: E402


def group(name):
    if name.startswith("mlp"):
        return "local"
    if ".lat-" in name:
        return "retr"
    if name == "emb":
        return "carry"
    if name.endswith(".edges"):
        return "edges"
    return "other"


@torch.no_grad()
def shares(m, K, g, br, E, head, Kp=10):
    T = lambda a: torch.as_tensor(a)
    x0 = m.embed_prefix(T(g.src), T(g.dst), T(g.start))
    P = x0.shape[1]
    h, cache, _ = m.encode(x0)
    te = m.type_emb.weight
    n = len(g); bi = np.arange(n)[:, None]
    reach = np.concatenate([E["A"][:, 1:], E["B"]], 1); decoy = np.concatenate([E["D"][:, 1:], E["C"]], 1)
    blk0 = m.blocks[0].attn; dh = blk0.dh
    Wq = blk0.qkv.weight[head * dh:(head + 1) * dh]
    keys = cache[0][0][:, head, :P - 1]
    kdiff = keys[bi, reach] - keys[bi, decoy]                         # (B, 11, dh)
    rows = []
    for k in range(1, Kp + 1):
        emb = (h + te[T_LAT] + (m.step_emb.weight[min(k, K)] if getattr(m, "use_step", True) else 0))[:, None]
        comps = {"emb": emb}
        xx = emb; new = []
        for li, (blk, kv) in enumerate(zip(m.blocks, cache)):
            parts, kv2 = attn_parts(blk.attn, blk.ln1(xx), kv, P, k - 1)
            for n_, c in parts.items():
                comps[f"a{li+1}.{n_}"] = c
            xx = xx + sum(parts.values())
            mo = blk.mlp(blk.ln2(xx)); comps[f"mlp{li+1}"] = mo
            xx = xx + mo; new.append(kv2)
        cache = new
        h = m._thought(xx[:, 0])
        f1 = ln_lin(m.ln_f, xx[:, 0]); f2 = ln_lin(m.blocks[0].ln1, (h[:, None] + te[T_LAT])[:, 0])
        prof = {}
        for name, c in comps.items():
            qc = f2(f1(c[:, 0])) @ Wq.T
            prof[name] = ((kdiff @ qc[:, :, None])[..., 0].mean(0).numpy() / math.sqrt(dh))
        total = sum(prof.values())
        j = int(np.argmax(total))
        grp = {}
        for name, v in prof.items():
            grp[group(name)] = grp.get(group(name), 0.0) + float(v[j])
        lag = {name: float(v[j]) for name, v in prof.items() if ".lat-" in name}
        rows.append({"k": k, "ptr": j + 1, "amp": float(total[j]), "groups": grp,
                     "top_lag": max(lag, key=lambda x: abs(lag[x])) if lag else None})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp4_shares.json")
    ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    Mj = json.load(open("theory/results/exp2_membership.json"))
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, br, E = probe(a.n)
    for name in Mj:
        if name in res:
            continue
        m, c = load("runs/" + name, "cpu")
        res[name] = {"K": c["K"], "head": Mj[name]["heads"][0], "rows": shares(m, c["K"], g, br, E, Mj[name]["heads"][0])}
        json.dump(res, open(a.out, "w"))
        print(name, " ".join("%d:%s%.0f/%.0f" % (r["ptr"], "", 100 * r["groups"].get("local", 0) / max(abs(r["amp"]), 1e-6),
                                                   100 * r["groups"].get("retr", 0) / max(abs(r["amp"]), 1e-6)) for r in res[name]["rows"]), flush=True)


if __name__ == "__main__":
    main()
