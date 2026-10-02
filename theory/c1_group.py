"""Premise check for the "beat SOTA" target 1 (not registered; descriptive): does the §67 discrete-latent learner acquire
larger groups' word problems from final answers? Generalises theory/c1_diag.py from S3 to any group from st_model.

Group G (S3, S4, A5, S5): operations uniform over G's elements; the state starts at the identity; the answer is the final
group element (|G|-way). Cell: bilinear (second-order) next-code logits out(ReLU((U code) * (V op))) over M codes
(over-complete, M = ratio * |G|), softmax temperature annealed tau0 -> 1 over the first half; decoder M -> |G|. Exact
forward marginalisation, curriculum lengths {1},{1,2},{1,2,4},{1,2,4,8}. Hard execution; product-automaton certificate
over reachable (code, true element) pairs from (0, identity).
    python theory/c1_group.py --group A5 --seeds 0 1 2 [--ratio 2] [--updates 8000] [--device cpu]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from st_model import group_elems  # noqa: E402


def comp_table(group):
    n, G = group_elems(group)
    idx = {g: i for i, g in enumerate(G)}
    # state = a group element s (as a permutation); applying op a gives a o s (same convention as st_model.sample)
    return np.array([[idx[tuple(np.array(G[a])[np.array(G[s])])] for a in range(len(G))] for s in range(len(G))]), \
        idx[tuple(range(n))], len(G)


class Bil(nn.Module):
    def __init__(self, M, A, hid=256):
        super().__init__()
        self.M, self.A = M, A
        self.U, self.V, self.out = nn.Linear(M, hid), nn.Linear(A, hid), nn.Linear(hid, M)
        self.D = nn.Parameter(torch.randn(M, A) * 0.1)

    def table(self):
        h = torch.relu((self.U.weight.T + self.U.bias)[:, None, :] * (self.V.weight.T + self.V.bias)[None, :, :])  # (M, A, hid)
        return self.out(h)                                                       # (M, A, M)


def run(group, seed, updates, lr, ratio, tau0, dev, bs=512, curriculum="std", hid=256):
    COMP, e, A = comp_table(group)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    M = int(ratio * A)
    m = Bil(M, A, hid).to(dev); opt = torch.optim.Adam(m.parameters(), lr=lr)
    stages = [[1], [1, 2], [1, 2, 4], [1, 2, 4, 8]]
    Ct = torch.as_tensor(COMP, device=dev)
    for u in range(updates):
        if curriculum == "zhang":                     # operator-first: length-2 only for 32 %, then <= 4 to 48 %, then <= 8
            f = u / updates
            L = 2 if f < 0.32 else int(rng.integers(1, 5)) if f < 0.48 else int(rng.integers(1, 9))
        else:
            L = int(rng.choice(stages[min(3, 4 * u // updates)]))
        ops = torch.as_tensor(rng.integers(0, A, (bs, L)), device=dev)
        s = torch.full((bs,), e, device=dev, dtype=torch.long)
        for t in range(L):
            s = Ct[s, ops[:, t]]
        tau = 1.0 + (tau0 - 1.0) * max(0.0, 1.0 - u / (0.5 * updates))
        T = F.softmax(m.table() / tau, -1)
        alpha = torch.zeros(bs, M, device=dev); alpha[:, 0] = 1.0
        for t in range(L):
            alpha = torch.einsum("bi,bij->bj", alpha, T[:, ops[:, t]].permute(1, 0, 2))
        p = (alpha * F.softmax(m.D, -1)[:, s].T).sum(1)
        loss = -torch.log(p.clamp_min(1e-12)).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Th = m.table().argmax(-1).cpu().numpy(); Dh = m.D.argmax(-1).cpu().numpy()
    seen, frontier, bad = set(), [(0, e)], 0
    while frontier:
        c, st = frontier.pop()
        for a in range(A):
            nxt = (int(Th[c, a]), int(COMP[st, a]))
            if nxt not in seen:
                seen.add(nxt); frontier.append(nxt); bad += int(Dh[nxt[0]] != nxt[1])
    acc = {}
    for N in (8, 9, 32, 128):
        ops = rng.integers(0, A, (2000, N)); c = np.zeros(2000, int); st = np.full(2000, e)
        for t in range(N):
            c = Th[c, ops[:, t]]; st = COMP[st, ops[:, t]]
        acc[N] = float((Dh[c] == st).mean())
    return {"group": group, "M": M, "loss": loss.item(), "reachable_pairs": len(seen), "bad_pairs": bad,
            "certified": bad == 0, "acc": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="A5"); ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--updates", type=int, default=8000); ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--ratio", type=float, default=2.0); ap.add_argument("--tau0", type=float, default=4.0)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--out", default=None)
    ap.add_argument("--curriculum", default="std", choices=["std", "zhang"]); ap.add_argument("--hid", type=int, default=256)
    a = ap.parse_args(); torch.set_num_threads(3)
    res = {}
    for s in a.seeds:
        r = run(a.group, s, a.updates, a.lr, a.ratio, a.tau0, a.device, curriculum=a.curriculum, hid=a.hid); res[s] = r
        print(a.group, "seed", s, "M", r["M"], "loss %.3f" % r["loss"], "certified", r["certified"], "bad", r["bad_pairs"],
              "acc", {k: round(v, 3) for k, v in r["acc"].items()}, flush=True)
    if a.out:
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
