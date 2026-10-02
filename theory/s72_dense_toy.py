"""THEORY §72 follow-up premise (descriptive): is the §72 GPU collapse an input-scale x learning-rate artefact of the
multiplicative head, independent of GPT-2?

"Dense toy": the §72 interface with GPT-2 replaced by the identity. Trainable code embeddings e_c in R^768 (init N(0,
0.14^2), as `distinct_codes`), fixed random op embeddings f_a (N(0, 0.14^2), standing in for wte rows) and a fixed answer
marker f_ans.
  h_code = e_c, h_op = e_c + f_a (the op position sees code + op), h_dec = e_c + f_ans
  next-code logits = W ReLU((U LN(h_code)) * (V LN(h_op))) (hid 256), answer logits = A LN(h_dec)
Exact marginalisation, S3, M 24, tau 4 -> 1 over the first half, std curriculum, 8,000 updates, bs 512, Adam at --lr for
every trainable parameter (the §72 setting was 1e-2). Product-automaton certificate (c1_rl.certify).
    python theory/s72_dense_toy.py --lr 1e-2 --seeds 990 991 992 [--onehot]   (--onehot: one-hot inputs, i.e. the TOY arm)
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
from c1_group import comp_table  # noqa: E402
from c1_rl import certify, length  # noqa: E402


class Dense(nn.Module):
    def __init__(self, M, A, d=768, hid=256, onehot=False):
        super().__init__()
        self.M, self.A, self.onehot = M, A, onehot
        if onehot:
            d = M + A
            self.register_buffer("Ec", torch.eye(d)[:M]); self.register_buffer("Fa", torch.eye(d)[M:])
            self.register_buffer("fans", torch.zeros(d))
        else:
            self.Ec = nn.Parameter(0.14 * torch.randn(M, d))
            self.register_buffer("Fa", 0.14 * torch.randn(A, d)); self.register_buffer("fans", 0.14 * torch.randn(d))
        self.ln_c, self.ln_o, self.ln_d = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.U, self.V, self.W = nn.Linear(d, hid), nn.Linear(d, hid), nn.Linear(hid, M)
        self.Ahead = nn.Linear(d, A)

    def table(self):                                                     # (M, A, M) next-code logits
        hc = self.Ec[:, None, :].expand(self.M, self.A, -1)
        ho = self.Ec[:, None, :] + self.Fa[None, :, :]
        return self.W(torch.relu(self.U(self.ln_c(hc)) * self.V(self.ln_o(ho))))

    @property
    def D(self):                                                         # (M, A) answer logits (certify uses .D)
        return self.Ahead(self.ln_d(self.Ec + self.fans))


def run(seed, lr, onehot, updates=8000, bs=512, M=24):
    COMP, e, A = comp_table("S3"); Ct = torch.as_tensor(COMP)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = Dense(M, A, onehot=onehot); opt = torch.optim.Adam(m.parameters(), lr=lr); trace = {}
    for u in range(updates):
        L = length(rng, u, updates, "std")
        ops = torch.as_tensor(rng.integers(0, A, (bs, L))); s = torch.full((bs,), e, dtype=torch.long)
        for t in range(L):
            s = Ct[s, ops[:, t]]
        tau = 1.0 + 3.0 * max(0.0, 1.0 - u / (0.5 * updates))
        logT = F.log_softmax(m.table() / tau, -1); logD = F.log_softmax(m.D, -1)
        la = torch.full((bs, M), -1e30); la[:, 0] = 0.0
        for t in range(L):
            la = torch.logsumexp(la[:, :, None] + logT[:, ops[:, t], :].permute(1, 0, 2), 1)
        loss = -torch.logsumexp(la + logD[:, s].T, 1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (u + 1) % 1000 == 0:
            with torch.no_grad():
                pre = m.U(m.ln_c(m.Ec[:, None, :].expand(M, A, -1))) * m.V(m.ln_o(m.Ec[:, None, :] + m.Fa[None]))
                trace[u + 1] = {"loss": round(float(loss), 4), "L": L, "dead_frac": round(float((pre <= 0).all(0).all(0).float().mean()), 3)}
    cert, bad, acc = certify(m, COMP, e, np.random.default_rng(seed + 7))
    return {"certified": bool(cert), "bad_pairs": bad, "acc": acc, "trace": trace}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-2); ap.add_argument("--seeds", type=int, nargs="*", default=[990, 991, 992])
    ap.add_argument("--onehot", action="store_true"); ap.add_argument("--out")
    a = ap.parse_args(); torch.set_num_threads(2); res = {}
    for s in a.seeds:
        r = run(s, a.lr, a.onehot); res[s] = r
        print("dense" if not a.onehot else "onehot", "lr", a.lr, "seed", s, "certified", r["certified"], "bad", r["bad_pairs"],
              "acc128", round(r["acc"][128], 3), "trace", {k: (v["loss"], v["dead_frac"]) for k, v in r["trace"].items()}, flush=True)
    if a.out:
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
