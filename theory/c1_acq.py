"""THEORY §70: discrete-latent acquisition from final answers, (A) operator-conditioned exact likelihood and (B) tempered
posterior continuation into hard EM / Viterbi (proposals A and B of docs/EXTERNAL_IDEAS_RL_2026-10-01.md).

Machine as in theory/c1_rl.py: M codes, start code 0, transition logits for every (code, op), decoder D (M x |G|), ops
uniform over the group, answer = final element; product-automaton certificate (c1_rl.certify). Transition temperature
tau(u) = 4 -> 1 linearly by update U/2, then 1. Batch 512, Adam lr 1e-2, std curriculum ({1}, {1,2}, {1,2,4}, {1,2,4,8}).
Cells (--cell):
  add   : additive  out(ReLU(W [onehot c; onehot a] + b))                         (hid matched in parameters)
  mul   : multiplicative out(ReLU((U c + b_U) * (V a + b_V)))                     (hid matched in parameters)
  opb   : operator-conditioned L(c, a, j) = e_out(j)^T B_a e_in(c) + b_j, rank r  (--rank; r = M is the full-rank reference)
  mlp   : the historical MLP (c1_rl.Cell "mlp", hid 256), used by the B arms
  For add / mul / opb the initial transition logits are rescaled to std 0.1 (the free table's init scale).
Methods (--method):
  mml   : exact forward marginal likelihood, one update per batch
  mml2  : the same, each batch used for two consecutive updates (gradient recomputed): B's paired control
  exr   : exact expected reward, maximise mean J (J = P(correct answer)), one update per batch
  temper: B. Outer steps of two updates on one batch. u < U/8: mml (twice). Then an E-step from the detached current
          model gives exact posterior counts under q_eta ∝ w^eta (log-space forward, unnormalised powered potentials;
          counts = autograd of log Z_eta): eta 0.5 -> 2 linearly over [U/8, U/2), eta 4 over [U/2, 3U/4), Viterbi
          (max-product; counts = autograd of the max path score, ties to the lowest index) over [3U/4, U). Each of the
          two updates minimises 0.25 (-log J) + 0.75 C, C = cross-entropy of the current T, D against the fixed counts.
  eta1  : temper's structure with eta = 1 throughout and no Viterbi phase (generalised-EM control)
    python theory/c1_acq.py --cell opb --rank 12 --M 24 --method mml --seeds 1300 1301 1302 --threads 1 --out f.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c1_group import comp_table  # noqa: E402
from c1_rl import Cell, certify, length  # noqa: E402


class Add(nn.Module):
    def __init__(self, M, A, hid):
        super().__init__()
        self.M, self.A = M, A
        self.f = nn.Sequential(nn.Linear(M + A, hid), nn.ReLU(), nn.Linear(hid, M))
        self.D = nn.Parameter(torch.randn(M, A) * 0.1)
        self.last = self.f[2]

    def table(self):
        X = torch.cat([torch.eye(self.M).repeat_interleave(self.A, 0), torch.eye(self.A).repeat(self.M, 1)], 1)
        return self.f(X).view(self.M, self.A, self.M)


class Mul(nn.Module):
    def __init__(self, M, A, hid):
        super().__init__()
        self.M, self.A = M, A
        self.U, self.V, self.out = nn.Linear(M, hid), nn.Linear(A, hid), nn.Linear(hid, M)
        self.D = nn.Parameter(torch.randn(M, A) * 0.1)
        self.last = self.out

    def table(self):
        return self.out(torch.relu((self.U.weight.T + self.U.bias)[:, None, :] * (self.V.weight.T + self.V.bias)[None, :, :]))


class OpB(nn.Module):
    def __init__(self, M, A, r):
        super().__init__()
        self.M, self.A = M, A
        self.e_in = nn.Parameter(torch.randn(M, r) / r ** 0.5)
        self.e_out = nn.Parameter(torch.randn(M, r) / r ** 0.5)
        self.B = nn.Parameter(torch.randn(A, r, r) / r ** 0.5)
        self.b = nn.Parameter(torch.zeros(M))
        self.D = nn.Parameter(torch.randn(M, A) * 0.1)

    def table(self):                                                             # (M_in, A, M_out)
        return torch.einsum("cr,ars,js->caj", self.e_in, self.B, self.e_out) + self.b


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p is not m.D)


def matched_hid(kind, M, A, target):
    """Smallest hid whose transition-parameter count >= target (add: (M+A+1+M) hid + M; mul: (M+1+A+1+M) hid + M)."""
    per = (M + A + 1 + M) if kind == "add" else (M + 1 + A + 1 + M)
    return int(np.ceil((target - M) / per))


def build(cell, M, A, rank):
    if cell == "mlp":
        return Cell("mlp", M, A)
    target = M * rank * 2 + A * rank * rank + M                                   # opb's transition parameters
    m = OpB(M, A, rank) if cell == "opb" else (Add if cell == "add" else Mul)(M, A, matched_hid(cell, M, A, target))
    with torch.no_grad():                                                         # initial logit std -> 0.1
        k = 0.1 / float(m.table().std())
        if cell == "opb":
            m.e_out.mul_(k); m.b.mul_(k)
        else:
            m.last.weight.mul_(k); m.last.bias.mul_(k)
    return m


def forward_logp(logT, logD, ops, s):
    """log P(y | x) by the exact forward pass in log space. logT (M, A, M), logD (M, |G|)."""
    bs, L = ops.shape
    la = torch.full((bs, logT.shape[0]), -1e30); la[:, 0] = 0.0
    for t in range(L):
        la = torch.logsumexp(la[:, :, None] + logT[:, ops[:, t], :].permute(1, 0, 2), 1)
    return torch.logsumexp(la + logD[:, s].T, 1)


def posterior_counts(logT, logD, ops, s, eta, viterbi):
    """Edge counts xi (bs, L, M, M) and terminal counts gam (bs, M) under q ∝ w^eta (or the Viterbi path)."""
    bs, L = ops.shape
    phi = (eta * logT[:, ops, :]).permute(1, 2, 0, 3).detach().requires_grad_(True)   # (bs, L, M_in, M_out)
    psi = (eta * logD[:, s].T).detach().requires_grad_(True)
    la = torch.full((bs, logT.shape[0]), -1e30); la[:, 0] = 0.0
    for t in range(L):
        x = la[:, :, None] + phi[:, t]
        la = x.max(1).values if viterbi else torch.logsumexp(x, 1)
    z = (la + psi).max(1).values if viterbi else torch.logsumexp(la + psi, 1)
    xi, gam = torch.autograd.grad(z.sum(), [phi, psi])
    return xi, gam


def train(group, cell, method, seed, M, rank, updates, lr, bs):
    COMP, e, A = comp_table(group)
    Ct = torch.as_tensor(COMP)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = build(cell, M, A, rank); opt = torch.optim.Adam(m.parameters(), lr=lr)
    tau = lambda u: 1.0 + 3.0 * max(0.0, 1.0 - u / (0.5 * updates))              # noqa: E731
    u, ent_trace = 0, {}
    while u < updates:
        L = length(rng, u, updates, "std")
        ops = torch.as_tensor(rng.integers(0, A, (bs, L)))
        s = torch.full((bs,), e, dtype=torch.long)
        for t in range(L):
            s = Ct[s, ops[:, t]]
        reps = 2 if method in ("mml2", "temper", "eta1") else 1
        counts = None
        if method in ("temper", "eta1") and u >= updates // 8:
            with torch.no_grad():
                lg = m.table() / tau(u)
            logT_old, logD_old = F.log_softmax(lg, -1), F.log_softmax(m.D.detach(), -1)
            if method == "eta1":
                eta, vit = 1.0, False
            elif u < updates // 2:
                eta, vit = 0.5 + 1.5 * (u - updates // 8) / (updates // 2 - updates // 8), False
            elif u < 3 * updates // 4:
                eta, vit = 4.0, False
            else:
                eta, vit = 1.0, True
            counts = posterior_counts(logT_old, logD_old, ops, s, eta, vit)
        for _ in range(reps):
            if u >= updates:
                break
            logT, logD = F.log_softmax(m.table() / tau(u), -1), F.log_softmax(m.D, -1)
            lp = forward_logp(logT, logD, ops, s)
            if method == "exr":
                loss = -lp.exp().mean()
            elif counts is None:
                loss = -lp.mean()
            else:
                xi, gam = counts
                C = -(xi * logT[:, ops, :].permute(1, 2, 0, 3)).sum((1, 2, 3)) - (gam * logD[:, s].T).sum(1)
                loss = 0.25 * (-lp.mean()) + 0.75 * C.mean()
            opt.zero_grad(); loss.backward(); opt.step(); u += 1
            if u % 500 == 0:                                                      # mean transition-row entropy (nats)
                with torch.no_grad():
                    P = F.softmax(m.table() / tau(u), -1); ent_trace[u] = round(float(-(P * P.clamp_min(1e-12).log()).sum(-1).mean()), 4)
    cert, bad, acc = certify(m, COMP, e, np.random.default_rng(seed + 7))
    with torch.no_grad():                                                         # length-stratified soft NLL, fixed panel
        logT, logD = F.log_softmax(m.table(), -1), F.log_softmax(m.D, -1); er = np.random.default_rng(seed + 11); nll = {}
        for N in (1, 2, 4, 8):
            ops = torch.as_tensor(er.integers(0, A, (1000, N))); s = torch.full((1000,), e, dtype=torch.long)
            for t in range(N):
                s = Ct[s, ops[:, t]]
            nll[N] = float(-forward_logp(logT, logD, ops, s).mean())
    return {"certified": bool(cert), "bad_pairs": bad, "acc": acc, "nll": nll, "params": n_params(m), "ent_trace": ent_trace}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S3"); ap.add_argument("--cell", default="opb", choices=["add", "mul", "opb", "mlp"])
    ap.add_argument("--method", default="mml", choices=["mml", "mml2", "exr", "temper", "eta1"])
    ap.add_argument("--M", type=int, default=24); ap.add_argument("--rank", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="*", default=[9999]); ap.add_argument("--updates", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-2); ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--threads", type=int, default=1); ap.add_argument("--out")
    a = ap.parse_args(); torch.set_num_threads(a.threads); res = {}
    for sd in a.seeds:
        t0, p0 = time.time(), time.process_time()
        r = train(a.group, a.cell, a.method, sd, a.M, a.rank, a.updates, a.lr, a.bs)
        r.update(cpu_sec=round(time.time() - t0, 1), proc_sec=round(time.process_time() - p0, 1)); res[sd] = r
        print(a.group, a.cell, a.method, "M", a.M, "r", a.rank, "seed", sd, "params", r["params"], "certified", r["certified"],
              "bad", r["bad_pairs"], "acc", {k: round(v, 3) for k, v in r["acc"].items()},
              "nll", {k: round(v, 3) for k, v in r["nll"].items()}, "sec", r["proc_sec"], flush=True)
    if a.out:
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
