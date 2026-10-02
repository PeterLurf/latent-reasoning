"""THEORY §67 diagnostic (Audit's request; not registered, descriptive): does a SHARED-parameter cell learn S3's 36-entry
transition table under the exact-marginalisation objective, or only a free table? An independent minimal implementation
(CPU), cross-checking c1/.

Codes M = 6, initial code 0. Cell: next-code distribution T[c, a] = softmax(f(c, a)), with f either a free table (tab)
or an MLP over [one-hot(c), one-hot(a)] -> hidden -> M (mlp). Decoder D[c] = softmax(table) over the 6 answers. Exact
forward marginalisation alpha_t = alpha_{t-1} T[:, a_t]; loss = -log sum_c alpha_N(c) D[c, y]. Curriculum (as §67): lengths
{1}, {1,2}, {1,2,4}, {1,2,4,8} in 4 equal stages. Hard execution: argmax T and argmax D; accuracy at N 8..128 by exact
table rollout (equivalent to hard execution of a stationary deterministic cell); certificate: a product-automaton search
over (code, true state) pairs reachable from (0, identity); certified iff every pair reached after >= 1 operation decodes to
its true state ("errs" = the number of reachable pairs that do not; "bij" = no code is reached with two true states).
    python theory/c1_diag.py [--cells tab mlp] [--seeds 0 1 2] [--updates 2000 8000] [--lrs 1e-2 1e-3]
"""
import argparse
import itertools
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

G = [tuple(p) for p in itertools.permutations(range(3))]
COMP = np.array([[G.index(tuple(np.array(G[a])[np.array(G[s])])) for a in range(6)] for s in range(6)])  # state s, op a -> s'


class Cell(nn.Module):
    def __init__(self, kind, M=6, hid=256):
        super().__init__()
        self.kind, self.M = kind, M
        if kind == "tab":
            self.T = nn.Parameter(torch.randn(M, 6, M) * 0.1)
        else:
            self.f = nn.Sequential(nn.Linear(M + 6, hid), nn.ReLU(), nn.Linear(hid, M))
        self.D = nn.Parameter(torch.randn(M, 6) * 0.1)
        self.X = torch.cat([torch.eye(M).repeat_interleave(6, 0), torch.eye(6).repeat(M, 1)], 1)   # rows (c, a)

    def table(self):
        return self.T if self.kind == "tab" else self.f(self.X).view(self.M, 6, self.M)


def run(kind, seed, updates, lr, bs=512, M=6, tau0=1.0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = Cell(kind, M); opt = torch.optim.Adam(m.parameters(), lr=lr)
    stages = [[1], [1, 2], [1, 2, 4], [1, 2, 4, 8]]
    for u in range(updates):
        lens = stages[min(3, 4 * u // updates)]
        L = int(rng.choice(lens))
        ops = torch.as_tensor(rng.integers(0, 6, (bs, L)))
        y = torch.zeros(bs, dtype=torch.long)
        s = np.zeros(bs, int)
        for t in range(L):
            s = COMP[s, ops[:, t].numpy()]
        y = torch.as_tensor(s)
        tau = 1.0 + (tau0 - 1.0) * max(0.0, 1.0 - u / (0.5 * updates))   # annealed temperature: tau0 -> 1 by halfway
        T = F.softmax(m.table() / tau, -1)                      # (M, 6, M)
        alpha = torch.zeros(bs, m.M); alpha[:, 0] = 1.0
        for t in range(L):
            alpha = torch.einsum("bi,bij->bj", alpha, T[:, ops[:, t]].permute(1, 0, 2))
        p = (alpha * F.softmax(m.D, -1)[:, y].T).sum(1)
        loss = -torch.log(p.clamp_min(1e-12)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Th = m.table().argmax(-1).numpy(); Dh = m.D.argmax(-1).numpy()
    # certificate: product-automaton search over (code, true state) pairs reachable from (0, identity); certified iff every
    # pair reached after >= 1 operation decodes to its true state (exact for every length >= 1)
    seen, frontier, bad = set(), [(0, 0)], 0
    while frontier:
        c, st = frontier.pop()
        for op in range(6):
            nxt = (int(Th[c, op]), int(COMP[st, op]))
            if nxt not in seen:
                seen.add(nxt); frontier.append(nxt); bad += int(Dh[nxt[0]] != nxt[1])
    codes = {c for c, _ in seen}
    ambiguous = sum(len({st for c2, st in seen if c2 == c}) > 1 for c in codes)
    errs, bij = bad, ambiguous == 0
    reach = codes
    acc = {}
    for N in (8, 9, 32, 33, 127, 128):
        ops = rng.integers(0, 6, (2000, N)); c = np.zeros(2000, int); s = np.zeros(2000, int)
        for t in range(N):
            c = Th[c, ops[:, t]]; s = COMP[s, ops[:, t]]
        acc[N] = float((Dh[c] == s).mean())
    return {"loss": loss.item(), "reach": len(reach), "bijection": bool(bij), "transition_errors": int(errs),
            "certified": bool(errs == 0), "acc": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=["tab", "mlp"]); ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--updates", type=int, nargs="*", default=[2000, 8000]); ap.add_argument("--lrs", type=float, nargs="*", default=[1e-2, 1e-3])
    ap.add_argument("--Ms", type=int, nargs="*", default=[6]); ap.add_argument("--out", default="theory/results/c1_diag.json")
    ap.add_argument("--tau0s", type=float, nargs="*", default=[1.0])
    a = ap.parse_args(); torch.set_num_threads(3)
    res = {}
    for M, kind, tau0, lr, upd, seed in itertools.product(a.Ms, a.cells, a.tau0s, a.lrs, a.updates, a.seeds):
        r = run(kind, seed, upd, lr, M=M, tau0=tau0); key = f"M{M}_{kind}_tau{tau0:g}_lr{lr:g}_u{upd}_s{seed}"; res[key] = r
        print("%-24s loss %.3f reach %d bij %s errs %2d certified %s acc N8/9/32/33/127/128 %s" % (
            key, r["loss"], r["reach"], r["bijection"], r["transition_errors"], r["certified"],
            "/".join("%.2f" % v for v in r["acc"].values())), flush=True)
    json.dump(res, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()


def counter_check(seed, M=12, updates=8000, lr=1e-2, n=4000, N=24):
    """Descriptive (Audit): does an M = 12 table's code identity carry length information beyond the S3 state? For each
    true state, the conditional entropy H(code | state) and the mutual information between code and (length mod q),
    q = 2, 3, 4, given the state, from n random sequences observed at every step 1..N."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = Cell("tab", M); opt = torch.optim.Adam(m.parameters(), lr=lr)
    stages = [[1], [1, 2], [1, 2, 4], [1, 2, 4, 8]]
    for u in range(updates):
        L = int(rng.choice(stages[min(3, 4 * u // updates)])); bs = 512
        ops = torch.as_tensor(rng.integers(0, 6, (bs, L))); s = np.zeros(bs, int)
        for t in range(L):
            s = COMP[s, ops[:, t].numpy()]
        T = F.softmax(m.table(), -1); alpha = torch.zeros(bs, M); alpha[:, 0] = 1.0
        for t in range(L):
            alpha = torch.einsum("bi,bij->bj", alpha, T[:, ops[:, t]].permute(1, 0, 2))
        p = (alpha * F.softmax(m.D, -1)[:, torch.as_tensor(s)].T).sum(1)
        loss = -torch.log(p.clamp_min(1e-12)).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Th = m.table().argmax(-1).numpy()
    ops = rng.integers(0, 6, (n, N)); c = np.zeros(n, int); s = np.zeros(n, int); rows = []
    for t in range(1, N + 1):
        c = Th[c, ops[:, t - 1]]; s = COMP[s, ops[:, t - 1]]
        rows.append(np.stack([c, s, np.full(n, t)], 1))
    R = np.concatenate(rows)

    def H(x):
        _, cnt = np.unique(x, axis=0, return_counts=True); p = cnt / cnt.sum(); return float(-(p * np.log2(p)).sum())
    out = {"H_code_given_state": H(R[:, :2]) - H(R[:, 1:2])}
    for q in (2, 3, 4):
        lm = (R[:, 2] % q)[:, None]
        # I(code; len mod q | state) = H(code,state) + H(lenmod,state) - H(code,lenmod,state) - H(state)
        out[f"MI_code_lenmod{q}_given_state"] = (H(R[:, :2]) + H(np.hstack([lm, R[:, 1:2]])) - H(np.hstack([R[:, :2], lm]))
                                                 - H(R[:, 1:2]))
    return out
