"""THEORY §69: RL-style training for discrete-latent acquisition from outcome-only reward (user request via Audit).

Same machine as theory/c1_group.py: M codes, start code 0, a cell gives next-code logits for every (code, op) (cells: tab,
mlp, bil), a decoder D gives answer logits per code; ops uniform over the group G (S3/S4/A5); the answer is the final
element. Methods (all trained from the final answer only):
  marg : exact forward marginalisation (the §67 objective; baseline)
  hem  : hard-EM / expert iteration (STaR-like): sample K code paths per problem from softmax(logits / tau), sample an
         answer from D; maximise the log-probability of the paths whose answer is correct (others ignored)
  grpo : REINFORCE with a group baseline: reward 1 if the sampled answer is correct; advantage = reward - group mean;
         loss = -A * log pi(path, answer) - beta * entropy, beta annealed beta0 -> 0 over the first half
  bon  : best-of-n restarts of marg (n = 4) selected on training reward (hard-execution accuracy on 2,000 training
         problems at N 1..8); reported as best-of-n with n
  hyb  : marg for the first half of updates, then hem
  rmer : RandoMer-style meritocratic update (Guu 1704.07926): over each problem's CORRECT sampled paths, weights uniform
         (beta = 0), so high-probability spurious paths do not dominate
Curricula: std ({1}, {1,2}, {1,2,4}, {1,2,4,8}) or zhang (N = 2 for 32 %, N <= 4 to 48 %, N <= 8 after).
Evaluation: hard execution (argmax codes and answer) at N 8, 9, 32, 128 and the product-automaton certificate.
    python theory/c1_rl.py --group S3 --cell mlp --method hem --seeds 100 101 ... [--M 12] [--updates 8000]
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


class Cell(nn.Module):
    def __init__(self, kind, M, A, hid=256):
        super().__init__()
        self.kind, self.M, self.A = kind, M, A
        if kind == "tab":
            self.T = nn.Parameter(torch.randn(M, A, M) * 0.1)
        elif kind == "bil":
            self.U, self.V, self.out = nn.Linear(M, hid), nn.Linear(A, hid), nn.Linear(hid, M)
        else:
            self.f = nn.Sequential(nn.Linear(M + A, hid), nn.ReLU(), nn.Linear(hid, M))
        self.D = nn.Parameter(torch.randn(M, A) * 0.1)

    def table(self):
        if self.kind == "tab":
            return self.T
        if self.kind == "bil":
            return self.out(torch.relu((self.U.weight.T + self.U.bias)[:, None, :] * (self.V.weight.T + self.V.bias)[None, :, :]))
        X = torch.cat([torch.eye(self.M, device=self.D.device).repeat_interleave(self.A, 0),
                       torch.eye(self.A, device=self.D.device).repeat(self.M, 1)], 1)
        return self.f(X).view(self.M, self.A, self.M)


def length(rng, u, updates, curriculum):
    if curriculum == "zhang":
        f = u / updates
        return 2 if f < 0.32 else int(rng.integers(1, 5)) if f < 0.48 else int(rng.integers(1, 9))
    stages = [[1], [1, 2], [1, 2, 4], [1, 2, 4, 8]]
    return int(rng.choice(stages[min(3, 4 * u // updates)]))


def certify(m, COMP, e, rng):
    with torch.no_grad():
        Th = m.table().argmax(-1).cpu().numpy(); Dh = m.D.argmax(-1).cpu().numpy()
    A = COMP.shape[1]
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
    return bad == 0, bad, acc


def train_acc(m, COMP, e, rng):
    with torch.no_grad():
        Th = m.table().argmax(-1).cpu().numpy(); Dh = m.D.argmax(-1).cpu().numpy()
    A = COMP.shape[1]; ok = []
    for N in range(1, 9):
        ops = rng.integers(0, A, (250, N)); c = np.zeros(250, int); st = np.full(250, e)
        for t in range(N):
            c = Th[c, ops[:, t]]; st = COMP[st, ops[:, t]]
        ok.append((Dh[c] == st).mean())
    return float(np.mean(ok))


def run(group, cell, method, seed, M, updates, lr, tau0, curriculum, K=8, beta0=0.05, bs=256, dev="cpu"):
    COMP, e, A = comp_table(group)
    Ct = torch.as_tensor(COMP, device=dev)
    if method == "bon":
        best = None
        for r in range(4):
            mm = run(group, cell, "marg", seed * 100 + r, M, updates, lr, tau0, curriculum, dev=dev, _model=True)
            ta = train_acc(mm, COMP, e, np.random.default_rng(seed))
            if best is None or ta > best[0]:
                best = (ta, mm)
        cert, bad, acc = certify(best[1], COMP, e, np.random.default_rng(seed + 7))
        return {"certified": bool(cert), "bad_pairs": bad, "acc": acc, "selected_train_acc": best[0], "n": 4}
    return _train(group, cell, method, seed, M, updates, lr, tau0, curriculum, K, beta0, bs, dev)


def _train(group, cell, method, seed, M, updates, lr, tau0, curriculum, K, beta0, bs, dev, model_only=False):
    COMP, e, A = comp_table(group)
    Ct = torch.as_tensor(COMP, device=dev)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = Cell(cell, M, A).to(dev); opt = torch.optim.Adam(m.parameters(), lr=lr)
    for u in range(updates):
        L = length(rng, u, updates, curriculum)
        tau = 1.0 + (tau0 - 1.0) * max(0.0, 1.0 - u / (0.5 * updates))
        meth = ("marg" if u < updates // 2 else "hem") if method == "hyb" else method
        ops = torch.as_tensor(rng.integers(0, A, (bs, L)), device=dev)
        s = torch.full((bs,), e, device=dev, dtype=torch.long)
        for t in range(L):
            s = Ct[s, ops[:, t]]
        logits = m.table() / tau
        if meth == "marg":
            T = F.softmax(logits, -1); alpha = torch.zeros(bs, M, device=dev); alpha[:, 0] = 1.0
            for t in range(L):
                alpha = torch.einsum("bi,bij->bj", alpha, T[:, ops[:, t]].permute(1, 0, 2))
            loss = -torch.log((alpha * F.softmax(m.D, -1)[:, s].T).sum(1).clamp_min(1e-12)).mean()
        else:
            opsK = ops.repeat_interleave(K, 0); sK = s.repeat_interleave(K, 0)
            c = torch.zeros(bs * K, dtype=torch.long, device=dev); logp = torch.zeros(bs * K, device=dev); ent = 0.0
            for t in range(L):
                lg = logits[c, opsK[:, t]]                                   # (bs*K, M)
                dist = torch.distributions.Categorical(logits=lg)
                nc = dist.sample(); logp = logp + dist.log_prob(nc); ent = ent + dist.entropy().mean(); c = nc
            dd = torch.distributions.Categorical(logits=m.D[c])
            ans = dd.sample(); logp = logp + dd.log_prob(ans)
            r = (ans == sK).float()
            if meth == "hem":
                loss = -(logp * r).sum() / r.sum().clamp_min(1.0)
            elif meth == "rmer":                                             # RandoMer (Guu 1704.07926): meritocratic
                w = torch.zeros(bs * K, device=dev)                         # weights over the CORRECT paths of each problem,
                lp = logp.detach().view(bs, K); rr = r.view(bs, K)          # q ∝ p^beta with beta = 0 (uniform over correct)
                q = torch.where(rr > 0, (lp * 0.0).exp(), torch.zeros_like(lp))
                q = q / q.sum(1, keepdim=True).clamp_min(1e-12)
                loss = -(q.view(-1) * logp).sum() / max(float((rr.sum(1) > 0).sum()), 1.0)
            else:                                                            # grpo
                adv = r.view(bs, K); adv = (adv - adv.mean(1, keepdim=True)).view(-1)
                beta = beta0 * max(0.0, 1.0 - u / (0.5 * updates))
                loss = -(adv * logp).mean() - beta * ent / max(L, 1)
        opt.zero_grad(); loss.backward(); opt.step()
    if model_only:
        return m
    cert, bad, acc = certify(m, COMP, e, np.random.default_rng(seed + 7))
    return {"certified": bool(cert), "bad_pairs": bad, "acc": acc, "final_loss": float(loss.item())}


def _run_marg_model(group, cell, seed, M, updates, lr, tau0, curriculum, dev):
    return _train(group, cell, "marg", seed, M, updates, lr, tau0, curriculum, 8, 0.0, 256, dev, model_only=True)


# best-of-n uses the marg trainer
def run(group, cell, method, seed, M, updates, lr, tau0, curriculum, K=8, beta0=0.05, bs=256, dev="cpu", _model=False):  # noqa: F811
    COMP, e, A = comp_table(group)
    if method == "bon":
        best = None
        for r_ in range(4):
            mm = _run_marg_model(group, cell, seed * 100 + r_, M, updates, lr, tau0, curriculum, dev)
            ta = train_acc(mm, COMP, e, np.random.default_rng(seed))
            if best is None or ta > best[0]:
                best = (ta, mm)
        cert, bad, acc = certify(best[1], COMP, e, np.random.default_rng(seed + 7))
        return {"certified": bool(cert), "bad_pairs": bad, "acc": acc, "selected_train_acc": best[0], "n": 4}
    return _train(group, cell, method, seed, M, updates, lr, tau0, curriculum, K, beta0, bs, dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S3"); ap.add_argument("--cell", default="mlp", choices=["tab", "mlp", "bil"])
    ap.add_argument("--method", default="hem", choices=["marg", "hem", "grpo", "bon", "hyb", "rmer"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[100, 101, 102]); ap.add_argument("--M", type=int, default=0)
    ap.add_argument("--updates", type=int, default=8000); ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--tau0", type=float, default=1.0); ap.add_argument("--curriculum", default="std", choices=["std", "zhang"])
    ap.add_argument("--K", type=int, default=8); ap.add_argument("--beta0", type=float, default=0.05)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--out")
    a = ap.parse_args(); torch.set_num_threads(3)
    _, _, A = comp_table(a.group); M = a.M or A
    res = {}
    for s in a.seeds:
        t0 = __import__("time").time()
        r = run(a.group, a.cell, a.method, s, M, a.updates, a.lr, a.tau0, a.curriculum, a.K, a.beta0, dev=a.device)
        r["cpu_sec"] = round(__import__("time").time() - t0, 1); res[s] = r
        print(a.group, a.cell, a.method, "M", M, "seed", s, "certified", r["certified"], "bad", r["bad_pairs"],
              "acc", {k: round(v, 3) for k, v in r["acc"].items()}, flush=True)
    if a.out:
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
