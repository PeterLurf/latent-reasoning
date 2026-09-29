"""Minimal model for THEORY §22 Prop. 5 (contraction needs a restoring map) and proposal T6 (amplitude jitter).

A pointer amplitude a moves one hop per step through a learned 1-D map phi (residual MLP, initialised near the identity,
as a transformer's residual carry is). Every hop adds noise; decoy chains start at 0 and carry only propagated noise.
Readout at every depth d <= D: sigmoid(kappa (a_d - theta)), BCE against reachable (1) / decoy (0) — anytime-style, so
slack plays no role here. Arm "jitter": during training the input amplitude of every hop is multiplied by
s ~ LogUniform[1/r, r] (the transfer-curve perturbation of exp9 turned into augmentation). Evaluation: no jitter, chains
of length 40, reach = first depth where balanced accuracy < 0.9; the learned map's fixed points and slope there.
    python theory/sim_jitter.py
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "3")))
L, KAPPA = 40, 10.0


class Phi(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, width), nn.Tanh(), nn.Linear(width, 1))
        nn.init.normal_(self.net[2].weight, std=1e-3); nn.init.zeros_(self.net[2].bias)
        self.theta = nn.Parameter(torch.tensor(0.5))

    def forward(self, a):
        return a + self.net(a[..., None])[..., 0]


def chains(phi, n, depth, sig, jitter=0.0, gen=None):
    a = 1.0 + sig * torch.randn(n, generator=gen); b = sig * torch.randn(n, generator=gen)
    A, Bs = [a], [b]
    for _ in range(depth - 1):
        if jitter:
            s = torch.exp((2 * torch.rand(2, n, generator=gen) - 1) * np.log(jitter))
            a, b = a * s[0], b * s[1]
        a = phi(a) + sig * torch.randn(n, generator=gen)
        b = phi(b) + sig * torch.randn(n, generator=gen)
        A.append(a); Bs.append(b)
    return torch.stack(A, 1), torch.stack(Bs, 1)


def train(seed, D, sig, jitter, steps=3000, n=512, wd=1e-4):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    phi = Phi()
    opt = torch.optim.Adam(phi.parameters(), lr=3e-3, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        A, Bs = chains(phi, n, D, sig, jitter, gen)
        z = torch.cat([A, Bs]); y = torch.cat([torch.ones_like(A), torch.zeros_like(Bs)])
        loss = bce(KAPPA * (z - phi.theta), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return phi


@torch.no_grad()
def evaluate(phi, sig, seed):
    gen = torch.Generator().manual_seed(10_000 + seed)
    A, Bs = chains(phi, 4000, L, sig, 0.0, gen)
    th = phi.theta.item()
    acc = 0.5 * ((A > th).float().mean(0) + (Bs <= th).float().mean(0)).numpy()
    ok = acc >= 0.9
    reach = int(np.argmin(ok)) if not ok.all() else L
    x = torch.linspace(-0.5, 2.5, 3001)
    y = phi(x) - x
    sgn = torch.sign(y)
    fps = [float(x[i]) for i in range(len(x) - 1) if sgn[i] != sgn[i + 1]]
    slope = lambda p: float((phi(torch.tensor([p + 1e-3])) - phi(torch.tensor([p - 1e-3]))) / 2e-3)
    fp = [(round(p, 3), round(slope(p), 3)) for p in fps]
    stable_hi = [p for p, s in fp if p > th and abs(s) < 1]
    gain = float(phi(torch.tensor([1.0])) / 1.0)
    return {"reach": reach, "theta": th, "acc_D+5": float(acc[min(L - 1, 9)]), "fixed_points": fp,
            "restoring": bool(stable_hi), "gain_at_1": gain}


if __name__ == "__main__":
    out = []
    for sig in (0.08, 0.15):
        for D in (3, 5):
            for arm, jit in (("plain", 0.0), ("jitter", 2.0)):
                for seed in range(6):
                    phi = train(seed, D, sig, jit)
                    r = evaluate(phi, sig, seed)
                    r.update(sig=sig, D=D, arm=arm, seed=seed)
                    out.append(r)
                    print("sig %.2f D %d %-6s s%d reach %2d restoring %s gain(1) %.3f theta %.2f fps %s" % (
                        sig, D, arm, seed, r["reach"], r["restoring"], r["gain_at_1"], r["theta"], r["fixed_points"]), flush=True)
    json.dump(out, open("theory/results/sim_jitter.json", "w", newline="\n"), indent=1)
    print("\nsummary (median reach, restoring share)")
    for sig in (0.08, 0.15):
        for D in (3, 5):
            for arm in ("plain", "jitter"):
                rr = [r for r in out if r["sig"] == sig and r["D"] == D and r["arm"] == arm]
                print("sig %.2f D %d %-6s reach %s  restoring %d/%d" % (sig, D, arm, sorted(r["reach"] for r in rr),
                      sum(r["restoring"] for r in rr), len(rr)))
