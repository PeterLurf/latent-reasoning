"""Red-team D3 (e): is N2's pass-1 gather a translation-invariant symbolic rule? Own implementation (CPU, 3 threads, idle).
In-range items (T 15..27, own rng 5150), equal-length batches. Target p = the trained pass-1 gather (c = m.c_anneal, here 0), per head.
Fits (per head, KL(p || q) by Adam), translation-invariant by construction:
  base : M[c_j, c_i] + w[clip(i - j, +-8)] + sink + ' =' anchors          (D3's family)
  rich : base + A[c_i, off] + B[c_j, off] + D[clip(depth_i - depth_j, +-3)] + Dj[depth_j] (off = clip(i - j, +-8), 19 buckets)
Scale: mean entropy H(p), KL(p || uniform), KL of offset-only. Held-out = 25% of items.
Usage: python rt_symfit.py <run> <out.json>"""
import json, math, sys
from collections import defaultdict
sys.path.insert(0, "D:/Latent")
try:
    import psutil; psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
except Exception:
    pass
import numpy as np
import torch
torch.set_num_threads(3)
import boxes.nested170 as N
import boxes.vbank as VB
import boxes.vbank_eval as VE
import boxes.deltamem as DM
R_OFF = 8; NB = 2 * R_OFF + 3


def depth_of(cls):
    d = 0; out = []
    for c in cls:
        if c == VB.CID[" ("]:
            d += 1
        out.append(min(d, 4))
        if c == VB.CID[" )"]:
            d -= 1
    return out


def main():
    run, fo = sys.argv[1], sys.argv[2]
    m, a = VE.load(run, "cpu"); H = m.H
    rng = np.random.default_rng(5150); byT = defaultdict(list)
    n_items = 0
    while n_items < 1400:
        mo = int(rng.integers(3, 9)); r, _t = N.generate(rng, mo); s = N.render(r); T = len(s) + 2
        if 15 <= T <= 27:
            byT[T].append({"root": r, "syms": s}); n_items += 1
    batches = []
    with torch.no_grad():
        for T, its in sorted(byT.items()):
            for c0 in range(0, len(its), 32):
                part = its[c0:c0 + 32]; x, pad, lens = VB.batch_x(part, "cpu")
                V0 = DM.gpt_lower(m.g, m.g.embed(x), VB.CORE[0])
                q = (DM.wpe_slice(m.g, T) @ m.gq_w + m.gq_b).view(T, H, m.hd).transpose(0, 1)
                lg = torch.einsum("htd,bhsd->bhts", q, m.gkey(V0)) / math.sqrt(m.hd) - m.c_anneal * (~torch.eye(T, dtype=torch.bool)).float()[None, None]
                p = torch.softmax(lg, -1)
                C = torch.tensor([VB.recode_targets(it["root"], 0)[0] for it in part]); Dp = torch.tensor([depth_of(c) for c in C.tolist()])
                batches.append((T, p, C, Dp))
    rs = np.random.default_rng(1); idx = rs.permutation(len(batches)); nte = max(1, len(batches) // 4)
    te = [batches[i] for i in idx[:nte]]; tr = [batches[i] for i in idx[nte:]]

    class Fit(torch.nn.Module):
        def __init__(self, rich):
            super().__init__(); self.rich = rich; z = lambda *s: torch.nn.Parameter(torch.zeros(*s))  # noqa: E731
            self.M = z(H, VB.NCLS, VB.NCLS); self.w = z(H, NB); self.sink = z(H); self.eq = z(H)
            if rich:
                self.A = z(H, VB.NCLS, NB); self.B = z(H, VB.NCLS, NB); self.D = z(H, 7); self.Dj = z(H, 5, NB)

        def forward(self, T, C, Dp):
            d = torch.arange(T)[None, :] - torch.arange(T)[:, None]
            b = torch.where(d < -R_OFF, torch.zeros_like(d), torch.where(d > R_OFF, torch.full_like(d, NB - 1), d + R_OFF + 1))
            lg = self.M[:, C[:, :, None], C[:, None, :]].permute(1, 0, 2, 3) + self.w[:, b][None]
            lg = lg.clone(); lg[..., 0] += self.sink[None, :, None]; lg[..., T - 1] += self.eq[None, :, None]
            if self.rich:
                lg = lg + self.A[:, C[:, None, :].expand(-1, T, -1), b[None].expand(C.shape[0], -1, -1)].permute(1, 0, 2, 3)
                lg = lg + self.B[:, C[:, :, None].expand(-1, -1, T), b[None].expand(C.shape[0], -1, -1)].permute(1, 0, 2, 3)
                dd = (Dp[:, None, :] - Dp[:, :, None]).clamp(-3, 3) + 3
                lg = lg + self.D[:, dd].permute(1, 0, 2, 3)
                lg = lg + self.Dj[:, Dp[:, :, None].expand(-1, -1, T), b[None].expand(C.shape[0], -1, -1)].permute(1, 0, 2, 3)
            return torch.log_softmax(lg, -1)

    def kl(f, bs):
        tot = 0.0; n = 0
        for T, p, C, Dp in bs:
            lq = f(T, C, Dp); k_ = (p * (torch.log(p.clamp_min(1e-30)) - lq)).sum(-1); tot = tot + k_.sum(); n += k_.numel()
        return tot / n
    out = {"run": run, "c_anneal": float(m.c_anneal), "n_batches": len(batches), "n_items": n_items}
    with torch.no_grad():
        ent = sum(float(-(p * torch.log(p.clamp_min(1e-30))).sum(-1).sum()) for _, p, _, _ in batches) / sum(p[..., 0].numel() for _, p, _, _ in batches)
        klu = sum(float((p * (torch.log(p.clamp_min(1e-30)) + math.log(T))).sum(-1).sum()) for T, p, _, _ in batches) / sum(p[..., 0].numel() for _, p, _, _ in batches)
    out["mean_entropy_nats"] = ent; out["kl_uniform"] = klu
    for name, rich, offonly in (("offset_only", False, True), ("base", False, False), ("rich", True, False)):
        f = Fit(rich)
        params = [f.w, f.sink, f.eq] if offonly else list(f.parameters())
        opt = torch.optim.Adam(params, lr=0.05)
        for it_ in range(400):
            opt.zero_grad(); L = kl(f, tr); L.backward(); opt.step()
            if it_ % 100 == 0:
                print(name, it_, float(L), flush=True)
        with torch.no_grad():
            out[name] = {"kl_train": float(kl(f, tr)), "kl_heldout": float(kl(f, te))}
        print(name, out[name], flush=True)
    json.dump(out, open(fo, "w", newline="\n"), indent=1); print("DONE", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
