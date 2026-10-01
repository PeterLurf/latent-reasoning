"""THEORY §54 (frontier F1): annotated-input reasoning (AIL) -- the reasoning state is a label on each node id of the
problem, the only thing carried between passes; each pass is the same transformer applied to [edge tokens + labels].

State: labels l in {0, 1}^V per graph (1 = reached), l_0 = {root}. A pass embeds each edge token as
  src_emb[src] + dst_emb[dst] + lab_src[l(src)] + lab_dst[l(dst)] + type,
runs an encoder (no positions; the edge list is a set), scores each edge token with "its target becomes reached", takes
the max over edges into each node, and updates
  monotone (default): l' = l OR (score > 0)            -- labels can only switch on; a fixed point is guaranteed
  free (--free)     : l' = (score_keep_or_set > 0)     -- labels may switch off (counter-risk ablation)
Hard labels in the forward pass, sigmoid straight-through gradients. No hidden state, step index or history is carried.
Answer for node q after P passes: l_P(q). Training: final answers only (reachable vs not, make_batch queries), P = 6
passes on training graphs of depth <= 6; --hint adds per-pass targets (reached within p hops) as an upper bound.
    python theory/ail.py --out theory/runs/ail_mono_s1 --seed 1 [--free] [--hint] [--steps 20000] [--device cuda:0]
    python theory/ail.py --eval theory/runs/ail_mono_s1          # probe evaluation + edit test -> eval_ail.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.data import make_batch  # noqa: E402
from latent.graphs import UNREACH, V, sample_graphs  # noqa: E402


class AIL(nn.Module):
    def __init__(self, d=128, layers=2, heads=4, free=False):
        super().__init__()
        self.free = free
        self.src_emb, self.dst_emb = nn.Embedding(V, d), nn.Embedding(V, d)
        self.lab_src, self.lab_dst = nn.Embedding(2, d), nn.Embedding(2, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, 2 if free else 1)
        with torch.no_grad():                    # start with "mark nothing": a random init floods monotone labels
            self.head.bias.fill_(-4.0)
            if free:
                self.head.bias[1] = 8.0          # keep-or-set: start by keeping existing labels

    def step(self, src, dst, lab):
        """lab: (B, V) float in {0,1} (with straight-through grads). Returns new labels (B, V)."""
        ls = torch.gather(lab, 1, src); ld = torch.gather(lab, 1, dst)
        x = (self.src_emb(src) + self.dst_emb(dst) + ls[..., None] * self.lab_src.weight[1] + (1 - ls[..., None]) * self.lab_src.weight[0]
             + ld[..., None] * self.lab_dst.weight[1] + (1 - ld[..., None]) * self.lab_dst.weight[0])
        h = self.enc(x)
        s = self.head(h)                                                     # (B, E, 1 or 2)
        B = src.shape[0]
        node = torch.full((B, V, s.shape[-1]), -30.0, device=src.device, dtype=s.dtype)
        node = node.scatter_reduce(1, dst[..., None].expand(-1, -1, s.shape[-1]), s, reduce="amax", include_self=True)
        if self.free:                                                        # keep-or-set logit, may switch labels off
            soft = torch.sigmoid(node[..., 0] + lab * node[..., 1])
        else:                                                                # monotone OR
            soft = torch.maximum(lab, torch.sigmoid(node[..., 0]))
        hard = (soft > 0.5).float()
        return hard + soft - soft.detach()

    def forward(self, src, dst, start, P, record=False):
        lab = F.one_hot(start, V).float()
        traj = [lab]
        for _ in range(P):
            lab = self.step(src, dst, lab)
            traj.append(lab)
        return (lab, traj) if record else lab


def bfs_within(g, p):
    d = g.dist()
    return ((d >= 1) & (d <= p)).astype(np.float32)


def train(a):
    dev = a.device
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    m = AIL(free=a.free).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n")
    t0 = time.time()
    for step in range(1, a.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = a.lr * min(1, step / 200) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        g = sample_graphs(rng, a.bs, split="train", max_bdepth=None)
        b = make_batch(rng, g, 16, dev)
        lab, traj = m(b["src"], b["dst"], b["start"], a.P, record=True)
        sq = lambda x: x * (1 - 2e-3) + 1e-3          # affine squash: keeps the straight-through gradient (clamp kills it)
        pq = sq(torch.gather(lab, 1, b["q"]))
        loss = F.binary_cross_entropy(pq, b["qy"])
        if a.hint:                                                           # upper bound: per-pass targets
            for p in range(1, a.P + 1):
                tgt = torch.as_tensor(bfs_within(g, p), device=dev)
                tgt[torch.arange(len(g)), b["start"]] = 1
                loss = loss + F.binary_cross_entropy(sq(traj[p]), tgt) / a.P
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 200 == 0 or step == a.steps:
            acc = float(((pq > 0.5).float() == b["qy"]).float().mean())
            log.write(json.dumps({"step": step, "loss": loss.item(), "train_acc": acc, "sec": time.time() - t0}) + "\n"); log.flush()
    torch.save({"model": m.state_dict(), "args": vars(a)}, os.path.join(a.out, "model.pt"))
    print("done", a.out, "sec %.0f" % (time.time() - t0), flush=True)


@torch.no_grad()
def evaluate(path, dev="cpu"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from deep_chain import deep_chain_graphs
    from rollout import probe
    ck = torch.load(os.path.join(path, "model.pt"), map_location=dev, weights_only=False)
    m = AIL(free=ck["args"]["free"]).to(dev); m.load_state_dict(ck["model"]); m.eval()
    T = lambda x: torch.as_tensor(x, device=dev)
    out = {}
    rng = np.random.default_rng(12345)
    g = sample_graphs(rng, 2048, split="discovery", max_bdepth=None)
    b = make_batch(rng, g, 16, dev)
    lab = m(b["src"], b["dst"], b["start"], 6)
    pr = torch.gather(lab, 1, b["q"]) > 0.5; y = b["qy"] > 0.5
    out["in_dist_bal"] = float(0.5 * (pr[y].float().mean() + (~pr[~y]).float().mean()))
    for R, P_list in ((2, (6, 12, 16)), (4, (12, 24, 32))):
        if R == 2:
            gg, br, _ = probe(512); bi = np.arange(len(gg))
            reach = np.concatenate([gg.nodes[bi, br["A"]], gg.nodes[bi, br["B"]]], 1)
            decoy = np.concatenate([gg.nodes[bi, br["D"]], gg.nodes[bi, br["C"]]], 1)
        else:
            gg, s = deep_chain_graphs(np.random.default_rng(7), 512, [2, 3], 4); reach, decoy = s["reach"], s["decoy"]
        for P in P_list:
            lab, traj = m(T(gg.src), T(gg.dst), T(gg.start), P, record=True)
            L_ = lab.cpu().numpy() > 0.5; bi = np.arange(len(gg))
            out[f"R{R}_P{P}"] = [float(0.5 * (L_[bi, reach[:, j]].mean() + 1 - L_[bi, decoy[:, j]].mean())) for j in range(reach.shape[1])]
            if P == P_list[-1]:                                             # fixed point: first pass with no change
                ch = [((traj[p] > 0.5) != (traj[p - 1] > 0.5)).any(1).cpu().numpy() for p in range(1, P + 1)]
                fp = np.array([next((p for p in range(P) if not ch[p][i]), P) for i in range(len(gg))])
                out[f"R{R}_fixed_point_median"] = float(np.median(fp))
    # edit test (C7): at pass 2, switch ON the look-alike chain's node at index 1 (D1) and OFF nothing; prediction: the
    # look-alike's successors (D2, D3) become reached by the end, the reachable chain is unchanged
    gg, br, _ = probe(512); bi = np.arange(len(gg))
    reach = np.concatenate([gg.nodes[bi, br["A"]], gg.nodes[bi, br["B"]]], 1)
    decoy = np.concatenate([gg.nodes[bi, br["D"]], gg.nodes[bi, br["C"]]], 1)
    src, dst, st = T(gg.src), T(gg.dst), T(gg.start)
    lab = F.one_hot(st, V).float()
    for p in range(12):
        if p == 2:
            lab = lab.clone(); lab[torch.arange(len(gg)), T(decoy[:, 1])] = 1.0
        lab = m.step(src, dst, lab)
    base = m(src, dst, st, 12)
    L1, L0 = lab.cpu().numpy() > 0.5, base.cpu().numpy() > 0.5
    ins = L1[bi, decoy[:, 2]] & L1[bi, decoy[:, 3]]
    keep = (L1[bi[:, None], reach] == L0[bi[:, None], reach]).all(1)
    out["edit"] = {"ins_d2_d3": float(ins.mean()), "reach_unchanged": float(keep.mean()), "IIA": float((ins & keep).mean())}
    json.dump(out, open(os.path.join(path, "eval_ail.json"), "w", newline="\n"), indent=1)
    print(path, "in-dist %.3f | R2 P12 d7-12 %.3f | R4 P24 d13-24 %.3f | edit IIA %.2f" % (
        out["in_dist_bal"], np.mean(out["R2_P12"][6:12]), np.mean(out["R4_P24"][12:24]), out["edit"]["IIA"]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out"); ap.add_argument("--eval")
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--P", type=int, default=6); ap.add_argument("--free", action="store_true"); ap.add_argument("--hint", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.eval:
        evaluate(a.eval, a.device)
    else:
        train(a)


if __name__ == "__main__":
    main()
