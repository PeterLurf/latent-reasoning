"""THEORY §64: permutation tracking (S3 / S4 word problems) in a Coconut-style latent loop — do the graph toy's findings
(a count-keyed stop beyond the trained length; written-state restarts fixing it) reappear outside graph search?

Task. n items start in an arrangement (identity with probability 1/2, else uniform random); N operations (uniform
random elements of the group: S3, S4, S5, or A5 = the even permutations of 5, non-solvable) are applied in order. Query: for each item, its final position (n-way).
Input. [n init tokens: item i at slot j] + [N op tokens at sorted random positions in [0, 64)] (randomized positions,
Ruoss 2305.16843, so every position embedding is trained). The prefix is read with full attention.
Loop. K latent steps (Coconut: the final-layer output at the last latent position, layer-normed, is fed back as the
next latent token; no step embedding). Latents attend to the prefix and earlier latents. n query tokens (one per item)
then attend to everything and are classified over the n positions. Training: final answers only, N ~ U{1..8}, K = 8.
Evaluation (CPU or GPU):
  single  : accuracy (all items right) vs N = 1..24 at K' = max(8, N) and at K' = 24
  trunc   : for N > 8, agreement of the answers with the arrangement after the first 8 operations (the count-keyed-stop
            signature) vs with the true final arrangement
  restart : written-state restarts: run 8 ops, decode the arrangement (the most probable permutation), write it as the
            init block, restart on the next 8 ops, ...; accuracy vs N
  pointer : per latent step, the rank of the op token most attended by layer 1 (mean over heads), on N = 16 inputs
    python theory/st_model.py --group A5 --seed 1 --out theory/runs/st/a5_s1 [--steps 20000] [--device cuda:0]
    python theory/st_model.py --eval theory/runs/st/s3_s1
"""
import argparse
import itertools
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PMAX, NMAX_TRAIN, KTRAIN = 64, 8, 8
T_INIT, T_OP, T_LAT, T_Q = 0, 1, 2, 3


def perms(n):
    return [tuple(p) for p in itertools.permutations(range(n))]


def group_elems(group):
    """'S3' / 'S4' / 'S5': all permutations; 'A5': the even ones (non-solvable, 60 elements). -> (n items, elements)."""
    n = int(group[1:])
    P = perms(n)
    if group[0] == "A":
        P = [p for p in P if sum(p[i] > p[j] for i in range(n) for j in range(i + 1, n)) % 2 == 0]
    return n, P


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h; self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.h)
        a = torch.softmax(s.masked_fill(~mask, -1e9), -1)
        x = x + self.o((a @ v).transpose(1, 2).reshape(B, L, D))
        return x + self.mlp(self.ln2(x)), a


class STReasoner(nn.Module):
    def __init__(self, group, d=256, layers=2, heads=8):
        super().__init__()
        n, self.G = group_elems(group)
        self.n = n
        self.op_emb = nn.Embedding(len(self.G), d); self.item_emb = nn.Embedding(n, d); self.slot_emb = nn.Embedding(n, d)
        self.pos_emb = nn.Embedding(PMAX, d); self.type_emb = nn.Embedding(4, d); self.q_emb = nn.Embedding(n, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d); self.head = nn.Linear(d, n)

    def prefix(self, init, ops, pos, nops):
        """init (B, n) slot of each item; ops (B, Nmax) element ids; pos (B, Nmax); nops (B,) -> (x, valid)."""
        B, Nm = ops.shape; te = self.type_emb.weight
        xi = self.item_emb.weight[None].expand(B, -1, -1) + self.slot_emb(init) + te[T_INIT]
        xo = self.op_emb(ops) + self.pos_emb(pos) + te[T_OP]
        valid = torch.cat([torch.ones(B, self.n, dtype=torch.bool, device=ops.device),
                           torch.arange(Nm, device=ops.device)[None] < nops[:, None]], 1)
        return torch.cat([xi, xo], 1), valid

    def run(self, init, ops, pos, nops, K, keep_attn=False):
        x, valid = self.prefix(init, ops, pos, nops)
        B, P, D = x.shape; te = self.type_emb.weight
        lat, attn1 = [], []
        h = torch.zeros(B, D, device=x.device)
        for k in range(K):
            xl = (h + te[T_LAT])[:, None]
            seq = torch.cat([x] + lat + [xl], 1)
            L = seq.shape[1]
            keys = torch.cat([valid, torch.ones(B, L - P, dtype=torch.bool, device=x.device)], 1)
            mask = keys[:, None, None, :].expand(B, 1, L, L)
            z = seq
            for li, blk in enumerate(self.blocks):
                z, a = blk(z, mask)
                if li == 0 and keep_attn:
                    attn1.append(a[:, :, -1, self.n:P].mean(1))           # last latent's attention over op tokens
            h = self.ln_f(z[:, -1]); lat.append(xl)
        xq = self.q_emb.weight[None].expand(B, -1, -1) + te[T_Q]
        seq = torch.cat([x] + lat + [xq], 1); L = seq.shape[1]
        keys = torch.cat([valid, torch.ones(B, L - P, dtype=torch.bool, device=x.device)], 1)
        z = seq
        for blk in self.blocks:
            z, _ = blk(z, keys[:, None, None, :].expand(B, 1, L, L))
        logits = self.head(self.ln_f(z[:, -self.n:]))                       # (B, n items, n positions)
        return (logits, attn1) if keep_attn else logits


def sample(rng, group, B, Nlo, Nhi, Nmax=None, ident_p=0.5):
    """Returns init (B, n) slot of each item, ops (B, Nmax), pos (B, Nmax), nops (B,), final (B, n), trunc8 (B, n)."""
    n, G = group_elems(group); Nmax = Nmax or Nhi
    nops = rng.integers(Nlo, Nhi + 1, B)
    init = np.array([np.arange(n) if rng.random() < ident_p else rng.permutation(n) for _ in range(B)])
    ops = rng.integers(0, len(G), (B, Nmax))
    pos = np.zeros((B, Nmax), np.int64)
    for b in range(B):
        pos[b, :nops[b]] = np.sort(rng.choice(PMAX, nops[b], replace=False))
    final, trunc = init.copy(), init.copy()
    for b in range(B):
        s = init[b].copy()
        for t in range(nops[b]):
            s = np.array(G[ops[b, t]])[s]                   # op maps slot -> slot; item i moves from s[i] to g[s[i]]
            if t == NMAX_TRAIN - 1:
                trunc[b] = s
        final[b] = s
        if nops[b] <= NMAX_TRAIN:
            trunc[b] = s
    return init, ops, pos, nops, final, trunc


def tt(dev, *xs):
    return [torch.as_tensor(x, device=dev) for x in xs]


def train(a):
    dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    m = STReasoner(a.group).to(dev); n = m.n
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time()
    for step in range(1, a.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = a.lr * min(1, step / 500) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        nlo, nhi = 1, NMAX_TRAIN
        if a.curriculum:                       # THEORY §64b: Zhang et al.'s operator-first curriculum, adapted to N <= 8:
            f = step / a.steps                 # length-2 problems only (the composition table) for the first 32 %,
            nlo, nhi = (2, 2) if f < 0.32 else (1, 4) if f < 0.48 else (1, NMAX_TRAIN)   # then N <= 4, then §64's U{1..8}
        init, ops, pos, nops, final, _ = sample(rng, a.group, a.bs, nlo, nhi, Nmax=NMAX_TRAIN)
        init, ops, pos, nops, final = tt(dev, init, ops, pos, nops, final)
        logits = m.run(init, ops, pos, nops, KTRAIN)
        loss = F.cross_entropy(logits.reshape(-1, n), final.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 500 == 0 or step == a.steps:
            acc = float((logits.argmax(-1) == final).all(1).float().mean())
            log.write(json.dumps({"step": step, "loss": loss.item(), "train_full_acc": acc, "sec": round(time.time() - t0)}) + "\n"); log.flush()
    torch.save({"model": m.state_dict(), "args": vars(a)}, os.path.join(a.out, "model.pt"))
    evaluate(a.out, dev, a.n_eval)


@torch.no_grad()
def evaluate(path, dev="cpu", n_eval=1024):
    ck = torch.load(os.path.join(path, "model.pt"), map_location=dev, weights_only=False)
    group = ck["args"]["group"]; m = STReasoner(group).to(dev); m.load_state_dict(ck["model"]); m.eval()
    n = m.n; Gt = torch.as_tensor(perms(n), device=dev)                      # decode over all arrangements of n items
    out = {"group": group, "single": {}, "single_K24": {}, "single_K0": {}, "trunc": {}, "true_vs_trunc": {}, "restart": {},
           "chance": 1 / math.factorial(n)}
    for N in list(range(1, 25)):
        rng = np.random.default_rng(1000 + N)
        init, ops, pos, nops, final, trunc = sample(rng, group, n_eval, N, N, Nmax=24)
        I, O, Pp, Nn, Fi, Tr = tt(dev, init, ops, pos, nops, final, trunc)
        pred = m.run(I, O, Pp, Nn, max(KTRAIN, N)).argmax(-1)
        out["single"][N] = float((pred == Fi).all(1).float().mean())
        out["single_K24"][N] = float((m.run(I, O, Pp, Nn, 24).argmax(-1) == Fi).all(1).float().mean())
        out["single_K0"][N] = float((m.run(I, O, Pp, Nn, 0).argmax(-1) == Fi).all(1).float().mean())   # no latent step
        out.setdefault("identity", {})[N] = float((I == Fi).all(1).float().mean())   # no-reasoning baseline: no update
        if N > NMAX_TRAIN:
            out["trunc"][N] = float((pred == Tr).all(1).float().mean())
            out["true_vs_trunc"][N] = float((Tr == Fi).all(1).float().mean())   # chance overlap of the two references
        # written-state restarts every 8 ops
        cur = I.clone(); done = 0
        while done < N:
            k = min(NMAX_TRAIN, N - done)
            seg_ops = O[:, done:done + k]; seg_pos = Pp[:, done:done + k]
            lp = F.log_softmax(m.run(cur, seg_ops, seg_pos, torch.full_like(Nn, k), KTRAIN), -1)   # (B, n, n)
            score = lp[:, torch.arange(n, device=dev)[None], Gt].sum(-1)        # (B, n!) joint log-prob of each permutation
            cur = Gt[score.argmax(1)]                                         # most probable valid arrangement
            done += k
        out["restart"][N] = float((cur == Fi).all(1).float().mean())
    # written-state edit (C7 for the written state; Literature): N = 8 inputs, the init block replaced by a random
    # arrangement; IIA = the answers equal the true composition from the edited state
    rng = np.random.default_rng(55)
    init, ops, pos, nops, final, _ = sample(rng, group, n_eval, 8, 8, Nmax=8, ident_p=0.0)
    _, Gl = group_elems(group)
    new_init = np.array([rng.permutation(n) for _ in range(n_eval)])
    tgt = new_init.copy()
    for b in range(n_eval):
        s = new_init[b].copy()
        for t in range(8):
            s = np.array(Gl[ops[b, t]])[s]
        tgt[b] = s
    NI, O, Pp, Nn, Tg = tt(dev, new_init, ops, pos, nops, tgt)
    out["edit_IIA"] = float((m.run(NI, O, Pp, Nn, KTRAIN).argmax(-1) == Tg).all(1).float().mean())
    # read pointer on N = 16
    rng = np.random.default_rng(77)
    init, ops, pos, nops, final, _ = sample(rng, group, 512, 16, 16, Nmax=24)
    I, O, Pp, Nn = tt(dev, init, ops, pos, nops)
    _, att = m.run(I, O, Pp, Nn, 24, keep_attn=True)
    out["pointer_rank"] = [float(np.median(at[:, :16].argmax(1).cpu().numpy() + 1)) for at in att]
    json.dump(out, open(os.path.join(path, "eval_st.json"), "w", newline="\n"), indent=1)
    s = out["single"]; r = out["restart"]
    print(path, "in-range N1-8 %.3f | single N9-12 %.3f N13-16 %.3f | trunc-agree N9-16 %.3f | restart N9-16 %.3f N17-24 %.3f | ptr %s" % (
        np.mean([s[k] for k in range(1, 9)]), np.mean([s[k] for k in range(9, 13)]), np.mean([s[k] for k in range(13, 17)]),
        np.mean([out["trunc"][k] for k in range(9, 17)]), np.mean([r[k] for k in range(9, 17)]), np.mean([r[k] for k in range(17, 25)]),
        out["pointer_rank"][:12]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="A5", choices=["S3", "S4", "A5", "S5"]); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--device", default="cpu")
    ap.add_argument("--out"); ap.add_argument("--eval"); ap.add_argument("--n_eval", type=int, default=1024)
    ap.add_argument("--curriculum", action="store_true", help="THEORY §64b: Zhang et al. operator-first curriculum adapted to N <= 8 (default off)")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "3")))
    if a.eval:
        evaluate(a.eval, a.device, a.n_eval)
    else:
        train(a)


if __name__ == "__main__":
    main()
