"""PROTOCOL12 (docs/PROTOCOL12.md; theory source THEORY §54, frontier F1): annotated-input reasoning in GPT-2 small on
ProsQA. The only state carried between passes is a binary label per entity of the problem.

Pass: one GPT-2-small forward over [token embeddings + label embedding on every token of an entity mention]; a linear
mark head scores every token; per entity the max over its mentions gives a logit; labels update
  mono (primary): l' = l OR (logit > 0)               (labels only switch on; a fixed point is guaranteed)
  free           : l' = (set_logit + l * keep_logit > 0)  (labels may switch off; counter-risk check)
Hard labels forward, sigmoid straight-through backward. No hidden state, latent token or step index is carried.
Start: root person = 1. Answer: the option whose label is on; if both or neither, the larger final option logit.
Training: final answers only (BCE on the affine-squashed option labels), problems of depth <= 6, P = 6 passes.
  --hint: per-pass targets (entities within p hops of the root), an upper bound that uses step labels.
    python prosqa/ail_gpt2.py --out runs_ail/mono_s1 --seed 1 [--arm mono|free|hint] [--epochs 2] [--device cuda:0]
    python prosqa/ail_gpt2.py --eval runs_ail/mono_s1 [--device cuda:0]
"""
import argparse
import json
import math
import os
import re
import time
from collections import deque

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ENC = tiktoken.get_encoding("gpt2")
HERE = os.path.dirname(os.path.abspath(__file__))
PIECE = re.compile(r" ?\w+| ?[^\w\s]+|\s+")


def encode(x):
    """Token ids of the question and, per token, the entity index (-1 if none)."""
    q = x["question"]; sym = x["idx_to_symbol"]; ix = {s: i for i, s in enumerate(sym)}
    ids, ent = [], []
    for m in PIECE.finditer(q):
        p = m.group(0); t = ENC.encode(p)
        ids += t; ent += [ix.get(p.strip(), -1)] * len(t)
    assert ids == ENC.encode(q), "piecewise tokenisation differs"
    return ids, ent


def hops(x):
    adj = {}
    for a, b in x["edges"]:
        adj.setdefault(a, []).append(b)
    d = {x["root"]: 0}; q = deque([x["root"]])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in d:
                d[v] = d[u] + 1; q.append(v)
    return d


class GPT2(nn.Module):
    """GPT-2 small, trainable, plain torch (weights from the original OpenAI checkpoint, HF Conv1D layout)."""
    def __init__(self, sd):
        super().__init__()
        self.L = sum(1 for k in sd if k.startswith("h.") and k.endswith(".ln_1.weight")); self.H = 12
        self.p = nn.ParameterDict({k.replace(".", "_"): nn.Parameter(v.float().clone()) for k, v in sd.items()
                                   if not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias")})

    def w(self, k):
        return self.p[k.replace(".", "_")]

    def block(self, h, i, pad_mask):
        B, T, D = h.shape; hd = D // self.H; P = f"h.{i}."
        a = F.layer_norm(h, (D,), self.w(P + "ln_1.weight"), self.w(P + "ln_1.bias"))
        q, k, v = (a @ self.w(P + "attn.c_attn.weight") + self.w(P + "attn.c_attn.bias")).split(D, -1)
        q, k, v = (t.view(B, T, self.H, hd).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=pad_mask)
        h = h + o.transpose(1, 2).reshape(B, T, D) @ self.w(P + "attn.c_proj.weight") + self.w(P + "attn.c_proj.bias")
        a = F.layer_norm(h, (D,), self.w(P + "ln_2.weight"), self.w(P + "ln_2.bias"))
        m = F.gelu(a @ self.w(P + "mlp.c_fc.weight") + self.w(P + "mlp.c_fc.bias"), approximate="tanh")
        return h + m @ self.w(P + "mlp.c_proj.weight") + self.w(P + "mlp.c_proj.bias")

    def forward(self, emb, pad_mask, ckpt=False):
        T = emb.shape[1]
        h = emb + self.w("wpe.weight")[:T][None]
        for i in range(self.L):
            h = checkpoint(self.block, h, i, pad_mask, use_reentrant=False) if ckpt else self.block(h, i, pad_mask)
        return F.layer_norm(h, (h.shape[-1],), self.w("ln_f.weight"), self.w("ln_f.bias"))


class AIL(nn.Module):
    def __init__(self, sd, arm="mono", lab_std=0.02, bias=-4.0):
        super().__init__()
        self.free = arm == "free"
        self.lm = GPT2(sd)
        self.lab = nn.Embedding(2, 768); nn.init.normal_(self.lab.weight, std=lab_std)
        self.head = nn.Linear(768, 2 if self.free else 1)
        nn.init.zeros_(self.head.weight)              # logits start at the bias exactly: GPT-2's large final
                                                       # activations times a random head flood the labels (smoke)
        with torch.no_grad():
            self.head.bias.fill_(bias)                 # start by marking nothing (a random init floods monotone labels)
            if self.free:
                self.head.bias[1] = 8.0                # keep-or-set: start by keeping existing labels

    tau = None                                       # set during the soft warm-up (setting #5): no hard threshold
    mode = "hard"                                    # §54b: "soft" (relaxed forward) | "stoch" (Bernoulli forward) | "hard"

    def step(self, b, lab, ckpt=False):
        """b: batch dict; lab: (B, V) float {0,1} with straight-through grads. Returns new labels and entity logits."""
        ent = b["ent"]; is_ent = (ent >= 0)
        tl = torch.gather(lab, 1, ent.clamp(min=0)) * is_ent                    # (B, T) label of each token's entity
        e = self.lm.w("wte.weight")[b["ids"]]
        e = e + is_ent[..., None] * (tl[..., None] * self.lab.weight[1] + (1 - tl[..., None]) * self.lab.weight[0])
        h = self.lm(e, b["pad"], ckpt)
        s = self.head(h).float()                                                 # (B, T, 1|2)
        s = s.masked_fill(~is_ent[..., None], -30.0)
        V = lab.shape[1]
        node = torch.full((lab.shape[0], V, s.shape[-1]), -30.0, device=s.device)
        node = node.scatter_reduce(1, ent.clamp(min=0)[..., None].expand(-1, -1, s.shape[-1]), s, reduce="amax", include_self=True)
        t = 1.0 if self.tau is None else self.tau
        if not self.free and self.mode == "soft":    # §54b AI-soft / AI-hint: l' = max(l, sigmoid(z / tau))
            logit = node[..., 0]
            return torch.maximum(lab, torch.sigmoid(logit / t)), logit
        if not self.free and self.mode == "stoch":   # §54b AI-stoch: l' = l OR b, b ~ Bernoulli(sigmoid(z)), ST via sigmoid
            logit = node[..., 0]; pr = torch.sigmoid(logit)
            bs = torch.bernoulli(pr.detach())
            return lab + (1 - lab) * (bs + pr - pr.detach()), logit
        if self.free:
            logit = node[..., 0] + lab * node[..., 1]
            soft = torch.sigmoid(logit / t)
        else:
            logit = node[..., 0]
            soft = torch.maximum(lab, torch.sigmoid(logit / t))
        if self.tau is not None:                     # soft forward: continuous labels, full gradient
            return soft, logit
        hard = (soft > 0.5).float()
        return hard + soft - soft.detach(), logit

    def run(self, b, P, edit=None, record=False, ckpt=False):
        lab = b["start"].clone()
        if edit is not None:
            lab = torch.maximum(lab, edit)
        traj, logit = [lab], None
        for _ in range(P):
            lab, logit = (checkpoint(self.step, b, lab, use_reentrant=False) if ckpt else self.step(b, lab))
            traj.append(lab)
        return lab, logit, traj


def cue_variant(x, rng):
    """§54b-1: x plus "<another person> is a <parent of the target>." at a random statement position (answer, root's
    reachable set and target depth unchanged); None if no other person or no concept parent exists."""
    sym = x["idx_to_symbol"]; root = sym[x["root"]]
    people = [s for s in sym if s[:1].isupper() and s != root]
    par = [u for u, v in x["edges"] if v == x["target"] and sym[u][:1].islower()]
    if not people or not par:
        return None
    pe, pa = str(rng.choice(people)), sym[int(rng.choice(par))]
    parts = re.split(r"(?<=\.) ", x["question"]); stmts, q = parts[:-1], parts[-1]
    pos = int(rng.integers(0, len(stmts) + 1))
    art = "an" if pa[:1] in "aeiou" else "a"
    stmts = stmts[:pos] + [f"{pe} is {art} {pa}."] + stmts[pos:]
    ix = {s_: i for i, s_ in enumerate(sym)}
    return {**x, "question": " ".join(stmts) + " " + q, "edges": x["edges"] + [[ix[pe], ix[pa]]]}


def collate(xs, dev):
    enc = [encode(x) for x in xs]
    T = max(len(i) for i, _ in enc); B = len(xs); V = max(len(x["idx_to_symbol"]) for x in xs)
    ids = torch.full((B, T), 50256, dtype=torch.long); ent = torch.full((B, T), -1, dtype=torch.long)
    pad = torch.zeros(B, 1, T, T, dtype=torch.bool)
    for j, (i, e) in enumerate(enc):
        n = len(i); ids[j, :n] = torch.tensor(i); ent[j, :n] = torch.tensor(e)
        pad[j, 0, :, :n] = torch.ones(T, n, dtype=torch.bool).tril()           # causal, keys limited to real tokens
        pad[j, 0, n:, :n] = True; pad[j, 0, n:, :] |= torch.eye(T, dtype=torch.bool)[n:]  # padded queries: no NaN rows
    start = torch.zeros(B, V); tgt = torch.zeros(B, dtype=torch.long); neg = torch.zeros(B, dtype=torch.long)
    for j, x in enumerate(xs):
        start[j, x["root"]] = 1; tgt[j] = x["target"]; neg[j] = x["neg_target"]
    b = {"ids": ids, "ent": ent, "pad": pad, "start": start, "tgt": tgt, "neg": neg}
    return {k: v.to(dev) for k, v in b.items()}


def answer(lab, logit, b):
    """Prediction correct? Option on/off; tie broken by the final logits."""
    r = torch.arange(lab.shape[0], device=lab.device)
    lt, ln = lab[r, b["tgt"]] > 0.5, lab[r, b["neg"]] > 0.5
    tie = lt == ln
    return torch.where(tie, logit[r, b["tgt"]] > logit[r, b["neg"]], lt & ~ln)


def train(a):
    dev = a.device
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    sd = torch.load(a.init, map_location="cpu", weights_only=True)
    arm = "free" if a.arm == "free" else "mono"
    m = AIL(sd, arm, a.lab_std, a.bias).to(dev)
    v2 = a.arm in ("soft", "stoch", "hint2")                                  # §54b schedules
    data = [x for x in json.load(open(a.train)) if len(x["steps"]) <= a.max_depth]
    by_depth = {d: [i for i, x in enumerate(data) if len(x["steps"]) <= d] for d in range(1, a.max_depth + 1)}
    sub = [x for x in data if len(x["steps"]) <= a.max_depth][:64]           # hardened train-accuracy monitor
    head_params = list(m.lab.parameters()) + list(m.head.parameters())
    opt = torch.optim.AdamW([{"params": m.lm.parameters(), "lr": a.lr}, {"params": head_params, "lr": a.lr_head}], weight_decay=0.01)
    base_lr = [a.lr, a.lr_head]
    steps = a.epochs * len(data) // (a.bs * a.accum)
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n")
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    sq = lambda x: x * (1 - 2 * a.eps) + a.eps                             # affine squash keeps the ST gradient
    t0, step, accs, losses = time.time(), 0, [], []
    order = []
    while step < steps:
        soft_steps = int(a.soft_frac * steps)
        lam = 0.0
        if v2:                                       # §54b: tau 1 -> tau_end over 2/3, lambda ramp over the last 1/3,
            fr = step / steps                        # depth 3 only for 1/6, widening linearly to 3..6 by 1/3
            m.mode = "stoch" if a.arm == "stoch" else "soft"
            m.tau = 1 + (a.tau_end - 1) * min(1, fr / (2 / 3))
            lam = a.lam * max(0.0, (fr - 2 / 3) / (1 / 3))
            dmax = 3 if fr < 1 / 6 else min(a.max_depth, 3 + int((fr - 1 / 6) / (1 / 6) * (a.max_depth - 3 + 1)))
            pool = by_depth[dmax]
            if not order or (dmax < a.max_depth and any(len(data[i]["steps"]) > dmax for i in order[-a.bs:])):
                order = [pool[i] for i in rng.permutation(len(pool))]
        elif step < soft_steps:                        # setting #5: soft forward, tau 1 -> 0.1, depth 3 -> max_depth
            fr = step / max(1, soft_steps)
            m.tau = a.tau0 * (a.tau1 / a.tau0) ** fr
            dmax = min(a.max_depth, 3 + int(fr * (a.max_depth - 2)))
            pool = by_depth[dmax]
            if not order or len(order) > len(pool):
                order = [pool[i] for i in rng.permutation(len(pool))]
        else:
            m.tau = None
        if not order:
            order = list(rng.permutation(len(data)))
        for gp, lr in zip(opt.param_groups, base_lr):
            gp["lr"] = lr * min(1, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * step / steps))
        opt.zero_grad()
        for _ in range(a.accum):
            idx = [order.pop() for _ in range(min(a.bs, len(order)))] or [int(rng.integers(len(data)))]
            xs = [data[i] for i in idx]
            b = collate(xs, dev)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                lab, logit, traj = m.run(b, a.P, record=True, ckpt=True)
            r = torch.arange(len(xs), device=dev)
            loss = F.binary_cross_entropy(sq(lab[r, b["tgt"]]), torch.ones(len(xs), device=dev)) + \
                   F.binary_cross_entropy(sq(lab[r, b["neg"]]), torch.zeros(len(xs), device=dev))
            if lam > 0:                              # binarisation penalty over entity labels, passes 1..P
                msk = torch.zeros_like(lab)
                for j, x in enumerate(xs):
                    msk[j, : len(x["idx_to_symbol"])] = 1
                pen = sum(((t_ * (1 - t_)) * msk).sum() for t_ in traj[1:]) / (msk.sum() * a.P)
                loss = loss + lam * pen
            if a.arm in ("hint", "hint2"):
                for p in range(1, a.P + 1):
                    tg = torch.zeros_like(lab)
                    for j, x in enumerate(xs):
                        for v, dd in hops(x).items():
                            if dd <= p:
                                tg[j, v] = 1
                    msk = torch.zeros_like(lab)
                    for j, x in enumerate(xs):
                        msk[j, : len(x["idx_to_symbol"])] = 1
                    loss = loss + (F.binary_cross_entropy(sq(traj[p]), tg, reduction="none") * msk).sum() / msk.sum() / a.P
            (loss / a.accum).backward()
            accs.append(float(answer(lab.detach(), logit.detach(), b).float().mean())); losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); step += 1
        if step % 25 == 0 or step == steps:
            rec = {"step": step, "of": steps, "loss": float(np.mean(losses)), "train_acc": float(np.mean(accs)),
                   "sec": round(time.time() - t0)}
            if v2:
                rec.update({"tau": m.tau, "lam": lam, "dmax": dmax})
                if step % 100 == 0 or step == steps:     # hardened accuracy and label binarity on 64 training problems
                    rec.update(monitor(m, sub, a.P, dev, amp))
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"step {step}/{steps} loss {np.mean(losses):.3f} acc {np.mean(accs):.3f} {time.time() - t0:.0f}s", flush=True)
            accs, losses = [], []
    torch.save({"model": m.state_dict(), "args": vars(a), "tau_end": m.tau}, os.path.join(a.out, "model.pt"))
    print("TRAIN_DONE", a.out, f"{time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def monitor(m, xs, P, dev, amp):
    """Hardened and soft answer accuracy, and label binarity (share of soft labels within 0.1 of 0 or 1)."""
    mode, tau = m.mode, m.tau
    out = {}
    for name, md in (("hard", "hard"), ("soft", mode)):
        m.mode = md; m.tau = None if md == "hard" else tau
        acc, binar = [], []
        for i in range(0, len(xs), 8):
            b = collate(xs[i:i + 8], dev)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                lab, logit, _ = m.run(b, P)
            acc += answer(lab, logit, b).float().tolist()
            if name == "soft":
                for j, x in enumerate(xs[i:i + 8]):
                    v = lab[j, : len(x["idx_to_symbol"])]
                    binar.append(float(((v < 0.1) | (v > 0.9)).float().mean()))
        out[f"acc_{name}"] = float(np.mean(acc))
        if binar:
            out["binarity"] = float(np.mean(binar))
    m.mode, m.tau = mode, tau
    return out


@torch.no_grad()
def evaluate(a):
    dev = a.device
    ck = torch.load(os.path.join(a.eval, "model.pt"), map_location="cpu", weights_only=False)
    arm0 = ck["args"]["arm"]; arm = "free" if arm0 == "free" else "mono"
    sd = torch.load(ck["args"]["init"], map_location="cpu", weights_only=True)
    m = AIL(sd, arm); m.load_state_dict(ck["model"]); m.to(dev).eval()
    m.mode, m.tau = "hard", None                                              # hardened labels: OR at threshold 0.5
    soft_cfg = {"soft": ("soft", ck.get("tau_end")), "hint2": ("soft", ck.get("tau_end")), "stoch": ("soft", 1.0)}.get(arm0)
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    data = json.load(open(a.test)); rng = np.random.default_rng(0)
    excl = {int(i) for i in a.exclude.split(",") if i}
    seen, sel = {}, []
    for ii, x in enumerate(data):
        d = len(x["steps"])
        if ii in excl:
            continue
        if seen.get(d, 0) < a.n_per_d:
            seen[d] = seen.get(d, 0) + 1; sel.append(x)
    trainP = ck["args"]["P"]
    Ps = sorted({trainP, 2 * trainP, 14})
    rows = []
    for i in range(0, len(sel), a.bs):
        xs = sel[i:i + a.bs]; b = collate(xs, dev)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            lab, logit, traj = m.run(b, max(Ps), record=True)
            # edit (C7): switch ON, before pass 1, the farthest ancestor (<= 3 hops) of the negative option;
            # BFS predicts the negative option's label turns on and the target's label is unchanged
            ed = torch.zeros_like(b["start"]); has = []
            for j, x in enumerate(xs):
                radj = {}
                for u, v in x["edges"]:
                    radj.setdefault(v, []).append(u)
                d = {x["neg_target"]: 0}; q = deque([x["neg_target"]])
                while q:
                    u = q.popleft()
                    for v in radj.get(u, []):
                        if v not in d and d[u] < 3:
                            d[v] = d[u] + 1; q.append(v)
                cands = [v for v, dd in d.items() if dd == max(d.values()) and dd > 0]
                if cands:
                    ed[j, int(rng.choice(cands))] = 1; has.append(True)
                else:
                    has.append(False)
            lab_e, _, _ = m.run(b, max(Ps), edit=ed)
            # §54b-1 cue control: one added statement "<another person> is a <parent of the target>" (answer unchanged)
            xc = [cue_variant(x, rng) for x in xs]
            bc = collate([c if c is not None else x for c, x in zip(xc, xs)], dev)
            lab_c, logit_c, _ = m.run(bc, max(Ps))
            acc_c = answer(lab_c, logit_c, bc)
            if soft_cfg:                                                          # soft-test accuracy (AG0)
                m.mode, m.tau = soft_cfg
                lab_s, logit_s, _ = m.run(b, max(Ps))
                acc_s = answer(lab_s, logit_s, b)
                m.mode, m.tau = "hard", None
        r = torch.arange(len(xs), device=dev)
        for j, x in enumerate(xs):
            T = [traj[p][j] > 0.5 for p in range(len(traj))]
            ch = [bool((T[p] != T[p - 1]).any()) for p in range(1, len(T))]
            fp = next((p for p in range(len(ch)) if not ch[p]), len(ch))
            row = {"depth": len(x["steps"]), "fixed_point": fp}
            for P in Ps:
                lb = traj[P]
                lt, ln = bool(lb[j, x["target"]] > 0.5), bool(lb[j, x["neg_target"]] > 0.5)
                row[f"P{P}"] = bool(lt and not ln) if lt != ln else None             # None = tie (logit only at max P)
            row["acc"] = bool(answer(lab, logit, b)[j])
            if xc[j] is not None:
                row["acc_cue"] = bool(acc_c[j])
            if soft_cfg:
                row["acc_soft"] = bool(acc_s[j])
            h = hops(x); nV = len(x["idx_to_symbol"])
            fin = traj[max(Ps)][j, :nV].cpu().numpy() > 0.5
            reach = np.array([v in h for v in range(nV)])
            row["label_prec"] = float(fin[reach].sum() / max(1, fin.sum())); row["label_recall"] = float(fin[reach].mean())
            if has[j]:
                row["edit_neg_on"] = bool(lab_e[j, x["neg_target"]] > 0.5)
                row["edit_tgt_same"] = bool((lab_e[j, x["target"]] > 0.5) == (lab[j, x["target"]] > 0.5))
            rows.append(row)
    out = {"model": a.eval, "Ps": Ps, "n": len(rows), "rows": rows}
    json.dump(out, open(os.path.join(a.eval, "eval.json"), "w", newline="\n"))
    for lo, hi in ((3, 6), (7, 10), (11, 14)):
        s_ = [r for r in rows if lo <= r["depth"] <= hi]
        print(f"d{lo}-{hi} n {len(s_)} fixed_point median {np.median([r['fixed_point'] for r in s_])}", flush=True)
    print("EVAL_DONE", a.eval, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out"); ap.add_argument("--eval")
    ap.add_argument("--arm", default="mono", choices=["mono", "free", "hint", "soft", "stoch", "hint2"]); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init", default=os.path.join(HERE, "ckpt", "gpt2_pretrained.bin"))
    ap.add_argument("--train", default=os.path.join(HERE, "p8data", "p8_train.json"))
    ap.add_argument("--test", default=os.path.join(HERE, "p8data", "p8_test.json"))
    ap.add_argument("--max_depth", type=int, default=6); ap.add_argument("--P", type=int, default=6)
    ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--bs", type=int, default=4); ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--eps", type=float, default=1e-3); ap.add_argument("--n_per_d", type=int, default=100)
    ap.add_argument("--lab_std", type=float, default=0.02); ap.add_argument("--bias", type=float, default=-4.0)
    ap.add_argument("--tau_end", type=float, default=0.3); ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--soft_frac", type=float, default=0.0); ap.add_argument("--tau0", type=float, default=1.0)
    ap.add_argument("--tau1", type=float, default=0.1)
    ap.add_argument("--exclude", default="0,1,2,3,4,5,6,7,15,16,27,47",
                    help="p8_test indices excluded from evaluation (setting #0's plumbing check evaluated them)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.eval:
        evaluate(a)
    else:
        train(a)


if __name__ == "__main__":
    main()
