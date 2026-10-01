"""THEORY §60 Phase A: data pools, training (BM-oc / BM-no) and evaluation.

    python bm/run.py gen  --out bm/data_pool --seed 0                 # fixed train pool + dev/eval sets (pickles)
    python bm/run.py train --pool bm/data_pool --out bm/runs/oc_s1 --lam 1 --seed 1 --steps N --bs B [--device cuda:0]
    python bm/run.py eval --pool bm/data_pool --run bm/runs/oc_s1 [--device cuda:0]
"""
import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bm.data import (HELDOUT_FAMILIES, MAXP, PATTERNS, TRAIN_FAMILIES, make_set, random_schedule,  # noqa: E402
                     row_mask, schedule_tables, true_table)
from bm.model import BM  # noqa: E402

NONE = 8


# ---------------------------------------------------------------- data pools
def with_alt(rng, t, mode):
    """Attach a second schedule of the same region (for order consistency / associativity)."""
    s = random_schedule(rng, t["edges"], set(t["pinned"]), mode)
    if s is None:
        return None
    nodes, kids = s; ports, depth = schedule_tables(t["edges"], t["pinned"], nodes, kids)
    t["alt"] = {"nodes": [sorted(x) for x in nodes], "kids": kids, "ports": ports, "depth": depth}
    return t


def gen(a):
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    meta = {}
    def build(name, n, fams, kmin, kmax, mode, alt_mode=None):
        out, rej = [], {"tried": {}, "rejected": {}}
        while len(out) < n:
            ts, r = make_set(rng, n - len(out), fams, kmin, kmax, mode)
            for f, c in r["tried"].items():
                rej["tried"][f] = rej["tried"].get(f, 0) + c
            for f, d in r["rejected"].items():
                for k, c in d.items():
                    rej["rejected"].setdefault(f, {})[k] = rej["rejected"].get(f, {}).get(k, 0) + c
            for t in ts:
                if alt_mode is None or with_alt(rng, t, alt_mode) is not None:
                    out.append(t)
        pickle.dump(out, open(os.path.join(a.out, name + ".pkl"), "wb"))
        meta[name] = {"n": len(out), "rejection": rej, "label_mean": float(np.mean([t["label"] for t in out])),
                      "nP": {p: sum(len(t["pinned"]) == p for t in out) for p in (2, 3, 4)}}
        print(name, meta[name]["n"], "label mean %.3f" % meta[name]["label_mean"], flush=True)
    build("train", a.n_train, TRAIN_FAMILIES, 2, 8, "train", "train")
    build("dev", 2000, TRAIN_FAMILIES, 2, 8, "train", "train")
    build("ev_instances", 3000, TRAIN_FAMILIES, 2, 8, "train", "train")
    build("ev_motifs", 3000, HELDOUT_FAMILIES, 2, 8, "train", "train")
    build("ev_orders", 3000, TRAIN_FAMILIES, 6, 8, "heldout", "heldout")
    build("ev_size", 1500, TRAIN_FAMILIES, 9, 16, "any")
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w", newline="\n"), indent=1)


# ---------------------------------------------------------------- batching
def build_batch(tasks, rng, sched="main"):
    """Flatten the schedules of a batch into leaf and merge-step tensors (CPU numpy)."""
    leaves, merges, roots, info = [], [], [], []
    off = 0
    for ti, t in enumerate(tasks):
        s = t if sched == "main" else t["alt"]
        nodes, kids, ports = s["nodes"], s["kids"], [list(p) for p in s["ports"]]
        R = len(nodes) - 1
        for i in range(R):                                    # random slot order for every non-root node
            rng.shuffle(ports[i])
        nP = len(ports[R])                                    # root too (note 60-3): slot s holds pinned[perm[s]]
        perm = list(rng.permutation(nP)); ports[R] = [ports[R][k] for k in perm]
        canon_row, canon_col = canon_index(perm)
        E = sum(k is None for k in kids)
        for i in range(E):
            u, v = t["edges"][nodes[i][0]]
            sl = lambda x: ports[i].index(x) if x in ports[i] else MAXP
            leaves.append((off + i, sl(u), sl(v), len(ports[i])))
        m = 0
        for i in range(E, len(nodes)):
            a_, b_ = kids[i]
            vs = sorted(set(ports[a_]) | set(ports[b_]) | set(ports[i]))
            perm = rng.permutation(8)
            lid = {v: int(perm[j]) for j, v in enumerate(vs)}
            ids = lambda p: [lid[v] for v in p] + [NONE] * (MAXP - len(p))
            wire = np.zeros((8, 3), np.float32)
            for v in vs:
                wire[lid[v]] = [v in ports[a_], v in ports[b_], v in ports[i]]
            merges.append((m, off + i, off + a_, off + b_, ids(ports[a_]), ids(ports[b_]), ids(ports[i]),
                           len(ports[a_]), len(ports[b_]), len(ports[i]), wire))
            m += 1
        roots.append(off + R)
        info.append({"depth": s["depth"][R], "k": t["k"], "nodes_depth": s["depth"], "crow": canon_row, "ccol": canon_col})
        off += len(nodes)
    return {"n_nodes": off, "leaves": leaves, "merges": merges, "roots": roots, "info": info}


def canon_index(perm):
    """Index arrays mapping a root table over permuted slots back to canonical (pinned-order) layout:
    T_can[r, j] = T[crow[r], ccol[j]]. perm[s] = pinned index held by slot s."""
    nP = len(perm); inv = {perm[s]: s for s in range(nP)}
    crow = []
    for r, pat in enumerate(PATTERNS):                    # canonical row: bit k = seed on pinned index k
        if any(pat[k] for k in range(nP, MAXP)):
            crow.append(r); continue                      # padding row: stays padding
        bits = [0] * MAXP
        for k in range(nP):
            bits[inv[k]] = pat[k]
        crow.append(PATTERNS.index(tuple(bits)))
    ccol = [inv[j] if j < nP else j for j in range(MAXP)]
    return crow, ccol


def masks(n, dev):
    rm = torch.tensor(np.stack([row_mask(k)[0] for k in n]), device=dev)
    cm = torch.tensor(np.stack([row_mask(k)[1] for k in n]), device=dev)
    return rm, cm


def forward(model, bb, dev):
    """Run all schedules of the batch; returns the root tables (B, 16, 4) and all node tables (list)."""
    tab = [None] * bb["n_nodes"]
    L = bb["leaves"]
    rm, cm = masks([x[3] for x in L], dev)
    T = model.leaves(torch.tensor([x[1] for x in L], device=dev), torch.tensor([x[2] for x in L], device=dev), rm, cm)
    for j, x in enumerate(L):
        tab[x[0]] = T[j]
    M = bb["merges"]
    for m in range(max((x[0] for x in M), default=-1) + 1):
        S = [x for x in M if x[0] == m]
        TA = torch.stack([tab[x[2]] for x in S]); TB = torch.stack([tab[x[3]] for x in S])
        idA, idB, idP = (torch.tensor([x[k] for x in S], device=dev) for k in (4, 5, 6))
        rmA, _ = masks([x[7] for x in S], dev); rmB, _ = masks([x[8] for x in S], dev); rmP, cmP = masks([x[9] for x in S], dev)
        wire = torch.tensor(np.stack([x[10] for x in S]), device=dev)
        P = model.merge(TA, TB, idA, idB, idP, rmA, rmB, rmP, cmP, wire)
        for j, x in enumerate(S):
            tab[x[1]] = P[j]
    R = torch.stack([tab[r] for r in bb["roots"]])         # permuted slots -> canonical pinned order (bookkeeping)
    crow = torch.tensor([i["crow"] for i in bb["info"]], device=dev); ccol = torch.tensor([i["ccol"] for i in bb["info"]], device=dev)
    R = torch.gather(R, 1, crow[:, :, None].expand(-1, -1, MAXP))
    R = torch.gather(R, 2, ccol[:, None, :].expand(-1, 16, -1))
    return R, tab


# ---------------------------------------------------------------- training
def train(a):
    dev = a.device
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    pool = pickle.load(open(os.path.join(a.pool, "train.pkl"), "rb"))
    if a.kmax:                                                   # diagnostic only: restrict training region size
        pool = [t for t in pool if t["k"] <= a.kmax]
    model = BM().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n")
    sq = lambda x: x * (1 - 2e-3) + 1e-3
    t0 = time.time(); hist = []
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1, step / 200) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        tasks = [pool[i] for i in rng.integers(0, len(pool), a.bs)]
        bb = build_batch(tasks, rng)
        root, _ = forward(model, bb, dev)
        r = torch.arange(len(tasks), device=dev)
        rows = torch.tensor([t["row"] for t in tasks], device=dev); cols = torch.tensor([t["col"] for t in tasks], device=dev)
        y = torch.tensor([t["label"] for t in tasks], device=dev)
        pr = root[r, rows, cols]
        loss = F.binary_cross_entropy(sq(pr), y)
        lo = torch.zeros((), device=dev)
        if a.lam > 0:
            root2, _ = forward(model, build_batch(tasks, rng, "alt"), dev)
            rm, cm = masks([len(t["pinned"]) for t in tasks], dev)
            mk = (rm[..., None] & cm[:, None, :]).float()
            lo = (((root - root2) ** 2) * mk).sum() / mk.sum()
            loss = loss + a.lam * lo
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        hist.append((loss.item(), float(((pr > 0.5).float() == y).float().mean()), lo.item()))
        if step % 100 == 0 or step == a.steps:
            h = np.array(hist); hist = []
            rec = {"step": step, "loss": float(h[:, 0].mean()), "acc": float(h[:, 1].mean()), "order_mse": float(h[:, 2].mean()),
                   "sec": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, flush=True)
    torch.save({"model": model.state_dict(), "args": vars(a)}, os.path.join(a.out, "model.pt"))
    print("TRAIN_DONE", a.out, f"{time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- evaluation
def laws_ok(T, nP):
    f = {}
    for r in range(16):
        if row_mask(nP)[0][r]:
            f[PATTERNS[r]] = tuple(int(T[r, j] > 0.5) if j < nP else 0 for j in range(4))
    for a_, fa in f.items():
        if any(fa[j] < a_[j] for j in range(4)) or f[fa] != fa:
            return False
        for b_, fb in f.items():
            if all(a_[j] <= b_[j] for j in range(4)) and any(fa[j] > fb[j] for j in range(4)):
                return False
            u = tuple(max(a_[j], b_[j]) for j in range(4))
            if f[u] != tuple(max(fa[j], fb[j]) for j in range(4)):
                return False
    return True


@torch.no_grad()
def run_set(model, tasks, dev, rng, sched="main", bs=256):
    roots, infos = [], []
    for i in range(0, len(tasks), bs):
        bb = build_batch(tasks[i:i + bs], rng, sched)
        R, _ = forward(model, bb, dev)
        roots.append(R.float().cpu().numpy()); infos += bb["info"]
    return np.concatenate(roots), infos


def ridge_cv(X, y, k=5, lam=1.0):
    """Cross-validated multiclass accuracy of a ridge one-vs-rest classifier, vs the majority baseline."""
    cls = sorted(set(y)); y = np.array(y); n = len(y)
    idx = np.random.default_rng(0).permutation(n); folds = np.array_split(idx, k); acc = []
    for f in folds:
        tr = np.setdiff1d(idx, f)
        Xt = np.c_[X[tr], np.ones(len(tr))]; Y = np.stack([(y[tr] == c).astype(float) for c in cls], 1)
        W = np.linalg.solve(Xt.T @ Xt + lam * np.eye(Xt.shape[1]), Xt.T @ Y)
        pred = np.array(cls)[np.argmax(np.c_[X[f], np.ones(len(f))] @ W, 1)]
        acc.append(np.mean(pred == y[f]))
    maj = max(np.mean(y == c) for c in cls)
    return float(np.mean(acc)), float(maj)


def evaluate(a):
    dev = a.device
    ck = torch.load(os.path.join(a.run, "model.pt"), map_location=dev, weights_only=False)
    model = BM().to(dev); model.load_state_dict(ck["model"]); model.eval()
    rng = np.random.default_rng(123)
    out = {}
    names = a.sets.split(",") if a.sets else ["ev_instances", "ev_motifs", "ev_orders", "ev_size", "dev"]
    sets = {k: pickle.load(open(os.path.join(a.pool, k + ".pkl"), "rb")) for k in names}
    for name, tasks in sets.items():
        R, info = run_set(model, tasks, dev, rng)
        probe = np.mean([(R[i, t["row"], t["col"]] > 0.5) == (t["label"] > 0.5) for i, t in enumerate(tasks)])
        ent, law = [], []
        for i, t in enumerate(tasks):
            nP = len(t["pinned"]); rm, cm = row_mask(nP); TR = np.array(t["table"])
            ent.append(np.mean(((R[i] > 0.5) == (TR > 0.5))[rm][:, cm])); law.append(laws_ok(R[i], nP))
        out[name] = {"probe": float(probe), "entry": float(np.mean(ent)), "laws": float(np.mean(law)),
                     "probe_nP2": float(np.mean([(R[i, t["row"], t["col"]] > 0.5) == (t["label"] > 0.5) for i, t in enumerate(tasks) if len(t["pinned"]) == 2])),
                     "probe_nP34": float(np.mean([(R[i, t["row"], t["col"]] > 0.5) == (t["label"] > 0.5) for i, t in enumerate(tasks) if len(t["pinned"]) > 2]))}
        if name == "ev_orders":                                   # residual associativity: two held-out orders
            R2, _ = run_set(model, tasks, dev, rng, "alt")
            d, dis = [], []
            for i, t in enumerate(tasks):
                rm, cm = row_mask(len(t["pinned"])); mk = rm[:, None] & cm[None, :]
                d.append(np.abs(R[i] - R2[i])[mk].mean()); dis.append(((R[i] > 0.5) != (R2[i] > 0.5))[mk].mean())
            out[name]["assoc_abs"] = float(np.mean(d)); out[name]["assoc_disagree"] = float(np.mean(dis))
        if name == "ev_instances":                                # age audit: within groups of identical true tables
            key = [json.dumps(t["table"]) + str(len(t["pinned"])) for t in tasks]
            groups = {}
            for i, k in enumerate(key):
                groups.setdefault(k, []).append(i)
            big = [g for g in groups.values() if len(g) >= 20]
            res = []
            for g in big:
                X = R[g].reshape(len(g), -1)
                for target in ("depth", "k"):
                    y = [info[i]["depth"] if target == "depth" else tasks[i]["k"] for i in g]
                    if len(set(y)) > 1:
                        acc, maj = ridge_cv(X, y)
                        res.append({"target": target, "n": len(g), "acc": acc, "majority": maj})
            out["age_audit"] = {t: {"groups": sum(r["target"] == t for r in res),
                                    "mean_acc": float(np.mean([r["acc"] for r in res if r["target"] == t] or [np.nan])),
                                    "mean_majority": float(np.mean([r["majority"] for r in res if r["target"] == t] or [np.nan]))}
                                for t in ("depth", "k")}
            marg = {}
            for i, t in enumerate(tasks):
                rm, cm = row_mask(len(t["pinned"])); mk = rm[:, None] & cm[None, :]
                marg.setdefault(info[i]["depth"], []).append(float(np.abs(R[i] - 0.5)[mk].mean()))
            out["margin_by_depth"] = {int(k): float(np.mean(v)) for k, v in sorted(marg.items())}
    model.binarise = True                                         # binarised-commit check
    for name in [n for n in ("ev_instances", "ev_orders") if n in sets]:
        R, _ = run_set(model, sets[name], dev, rng)
        out[name]["probe_binarised"] = float(np.mean([(R[i, t["row"], t["col"]] > 0.5) == (t["label"] > 0.5) for i, t in enumerate(sets[name])]))
    fn = "eval.json" if not a.sets else "eval_" + a.sets.replace(",", "_") + ".json"
    json.dump(out, open(os.path.join(a.run, fn), "w", newline="\n"), indent=1)
    print(json.dumps({k: v for k, v in out.items()}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "train", "eval"])
    ap.add_argument("--out"); ap.add_argument("--pool", default="bm/data_pool"); ap.add_argument("--run")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n_train", type=int, default=50000)
    ap.add_argument("--lam", type=float, default=0.0); ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--kmax", type=int, default=0, help="diagnostic: train only on regions with <= kmax edges")
    ap.add_argument("--sets", default="", help="eval only these sets (the development pilot uses dev only)")
    a = ap.parse_args()
    {"gen": gen, "train": train, "eval": evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
