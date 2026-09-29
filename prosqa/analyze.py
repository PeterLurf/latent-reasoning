"""PROTOCOL6 addendum C: frontier sweep, step-specific blocking and end activity in Coconut GPT-2 on ProsQA.

    python prosqa/analyze.py --ckpt <checkpoint> --data <prosqa_*.json> --what trace|block|kprime --out <json> [--n N]

Terms (as in the synthetic task, docs/PROTOCOL5.md):
  * hop-h edge: a statement "X is a Y" / "Every X is a Y" whose subject is at BFS distance h-1 from the queried person
    (reachable); decoy hop-h edge: subject unreachable from the queried person and at distance h-1 from the other person.
  * thought step k: the forward position whose final hidden state becomes continuous thought k (k = 1 is the
    <start-latent> token). Per-edge attention = attention from that position to the statement's tokens, summed over
    the statement and averaged over heads, per layer.
  * excess(k, h) = mean per-edge attention on reachable hop-h edges - on decoy hop-h edges.
  * end activity (per layer) = max_h excess(K, h) / max_{k<=K, h} excess(k, h), h = 1..5.
  * blocking: attention from step k to the tokens of the answer path's hop-h statement is removed in every layer and
    head; effect = change of the answer margin log p(correct answer) - log p(other option).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import deque

import numpy as np
import tiktoken
import torch

from gpt2c import GPT2C, load_state

ENC = tiktoken.get_encoding("gpt2")
STMT = re.compile(r"^(?:Every (\w+)|(\w+)) is an? (\w+)\.$")


def bfs(src, adj, allowed=None):
    d = {src: 0}
    dq = deque([src])
    while dq:
        u = dq.popleft()
        for v in adj.get(u, []):
            if v not in d and (allowed is None or v in allowed):
                d[v] = d[u] + 1
                dq.append(v)
    return d


def parse(x):
    """Token ids of question + '\\n', per-statement token spans, edges and hop labels. Delayed-query samples
    (make_delayed.py, addendum J) keep the statements first, so spans are unchanged; the prefix ids end with
    " Start: <root>.\\n" and the query tokens " <query>\\n" are prepended to both candidate answers (they come right
    after <end-latent>, as in training; identical for both candidates, so margins are unaffected)."""
    if x.get("query"):
        stm = x["question"].rsplit(" Start: ", 1)[0]
        P = _parse({**x, "question": stm + " " + x["query"]})
        ids = ENC.encode(x["question"] + "\n")
        assert ids[: P["spans"][-1][1]] == P["ids"][: P["spans"][-1][1]], "statement tokens differ"
        mid = ENC.encode(" " + x["query"] + "\n")
        return {**P, "ids": ids, "answers": [mid + a for a in P["answers"]], "mid": mid}
    return _parse(x)


def _parse(x):
    sym = x["idx_to_symbol"]; ix = {s: i for i, s in enumerate(sym)}
    q = x["question"]
    parts = re.split(r"(?<=\.) ", q)
    stmts, question = parts[:-1], parts[-1]
    ids, spans, edges = [], [], []
    for i, s in enumerate(stmts):
        t = ENC.encode(s if i == 0 else " " + s)
        spans.append((len(ids), len(ids) + len(t)))
        ids += t
        m = STMT.match(s)
        a = m.group(1) or m.group(2)
        edges.append((ix[a], ix[m.group(3)]))
    ids += ENC.encode(" " + question + "\n")
    assert ids == ENC.encode(q + "\n"), "sentence-wise tokenisation differs"
    assert sorted(map(tuple, x["edges"])) == sorted(edges), "statements do not match edges"
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
    root = x["root"]
    dr = bfs(root, adj)
    people = [i for i, s in enumerate(sym) if s[:1].isupper() and i != root]
    unreach = set(range(len(sym))) - set(dr)
    do = {}
    for p in people:
        for n, dd in bfs(p, adj, unreach | {p}).items():
            do[n] = min(dd, do.get(n, 99))
    hop = [dr[u] + 1 if u in dr else None for u, _ in edges]
    dhop = [do[u] + 1 if u not in dr and u in do else None for u, _ in edges]
    path = []
    for s in x["steps"]:
        path.append(stmts.index(s))
    ans = x["answer"]
    tgt, neg = sym[x["target"]], sym[x["neg_target"]]
    a_neg = re.sub(rf"\b{tgt}\.$", f"{neg}.", ans)
    assert a_neg != ans
    answers = [ENC.encode("### " + ans), ENC.encode("### " + a_neg)]
    return {"ids": ids, "spans": spans, "hop": hop, "dhop": dhop, "path": path, "answers": answers}


def trace(model, data, K, n):
    L, H = model.L, 6
    acc = {k: np.zeros((L, K, H)) for k in ("r_sum", "r_n", "d_sum", "d_n", "p_sum", "p_n")}
    for x in data[:n]:
        P = parse(x)
        lp, qpos, att = model.rollout(P["ids"], K, P["answers"][:1], trace=True)
        A = att[:, 0].mean(1).cpu().numpy()                       # (L, K, T)
        per = np.stack([A[:, :, a:b].sum(-1) for a, b in P["spans"]], -1)    # (L, K, n_edges)
        for e, (h, dh) in enumerate(zip(P["hop"], P["dhop"])):
            if h is not None and h <= H:
                acc["r_sum"][:, :, h - 1] += per[:, :, e]; acc["r_n"][:, :, h - 1] += 1
            if dh is not None and dh <= H:
                acc["d_sum"][:, :, dh - 1] += per[:, :, e]; acc["d_n"][:, :, dh - 1] += 1
        for j, e in enumerate(P["path"]):
            acc["p_sum"][:, :, j] += per[:, :, e]; acc["p_n"][:, :, j] += 1
    R = acc["r_sum"] / np.maximum(acc["r_n"], 1)
    D = acc["d_sum"] / np.maximum(acc["d_n"], 1)
    Pp = acc["p_sum"] / np.maximum(acc["p_n"], 1)
    ex = R - D
    out = {"K": K, "n": n, "reachable": R.tolist(), "decoy": D.tolist(), "path": Pp.tolist(), "excess": ex.tolist(),
           "n_reach": acc["r_n"][0, 0].tolist(), "n_decoy": acc["d_n"][0, 0].tolist(), "n_path": acc["p_n"][0, 0].tolist()}
    e5 = ex[:, :, :5]
    pk = e5.max((1, 2))
    out["end_activity"] = [float(e5[l, K - 1].max() / pk[l]) if pk[l] > 0 else 0.0 for l in range(L)]
    out["frontier_score"] = [float(e5[l].max(1).mean()) for l in range(L)]      # layer choice (discovery split)
    out["path_clock"] = [[int(np.argmax(Pp[l, :, j])) + 1 for j in range(H)] for l in range(L)]
    return out


def block(model, data, K, n, chunk=16):
    """Delta margin for blocking the answer path's hop-h statement at thought step k, at every step ("all", positive
    control), and a same-hop decoy statement at step k (control). Per-example deltas are kept for bootstrap CIs."""
    H = 6
    rng = np.random.default_rng(0)
    rows = []
    for x in data[:n]:
        P = parse(x)
        T = len(P["ids"]) + 1 + K + 1 + max(map(len, P["answers"]))
        conds = [None]
        for h, e in enumerate(P["path"], 1):
            same = [i for i, dh in enumerate(P["dhop"]) if dh == h]
            ctrl = int(rng.choice(same)) if same else None
            conds.append(("all", 0, h, e))
            for k in range(1, K + 1):
                conds.append(("path", k, h, e))
                if ctrl is not None:
                    conds.append(("ctrl", k, h, ctrl))
        margins = []
        for i in range(0, len(conds), chunk):
            cc = conds[i:i + chunk]
            ex = torch.zeros(len(cc), T, T, device=model.wte.device)
            for b, c in enumerate(cc):
                if c is not None:
                    kind, k, _, e = c
                    a, bb = P["spans"][e]
                    ks = range(1, K + 1) if kind == "all" else [k]
                    for kk in ks:
                        ex[b, len(P["ids"]) + kk - 1, a:bb] = float("-inf")
            lp, _, _ = model.rollout(P["ids"], K, P["answers"], extra=ex)
            margins += (lp[:, 0] - lp[:, 1]).tolist()
        m0 = margins[0]
        rows.append({"base": m0, "path_len": len(P["path"]),
                     "d": [[c[0], c[1], c[2], m - m0] for c, m in zip(conds[1:], margins[1:]) if c[2] <= H]})
    s = {c: np.zeros((K + 1, H)) for c in ("path", "ctrl", "all")}; cnt = {c: np.zeros((K + 1, H)) for c in s}
    for r in rows:
        for kind, k, h, d in r["d"]:
            s[kind][k, h - 1] += d; cnt[kind][k, h - 1] += 1
    mean = lambda c: (s[c] / np.maximum(cnt[c], 1))
    base = np.array([r["base"] for r in rows])
    return {"K": K, "n": n, "base_margin_mean": float(base.mean()), "base_acc": float((base > 0).mean()),
            "delta_path": mean("path")[1:].tolist(), "n_path": cnt["path"][1:].tolist(),
            "delta_ctrl": mean("ctrl")[1:].tolist(), "n_ctrl": cnt["ctrl"][1:].tolist(),
            "delta_all": mean("all")[0].tolist(), "n_all": cnt["all"][0].tolist(), "per_example": rows}


def noreread(model, data, n, budgets=(0, 6)):
    """Exploratory: the answer positions (<end-latent> onwards) may not attend to any statement token, so only the
    thoughts (and the question sentence) can carry the graph. Candidate accuracy with K' thoughts."""
    out = {}
    for Kp in budgets:
        acc = []
        for x in data[:n]:
            P = parse(x)
            T = len(P["ids"]) + 1 + Kp + 1 + max(map(len, P["answers"]))
            ex = torch.zeros(1, T, T, device=model.wte.device)
            ex[0, len(P["ids"]) + 1 + Kp:, : P["spans"][-1][1]] = float("-inf")
            lp, _, _ = model.rollout(P["ids"], Kp, P["answers"], extra=ex)
            acc.append(float(lp[0, 0] - lp[0, 1]) > 0)
        out[Kp] = {"acc": float(np.mean(acc)), "n": len(acc)}
        print("noreread K'", Kp, out[Kp], flush=True)
    return out


def kprime(model, data, n, budgets=(0, 2, 4, 6, 8, 10, 12)):
    out = {}
    for Kp in budgets:
        by = {}
        for x in data[:n]:
            P = parse(x)
            lp, _, _ = model.rollout(P["ids"], Kp, P["answers"])
            by.setdefault(len(P["path"]), []).append(float(lp[0, 0] - lp[0, 1]) > 0)
        out[Kp] = {"acc": float(np.mean(sum(by.values(), []))), "by_path_len": {L: float(np.mean(v)) for L, v in sorted(by.items())},
                   "n_by_path_len": {L: len(v) for L, v in sorted(by.items())}}
        print("K'", Kp, out[Kp], flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--what", choices=["trace", "block", "kprime", "noreread"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mask_answer", action="store_true", help="addendum G model: answer cannot see the question")
    ap.add_argument("--budgets", default="0,2,4,6,8,10,12", help="K' values for --what kprime")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = GPT2C(load_state(a.ckpt, a.device))
    model.mask_answer = a.mask_answer
    data = json.load(open(a.data))
    res = {"trace": lambda: trace(model, data, a.K, a.n), "block": lambda: block(model, data, a.K, a.n),
           "kprime": lambda: kprime(model, data, a.n, tuple(int(b) for b in a.budgets.split(","))), "noreread": lambda: noreread(model, data, a.n)}[a.what]()
    res.update(ckpt=a.ckpt, data=a.data)
    json.dump(res, open(a.out, "w"), indent=1)
    print("DONE", a.what, a.out, flush=True)
