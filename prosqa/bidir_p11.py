"""PROTOCOL11 (docs/PROTOCOL11.md): bidirectional latent search in Coconut GPT-2 on ProsQA, zero-shot (THEORY §52 F2,
non-toy test; the toy version is THEORY §53).

    python bidir_p11.py --ckpt <ckpt> --data p8data/p8_test.json --out <json> [--m 10] [--Kp 10] [--device cuda:0]

Forward episode: the problem as is (K' thoughts). Backward episode for each answer option O: every statement reversed
("Every A is a B." -> "Every B is a A."; "Name is a B." -> "Every B is a Name."), plus "<Z> is a O." for a fresh person Z,
and the question "Is <Z> a <root> or <other person>?" (K' thoughts). Frontier of an episode = the objects of the m
statements with the most sweep-head attention summed over all thoughts (sweep head calibrated as in reanchor_p8.py on
the forward problems). Answer: if the forward frontier meets the backward frontier of exactly one option, that option;
otherwise the forward episode's answer. Prints progress only.
"""
import argparse
import json
import re

import numpy as np
import torch

from reanchor_p8 import STMT, lp_of, split, stmt_att
from gpt2c import GPT2C, load_state

NAMES = ["Zed", "Sam", "Max", "Rex", "Jon", "Kim", "Lou", "Ned"]


def reverse(x, opt):
    """Backward problem from option index opt: reversed statements, a fresh person Z with 'Z is a <opt>.'."""
    stmts, _ = split(x)
    sym = list(x["idx_to_symbol"]); root = sym[x["root"]]
    people = [s for s in sym if s[:1].isupper() and s != root]
    z = next(n for n in NAMES if n not in sym)
    sym.append(z); ix = {s: i for i, s in enumerate(sym)}
    rev, edges = [], []
    for s in stmts:
        m = STMT.match(s); a, b = m.group(1) or m.group(2), m.group(3)
        art = "an" if a[:1].lower() in "aeiou" else "a"
        rev.append(f"Every {b} is {art} {a}."); edges.append([ix[b], ix[a]])
    o = sym[opt]
    art = "an" if o[:1] in "aeiou" else "a"
    rev.append(f"{z} is {art} {o}."); edges.append([ix[z], ix[o]])
    other = people[0] if people else next(t for t in sym if t[:1].islower() and t != o)
    q = f"Is {z} a {root} or {other}?"
    return {"question": " ".join(rev) + " " + q, "answer": f"{z} is a {root}.", "idx_to_symbol": sym, "edges": edges,
            "root": ix[z], "target": ix[root], "neg_target": ix[other], "steps": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--m", type=int, nargs="+", default=[10, 5, 15]); ap.add_argument("--Kp", type=int, default=10)
    ap.add_argument("--n_per_d", type=int, default=100); ap.add_argument("--dmin", type=int, default=3)
    ap.add_argument("--dmax", type=int, default=14); ap.add_argument("--n_cal", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save_sets", action="store_true", help="PROTOCOL11 amendment 1: store the top-15 forward and backward "
                    "frontiers with their gold labels (reached from the root / reaching the option) for per-side precision")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = GPT2C(load_state(a.ckpt, a.device)); model.mask_answer = True
    data = json.load(open(a.data))
    cal = [x for x in data if x["ext"] <= 2][: a.n_cal]; cal_ids = {id(x) for x in cal}
    score = None
    for x in cal:
        _, P, att = lp_of(model, x, a.Kp, trace=True)
        S = stmt_att(att, P)
        gold = np.zeros(S.shape[-1], bool); gold[P["path"]] = True
        mm = S[..., gold].sum(-1).mean(-1)
        score = mm if score is None else score + mm
    L_, H_ = np.unravel_index(np.argmax(score), score.shape)
    seen, rows = {}, []
    for x in data:
        d = len(x["steps"])
        if not a.dmin <= d <= a.dmax or seen.get(d, 0) >= a.n_per_d or id(x) in cal_ids:
            continue
        seen[d] = seen.get(d, 0) + 1
        lp, P, att = lp_of(model, x, a.Kp, trace=True)
        Sf = stmt_att(att, P)[L_, H_].sum(0)
        stmts, _ = split(x)
        objs = [STMT.match(s).group(3) for s in stmts]
        order_f = [objs[i] for i in np.argsort(-Sf)]
        back = {}
        for name, opt in (("t", x["target"]), ("n", x["neg_target"])):
            xb = reverse(x, opt)
            lpb, Pb, attb = lp_of(model, xb, a.Kp, trace=True)
            Sb = stmt_att(attb, Pb)[L_, H_].sum(0)
            sb, _ = split(xb)
            ob = [STMT.match(s).group(3) for s in sb]
            back[name] = [ob[i] for i in np.argsort(-Sb)]
        row = {"depth": d, "fwd": float(lp[0] - lp[1])}
        if a.save_sets:
            sym = x["idx_to_symbol"]; ix = {v: i for i, v in enumerate(sym)}; adj, radj = {}, {}
            for u, v in x["edges"]:
                adj.setdefault(u, []).append(v); radj.setdefault(v, []).append(u)
            def closure(src, g):
                seen, st = {src}, [src]
                while st:
                    u = st.pop()
                    for v in g.get(u, []):
                        if v not in seen:
                            seen.add(v); st.append(v)
                return seen
            fr, bt_true, bn_true = closure(x["root"], adj), closure(x["target"], radj), closure(x["neg_target"], radj)
            row["F"] = [[e, ix[e] in fr] for e in order_f[:15]]
            row["Bt"] = [[e, e in ix and ix[e] in bt_true] for e in back["t"][:15]]
            row["Bn"] = [[e, e in ix and ix[e] in bn_true] for e in back["n"][:15]]
        for m in a.m:
            F = set(order_f[:m]); bt, bn = set(back["t"][:m]), set(back["n"][:m])
            row[f"hit_t_m{m}"], row[f"hit_n_m{m}"] = bool(F & bt), bool(F & bn)
        rows.append(row)
        if len(rows) % 50 == 0:
            print(f"{len(rows)} problems", flush=True)
    json.dump({"ckpt": a.ckpt, "Kp": a.Kp, "m": a.m, "sweep_layer": int(L_), "sweep_head": int(H_), "n": len(rows),
               "rows": rows}, open(a.out, "w"))
    print("DONE", a.out, len(rows))


if __name__ == "__main__":
    main()
