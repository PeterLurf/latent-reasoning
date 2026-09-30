"""PROTOCOL8b pilot (exploratory): re-anchoring a trained Coconut GPT-2 on deep ProsQA with a written, readable
checkpoint (THEORY §26 / §32 idea, carried to a pretrained model).

    python reanchor_p8.py --ckpt <ckpt> --data p8data/p8_test.json --out <json> [--n_per_m 150] [--Kp 10] [--device cuda:0]

1. Calibration (the first --n_cal problems with m <= 2, excluded from evaluation): pick the (layer, head) whose attention
   from thought positions 1..K onto statements puts the most mass on gold-path statements ("sweep head").
2. Per evaluation problem: baseline episode (K' thoughts). At the last thought, the sweep head's attention per statement
   gives the top-3 statements; the anchor is each statement's object entity ("Every X is a Y." / "Name is a Y." -> Y).
3. Restart: for each anchor Y the checkpoint is written as text, "<root> is a Y.", appended to the statements, and a
   fresh K'-thought episode is run. The answer combines the episodes by max log-probability per option.
4. Controls with the same combination: random (3 random concept entities as anchors) and oracle (the gold-path entity
   4 hops before the target, or the root's first step if the path is shorter).
Reports candidate accuracy by extension m and by answer depth for baseline / reanchor / random / oracle, plus anchor
statistics (share of top-1 anchors on the gold path; the anchor's hop).
"""
import argparse
import json
import re

import numpy as np
import torch

from analyze import parse
from gpt2c import GPT2C, load_state

STMT = re.compile(r"^(?:Every (\w+)|(\w+)) is an? (\w+)\.$")


def split(x):
    parts = re.split(r"(?<=\.) ", x["question"])
    return parts[:-1], parts[-1]


def with_checkpoint(x, anchor_sym):
    """x with the statement '<root> is a <anchor>.' added (and its edge), everything else unchanged."""
    stmts, q = split(x)
    sym = x["idx_to_symbol"]; root = sym[x["root"]]
    art = "an" if anchor_sym[:1] in "aeiou" else "a"
    new = f"{root} is {art} {anchor_sym}."
    ix = {s: i for i, s in enumerate(sym)}
    return {**x, "question": " ".join(stmts + [new]) + " " + q, "edges": x["edges"] + [[x["root"], ix[anchor_sym]]]}


@torch.no_grad()
def lp_of(model, x, Kp, trace=False):
    P = parse(x)
    lp, _, att = model.rollout(P["ids"], Kp, P["answers"], trace=trace)
    return lp[0].cpu().numpy(), P, att


def stmt_att(att, P):
    """att (L, 1, H, K, T) -> (L, H, K, n_stmt): attention mass per statement from each thought position."""
    A = att[:, 0]
    return torch.stack([A[..., s:e].sum(-1) for s, e in P["spans"]], -1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n_per_m", type=int, default=150); ap.add_argument("--n_cal", type=int, default=100)
    ap.add_argument("--Kp", type=int, default=10); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = GPT2C(load_state(a.ckpt, a.device)); model.mask_answer = True
    rng = np.random.default_rng(a.seed)
    data = json.load(open(a.data))
    cal = [x for x in data if x["ext"] <= 2][: a.n_cal]
    cal_ids = {id(x) for x in cal}
    # 1. sweep head
    score = None
    for x in cal:
        _, P, att = lp_of(model, x, a.Kp, trace=True)
        S = stmt_att(att, P)                                        # (L, H, K, n)
        gold = np.zeros(S.shape[-1], bool); gold[P["path"]] = True
        m = S[..., gold].sum(-1).mean(-1)                          # (L, H): gold-path mass averaged over thoughts
        score = m if score is None else score + m
    L_, H_ = np.unravel_index(np.argmax(score), score.shape)
    # 2-4. evaluation
    seen, rows = {}, []
    for x in data:
        if id(x) in cal_ids or seen.get(x["ext"], 0) >= a.n_per_m:
            continue
        seen[x["ext"]] = seen.get(x["ext"], 0) + 1
        base, P, att = lp_of(model, x, a.Kp, trace=True)
        S = stmt_att(att, P)[L_, H_, -1]                            # last thought, sweep head: (n_stmt,)
        stmts, _ = split(x)
        objs = [STMT.match(s).group(3) for s in stmts]
        top = list(np.argsort(-S)[:3])
        anchors = [objs[i] for i in top]
        sym = x["idx_to_symbol"]
        concepts = [s for s in sym if s[:1].islower() and s not in (sym[x["target"]], sym[x["neg_target"]])]
        rand = list(rng.choice(concepts, 3, replace=False))
        path_objs = [objs[i] for i in P["path"]]
        oracle = path_objs[max(0, len(path_objs) - 5)]             # 4 hops before the target
        comb = lambda anc: np.maximum.reduce([base] + [lp_of(model, with_checkpoint(x, y), a.Kp)[0] for y in anc])
        re_lp, rnd_lp, or_lp = comb(anchors), comb(rand), comb([oracle])
        hop_top1 = P["hop"][top[0]]
        rows.append({"m": x["ext"], "depth": len(x["steps"]), "base": float(base[0] - base[1]),
                     "reanchor": float(re_lp[0] - re_lp[1]), "random": float(rnd_lp[0] - rnd_lp[1]),
                     "oracle": float(or_lp[0] - or_lp[1]), "top1_on_path": bool(top[0] in P["path"]),
                     "top1_hop": hop_top1})
    R = rows
    def acc(key, by):
        vals = sorted(set(r[by] for r in R))
        return {int(v): float(np.mean([r[key] > 0 for r in R if r[by] == v])) for v in vals}
    out = {"ckpt": a.ckpt, "Kp": a.Kp, "sweep_layer": int(L_), "sweep_head": int(H_), "n": len(R),
           "by_m": {k: acc(k, "m") for k in ("base", "reanchor", "random", "oracle")},
           "by_depth": {k: acc(k, "depth") for k in ("base", "reanchor", "random", "oracle")},
           "top1_on_path_share": float(np.mean([r["top1_on_path"] for r in R])), "rows": R}
    json.dump(out, open(a.out, "w"))
    print("sweep head", L_, H_, "top1 on path", round(out["top1_on_path_share"], 3))
    for k in ("base", "reanchor", "random", "oracle"):
        print(k, {d: round(v, 3) for d, v in out["by_depth"][k].items()})
    print("DONE", a.out)


if __name__ == "__main__":
    main()
