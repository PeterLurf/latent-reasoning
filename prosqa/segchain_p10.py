"""PROTOCOL10 (docs/PROTOCOL10.md): segmented Coconut with written entity checkpoints, inference only (THEORY §47/§48).

    python segchain_p10.py --ckpt <ckpt> --data p8data/p8_test.json --out <json> [--S 4] [--Kp 10] [--device cuda:1]

1. Calibration as in reanchor_p8.py: sweep head = the (layer, head) whose attention from thoughts onto statements puts
   the most mass on gold-path statements (first 100 problems with ext <= 2; none of them has depth >= 11).
2. Per problem (answer depth 11..14; the first 100 per depth, all of depth 14): segment 1 is a plain K'-thought episode.
   After segment s, the write step takes the top statement under the sweep head at the segment's last thought, excluding
   statements whose object is already written or is the root; its object Y is written as "<root> is a Y." at a random
   statement position, and segment s + 1 is a fresh K'-thought episode on the new text (isolation: no latent thought of
   an earlier segment is visible; the written text is the only channel). Halt (C5 v0) when Y is an answer option or no
   candidate is left; later S reuse the halting segment's answer.
3. Edit check (C7, secondary): on problems that reach segment 2, re-run segment 2 with the written entity replaced by a
   distractor at the same hop distance from the root that reaches the negative option but not the target (skipped if
   none). Reported: the share of such problems answered with the negative option, edited vs unedited.
Prints progress only (no accuracies) so the run can go ahead blind before the PROTOCOL8b step-1 verdict.
"""
import argparse
import json
from collections import deque

import numpy as np
import torch

from reanchor_p8 import STMT, lp_of, split, stmt_att, with_checkpoint
from gpt2c import GPT2C, load_state


def hops_from(edges, src, n):
    d = [None] * n; d[src] = 0; q = deque([src]); adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if d[v] is None:
                d[v] = d[u] + 1; q.append(v)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--S", type=int, default=4); ap.add_argument("--Kp", type=int, default=10)
    ap.add_argument("--n_per_d", type=int, default=100); ap.add_argument("--dmin", type=int, default=11)
    ap.add_argument("--dmax", type=int, default=14); ap.add_argument("--n_cal", type=int, default=100)
    ap.add_argument("--device", default="cuda:1"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude_cal", action="store_true", help="skip the calibration problems in the evaluation "
                    "(PROTOCOL10 amendment 2; they cannot occur at depth >= 11)")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = GPT2C(load_state(a.ckpt, a.device)); model.mask_answer = True
    rng = np.random.default_rng(a.seed)
    data = json.load(open(a.data))
    score = None
    cal = [x for x in data if x["ext"] <= 2][: a.n_cal]
    cal_ids = {id(x) for x in cal} if a.exclude_cal else set()
    for x in cal:
        _, P, att = lp_of(model, x, a.Kp, trace=True)
        S = stmt_att(att, P)
        gold = np.zeros(S.shape[-1], bool); gold[P["path"]] = True
        m = S[..., gold].sum(-1).mean(-1)
        score = m if score is None else score + m
    L_, H_ = np.unravel_index(np.argmax(score), score.shape)
    seen, rows = {}, []
    for x in data:
        d = len(x["steps"])
        if not a.dmin <= d <= a.dmax or seen.get(d, 0) >= a.n_per_d or id(x) in cal_ids:
            continue
        seen[d] = seen.get(d, 0) + 1
        sym = x["idx_to_symbol"]; ix = {s: i for i, s in enumerate(sym)}; n = len(sym)
        root, tgt, neg = x["root"], x["target"], x["neg_target"]
        dist = hops_from(x["edges"], root, n)
        gold_objs = {STMT.match(s).group(3) for s in x["steps"]}
        cur, written, margins, writes, halted = x, [], [], [], None
        for s in range(1, a.S + 1):
            lp, P, att = lp_of(model, cur, a.Kp, trace=True)
            margins.append(float(lp[0] - lp[1]))
            if s == a.S:
                break
            att_s = stmt_att(att, P)[L_, H_, -1]
            stmts, _ = split(cur)
            objs = [STMT.match(t).group(3) for t in stmts]
            y = next((objs[i] for i in np.argsort(-att_s) if objs[i] not in written and ix[objs[i]] != root), None)
            if y is None or ix[y] in (tgt, neg):
                halted = s; break
            written.append(y)
            writes.append({"entity": y, "on_path": y in gold_objs, "hops": dist[ix[y]]})
            pos = int(rng.integers(0, len(stmts) + 1))
            if s == 1:
                seg2_pos, seg2_prev = pos, cur
            cur = with_checkpoint(cur, y, pos)
        margins += [margins[-1]] * (a.S - len(margins))
        row = {"depth": d, "margins": margins, "halted": halted, "writes": writes}
        if writes and writes[0]["hops"] is not None:        # C7 edit check on segment 2
            h = writes[0]["hops"]
            dneg = {i for i in range(n) if hops_from(x["edges"], i, n)[neg] is not None}
            dtgt = {i for i in range(n) if hops_from(x["edges"], i, n)[tgt] is not None}
            cands = [i for i in range(n) if dist[i] == h and i in dneg and i not in dtgt and i not in (tgt, neg)
                     and sym[i][:1].islower()]
            if cands:
                z = sym[int(rng.choice(cands))]
                lp_e = lp_of(model, with_checkpoint(seg2_prev, z, seg2_pos), a.Kp)[0]
                row["edit"] = {"entity": z, "margin_edited": float(lp_e[0] - lp_e[1]), "margin_unedited": margins[1]}
        rows.append(row)
        if len(rows) % 25 == 0:
            print(f"{len(rows)} problems", flush=True)
    json.dump({"ckpt": a.ckpt, "Kp": a.Kp, "S": a.S, "sweep_layer": int(L_), "sweep_head": int(H_), "n": len(rows),
               "rows": rows}, open(a.out, "w"))
    print("DONE", a.out, len(rows), "problems")


if __name__ == "__main__":
    main()
