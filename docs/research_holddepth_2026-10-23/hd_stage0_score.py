"""HOLD-DEPTH Stage 0 scorer (HD_draft.md §2), $0: reads hd_check_cpu.<run>.json (check B) for the bases and the far cells, writes hd_stage0.out.json.
    python -B docs/research_holddepth_2026-10-23/hd_stage0_score.py [--fixtures]
Pairs (same seed): RD s1 / FH s1 vs base s1 (s213_d3t_s1); RD s2 / FH s2 vs base s2 (s216_d3t_s2). RD s3 has no same-seed base: shown vs both, never labelled.
Aggregates over the deep items (n 9..11 pooled; K' = n + 2 is the answer read):
  U_share     share of wrong items whose final state R(V_{n+2}) is unfinished (alive_fin != 1)
  act_late    mean over items of mean act[t], t in [9, n]; act_early: t in [2, 7]; ratios arm / base
  idle_pre    mean idle[t], t in [8, n - 1]
  PA_S, PA_nS P(answer right | S), P(answer right | not S)  (S = finished final state holding the gold value)
  ahead       share of items with t_ans <= n - 1;  PAR share: of the items right in base and wrong in the arm (same item), the share with base t_ans <= n - 1
Pre-stated labels per pair (HD_draft §2.3):
  SETTLE   dU >= +0.15 and (act_late ratio <= 0.70 or d idle_pre >= +0.15)
  VALUE    |dU| < 0.15 and 0.70 < act_late ratio < 1.43 and |d idle_pre| < 0.15
  MIXED    otherwise
  Supporting (not labels): REPAIR-SIG d PA_nS >= +0.05 (>= 10 not-S items in both); PAR-SIG PAR share >= 0.60 and d ahead <= -0.05.
Mechanism reading = the RD label if RD s1 and s2 agree, else MIXED. FH pairs are descriptive."""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = {"s1": "s213_d3t_s1", "s2": "s216_d3t_s2"}
ARMS = {("rd", "s1"): "s224_rd_s1", ("rd", "s2"): "s224_rd_s2", ("rd", "s3"): "s224_rd_s3", ("fh", "s1"): "s224_fh_s1", ("fh", "s2"): "s224_fh_s2"}


def agg(o):
    R = [(int(n), r) for n, rs in o["deep"].items() for r in rs]
    wrong = [(n, r) for n, r in R if r["ans"][n + 2] != r["y"]]
    act_l = [np.mean([r["act"][t] for t in range(9, n + 1)]) for n, r in R]
    act_e = [np.mean([r["act"][t] for t in range(2, 8)]) for n, r in R]
    idle = [np.mean([r["idle"][t] for t in range(8, n)]) for n, r in R]
    S = [(n, r) for n, r in R if r["S"]]; nS = [(n, r) for n, r in R if not r["S"]]
    pa = lambda xs: float(np.mean([r["ans"][n + 2] == r["y"] for n, r in xs])) if xs else None   # noqa: E731
    return {"items": len(R), "acc": pa(R), "U_share": float(np.mean([not r["finished"] for _, r in wrong])) if wrong else None, "n_wrong": len(wrong),
            "act_late": float(np.mean(act_l)), "act_early": float(np.mean(act_e)), "idle_pre": float(np.mean(idle)),
            "PA_S": pa(S), "PA_nS": pa(nS), "n_S": len(S), "n_nS": len(nS),
            "ahead": float(np.mean([r["t_ans"] is not None and r["t_ans"] <= n - 1 for n, r in R]))}


def par_share(ob, oa):
    k = []
    for n in ob["deep"]:
        for rb, ra in zip(ob["deep"][n], oa["deep"][n]):
            assert rb["y"] == ra["y"], "item order differs"
            nn = int(n)
            if rb["ans"][nn + 2] == rb["y"] and ra["ans"][nn + 2] != ra["y"]:
                k.append(rb["t_ans"] is not None and rb["t_ans"] <= nn - 1)
    return (float(np.mean(k)) if k else None), len(k)


def judge(A, B, ps):
    dU = A["U_share"] - B["U_share"]; ar = A["act_late"] / B["act_late"]; di = A["idle_pre"] - B["idle_pre"]
    if dU >= 0.15 and (ar <= 0.70 or di >= 0.15):
        lab = "SETTLE"
    elif abs(dU) < 0.15 and 0.70 < ar < 1.43 and abs(di) < 0.15:
        lab = "VALUE"
    else:
        lab = "MIXED"
    rep = A["n_nS"] >= 10 and B["n_nS"] >= 10 and A["PA_nS"] - B["PA_nS"] >= 0.05
    par = ps[0] is not None and ps[0] >= 0.60 and A["ahead"] - B["ahead"] <= -0.05
    return {"label": lab, "dU": dU, "act_late_ratio": ar, "act_early_ratio": A["act_early"] / B["act_early"], "d_idle_pre": di,
            "d_PA_nS": None if A["PA_nS"] is None or B["PA_nS"] is None else A["PA_nS"] - B["PA_nS"], "REPAIR_SIG": rep,
            "PAR_share": ps[0], "PAR_n": ps[1], "d_ahead": A["ahead"] - B["ahead"], "PAR_SIG": par, "d_acc": A["acc"] - B["acc"]}


def score(files):
    out = {"def": __doc__, "cells": {}, "pairs": {}}
    for s, rn in BASE.items():
        out["cells"]["base_" + s] = agg(files[rn])
    for (arm, s), rn in ARMS.items():
        if rn not in files:
            continue
        out["cells"]["%s_%s" % (arm, s)] = A = agg(files[rn])
        bs = [s] if s in BASE else ["s1", "s2"]
        for b in bs:
            out["pairs"]["%s_%s_vs_base_%s" % (arm, s, b)] = dict(judge(A, out["cells"]["base_" + b], par_share(files[BASE[b]], files[rn])) if s == b else
                                                               judge(A, out["cells"]["base_" + b], (None, 0)), same_seed=(s == b))
    l1, l2 = out["pairs"].get("rd_s1_vs_base_s1", {}).get("label"), out["pairs"].get("rd_s2_vs_base_s2", {}).get("label")
    out["reading"] = l1 if (l1 is not None and l1 == l2) else "MIXED"
    out["lever_gate"] = "G-FAR WITHDRAWN (SETTLE)" if out["reading"] == "SETTLE" else "G-FAR proceeds to review"
    return out


def fixtures():
    rng = np.random.default_rng(0)

    def mk(acc, unfinished, act9, idle8, ahead):
        d = {}
        for n in (9, 10, 11):
            rs = []
            for i in range(128):
                y = (7 * i + n) % 11; right = rng.random() < acc; fin = right or rng.random() >= unfinished
                a_ = [y if right else (y + 1) % 11] * 25; ta = (n - 1 if rng.random() < ahead else n) if right else None
                rs.append({"y": y, "ans": a_, "t_ans": ta, "finished": fin, "S": fin and right, "act": [None] + [0.13] * 8 + [act9] * (n - 6),
                           "idle": [0.0] * 8 + [idle8] * 17})
            d[str(n)] = rs
        return {"deep": d}
    res = []
    rng = np.random.default_rng(1); b = mk(0.6, 0.2, 0.12, 0.1, 0.3); rng = np.random.default_rng(1); a_ = mk(0.45, 0.6, 0.05, 0.4, 0.3)
    o = score({"s213_d3t_s1": b, "s216_d3t_s2": b, "s224_rd_s1": a_, "s224_rd_s2": a_}); res.append(("settle", o["reading"] == "SETTLE", o["reading"]))
    rng = np.random.default_rng(2); b = mk(0.6, 0.2, 0.12, 0.1, 0.3); rng = np.random.default_rng(2); a_ = mk(0.45, 0.2, 0.12, 0.1, 0.3)
    o = score({"s213_d3t_s1": b, "s216_d3t_s2": b, "s224_rd_s1": a_, "s224_rd_s2": a_}); res.append(("value", o["reading"] == "VALUE", o["reading"]))
    rng = np.random.default_rng(3); b = mk(0.6, 0.2, 0.12, 0.1, 0.3); c = mk(0.45, 0.6, 0.05, 0.4, 0.3); d = mk(0.45, 0.2, 0.12, 0.1, 0.3)
    o = score({"s213_d3t_s1": b, "s216_d3t_s2": b, "s224_rd_s1": c, "s224_rd_s2": d}); res.append(("split -> MIXED", o["reading"] == "MIXED", o["reading"]))
    o = score({"s213_d3t_s1": b, "s216_d3t_s2": b, "s224_rd_s1": b, "s224_rd_s2": b, "s224_rd_s3": d})
    res.append(("s3 never same-seed", all(not v["same_seed"] for k, v in o["pairs"].items() if k.startswith("rd_s3")), [k for k in o["pairs"] if k.startswith("rd_s3")]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    return all(r[1] for r in res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        raise SystemExit(0 if fixtures() else 1)
    names = list(BASE.values()) + list(ARMS.values()); files = {}
    for rn in names:
        p = os.path.join(HERE, "hd_check_cpu.%s.json" % rn)
        if os.path.exists(p):
            files[rn] = json.load(open(p))
    miss = [rn for rn in list(BASE.values()) + ["s224_rd_s1", "s224_rd_s2"] if rn not in files]
    if miss:
        raise SystemExit("missing check-B files: %s" % miss)
    o = score(files); json.dump(o, open(os.path.join(HERE, "hd_stage0.out.json"), "w", newline="\n"), indent=1)
    print("reading", o["reading"], "|", o["lever_gate"])
    for k, v in o["pairs"].items():
        print(k, {q: (round(x, 3) if isinstance(x, float) else x) for q, x in v.items()})
