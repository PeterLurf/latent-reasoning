"""THEORY §267 EXACT-EM-11 stage A scorer (EM_v1 in THEORY §267; Theory is the only scorer; committed before output).
Cells (OPS_CHAIN word programs, d_max 3, askname init, 12k, plan given by the step cue, NO per-step value in the loss):
  em_s1 / em_s2   K1a RESET-DM: 11-way categorical at each pass's ':'; S_t = write(emb(v_t)) on the zero state; loss = -log sum over v_1..v_d-1 (exact forward DP)
  emh_s1          --em_hard 1 (max instead of logsumexp; descriptive CTRL)
  emt_s1          TOKEN-INJECT-11 (previous value as text, exact enumeration, no DeltaMem; descriptive CTRL / upper bound)
  aoc_s1          the AO recipe (answer CE through the continuous DeltaMem loop, no aux) on the same OPS_CHAIN d_max 3 stream (descriptive CTRL: the drift)
Fields (per cell): eval_em.json = {"acc": {d: {acc, n}} (d 1-5, the pass-d greedy readout; TEST2-chain), "ws": {d: {t: acc}} (pass-t greedy ':' readout == v_t),
  "swap_v": {"3": {follow, floor, n}} (pass 1's write replaced by write(v') with v' != v_1, seeded; follow = final == the program recomputed with v_1 := v';
  floor = P(intact final == that answer)), "mem_ablate": {"3": {acc, n}} (pass 3 run on the zero state)}; log.jsonl rows {"step", "ws_readout": {d: {t: acc}}}.
Binding per em seed (EM-C4, stage A):
  ONSET   ws_readout['3']['1'] >= 0.80 at the log rows step 1000 and 2000 (the AO drift showed 0.09-0.27 there);
  FORM    final ws d 3 passes 1, 2 >= 0.90; acc d 2 >= 0.90 and d 3 >= 0.80;
  USE     swap_v d 3 follow >= 0.80 and >= floor + 0.30; mem_ablate d 3 acc <= 0.30 (pass 3 cannot recompute the chain from text alone);
  Label EM-C4 if all hold; PARK if ws_readout['3']['1'] <= 0.30 at step 2000 in both seeds; else FAIL: <criteria>. Wilson 95% CIs + NEAR on the binding cells.
Descriptive: CTRL cells' readings; d 4-5 acc (extrapolation with given cues); posterior entropy (log, if present);
  Note 267-1 (Audit): PERM-INVARIANT readout = best-bijection accuracy of the pass-t argmax vs v_t (max over permutations of Z_11, exact bitmask DP) on
  eval_em.json "ws_items" {d: {t: {"pred": [N], "v": [N]}}}; PERMUTED-LATENT if bijection acc >= 0.90 while raw ws < 0.50 (a latent formed under relabelling).
    python -B theory/score_em.py [--fixtures]
"""
import argparse, json, math, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_em.json")
EM = ("em_s1", "em_s2")
CTRL = ("emh_s1", "emt_s1", "aoc_s1")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def rows(p):
    out = []
    try:
        for ln in open(p):
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n); return (c - h, c + h)


def near(p, n, bar):
    return abs(p - bar) <= 2 * math.sqrt(max(p * (1 - p), 1e-12) / n)


def onset(runs, run):
    o = {}
    for r in rows(os.path.join(runs, run, "log.jsonl")):
        if r.get("step") in (1000, 2000) and "ws_readout" in r:
            o[r["step"]] = r["ws_readout"]["3"]["1"]
    return o


def best_bijection(pred, v, k=11):
    """max over bijections pi of Z_k of mean(pi(pred) == v), exact by DP over subsets of targets."""
    C = [[0] * k for _ in range(k)]
    for a, b in zip(pred, v):
        C[a][b] += 1
    best = {0: 0}
    for i in range(k):
        nxt = {}
        for m, val in best.items():
            for j in range(k):
                if not m >> j & 1:
                    m2 = m | 1 << j; w = val + C[i][j]
                    if nxt.get(m2, -1) < w:
                        nxt[m2] = w
        best = nxt
    return max(best.values()) / max(len(v), 1)


def cell(runs, run):
    e = jl(os.path.join(runs, run, "eval_em.json"))
    if e is None:
        return None
    on = onset(runs, run)
    perm = {d: {t: best_bijection(x["pred"], x["v"]) for t, x in tt.items()} for d, tt in e.get("ws_items", {}).items()}
    return {"acc": {d: e["acc"][d]["acc"] for d in e["acc"]}, "n3": e["acc"]["3"]["n"], "ws3": e["ws"]["3"], "swap": e["swap_v"]["3"], "ablate": e["mem_ablate"]["3"]["acc"],
            "onset": on, "perm_ws": perm,
            "permuted_latent": any(perm.get(d, {}).get(t, 0) >= 0.90 and e["ws"].get(d, {}).get(t, 1) < 0.50 for d in perm for t in perm[d])}


def seed(c):
    on = c["onset"]
    if 1000 not in on or 2000 not in on:
        return {"label": "pending (onset rows missing)"}
    crit = {"ONSET": on[1000] >= 0.80 and on[2000] >= 0.80,
            "FORM": c["ws3"]["1"] >= 0.90 and c["ws3"]["2"] >= 0.90 and c["acc"]["2"] >= 0.90 and c["acc"]["3"] >= 0.80,
            "USE": c["swap"]["follow"] >= 0.80 and c["swap"]["follow"] >= c["swap"]["floor"] + 0.30 and c["ablate"] <= 0.30}
    fail = [k for k, v in crit.items() if not v]; n = c["n3"]
    return {"label": "EM-C4" if not fail else "FAIL: " + ",".join(fail), "crit": crit, "park_sig": on[2000] <= 0.30,
            "ci": {"acc3": wilson(c["acc"]["3"], n), "swap": wilson(c["swap"]["follow"], c["swap"]["n"])},
            "near": {"acc3": near(c["acc"]["3"], n, 0.80), "swap": near(c["swap"]["follow"], c["swap"]["n"], 0.80)}}


def score(runs=R):
    o = {"section": 267, "cells": {r: cell(runs, r) for r in EM + CTRL}}
    if any(o["cells"][r] is None for r in EM):
        o["verdict"] = "pending"; return o
    o["seeds"] = {r: seed(o["cells"][r]) for r in EM}; labs = [o["seeds"][r]["label"] for r in EM]
    if any(l.startswith("pending") for l in labs):
        o["verdict"] = "pending"; return o
    if all(o["seeds"][r]["park_sig"] for r in EM):
        o["verdict"] = "PARK (onset drift in both seeds: ws d3 pass 1 <= 0.30 at 2k) (%s)" % " / ".join(labs); return o
    o["verdict"] = "EM-C4 %d/2 (%s)" % (labs.count("EM-C4"), " / ".join(labs)); return o


def fixtures():
    res = []

    def world(on1=0.95, on2=0.97, ws1=0.98, ws2=0.97, a2=0.97, a3=0.9, fol=0.92, flo=0.1, abl=0.12, missing_log=False):
        dd = tempfile.mkdtemp()
        for r in EM + CTRL:
            p = os.path.join(dd, r); os.makedirs(p)
            json.dump({"acc": {str(d): {"acc": {1: 1.0, 2: a2, 3: a3, 4: 0.6, 5: 0.4}[d], "n": 1000} for d in range(1, 6)},
                       "ws": {"3": {"1": ws1, "2": ws2, "3": a3}}, "swap_v": {"3": {"follow": fol, "floor": flo, "n": 1000}}, "mem_ablate": {"3": {"acc": abl, "n": 1000}}},
                      open(os.path.join(p, "eval_em.json"), "w"))
            with open(os.path.join(p, "log.jsonl"), "w") as fh:
                if not missing_log:
                    for st, v in ((1000, on1), (2000, on2)):
                        fh.write(json.dumps({"step": st, "ws_readout": {"3": {"1": v, "2": 0.5}}}) + chr(10))
        return dd
    for name, kw, want in (("all", {}, "EM-C4 2/2"), ("onset drift", {"on1": 0.5}, "FAIL: ONSET"), ("park", {"on1": 0.2, "on2": 0.2}, "PARK"),
                           ("form ws", {"ws2": 0.8}, "FAIL: FORM"), ("form d3", {"a3": 0.7}, "FAIL: FORM"), ("swap", {"fol": 0.5}, "FAIL: USE"),
                           ("swap vs floor", {"fol": 0.85, "flo": 0.6}, "FAIL: USE"), ("in-pass recompute", {"abl": 0.6}, "FAIL: USE"),
                           ("missing log", {"missing_log": True}, "pending")):
        o = score(world(**kw)); res.append((name, want in o["verdict"], o["verdict"]))
    import random
    rg = random.Random(0); v = [rg.randrange(11) for _ in range(500)]; pi = list(range(11)); rg.shuffle(pi)
    res.append(("bijection recovers a permuted code", abs(best_bijection([pi[x] for x in v], v) - 1.0) < 1e-9, "perm"))
    res.append(("bijection of a constant readout is ~1/11", best_bijection([0] * 500, v) < 0.2, "const"))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], "->", r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
