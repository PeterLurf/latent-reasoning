"""Measure and score docs/PROTOCOL5.md (addenda A-C).

    python -m latent.protocol5_confirm strict <runs...> --device cuda:0   # addendum C measurements -> results/strict_clock.json
    python -m latent.protocol5_confirm endact <runs...> --device cuda:0   # P26 end activity          -> results/p5_endact.json
    python -m latent.protocol5_confirm score                              # verdicts                  -> results/protocol5_confirm.json

Definitions are those frozen in the addenda; nothing here is tuned after seeing results.
"""
from __future__ import annotations

import argparse
import json
import os
from math import comb

import numpy as np

from .competence import load
from .confirm_sweep import MODELS as E_MODELS, DEPTHS, blocking, clock, trace
from .protocol4_confirm import attempts, final_stage
from .protocol3_confirm import jload

STRICT, ENDACT, OUT = "results/strict_clock.json", "results/p5_endact.json", "results/protocol5_confirm.json"
PILOT = [f"rel_{arm}_s{s}" for s in range(20, 28) for arm in ("base", "lr5")]
CONC = 0.35          # addendum C concentration rule
END_T = 0.25         # addendum B threshold


def excess(tr):
    return np.array(tr["reachable"]) - np.array(tr["decoy"])


def single_step(tr):
    """Addendum C: k*(1) < k*(2) < k*(3) and off-by->=2 steps carry <= CONC x the excess at k*(x)."""
    R, ex = np.array(tr["reachable"]), excess(tr)
    ks = [int(np.argmax(R[:, x])) for x in DEPTHS]
    ok = ks[0] < ks[1] < ks[2] and all(
        ex[k, x] > 0 and all(ex[j, x] <= CONC * ex[k, x] for j in range(len(ex)) if abs(j - k) >= 2)
        for x, k in zip(DEPTHS, ks))
    return [k + 1 for k in ks], bool(ok)


def end_activity(tr, K):
    ex = excess(tr)[:K, 1:]
    pk = ex.max()
    return float(ex[K - 1].max() / pk) if pk > 0 else 0.0


def _save(path, key, val):
    res = json.load(open(path)) if os.path.exists(path) else {}
    res[key] = val
    json.dump(res, open(path, "w"), indent=1)


def measure_strict(runs, device):
    done = json.load(open(STRICT)) if os.path.exists(STRICT) else {}
    for r in runs:
        if r in done:
            continue
        m, c = load(r, device)
        K = c["K"]; Kp = max(8, K)
        disc = trace(m, K, Kp, 4, device, seed=5050, split="discovery")
        kd, qual = single_step(disc)
        o = {"K": K, "Kp": Kp, "disc_trace4": disc, "disc_clock": kd, "single_step": qual}
        o["trace"] = {W: trace(m, K, Kp, W, device, seed=3030, split="confirm") for W in (2, 4, 8)}
        o["clock"], o["sweeps"] = clock(o["trace"][4])
        o["clock_by_W"] = {W: clock(o["trace"][W])[0] for W in (2, 4, 8)}
        o["block"] = blocking(m, K, Kp, device, seed=3037, split="confirm")
        _save(STRICT, r, o)
        print(f"{r}: disc clock {kd} single-step {qual}; confirm clock {o['clock']}", flush=True)


def measure_endact(runs, device):
    done = json.load(open(ENDACT)) if os.path.exists(ENDACT) else {}
    for r in runs:
        if r in done:
            continue
        m, c = load(r, device)
        K = c["K"]; Kp = max(8, K)
        tr = trace(m, K, Kp, 4, device)                 # seed 2026 + 4, confirm split: addendum B
        _save(ENDACT, r, {"K": K, "trace4": tr, "end_activity": end_activity(tr, K)})
        print(f"{r}: end activity {end_activity(tr, K):.3f}", flush=True)


def p23_p24(o):
    """PROTOCOL4 addendum-E rules, unchanged (as in confirm_sweep.score)."""
    ks, B = o["clock"], o["block"]
    dn = {int(k): v for k, v in B["downstream"].items()}
    ct = {int(k): v for k, v in B["control"].items()}
    diag = all(dn[k][x] <= -0.05 for x, k in zip(DEPTHS, ks))
    off = all(abs(dn[k][x]) <= 0.02 for x, kx in zip(DEPTHS, ks) for k in dn if abs(k - kx) >= 2)
    ctl = all(v >= -0.01 for k in ct for v in ct[k])
    T = o["trace"]; k1 = ks[0] - 1
    ex = lambda W: T[str(W)]["reachable"][k1][1] - T[str(W)]["decoy"][k1][1]
    same = all(o["clock_by_W"][str(W)] == ks for W in (2, 4, 8))
    r_edge = ex(8) / ex(2)
    r_tot = ex(8) * T["8"]["n_depth1"] / (ex(2) * T["2"]["n_depth1"])
    return ({"diag": diag, "off": off, "control": ctl, "pass": bool(diag and off and ctl)},
            {"same_clock": same, "edge_W8_W2": r_edge, "total_W8_W2": r_tot,
             "pass": bool(same and 0.10 <= r_edge <= 0.45 and 0.6 <= r_tot <= 1.6)})


def reading(ps):
    if not ps:
        return {"n": 0, "n_pass": 0, "frac": None, "reading": None}
    k = sum(ps); f = k / len(ps)
    return {"n": len(ps), "n_pass": k, "frac": f, "reading": "confirmed" if f >= 0.8 else "fails" if f < 0.5 else "mixed"}


def fisher_one_sided(a, b, c, d):
    """P(X >= a) for the 2x2 table [[a, b], [c, d]] with fixed margins (a = arm-1 successes)."""
    n1, n2, k = a + b, c + d, a + c
    tot = comb(n1 + n2, k)
    return sum(comb(n1, x) * comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / tot


def pilot_run(r):
    run = f"runs/{r}"
    o = {"trained": os.path.exists(f"{run}/model.pt"), "final_stage": final_stage(r), "attempts": attempts(r)}
    if os.path.exists(f"{run}/log.jsonl"):
        o["advances"] = [(x["step"], x["curriculum_advance_to"]) for x in map(json.loads, open(f"{run}/log.jsonl"))
                         if "curriculum_advance_to" in x]
    c, m = jload(f"results/competence_{r}.json"), jload(f"results/competence_matched_{r}.json")
    L = (jload(f"results/length_gen_{r}.json") or {}).get(run)
    if c is not None and m is not None:
        c = c[0]
        top = c["budgets"][str(max(int(k) for k in c["budgets"]))]
        o["delayed"] = top["delayed"]["bal_acc"][0]
        o["matched5"], o["matched6"] = m[run]["5"]["bal_acc"], m[run]["6"]["bal_acc"]
        o["gate"] = bool(o["delayed"] >= 0.85 and o["matched5"] >= 0.60 and o["final_stage"] == "full")
    if L:
        row = next((x for x in L["rows"] if x["K'"] == 12), None)
        if row:
            ci = row.get("far_ci", {}).get("7")
            o["d7_K12"], o["d7_K12_ci"] = row["far"]["7"], ci
            o["extrap_K12"] = bool(row["far"]["7"] >= 0.60 and ci is not None and ci[0] > 0.55)
    return o


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool)
    pos, neg = x[y], x[~y]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def score():
    out = {"pilot": {r: pilot_run(r) for r in PILOT}}
    P = out["pilot"]
    # R1 (addendum A)
    done = {r: p for r, p in P.items() if "gate" in p}
    arm = lambda a: [r for r in done if f"_{a}_" in r]
    gb, gl = sum(done[r]["gate"] for r in arm("base")), sum(done[r]["gate"] for r in arm("lr5"))
    nb, nl = len(arm("base")), len(arm("lr5"))
    R1 = {"base": f"{gb}/{nb}", "lr5e-4": f"{gl}/{nl}", "complete": nb == 8 and nl == 8,
          "fisher_p_one_sided": fisher_one_sided(gl, nl - gl, gb, nb - gb) if nb and nl else None,
          "full_depth": {a: sum(done[r]["final_stage"] == "full" for r in arm(a)) for a in ("base", "lr5")},
          "stage1_relaunches": {a: sum((P[r]["attempts"] or 1) - 1 for r in P if f"_{a}_" in r) for a in ("base", "lr5")}}
    R1["reading"] = ("holds" if gl > gb and gl >= 6 else "fails" if gl <= gb else "partial") if R1["complete"] else "incomplete"
    for a in ("base", "lr5"):
        full = [done[r] for r in arm(a) if done[r]["final_stage"] == "full"]
        R1[f"accuracy_{a}"] = {k: (float(np.mean([f[k] for f in full])) if full else None) for k in ("delayed", "matched5", "matched6")}
    out["R1"] = R1
    # P26 (addendum B)
    EA = jload(ENDACT) or {}
    rows = [(r, EA[f"runs/{r}"]["end_activity"], done[r]["extrap_K12"]) for r in done
            if done[r]["gate"] and f"runs/{r}" in EA and "extrap_K12" in done[r]]
    lo = [e for _, v, e in rows if v < END_T]; hi = [e for _, v, e in rows if v >= END_T]
    a_ok = sum(lo) <= 1
    rate = lambda xs: (sum(xs) / len(xs)) if xs else None
    b_dir = None if not lo or not hi else (rate(hi) > rate(lo)) if rate(hi) != rate(lo) else None
    c_auc = (auc([v for _, v, _ in rows], [e for _, _, e in rows])
             if len(lo) >= 2 and len(hi) >= 2 and 0 < sum(e for *_, e in rows) < len(rows) else None)
    if not a_ok or b_dir is False:
        rd = "fails"
    elif b_dir and (c_auc is None or c_auc >= 0.70):
        rd = "confirmed"
    else:
        rd = "unresolved"
    out["P26"] = {"runs": {r: {"end_activity": v, "extrapolates": e} for r, v, e in rows},
                  "low": {"n": len(lo), "n_extrap": sum(lo)}, "high": {"n": len(hi), "n_extrap": sum(hi)},
                  "a_necessity": a_ok, "b_direction": b_dir, "c_auc": c_auc, "reading": rd}
    # P23' / P24' (addendum C)
    S = jload(STRICT) or {}
    per = {}
    for r, o in S.items():
        a, b = p23_p24(o)
        per[r] = {"single_step": o["single_step"], "disc_clock": o["disc_clock"], "clock": o["clock"], "P23": a, "P24": b}
    main = [r for r in per if r in E_MODELS]
    pil = [r for r in per if r not in E_MODELS and P.get(r[len("runs/"):], {}).get("gate")]   # addendum C: gate-passing pilot runs
    ss = lambda rs: [r for r in rs if per[r]["single_step"]]
    ns = lambda rs: [r for r in rs if not per[r]["single_step"]]
    out["strict_clock"] = {
        "per_model": per,
        "P23_single_step": reading([per[r]["P23"]["pass"] for r in ss(main)]),
        "P24_single_step": reading([per[r]["P24"]["pass"] for r in ss(main)]),
        "P23_other": reading([per[r]["P23"]["pass"] for r in ns(main)]),
        "P24_other": reading([per[r]["P24"]["pass"] for r in ns(main)]),
        "single_step_models": ss(main),
        "pilot_P23_single_step": reading([per[r]["P23"]["pass"] for r in ss(pil)]),
        "pilot_P24_single_step": reading([per[r]["P24"]["pass"] for r in ss(pil)]),
        "pilot_single_step_models": ss(pil)}
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("R1", "P26")}, indent=1, default=float))
    sc = out["strict_clock"]
    print(json.dumps({k: v for k, v in sc.items() if k != "per_model"}, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["strict", "endact", "score"])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.cmd == "strict":
        measure_strict(a.runs or E_MODELS, a.device)
    elif a.cmd == "endact":
        measure_endact(a.runs, a.device)
    else:
        score()
