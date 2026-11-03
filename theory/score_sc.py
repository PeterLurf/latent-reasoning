"""THEORY §264 SCORECARD-CHECK scorer (SC_v1 in THEORY §264; Theory is the only scorer; committed before output).
Source: bun_s1 / bun_s2 eval_cue_clock.json. Layout per schedule sch in STD / ER / LF / CY and depth d (str):
  items[sch][d] = {"ans": [N], "pred": [N][KM + 1] (decoder answer after K' = 0..KM passes), "rel_inc": [N][KM - 1] (t = 2..KM: ||S_t - S_t-1|| / ||S_t-1||,
                   Frobenius over every state tensor), "norm": [N][KM] (t = 1..KM: ||S_t||)}; KM = 24.
  STD = the run's cue S_min(t, d) (d 3-8); ER = EARLY-REPEAT S_min(t, d - 2) (d 3-6); LF = LATE-FRESH S_t for t <= d, then a name absent from the item, a new one per pass
  (d 3-6); CY = CYCLE S_((t - 1) mod d) + 1 (d 3-6).
Per seed, d 3-6 (binding for SCOPE wording, Audit 11-03):
  c = median_t in 2..d of med_items rel_inc_t (STD); h = median_t in d+2..d+6 (STD); CONTRAST iff c >= 3 h; tau = sqrt(c h).
  ER (r = d - 2): HOLD_ER = median_t in r+2..r+4 med rel_inc (ER) < tau; EARLY = P(r = 3 halt khalt <= r + 1) (score_c5c's rule on pred[0..KM]).
      PLAN if HOLD_ER and EARLY >= 0.8; COMP if not HOLD_ER or EARLY <= 0.2; else MIXED.
  LF: HOLD_LF = median_t in d+2..d+6 med rel_inc (LF) < tau; acc24 = P(pred[KM] == ans).
      COMPLETION if HOLD_LF and acc24(LF) >= acc24(STD) - 0.02; CUE if not HOLD_LF or acc24(LF) <= acc24(STD) - 0.10; else MIXED.
  CY: acc(k d) for k = 1..KM // d; ratio = median_items norm[KM] / norm[d].
      FIXED if acc(k d) >= acc(d) - 0.02 for every k >= 2 and ratio <= 1.5; FREEZE if some acc(k d) < acc(d) - 0.05 or ratio > 1.5; else MIXED.
  No CONTRAST at d -> ER / LF label NO-CONTRAST at that d. Per seed a schedule's label = the label held at >= 3 of d 3-6, else MIXED.
  VALIDITY: STD acc at K' 8 within 0.015 of the run's eval_acc by_d[d].acc at d 3-6, else VOID (harness mismatch).
Verdict (both seeds must agree):
  C5: ER PLAN in both -> 'C5 PLAN-CLOCKED' (SOTA C5 -> partial: the stop follows the given plan's end); ER COMP in both -> 'C5 unchanged'; else 'C5 MIXED'.
  C6: (LF COMPLETION or CY FIXED) in both -> 'C6 unchanged'; (LF CUE and CY FREEZE) in both -> 'C6 SCOPED (holds under a repeated cue)'; else 'C6 MIXED'.
Descriptive: ER accuracy at the r = 3 halt point (Note 264-1, Audit: does an early halt return a pre-target answer); STD d 7-8 acc at K' 8 / 12 / 16 / 24; CY flips K' d -> 2 d (wrong->right, right->wrong); ER / LF / CY acc by K'; per-schedule median rel_inc by t.
    python -B theory/score_sc.py [--fixtures]
"""
import argparse, json, math, os, sys, tempfile
from statistics import median

_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_sc.json")
RUNS = ("bun_s1", "bun_s2")
KM = 24
DS = (3, 4, 5, 6)


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def halt_k(pred, r=3):
    for k in range(0, KM - r + 1):
        if all(pred[k + j] == pred[k] for j in range(1, r + 1)):
            return k
    return KM


def med_t(it, t):
    return median(x[t - 2] for x in it["rel_inc"])


def acc(it, k):
    return sum(p[k] == y for p, y in zip(it["pred"], it["ans"])) / len(it["ans"])


def at_d(I, d):
    s = str(d); st, er, lf, cy = I["STD"][s], I["ER"][s], I["LF"][s], I["CY"][s]
    c = median(med_t(st, t) for t in range(2, d + 1)); h = median(med_t(st, t) for t in range(d + 2, d + 7))
    contrast = c >= 3 * h; tau = math.sqrt(c * h) if c > 0 and h > 0 else 0.0
    r = d - 2; hold_er = median(med_t(er, t) for t in range(r + 2, r + 5)) < tau
    early = sum(halt_k(p) <= r + 1 for p in er["pred"]) / len(er["pred"])
    er_lab = "PLAN" if hold_er and early >= 0.8 else "COMP" if (not hold_er or early <= 0.2) else "MIXED"
    hold_lf = median(med_t(lf, t) for t in range(d + 2, d + 7)) < tau; a_lf, a_st = acc(lf, KM), acc(st, KM)
    lf_lab = "COMPLETION" if hold_lf and a_lf >= a_st - 0.02 else "CUE" if (not hold_lf or a_lf <= a_st - 0.10) else "MIXED"
    if not contrast:
        er_lab = lf_lab = "NO-CONTRAST"
    ks = list(range(d, KM + 1, d)); ac = {k: acc(cy, k) for k in ks}; ratio = median(n[KM - 1] / n[d - 1] for n in cy["norm"])
    cy_lab = ("FIXED" if all(ac[k] >= ac[d] - 0.02 for k in ks[1:]) and ratio <= 1.5 else
              "FREEZE" if any(ac[k] < ac[d] - 0.05 for k in ks[1:]) or ratio > 1.5 else "MIXED")
    w2r = sum(p[d] != y and p[2 * d] == y for p, y in zip(cy["pred"], cy["ans"])) / len(cy["ans"])
    r2w = sum(p[d] == y and p[2 * d] != y for p, y in zip(cy["pred"], cy["ans"])) / len(cy["ans"])
    return {"c": c, "h": h, "tau": tau, "contrast": contrast, "ER": er_lab, "hold_er": hold_er, "early": early, "LF": lf_lab, "hold_lf": hold_lf,
            "acc24_lf": a_lf, "acc24_std": a_st, "CY": cy_lab, "cy_acc": ac, "cy_norm_ratio": ratio, "cy_flips_d_2d": {"w2r": w2r, "r2w": r2w}}


def agg(per, key):
    labs = [per[str(d)][key] for d in DS]
    for lab in set(labs):
        if labs.count(lab) >= 3:
            return lab
    return "MIXED"


def seed(runs, run):
    f = jl(os.path.join(runs, run, "eval_cue_clock.json")); ea = jl(os.path.join(runs, run, "eval_acc.json"))
    if f is None or ea is None:
        return {"label": "pending"}
    I = f["items"]
    valid = all(abs(acc(I["STD"][str(d)], 8) - ea["by_d"][str(d)]["acc"]) <= 0.015 for d in DS)
    per = {str(d): at_d(I, d) for d in DS}
    desc = {"std_long": {str(d): {str(k): acc(I["STD"][str(d)], k) for k in (8, 12, 16, 24)} for d in (7, 8) if str(d) in I["STD"]},
            "acc_by_k": {s: {str(d): [acc(I[s][str(d)], k) for k in range(KM + 1)] for d in DS} for s in ("ER", "LF", "CY")},
            "rel_inc_by_t": {s: {str(d): [med_t(I[s][str(d)], t) for t in range(2, KM + 1)] for d in DS} for s in ("STD", "ER", "LF", "CY")},
            "er_acc_at_halt": {str(d): sum(p[halt_k(p)] == y for p, y in zip(I["ER"][str(d)]["pred"], I["ER"][str(d)]["ans"])) / len(I["ER"][str(d)]["ans"]) for d in DS}}   # Note 264-1 (Audit)
    return {"label": "VOID" if not valid else "ok", "valid": valid, "per_d": per, "ER": agg(per, "ER"), "LF": agg(per, "LF"), "CY": agg(per, "CY"), "desc": desc}


def score(runs=R):
    o = {"section": 264, "seeds": {r: seed(runs, r) for r in RUNS}}; S = list(o["seeds"].values())
    if any(s["label"] == "pending" for s in S):
        o["verdict"] = "pending"; return o
    if any(s["label"] == "VOID" for s in S):
        o["verdict"] = "VOID (STD does not reproduce eval_acc)"; return o
    er = [s["ER"] for s in S]; c5 = "C5 PLAN-CLOCKED" if er == ["PLAN"] * 2 else "C5 unchanged" if er == ["COMP"] * 2 else "C5 MIXED"
    keep = [s["LF"] == "COMPLETION" or s["CY"] == "FIXED" for s in S]; scoped = [s["LF"] == "CUE" and s["CY"] == "FREEZE" for s in S]
    c6 = "C6 unchanged" if all(keep) else "C6 SCOPED (holds under a repeated cue)" if all(scoped) else "C6 MIXED"
    o["verdict"] = "%s; %s (ER %s / LF %s / CY %s)" % (c5, c6, "/".join(er), "/".join(s["LF"] for s in S), "/".join(s["CY"] for s in S)); return o


def fixtures():
    res = []

    def mk_items(d, n, comp_inc, hold_inc, plan_at, ans_ok_until, drift=False):
        # rel_inc: compute-level up to pass plan_at, hold-level after; pred: correct from K' >= plan_at (or wrong if > ans_ok_until)
        rel = [[comp_inc if t <= plan_at else hold_inc for t in range(2, KM + 1)] for _ in range(n)]
        pred = [[(7 if (plan_at <= k <= ans_ok_until) else (k % 11 if (drift and k > ans_ok_until) else 3)) for k in range(KM + 1)] for _ in range(n)]
        norm = [[1.0 + (0.1 * t if drift else 0.0) for t in range(1, KM + 1)] for _ in range(n)]
        return {"ans": [7] * n, "pred": pred, "rel_inc": rel, "norm": norm}

    def world(er="plan", lf="completion", cy="fixed", std_acc_shift=0.0, flat=False):
        dd = tempfile.mkdtemp()
        for run in RUNS:
            p = os.path.join(dd, run); os.makedirs(p); I = {"STD": {}, "ER": {}, "LF": {}, "CY": {}}
            for d in (3, 4, 5, 6, 7, 8):
                hold = 0.3 if flat else 0.01
                I["STD"][str(d)] = mk_items(d, 50, 0.3, hold, d, KM)
                if d not in DS:
                    continue
                r = d - 2
                I["ER"][str(d)] = mk_items(d, 50, 0.3, hold, r, KM) if er == "plan" else mk_items(d, 50, 0.3, 0.3, KM, KM)
                I["LF"][str(d)] = mk_items(d, 50, 0.3, hold, d, KM) if lf == "completion" else mk_items(d, 50, 0.3, 0.3, d, d + 1, drift=True)
                I["CY"][str(d)] = mk_items(d, 50, 0.3, hold, d, KM) if cy == "fixed" else mk_items(d, 50, 0.3, 0.3, d, d + 1, drift=True)
            json.dump({"items": I}, open(os.path.join(p, "eval_cue_clock.json"), "w"))
            json.dump({"by_d": {str(d): {"acc": 1.0 - std_acc_shift} for d in DS}}, open(os.path.join(p, "eval_acc.json"), "w"))
        return dd
    for name, kw, want in (("plan-clocked, C6 kept", {}, "C5 PLAN-CLOCKED; C6 unchanged"),
                           ("computation-driven", {"er": "comp"}, "C5 unchanged; C6 unchanged"),
                           ("C6 scoped", {"lf": "cue", "cy": "freeze"}, "C6 SCOPED"),
                           ("C6 kept by CY alone", {"lf": "cue"}, "C6 unchanged"),
                           ("C6 kept by LF alone", {"cy": "freeze"}, "C6 unchanged"),
                           ("void", {"std_acc_shift": 0.05}, "VOID"),
                           ("no contrast", {"flat": True}, "C5 MIXED")):
        o = score(world(**kw)); res.append((name, want in o["verdict"], o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], "->", r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
