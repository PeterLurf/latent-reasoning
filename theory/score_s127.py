"""THEORY §127 scoring (rules fixed in §127; written before any §127 run; Theory is the only scorer).
Runs: boxes/runs_fsq/s127_s5_{var,skip,std2,lr10}_s{1..4}, s127_ff_{arm}_s{1,2}; control s127_s5_ctrl_s{3,4} plus §120 FINAL
S5 s1/s2 (s120_s5_final_s{1,2}). eval.json (long_acc, carried_norm, probe, train_gpu), log.jsonl, eval_contraction_fp32.json.
Plateau flag (123-3, at 2k; needs the log to reach step 2000): loss(2k) >= 0.9 loss(1k) and loss(2k) >= 0.25 loss(100).
A run stopped at 2k after its flag fired has no eval.json and is scored plateaued (long_acc counted as < 0.99).
ESCAPE(arm): seed 2 not plateaued.  ENTRY-FIX(arm): ESCAPE, S5 plateau count <= 1/4 (s1-4), long_acc >= 0.99 in >= 3/4, and
rho_ff (123-9, fp32) <= 0.28 in every seed with long_acc >= 0.99.  FF guard: flip-flop long_acc >= 0.99 in 2/2.
ADOPT: ENTRY-FIX and FF guard (ties: lowest cost per accepted seed, then fewest new terms D, C, B, A).
C (std2) validity: carried row norm at 2k within 0.5x-2x wte's, else "not delivered".
Cost per accepted seed (descriptive): (1 + (p / (1 - p)) / 12) / a; p = S5 plateau rate, a = share of escaped seeds with
long_acc >= 0.99.
    python theory/score_s127.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
ARMS = ("var", "skip", "std2", "lr10")
TIE = {"lr10": 0, "std2": 1, "skip": 2, "var": 3}                           # fewest new terms first


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def plateau(d):
    p = os.path.join(R, d, "log.jsonl")
    if not os.path.exists(p):
        return None
    L = {}
    for line in open(p):
        x = json.loads(line)
        if "loss" in x and "step" in x:
            L[x["step"]] = x["loss"]
    if not L or max(L) < 2000:
        return None                                                          # 2k not reached: undecided
    g = lambda t: L[min(L, key=lambda s: abs(s - t))]  # noqa: E731
    return bool(g(2000) >= 0.9 * g(1000) and g(2000) >= 0.25 * g(100))


def rho_ff(cf, eps="0.1", floor=1e-3):                                       # note 123-9
    try:
        e = cf["detail"]["eps"][eps]; rk, dk = e["rho_k"], e["dist_med_k"]
    except (KeyError, TypeError):
        return None
    use = [rk[k] for k in range(len(rk)) if k < len(dk) and dk[k] > floor and rk[k] > 0]
    if len(use) < 2:
        return None
    return math.exp(sum(math.log(x) for x in use) / len(use))


def cell(d):
    r, pl = load(d), plateau(d)
    if r is None and pl is None:
        return None                                                          # not started / before 2k
    if r is None:
        return {"plateau": pl, "long_acc": None, "status": "stopped at 2k (plateau)" if pl else "running", "rho_ff": None}
    cn = r.get("carried_norm") or {}; k2 = cn.get("2k") or {}
    return {"plateau": pl, "long_acc": r.get("long_acc"), "status": "done", "rho_ff": rho_ff(load(d, "eval_contraction_fp32.json")),
            "ext_2048": (r.get("ext_2048") or {}).get("cumulative"), "codes_per_state": r.get("codes_per_state"),
            "row_norm_2k": k2.get("row_norm"), "wte_row_norm": k2.get("wte_row_norm"), "probe": r.get("probe"),
            "gpu": r.get("train_gpu", r.get("gpu"))}


def cost(cells):
    done = [v for v in cells.values() if v and v["plateau"] is not None]
    if not done:
        return None
    p = sum(bool(v["plateau"]) for v in done) / len(done); esc = [v for v in done if not v["plateau"]]
    if not esc or any(v["long_acc"] is None for v in esc):
        return {"p": p, "a": None, "cost": None, "n": len(done)}
    a = sum((v["long_acc"] or 0) >= 0.99 for v in esc) / len(esc)
    return {"p": p, "a": a, "cost": (1 + (p / (1 - p)) / 12) / a if a > 0 and p < 1 else math.inf, "n": len(done)}


def main():
    res = {"arms": {}, "control": None}
    for arm in ARMS:
        s5 = {s: cell("s127_s5_%s_s%d" % (arm, s)) for s in (1, 2, 3, 4)}; s5 = {k: v for k, v in s5.items() if v}
        ff = {s: cell("s127_ff_%s_s%d" % (arm, s)) for s in (1, 2)}; ff = {k: v for k, v in ff.items() if v}
        esc = (not s5[2]["plateau"]) if 2 in s5 and s5[2]["plateau"] is not None else "pending"
        valid = True
        if arm == "std2" and 2 in s5 and s5[2].get("row_norm_2k") is not None:
            ratio = s5[2]["row_norm_2k"] / s5[2]["wte_row_norm"]; valid = 0.5 <= ratio <= 2.0
        decided = [v for v in s5.values() if v["plateau"] is not None and (v["plateau"] or v["status"] == "done")]
        if esc is False:
            ef = False
        elif len(decided) < 4:
            ef = "incomplete (%d / 4)" % len(decided)
        else:
            n_pl = sum(bool(v["plateau"]) for v in s5.values()); ok = [v for v in s5.values() if (v["long_acc"] or 0) >= 0.99]
            if n_pl > 1 or len(ok) < 3:
                ef = False
            elif any(v["rho_ff"] is None for v in ok):
                ef = "incomplete (rho_ff)"
            else:
                ef = all(v["rho_ff"] <= 0.28 for v in ok)
        ffg = (len(ff) == 2 and all((v["long_acc"] or 0) >= 0.99 for v in ff.values())) if len(ff) == 2 and all(v["status"] == "done" for v in ff.values()) else "incomplete"
        adopt = (ef is True and ffg is True) if isinstance(ef, bool) and (ef is False or isinstance(ffg, bool)) else "pending"
        res["arms"][arm] = {"S5": s5, "FF": ff, "ESCAPE": esc, "delivered": valid if arm == "std2" else True, "ENTRY_FIX": ef if valid else "not delivered",
                            "FF_GUARD": ffg, "ADOPT": adopt if valid else False, "cost": cost(s5)}
    ctrl = {s: cell("s127_s5_ctrl_s%d" % s) for s in (3, 4)}; ctrl.update({s: cell("s120_s5_final_s%d" % s) for s in (1, 2)})
    ctrl = {k: v for k, v in ctrl.items() if v}; res["control"] = {"S5": ctrl, "cost": cost(ctrl)}
    ad = [a for a in ARMS if res["arms"][a]["ADOPT"] is True]
    ad.sort(key=lambda a: ((res["arms"][a]["cost"] or {}).get("cost") or math.inf, TIE[a])); res["DEFAULT"] = ad[0] if ad else None
    for a in ARMS:
        v = res["arms"][a]
        print(a.upper(), {s: (c["plateau"], c["long_acc"], c["rho_ff"], c["status"]) for s, c in v["S5"].items()},
              "| ESCAPE", v["ESCAPE"], "| ENTRY-FIX", v["ENTRY_FIX"], "| FF", v["FF_GUARD"], "| ADOPT", v["ADOPT"], "| cost", v["cost"])
    print("CONTROL", {s: (c["plateau"], c["long_acc"], c.get("gpu")) for s, c in ctrl.items()}, "| cost", res["control"]["cost"])
    print("DEFAULT", res["DEFAULT"], "| no arm escapes -> restart-on-plateau" if all(res["arms"][a]["ESCAPE"] is False for a in ARMS) else "")
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s127.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
