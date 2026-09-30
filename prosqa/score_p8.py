"""Score PROTOCOL8 (confirmatory; docs/PROTOCOL8.md). Written 2026-09-30 before any PROTOCOL8 run started.

Inputs per seed s in {1,2,3,4} (fetched to prosqa/res_p8c/<arm>_s<s>/):
  latent arms coco / at / rt: coco_test_e48.json (long_eval2, K' in {0,1,2,3,4,6,8,10,12,14,16})
  text-CoT arm cot:            cot_test_e30.json (eval_p8 --mode cot, 150 per m, generated-token counts)
Measures (candidate accuracy, as in Phase 0):
  D710(arm, s)  accuracy at K' = 10 over test problems with answer depth 7..10 (deep, inside the trained range)
  D11(arm, s)   accuracy at answer depth 11 (first depth past training), K' = 16
  D12p(arm, s)  accuracy at answer depths >= 12, K' = 16
  SHAL(arm, s)  accuracy at answer depths 3..6, K' = 10;  LU(arm, s) = acc(m 0..6, K'=10) - acc(m 0..6, K'=0)
Rules (seed is the unit; exact one-sided Mann-Whitney on 4 vs 4 seed values):
  P8-1 (primary, per arm at / rt vs coco): D710 higher for the arm: supported if exact p <= 0.05 and mean difference
       >= 0.05; not supported if mean difference <= 0; else inconclusive.
  P8-2 (secondary): the same rule on D11; D12p reported (prediction: no gain).
  P8-3 (latent use, per latent arm): LU >= 0.15 in >= 3 of 4 seeds.
  P8-4 (C3, descriptive with a bar): best latent arm "competitive with text CoT" if its mean D710 >= CoT mean D710 - 0.03;
       reported with compute per answer: latent K' + answer tokens vs CoT generated tokens + answer tokens.
  P8-5 (mechanistic signature, theory prediction): K'sat(arm, s) = smallest K' with acc(depth 7..10, K') >=
       acc(depth 7..10, K'=16) - 0.02. Predicted: coco ~3; anytime >= 6 in >= 3 of 4 seeds. Descriptive decision:
       "signature met" if anytime K'sat >= 6 in >= 3 seeds and coco K'sat <= 4 in >= 3 seeds.
  Learnability: competent = acc(m 0..6, K'=10) >= 0.75 (the Phase 0 F1 bar); counts per arm.
    python prosqa/score_p8.py
"""
import itertools
import json
import os

import numpy as np

R = "prosqa/res_p8c"
SEEDS = (1, 2, 3, 4)


def load(arm, s):
    f = f"{R}/{arm}_s{s}/" + ("cot_test_e30.json" if arm == "cot" else "coco_test_e48.json")
    return json.load(open(f)) if os.path.exists(f) else None


def acc_depth(d, Kp, lo, hi):
    b = d["budgets"][str(Kp)] if "budgets" in d else d
    pl = np.array(b["path_len"]) if "path_len" in b else None
    mg = np.array(b["margins"])
    if pl is None:                                                  # cot file: path length via by_path_len counts
        return None
    sel = (pl >= lo) & (pl <= hi)
    return float((mg[sel] > 0).mean()) if sel.any() else None


def acc_m(d, Kp, ms):
    b = d["budgets"][str(Kp)]
    e, mg = np.array(b["ext"]), np.array(b["margins"])
    sel = np.isin(e, ms)
    return float((mg[sel] > 0).mean())


def cot_depth(d, lo, hi):
    pl = np.array([x for x in d.get("path_len_list", [])]) if "path_len_list" in d else None
    bp, n = d["by_path_len"], d["n_by_path_len"]
    tot = sum(n[k] for k in n if lo <= int(k) <= hi)
    return None if not tot else float(sum(bp[k] * n[k] for k in bp if lo <= int(k) <= hi) / tot)


def ksat(d):
    ks = sorted(int(k) for k in d["budgets"])
    top = acc_depth(d, 16, 7, 10)
    for k in ks:
        a = acc_depth(d, k, 7, 10)
        if a is not None and top is not None and a >= top - 0.02:
            return k
    return None


def mw_p(x, y):
    """Exact one-sided Mann-Whitney p for x > y (small n, all splits)."""
    allv = list(x) + list(y); n = len(x)
    obs = sum((a > b) + 0.5 * (a == b) for a in x for b in y)
    cnt = tot = 0
    for idx in itertools.combinations(range(len(allv)), n):
        xs = [allv[i] for i in idx]; ys = [allv[i] for i in range(len(allv)) if i not in idx]
        u = sum((a > b) + 0.5 * (a == b) for a in xs for b in ys)
        cnt += u >= obs - 1e-12; tot += 1
    return cnt / tot


def rule(x, y):
    x, y = [v for v in x if v is not None], [v for v in y if v is not None]
    if len(x) < 3 or len(y) < 3:
        return {"decision": "unresolved (fewer than 3 seeds per arm)", "x": x, "y": y}
    d = float(np.mean(x) - np.mean(y)); p = mw_p(x, y)
    dec = "supported" if (p <= 0.05 and d >= 0.05) else "not supported" if d <= 0 else "inconclusive"
    return {"decision": dec, "mean_diff": d, "mw_p": p, "arm": x, "coco": y}


def main():
    per = {}
    for arm in ("coco", "at", "rt"):
        for s in SEEDS:
            d = load(arm, s)
            if d is None:
                continue
            per[f"{arm}_s{s}"] = {"D710": acc_depth(d, 10, 7, 10), "D11": acc_depth(d, 16, 11, 11),
                                   "D12p": acc_depth(d, 16, 12, 99), "SHAL": acc_depth(d, 10, 3, 6),
                                   "LU": acc_m(d, 10, range(7)) - acc_m(d, 0, range(7)),
                                   "competent": acc_m(d, 10, range(7)) >= 0.75, "Ksat": ksat(d)}
    for s in SEEDS:
        d = load("cot", s)
        if d is not None:
            per[f"cot_s{s}"] = {"D710": cot_depth(d, 7, 10), "D11": cot_depth(d, 11, 11), "D12p": cot_depth(d, 12, 99),
                                "SHAL": cot_depth(d, 3, 6),
                                "competent": np.mean([v for k, v in d["by_ext"].items() if int(k) <= 6]) >= 0.75,
                                "mean_gen_tokens": float(np.mean(d.get("gen_tokens", [np.nan])))}
    get = lambda arm, k: [per[f"{arm}_s{s}"][k] for s in SEEDS if f"{arm}_s{s}" in per]
    out = {"per_seed": per}
    for arm in ("at", "rt"):
        out[f"P8-1_{arm}"] = rule(get(arm, "D710"), get("coco", "D710"))
        out[f"P8-2_{arm}"] = rule(get(arm, "D11"), get("coco", "D11"))
        out[f"D12p_{arm}_minus_coco"] = (float(np.mean(get(arm, "D12p")) - np.mean(get("coco", "D12p")))
                                         if get(arm, "D12p") and get("coco", "D12p") else None)
    for arm in ("coco", "at", "rt"):
        lu = get(arm, "LU")
        out[f"P8-3_{arm}"] = {"LU": lu, "met": sum(v >= 0.15 for v in lu) >= 3 if len(lu) == 4 else "incomplete"}
        out[f"competent_{arm}"] = sum(bool(v) for v in get(arm, "competent"))
    at_k, co_k = get("at", "Ksat"), get("coco", "Ksat")
    out["P8-5"] = {"Ksat_at": at_k, "Ksat_coco": co_k, "Ksat_rt": get("rt", "Ksat"),
                   "signature_met": (sum(k is not None and k >= 6 for k in at_k) >= 3 and
                                     sum(k is not None and k <= 4 for k in co_k) >= 3)}
    cot = get("cot", "D710")
    if cot:
        best = max(("coco", "at", "rt"), key=lambda a: np.mean(get(a, "D710")) if get(a, "D710") else -1)
        out["P8-4"] = {"best_latent_arm": best, "best_latent_D710": float(np.mean(get(best, "D710"))),
                       "cot_D710": float(np.mean(cot)),
                       "competitive": float(np.mean(get(best, "D710"))) >= float(np.mean(cot)) - 0.03,
                       "cot_mean_gen_tokens": float(np.nanmean(get("cot", "mean_gen_tokens"))),
                       "latent_steps": 10, "competent_cot": sum(bool(v) for v in get("cot", "competent"))}
    json.dump(out, open("results/p8.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_seed"}, indent=1))


if __name__ == "__main__":
    main()
