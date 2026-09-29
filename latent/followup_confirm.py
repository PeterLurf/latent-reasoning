"""Score the frozen predictions of docs/protocol2_frozen.json on fresh-confirmation summaries.

    python -m latent.followup_confirm
"""
from __future__ import annotations

import json
import os

import numpy as np

REC = {"s0": ("runs/r3_pc_gated_gpt2", ""), "s0@k0": ("runs/r3_pc_gated_gpt2", "_k0"),
       "s1": ("runs/main_s1", ""), "s2": ("runs/main_s2", ""), "s3": ("runs/main_s3", ""),
       "s4": ("runs/dev_s4", ""), "s5": ("runs/main_s5", "")}
HELDOUT = ("s3", "s4", "s5")      # PROTOCOL3 addenda C/D: scored separately, never folded into the Part II verdicts
PAUSE = {"p0": "runs/pause_s0", "p1": "runs/pause_s1", "p2": "runs/pause_s2"}


def load(run, split, tag=""):
    p = os.path.join(run, "followup", split + tag, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def frac(cache, cond):
    return cache["conds"][cond]["targeted"][0] / cache["clean_gap"][0]


def ols_r2(Xtr, ytr, Xte, yte):
    ok_tr = ~np.isnan(Xtr).any(1) & ~np.isnan(ytr)
    ok_te = ~np.isnan(Xte).any(1) & ~np.isnan(yte)
    Xtr, ytr, Xte, yte = Xtr[ok_tr], ytr[ok_tr], Xte[ok_te], yte[ok_te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd]
    beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    pred = np.c_[np.ones(len(Xte)), (Xte - mu) / sd] @ beta
    return float(1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())


BASE = ["W", "diff_norm", "base_conf"]
GEOM = BASE + ["gain", "resid_ratio", "pred_target_logit", "pred_col_logit"]


def rows(summary):
    return [r for W, R in summary["results"].items() for r in R.get("jac_rows", [])]


def main():
    out = {"recurrent": {}, "pause": {}, "verdicts": {}}
    V = {f"P{i}": [] for i in ("1", "1b", "2", "3", "4", "5", "6", "7", "8")}
    for name, (run, tag) in REC.items():
        S = load(run, "fresh", tag)
        if S is None:
            continue
        R = S["results"]
        o = {"k": S["k"]}
        # P1 route shift
        sh = {W: {c: frac(R[W]["cache"], c) for c in ("T", "L", "TL", "clamp", "Gq", "Gl", "TLGq")} for W in R}
        o["route_shares"] = sh
        V["P1"].append((name, sh["8"]["TL"] < sh["2"]["TL"] and sh["8"]["Gq"] > sh["2"]["Gq"]))
        # P2 dose rescue
        d = {W: R[W]["dose_remove"]["kinds"] for W in R}
        t15_8 = d["8"]["full"]["doses"]["1.5"]["targeted"][0]
        t10_2 = d["2"]["full"]["doses"]["1.0"]["targeted"][0]
        o["dose"] = {W: {kind: {lam: [e["targeted"], e["collateral_mean"][0], e["perturb_norm"]] for lam, e in kd["doses"].items()}
                         for kind, kd in d[W].items()} for W in d}
        o["recovery_ratio"] = {W: {kind: d[W][kind]["doses"]["1.0"]["recovery_ratio_of_means"] for kind in d[W]} for W in d}
        o["clean_gap"] = {W: R[W]["dose_remove"]["clean_gap"] for W in R}
        V["P2"].append((name, t15_8 < t10_2))
        # P3 direction not size
        p01 = d["4"]["pattern"]["doses"]["0.1"]; f15 = d["4"]["filter"]["doses"]["1.5"]
        o["P3"] = {"pattern@0.1": [p01["targeted"][0], p01["perturb_norm"]], "filter@1.5": [f15["targeted"][0], f15["perturb_norm"]]}
        V["P3"].append((name, p01["targeted"][0] > f15["targeted"][0]))
        # P4 swap
        sw = {}
        ok = True
        for W in R:
            c = R[W]["swap"]["conds"]
            s1 = c["swap"]["kinds"]["full"]["1.0"]; r1 = c["remove_A"]["kinds"]["full"]["1.0"]; i1 = c["insert_B"]["kinds"]["full"]["1.0"]
            sw[W] = {"swap": [s1["collateral_mean"][0], s1["collateral_signed"][0], s1["A_drop"][0], s1["B_rise"][0]],
                     "remove": [r1["collateral_mean"][0], r1["collateral_signed"][0], r1["A_drop"][0]],
                     "insert": [i1["collateral_mean"][0], i1["collateral_signed"][0], i1["B_rise"][0]],
                     "clean_swap_col": c["swap"]["clean"]["collateral_mean"][0], "clean_remove_col": c["remove_A"]["clean"]["collateral_mean"][0],
                     "same_status_col": c["same_status"]["kinds"]["full"]["1.0"]["collateral_mean"][0]}
            ok &= abs(s1["collateral_signed"][0]) < abs(r1["collateral_signed"][0]) and s1["collateral_mean"][0] >= 0.8 * r1["collateral_mean"][0]
        o["swap"] = sw
        V["P4"].append((name, bool(ok)))
        # P5 continuation
        ct, ok = {}, True
        for W in R:
            for j in (2, 3):
                c = R[W][f"cont_j{j}"]
                fi, fw = c["follow_index_full_Dn"], c["follow_index_full_Wr"]
                ct[f"W{W}_j{j}"] = {"follow_true": fi, "follow_wrong": fw, "follow_pattern": c["follow_index_pattern_Dn"],
                                    "natural_C_rec": c["sets"]["C_rec"]["natural_Nt"][0], "edit_C_rec": c["sets"]["C_rec"]["full_Dn"][0],
                                    "edit_C_don": c["sets"]["C_don"]["full_Dn"][0], "competence_Nt": c["competence"]["Nt_C_rec_reachable"][0], "n": c["n"]}
                ok &= fi[1] > 0 and fw[0] < fi[1]
        o["continuation"] = ct
        V["P5"].append((name, bool(ok)))
        # P6 competition
        at, ok = {}, True
        for W in R:
            e = R[W]["attn"]["doses"]["1.0"]
            at[W] = {k_: e[k_] for k_ in e}
            oth = e["att_boost_other_attached"]
            ok &= oth[2] < 0 and abs(e["dec_boost_attached"][0]) < abs(oth[0]) / 3 and abs(e["all_boost_attached"][0]) < abs(oth[0]) / 3
        o["attention"] = at
        V["P6"].append((name, bool(ok)))
        # P7 gain restoration
        gr = {W: {t: R[W]["gain"][t]["targeted"][0] for t in ("plain", "restored")} for W in R if "gain" in R[W]}
        o["gain"] = gr
        V["P7"].append((name, all(abs(v["restored"] - v["plain"]) < 0.03 for v in gr.values())))
        # P8 controllability
        Sd = load(run, "discovery")
        if Sd is not None and tag == "":
            tr, te = rows(Sd), rows(S)
            f = lambda rs, cols: np.array([[r_[c] for c in cols] for r_ in rs], float)
            res = {}
            for y in ("obs_target_prob", "obs_col_prob"):
                ytr, yte = np.array([r_[y] for r_ in tr], float), np.array([r_[y] for r_ in te], float)
                res[y] = {"baseline": ols_r2(f(tr, BASE), ytr, f(te, BASE), yte), "geometry": ols_r2(f(tr, GEOM), ytr, f(te, GEOM), yte)}
            corr = lambda a_, b_: float(np.corrcoef(a_, b_)[0, 1])
            res["linear_validity_corr_target"] = corr(f(te, ["pred_target_logit"])[:, 0], f(te, ["obs_target_logit"])[:, 0])
            res["resid_ratio_by_W"] = {W: float(np.mean([r_["resid_ratio"] for r_ in S["results"][W]["jac_rows"]])) for W in S["results"]}
            o["controllability"] = res
            V["P8"].append((name, all(res[y]["geometry"] - res[y]["baseline"] < 0.05 for y in ("obs_target_prob", "obs_col_prob"))))
        out["recurrent"][name] = o
    # addendum A: P9 composition, P10 gain model
    V["P9"], V["P10"] = [], []
    for name, (run, tag) in REC.items():
        if tag:
            continue
        Sp = load(run, "fresh", "_pair")
        if Sp is not None:
            ok, rec = True, {}
            for W, R_ in Sp["results"].items():
                p = R_["pair"]
                rec[W] = {"both_f1": p["both"]["follow1"], "both_f2": p["both"]["follow2"],
                          "only1_f2": p["only1"]["follow2"][0], "only2_f1": p["only2"]["follow1"][0],
                          "additivity": [p["additivity_branch1"][0], p["additivity_branch2"][0]], "n": p["n"]}
                if W in ("4", "8"):
                    ok &= p["both"]["follow1"][1] > 0 and p["both"]["follow2"][1] > 0
                    ok &= abs(p["only1"]["follow2"][0]) < 0.03 and abs(p["only2"]["follow1"][0]) < 0.03
                    ok &= (p["additivity_branch1"][0] + p["additivity_branch2"][0]) / 2 < 0
            out["recurrent"][name]["composition"] = rec
            V["P9"].append((name, bool(ok)))
        Sd, Sf = load(run, "discovery", "_gainfit"), load(run, "fresh", "_gainfit")
        if Sd is not None and Sf is not None:
            def xy(S):
                x, y = [], []
                for W, R_ in S["results"].items():
                    d = np.array(R_["gainfit"]["diff_norm"])
                    x += [np.log(int(W))] * len(d); y += list(np.log(d))
                return np.array(x), np.array(y)
            xd, yd = xy(Sd); xf, yf = xy(Sf)
            res = {}
            for beta in (0.0, 0.5, 1.0):
                a = np.mean(yd + beta * xd)                 # intercept fitted on discovery
                res[f"beta={beta}"] = float(np.mean((yf - (a - beta * xf)) ** 2))
            bfree = -np.polyfit(xd, yd, 1)[0]
            a = np.mean(yd + bfree * xd)
            res["free_beta_discovery"] = float(bfree)
            res["mse_free"] = float(np.mean((yf - (a - bfree * xf)) ** 2))
            res["free_beta_fresh"] = float(-np.polyfit(xf, yf, 1)[0])
            res["mean_diff_by_W"] = {W: float(np.mean(R_["gainfit"]["diff_norm"])) for W, R_ in Sf["results"].items()}
            out["recurrent"][name]["gain_model"] = res
            best = min(("beta=0.0", "beta=0.5", "beta=1.0"), key=lambda k_: res[k_])
            V["P10"].append((name, best == "beta=0.5" and 0.4 <= res["free_beta_fresh"] <= 0.9))
    # addendum B (PROTOCOL3): P14 within-state relative code
    V["P14"], comp = [], []
    for name, (run, tag) in REC.items():
        if tag:
            continue
        Sa = load(run, "fresh", "_amp")
        if Sa is None:
            continue
        ok, rec = True, {}
        for W in ("4", "8"):
            a = Sa["results"][W]["amp"]["lams"]
            e0, e2 = a["0.0"], a["2.0"]
            rec[W] = {"erase_own_deep": e0["own_deep"], "double_own_deep": e2["own_deep"], "double_others": e2["other_attached"],
                      "random_abs": e0["random_abs_attached"][0]}
            ok &= e0["own_deep"][2] < 0 and e2["own_deep"][1] > 0 and e0["random_abs_attached"][0] < abs(e0["own_deep"][0]) / 3
        comp.append((name, bool(Sa["results"]["4"]["amp"]["lams"]["2.0"]["other_attached"][2] < 0)))
        out["recurrent"][name]["amplify"] = rec
        V["P14"].append((name, bool(ok)))
    out["P14_competition_W4_seeds"] = int(sum(c for n, c in comp if n not in HELDOUT))
    out["P14_competition_W4_heldout"] = {n: c for n, c in comp if n in HELDOUT}
    for name, run in PAUSE.items():
        S = load(run, "fresh")
        if S is None:
            continue
        sh = {W: {c: frac(R_["cache"], c) for c in ("L", "Gq", "Gl", "LGq")} for W, R_ in S["results"].items()}
        at = {W: R_["attn"]["doses"]["1.0"] for W, R_ in S["results"].items()}
        out["pause"][name] = {"route_shares": sh, "attention": at}
        V["P1b"].append((name, all(v["Gq"] >= 0.85 and v["L"] <= 0.15 for v in sh.values())))
    part2 = lambda v: [x for n, x in v if n not in HELDOUT]
    out["verdicts"] = {p: {"per_model": dict(v), "all": all(part2(v)) if part2(v) else None,
                           "heldout": {n: x for n, x in v if n in HELDOUT}} for p, v in V.items()}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/followup_confirm.json", "w"), indent=1)
    for p, v in out["verdicts"].items():
        print(p, v)


if __name__ == "__main__":
    main()
