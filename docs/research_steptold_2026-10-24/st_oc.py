"""STEP-TOLD $0 OC + cost (CPU, 1 thread, idle priority; numpy only). Every gate of ST_draft sec. 6-7 under named truths, at ONE seed, plus the
cost model from the measured §228 / §229 pilot files and the expected spend. Inputs read (closed runs, not blinded): boxes/runs_fsq/{s228_pdir2c_s1,
s228_plat2_s1, s229_plat2c_s1}/{train_summary,eval_acc,log}.json(l); st_cue_check.json (token lengths).
    python -B st_oc.py
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
except Exception:  # noqa: BLE001
    pass
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..")); RUNS = os.path.join(ROOT, "boxes", "runs_fsq")
OUT = os.path.join(HERE, "st_oc.json"); REPS = 20000; rng = np.random.default_rng(23246)
MAJ = 0.11                                                               # answer / value majority share ~0.10-0.12 (REVIEW_LB #10; st_cue_check C3 0.095)


def b(p, n):
    return rng.binomial(n, p, REPS) / n


def lp2(q, n_eff=495):
    """pass-1 aux value acc on d = 2 items in the last 1k bin; worst-case clustering: one effective draw per d = 2 step (495 steps x 64 items)."""
    return b(q, n_eff) >= 0.80


def lp3(pl, pc, n=1000):
    L, C = b(pl, n), b(pc, n); return (L >= 0.70) & (L >= C + 0.30)


def lp5(pl, drop, n=1000):
    """MEAN-AT-1 on TEST2 d = 2: paired; intact acc pl, ablated acc pl - drop (per-item: ablated correct only if intact correct, the conservative pairing)."""
    I = rng.random((REPS, n)) < pl; keep = rng.random((REPS, n)) < (max(pl - drop, 0) / pl if pl > 0 else 0); A = I & keep
    return (I.mean(1) - A.mean(1)) >= 0.30


def leak(q_aux1, q_aux2, q_ans, n=1000):
    """CUE-ONLY guard: any of the three readings > MAJ + 0.05 -> LEAK (pilot VOID)."""
    return (b(q_aux1, n) > MAJ + 0.05) | (b(q_aux2, n) > MAJ + 0.05) | (b(q_ans, n) > MAJ + 0.05)


def ask(a1, a2, n=1000):
    A1, A2 = b(a1, n), b(a2, n); return (A2 >= 0.70) & (A1 >= 0.80)              # d1 < 0.80 = UNDERTRAINED (not a pass)


def main():
    res = {"reps": REPS, "majority": MAJ}
    # ---------------- P1-ASK (local, $0)
    res["P1_ASK"] = {nm: float(ask(a1, a2).mean()) for nm, (a1, a2) in {
        "told_learned (d1 0.95, d2 0.92)": (0.95, 0.92), "no_told (d1 0.95, d2 0.15)": (0.95, 0.15), "partial (d1 0.95, d2 0.60)": (0.95, 0.60),
        "at_bar (d1 0.85, d2 0.70)": (0.85, 0.70), "distractor_cost (d1 0.98, d2 0.80)": (0.98, 0.80)}.items()}
    # ---------------- pilot P-2 (cloud), joint over LP2, LP3, LP5 and no LEAK (LP0 / LP1 treated as met under every truth: not sampling gates)
    truths = {
        "no_effect (LAT = CD = 0.10; aux 0.25; drop 0)": dict(q=0.25, pl=0.10, pc=0.10, drop=0.0),
        "strong_serial (LAT 0.90, CD 0.12; aux 0.90; drop 0.75)": dict(q=0.90, pl=0.90, pc=0.12, drop=0.75),
        "marginal (LAT 0.75, CD 0.40; aux 0.82; drop 0.35)": dict(q=0.82, pl=0.75, pc=0.40, drop=0.35),
        "at_bars (LAT 0.70, CD 0.40; aux 0.80; drop 0.30)": dict(q=0.80, pl=0.70, pc=0.40, drop=0.30),
        "recompute (LAT 0.85, CD 0.12; aux 0.90; drop 0.02)": dict(q=0.90, pl=0.85, pc=0.12, drop=0.02),
        "cue_solves_one_pass (LAT 0.90, CD 0.70)": dict(q=0.90, pl=0.90, pc=0.70, drop=0.75),
        "pass1_only (aux 0.90; LAT 0.15, CD 0.12)": dict(q=0.90, pl=0.15, pc=0.12, drop=0.0),
    }
    P2 = {}
    for nm, t in truths.items():
        g2, g3, g5 = lp2(t["q"]), lp3(t["pl"], t["pc"]), lp5(t["pl"], t["drop"]); lk = leak(MAJ, MAJ, MAJ)
        P2[nm] = {"LP2": float(g2.mean()), "LP3": float(g3.mean()), "LP5": float(g5.mean()), "PASS": float((g2 & g3 & g5 & ~lk).mean())}
    res["P2_pilot"] = P2; res["LEAK_false_alarm_at_majority"] = float(leak(MAJ, MAJ, MAJ).mean()); res["LEAK_power_at_0.25"] = float(leak(0.25, MAJ, MAJ).mean())
    # ---------------- P-4 (cloud), LP4: LAT d3 >= 0.60, d4 >= 0.50, each >= CD + 0.25 (1,000 items per d)
    def lp4(l3, l4, c3, c4, n=1000):
        L3, L4, C3, C4 = b(l3, n), b(l4, n), b(c3, n), b(c4, n); return (L3 >= 0.60) & (L4 >= 0.50) & (L3 >= C3 + 0.25) & (L4 >= C4 + 0.25)
    res["P4_LP4"] = {"no_effect": float(lp4(0.1, 0.1, 0.1, 0.1).mean()), "strong (0.85 / 0.75 vs 0.12)": float(lp4(0.85, 0.75, 0.12, 0.12).mean()),
                     "at_bars (0.60 / 0.50 vs 0.35 / 0.25)": float(lp4(0.60, 0.50, 0.35, 0.25).mean())}
    # ---------------- Stage 1 R2' MEAN-AT-t* (pooled over B >= 3 depths x 1,000 items): drop >= 0.30
    res["S1_R2prime"] = {"serial (LAT 0.75, drop 0.55)": float(lp5(0.75, 0.55, 3000).mean()), "recompute (drop 0.03)": float(lp5(0.75, 0.03, 3000).mean()),
                         "at_bar (drop 0.30)": float(lp5(0.75, 0.30, 3000).mean())}
    # ---------------- cost (measured inputs)
    ts = {r: json.load(open(os.path.join(RUNS, r, "train_summary.json"))) for r in ("s228_pdir2c_s1", "s228_plat2_s1", "s229_plat2c_s1")}
    ev = {r: json.load(open(os.path.join(RUNS, r, "eval_acc.json"))).get("sec") for r in ts}
    L = json.load(open(os.path.join(HERE, "st_cue_check.json")))["C4_lengths"]
    Lp = lambda dmax: np.mean([L[str(d)]["think_plain_mean"] for d in range(1, dmax + 1)])  # noqa: E731
    Ls = lambda dmax: np.mean([L[str(d)]["think_st_mean"] for d in range(1, dmax + 1)])  # noqa: E731
    Lc = lambda dmax: np.mean([L[str(d)]["cued_direct_dec_mean"] for d in range(1, dmax + 1)])  # noqa: E731
    Ld = lambda dmax: np.mean([L[str(d)]["plain_dec_mean"] for d in range(1, dmax + 1)])  # noqa: E731
    K = 8

    def lat_units(dmax, st):
        """[ASSUMED] token-linear work per step, in units of 6 blocks over the plain d <= 2 think length: lower blocks once (cached) or once per distinct cue
        (mean d) for ST; upper blocks K times; decoder 12 blocks over dec; aux reads 12 blocks over ~13 tokens for mean d passes."""
        md = np.mean(range(1, dmax + 1)); Lt = Ls(dmax) if st else Lp(dmax)
        return ((md if st else 1.0) + K) * Lt + 2 * Ld(dmax) + md * 2 * 13
    base_u = lat_units(2, False); sec_lat2 = [ts["s228_plat2_s1"]["train_sec"], ts["s229_plat2c_s1"]["train_sec"]]
    f_st2 = lat_units(2, True) / base_u; f_st4 = lat_units(4, True) / base_u; f_st8 = lat_units(8, True) / base_u
    f_cd2 = Lc(2) / Ld(2); f_cd4 = Lc(4) / Ld(2); f_cd8 = Lc(8) / Ld(2); f_d8 = Ld(8) / Ld(2); f_d4 = Ld(4) / Ld(2)
    rate = (0.20, 0.27)                                                  # $/h effective (realised: §228 pair $0.58, §229 $0.54 over their host hours)
    setup = 900
    sdir = ts["s228_pdir2c_s1"]["train_sec"]

    def dollars(sec):
        return [round(sec / 3600 * r, 2) for r in rate]
    lat2 = [s * f_st2 for s in sec_lat2]; ev_lat2 = ev["s228_plat2_s1"] * 1.6 + 200          # [ASSUMED] eval: lower per distinct cue (x1.37) x length, + MEAN-AT-1, CUE-ONLY, D-SWAP2
    cd2 = sdir * f_cd2; p2_sec = [l + ev_lat2 + cd2 + 30 + setup for l in lat2]
    p2_fresh_dir = sdir + 20
    lat4 = [s * f_st4 * 2 for s in sec_lat2]; cd4 = sdir * f_cd4 * 2; p4_sec = [l + ev_lat2 * 2 + cd4 + 60 + setup for l in lat4]
    lat8 = [s * f_st8 * 6 for s in sec_lat2]; cd8 = sdir * f_cd8 * 6; d8 = sdir * f_d8 * 6; s1_eval = 3 * 3600                # Stage 1: 48k steps = 6 x 8k
    s1_sec = [l + cd8 + d8 + s1_eval + 2 * setup for l in lat8]
    cost = {"factors": {"lat_st_vs_s228_lat_d2": f_st2, "lat_st_d4": f_st4, "lat_st_d8": f_st8, "cued_direct_d2": f_cd2, "cued_direct_d8": f_cd8, "direct_d8": f_d8},
            "measured_train_sec": {"s228_pdir2c_s1": sdir, "s228_plat2_s1": sec_lat2[0], "s229_plat2c_s1": sec_lat2[1]}, "measured_eval_acc_sec": ev,
            "rate_usd_per_h": rate,
            "P2_pilot": {"host_h": [round(s / 3600, 2) for s in p2_sec], "usd_low_high": [dollars(p2_sec[0])[0], dollars(p2_sec[1])[1]]},
            "P2_fresh_plain_direct_if_not_3090": dollars(p2_fresh_dir)[1],
            "P4_pilot": {"host_h": [round(s / 3600, 2) for s in p4_sec], "usd_low_high": [dollars(p4_sec[0])[0], dollars(p4_sec[1])[1]]},
            "Stage1": {"host_h": [round(s / 3600, 1) for s in s1_sec], "usd_low_high": [dollars(s1_sec[0])[0], dollars(s1_sec[1])[1]]}}
    hi = lambda k: cost[k]["usd_low_high"][1]  # noqa: E731
    cost["guard_x1.3"] = {"P2": round(1.3 * (hi("P2_pilot") + cost["P2_fresh_plain_direct_if_not_3090"]), 2), "P4": round(1.3 * hi("P4_pilot"), 2),
                          "Stage1": round(1.3 * hi("Stage1"), 2)}
    cost["family_cap_guard"] = round(sum(cost["guard_x1.3"].values()), 2)
    pri = {"P1_ASK": 0.5, "P2_given_ASK": 0.35, "P4_given_P2": 0.6, "C1_TOLD_given_P4": 0.4}
    e_spend = pri["P1_ASK"] * (hi("P2_pilot") + pri["P2_given_ASK"] * (hi("P4_pilot") + pri["P4_given_P2"] * hi("Stage1")))
    cost["priors_ASSUMED"] = pri; cost["expected_spend_high_rate"] = round(e_spend, 2)
    cost["P_C1_TOLD_overall"] = round(pri["P1_ASK"] * pri["P2_given_ASK"] * pri["P4_given_P2"] * pri["C1_TOLD_given_P4"], 3)
    res["cost"] = cost
    with open(OUT, "w", newline="\n") as f:
        json.dump(res, f, indent=1); f.write("\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
