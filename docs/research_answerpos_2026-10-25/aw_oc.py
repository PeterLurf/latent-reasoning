"""ANSWER-WRITE (AW) $0 OC + cost (AW_draft.md sec. 5). CPU, 1 thread, idle priority; numpy only. Every binding P-2 gate under named truths at ONE seed,
the G-AW0 probe gate's OC, and the cost model from measured §228 / §233 files. Inputs (closed runs, not blinded): boxes/runs_fsq/{st_plst2_s1, st_pcd2_s1,
s228_pdir2c_s1}/{train_summary, eval}.json, {s228_plat2_s1_gtime, st_plst2_s1_gtime}/timing.json, st_cue_check.json (token lengths).
    python -B docs/research_answerpos_2026-10-25/aw_oc.py      -> aw_oc.json
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
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..")); RUNS = os.path.join(ROOT, "boxes", "runs_fsq")
OUT = os.path.join(HERE, "aw_oc.json"); REPS = 20000; rng = np.random.default_rng(23472); MAJ = 0.11; DIR_D2 = 0.093   # s228_pdir2c_s1 TEST2 d = 2 (233-4)


def b(p, n):
    return rng.binomial(n, min(max(p, 0.0), 1.0), REPS) / n


def lp2(q, n_eff=495):
    """pass-1 aux value acc on d = 2 items, last 1k bin; worst-case clustering: one effective draw per d = 2 step (st_oc's model)."""
    return b(q, n_eff) >= 0.80


def lp3(pl, pc, n=1000):
    L, C = b(pl, n), b(pc, n); return (L >= 0.70) & (L >= C + 0.30), C


def lp5(pl, drop, n=1000):
    """MEAN-AT-1 at K' = d, paired (ablated correct only if intact correct: the conservative pairing)."""
    I = rng.random((REPS, n)) < pl; keep = rng.random((REPS, n)) < (max(pl - drop, 0) / pl if pl > 0 else 0); A = I & keep
    return (I.mean(1) - A.mean(1)) >= 0.30


def swapk(f, fl, n=1000):
    """STATE-USE SWAP-K: follow_U >= 0.50 and >= floor + 0.30 (floor = MEAN-K answers' share in U on the same pairs)."""
    Fo, Fl = b(f, n), b(fl, n); return (Fo >= 0.50) & (Fo >= Fl + 0.30)


def cue_only_trip(q, n=1000):
    return b(q, n) > max(MAJ, DIR_D2) + 0.05


def main():
    res = {"reps": REPS, "majority": MAJ}
    # G-AW0: probe test acc on n_test = 600 >= 0.80
    res["G_AW0"] = {"truth %.3f" % t: float((b(t, 600) >= 0.80).mean()) for t in (1.0, 0.95, 0.855, 0.82, 0.80, 0.78, 0.75, 0.30)}
    truths = {
        "no_effect (aux 0.33 as s233; LAT = CD = 0.10)": dict(q=0.33, pl=0.10, pc=0.10, drop=0.0, ms=0.10, f=0.18, fl=0.18),
        "pass1_only (S_1 carries v_1, nothing reads it)": dict(q=0.90, pl=0.10, pc=0.10, drop=0.0, ms=0.10, f=0.18, fl=0.18),
        "pass1_plus_decoder (decoder applies op 2 to the stored v_1)": dict(q=0.92, pl=0.88, pc=0.12, drop=0.70, ms=0.10, f=0.80, fl=0.18),
        "strong_serial (passes chain)": dict(q=0.92, pl=0.90, pc=0.12, drop=0.75, ms=0.10, f=0.85, fl=0.18),
        "marginal": dict(q=0.82, pl=0.75, pc=0.40, drop=0.35, ms=0.30, f=0.60, fl=0.18),
        "at_bars": dict(q=0.80, pl=0.70, pc=0.40, drop=0.30, ms=0.50, f=0.50, fl=0.18),
        "weak_read (state read, half the time)": dict(q=0.90, pl=0.45, pc=0.12, drop=0.30, ms=0.10, f=0.40, fl=0.18),
        "recompute (trunk got better one-pass; state unused)": dict(q=0.90, pl=0.80, pc=0.12, drop=0.02, ms=0.78, f=0.20, fl=0.18),
        "init_solves_one_pass (CD-AN high)": dict(q=0.90, pl=0.90, pc=0.75, drop=0.75, ms=0.10, f=0.85, fl=0.18),
    }
    P2 = {}
    for nm, t in truths.items():
        g2 = lp2(t["q"]); g3, C = lp3(t["pl"], t["pc"]); g5 = lp5(t["pl"], t["drop"]); M = b(t["ms"], 1000); gm = M <= C + 0.10; gs = swapk(t["f"], t["fl"])
        trip = cue_only_trip(MAJ); ok = g2 & g3 & g5 & gm & gs & ~trip
        P2[nm] = {k: float(v.mean()) for k, v in (("LP2", g2), ("LP3", g3), ("LP5", g5), ("MEAN_STATE", gm), ("SWAP_K", gs), ("PASS", ok))}
    res["P2_pilot"] = P2; res["CUE_ONLY_false_trip_at_majority"] = float(cue_only_trip(MAJ).mean()); res["CUE_ONLY_power_at_0.25"] = float(cue_only_trip(0.25).mean())
    # ---------------- cost (measured inputs)
    ts = {r: json.load(open(os.path.join(RUNS, r, "train_summary.json"))) for r in ("st_plst2_s1", "st_pcd2_s1", "s228_pdir2c_s1")}
    ev = {r: {k: v.get("sec") for k, v in json.load(open(os.path.join(RUNS, r, "eval.json")))["parts"].items()} for r in ("st_plst2_s1", "st_pcd2_s1")}
    tm = {r: json.load(open(os.path.join(RUNS, r, "timing.json"))) for r in ("s228_plat2_s1_gtime", "st_plst2_s1_gtime")}
    L = json.load(open(os.path.join(ROOT, "docs", "research_steptold_2026-10-24", "st_cue_check.json")))["C4_lengths"]
    mean = lambda key, dmax, add=0.0: float(np.mean([L[str(d)][key] for d in range(1, dmax + 1)])) + add  # noqa: E731
    K = 8

    def units(dmax, kind):
        """[ASSUMED] token-linear work per step (st_oc's model): lower blocks once per distinct input, upper blocks K times, decoder 12 blocks, aux reads."""
        md = float(np.mean(range(1, dmax + 1))); Lt = {"plain": mean("think_plain_mean", dmax), "st": mean("think_st_mean", dmax), "aw": mean("think_st_mean", dmax, 4.0)}[kind]
        return ((1.0 if kind == "plain" else md) + K) * Lt + 2 * mean("plain_dec_mean", dmax) + md * 2 * 13
    lat_hi = ts["st_plst2_s1"]["train_sec"] * units(2, "aw") / units(2, "st")              # ST's measured time (micro 2, T-token write loop) x AW tokens
    lat_lo = tm["s228_plat2_s1_gtime"]["sec_per_step"] * 12000 * units(2, "aw") / units(2, "plain")   # micro 1 (s228 G-TIME) x AW tokens [ASSUMED fits]
    ev_lat = sum(v for v in ev["st_plst2_s1"].values() if v) + 160                          # + WS-READOUT / SWAP-1 / SWAP-K [ASSUMED 160 s]
    cd = ts["st_pcd2_s1"]["train_sec"] + (ev["st_pcd2_s1"].get("acc") or 30); setup = 900; rate = (0.20, 0.27)
    usd = lambda s, r: round(s / 3600 * r, 2)  # noqa: E731
    p2 = [lat_lo + ev_lat + cd + setup, lat_hi + ev_lat + cd + setup]
    stan = ts["st_plst2_s1"]["train_sec"] + sum(v for v in ev["st_plst2_s1"].values() if v) + setup        # ST-AN attribution control = §233 LAT-ST cost
    f4 = units(4, "aw") / units(2, "aw") * 16000 / 12000
    cdf = lambda dmax, steps: mean("cued_direct_dec_mean", dmax) / mean("cued_direct_dec_mean", 2) * steps / 12000  # noqa: E731
    p4 = [lat_lo * f4 + 2 * ev_lat + ts["st_pcd2_s1"]["train_sec"] * cdf(4, 16000) + setup, lat_hi * f4 + 2 * ev_lat + ts["st_pcd2_s1"]["train_sec"] * cdf(4, 16000) + setup]
    f8 = units(8, "aw") / units(2, "aw") * 48000 / 12000; s1_eval = 3 * 3600
    dir8 = ts["s228_pdir2c_s1"]["train_sec"] * mean("plain_dec_mean", 8) / mean("plain_dec_mean", 2) * 6
    s1 = [lat_lo * f8 + ts["st_pcd2_s1"]["train_sec"] * cdf(8, 48000) + dir8 + s1_eval + 2 * setup,
          lat_hi * f8 + ts["st_pcd2_s1"]["train_sec"] * cdf(8, 48000) + dir8 + s1_eval + 2 * setup]
    cost = {"measured": {"st_plst2_s1_train_sec": ts["st_plst2_s1"]["train_sec"], "st_pcd2_s1_train_sec": ts["st_pcd2_s1"]["train_sec"],
                         "s228_pdir2c_s1_train_sec": ts["s228_pdir2c_s1"]["train_sec"], "st_plst2_s1_eval_sec": ev["st_plst2_s1"],
                         "s228_plat2_s1_gtime_sps_micro1": tm["s228_plat2_s1_gtime"]["sec_per_step"], "st_plst2_s1_gtime_sps_micro2": tm["st_plst2_s1_gtime"]["sec_per_step"]},
            "factors": {"aw_vs_st_tokens_d2": units(2, "aw") / units(2, "st"), "aw_vs_plain_d2": units(2, "aw") / units(2, "plain"), "p4_lat": f4, "s1_lat": f8},
            "lat_aw_train_h": [round(lat_lo / 3600, 2), round(lat_hi / 3600, 2)], "rate_usd_per_h": rate}
    for nm, s in (("P2", p2), ("ST_AN_attribution", [stan, stan]), ("P4", p4), ("Stage1", s1)):
        cost[nm] = {"host_h": [round(s[0] / 3600, 2), round(s[1] / 3600, 2)], "usd_low_high": [usd(s[0], rate[0]), usd(s[1], rate[1])], "guard_x1.3": round(1.3 * usd(s[1], rate[1]), 2)}
    cost["P2"]["fresh_plain_direct_if_not_3090_usd"] = usd(ts["s228_pdir2c_s1"]["train_sec"] + 30, rate[1])
    cost["family_guard_sum"] = round(sum(cost[k]["guard_x1.3"] for k in ("P2", "ST_AN_attribution", "P4", "Stage1")), 2)
    cost["Stage1_wall_h_two_hosts"] = [round((lat_lo * f8 + s1_eval / 2 + setup) / 3600, 1), round((lat_hi * f8 + s1_eval / 2 + setup) / 3600, 1)]
    pri = {"G_AW0": 0.97, "LP2_given_gate": 0.65, "rest_of_P2_given_LP2": 0.30, "P4_given_P2": 0.5, "C1_given_P4": 0.4}
    pp2 = pri["G_AW0"] * pri["LP2_given_gate"] * pri["rest_of_P2_given_LP2"]
    hi = lambda k: cost[k]["usd_low_high"][1]  # noqa: E731
    e = pri["G_AW0"] * hi("P2") + pp2 * (hi("ST_AN_attribution") + hi("P4") + pri["P4_given_P2"] * hi("Stage1"))
    cost["priors_ASSUMED"] = pri; cost["P_P2_pass"] = round(pp2, 3); cost["P_C1_overall"] = round(pp2 * pri["P4_given_P2"] * pri["C1_given_P4"], 3)
    cost["expected_spend_high_rate"] = round(e, 2); res["cost"] = cost
    with open(OUT, "w", newline="\n") as f:
        json.dump(res, f, indent=1); f.write("\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
