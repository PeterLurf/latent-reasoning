"""Score THEORY §79 (75ec955) from c1/runs/p79_<arm>[_f<k>]_s<seed>/eval.json; arms pfstd, rfstd, bow, tab (8 training
families) and the PF-std sweep pfstd_f2 / pfstd_f4; seeds 2081-2085. Committed before any §79 run.
Primary statistics use the frozen sigma; the sigma-floor values are reported alongside.
* P1: PF-std certified on the training phrasings (product automaton over every training phrasing) in >= 4/5 seeds.
* P2: for at least one held-out family H in H1-H4, all of:
    - median PF-std agreement >= 0.90;
    - PF-std's family certificate holds in >= 3/5 seeds;
    - median over seeds of (PF-std - BOW) and of (PF-std - RF-std) agreement, paired by seed, both >= 0.20;
  and PF-std's held-out minimal pairs (H4 rotations) median accuracy >= 0.90.
* Descriptive: every arm's median agreement / certificates / accuracy N 8-128 per family, POS / NEG controls, T6
  (training) minimal pairs, the sweep, TAB (0 by construction on held-out phrasings), phrasing consistency.
A verdict with fewer than 5 PF-std (or BOW / RF-std) seeds is "incomplete".
    python c1/score_79.py
"""
import json
import os

import numpy as np

SEEDS = range(2081, 2086); ARMS = ("pfstd", "rfstd", "bow", "tab", "pfstd_f2", "pfstd_f4"); HELD = ("H1", "H2", "H3", "H4")
FAMS = HELD + ("POS", "NEG", "T6")


def main():
    R = {a: {s: json.load(open(f"c1/runs/p79_{a}_s{s}/eval.json")) for s in SEEDS if os.path.exists(f"c1/runs/p79_{a}_s{s}/eval.json")}
         for a in ARMS}
    g = lambda e, f, k, col="frozen_sigma": e["eval"].get(f, {}).get(col, {}).get(k)          # noqa: E731
    pf, bw, rf = R["pfstd"], R["bow"], R["rfstd"]
    v = {}
    k1 = sum(bool(e["train"]["certificate"]["certified"]) for e in pf.values())
    v["P1"] = {"certified": k1, "of": len(pf), "met": k1 >= 4}
    p2 = {}
    for H in HELD:
        ag = [g(e, H, "agreement") for e in pf.values()]
        cert = sum(bool((g(e, H, "certificate") or {}).get("certified")) for e in pf.values())
        dB = [g(pf[s], H, "agreement") - g(bw[s], H, "agreement") for s in pf if s in bw]
        dR = [g(pf[s], H, "agreement") - g(rf[s], H, "agreement") for s in pf if s in rf]
        r = {"pf_median_agreement": float(np.median(ag)) if ag else None, "pf_family_cert": cert,
             "median_diff_vs_bow": float(np.median(dB)) if dB else None, "median_diff_vs_rf": float(np.median(dR)) if dR else None}
        r["family_ok"] = bool(ag and r["pf_median_agreement"] >= 0.90 and cert >= 3 and dB and dR
                              and r["median_diff_vs_bow"] >= 0.20 and r["median_diff_vs_rf"] >= 0.20)
        p2[H] = r
    mp = [g(e, "H4", "minimal_pair_acc") for e in pf.values()]
    mp_med = float(np.median(mp)) if mp else None
    v["P2"] = {"families": p2, "pf_minimal_pairs_H4_median": mp_med,
               "met": any(r["family_ok"] for r in p2.values()) and mp_med is not None and mp_med >= 0.90}
    if len(pf) < 5 or len(bw) < 5 or len(rf) < 5:
        for k in ("P1", "P2"):
            v[k]["status"] = "incomplete"
    table = {}
    for a in ARMS:
        if not R[a]:
            continue
        t = {"n": len(R[a]), "train_certified": sum(bool(e["train"]["certificate"]["certified"]) for e in R[a].values()),
             "train_acc128_median": float(np.median([e["train"]["acc"]["128"] for e in R[a].values()])),
             "phrasing_consistency_median": float(np.median([e["train"]["phrasing_consistency"] for e in R[a].values()]))}
        for f in FAMS:
            for col in ("frozen_sigma", "sigma_floor"):
                ag = [g(e, f, "agreement", col) for e in R[a].values() if g(e, f, "agreement", col) is not None]
                if ag:
                    t[f"{f}_{col}_agreement_median"] = float(np.median(ag))
            cs = [bool((g(e, f, "certificate") or {}).get("certified")) for e in R[a].values() if g(e, f, "certificate") is not None]
            if cs:
                t[f"{f}_cert"] = sum(cs)
            accs = [g(e, f, "acc")["128"] for e in R[a].values() if g(e, f, "acc")]
            if accs:
                t[f"{f}_acc128_median"] = float(np.median(accs))
            mps = [g(e, f, "minimal_pair_acc") for e in R[a].values() if g(e, f, "minimal_pair_acc") is not None]
            if mps:
                t[f"{f}_minimal_pairs_median"] = float(np.median(mps))
        table[a] = t
    res = {"verdicts": v, "arms": table}
    json.dump(res, open("theory/results/score_s79.json", "w", newline="\n"), indent=1)
    for a, t in table.items():
        print(a, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in t.items() if "sigma_floor" not in k})
    print("P1", v["P1"]); print("P2", json.dumps(v["P2"]))


if __name__ == "__main__":
    main()
