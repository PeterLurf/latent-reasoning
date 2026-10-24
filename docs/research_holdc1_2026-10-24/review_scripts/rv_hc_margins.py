"""REVIEW_HC check R1 (CPU only, 2 threads, idle priority; committed eval_vb.json files only, no model, no torch).
Recomputes HC_draft's C1-KEPT margins / OC and adds:
  (a) the joint OC product and the COEXIST product as written in HC_draft section 6;
  (b) minimum detectable harm (50 % / 90 % fail power) per decisive statistic under the draft's own null sd;
  (c) PAIRED item-bootstrap SD of the slope_h DIFFERENCE for each FT pair (same resampled items in parent and child),
      vs the draft's unpaired 3 x sqrt(2) x SE term;
  (d) an unquantised onset statistic: mean onset (T_SET units) over n 3..8 items with an onset, and its slope on height
      (OLS of per-height MEAN onset), with FT-pair diffs and paired-bootstrap SDs;
  (e) PROBE(3,4,6) mean (the unsaturated part of the probe curve) FT-pair diffs.
    python -B docs/research_holdc1_2026-10-24/review_scripts/rv_hc_margins.py
Writes rv_hc_margins.out.json next to this script (LF)."""
import json
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS if hasattr(psutil, "IDLE_PRIORITY_CLASS") else 19)
except Exception:                                                          # noqa: BLE001
    pass
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); DRAFT = os.path.dirname(HERE)
sys.path.insert(0, DRAFT)
import hc_c1_noise_cpu as H  # noqa: E402  (module import has no side effects beyond env / nice)

INF = H.INF; NS = H.NS; T_SET = H.T_SET


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def mean_onset_stats(ns, hs, on, idx):
    byh, vals = {}, []
    for i in idx:
        if 3 <= ns[i] <= 8 and on[i] < INF:
            byh.setdefault(hs[i], []).append(on[i]); vals.append(on[i])
    hk = sorted(h for h in byh if len(byh[h]) >= 30)
    slope = H.ols(hk, [float(np.mean(byh[h])) for h in hk]) if len(hk) >= 2 else None
    return (float(np.mean(vals)) if vals else None), slope


def main():
    prev = json.load(open(os.path.join(DRAFT, "hc_c1_noise_cpu.out.json")))
    out = {}
    # (a) products
    joint = 1.0
    for k, v in prev["oc"].items():
        joint *= v["P_keep_2seeds_no_effect"]
    per_seed = math.sqrt(joint)
    out["a_joint_C1KEPT_2seeds"] = joint
    out["a_COEXIST_as_written"] = per_seed * 0.85 * 0.85 * 0.97 * 0.99 * per_seed
    out["a_COEXIST_with_VBC1_both_seeds"] = per_seed * 0.85 * 0.85 * 0.97 * 0.99 * 0.99 * per_seed
    # (b) minimum detectable harm under the draft's null sd (vs FT-CTRL: sqrt(2) sd, the binding comparison)
    mdh = {}
    for k, m in prev["margins"].items():
        sd = math.sqrt(2) * m["sd_null_assumed"]
        mdh[k] = {"delta": m["delta"], "sd_vs_ctrl": sd, "harm_at_50pct_fail": m["delta"], "harm_at_90pct_fail": m["delta"] + 1.2816 * sd,
                  "delta_over_maxFT": (m["delta"] / m["max_abs_FT_diff"]) if m["max_abs_FT_diff"] > 0 else None,
                  "delta_over_sd": m["delta"] / sd if sd > 0 else None}
        # one-lineage harm: P(C1-KEPT passes in the harmed lineage) at the draft's harm size
    out["b_min_detectable_harm"] = mdh
    out["b_slope_one_lineage_pass_given_harm"] = prev["oc"]["slope_h"]["P_pass_per_seed_given_harm"]
    # (c)-(e) item-level
    runs = sorted({r for a, b, _ in H.FT_PAIRS for r in (a, b)})
    ev = {r: json.load(open(os.path.join(H.R, r, "eval_vb.json"))) for r in runs}
    on = {r: H.onsets(ev[r]) for r in runs}
    ns = ev[runs[0]]["test_meta"]["n"]; hs = ev[runs[0]]["test_meta"]["height"]
    assert all(ev[r]["test_meta"]["n"] == ns and ev[r]["test_meta"]["height"] == hs for r in runs)
    allidx = list(range(len(ns)))
    point = {}
    for r in runs:
        s_, _, _ = H.onset_stats(ns, hs, on[r]); mo, ms = mean_onset_stats(ns, hs, on[r], allidx)
        p = ev[r]["probe"]; p346 = float(np.mean([sum(p[str(t)]["acc_by_n"][str(n)] for n in NS) / 6 for t in (3, 4, 6)]))
        point[r] = {"slope_h": s_, "mean_onset": mo, "slope_mean_onset": ms, "PROBE_346": p346,
                    "onset_hist": {str(t): sum(1 for i in allidx if 3 <= ns[i] <= 8 and on[r][i] == t) for t in list(T_SET) + [INF]}}
    out["point"] = point
    rng = np.random.default_rng(24024); byn = {n: [i for i, m in enumerate(ns) if m == n] for n in NS}; B = 300
    pairs = {}
    for a, b, lab in H.FT_PAIRS:
        ds, dmo, dms = [], [], []
        for _ in range(B):
            idx = [int(j) for n in NS for j in rng.choice(byn[n], size=len(byn[n]), replace=True)]
            sa, _, _ = H.onset_stats(ns, hs, on[a], idx); sb, _, _ = H.onset_stats(ns, hs, on[b], idx)
            moa, msa = mean_onset_stats(ns, hs, on[a], idx); mob, msb = mean_onset_stats(ns, hs, on[b], idx)
            if sa is not None and sb is not None:
                ds.append(sb - sa)
            dmo.append(mob - moa); dms.append(msb - msa)
        same = sum(1 for i in allidx if 3 <= ns[i] <= 8 and on[a][i] == on[b][i]) / sum(1 for i in allidx if 3 <= ns[i] <= 8)
        pairs[lab] = {"a": a, "b": b, "item_onset_agreement_n3_8": same,
                      "d_slope_h": point[b]["slope_h"] - point[a]["slope_h"], "paired_sd_d_slope_h": float(np.std(ds, ddof=1)),
                      "d_mean_onset": point[b]["mean_onset"] - point[a]["mean_onset"], "paired_sd_d_mean_onset": float(np.std(dmo, ddof=1)),
                      "d_slope_mean_onset": point[b]["slope_mean_onset"] - point[a]["slope_mean_onset"], "paired_sd_d_slope_mean_onset": float(np.std(dms, ddof=1)),
                      "d_PROBE_346": point[b]["PROBE_346"] - point[a]["PROBE_346"]}
    out["c_d_pairs"] = pairs
    out["c_draft_slope_delta"] = prev["margins"]["slope_h"]["delta"]
    fo = os.path.join(HERE, "rv_hc_margins.out.json")
    with open(fo, "w", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(json.dumps({k: out[k] for k in ("a_joint_C1KEPT_2seeds", "a_COEXIST_as_written", "a_COEXIST_with_VBC1_both_seeds", "b_slope_one_lineage_pass_given_harm")}, indent=1))
    for k, v in mdh.items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    for r, v in point.items():
        print(r, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    for lab, v in pairs.items():
        print(lab, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    print("wrote", fo)


if __name__ == "__main__":
    sys.exit(main())
