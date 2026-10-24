"""REVIEW_LB check O ($0, CPU, pure Python): re-runs of lb_oc's LP3 OC where its applicability rule bites, and the expected spend under the review's
fixes. Imports lb_oc's exact-binomial helpers; reads lb_oc.json / lb_cost.json (draft files, unchanged).
Usage: python -B rv_lb_oc_spend.py [--out rv_lb_oc_spend.json]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); DRAFT = os.path.dirname(HERE)
sys.path.insert(0, DRAFT)
import lb_oc  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "rv_lb_oc_spend.json")); a = ap.parse_args(argv)
    N = 1000; res = {}
    # LP3 (LAT >= 0.70 and >= DIR + 0.30) in the band the draft marks NOT-APPLICABLE (DIR d = 2 > 0.40)
    res["LP3_dir_band"] = {"lat%.2f_dir%.2f" % (pl, pd): lb_oc.p_two_arm(N, pl, pd, 0.70, 0.30)
                           for pl in (0.90, 0.95, 0.99) for pd in (0.40, 0.50, 0.55, 0.60, 0.65)}
    # LP2 on d = 2 items only: a training-bin mean; with ~32 d = 2 items x ~500 d = 2 steps per 1k bin, binomial noise on the bin mean is
    # sd <= sqrt(0.25 / 16000) = 0.004 (items within a step are not independent of the model state, so this is the sampling floor only)
    res["LP2_d2_bin_sd_floor"] = (0.25 / 16000) ** 0.5
    sp = json.load(open(os.path.join(DRAFT, "lb_oc.json")))["spend_point_guard"]["chunk"]
    A1, A2, TX, B6, BUILD = (sp[k] for k in ("A1_pilot_d2", "A2_pilot_d4", "TEXT_22k", "B_stage1_d6_36k", "build"))
    C = json.load(open(os.path.join(DRAFT, "lb_cost.json")))["bases"]["qwen25_05b"]["cells"]
    lat_a1 = [C["pilot_d2_8k"]["inject_chunk"]["usd"], C["pilot_d2_8k"]["inject_chunk"]["usd_guard"]]
    tx30 = [x * 30 / 22 for x in TX]
    def exp(pA1, pA2, extra_a1=(0.0, 0.0), tx=TX):
        return [round(BUILD[i] + A1[i] + extra_a1[i] + pA1 * (A2[i] + tx[i] + pA2 * B6[i]), 2) for i in range(2)]
    res["expected"] = {
        "draft (pA1 0.23, pA2 0.5)": exp(0.23, 0.5),
        "applicability counted (P(N/A) 0.3 -> pA1 0.274)": exp(0.95 * 0.40 * (0.3 + 0.7 * 0.6), 0.5),
        "LP2 on d2 only, prior 0.25 (pA1 0.95 x 0.25 x 0.72)": exp(0.95 * 0.25 * 0.72, 0.5),
        "+ A1 lr-bracket LAT arm (fix 4), pA1 0.274": exp(0.274, 0.5, lat_a1),
        "+ lr bracket + TEXT to 30k, pA1 0.274": exp(0.274, 0.5, lat_a1, tx30)}
    res["staged_guard_with_fixes"] = round(BUILD[1] + A1[1] + lat_a1[1] + tx30[1] + A2[1] + B6[1], 2)
    res["staged_point_with_fixes"] = round(BUILD[0] + A1[0] + lat_a1[0] + tx30[0] + A2[0] + B6[0], 2)
    res["lat_a1_arm_point_guard"] = lat_a1; res["text30k_point_guard"] = [round(x, 2) for x in tx30]
    with open(a.out, "w", newline="\n") as f:
        json.dump(res, f, indent=1); f.write("\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
