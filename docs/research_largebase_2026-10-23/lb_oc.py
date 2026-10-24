"""LB draft check D ($0, CPU): operating characteristics of the LB gates and the expected spend tree (LB_draft.md §8).

Exact binomial tails (no normal approximation). Two-arm gates (LAT >= thr and LAT - DIRECT >= margin) use independent binomials at N items
per arm per depth (TEST2: 1,000 per d). Spend per branch from lb_cost.json (3090 point estimates; guard x 1.3 shown separately).
Usage: python -B lb_oc.py [--cost lb_cost.json] [--out lb_oc.json]
"""
import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def pmf(n, p):
    lp, lq = (math.log(p) if p > 0 else -1e300), (math.log(1 - p) if p < 1 else -1e300)
    return [math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * lp + (n - k) * lq) for k in range(n + 1)]


def p_two_arm(n, p_lat, p_dir, thr, margin):
    """P(acc_LAT >= thr and acc_LAT - acc_DIR >= margin), N items each."""
    fl, fd = pmf(n, p_lat), pmf(n, p_dir); cd = [0.0] * (n + 1); s = 0.0
    for k in range(n + 1):
        s += fd[k]; cd[k] = s
    tot = 0.0
    for k in range(n + 1):
        if k / n + 1e-12 < thr:
            continue
        kd_max = math.floor((k / n - margin) * n + 1e-9)
        if kd_max >= 0:
            tot += fl[k] * cd[min(kd_max, n)]
    return tot


def p_one(n, p, thr):
    return sum(pr for k, pr in enumerate(pmf(n, p)) if k / n + 1e-12 >= thr)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--cost", default=os.path.join(HERE, "lb_cost.json")); ap.add_argument("--out", default=os.path.join(HERE, "lb_oc.json"))
    a = ap.parse_args(argv); C = json.load(open(a.cost))["bases"]["qwen25_05b"]["cells"]; N = 1000; res = {"N_per_d": N}

    # LP3 (binding, pilot A1): TEST2 d = 2 LAT >= 0.70 and >= DIRECT + 0.30
    res["LP3"] = {s: p_two_arm(N, pl, pd, 0.70, 0.30) for s, (pl, pd) in {
        "null_chance (0.091, 0.091)": (1 / 11, 1 / 11), "dir_partly_learns (0.40, 0.30)": (0.40, 0.30), "lat_0.68_dir_0.09": (0.68, 1 / 11),
        "lat_0.70_dir_0.09": (0.70, 1 / 11), "lat_0.72_dir_0.09": (0.72, 1 / 11), "lat_0.90_dir_0.09": (0.90, 1 / 11), "lat_0.90_dir_0.62": (0.90, 0.62)}.items()}
    # LP4 (pilot A2): d 3 >= 0.60 and d 4 >= 0.50, each >= DIRECT + 0.25 (independent depths)
    def lp4(p3, p4, d3, d4):
        return p_two_arm(N, p3, d3, 0.60, 0.25) * p_two_arm(N, p4, d4, 0.50, 0.25)
    res["LP4"] = {"null_chance": lp4(1 / 11, 1 / 11, 1 / 11, 1 / 11), "working_0.85_0.75": lp4(0.85, 0.75, 0.10, 0.10), "marginal_0.62_0.52": lp4(0.62, 0.52, 0.10, 0.10),
                  "shallow_0.80_0.30": lp4(0.80, 0.30, 0.10, 0.10)}
    # PB (L0 BIND-PROBE): Qwen OPD test acc >= 0.80 and >= GPT-2 small + 0.15, N_TEST 1,000 each
    res["PB"] = {s: p_two_arm(N, pq, pg, 0.80, 0.15) for s, (pq, pg) in {"equal_0.60": (0.60, 0.60), "equal_0.95": (0.95, 0.95), "qwen_0.85_gpt2_0.65": (0.85, 0.65),
                                                                          "qwen_0.82_gpt2_0.66": (0.82, 0.66), "qwen_0.95_gpt2_0.50": (0.95, 0.50)}.items()}
    # TEXT-SERIAL on Qwen (§227 rule): d 2 >= 0.80 and d 3 >= 0.70, 200 items per d
    res["TEXT"] = {s: p_one(200, p2, 0.80) * p_one(200, p3, 0.70) for s, (p2, p3) in {"like_s227 (1.0, 1.0)": (0.999, 0.999), "weak (0.75, 0.65)": (0.75, 0.65),
                                                                                      "dir_like (0.2, 0.1)": (0.2, 0.1)}.items()}

    # spend tree (chunk variant = CHUNK_DELTA adopted; plain = no chunk, ckpt where cheaper); point and guard
    def cell(tag, key):
        r = C[tag][key]; return r["usd"], r["usd_guard"]
    var = {}
    for name, keys in {"chunk": ("inject_chunk", "inject_chunk", "inject_chunk", "inject_chunk"), "plain": ("inject", "inject_ckpt", "inject_ckpt", "inject_ckpt")}.items():
        A1 = [x + y for x, y in zip(cell("pilot_d2_8k", keys[0]), cell("pilot_d2_8k", "direct"))]
        A2 = [x + y for x, y in zip(cell("pilot_d4_16k", keys[1]), cell("pilot_d4_16k", "direct"))]
        TX = [C["text_s227_18k"]["usd"] * 22 / 18, C["text_s227_18k"]["usd_guard"] * 22 / 18]           # TEXT run to 22k (draft §6)
        B6 = [x + y for x, y in zip(cell("stage1_d6_36k", keys[2]), cell("stage1_d6_36k", "direct"))]
        B8 = [x + y for x, y in zip(cell("stage1_d8_48k", keys[3]), cell("stage1_d8_48k", "direct"))]
        B5 = [x + y for x, y in zip(cell("stage1_d5_30k", keys[3]), cell("stage1_d5_30k", "direct"))]
        BUILD = [0.15, 0.20]                                                                              # cloud F2 timing + chunk trajectory gate
        full = [BUILD[i] + A1[i] + TX[i] + A2[i] + B8[i] for i in range(2)]
        staged6 = [BUILD[i] + A1[i] + TX[i] + A2[i] + B6[i] for i in range(2)]
        staged5 = [BUILD[i] + A1[i] + TX[i] + A2[i] + B5[i] for i in range(2)]
        # priors [ASSUMED, draft §8]: P(A1 pass) 0.23, P(A2 pass | A1) 0.5
        pA1, pA2 = 0.23, 0.5
        exp6 = [BUILD[i] + A1[i] + pA1 * (A2[i] + TX[i] + pA2 * B6[i]) for i in range(2)]               # TEXT runs only after an A1 pass (draft §6)
        var[name] = {"A1_pilot_d2": A1, "TEXT_22k": TX, "A2_pilot_d4": A2, "B_stage1_d6_36k": B6, "B_stage1_d8_48k": B8, "B_stage1_d5_30k": B5, "build": BUILD,
                     "full_design_d8": full, "staged_d6": staged6, "staged_d5": staged5, "expected_staged_d6": exp6}
    res["spend_point_guard"] = {k: {kk: [round(x, 2) for x in vv] for kk, vv in v.items()} for k, v in var.items()}
    with open(a.out, "w", newline="\n") as f:
        json.dump(res, f, indent=1); f.write("\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
