"""TS_v1 check C (T12, T2, RT m6 / m7, PROOFS P6): Plan B' cost and hours for the s1-only lineage, and the reuse-free pool sizes with >= 1 % spare.
Cloud (RTX 3090): T1 and T2 training only (+ per-host deploy, HOSTPROBE, G-T9 MEM, G-T10 TIME). Local RTX 3070, fp32: base re-eval, labelling (gates,
EQUIV, pools), every eval_text / eval_textself. Inputs (read only): the draft's ts_cost.json (training-time model, eval calibration), PROOFS pc_cost.json
(TF padding, MEMO-GAP-T generations), RT rt_reach.json (decode-step counts). Every throughput / price is [ASSUMED]; G-T10 / G-T18 decide.
python -B v1_cost.py"""
import json
import math
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "v1_cost.json"); R6 = os.path.dirname(HERE)
TC = json.load(open(os.path.join(R6, "scripts", "ts_cost.json")))
PC = json.load(open(os.path.join(R6, "proofs_ts", "pc_cost.json")))
RT = json.load(open(os.path.join(R6, "rt_ts", "rt_reach.json")))
PRICE = (0.17, 0.26)
CONSUMED = 383488                                                        # 5,992 self steps x 64 (PROOFS P4, seed-1 replay, exact)


def hyper_sd(N, K, m):
    p = K / N; return math.sqrt(m * p * (1 - p) * (N - m) / (N - 1))


def main():
    res = {}
    # pools (m7: >= 1 % spare)
    pools = {"T1": {"per_n": {n: 97000 for n in (9, 10, 11, 12)}}, "T2": {"per_n": {**{n: 65000 for n in (13, 14, 15, 16)}, **{n: 32500 for n in (9, 10, 11, 12)}}}}
    for k, p in pools.items():
        N = sum(p["per_n"].values()); p["total"] = N; p["spare_records"] = N - CONSUMED; p["spare_frac"] = (N - CONSUMED) / CONSUMED
        p["spare_self_steps"] = (N - CONSUMED) / 64
        p["consumed_per_n"] = {n: {"mean": CONSUMED * K / N, "sd": hyper_sd(N, K, CONSUMED)} for n, K in p["per_n"].items()}
        if k == "T2":
            p["new_window_share_mean"] = sum(K for n, K in p["per_n"].items() if n >= 13) / N
    res["pools"] = pools
    res["latent_reference"] = {"R1 pool 440,000 (110,000 per n)": 440000, "R2 pool 444,000 (74,000 / 37,000)": 444000,
                               "latent consumed per n R1": CONSUMED / 4, "latent consumed per new n R2": CONSUMED * 74000 / 444000}
    # cloud training (draft's steady-rate model; x1..x2 host variance, memory note 'host speed variance')
    t1, t2 = TC["train"]["T1_hours"], TC["train"]["T2_hours"]
    dep = (0.25, 0.35)                                                   # per host: deploy, HOSTPROBE, G-T9 (2 steps), G-T10 forced G-TIME, model / pool upload
    cells = {}
    for nm, th in (("s219_t1_s1", t1), ("s219_t2_s1", t2)):
        h = (th + dep[0], 2 * th + dep[1])
        cells[nm] = {"train_h_steady": th, "hours": h, "usd": (h[0] * PRICE[0], h[1] * PRICE[1])}
    egress = 0.10                                                        # off-host full-state mirrors (6 x ~1.5 GB per cell) + uploads [ASSUMED price]
    tot = (sum(c["usd"][0] for c in cells.values()), sum(c["usd"][1] for c in cells.values()) + egress)
    res["cloud"] = {"cells": cells, "egress_allowance": egress, "total_usd": tot, "price_band": PRICE,
                    "admission_usd": {"s219_t1_s1": 1.00, "s219_t2_s1": 1.45}, "guard_usd": {"s219_t1_s1": 1.20, "s219_t2_s1": 1.70},
                    "family_cap_usd": 3.00}
    res["cloud"]["admission_headroom_vs_projected_upper"] = {k: res["cloud"]["admission_usd"][k] / cells[k]["usd"][1] - 1 for k in cells}
    # local 3070 hours (fp32)
    ev = (PC["eval_text_hours_3070"][0], max(PC["eval_text_hours_3070"][1], 2.9))   # PROOFS / draft model .. RT's context term (2.2-2.9 h)
    pool1 = (PC["tf_positions_per_item"]["T1"][1] / PC["tf_positions_per_item"]["T1"][0]) * TC["label_hours"]["T1_10T"] * 388000 / 392000
    pool2 = (PC["tf_positions_per_item"]["T2"][1] / PC["tf_positions_per_item"]["T2"][0]) * TC["label_hours"]["T2_10T"] * 390000 / 392000
    memo1, memo2 = PC["memo_gap"]["T1"]["hours_3070_fp32"], PC["memo_gap"]["T2"]["hours_3070_fp32"]
    stages = {
        "G-T18 eval smoke (base, fresh rng 23195 n-24 shapes) + timing": (0.1, 0.2),
        "base eval_text (n 1-24, 24,000 items)": ev,
        "GATE-T1 (8k) + LABELLER-FULL + G-T16 EQUIV (REROOT + FULL generation, 250 per n)": (0.1, 0.2),
        "T1 pool (388k, TF-GREEDY, padded)": (0.8 * pool1, pool1),
        "T1 eval_text": ev,
        "T1 eval_textself GPU parts (MEMO-GAP-T 16k gens, LABELLER-KEPT-T 8k REROOT)": (memo1[0] + 0.05, memo1[1] + 0.1),
        "GATE-T2 + GATE-T1' (16k) + LABELLER-FULL + EQUIV": (0.15, 0.3),
        "T2 pool (390k, TF-GREEDY, padded)": (0.8 * pool2, pool2),
        "T2 eval_text": ev,
        "T2 eval_textself GPU parts (MEMO-GAP-T 32k gens, LABELLER-KEPT-T 8k)": (memo2[0] + 0.05, memo2[1] + 0.1),
    }
    loc = (sum(v[0] for v in stages.values()), sum(v[1] for v in stages.values()))
    res["local_3070"] = {"stages": stages, "total_hours": loc, "pool_hours_note": "pool1/pool2 = draft label_hours at 10 TFLOPS x PROOFS padding ratio x pool size; lower = 0.8 x"}
    # critical path (serial; T2 labelling only after T1's eval_text + the round-2 condition note)
    crit = {"base eval_text": ev, "GATE-T1 + T1 pool": (0.1 + 0.8 * pool1, 0.2 + pool1), "T1 cloud (deploy + train)": cells["s219_t1_s1"]["hours"],
            "T1 eval_text (+ CPU LABEL-AUDIT-T, contamination)": ev, "GATE-T2 + T2 pool": (0.15 + 0.8 * pool2, 0.3 + pool2),
            "T2 cloud": cells["s219_t2_s1"]["hours"], "T2 eval_text": ev, "T2 eval_textself": (memo2[0] + 0.05, memo2[1] + 0.1)}
    res["critical_path_hours"] = {"stages": crit, "total": (sum(v[0] for v in crit.values()), sum(v[1] for v in crit.values())),
                                  "note": "base eval_text can overlap GATE-T1 + pool only if a second local card is free; T1 eval_textself overlaps T2 cloud training"}
    # Plan A for reference (PROOFS' corrected per-lineage, one 3090 host)
    res["plan_A_reference"] = {"per_lineage_usd": PC["per_lineage_usd_revised"], "with_padding": (1.9, 5.6)}
    json.dump(res, open(OUT, "w", newline="\n"), indent=1)
    print(json.dumps({"pools": {k: {kk: vv for kk, vv in v.items() if kk != "consumed_per_n"} for k, v in pools.items()}}, indent=0))
    print("consumed per n", {k: {n: (round(x["mean"], 1), round(x["sd"], 2)) for n, x in v["consumed_per_n"].items()} for k, v in pools.items()})
    print("cloud", json.dumps(res["cloud"], indent=0))
    print("local stages", {k: (round(v[0], 2), round(v[1], 2)) for k, v in stages.items()}); print("local total", [round(x, 2) for x in loc])
    print("critical path", [round(x, 1) for x in res["critical_path_hours"]["total"]])


if __name__ == "__main__":
    main()
