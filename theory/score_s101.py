"""THEORY §101 scoring (rules fixed in §101; written before any §101 implementation or run; Theory is the only scorer).
Runs: boxes/runs_s5/s101_b{1,0.1,0.01,0}_s{1,2,3}/eval.json (HYB-4 with a soft snap at beta_min; keys as latent86/gen97 plus
beta_min, acc_vs_length, drift, u_code_norm_ratio).
Per arm: PASS iff cumulative >= 0.99 at 512 in >= 2/3 seeds AND zero-removal acc after the first ablated slot <= 0.10 in
those seeds (§92's rule). Primary reading: the smallest beta_min that passes.
    python theory/score_s101.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BETAS = ("1", "0.1", "0.01", "0")


def main():
    res = {"arms": {}}
    for b in BETAS:
        rows = {}
        for s in (1, 2, 3):
            f = os.path.join(ROOT, "boxes", "runs_s5", "s101_b%s_s%d" % (b, s), "eval.json")
            if os.path.exists(f):
                r = json.load(open(f))
                rows[s] = {"cum512": r["cumulative"]["512"], "ext2048": r.get("extension_2048", {}).get("cumulative", {}).get("2048"),
                           "zero_after": r["removal"]["zero"]["acc_after_first_ablated_slot"],
                           "mean_after": r["removal"]["mean"]["acc_after_first_ablated_slot"],
                           "acc_vs_length": r.get("acc_vs_length"), "u_code_norm_ratio": r.get("u_code_norm_ratio"),
                           "drift": r.get("drift")}
        passing = [s for s, v in rows.items() if v["cum512"] >= 0.99 and v["zero_after"] <= 0.10]
        res["arms"][b] = {"n": len(rows), "pass_seeds": passing, "pass": len(passing) >= 2, "runs": rows}
        print("beta_min %-5s n %d pass %s %s | cum512 %s | ext2048 %s | zero_after %s" % (
            b, len(rows), passing, len(passing) >= 2, {s: round(v["cum512"], 4) for s, v in rows.items()},
            {s: v["ext2048"] for s, v in rows.items()}, {s: v["zero_after"] for s, v in rows.items()}))
    passed = [float(b) for b in BETAS if res["arms"][b]["pass"]]
    res["smallest_passing_beta_min"] = min(passed) if passed else None
    print("smallest passing beta_min:", res["smallest_passing_beta_min"])
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s101.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
