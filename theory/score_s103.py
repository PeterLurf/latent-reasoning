"""THEORY §103 scoring (rules fixed in §103; written before any §103 implementation or run; Theory is the only scorer).
Runs: boxes/runs_s5/s103_s{1,2,3}/eval.json (gen97 GEN-4 trained by filtered self-training by length; keys as gen97 plus
short_fraction, rounds = [{length, filtered_by {validity, final, vote}, kept_share, acc_new_length_before,
pseudo_label_precision}]).
SELF: cumulative >= 0.99 at 512 in >= 2/3 seeds AND zero-removal acc after the first ablated slot <= 0.10 in those seeds;
every run must have short_fraction <= 0.05 (asserted). Pseudo-label precision is descriptive (truth used by the scorer only).
    python theory/score_s103.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    rows = {}
    for s in (1, 2, 3):
        f = os.path.join(ROOT, "boxes", "runs_s5", "s103_s%d" % s, "eval.json")
        if os.path.exists(f):
            r = json.load(open(f))
            assert r["short_fraction"] <= 0.05 + 1e-9, (f, r["short_fraction"])
            rows[s] = {"cum512": r["cumulative"]["512"], "ext2048": r.get("extension_2048", {}).get("cumulative", {}).get("2048"),
                       "zero_after": r["removal"]["zero"]["acc_after_first_ablated_slot"],
                       "mean_after": r["removal"]["mean"]["acc_after_first_ablated_slot"],
                       "per_slot": r.get("chunk_check", {}).get("per_slot_acc"), "edit_iia": (r.get("edit_iia") or {}).get("iia"),
                       "short_fraction": r["short_fraction"], "rounds": r.get("rounds")}
    passing = [s for s, v in rows.items() if v["cum512"] >= 0.99 and v["zero_after"] <= 0.10]
    res = {"n": len(rows), "pass_seeds": passing, "SELF": len(passing) >= 2, "runs": rows}
    for s, v in rows.items():
        print("s%d cum512 %.4f ext2048 %s zero %s mean %s short %.3f | rounds %s" % (
            s, v["cum512"], v["ext2048"], v["zero_after"], v["mean_after"], v["short_fraction"],
            [(x.get("length"), x.get("kept_share"), x.get("pseudo_label_precision")) for x in (v["rounds"] or [])]))
    print("SELF", res["SELF"], "pass seeds", passing)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s103.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
