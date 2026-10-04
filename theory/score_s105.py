"""THEORY §105 scoring (rules fixed in §105; written before any §105 implementation or run; Theory is the only scorer).
Runs: boxes/runs_s5/g105_mask4_s{1..5}/eval.json (gen97 GEN-4 with the masked fork; gen97's keys).
Donor protocol: theory/results/s104_serial_s105.json (theory/s104_serial.py run on these models; per model "patch").
G4-M: cumulative >= 0.99 at 512 in >= 2/3 of seeds 1-3 with zero-removal <= 0.10 in those seeds.
MASK-DEC: in those seeds, mean-removal acc after the first ablated slot <= 0.06.
HANDOFF: USE(j) (single-patch donor-consistent >= 0.90) for j = 1..3 with no BYPASS (recipient-consistent >= 0.10) in
>= 2/3 models.  Unqualified C1 = G4-M + MASK-DEC + HANDOFF.
    python theory/score_s105.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    runs = {}
    for s in range(1, 6):
        f = os.path.join(ROOT, "boxes", "runs_s5", "g105_mask4_s%d" % s, "eval.json")
        if os.path.exists(f):
            r = json.load(open(f))
            runs[s] = {"cum512": r["cumulative"]["512"], "ext2048": r.get("extension_2048", {}).get("cumulative", {}).get("2048"),
                       "zero_after": r["removal"]["zero"]["acc_after_first_ablated_slot"],
                       "mean_after": r["removal"]["mean"]["acc_after_first_ablated_slot"]}
    g4 = [s for s, v in runs.items() if s <= 3 and v["cum512"] >= 0.99 and v["zero_after"] <= 0.10]
    res = {"runs": runs, "G4_M_seeds": g4, "G4_M": len(g4) >= 2}
    res["MASK_DEC"] = res["G4_M"] and all(runs[s]["mean_after"] <= 0.06 for s in g4)
    f = os.path.join(ROOT, "theory", "results", "s104_serial_s105.json")
    if os.path.exists(f):
        d = json.load(open(f)); ok = 0
        for k, r in d.items():
            p = r["patch"]
            ok += all(p["slot%d_single" % j]["donor_consistent"] >= 0.90 and p["slot%d_single" % j]["recipient_consistent"] < 0.10 for j in (1, 2, 3))
        res["HANDOFF_models"] = ok; res["HANDOFF"] = ok >= 2 and len(d) >= 3
    else:
        res["HANDOFF"] = None
    res["C1_unqualified"] = bool(res["G4_M"] and res["MASK_DEC"] and res["HANDOFF"])
    print({k: v for k, v in res.items() if k != "runs"}, "| runs", runs)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s105.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
