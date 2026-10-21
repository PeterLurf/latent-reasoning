"""R3 draft $0 check 2 (CPU, read-only on committed log.jsonl + draws_log.json; writes only r3_logsplit.json beside this script).
Which batch type puts gradient into SLVA's addressing? Every log_every-th step logs slva_gn (pre-clip grad norm of slva_* params), gn_preclip
(all params) and SLVA's off-newest mass on changed / copy slots (boxes/slva.mass_stats, on that step's own batch). draws_log.json gives the
coin per step (True = self batch), so each logged row is typed self / plain. Reported per run and per third of training (1-4k, 4-8k, 8-12k):
median slva_gn, median gn_preclip, median slva_gn / gn_preclip, mean off-newest chg / copy, by batch type.
FT-CTRL has no pool (every row plain). Usage: python -B r3_logsplit.py."""
import json
import os
import statistics as S

R = "D:/Latent/boxes/runs_fsq"
RUNS = ["s215_ftctrl_s1", "s215_ftself_s1", "s216_r2_s1", "s216_r1_s2"]
out = {"def": __doc__, "runs": {}}
for r in RUNS:
    rows = [json.loads(l) for l in open(os.path.join(R, r, "log.jsonl")) if l.strip()]
    rows = [x for x in rows if "slva_gn" in x and "step" in x]
    dl_p = os.path.join(R, r, "draws_log.json"); coin = {}
    if os.path.exists(dl_p):
        coin = {d["step"]: bool(d.get("coin")) for d in json.load(open(dl_p))}
    res = {}
    for lo, hi in ((1, 4000), (4001, 8000), (8001, 12000)):
        for typ in ("self", "plain"):
            xs = [x for x in rows if lo <= x["step"] <= hi and (coin.get(x["step"], False) == (typ == "self"))]
            if not xs:
                continue
            res["%d-%d_%s" % (lo, hi, typ)] = {
                "n_rows": len(xs), "slva_gn_med": S.median(x["slva_gn"] for x in xs), "gn_preclip_med": S.median(x["gn_preclip"] for x in xs),
                "slva_share_med": S.median(x["slva_gn"] / x["gn_preclip"] for x in xs if x["gn_preclip"] > 0),
                "offnewest_chg": S.mean(x["slva_offnewest_chg"] for x in xs if x["slva_offnewest_chg"] is not None),
                "offnewest_copy": S.mean(x["slva_offnewest_copy"] for x in xs if x["slva_offnewest_copy"] is not None)}
    out["runs"][r] = res
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "r3_logsplit.json"), "w", newline="\n"), indent=1)
for r, d in out["runs"].items():
    print("==", r)
    for k, v in d.items():
        print("  %-18s n %3d  slva_gn %.2e  gn %.2e  share %.3f  offnew chg %.3f copy %.3f" % (
            k, v["n_rows"], v["slva_gn_med"], v["gn_preclip_med"], v["slva_share_med"], v["offnewest_chg"], v["offnewest_copy"]))
