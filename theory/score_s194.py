"""THEORY §194 scoring (§190-S2, a descriptive replication; criteria fixed in §194; written before any s2 run; Theory is the only scorer).
s190_rmsnopass_s2 = s190_rmsnopass_s1's args with seed 2. REPLICATES (descriptive, 2 seeds; never a claim) needs all of:
  (i)   187-3 rows: mean handle t 3-6 at n 12 >= its n 9 value - 0.05 (nesting <= 2);
  (ii)  187-3 rows: n 10 has no onset (handle < 0.80) before t = 9;
  (iii) eval_s182_auxdone.json: n 10 median halt >= 9.5 and frac_capped < 0.5.
Reported beside: basin entry (theory/basin_entry.py) and final loss; §190's registered labels applied to s2 vs RMS s2 (score_s190).
    python theory/score_s194.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
D, EPS = "s190_rmsnopass_s2", 1e-9


def rows_table(d):
    p = os.path.join(R, d, "eval_s187_deepaux_items_n9_12.json")
    if not os.path.exists(p):
        return None
    rows = json.load(open(p))["items"]; out = {}
    for n in (9, 10, 11, 12):
        it = [r for r in rows if r["n"] == n and r["nest_depth"] <= 2]; T = max(int(t) for r in it for t in r["handle_by_t"])
        h = [sum(r["handle_by_t"][str(t)] for r in it if str(t) in r["handle_by_t"]) / max(1, sum(str(t) in r["handle_by_t"] for r in it)) for t in range(1, T + 1)]
        out[n] = {"mean_t3_6": sum(h[2:6]) / 4, "onset": next((t for t, v in enumerate(h, 1) if v < 0.80), T + 1)}
    return out


def main():
    res = {}; tb = rows_table(D); ap = os.path.join(R, D, "eval_s182_auxdone.json")
    ad = (json.load(open(ap)).get("auxdone") or json.load(open(ap)))["by_n"].get("10") if os.path.exists(ap) else None
    if tb is None or ad is None:
        rd = "pending (rows %s, auxdone %s)" % ("in" if tb else "missing", "in" if ad else "missing")
    else:
        c1 = tb[12]["mean_t3_6"] >= tb[9]["mean_t3_6"] - 0.05 - EPS; c2 = tb[10]["onset"] >= 9; c3 = ad["median_halt"] >= 9.5 - EPS and ad["frac_capped"] < 0.5
        rep = c1 and c2 and c3
        rd = "%s (descriptive, 2 seeds; not a claim): (i) n12 t3-6 %.3f vs n9 %.3f %s; (ii) n10 onset %d %s; (iii) AUX-DONE n10 median %.1f, capped %.2f %s" % (
            "REPLICATES" if rep else "DOES NOT REPLICATE", tb[12]["mean_t3_6"], tb[9]["mean_t3_6"], c1, tb[10]["onset"], c2, ad["median_halt"], ad["frac_capped"], c3)
        res.update(rows=tb, auxdone_n10=ad, replicates=rep)
    try:
        import basin_entry as BE
        if os.path.exists(os.path.join(R, D, "log.jsonl")):
            res["entry"] = BE.summary(D); rd += "; entry(0.5/0.1) %s/%s, last bin %.4f" % (res["entry"]["entry_0.5"], res["entry"]["entry_0.1"], res["entry"]["last_bin"])
    except Exception as ex:  # noqa: BLE001
        rd += "; entry unavailable (%s)" % ex
    res["reading"] = rd; print("READING:", rd)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s194.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
