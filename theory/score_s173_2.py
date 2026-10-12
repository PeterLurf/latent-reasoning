"""THEORY note 173-2 / 173-2a scoring (descriptive beside §173's rules; written before any of the three halt files exist; Theory is the only scorer).
§173 CONV-HALT (answer2, cap 2K) on s184_reweight_s1 / s2 and s182_window_s1, from <run>/eval_s173_halt.json (boxes/halt173).
At n = 10 and 12 (173-2a: halt173's n set is fixed), comparator = the same file's acc_by_t[n][n + 2] (same items):
  SIZE-FREE HALT(model) = at both n, |acc at halt - acc_by_t[n][n+2]| <= 0.03 and mean halt >= n - 1;
  PREMATURE(model)      = at either n, acc at halt <= acc_by_t[n][n+2] - 0.05;  else mixed.
  SIZE-FREE HALT (the 173-2 reading) needs both §184 seeds; WINDOW s1 is read beside them. A halt more than 0.03 ABOVE K'=n+2 is flagged, not counted.
Note 173-3: rule "moved2" (settled AND a(t) != a(0)) is read the same way, the reading on held-out s184 s2 only (s1 = design data, WINDOW beside);
plus n 2..8 vs fixed K (HALT-EQUAL form: acc >= fixed - 0.01, mean h < K). Descriptive: mean halt and frac_capped by n (2..16) against n; n = 16 has no comparator for §184 (K' 18 > cap 16). WINDOW's think_k is 12, so halt173
gives it cap 24 (2K) - a larger halting budget than §184's; reported beside, not in the reading.
    python theory/score_s173_2.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
RUNS = {"s184_s1": "s184_reweight_s1", "s184_s2": "s184_reweight_s2", "window_s1": "s182_window_s1", "rms_s1": "s185_rms_s1"}   # 173-4: rms_s1 descriptive (beside)
KEXP = {"s184_reweight_s1": 8, "s184_reweight_s2": 8, "s182_window_s1": 12, "s185_rms_s1": 8}   # each run's think_k (args.json); halt173 cap = 2K
READ_N, TOL, PRE, EPS = (10, 12), 0.03, 0.05, 1e-9


def model(d, RULE="answer2"):
    p = os.path.join(R, d, "eval_s173_halt.json")
    if not os.path.exists(p):
        return {"status": "open"}
    x = json.load(open(p))
    if RULE not in x["halt"]:
        return {"status": "no %s in file" % RULE}
    h, bt, cap = x["halt"][RULE], x["acc_by_t"], x["cap"]
    assert int(cap) == 2 * int(x["think_k"]) and int(x["think_k"]) == KEXP.get(d, int(x["think_k"])) and x["kind"] == "nested", (d, x["think_k"], cap, x["kind"])
    rows = {}
    for s in sorted(h, key=int):
        n = int(s); r = {"acc_halt": h[s]["acc"], "mean_h": h[s]["mean_h"], "frac_capped": h[s]["frac_capped"]}
        fc = r["frac_capped"]; r["mean_h_uncapped"] = None if fc >= 1 else (r["mean_h"] - fc * cap) / (1 - fc)   # 173-3a: capped items have h = cap
        assert len(bt[s]) == cap + 1, (d, s, len(bt[s]))
        if n + 2 <= cap:
            r["acc_k_n2"] = bt[s][n + 2]; r["diff"] = r["acc_halt"] - r["acc_k_n2"]
        if n <= 8:
            r["fixed_k"] = x["halt_fixed_k"][s]; r["halt_equal"] = bool(r["acc_halt"] >= r["fixed_k"] - 0.01 - EPS and r["mean_h"] < int(x["think_k"]))
        rows[n] = r
    rd = [rows[n] for n in READ_N]
    free = all(abs(r["diff"]) <= TOL + EPS and r["mean_h"] >= n - 1 - EPS for n, r in zip(READ_N, rd))
    pre = any(r["diff"] <= -PRE + EPS for r in rd)
    above = [n for n, r in zip(READ_N, rd) if r["diff"] > TOL + EPS]
    lab = "SIZE-FREE HALT" if free else "PREMATURE" if pre else "mixed"
    return {"status": "done", "label": lab, "above_fixed": above, "by_n": rows, "cap": int(cap), "gpu": x.get("gpu"), "n_items": x.get("n")}


def line(k, m):
    if m["status"] != "done":
        return "%s %s" % (k, m["status"])
    return "%s %s: " % (k, m["label"]) + "; ".join("n %d halt %.3f vs K'=n+2 %.3f (%+.3f), mean h %.2f, capped %.3f" % (
        n, m["by_n"][n]["acc_halt"], m["by_n"][n]["acc_k_n2"], m["by_n"][n]["diff"], m["by_n"][n]["mean_h"], m["by_n"][n]["frac_capped"]) + (
        " (uncapped mean h %.2f%s)" % (m["by_n"][n]["mean_h_uncapped"], "; below n - 1: caveat" if m["by_n"][n]["mean_h_uncapped"] < n - 1 else "") if m["by_n"][n]["mean_h_uncapped"] is not None else "") for n in READ_N) + (
        "; ABOVE fixed K' at n %s" % m["above_fixed"] if m["above_fixed"] else "")


def main():
    res = {}
    for rule in ("answer2", "moved2"):
        rr = {k: model(d, rule) for k, d in RUNS.items()}; a, b = rr["s184_s1"], rr["s184_s2"]
        if rule == "answer2":
            if a["status"] == "done" and b["status"] == "done":
                rd = "SIZE-FREE HALT (both §184 seeds)" if a["label"] == b["label"] == "SIZE-FREE HALT" else                      "PREMATURE (%s)" % ", ".join(k for k in ("s184_s1", "s184_s2") if rr[k]["label"] == "PREMATURE") if "PREMATURE" in (a["label"], b["label"]) else "mixed"
            elif "PREMATURE" in (a.get("label"), b.get("label")):
                rd = "PREMATURE (decided by %s; other seed %s)" % ("s184_s1" if a.get("label") == "PREMATURE" else "s184_s2", b["status"] if a.get("label") == "PREMATURE" else a["status"])
            else:
                rd = "pending (s1 %s; s2 %s)" % (a["status"], b["status"])
        else:                                                               # note 173-3: held-out s184 s2 only
            rd = b["label"] + " (held-out s184 s2)" if b["status"] == "done" else "pending (s184 s2 %s)" % b["status"]
        rr["reading"] = rd; res[rule] = rr; print("READING %s:" % rule, rd)
        for k, m in rr.items():
            if isinstance(m, dict) and "status" in m:
                print("  ", line(k, m) + (" [design data]" if rule == "moved2" and k == "s184_s1" else ""))
                if m["status"] == "done":
                    print("     mean halt by n:", {n: round(r["mean_h"], 2) for n, r in m["by_n"].items()}, "| n 2-8 halt-equal:", {n: r["halt_equal"] for n, r in m["by_n"].items() if n <= 8})
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s173_2.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
