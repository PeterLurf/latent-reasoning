"""THEORY §132 scoring (rules fixed in §132; written before any §132 run; Theory is the only scorer).
In-context transition tables on §120's FSQ carrier: per-problem random permutation tables, written as explicit pairs
(" a0>3 a1>5 ...") in every segment's prefix; the state only in the 8-d code; test on held-out tables.
Size ladder: rung 1 = k 4 states, m 2 ops (dirs s132_k4_s{seed}); rung 2 = k 6, m 3 (s132_k6_s{seed}), run only after a rung-1
PASS. Restart-on-plateau slots as §128: slot k in 1..4 tries seeds k, k+100, k+200, k+300; the accepted run is the slot's first
attempt whose 123-3 flag does not fire. eval.json: long_acc (cumulative 512 on held-out tables), ext_2048, donor (same-table),
donor_difftable, table_swap, ood_acc, cert_sample, probe_ophist (descriptive).
Per rung: PASS = >= 3/4 slots exact (>= 0.99) and SERIAL (same-table donor >= 0.90, recipient - chance < 0.10 at every position);
ENTRY-FAIL = >= 2 slots exhaust their attempts; MEMORISED (132-1) = >= 2 accepted runs with train-table accuracy >= 0.99 and
held-out < 0.5; NOT-LEARNED = >= 2 accepted runs below 0.99. Errors: args k / m not matching the rung, held-out overlap != 0. §132's verdict is rung 2's; a rung-1
failure stops the ladder and is the verdict ("stops at rung 1: ...").
Audit nits: smoke dirs (args steps != 24000 or sched_scale != 1 or cpu) are skipped and printed; a run before 2k is "running
(phase 1)"; an eval.json whose log has no 2k decision is an error row; accepted runs on more than one GPU name print a warning.
    python theory/score_s132.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
OFFS = (0, 100, 200, 300)
SKIPPED, ERRORS = [], []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def smoke(d):
    a = load(d, "args.json")
    if not a:
        return False
    why = [w for w, bad in (("steps %s" % a.get("steps"), int(a.get("steps", 24000)) != 24000),
                            ("sched_scale %s" % a.get("sched_scale"), float(a.get("sched_scale", 1.0)) != 1.0),
                            ("cpu", str(a.get("device", "")).startswith("cpu") or str(a.get("train_gpu", "")).lower() == "cpu")) if bad]
    if why:
        SKIPPED.append((d, why))
    return bool(why)


def log_rows(d):
    p = os.path.join(R, d, "log.jsonl")
    if os.path.exists(p):
        return [json.loads(x) for x in open(p)]
    p2 = os.path.join(R, d + ".void_partial", "log.jsonl")                  # a 2k plateau stop drained as .void_partial (127-2)
    if os.path.exists(p2):
        rows = [json.loads(x) for x in open(p2)]
        if any(str(x.get("stopped", "")).startswith("plateau") for x in rows):
            return rows
    return None


def plateau(rows):
    L = {x["step"]: x["loss"] for x in rows if "loss" in x and "step" in x}
    if not L or max(L) < 2000:
        return None
    g = lambda t: L[min(L, key=lambda s: abs(s - t))]  # noqa: E731
    return bool(g(2000) >= 0.9 * g(1000) and g(2000) >= 0.25 * g(100))


def serial(dn):
    return bool(dn) and all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] - v.get("chance_overlap", 0.0) < 0.10 for v in dn.values())


def slot(k, rung):
    att = []
    for o in OFFS:
        d = "s132_%s_s%d" % (rung, k + o)
        if smoke(d):
            break
        rows = log_rows(d); r = load(d)
        if rows is None and r is None:
            break                                                            # not started
        pl = plateau(rows) if rows else None
        if pl is None:
            if r is not None:
                ERRORS.append((d, "eval.json present but no 2k decision in log.jsonl")); return {"attempts": att, "status": "error", "accepted": None}
            return {"attempts": att, "status": "running (phase 1)", "accepted": None}
        att.append({"seed": k + o, "plateau": pl})
        if pl is False:
            if r is None:
                return {"attempts": att, "status": "running", "accepted": None}
            a_ = load(d, "args.json") or {}; kk, mm = {"k4": (4, 2), "k6": (6, 3)}[rung]
            bad = []
            if (a_.get("k"), a_.get("m")) != (kk, mm):
                bad.append("args k/m %s/%s != %s" % (a_.get("k"), a_.get("m"), rung))
            if r.get("heldout_overlap", 0):
                bad.append("held-out overlap %s != 0" % r.get("heldout_overlap"))
            if bad:                                                          # 132-1: not a valid run of this rung
                ERRORS.extend((d, b_) for b_ in bad); return {"attempts": att, "status": "error", "accepted": None}
            acc = {"seed": k + o, "long_acc": r.get("long_acc"), "train_acc": r.get("train_acc"), "heldout_n_tables": r.get("heldout_n_tables"), "serial": serial(r.get("donor")), "ext_2048": (r.get("ext_2048") or {}).get("cumulative"),
                   "donor_difftable": r.get("donor_difftable"), "table_swap": r.get("table_swap"), "ood_acc": r.get("ood_acc"),
                   "cert_sample": r.get("cert_sample"), "probe_ophist": r.get("probe_ophist"),
                   "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver")}
            acc["exact"] = (acc["long_acc"] or 0) >= 0.99; acc["pass"] = bool(acc["exact"] and acc["serial"])
            return {"attempts": att, "status": "accepted", "accepted": acc}
    if len(att) == len(OFFS) and all(a["plateau"] for a in att):
        return {"attempts": att, "status": "exhausted", "accepted": None}
    return {"attempts": att, "status": "open", "accepted": None}


def rung_verdict(rung):
    sl = {k: slot(k, rung) for k in (1, 2, 3, 4)}
    n_pass = sum(bool(v["accepted"] and v["accepted"]["pass"]) for v in sl.values())
    n_notlearned = sum(bool(v["accepted"] and not v["accepted"]["exact"]) for v in sl.values())
    n_exh = sum(v["status"] == "exhausted" for v in sl.values())
    n_mem = sum(bool(v["accepted"] and (v["accepted"]["train_acc"] or 0) >= 0.99 and (v["accepted"]["long_acc"] or 0) < 0.5) for v in sl.values())
    decided = sum(1 for v in sl.values() if v["status"] == "exhausted" or (v["status"] == "accepted" and v["accepted"] is not None))
    if n_pass >= 3:
        verdict = "PASS"
    elif n_exh >= 2:
        verdict = "ENTRY-FAIL"
    elif n_mem >= 2:
        verdict = "MEMORISED"                                                # 132-1: train-table exact, held-out < 0.5
    elif n_notlearned >= 2:
        verdict = "NOT-LEARNED"
    elif decided == 4:
        verdict = "descriptive (%d pass, %d not learned, %d exhausted)" % (n_pass, n_notlearned, n_exh)
    else:
        verdict = "incomplete (%d / 4 slots decided)" % decided
    gpus = {v["accepted"]["gpu"] for v in sl.values() if v["accepted"]}
    if len(gpus) > 1:
        print("WARNING (%s): accepted runs span GPU names %s (125-2)" % (rung, sorted(map(str, gpus))))
    for k, v in sl.items():
        a = v["accepted"]
        print(rung, "slot", k, v["status"], [(x["seed"], x["plateau"]) for x in v["attempts"]], a and {x: a[x] for x in ("seed", "long_acc", "train_acc", "heldout_n_tables", "serial", "ext_2048", "pass")})
    return sl, verdict


def main():
    s1, v1 = rung_verdict("k4")                                             # rung 1: k = 4 states, m = 2 ops
    s2, v2 = rung_verdict("k6")                                             # rung 2: k = 6, m = 3 (only after a rung-1 PASS)
    if v1 == "PASS":
        overall = v2 if not str(v2).startswith("incomplete") else "rung 1 PASS; rung 2 %s" % v2
    elif str(v1).startswith("incomplete"):
        overall = "rung 1 %s" % v1
    else:
        overall = "stops at rung 1: %s" % v1
    if SKIPPED:
        print("SKIPPED (smoke, not scored):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    print("RUNG 1 (k4)", v1, "| RUNG 2 (k6)", v2, "| §132", overall)
    json.dump({"rung1": {"slots": s1, "verdict": v1}, "rung2": {"slots": s2, "verdict": v2}, "verdict": overall,
               "skipped_smoke": SKIPPED, "errors": ERRORS},
              open(os.path.join(ROOT, "theory", "results", "score_s132.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
