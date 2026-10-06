"""THEORY §132 scoring (rules fixed in §132; written before any §132 run; Theory is the only scorer).
In-context transition tables on §120's FSQ carrier: per-problem random permutation tables in every segment's prefix, the state
only in the 8-d code; test on held-out tables. Restart-on-plateau slots as §128: slot k in 1..4 tries seeds k, k+100, k+200,
k+300; the slot's accepted run is its first attempt whose 123-3 flag does not fire. Pilot = slot 1; slots 2-4 only if it enters.
Runs: boxes/runs_fsq/s132_tab_s{seed} (eval.json: long_acc = cumulative 512 on held-out tables, ext_2048, donor (same-table),
donor_difftable, table_swap, cert_sample; log.jsonl).
PASS: >= 3/4 slots whose accepted run is exact (long_acc >= 0.99) and SERIAL (same-table donor: donor_consistent >= 0.90 and
recipient_consistent - chance_overlap < 0.10 at every position). NOT-LEARNED: accepted runs below 0.99 in >= 2 slots.
ENTRY-FAIL: >= 2 slots exhaust their four attempts. Descriptive: different-table donor, table swap, sampled CERT, cost.
    python theory/score_s132.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
OFFS = (0, 100, 200, 300)


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def plateau(d):
    p = os.path.join(R, d, "log.jsonl")
    if not os.path.exists(p):
        p2 = os.path.join(R, d + ".void_partial", "log.jsonl")
        if not os.path.exists(p2):
            return None
        rows = [json.loads(x) for x in open(p2)]
        if not any(str(x.get("stopped", "")).startswith("plateau") for x in rows):
            return None
    else:
        rows = [json.loads(x) for x in open(p)]
    L = {x["step"]: x["loss"] for x in rows if "loss" in x and "step" in x}
    if not L or max(L) < 2000:
        return None
    g = lambda t: L[min(L, key=lambda s: abs(s - t))]  # noqa: E731
    return bool(g(2000) >= 0.9 * g(1000) and g(2000) >= 0.25 * g(100))


def serial(dn):
    return bool(dn) and all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] - v.get("chance_overlap", 0.0) < 0.10 for v in dn.values())


def slot(k):
    att = []
    for o in OFFS:
        d = "s132_tab_s%d" % (k + o); pl = plateau(d)
        if pl is None and load(d) is None:
            break
        att.append({"seed": k + o, "plateau": pl})
        if pl is False:
            r = load(d)
            if r is None:
                return {"attempts": att, "status": "running", "accepted": None}
            acc = {"seed": k + o, "long_acc": r.get("long_acc"), "serial": serial(r.get("donor")), "ext_2048": (r.get("ext_2048") or {}).get("cumulative"),
                   "donor_difftable": r.get("donor_difftable"), "table_swap": r.get("table_swap"), "cert_sample": r.get("cert_sample"),
                   "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver")}
            acc["exact"] = (acc["long_acc"] or 0) >= 0.99; acc["pass"] = bool(acc["exact"] and acc["serial"])
            return {"attempts": att, "status": "accepted", "accepted": acc}
    if len(att) == len(OFFS) and all(a["plateau"] for a in att):
        return {"attempts": att, "status": "exhausted", "accepted": None}
    return {"attempts": att, "status": "open", "accepted": None}


def main():
    sl = {k: slot(k) for k in (1, 2, 3, 4)}
    n_pass = sum(bool(v["accepted"] and v["accepted"]["pass"]) for v in sl.values())
    n_notlearned = sum(bool(v["accepted"] and not v["accepted"]["exact"]) for v in sl.values())
    n_exh = sum(v["status"] == "exhausted" for v in sl.values())
    decided = sum(v["status"] in ("accepted", "exhausted") and (v["status"] == "exhausted" or v["accepted"] is not None) for v in sl.values())
    if n_pass >= 3:
        verdict = "PASS"
    elif n_exh >= 2:
        verdict = "ENTRY-FAIL"
    elif n_notlearned >= 2:
        verdict = "NOT-LEARNED"
    elif decided == 4:
        verdict = "descriptive (%d pass, %d not learned, %d exhausted)" % (n_pass, n_notlearned, n_exh)
    else:
        verdict = "incomplete (%d / 4 slots decided)" % decided
    for k, v in sl.items():
        a = v["accepted"]
        print("slot", k, v["status"], [(x["seed"], x["plateau"]) for x in v["attempts"]], a and {x: a[x] for x in ("seed", "long_acc", "serial", "ext_2048", "pass")})
    print("VERDICT", verdict)
    json.dump({"slots": sl, "verdict": verdict}, open(os.path.join(ROOT, "theory", "results", "score_s132.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
