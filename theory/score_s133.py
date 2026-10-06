"""THEORY §133 scoring (rules fixed in §133; written before any §133 run; Theory is the only scorer).
Wide-carrier error correction on §128's S5 recipe (fsq120 FINAL, restart-on-plateau slots k, k+100, k+200, k+300; RTX 3090 cloud,
checkpoints). Arms (carrier width d = 64): cont64, fsq64, soft64, noise64, pin64 (+ jac64 if run).
Stage 1 dirs: boxes/runs_fsq/s133_{arm}_s{seed}; eval.json long_acc (512), ext_2048.cumulative, donor; eval_contraction_fp32.json.
PASS(arm): >= 3/4 slots whose accepted run is exact at 2,048 (ext_2048 >= 0.99), SERIAL (donor >= 0.90, recipient - chance < 0.10
at every position) and contractive (rho_1 <= 0.28, as 127-12's clause: rho_ff if defined, else the first-hand-off ratio).
DROP (no follow-up): the proof slot (slot 1) is accepted but not exact at 2,048 -> the arm stops ("dropped at proof").
Stage 2 (capacity; passing arms + the FSQ-8 control): dirs boxes/runs_fsq/s133cap_{arm}_s{seed} on S5x4; CAP-PASS = long_acc >= 0.99
at 512 in >= 3/4 slots.
    python theory/score_s133.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
ARMS = ("cont64", "fsq64", "soft64", "noise64", "pin64", "jac64")
OFFS = (0, 100, 200, 300)
SKIPPED = []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def smoke(d):
    a = load(d, "args.json")
    if not a:
        return False
    bad = int(a.get("steps", 24000)) != 24000 or float(a.get("sched_scale", 1.0)) != 1.0 or str(a.get("device", "")).startswith("cpu")
    if bad:
        SKIPPED.append(d)
    return bad


def plateau(d):
    p = os.path.join(R, d, "log.jsonl")
    if not os.path.exists(p):
        return None
    L = {x["step"]: x["loss"] for x in map(json.loads, open(p)) if "loss" in x and "step" in x}
    if not L or max(L) < 2000:
        return None
    g = lambda t: L[min(L, key=lambda s: abs(s - t))]  # noqa: E731
    return bool(g(2000) >= 0.9 * g(1000) and g(2000) >= 0.25 * g(100))


def rho_clause(cf, eps="0.1", floor=1e-3):
    try:
        e = cf["detail"]["eps"][eps]; rk, dk = e["rho_k"], e["dist_med_k"]
    except (KeyError, TypeError):
        return None
    use = [rk[k] for k in range(len(rk)) if k < len(dk) and dk[k] > floor and rk[k] > 0]
    if len(use) >= 2:
        return math.exp(sum(math.log(x) for x in use) / len(use))
    return rk[0] if rk and any(x <= floor for x in dk) else None


def serial(dn):
    return bool(dn) and all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] - v.get("chance_overlap", 0.0) < 0.10 for v in dn.values())


def slot(prefix, k):
    att = []
    for o in OFFS:
        d = "%s_s%d" % (prefix, k + o)
        if smoke(d):
            break
        pl = plateau(d); r = load(d)
        if pl is None and r is None:
            break
        if pl is None:
            return {"attempts": att, "status": "running (phase 1)", "accepted": None}
        att.append({"seed": k + o, "plateau": pl})
        if pl is False:
            if r is None:
                return {"attempts": att, "status": "running", "accepted": None}
            a = {"seed": k + o, "long_acc": r.get("long_acc"), "ext_2048": (r.get("ext_2048") or {}).get("cumulative"),
                 "serial": serial(r.get("donor")), "rho": rho_clause(load(d, "eval_contraction_fp32.json")), "grid_share": r.get("grid_share"),
                 "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver")}
            a["exact2048"] = (a["ext_2048"] or 0) >= 0.99
            a["pass"] = None if a["rho"] is None else bool(a["exact2048"] and a["serial"] and a["rho"] <= 0.28)
            return {"attempts": att, "status": "accepted", "accepted": a}
    if len(att) == len(OFFS) and all(x["plateau"] for x in att):
        return {"attempts": att, "status": "exhausted", "accepted": None}
    return {"attempts": att, "status": "open", "accepted": None}


def arm_verdict(arm):
    sl = {k: slot("s133_%s" % arm, k) for k in (1, 2, 3, 4)}
    s1 = sl[1]["accepted"]
    if s1 is not None and not s1["exact2048"]:
        return sl, "dropped at proof (s1 not exact at 2,048)"
    if not any(v["attempts"] or v["status"] != "open" for v in sl.values()):
        return sl, "not run"
    passes = [v["accepted"]["pass"] for v in sl.values() if v["accepted"]]
    decided = [v for v in sl.values() if v["status"] == "exhausted" or (v["accepted"] and v["accepted"]["pass"] is not None)]
    n_pass = sum(1 for p in passes if p)
    if n_pass >= 3:
        return sl, "PASS"
    if len(decided) - n_pass >= 2:
        return sl, "FAIL"
    return sl, "incomplete (%d / 4 slots decided)" % len(decided)


def cap_verdict(arm):
    sl = {k: slot("s133cap_%s" % arm, k) for k in (1, 2, 3, 4)}
    acc = [v["accepted"] for v in sl.values() if v["accepted"]]
    n_ok = sum((a["long_acc"] or 0) >= 0.99 for a in acc)
    dec = sum(1 for v in sl.values() if v["status"] in ("accepted", "exhausted") and (v["status"] == "exhausted" or v["accepted"]))
    if not dec:
        return sl, "not run"
    return sl, ("CAP-PASS" if n_ok >= 3 else ("CAP-FAIL" if dec - n_ok >= 2 else "incomplete (%d / 4)" % dec))


def main():
    res = {"stage1": {}, "stage2": {}}
    for arm in ARMS:
        sl, v = arm_verdict(arm); res["stage1"][arm] = {"slots": sl, "verdict": v}
        print("S1", arm, v, {k: (s["status"], s["accepted"] and (s["accepted"]["ext_2048"], s["accepted"]["serial"], s["accepted"]["rho"])) for k, s in sl.items()})
    for arm in ARMS + ("fsq8",):
        sl, v = cap_verdict(arm)
        if v != "not run":
            res["stage2"][arm] = {"slots": sl, "verdict": v}; print("S2", arm, v)
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s133.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
