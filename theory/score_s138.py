"""THEORY §138 scoring (rules fixed in §138; written before any §138 implementation or run; Theory is the only scorer).
FULL WIDTH (the user's trigger: §133's 64-d arms passed stage 1): §133's harness exactly (§128's S5 recipe, restart slots k, k+100, k+200, k+300,
RTX 3090 cloud, checkpoints, no COPY) with the carrier width d 768. One change from §133 per arm (the width), plus one hinge arm.
Arms (dirs boxes/runs_fsq/s138_{arm}_s{seed}):
  full_fsq    d 768, hard FSQ (8 levels)
  full_noise  d 768, continuous + dither U(+-0.25) at training hand-offs after 2k (NOISE-64's)
  full_cont   d 768, continuous, no correction (CONT-64's; the direct contrast with §126 RAW-H)
  full_hinge  full_cont + the local contraction hinge (candidate 6; Literature d4f0d18b): from 2k, perturb the pre-quantisation z by eps =
              0.15 x the nearest-neighbour state-centroid distance (measured at 2k), one segment, penalise max(0, rho_hat - 0.3), weight 1
PASS(arm): >= 3/4 slots whose accepted run is exact at 2,048 (ext_2048 >= 0.99), SERIAL and contractive (rho <= 0.28: 127-12's clause for
full_fsq; 133-1's first-3 form for the continuous arms). DROP at proof: slot 1 accepted with ext_2048 < 0.95 stops the arm.
    python theory/score_s138.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
ARMS = ("full_fsq", "full_noise", "full_cont", "full_hinge")
OFFS = (0, 100, 200, 300)
SKIPPED = []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def smoke(d):
    a = load(d, "args.json")
    if not a:
        return False
    bad = (int(a.get("steps", 24000)) != 24000 or float(a.get("sched_scale", 1.0)) != 1.0 or str(a.get("device", "")).startswith("cpu")
           or bool(a.get("mem_check")))
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


CONTINUOUS = {"full_noise", "full_cont", "full_hinge"}


def rho_k3(cf, eps="0.1"):
    """133-1 (a), continuous / soft arms: geometric mean of rho_k over the first 3 hand-offs (avoids fp32-floor hand-offs)."""
    try:
        e = cf["detail"]["eps"][eps]; rk, dk = e["rho_k"], e["dist_med_k"]
    except (KeyError, TypeError):
        return None
    use = [rk[k] for k in range(min(3, len(rk), len(dk))) if dk[k] > 1e-2 and rk[k] > 0]   # 133-2: drop hand-offs landing near the floor
    if not use:
        return rk[0] if rk else None
    return math.exp(sum(math.log(x) for x in use) / len(use))


EXPECT = {"full_fsq": {"d_q": 768, "levels": 8}, "full_noise": {"d_q": 768, "no_round": True, "dither": 0.25},
          "full_cont": {"d_q": 768, "no_round": True}, "full_hinge": {"d_q": 768, "no_round": True, "hinge": True, "hinge_eps_frac": 0.15}}
ERRORS = []


def args_ok(d, arm):
    """133-1 (b): args must match the dir's arm; mismatch -> error row (slot invalid)."""
    a = load(d, "args.json") or {}
    bad = [k for k, v in EXPECT.get(arm, {}).items() if a.get(k) != v]
    if bad:
        ERRORS.append((d, "args mismatch for %s: %s" % (arm, {k: a.get(k) for k in bad})))
    return not bad


def serial(dn):
    return bool(dn) and all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] - v.get("chance_overlap", 0.0) < 0.10 for v in dn.values())


def slot(prefix, k, arm=None):
    att = []
    for o in OFFS:
        d = "%s_s%d" % (prefix, k + o)
        if smoke(d):
            break
        pl = plateau(d); r = load(d)
        if pl is None and r is None:
            sf = None
            got = [x for x in (sf or {}).get("attempts", []) if x.get("seed") == k + o and x.get("status") == "plateau"]    # attempt not fetched:
            if got:                                                                                                      # read the host's slot
                att.append({"seed": k + o, "plateau": True, "source": "slot file (raw log not fetched)"}); continue        # record, disclosed
            break
        if pl is None:
            return {"attempts": att, "status": "running (phase 1)", "accepted": None}
        att.append({"seed": k + o, "plateau": pl})
        if pl is False:
            if r is None:
                return {"attempts": att, "status": "running", "accepted": None}
            if arm and not args_ok(d, arm):
                return {"attempts": att, "status": "error", "accepted": None}
            a = {"seed": k + o, "long_acc": r.get("long_acc"), "ext_2048": (r.get("ext_2048") or {}).get("cumulative"),
                 "serial": serial(r.get("donor")), "grid_share": r.get("grid_share"),
                 "rho": (rho_k3 if arm in CONTINUOUS else rho_clause)(load(d, "eval_contraction_fp32.json")),
                 "rho_k_row": ((load(d, "eval_contraction_fp32.json") or {}).get("detail") or {}).get("eps", {}).get("0.1", {}).get("rho_k"),
                 "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver")}
            a["exact2048"] = (a["ext_2048"] or 0) >= 0.99
            a["pass"] = None if a["rho"] is None else bool(a["exact2048"] and a["serial"] and a["rho"] <= 0.28)
            return {"attempts": att, "status": "accepted", "accepted": a}
    if len(att) == len(OFFS) and all(x["plateau"] for x in att):
        return {"attempts": att, "status": "exhausted", "accepted": None}
    return {"attempts": att, "status": "open", "accepted": None}


def arm_verdict(arm):
    sl = {k: slot("s138_%s" % arm, k, arm) for k in (1, 2, 3, 4)}
    gpus = {v["accepted"]["gpu"] for v in sl.values() if v["accepted"]}
    if len(gpus) > 1:
        print("WARNING (%s): accepted runs span GPU names %s (125-2)" % (arm, sorted(map(str, gpus))))
    s1 = sl[1]["accepted"]
    if s1 is not None and (s1["ext_2048"] or 0) < 0.95:                     # 133-2: drop only below 0.95; 0.95-0.99 = non-exact slot
        return sl, "dropped at proof (s1 < 0.95 at 2,048)"
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


def main():
    res = {"stage1": {}}
    for arm in ARMS:
        sl, v = arm_verdict(arm); res["stage1"][arm] = {"slots": sl, "verdict": v}
        print("S1", arm, v, {k: (s["status"], s["accepted"] and (s["accepted"]["ext_2048"], s["accepted"]["serial"], s["accepted"]["rho"])) for k, s in sl.items()})
    st2 = {}
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s138.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
