"""THEORY §133 scoring (rules fixed in §133; written before any §133 run; Theory is the only scorer).
Wide-carrier error correction on §128's S5 recipe (fsq120 FINAL, restart-on-plateau slots k, k+100, k+200, k+300; RTX 3090 cloud,
checkpoints). Arms: cont64, fsq16, fsq64, soft16, soft64, noise64, pin64 (JAC dropped after Literature 7dbdb426).
Stage 1 dirs: boxes/runs_fsq/s133_{arm}_s{seed}; eval.json long_acc (512), ext_2048.cumulative, donor; eval_contraction_fp32.json.
PASS(arm): >= 3/4 slots whose accepted run is exact at 2,048 (ext_2048 >= 0.99), SERIAL (donor >= 0.90, recipient - chance < 0.10
at every position) and contractive (rho <= 0.28; 133-1 (a): rounded arms use 127-12's clause, continuous / soft arms the geometric mean over the first
3 hand-offs; the bound is carried over from d = 8). Args must match the arm (133-1 (b)); JAC was not run and counts for nothing.
DROP (no follow-up): the proof slot (slot 1) is accepted but not exact at 2,048 -> the arm stops ("dropped at proof").
Stage 2 (capacity): S5x2 (in-capacity control, dirs s133cap2_{arm}_s{seed}) and S5x4 (probe, dirs s133cap4_{arm}_s{seed}), all registers
queried at the end; arms = passing wide arms + fsq8 (8 levels, 24 bits) + fsq8L16 (16 levels, 32 bits); CAP-PASS = long_acc >= 0.99 at
512 in >= 3/4 slots.
133-11 (lean stage 2): cells cap2|fsq8, cap4|fsq8, cap4|fsq8L16, cap4|noise64, cap4|cgrownoise (§136 stage 2, note 136-2: COPY + GROW-NOISE with the
PLATEAU trigger, no restarts: the seed-k run is the slot). args family must be s5x2 / s5x4. Proof = slot 1 per cell; a proof accepted with long_acc < 0.95
at 512 drops the cell, except cap4|fsq8 (the probe's expected failure completes). Proof-level capacity reading (descriptive, n = 1) printed beside 133-2 (2).
    python theory/score_s133.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
ARMS = ("cont64", "fsq16", "fsq64", "soft16", "soft64", "noise64", "pin64")   # 133 after Literature 7dbdb426: JAC dropped, d 16 added
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


CONTINUOUS = {"cont64", "soft16", "soft64", "noise64", "pin64"}             # 133-1 (a): carry never collapses at test


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


EXPECT = {"cont64": {"d_q": 64, "no_round": True}, "fsq16": {"d_q": 16}, "fsq64": {"d_q": 64}, "soft16": {"d_q": 16, "soft_beta": 0.1},
          "soft64": {"d_q": 64, "soft_beta": 0.1}, "noise64": {"d_q": 64, "no_round": True, "dither": 0.25},
          "pin64": {"d_q": 64, "no_round": True, "carrier_ln": True}, "fsq8": {"d_q": 8, "levels": 8}, "fsq8L16": {"d_q": 8, "levels": 16},
          "cgrownoise": {"d_q": 8, "grow": "8:64", "grow_trigger": "plateau", "grow_noise": 0.25, "copy_frac": 0.1},   # 133-11 / 136-2
          "fullgrow": {"d_q": 8, "levels": 8, "grow": "8:64", "grow_trigger": "plateau", "grow_noise": 0.25, "copy_frac": 0.1, "grow2": 768,
                       "grow2_lr_scale": 0.0833, "grow2_min_step": 6000}}   # 133-12: §143's recipe with 136-2's PLATEAU trigger
NORESTART = {"cgrownoise", "fullgrow"}                                                  # §136's COPY arms: no restarts, the seed-k run is the slot
CAP_ARMS = ("fsq8", "fsq8L16", "noise64", "cgrownoise", "fullgrow")                     # 133-11's lean stage-2 cells
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
    if arm in NORESTART:                                                    # 133-11: no restart offsets; accepted whatever the 2k flag
        d = "%s_s%d" % (prefix, k); r = load(d)
        if smoke(d) or r is None:
            return {"attempts": att, "status": "open" if not os.path.exists(os.path.join(R, d)) else "running", "accepted": None}
        if not args_ok(d, arm):
            return {"attempts": att, "status": "error", "accepted": None}
        a = {"seed": k, "long_acc": r.get("long_acc"), "ext_2048": (r.get("ext_2048") or {}).get("cumulative"), "serial": serial(r.get("donor")),
             "family": (load(d, "args.json") or {}).get("family"), "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver")}
        return {"attempts": [{"seed": k, "plateau": plateau(d)}], "status": "accepted", "accepted": a}
    for o in OFFS:
        d = "%s_s%d" % (prefix, k + o)
        if smoke(d):
            break
        pl = plateau(d); r = load(d)
        if pl is None and r is None:
            sf = load("s133_slots/%s_k%d" % (prefix.split("_", 1)[1], k)) if prefix.startswith("s133_") else None   # 133-5: plateau
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
                 "serial": serial(r.get("donor")), "grid_share": r.get("grid_share"), "family": (load(d, "args.json") or {}).get("family"),
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
    sl = {k: slot("s133_%s" % arm, k, arm) for k in (1, 2, 3, 4)}
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


def cap_verdict(arm, task="cap4"):
    sl = {k: slot("s133%s_%s" % (task, arm), k, arm) for k in (1, 2, 3, 4)}
    fam = {"cap2": "s5x2", "cap4": "s5x4"}[task]
    for k_, v_ in sl.items():                                                # 133-11: the run must be on its task's family
        if v_["accepted"] and v_["accepted"].get("family") != fam:
            ERRORS.append(("s133%s_%s slot %d" % (task, arm, k_), "family %s != %s" % (v_["accepted"].get("family"), fam))); v_["accepted"] = None; v_["status"] = "error"
    acc = [v["accepted"] for v in sl.values() if v["accepted"]]
    s1 = sl[1]["accepted"]
    if s1 is not None and (s1["long_acc"] or 0) < 0.95 and not (task == "cap4" and arm == "fsq8"):   # 133-11 proof rule
        return sl, "dropped at proof (slot 1 < 0.95 at 512)"
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
    for task in ("cap2", "cap4"):                                            # S5x2 (in-capacity control), S5x4 (capacity probe)
        for arm in ARMS + ("fsq8", "fsq8L16", "cgrownoise", "fullgrow"):
            sl, v = cap_verdict(arm, task)
            if v != "not run":
                res["stage2"]["%s|%s" % (task, arm)] = {"slots": sl, "verdict": v}; print("S2", task, arm, v)
    st2 = res["stage2"]
    def v(key):
        return (st2.get(key) or {}).get("verdict")
    def entered(key):
        sl = (st2.get(key) or {}).get("slots") or {}
        return bool(sl) and not any(x["status"] == "exhausted" for x in sl.values()) and any(x["accepted"] for x in sl.values())
    wide_cap = [a for a in ARMS + ("cgrownoise", "fullgrow") if v("cap4|%s" % a) == "CAP-PASS"]
    def p1(key):                                                             # 133-11: proof-level (slot 1) long_acc, descriptive
        x = ((st2.get(key) or {}).get("slots") or {}).get(1) or {}
        return (x.get("accepted") or {}).get("long_acc")
    pr = {k: p1(k) for k in ("cap2|fsq8", "cap4|fsq8", "cap4|fsq8L16", "cap4|noise64", "cap4|cgrownoise")}
    if all(x is not None for x in pr.values()):
        ind = ((pr["cap2|fsq8"] >= 0.99) and (pr["cap4|fsq8"] < 0.99) and any(pr[k] >= 0.99 for k in ("cap4|fsq8L16", "cap4|noise64", "cap4|cgrownoise")))
        res["proof_capacity_reading"] = {"slot1_long_acc": pr, "capacity_indicated_at_proof": ind}
        print("PROOF-LEVEL (133-11, descriptive, n = 1 per cell):", pr, "capacity indicated:", ind)
    def decided(key):                                                        # 133-11: an incomplete cell is not yet a failure
        x = v(key)
        return x is not None and not x.startswith("incomplete") and x != "not run"
    pending_wide = any((v(k) or "").startswith("incomplete") for k in ["cap4|%s" % a for a in ARMS + ("cgrownoise", "fsq8L16", "fullgrow")])
    if not (decided("cap2|fsq8") and decided("cap4|fsq8")):
        cap = "incomplete"
    elif v("cap2|fsq8") != "CAP-PASS":
        cap = "difficulty (FSQ-8 fails S5x2)"
    elif not entered("cap4|fsq8"):
        cap = "entry (FSQ-8 does not enter S5x4)"
    elif v("cap4|fsq8") == "CAP-FAIL" and (wide_cap or v("cap4|fsq8L16") == "CAP-PASS"):
        cap = "CAPACITY (FSQ-8 fails S5x4 after entering; passes: %s)" % (wide_cap + (["fsq8L16"] if v("cap4|fsq8L16") == "CAP-PASS" else []))
    elif v("cap4|fsq8") == "CAP-FAIL" and pending_wide:
        cap = "incomplete (FSQ-8 fails S5x4; wide cells still open)"
    elif v("cap4|fsq8") == "CAP-FAIL":
        cap = "difficulty (nothing passes S5x4)"
    else:
        cap = "no capacity limit shown (FSQ-8 %s on S5x4)" % v("cap4|fsq8")
    res["capacity_reading"] = cap; print("CAPACITY READING (133-2):", cap)
    if decided("cap4|fullgrow") and decided("cap4|cgrownoise"):                # 133-12: full width against the 64-d growth arm at capacity
        fw, g64 = v("cap4|fullgrow") == "CAP-PASS", v("cap4|cgrownoise") == "CAP-PASS"
        res["fullwidth_reading"] = ("768 supplies capacity (FULL-GROW passes where cgrownoise fails)" if fw and not g64 else
                                    "full width costs nothing at capacity (both pass)" if fw and g64 else "no full-width benefit shown")
        print("FULL-WIDTH READING (133-12):", res["fullwidth_reading"])
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s133.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
