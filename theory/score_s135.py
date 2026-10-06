"""THEORY §135 scoring (rules fixed in §135; written before any §135 implementation or run; Theory is the only scorer).
TEACHER-STATE DISTILLATION on §128's S5 recipe (fsq120 FINAL, FSQ-8, RTX 3090 cloud, checkpoints, NO restarts and no plateau stop).
Teacher: boxes/runs_fsq/s135_teacher_s5 - gate A1 (eval.json acc_L32 >= 0.99 on 1,000 held-out at L = 32) and gate A2
(teacher_probe.json: per-layer 120-way state probe accuracy at the last token of each step's state text; chosen layer = argmax, ties
to the later layer; needs >= 0.95). No student run is read unless both gates pass.
Students: boxes/runs_fsq/s135_{tsd,base}_s5_s{1,2,3,4} (135-1: the control is BASE, the pretrained GPT-2 on the same CoT text); args must
be d_q 8, levels 8, family s5, p_hard 2000 / 4000 (135-1: missing keys = an old image = error), tsd = arm ("teacher" / "base"), tsd_w 1.0, tsd_tau 0.1, no stop_plateau, 24k, sched_scale 1 (smokes skipped).
PASS(arm): all four seeds exact at 2,048 (ext_2048 >= 0.99), SERIAL (donor >= 0.90, recipient - chance < 0.10) and contractive
(127-12's rho clause as in score_s134: geomean of rho_k over hand-offs with dist_med > 1e-3 and rho > 0, else rho_1;
<= 0.28). IMPROVES (descriptive) 3/4; FAIL <= 2/4. The user's claim on S5: TSD PASS and BASE not -> the CoT model's
internals help beyond a generic encoding of the same text; both PASS -> S5 does not separate; BASE PASS and TSD not -> the generic
encoding is better here.
Descriptive per seed: prefix accuracy (eval_prefix.json or eval.json acc_by_length 1 / 2 / 4 / 8), end probe, InfoNCE retrieval.
    python theory/score_s135.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
ARMS = {"tsd": "teacher", "base": "base"}                               # 135-1: TEXT -> BASE (stationary, same input)
SKIPPED, ERRORS = [], []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


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


def teacher_gate():
    e, p = load("s135_teacher_s5"), load("s135_teacher_s5", "teacher_probe.json")
    if e is None or p is None:
        return None, "teacher pending"
    a1 = (e.get("acc_L32") or 0) >= 0.99
    acc = {int(k): v for k, v in (p.get("probe_acc_by_layer") or {}).items()}
    if not acc:
        return False, "teacher_probe.json has no probe_acc_by_layer"
    best = max(acc.values()); layer = max(k for k, v in acc.items() if v == best)        # ties -> the later layer
    if p.get("chosen_layer") is not None and int(p["chosen_layer"]) != layer:
        ERRORS.append("teacher chosen_layer %s != argmax layer %d" % (p.get("chosen_layer"), layer))
    a2 = best >= 0.95
    return bool(a1 and a2), "A1 acc_L32 %s %s | A2 layer %d probe %.4f %s" % (e.get("acc_L32"), a1, layer, best, a2)


def bad_args(d, arm):
    a = load(d, "args.json") or {}
    if int(a.get("steps", 24000)) != 24000 or float(a.get("sched_scale", 1.0)) != 1.0 or str(a.get("device", "")).startswith("cpu") or a.get("mem_check"):
        SKIPPED.append(d); return "smoke"
    exp = {"d_q": 8, "levels": 8, "family": "s5", "p_hard_start": 2000, "p_hard_end": 4000, "tsd": ARMS[arm], "tsd_w": 1.0, "tsd_tau": 0.1}
    bad = {k: a.get(k) for k, v in exp.items() if a.get(k, v if k in ("d_q", "levels") else None) != v}
    if a.get("stop_plateau"):
        bad["stop_plateau"] = a.get("stop_plateau")
    if bad:
        ERRORS.append((d, bad)); return "args mismatch %s" % bad
    return None


def prefix(d):
    p = load(d, "eval_prefix.json")
    if p:
        return p.get("acc_by_prefix")
    ab = (load(d) or {}).get("acc_by_length") or {}
    return {k: ab.get(k) for k in ("1", "2", "4", "8") if k in ab} or None


def run(d, arm):
    r = load(d)
    if r is None:
        return None
    b = bad_args(d, arm)
    if b:
        return {"error": b}
    out = {"long_acc": r.get("long_acc"), "ext_2048": (r.get("ext_2048") or {}).get("cumulative"), "serial": serial(r.get("donor")),
           "rho": rho_clause(load(d, "eval_contraction_fp32.json")), "prefix": prefix(d), "gpu": r.get("train_gpu", r.get("gpu")),
           "driver": r.get("train_driver"), "tsd_retrieval": (r.get("s135") or {}).get("retrieval_top1"),
           "masked_share": (r.get("s135") or {}).get("masked_share")}           # 135-1 (d): expected about 1/120 (< 0.01)
    out["exact"] = (out["ext_2048"] or 0) >= 0.99
    out["pass"] = None if out["rho"] is None else bool(out["exact"] and out["serial"] and out["rho"] <= 0.28)
    return out


def verdict(runs):
    errs = [s for s, v in runs.items() if v and "error" in v]
    done = {s: v for s, v in runs.items() if v and "error" not in v}
    if errs:
        return "invalid runs %s" % errs
    if len(done) < 4 or any(v["pass"] is None for v in done.values()):
        return "incomplete (%d / 4)" % len(done)
    n = sum(1 for v in done.values() if v["pass"])
    return "PASS" if n == 4 else ("IMPROVES (3/4, descriptive)" if n == 3 else "FAIL (%d / 4)" % n)


def main():
    ok, why = teacher_gate()
    print("TEACHER GATE:", ok, why)
    res = {"teacher_gate": [ok, why], "arms": {}}
    if ok:
        for arm in ARMS:
            runs = {s: run("s135_%s_s5_s%d" % (arm, s), arm) for s in (1, 2, 3, 4)}
            v = verdict(runs); res["arms"][arm] = {"runs": runs, "verdict": v}
            for s, x in runs.items():
                print(arm, "s%d" % s, x and {k: x.get(k) for k in ("long_acc", "ext_2048", "serial", "rho", "pass", "prefix", "gpu", "tsd_retrieval", "masked_share", "error")})
            print("VERDICT", arm, v)
        vt, vx = res["arms"]["tsd"]["verdict"], res["arms"]["base"]["verdict"]
        if vt.startswith(("incomplete", "invalid")) or vx.startswith(("incomplete", "invalid")):
            claim = "pending"
        elif vt == "PASS" and vx != "PASS":
            claim = "the CoT model's internals help beyond a generic encoding of the same text"
        elif vt == "PASS" and vx == "PASS":
            claim = "S5 does not separate (both PASS)"
        elif vx == "PASS":
            claim = "the generic (pretrained) encoding is better here"
        else:
            claim = "neither PASS"
        res["claim_s5"] = claim; print("USER CLAIM ON S5:", claim)
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s135.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
