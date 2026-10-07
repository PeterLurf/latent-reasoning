"""THEORY §139 scoring (rules fixed in §139; written before any §139 implementation or run; Theory is the only scorer).
DISTILL-2 on multiplier-major multiplication (fsq120 --family mult_row --task 4x4): op j = b_j with a's digits and the position in context;
carried state S_j; FINAL only (the product decoded from the final code; curriculum over L multiplier digits, answer S_L).
Carrier SOFT-16 (d 16, levels 8, soft_beta 0.1; §133's only stage-1 PASS) + §134 COPY rows (copy_frac 0.1), no restarts, no plateau stop.
Arms (dirs boxes/runs_fsq/s139_{arm}_4x4_s{1..4}):
  a  (comparator): the base above (FINAL + COPY: final-answer content only, strict C4)
  b  (DISTILL-2): a + a readout head on W_up(code_j) trained toward the frozen TC-M teacher's state S_j (139-0: KL to its per-digit state-probe
     distribution if the probe agrees with the teacher's decoded S_j >= 0.99 per digit, else CE on the teacher's own generated S_j text; C4 as
     reworded), weight 1 until step 3,000 (the curriculum reaches L 4), then linearly to 0 at 4,000 (distill_full 3000, distill_end 4000)
  c  (constant control, Literature ff34ece2): b with the weight held at 1 for all 24k (distill_end 0 = never fades)
Teacher gate: boxes/runs_fsq/s139_teacher_probe/teacher_probe.json - heldout_exact_by_layer (exact match of all S_j digits, held-out half),
chosen_layer = argmax (ties -> later), gate needs >= 0.95. No B run is read unless the gate passes.
PASS(arm): >= 3/4 seeds with test_acc >= 0.99 on the 1,000 published 4x4 items, SERIAL (donor >= 0.90, recipient - chance < 0.10; donor S_2 continued
with the recipient's a and remaining digits) and contractive (133-1's first-3 form, rho_k3 <= 0.28: SOFT is continuous). IMPROVES 3/4 is folded in;
FAIL <= 2/4.
The user's question (descriptive comparison): B PASS and A not -> the teacher's state helps where final answers alone fail; both PASS -> compare exact
counts and steps-to-exact; B worse than A -> harm (as §135).
    python theory/score_s139.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
ARMS = ("a", "b", "c")
SKIPPED, ERRORS = [], []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def rho_k3(cf, eps="0.1"):
    try:
        e = cf["detail"]["eps"][eps]; rk, dk = e["rho_k"], e["dist_med_k"]
    except (KeyError, TypeError):
        return None
    use = [rk[k] for k in range(min(3, len(rk), len(dk))) if dk[k] > 1e-2 and rk[k] > 0]
    if not use:
        return rk[0] if rk else None
    return math.exp(sum(math.log(x) for x in use) / len(use))


def serial(dn):
    return bool(dn) and all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] - v.get("chance_overlap", 0.0) < 0.10 for v in dn.values())


def teacher_gate():
    p = load("s139_teacher_probe", "teacher_probe.json")
    if p is None:
        return None, "teacher probe pending"
    acc = {int(k): v for k, v in (p.get("heldout_exact_by_layer") or {}).items()}
    if not acc:
        return False, "no heldout_exact_by_layer"
    best = max(acc.values()); layer = max(k for k, v in acc.items() if v == best)
    if p.get("chosen_layer") is not None and int(p["chosen_layer"]) != layer:
        ERRORS.append("teacher chosen_layer %s != argmax %d" % (p.get("chosen_layer"), layer))
    if p.get("label_source") != "teacher_generated":
        ERRORS.append("probe labels must come from the teacher's own generated CoT (C4 as reworded): %s" % p.get("label_source"))
    return bool(best >= 0.95), "layer %d heldout exact %.4f" % (layer, best)


def bad_args(d, arm):
    a = load(d, "args.json") or {}
    if int(a.get("steps", 24000)) != 24000 or float(a.get("sched_scale", 1.0)) != 1.0 or str(a.get("device", "")).startswith("cpu") or a.get("mem_check"):
        SKIPPED.append(d); return "smoke"
    exp = {"family": "mult_row", "task": "4x4", "d_q": 16, "levels": 8, "soft_beta": 0.1, "copy_frac": 0.1}
    if arm == "b":
        exp.update({"distill": True, "distill_full": 3000, "distill_end": 4000})
    if arm == "c":
        exp.update({"distill": True, "distill_end": 0})
    bad = {k: a.get(k) for k, v in exp.items() if a.get(k) != v}
    if arm == "a" and a.get("distill"):
        bad["distill"] = a.get("distill")
    if a.get("stop_plateau"):
        bad["stop_plateau"] = True
    if bad:
        ERRORS.append((d, bad)); return "args mismatch %s" % bad
    return None


def run(d, arm):
    r = load(d)
    if r is None:
        return None
    b = bad_args(d, arm)
    if b:
        return {"error": b}
    out = {"test_acc": r.get("test_acc"), "serial": serial(r.get("donor")), "rho": rho_k3(load(d, "eval_contraction_fp32.json")),
           "acc_by_length": r.get("acc_by_length"), "probe": r.get("s139_probe"), "distill_kl_end": (r.get("s139") or {}).get("kl_at_fade"),
           "first_exact_step": (r.get("s139") or {}).get("first_exact_step"), "readout_acc_end": (r.get("s139") or {}).get("readout_sj_acc_end"),
           "target_mode": (r.get("s139") or {}).get("target_mode"), "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver")}
    out["exact"] = (out["test_acc"] or 0) >= 0.99
    out["pass"] = None if out["rho"] is None else bool(out["exact"] and out["serial"] and out["rho"] <= 0.28)
    return out


def verdict(runs):
    errs = [s for s, v in runs.items() if v and "error" in v]
    done = {s: v for s, v in runs.items() if v and "error" not in v}
    if errs:
        return "invalid runs %s" % errs, 0
    n = sum(1 for v in done.values() if v["pass"])
    if len(done) < 4 or any(v["pass"] is None for v in done.values()):
        return "incomplete (%d / 4)" % len(done), n
    return ("PASS" if n >= 3 else "FAIL (%d / 4)" % n), n


def main():
    ok, why = teacher_gate()
    print("TEACHER GATE:", ok, why)
    res = {"teacher_gate": [ok, why], "arms": {}}
    for arm in ARMS:
        if arm in ("b", "c") and not ok:
            res["arms"][arm] = {"verdict": "not read (teacher gate)"}; print("b: not read (teacher gate)"); continue
        runs = {s: run("s139_%s_4x4_s%d" % (arm, s), arm) for s in (1, 2, 3, 4)}
        v, n = verdict(runs); res["arms"][arm] = {"runs": runs, "verdict": v, "n_pass": n}
        for s, x in runs.items():
            print(arm, "s%d" % s, x and {k: x.get(k) for k in ("test_acc", "serial", "rho", "pass", "first_exact_step", "readout_acc_end", "target_mode", "error")})
        print("VERDICT", arm, v)
    va = res["arms"].get("a", {}).get("verdict", ""); vb = res["arms"].get("b", {}).get("verdict", "")
    if va.startswith("PASS") or va.startswith("FAIL"):
        if vb.startswith("PASS") and not va.startswith("PASS"):
            claim = "the teacher's state helps where final answers alone fail"
        elif vb.startswith("PASS") and va.startswith("PASS"):
            claim = "both pass: compare exact counts and steps-to-exact (descriptive)"
        elif vb.startswith("FAIL") and va.startswith("FAIL"):
            claim = "neither passes: compare exact counts (descriptive)"
        elif vb.startswith("FAIL") and va.startswith("PASS"):
            claim = "DISTILL-2 harms (as §135)"
        else:
            claim = "pending"
    else:
        claim = "pending"
    res["claim"] = claim; print("USER'S QUESTION:", claim)
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s139.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
