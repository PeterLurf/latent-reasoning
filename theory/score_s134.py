"""THEORY §134 scoring (rules fixed in §134; written before any §134 run; Theory is the only scorer).
COPY-ANCHOR on §128's S5 recipe: fsq120 FINAL, FSQ-8, plus label-free copy rows (a final-answer text autoencoded through the carrier,
decoded from the code alone), RTX 3090 cloud, checkpoints, NO restarts and no plateau stop (each seed is one 24k run).
Main: boxes/runs_fsq/s134_copy_s5_s{1,2,3,4}. Guard (only after PASS): s134_copy_{ff,cyc,mod,rdfa}_s1.
PASS: all four S5 seeds exact at 2,048 (ext_2048 >= 0.99), SERIAL (donor >= 0.90, recipient - chance < 0.10) and contractive (rho by 127-12's
clause <= 0.28). IMPROVES (descriptive): 3/4. FAIL: <= 2/4. Plain comparator on the same seeds / card class: s1 exact; s2, s4 plateau; s3
escapes without learning (1/4).
GUARD: each guard family's run exact (long_acc >= 0.99), SERIAL, and CLOSED where §128 required it (CERT holds, Moore = minimal states).
Mechanism (descriptive): probe at 2k and the end - R2_state, R2_item, op-history R2 (last 1-3 ops).
    python theory/score_s134.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
NST = {"cyc": 5, "mod": 5, "rdfa": 8}


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


def probes(d):
    p = os.path.join(R, d, "log.jsonl")
    if not os.path.exists(p):
        return {}
    return {x["step"]: x["probe"] for x in map(json.loads, open(p)) if "probe" in x}


def mech2k(d):
    """fsq120 logs the §134 mechanism probes at 2k as {"step", "s134": {dialect, copy_donor, ophist}} in log.jsonl."""
    p = os.path.join(R, d, "log.jsonl")
    if not os.path.exists(p):
        return None
    return next((x["s134"] for x in map(json.loads, open(p)) if "s134" in x), None)


def smoke_or_bad(d, family):
    """134-3 (b): smoke runs skipped; args must be d_q 8, levels 8, the family, copy_frac 0.1."""
    a = load(d, "args.json") or {}
    if int(a.get("steps", 24000)) != 24000 or float(a.get("sched_scale", 1.0)) != 1.0 or str(a.get("device", "")).startswith("cpu") or a.get("mem_check"):
        return "smoke"
    exp = {"d_q": 8, "levels": 8, "family": family, "copy_frac": 0.1, "p_hard_start": 2000, "p_hard_end": 4000}   # 134-4 (c): §120's ramp pinned
    bad = {k: a.get(k) for k, v in exp.items() if a.get(k, v if k in ("d_q", "levels") else None) != v}
    if bad:
        return "args mismatch %s" % bad
    lp = os.path.join(R, d, "log.jsonl")                                    # 134-4 (a): "loss" must be the main chain alone
    if os.path.exists(lp):
        recs = [x for x in map(json.loads, open(lp)) if "copy_loss" in x]
        if recs and not any("loss_total" in x for x in recs):
            return "old harness: log has copy_loss but no loss_total (loss contaminated by the copy term)"
    return None


def run(d, family="s5"):
    r = load(d)
    if r is None:
        return None
    a = load(d, "args.json") or {}
    sb = smoke_or_bad(d, family)
    if sb:
        return {"error": sb}
    out = {"long_acc": r.get("long_acc"), "ext_2048": (r.get("ext_2048") or {}).get("cumulative"), "serial": serial(r.get("donor")),
           "rho": rho_clause(load(d, "eval_contraction_fp32.json")), "copy_frac": a.get("copy_frac"), "stop_plateau": a.get("stop_plateau"),
           "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver"), "probes": probes(d),
           "dialect": (r.get("s134") or {}).get("dialect"), "copy_donor": (r.get("s134") or {}).get("copy_donor"),
           "copy_acc": (r.get("s134") or {}).get("copy_acc"), "ophist_end": (r.get("s134") or {}).get("ophist"),
           "mech_2k": mech2k(d)}                                           # 134-1 / 134-3 (c): end in eval.json["s134"], 2k in log.jsonl
    out["exact"] = (out["ext_2048"] if out["ext_2048"] is not None else out["long_acc"] or 0) >= 0.99
    out["pass"] = None if out["rho"] is None else bool(out["exact"] and out["serial"] and out["rho"] <= 0.28)
    out["valid"] = bool(a.get("copy_frac")) and not a.get("stop_plateau")       # COPY on, no plateau stop (no restarts)
    return out


def main():
    main_runs = {s: run("s134_copy_s5_s%d" % s) for s in (1, 2, 3, 4)}
    errs = {s: v["error"] for s, v in main_runs.items() if v and "error" in v}
    done = {s: v for s, v in main_runs.items() if v and "error" not in v}
    bad = [s for s, v in done.items() if not v["valid"]] + list(errs)
    n_pass = sum(1 for v in done.values() if v["pass"])
    if bad:
        verdict = "invalid runs %s (copy off or plateau stop on)" % bad
    elif len(done) < 4 or any(v["pass"] is None for v in done.values()):
        verdict = "incomplete (%d / 4)" % len(done)
    else:
        verdict = "PASS" if n_pass == 4 else ("IMPROVES (3/4, descriptive)" if n_pass == 3 else "FAIL (%d / 4)" % n_pass)
    guard = {}
    for f in ("ff", "cyc", "mod", "rdfa"):
        d = "s134_copy_%s_s1" % f; r = load(d)
        if r is None:
            continue
        c = load(d, "eval_cert_fp32.json") or load(d, "eval_cert.json") or {}
        if f not in NST:
            closed = None
        elif not c or c.get("holds") is None:
            closed = "pending"                                              # 134-3 (a): no cert yet -> incomplete, not ok
        else:
            closed = bool(c.get("holds") and not c.get("capped") and c.get("n_moore") == NST[f])
        acc_ok = (r.get("long_acc") or 0) >= 0.99 and serial(r.get("donor"))
        guard[f] = {"long_acc": r.get("long_acc"), "serial": serial(r.get("donor")), "closed": closed,
                    "ok": "incomplete (CERT pending)" if closed == "pending" else bool(acc_ok and closed is not False)}
    for s, v in main_runs.items():
        if v and "error" in v:
            print("S5 s%d ERROR" % s, v["error"]); continue
        print("S5 s%d" % s, v and {k: v[k] for k in ("long_acc", "ext_2048", "serial", "rho", "pass", "valid", "gpu", "copy_acc")},
              v and {"dialect": v["dialect"], "copy_donor": v["copy_donor"]},
              v and {st: {k: p.get(k) for k in ("r2_state", "r2_item", "r2_last1", "r2_last2", "r2_last3") if k in p} for st, p in v["probes"].items()})
    print("GUARD", {f: g["ok"] for f, g in guard.items()})
    print("VERDICT", verdict)
    json.dump({"main": main_runs, "verdict": verdict, "guard": guard}, open(os.path.join(ROOT, "theory", "results", "score_s134.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
