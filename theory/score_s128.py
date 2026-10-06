"""THEORY §128 scoring (rules fixed in §128; written before any §128 run; Theory is the only scorer).
One recipe (§120 FINAL fsq120 + the 127-3 non-finite guard; d 8, L 8; one code version) on four families, with label-free
restart-on-plateau (§126 reading): seed slot k in 1..4 tries seeds k, k+100, k+200, k+300 in order; an attempt whose 123-3 flag
fires is stopped at 2k (log kept) and the next seed runs; the slot's accepted run is its first non-plateau attempt.
Runs: boxes/runs_fsq/s128_{s5,ff,cyc,mod}_s{seed}  (eval.json: long_acc, ext_2048, donor, cert; log.jsonl; eval_contraction_fp32.json).
FAMILY-PASS(f): >= 3/4 slots whose accepted run has long_acc >= 0.99 (512) and SERIAL (every donor position: donor_consistent >= 0.90
and recipient_consistent - chance_overlap < 0.10), and for CYC / MOD / RDFA also CLOSED (Literature 4d50870d: a counter code is exact
at 512 without being the automaton): CERT holds, not capped / timed out, Moore-minimised size = the minimal state count (5 / 5 / 8);
CERT from eval_cert.json if present, else eval.json.  TASK-AGNOSTIC: FAMILY-PASS in all five families.
Descriptive: 2,048, CERT, rho_ff (123-9), restarts per slot, cost per accepted exact seed = (accepted runs + plateaued attempts / 12)
/ exact accepted runs (run-equivalents), non-finite skipped-step share (> 1 % -> the run is not delivered, scored as a failed slot).
    python theory/score_s128.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
FAMS = ("s5", "ff", "cyc", "mod", "rdfa")
NST = {"cyc": 5, "mod": 5, "rdfa": 8}                                       # CLOSED families: minimal state count
OFFS = (0, 100, 200, 300)


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def log_lines(d):
    p = os.path.join(R, d, "log.jsonl")
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else None


def plateau(d):
    lines = log_lines(d)
    if lines is None:
        return None
    L = {x["step"]: x["loss"] for x in lines if "loss" in x and "step" in x}
    if not L or max(L) < 2000:
        return None                                                          # 2k not reached: undecided
    g = lambda t: L[min(L, key=lambda s: abs(s - t))]  # noqa: E731
    return bool(g(2000) >= 0.9 * g(1000) and g(2000) >= 0.25 * g(100))


def nonfinite_share(d):
    lines = log_lines(d) or []
    last = max([int(x.get("step", 0)) for x in lines] or [0])
    return sum(1 for x in lines if x.get("nonfinite")) / last if last else 0.0


def serial(dn):
    return bool(dn) and all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] - v.get("chance_overlap", 0.0) < 0.10 for v in dn.values())


def rho_ff(cf, eps="0.1", floor=1e-3):                                       # note 123-9
    try:
        e = cf["detail"]["eps"][eps]; rk, dk = e["rho_k"], e["dist_med_k"]
    except (KeyError, TypeError):
        return None
    use = [rk[k] for k in range(len(rk)) if k < len(dk) and dk[k] > floor and rk[k] > 0]
    return math.exp(sum(math.log(x) for x in use) / len(use)) if len(use) >= 2 else None


def slot(fam, k):
    """The slot's attempts in seed order; accepted = first non-plateau attempt (needs its eval.json to be scored)."""
    att = []
    for o in OFFS:
        d = "s128_%s_s%d" % (fam, k + o)
        pl = plateau(d)
        if pl is None and load(d) is None:
            break                                                            # not run yet (or before 2k): the slot is open
        att.append({"seed": k + o, "plateau": pl})
        if pl and load(d) is not None:                                       # note 128-1: a flagged attempt trained on (flag check)
            att[-1]["flag_check_long_acc"] = load(d).get("long_acc")
        if pl is False:
            r = load(d)
            if r is None:
                return {"attempts": att, "status": "running", "accepted": None}
            nf = nonfinite_share(d)
            ok = (r.get("long_acc") or 0) >= 0.99 and nf <= 0.01
            acc = {"seed": k + o, "long_acc": r.get("long_acc"), "serial": serial(r.get("donor")), "nonfinite_share": nf,
                   "ext_2048": (r.get("ext_2048") or {}).get("cumulative"), "cert_holds": (r.get("cert") or {}).get("holds"),
                   "rho_ff": rho_ff(load(d, "eval_contraction_fp32.json")), "gpu": r.get("train_gpu", r.get("gpu"))}
            if fam in NST:
                c = load(d, "eval_cert.json") or r.get("cert") or {}
                acc["n_moore"] = c.get("n_moore")
                acc["closed"] = (None if c.get("holds") is None and c.get("status") in (None, "deferred") else
                                 bool(c.get("holds") and not c.get("capped") and not c.get("timed_out") and c.get("n_moore") == NST[fam]))
                if acc["closed"] is None:
                    return {"attempts": att, "status": "accepted (CERT pending)", "accepted": None}
            acc["pass"] = bool(ok and acc["serial"] and acc.get("closed", True))
            return {"attempts": att, "status": "accepted", "accepted": acc}
    if len(att) == len(OFFS) and all(a["plateau"] for a in att):
        return {"attempts": att, "status": "failed (all attempts plateaued)", "accepted": None}
    return {"attempts": att, "status": "open", "accepted": None}


def family(fam):
    sl = {k: slot(fam, k) for k in (1, 2, 3, 4)}
    done = [v for v in sl.values() if v["status"] == "accepted" or v["status"].startswith("failed")]
    n_pass = sum(bool(v["accepted"] and v["accepted"]["pass"]) for v in sl.values())
    n_fail = sum(1 for v in done if not (v["accepted"] and v["accepted"]["pass"]))
    fp = True if n_pass >= 3 else (False if n_fail >= 2 else "incomplete (%d / 4 slots decided)" % len(done))
    acc = [v["accepted"] for v in sl.values() if v["accepted"]]
    n_pl = sum(sum(bool(a["plateau"]) for a in v["attempts"]) for v in sl.values())
    exact = sum((a["long_acc"] or 0) >= 0.99 for a in acc)
    cost = (len(acc) + n_pl / 12) / exact if exact else None
    fc = [a for v in sl.values() for a in v["attempts"] if "flag_check_long_acc" in a]
    return {"slots": sl, "FAMILY_PASS": fp, "flag_checks": fc, "n_pass": n_pass, "plateaued_attempts": n_pl, "cost_per_exact_seed": cost}


def main():
    res = {f: family(f) for f in FAMS}
    fps = [res[f]["FAMILY_PASS"] for f in FAMS]
    res["TASK_AGNOSTIC"] = True if all(x is True for x in fps) else (False if any(x is False for x in fps) else "incomplete")
    for f in FAMS:
        v = res[f]
        print(f.upper(), {k: (s["status"], s["accepted"] and (s["accepted"]["long_acc"], s["accepted"]["serial"], s["accepted"].get("closed"), s["accepted"]["pass"])) for k, s in v["slots"].items()},
              "| FAMILY-PASS", v["FAMILY_PASS"], "| plateaued attempts", v["plateaued_attempts"], "| cost", v["cost_per_exact_seed"], "| flag checks (128-1)", [(a["seed"], a["flag_check_long_acc"]) for a in v["flag_checks"]])
    print("TASK-AGNOSTIC", res["TASK_AGNOSTIC"])
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s128.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
