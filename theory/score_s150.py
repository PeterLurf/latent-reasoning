"""THEORY §150 scoring (rules fixed in §150; written before any §150 implementation or run; Theory is the only scorer).
VOCAB-SLOT working memory (program B4; the user's "working memory... that mimics discretised tokens in thought") on §140's digit-pair 4x4 harness with §145's
K = 4 write positions: slot_k = sum_v softmax(LM head(h(w_k)) / tau)_v * wte(v) (soft), hardened by straight-through argmax (p_hard ramp), real tokens at test;
the K slots are the ONLY cross-segment channel; the decoder reads [EOT, slot_1..slot_K]; streamed product digits (C4 as reworded); restart slots k, k+100,
k+200, k+300 (123-3's 2k flag, 147-1's misfire rule). Arms (Literature c583bb23): full (the whole GPT-2 vocabulary) and digit (10 digit tokens + blank: an
abacus register); Gumbel-softmax straight-through; the slot path uses a detached wte. Dirs boxes/runs_fsq/s150_{full,digit}_4x4_s{seed}.
PASS: >= 3/4 slots whose accepted run has test_acc >= 0.99 at HARD inference, SERIAL (all K slot tokens of a donor swapped in after pair 4 / 8 / 12; row level)
and contractive (127-12's clause on the hard-token carrier, perturbation on h(w_k) before the LM head). DROP at proof: slot 1 accepted with test_acc < 0.95.
FAIL: >= 2 decided slots not passing. IMPROVES (descriptive): columns 3-7 all >= 0.9 on the proof. Descriptive: soft-at-inference accuracy (hard vs soft),
slot readability (token histogram; the share of steps whose slots spell the accumulator's digits), per-slot x per-digit probe, slot collapse (distinct tokens).
    python theory/score_s150.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
OFFS = (0, 100, 200, 300)
ARMS = {"full": "vocab", "digit": "digits"}                                  # 150: slot vocabulary (full GPT-2 vocab / 10 digits + blank)
S140_PROOF = [1.0, 1.0, 0.315, 0.132, 0.104, 0.24, 0.50, 0.936]              # §140 result: s140_pair_4x4_s1 per column (comparator)
SKIPPED, ERRORS, MISFIRES = [], [], []


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


def args_ok(d, arm):
    a = load(d, "args.json") or {}
    exp = {"family": "mult_pair", "task": "4x4", "slots": 4, "slot_space": ARMS[arm], "slot_gumbel": True, "slot_wte_detached": True}   # 150-1: logits too
    bad = {k: a.get(k) for k, v in exp.items() if a.get(k) != v}
    if a.get("copy_frac"):
        bad["copy_frac"] = a.get("copy_frac")
    if bad:
        ERRORS.append((d, bad))
    return not bad


def plateau(d):
    p = os.path.join(R, d, "log.jsonl")
    if not os.path.exists(p):
        return None
    L = {x["step"]: x["loss"] for x in map(json.loads, open(p)) if "loss" in x and "step" in x}
    if not L or max(L) < 2000:
        return None
    g = lambda t: L[min(L, key=lambda s: abs(s - t))]  # noqa: E731
    fl = bool(g(2000) >= 0.9 * g(1000) and g(2000) >= 0.25 * g(100))
    if fl and g(2000) < 1.0:                                                 # 147-1's misfire rule (137-1 (3)): logged, not a stop
        MISFIRES.append(d); return False
    return fl


def rho_clause(cf, eps="0.1", floor=1e-3):
    """127-12's clause for a rounded carrier (here hard tokens): geomean over hand-offs with dist > 1e-3 and rho > 0 (>= 2), else rho_1 if rounding recovers."""
    try:
        e = cf["detail"]["eps"][eps]; rk, dk = e["rho_k"], e["dist_med_k"]
    except (KeyError, TypeError):
        return None
    use = [rk[k] for k in range(len(rk)) if k < len(dk) and dk[k] > floor and rk[k] > 0]
    if len(use) >= 2:
        return math.exp(sum(math.log(x) for x in use) / len(use))
    return rk[0] if rk and any(x <= floor for x in dk) else None


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
    """Row level where present (multpair writes digit-level + row_level per position): donor >= 0.90, recipient - chance < 0.10."""
    if not dn:
        return False
    out = []
    for v in dn.values():
        x = v.get("row_level", v)
        out.append(x["donor_consistent"] >= 0.90 and x["recipient_consistent"] - x.get("chance_overlap", 0.0) < 0.10)
    return all(out)


def slot(arm, k):
    att = []
    for o in OFFS:
        d = "s150_%s_4x4_s%d" % (arm, k + o)
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
            if not args_ok(d, arm):
                return {"attempts": att, "status": "error", "accepted": None}
            s0 = r.get("s150") or {}
            a = {"seed": k + o, "test_acc": r.get("test_acc"), "serial": serial(r.get("donor")), "rho": rho_clause(load(d, "eval_contraction_fp32.json")),
                 "per_column": (r.get("acc_by_column") or {}).get("per_column"), "ext_4x8": r.get("ext_4x8"), "probe": r.get("s150_probe"),
                 "acc_by_magnitude": s0.get("acc_by_magnitude"), "soft_test_acc": s0.get("soft_test_acc"), "slot_tokens": s0.get("slot_token_hist"),
                 "spells_accumulator": s0.get("spells_accumulator_share"), "distinct_tokens_per_slot": s0.get("distinct_tokens_per_slot"),
                 "slot_margin": s0.get("slot_margin"), "wte_drift": s0.get("wte_drift"),                 # note 150-1
                 "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver")}
            a["exact"] = (a["test_acc"] or 0) >= 0.99
            a["pass"] = None if a["rho"] is None else bool(a["exact"] and a["serial"] and a["rho"] <= 0.28)
            a["reading"] = "IN-PASS / shortcut, not a latent carry (exact without SERIAL)" if (a["exact"] and not a["serial"]) else None
            return {"attempts": att, "status": "accepted", "accepted": a}
    if len(att) == len(OFFS) and all(x["plateau"] for x in att):
        return {"attempts": att, "status": "exhausted", "accepted": None}
    return {"attempts": att, "status": "open", "accepted": None}


def arm_verdict(arm):
    sl = {k: slot(arm, k) for k in (1, 2, 3, 4)}
    s1 = sl[1]["accepted"]
    if s1 is not None and (s1["test_acc"] or 0) < 0.95:
        return sl, "dropped at proof (slot 1 test_acc < 0.95)"
    if not any(v["attempts"] or v["status"] != "open" for v in sl.values()):
        return sl, "not run"
    passes = [v["accepted"]["pass"] for v in sl.values() if v["accepted"]]
    decided = [v for v in sl.values() if v["status"] == "exhausted" or (v["accepted"] and v["accepted"]["pass"] is not None)]
    n = sum(1 for p in passes if p)
    if n >= 3:
        return sl, "PASS"
    if len(decided) - n >= 2:
        return sl, "FAIL"
    return sl, "incomplete (%d / 4 slots decided)" % len(decided)


def improves(a):
    pc = (a or {}).get("per_column")
    return None if not pc or len(pc) < 7 else all(x >= 0.9 for x in pc[2:7])


def main():
    res = {}
    for arm in ARMS:
        sl, v = arm_verdict(arm); res[arm] = {"slots": sl, "verdict": v}
        print(arm, v, {k: (s_["status"], s_["accepted"] and (s_["accepted"]["test_acc"], s_["accepted"]["serial"], s_["accepted"]["rho"])) for k, s_ in sl.items()})
        s1 = sl[1]["accepted"]
        if s1:
            print("   proof per column", s1["per_column"], "| §140 1 x FSQ-8", S140_PROOF, "| IMPROVES (cols 3-7 >= 0.9):", improves(s1))
            print("   hard vs soft at test:", s1["test_acc"], "/", s1["soft_test_acc"], "| spells accumulator:", s1["spells_accumulator"],
                  "| distinct tokens per slot:", s1["distinct_tokens_per_slot"], "| by magnitude:", s1["acc_by_magnitude"])
            print("   slot margin:", s1["slot_margin"], "| wte drift:", s1["wte_drift"])
        res[arm]["improves"] = improves(s1)
    vf, vd = res["full"]["verdict"], res["digit"]["verdict"]; i_f, i_d = res["full"]["improves"], res["digit"]["improves"]
    out = lambda v: v.startswith("dropped") or v.startswith("FAIL")  # noqa: E731
    if vd == "PASS":
        rd = "an abacus of digit tokens, never supervised, carries multiplication exactly (a token-shaped working memory works for numbers)"
    elif vf == "PASS":
        rd = "model-chosen full-vocabulary tokens carry multiplication exactly (C7: the state is K tokens)"
    elif i_d and not i_f:
        rd = "the restricted digit manifold improves the accumulator where full-vocabulary slots do not (exactness short at this budget)"
    elif out(vf) and out(vd):
        rd = "token-shaped slots do not fix in-pass computation at this budget (Literature's prediction)"
    else:
        rd = "pending"
    res["reading"] = rd; print("READING:", rd)
    if MISFIRES:
        print("2k-flag misfires (not stops):", MISFIRES); res["misfires"] = MISFIRES
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s150.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
