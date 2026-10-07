"""THEORY §136 scoring (rules fixed in §136; written before any §136 implementation or run; Theory is the only scorer).
GROW on §128's S5 recipe (fsq120 FINAL, FSQ 8 levels, RTX 3090 cloud, checkpoints, NO restarts and no plateau stop).
Arms (dirs boxes/runs_fsq/s136_{arm}_s5_s{1..4}; nested: s136_nested_s5_s1 only):
  growp     plain base, d 8 -> 64 once at PLATEAU (non-zero loss: mean over (t-500, t] >= 0.5 and >= 0.98 x mean over (t-1000, t-500], t >= 4,000)
  cgrow     §134 COPY base, d 8 -> 64 once at THRESHOLD (first logged step >= 2,000 with main-chain loss <= 2.5)
  cstaged   COPY base, 8 -> 16 at THRESHOLD, then x2 (32, 64) whenever >= 1,000 steps since the last widening and loss <= 2.5
  cnoise    cgrow with the 56 new dims continuous + dither U(+-0.25) (first 8 stay hard FSQ)
  grow4     plain base, d 4 -> 64 once at THRESHOLD (entry at d 4 vs d 8's 1/4)
  nested    d 64 from step 0, a random prefix width per training chain from {4, 8, 16, 32, 64} (one run, descriptive)
PASS(arm): all four seeds exact at 2,048 (ext_2048 >= 0.99), SERIAL (donor >= 0.90, recipient - chance < 0.10) and contractive
(rho <= 0.28: 127-12's clause as in score_s134 for grow / staged / growcopy / nested; 133-1's first-3 geomean (rho_k3) for grownoise).
IMPROVES (descriptive) 3/4; FAIL <= 2/4.
Descriptive: ENTRY at d 4 (seeds whose trigger fired; grow / staged / grownoise share the pre-trigger trajectory) vs FSQ-8's 1/4;
GROWTH-SAFE (grown runs end exact; switch spike = max loss over the 1,000 steps after a widening minus the loss at the switch record);
DIMENSION USE (eval.json s136: new-dim variance share, new-dim probe R2_state, effective rank); prefix 1 / 2 / 4 / 8; codes per state.
The run must log each widening as a record {"step": t, "grow": {"from": d, "to": d2}}.
    python theory/score_s136.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
ARMS = ("growp", "cgrow", "cstaged", "cnoise", "grow4", "nested")
EXPECT = {"growp": {"d_q": 8, "grow": "8:64", "grow_trigger": "plateau"},
          "cgrow": {"d_q": 8, "grow": "8:64", "grow_trigger": 2.5, "copy_frac": 0.1},
          "cstaged": {"d_q": 8, "grow": "8:64", "grow_trigger": 2.5, "grow_staged": True, "copy_frac": 0.1},
          "cnoise": {"d_q": 8, "grow": "8:64", "grow_trigger": 2.5, "grow_noise": 0.25, "copy_frac": 0.1},
          "grow4": {"d_q": 4, "grow": "4:64", "grow_trigger": 2.5},
          "nested": {"d_q": 64, "nested": True}}
SEEDS = {"nested": (1,)}
SKIPPED, ERRORS = [], []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def log_recs(d):
    p = os.path.join(R, d, "log.jsonl")
    return list(map(json.loads, open(p))) if os.path.exists(p) else []


def rho_clause(cf, eps="0.1", floor=1e-3):
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
    return bool(dn) and all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] - v.get("chance_overlap", 0.0) < 0.10 for v in dn.values())


def bad_args(d, arm):
    a = load(d, "args.json") or {}
    if int(a.get("steps", 24000)) != 24000 or float(a.get("sched_scale", 1.0)) != 1.0 or str(a.get("device", "")).startswith("cpu") or a.get("mem_check"):
        SKIPPED.append(d); return "smoke"
    exp = dict(EXPECT[arm], levels=8, family="s5", p_hard_start=2000, p_hard_end=4000)
    def same(k, v):                                                         # grow_trigger may be stored as "2.5" or 2.5
        x = a.get(k, v if k == "levels" else None)
        return str(x) == str(v) if k == "grow_trigger" else x == v
    bad = {k: a.get(k) for k, v in exp.items() if not same(k, v)}
    if a.get("stop_plateau"):
        bad["stop_plateau"] = True
    for k in ("grow_staged", "copy_frac", "grow_noise", "nested"):            # one change per arm: the others' flags must be off
        if k not in EXPECT[arm] and a.get(k):
            bad[k] = a.get(k)
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
    recs = log_recs(d)
    L = {x["step"]: x["loss"] for x in recs if "loss" in x and "step" in x}
    grows = [(x["step"], x["grow"]) for x in recs if "grow" in x]
    spikes = []
    for st, _ in grows:
        if L:
            base = L[min(L, key=lambda s: abs(s - st))]
            after = [v for s, v in L.items() if st < s <= st + 1000]
            spikes.append(round(max(after) - base, 3) if after else None)
    cf = load(d, "eval_contraction_fp32.json")
    out = {"long_acc": r.get("long_acc"), "ext_2048": (r.get("ext_2048") or {}).get("cumulative"), "serial": serial(r.get("donor")),
           "rho": (rho_k3 if arm == "cnoise" else rho_clause)(cf), "grows": grows, "switch_spikes": spikes,
           "entered": bool(grows) if arm in ("cgrow", "cstaged", "cnoise", "grow4") else None, "width_final": (r.get("s136") or {}).get("width_final"),
           "dim_use": {k: (r.get("s136") or {}).get(k) for k in ("new_dim_var_share", "new_dim_r2_state", "eff_rank", "acc_w8")},
           "codes_per_state": r.get("codes_per_state"), "prefix": (load(d, "eval_prefix.json") or {}).get("acc_by_prefix"),
           "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver")}
    out["exact"] = (out["ext_2048"] or 0) >= 0.99
    out["pass"] = None if out["rho"] is None else bool(out["exact"] and out["serial"] and out["rho"] <= 0.28)
    return out


def verdict(runs, arm=None):
    errs = [s for s, v in runs.items() if v and "error" in v]
    done = {s: v for s, v in runs.items() if v and "error" not in v}
    if errs:
        return "invalid runs %s" % errs
    if arm == "nested":                                                      # one run, descriptive
        v = done.get(1)
        return "pending" if not v or v["pass"] is None else ("exact + SERIAL + contractive (descriptive)" if v["pass"] else "not passing (descriptive)")
    if len(done) < 4 or any(v["pass"] is None for v in done.values()):
        return "incomplete (%d / 4)" % len(done)
    n = sum(1 for v in done.values() if v["pass"])
    return "PASS" if n == 4 else ("IMPROVES (3/4, descriptive)" if n == 3 else "FAIL (%d / 4)" % n)


def main():
    res = {}
    for arm in ARMS:
        runs = {s: run("s136_%s_s5_s%d" % (arm, s), arm) for s in SEEDS.get(arm, (1, 2, 3, 4))}
        v = verdict(runs, arm); res[arm] = {"runs": runs, "verdict": v}
        for s, x in runs.items():
            print(arm, "s%d" % s, x and {k: x.get(k) for k in ("ext_2048", "serial", "rho", "pass", "entered", "grows", "switch_spikes", "width_final", "dim_use", "prefix", "error")})
        print("VERDICT", arm, v)
    ent = {s: (res["grow4"]["runs"].get(s) or {}).get("entered") for s in (1, 2, 3, 4)}
    res["entry_d4"] = ent; print("ENTRY at d 4 (grow4's THRESHOLD fired; plain d 8 comparator 1/4 on these seeds):", ent)
    fired = {s: (res["growp"]["runs"].get(s) or {}).get("grows") for s in (1, 2, 3, 4)}
    res["growp_fired"] = fired; print("GROW-P plateau widenings (rescue attempts):", fired)
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s136.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
