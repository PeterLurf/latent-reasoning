"""THEORY §143 scoring (rules fixed in §143; written before any §143 implementation or run; Theory is the only scorer).
FULL-GROW, §138's one revised method: §136 COPY+GROW-NOISE exactly (fsq120 S5 FINAL, COPY rows f 0.1 from step 0, FSQ d 8, no restarts, THRESHOLD
8 -> 64 at the first logged step >= 2,000 with loss <= 2.5, the 56 new dims continuous + dither U(+-0.25)), then a SECOND widening 64 -> 768 at the first
logged step >= 6,000 that is >= 1,000 steps after the first and has loss <= 2.5; the 704 new dims continuous + dither U(+-0.25); new W_up columns at 0,
500-step fade-in, fresh Adam group with a 300-step warm-up and lr x 64/768 (muP-style; Literature 9dc25047). Dirs boxes/runs_fsq/s143_fullgrow_s5_s{1..4}.
Each seed is identical to s136_cnoise_s5_s{seed} up to the second widening (same seed, base and card class): the comparison is paired per seed.
PASS: >= 3/4 seeds exact at 2,048 (ext_2048 >= 0.99), SERIAL (donor >= 0.90, recipient - chance < 0.10) and contractive: the E-SPACE contraction (fp32
contraction() perturbs all 768 z_b coordinates as usual, but every distance is measured as ||delta e|| with e = W_up(z): eval_contraction_fp32.json detail.eps
["0.1"].rho_k_e / dist_med_k_e), 133-1's first-3 form, <= 0.28 (Audit / Literature 5b22bd7f: a z-space ratio is deflated while W_up_new ~ 0). Descriptive
beside it: the z-space rho_k3, the 64-block e-space rho (s143.rho_block64_e) and the Jacobian top (fp32 power iteration).
A seed whose second widening never fires never reaches full width and does not pass. DROP at proof: s1 ext_2048 < 0.95 stops the section.
Readings per seed (pre-stated; Literature 5b22bd7f): LOCATED at 64 -> 768 = exact right before the second switch (grow.json pre_eval acc_512 >= 0.99) but
not exact at the end; INHERITS FULL-CONT's expansion = not passing AND (end Jacobian top rho > 1 OR end carried-embedding norm ||e|| = ||W_up z|| (carried_norm
row_norm, function-preserved at the switch) >= 2 x its value right before the second switch); AVOIDS it = passing AND end Jacobian top rho <= 1 (the
e-space rule already holds). A pass reads "stable at 768", not a benefit. Pairing: loss records identical to s136_cnoise up to switch 2 on the same 3090 /
driver; a mismatch makes the comparison unpaired (stated; readings use the run's own pre-switch-2 state). If the section fails, full width stops (the
user's rule): 64-d stays the default, no further fixes.
    python theory/score_s143.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
EXPECT = {"family": "s5", "d_q": 8, "levels": 8, "grow": "8:64", "grow_noise": 0.25, "copy_frac": 0.1, "p_hard_start": 2000, "p_hard_end": 4000,
          "grow2": 768, "grow2_min_step": 6000, "grow2_lr_scale": round(64 / 768, 4)}
SKIPPED, ERRORS = [], []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def log_recs(d):
    p = os.path.join(R, d, "log.jsonl")
    return list(map(json.loads, open(p))) if os.path.exists(p) else []


def rho_k3(cf, eps="0.1", space=""):
    try:
        e = cf["detail"]["eps"][eps]; rk, dk = e["rho_k" + space], e["dist_med_k" + space]
    except (KeyError, TypeError):
        return None
    use = [rk[k] for k in range(min(3, len(rk), len(dk))) if dk[k] > 1e-2 and rk[k] > 0]
    if not use:
        return rk[0] if rk else None
    return math.exp(sum(math.log(x) for x in use) / len(use))


def serial(dn):
    return bool(dn) and all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] - v.get("chance_overlap", 0.0) < 0.10 for v in dn.values())


def bad_args(d):
    a = load(d, "args.json") or {}
    if int(a.get("steps", 24000)) != 24000 or float(a.get("sched_scale", 1.0)) != 1.0 or str(a.get("device", "")).startswith("cpu") or a.get("mem_check"):
        SKIPPED.append(d); return "smoke"
    def same(k, v):
        x = a.get(k)
        if k == "grow_trigger":
            return str(x) == str(v)
        if isinstance(v, float) and x is not None:
            return abs(float(x) - v) < 1e-3
        return x == v
    exp = dict(EXPECT, grow_trigger=2.5)
    bad = {k: a.get(k) for k, v in exp.items() if not same(k, v)}
    for k in ("stop_plateau", "grow_staged", "nested", "no_round", "soft_beta", "hinge"):
        if a.get(k):
            bad[k] = a.get(k)
    if bad:
        ERRORS.append((d, bad)); return "args mismatch %s" % bad
    return None


def paired_identity(d, seed, t2):
    """Max |loss difference| against s136_cnoise_s5_s{seed} over the logged steps before the second widening (identity expected)."""
    a = {x["step"]: x["loss"] for x in log_recs(d) if "loss" in x and "step" in x}
    b = {x["step"]: x["loss"] for x in log_recs("s136_cnoise_s5_s%d" % seed) if "loss" in x and "step" in x}
    common = [s for s in a if s in b and (t2 is None or s < t2)]
    return round(max(abs(a[s] - b[s]) for s in common), 6) if common else None


def run(seed):
    d = "s143_fullgrow_s5_s%d" % seed
    r = load(d)
    if r is None:
        return None
    b = bad_args(d)
    if b:
        return {"error": b}
    recs = log_recs(d)
    L = {x["step"]: x["loss"] for x in recs if "loss" in x and "step" in x}
    grows = [(x["step"], x["grow"]) for x in recs if "grow" in x]
    spikes = []
    for st, _ in grows:
        after = [v for s, v in L.items() if st < s <= st + 1000]
        spikes.append(round(max(after) - L[min(L, key=lambda s: abs(s - st))], 3) if (after and L) else None)
    t2 = next((st for st, g in grows if int(g.get("to", 0)) == 768), None)
    gj = load(d, "grow.json") or {}
    ev2 = next((e for e in gj.get("events", []) if int(e.get("to", 0)) == 768), {})
    pre = ev2.get("pre_eval") or {}
    s138 = r.get("s138") or {}
    cn_end = ((r.get("carried_norm") or {}).get("end") or {}).get("row_norm")
    cf = load(d, "eval_contraction_fp32.json")
    out = {"ext_2048": (r.get("ext_2048") or {}).get("cumulative"), "serial": serial(r.get("donor")), "rho": rho_k3(cf, space="_e"), "rho_z_k3": rho_k3(cf),
           "grows": grows, "switch_spikes": spikes, "second_switch": t2, "width_final": (r.get("s136") or {}).get("width_final"),
           "pre_switch2": {k: pre.get(k) for k in ("acc_512", "rho_k3", "rho_dir_onestep_median", "rho_jacobian_power_median", "carried_norm")},
           "end": {"rho_dir_onestep_median": s138.get("rho_dir_onestep_median"), "rho_jacobian_power_median": s138.get("rho_jacobian_power_median"),
                   "carried_norm": cn_end, "eff_rank": s138.get("eff_rank"), "spread_over_between": s138.get("spread_over_between"),
                   "new_block_var_share": (r.get("s143") or {}).get("block704_var_share"), "new_block_r2_state": (r.get("s143") or {}).get("block704_r2_state"),
                   "w_up_new_norm": (r.get("s143") or {}).get("w_up_new_norm"), "rho_by_block": (r.get("s143") or {}).get("rho_by_block"),
                   "rho_block64_e": (r.get("s143") or {}).get("rho_block64_e"),
                   "w_up_new_out_share": (r.get("s143") or {}).get("w_up_new_out_share")},
           "paired_max_loss_diff_vs_s136_cnoise": paired_identity(d, seed, t2), "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver")}
    ab = (r.get("s143") or {}).get("ablate704")                              # note 143-2: zero the new block's W_up output at test
    if ab:
        ab_ext = (ab.get("ext_2048") or {}).get("cumulative") if isinstance(ab.get("ext_2048"), dict) else ab.get("ext_2048")
        out["ablate704"] = {"ext_2048": ab_ext, "serial": serial(ab.get("donor")),
                            "width": "UNUSED (works through the 64 original dims)" if ((ab_ext or 0) >= 0.99 and serial(ab.get("donor"))) else "USED"}
    out["exact"] = (out["ext_2048"] or 0) >= 0.99
    full = t2 is not None and out["width_final"] == 768
    out["pass"] = None if out["rho"] is None else bool(full and out["exact"] and out["serial"] and out["rho"] <= 0.28)
    jac = out["end"]["rho_jacobian_power_median"]; cn0 = out["pre_switch2"]["carried_norm"]
    if out["pass"] is False:
        if not full:
            out["reading"] = "never reached full width (second trigger did not fire)"
        else:
            loc = (pre.get("acc_512") or 0) >= 0.99
            inh = (jac is not None and jac > 1) or (cn_end is not None and cn0 and cn_end >= 2 * cn0)
            out["reading"] = "; ".join(x for x in ("LOCATED at the 64 -> 768 step" if loc else "not exact before the second switch",
                                                    "INHERITS FULL-CONT's expansion" if inh else "") if x)
    elif out["pass"]:
        out["reading"] = ("AVOIDS FULL-CONT's expansion (stable at 768)" if (jac is not None and jac <= 1)
                          else "passes; AVOIDS not shown (Jacobian top %s)" % jac)
    return out


def main():
    runs = {s: run(s) for s in (1, 2, 3, 4)}
    s1 = runs[1]
    if s1 and "error" not in s1 and (s1.get("ext_2048") or 0) < 0.95:
        v = "DROP at proof (s1 ext_2048 %s < 0.95): full width stops; 64-d stays the default (no further fixes)" % s1.get("ext_2048")
    elif any(x and "error" in x for x in runs.values()):
        v = "invalid runs %s" % [s for s, x in runs.items() if x and "error" in x]
    else:
        dec = {s: x for s, x in runs.items() if x and x.get("pass") is not None}
        n = sum(1 for x in dec.values() if x["pass"])
        if n >= 3:
            v = "PASS: full width reached by staged growth (joins S5x4 as the full-width arm)"
        elif len(dec) - n >= 2:
            v = "FAIL (%d / 4): full width stops; 64-d stays the default (no further fixes)" % n
        else:
            v = "incomplete (%d / 4 decided)" % len(dec)
    for s, x in runs.items():
        print("s%d" % s, x and {k: x.get(k) for k in ("ext_2048", "serial", "rho", "rho_z_k3", "pass", "second_switch", "switch_spikes", "reading", "ablate704",
                                                     "paired_max_loss_diff_vs_s136_cnoise", "error")})
        if x and "error" not in x:
            print("    pre-switch-2", x["pre_switch2"], "| end", x["end"])
    print("VERDICT", v)
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump({"runs": runs, "verdict": v}, open(os.path.join(ROOT, "theory", "results", "score_s143.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
