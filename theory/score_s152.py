"""THEORY §152 scoring (rules fixed in §152; written before any §152 implementation or run; Theory is the only scorer).
DELTA-READ: multiplier-major 4x4 (multrow --family mult_row) with §147's DeltaMem branch as the ONLY cross-segment channel (--carrier deltamem; no carried code token);
a side-branch reader / decoder reads the memory (writes nothing back). Arms (dirs boxes/runs_fsq/s152_{arm}_4x4_s{seed}):
  a       READ: the reader decodes the teacher's S_j (cache; label_source teacher_generated) at every hand-off; 48k; removal gate (L 4, train exact >= 0.95 on 512)
  bearly  forked from a's gate.pt, fixed_L 4, S_1 / S_2 / S_3 removed at +0 / +4k / +8k, final only to +20k
  ctrl    a without the reader (FINAL only), s1
PASS(arm): >= 3/4 seeds test_acc >= 0.99 + SERIAL (the whole memory state swapped after segment 2; row level). No contraction term (integrator by design).
DROP at proof: a s1 < 0.95. HEADLINE (the user's standard, C3 + C1): LATENT BEATS TEXT = a passing arm with median test_acc >= TC-M - 0.01 and FEWER FLOPs and FEWER
sequential passes per item than TC-M (1.000; 1.55e10; 57), with SERIAL; ICoT-SI (0.998; 4.93e9; 13) reported beside it (descriptive).
    python theory/score_s152.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
ARMS = ("a", "bearly", "ctrl")
TCM = {"acc": 1.0, "flops": 15459259392.0, "passes": 57}; ICOT = {"acc": 0.998, "flops": 4926282240.0, "passes": 13}   # §98 (m98_tc_4x4_s1 / score_s98)
SEEDS = {"a": (1, 2, 3, 4), "bearly": (1, 2, 3, 4), "ctrl": (1,)}
BASE = {"family": "mult_row", "task": "4x4", "carrier": "deltamem", "mem_add_heads": 6, "mem_delta_heads": 6, "mem_group_gain": True, "lmax_steps": "2000,5000,9000"}
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


def bad_args(d, arm, seed):
    a = load(d, "args.json") or {}
    n = 20000 if arm == "bearly" else 48000
    k = "fork_steps" if arm == "bearly" else "steps"
    if int(a.get(k, 0) or 0) != n or float(a.get("sched_scale", 1.0)) != 1.0 or str(a.get("device", "")).startswith("cpu") or a.get("mem_check") or a.get("train_n"):
        SKIPPED.append(d); return "smoke"
    exp = dict(BASE, reader=(arm != "ctrl"), seg_order="b_first")                # note 152-1 (2): [EOT, b_j, j, a's digits]
    if arm == "bearly":
        exp.update({"removal": arm[1:], "removal_every": 4000, "rewarm": 200, "fork_from": "boxes/runs_fsq/s152_a_4x4_s%d/gate.pt" % seed,
                    "no_opt_reset": True})                                  # §152: AdamW is NOT reset at removal (Literature (2)); LR re-warm only
    bad = {k_: a.get(k_) for k_, v in exp.items() if a.get(k_) != v}
    if a.get("no_round") or a.get("dither") or float(a.get("copy_frac") or 0) != 0:      # §152: DeltaMem is the carrier (no code token)
        bad["code_carrier_flags"] = (a.get("no_round"), a.get("dither"), a.get("copy_frac"))
    if arm in ("a", "ctrl") and a.get("removal"):
        bad["removal"] = a.get("removal")
    if arm != "ctrl" and a.get("label_source") != "teacher_generated":                 # C4 as reworded: the teacher's outputs, never gold S_j
        bad["label_source"] = a.get("label_source")
    if a.get("stop_plateau"):
        bad["stop_plateau"] = True
    if int(a.get("fixed_L") or 0) != (4 if arm == "bearly" else 0):        # B: every fork row is 4x4 (no prefix problems re-supervising S_j)
        bad["fixed_L"] = a.get("fixed_L")
    if bad:
        ERRORS.append((d, bad)); return "args mismatch %s" % bad
    return None


def run(arm, seed):
    d = "s152_%s_4x4_s%d" % (arm, seed)
    if arm == "bearly":
        ra = load("s152_a_4x4_s%d" % seed)
        if ra is not None and (ra.get("s152") or {}).get("gate_step") is None:
            return {"status": "not reached", "pass": False}                             # a finished without opening the gate: B is not run
    r = load(d)
    if r is None:
        return None
    b = bad_args(d, arm, seed)
    if b:
        return {"status": "error", "error": b}
    s = r.get("s152") or {}; dn = (r.get("donor") or {}).get("pos2") or {}
    out = {"status": "done", "test_acc": r.get("test_acc"), "serial": serial(r.get("donor")), "rho": rho_k3(load(d, "eval_contraction_fp32.json")),
           "acc_by_length": r.get("acc_by_length"), "train_acc": r.get("train_acc"), "n_nonfinite": r.get("n_nonfinite"),
           "donor": {k: dn.get(k) for k in ("donor_consistent", "recipient_consistent", "chance_overlap", "bprefix_consistent", "operands_consistent")},
           "serial_shares": {"carry_save": dn.get("donor_consistent"), "bprefix": dn.get("bprefix_consistent"), "recompute": dn.get("recipient_consistent")},   # Literature (4)
           "add_state_norm_by_segment": s.get("add_state_norm_by_segment"), "bprefix_probe_after_removal": s.get("bprefix_probe"),
           "pp_probe": s.get("pp_probe"),                                                                                       # note 152-1 (3)
           "sj_decode_acc": s.get("sj_decode_acc"), "head_ablation": s.get("head_ablation"), "alpha_by_head": s.get("alpha_by_head"),
           "colsum_readout": s.get("colsum_readout"),                                                                           # §152 descriptive
           "probe_r2": s.get("probe_r2"), "ext_4x8_low8": r.get("ext_4x8_low8"), "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver")}
    if arm == "a":
        out.update({"gate_step": s.get("gate_step"), "gate_train_exact": s.get("gate_train_exact"),
                    "teacher_vs_gold_mismatch": s.get("teacher_vs_gold_mismatch_share"), "teacher_invalid": s.get("teacher_invalid_share")})
    if arm == "bearly":
        fs = s.get("fork_start") or {}
        out["serial_before"] = serial(fs.get("donor")); out["test_acc_before"] = fs.get("test_acc")
        out["retention"] = [{"step": st.get("step"), "removed": st.get("removed"), "acc_by_length": st.get("acc_by_length"),
                             "test_acc": st.get("test_acc"), "serial": serial(st.get("donor")), "sj_decode_acc": st.get("sj_decode_acc"),
                             "final_ce_before_after": (st.get("final_ce_before"), st.get("final_ce_after"))} for st in (s.get("stages") or [])]
    c = r.get("compute") or {}
    out["compute"] = {"flops": c.get("flops"), "passes": c.get("sequential_passes")}
    out["serial_by_ckpt"] = s.get("serial_by_ckpt")
    out["exact"] = (out["test_acc"] or 0) >= 0.99
    out["pass"] = bool(out["exact"] and out["serial"])                           # §152: no contraction term (integrator)
    out["reading"] = "IN-PASS / shortcut, not a carried S_j (exact without SERIAL)" if (out["exact"] and not out["serial"]) else None
    return out


def verdict(arm, runs):
    if any(v and v.get("status") == "error" for v in runs.values()):
        return "invalid runs %s" % [s for s, v in runs.items() if v and v.get("status") == "error"]
    if arm == "ctrl":
        v = runs.get(1)
        return "pending" if not v or v.get("status") != "done" else "descriptive: test_acc %s" % v.get("test_acc")
    decided = {s: v for s, v in runs.items() if v and v.get("pass") is not None}
    n = sum(1 for v in decided.values() if v["pass"])
    if n >= 3:
        return "PASS"
    if len(decided) - n >= 2:
        return "FAIL (%d / 4)" % n
    return "incomplete (%d / 4 decided)" % len(decided)


def headline(runs):
    """LATENT BEATS TEXT (the user's standard): passing seeds' median acc >= TC-M - 0.01, fewer FLOPs and fewer sequential passes than TC-M, SERIAL."""
    ok = [v for v in runs.values() if v and v.get("pass")]
    if not ok:
        return None
    accs = sorted(v["test_acc"] for v in ok); fl = sorted(v["compute"]["flops"] for v in ok if v["compute"]["flops"]); ps = sorted(v["compute"]["passes"] for v in ok if v["compute"]["passes"])
    med = lambda x: x[len(x) // 2] if x else None  # noqa: E731
    m_acc, m_fl, m_ps = med(accs), med(fl), med(ps)
    beats = bool(m_acc is not None and m_fl is not None and m_ps is not None and m_acc >= TCM["acc"] - 0.01 and m_fl < TCM["flops"] and m_ps < TCM["passes"])
    return {"median_acc": m_acc, "median_flops": m_fl, "median_passes": m_ps, "flops_vs_tcm": None if not m_fl else m_fl / TCM["flops"],
            "flops_vs_icotsi": None if not m_fl else m_fl / ICOT["flops"], "LATENT_BEATS_TEXT": beats}


def main():
    res = {"arms": {}, "bars": {"tc_m": TCM, "icot_si": ICOT}}
    a1 = run("a", 1); c1r = run("ctrl", 1)
    c1_drop = bool(c1r and c1r.get("status") == "done" and (c1r.get("test_acc") or 0) < 0.95)     # note 152-1 (1): CTRL "drops" = below the proof bar
    if a1 and a1.get("status") == "done" and (a1.get("test_acc") or 0) < 0.95:
        res["verdict"] = "DROP at proof (a s1 test_acc %s < 0.95): not acquired within budget" % a1.get("test_acc")
        if c1_drop:
            res["verdict"] += ("; CTRL s1 %s also < 0.95: the multiplier-major layout's in-pass a x b_j is the block, not 'DeltaMem + reader fails' "
                               "(note 152-1 (1)); pp_probe locates it" % c1r.get("test_acc"))
        elif c1r and c1r.get("status") == "done":
            res["verdict"] += "; CTRL s1 %s >= 0.95 (descriptive: the reader's targets impede)" % c1r.get("test_acc")
        res["arms"]["a"] = {"runs": {1: a1}}; res["arms"]["ctrl"] = {"runs": {1: c1r}}
        print("PROOF:", res["verdict"]); print("a s1", {k: a1.get(k) for k in ("test_acc", "acc_by_length", "serial", "gate_step", "donor", "compute")})
    else:
        for arm in ARMS:
            runs = {s_: run(arm, s_) for s_ in SEEDS[arm]}
            v = verdict(arm, runs); hl = headline(runs) if v == "PASS" else None
            res["arms"][arm] = {"runs": runs, "verdict": v, "headline": hl}
            for s_, x in runs.items():
                print(arm, "s%d" % s_, x and {k: x.get(k) for k in ("status", "test_acc", "serial", "pass", "gate_step", "serial_before", "compute", "reading", "error")})
                for st in (x or {}).get("retention") or []:
                    print("   retention", st)
            print("VERDICT", arm, v, "| HEADLINE", hl)
        va, vb = res["arms"]["a"]["verdict"], res["arms"]["bearly"]["verdict"]
        hb = (res["arms"]["bearly"].get("headline") or {}).get("LATENT_BEATS_TEXT")
        if vb == "PASS" and hb:
            claim = ("B-EARLY PASS + LATENT BEATS TEXT: exact multiplication through a latent memory that carries the computation (SERIAL), final-answer only after "
                     "CoT-guided training, at fewer FLOPs and sequential passes than TC-M (the user's headline)")
        elif va == "PASS" and vb.startswith("FAIL"):
            claim = "A PASS, B-EARLY FAIL: removal loses the skill (retention says which S_j goes first)"
        elif va == "PASS":
            claim = "A PASS (B-EARLY %s)" % vb
        elif va.startswith("FAIL"):
            claim = "A FAIL: not acquired within budget"
        else:
            claim = "pending (A %s)" % va
        c1 = (res["arms"]["ctrl"]["runs"].get(1) or {}).get("test_acc")
        if c1 is not None:
            claim += "; CTRL (no reader) %.3f" % c1
            if c1_drop and (a1 or {}).get("status") == "done":
                claim += " < 0.95 while A s1 >= 0.95: the CoT reader is what makes multiplier-major learnable (note 152-1 (1))"
        res["verdict"] = claim; print("READING:", claim)
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s152.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
