"""THEORY §153 scoring (rules fixed in §153; written before any §153 implementation or run; Theory is the only scorer).
THINK-PASS (program B5): §152's recipe (DeltaMem as the ONLY cross-pass channel, the read-only reader / decoder pass, the teacher's S_j targets, earliest-first
removal without an AdamW reset) with item-agnostic passes (multrow --carrier deltamem --pass_input visible|pure|repeat):
  visible  pass j = [EOT, a's digits, " ×", b's digits, step token j]: the whole problem every pass; the model picks b_j itself
  pure     pass 0 = [EOT, a's digits, " ×", b's digits] (writes only); passes j = [EOT, step token j, 4 register tokens " _"]: the memory holds operands + accumulator
  repeat   stage 2 (only if PURE passes at 4x4): pure with ONE repeated think token (no step index), so the memory keeps its own pointer
Arms per input (dirs boxes/runs_fsq/s153_{input}_{arm}_4x4_s{seed}): a (reader; gate 0.95), bearly (fork of a's gate.pt; removal early, no_opt_reset), ctrl (final only, s1).
PASS(input, arm): >= 3/4 seeds with test_acc >= 0.99 and SERIAL.
  visible: §152's swap (the whole memory from a donor after pass 2; donors a_donor != a_rec, different b-prefix): carry-save >= 0.90, recompute - chance < 0.10.
  pure / repeat: the memory also holds the operands, so a whole-memory swap moves the problem and cannot discriminate; SERIAL = the pass-skip test (C1's step-specific
  blocking): pass 2 and pass 3 each skipped (memory passed through unchanged): row-skip-consistent (S_4 minus that row's a x b_j) >= 0.90 and full product < 0.10,
  over items with b_j != 0 (note 153-1; the step-j pass not run, later passes keep their own step tokens).
DROP at proof: a s1 < 0.95 (per input). Placement gate: §152 a s1 >= 0.95. HEADLINE as §152 (LATENT BEATS TEXT vs TC-M; ICoT-SI beside it).
Predictions (§153, stated before any run): (i) VISIBLE's recompute share > §152 A's; (ii) PURE's ext_4x5 (K = 5, untrained step key) fails (< 0.5).
ext_4x5.test_acc = S_5 mod 1e8 exact (note 153-2: the reader emits 8 digits); times token " ×" (13958), register token " _" (args times_token / register_token).
Stage 2 (REPEAT): C2 = ext_4x5 test_acc >= 0.9 at K = 5 in every passing seed; no-op stability: 4x4 test_acc at K' = 6 and 8 stays >= 0.99.
    python theory/score_s153.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s152 as S152  # noqa: E402  (§152's bars, loader, headline and SERIAL rule; §153's gate reads §152 a s1)

R = os.path.join(ROOT, "boxes", "runs_fsq")
INPUTS = ("visible", "pure", "repeat")
ARMS = ("a", "bearly", "ctrl")
SEEDS = {"a": (1, 2, 3, 4), "bearly": (1, 2, 3, 4), "ctrl": (1,)}
BASE = {"family": "mult_row", "task": "4x4", "carrier": "deltamem", "mem_add_heads": 6, "mem_delta_heads": 6, "mem_group_gain": True, "lmax_steps": "2000,5000,9000"}
SKIPPED, ERRORS = [], []
load = S152.load


def serial_swap(r):
    return S152.serial({"pos2": ((r.get("donor") or {}).get("pos2"))} if (r.get("donor") or {}).get("pos2") else None)


def serial_skip(r, d=None):
    """pure / repeat (note 153-1): pass 2 and pass 3 each skipped (the step-j pass not run; later passes keep their own step tokens); over items with b_j != 0
    (excludes_bj0 true; n_bj0_excluded reported): row_skip_consistent >= 0.90 and full_product < 0.10 at both skips."""
    sk = r.get("skip") or {}
    if not all(k in sk for k in ("pass2", "pass3")):
        return False
    if not all(sk[k].get("excludes_bj0") is True for k in ("pass2", "pass3")):
        ERRORS.append((d, "skip fields not computed over b_j != 0 (note 153-1)")); return False
    return all(sk[k]["row_skip_consistent"] >= 0.90 and sk[k]["full_product"] < 0.10 for k in ("pass2", "pass3"))


def bad_args(d, inp, arm, seed):
    a = load(d, "args.json") or {}
    n = 20000 if arm == "bearly" else 48000
    k = "fork_steps" if arm == "bearly" else "steps"
    if int(a.get(k, 0) or 0) != n or float(a.get("sched_scale", 1.0)) != 1.0 or str(a.get("device", "")).startswith("cpu") or a.get("mem_check") or a.get("train_n"):
        SKIPPED.append(d); return "smoke"
    exp = dict(BASE, reader=(arm != "ctrl"), pass_input=inp)
    if arm == "bearly":
        exp.update({"removal": "early", "removal_every": 4000, "rewarm": 200, "no_opt_reset": True,
                    "fork_from": "boxes/runs_fsq/s153_%s_a_4x4_s%d/gate.pt" % (inp, seed)})
    bad = {k_: a.get(k_) for k_, v in exp.items() if a.get(k_) != v}
    if a.get("no_round") or a.get("dither") or float(a.get("copy_frac") or 0) != 0:
        bad["code_carrier_flags"] = (a.get("no_round"), a.get("dither"), a.get("copy_frac"))
    if arm in ("a", "ctrl") and a.get("removal"):
        bad["removal"] = a.get("removal")
    if arm != "ctrl" and a.get("label_source") != "teacher_generated":
        bad["label_source"] = a.get("label_source")
    if a.get("stop_plateau"):
        bad["stop_plateau"] = True
    if int(a.get("fixed_L") or 0) != (4 if arm == "bearly" else 0):
        bad["fixed_L"] = a.get("fixed_L")
    if inp == "repeat" and not a.get("think_token"):
        bad["think_token"] = None
    if bad:
        ERRORS.append((d, bad)); return "args mismatch %s" % bad
    return None


def run(inp, arm, seed):
    d = "s153_%s_%s_4x4_s%d" % (inp, arm, seed)
    if arm == "bearly":
        ra = load("s153_%s_a_4x4_s%d" % (inp, seed))
        if ra is not None and (ra.get("s153") or {}).get("gate_step") is None:
            return {"status": "not reached", "pass": False}
    r = load(d)
    if r is None:
        return None
    b = bad_args(d, inp, arm, seed)
    if b:
        return {"status": "error" if b != "smoke" else "smoke", "error": b}
    s = r.get("s153") or {}; dn = (r.get("donor") or {}).get("pos2") or {}
    ser = serial_swap(r) if inp == "visible" else serial_skip(r, d)
    out = {"status": "done", "test_acc": r.get("test_acc"), "serial": ser, "serial_rule": "swap" if inp == "visible" else "pass-skip",
           "acc_by_length": r.get("acc_by_length"), "train_acc": r.get("train_acc"), "n_nonfinite": r.get("n_nonfinite"),
           "serial_shares": {"carry_save": dn.get("donor_consistent"), "bprefix": dn.get("bprefix_consistent"), "recompute": dn.get("recipient_consistent"),
                             "chance": dn.get("chance_overlap")},
           "skip": r.get("skip"), "ext_4x5": (r.get("ext_4x5") or {}).get("test_acc"), "noop": s.get("noop"),
           "attn_bj_share": s.get("attn_bj_share"), "operand_recall_pass1": s.get("operand_recall_pass1"), "bj_recall_by_pass": s.get("bj_recall_by_pass"),
           "head_ablation": s.get("head_ablation"), "head_roles": s.get("head_roles"), "mem_norm_by_pass": s.get("mem_norm_by_pass"),
           "sj_decode_acc": s.get("sj_decode_acc"), "reader_text": s.get("reader_text"), "serial_by_ckpt": s.get("serial_by_ckpt"),
           "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver")}
    if arm == "a":
        out.update({"gate_step": s.get("gate_step"), "gate_train_exact": s.get("gate_train_exact"),
                    "teacher_vs_gold_mismatch": s.get("teacher_vs_gold_mismatch_share"), "teacher_invalid": s.get("teacher_invalid_share")})
    if arm == "bearly":
        fs = s.get("fork_start") or {}
        out["test_acc_before"] = fs.get("test_acc")
        out["retention"] = [{"step": st.get("step"), "removed": st.get("removed"), "test_acc": st.get("test_acc"), "sj_decode_acc": st.get("sj_decode_acc"),
                             "final_ce_before_after": (st.get("final_ce_before"), st.get("final_ce_after"))} for st in (s.get("stages") or [])]
    c = r.get("compute") or {}
    out["compute"] = {"flops": c.get("flops"), "passes": c.get("sequential_passes")}
    out["exact"] = (out["test_acc"] or 0) >= 0.99
    out["pass"] = bool(out["exact"] and ser)
    out["reading"] = "exact without SERIAL (%s): in-pass / shortcut, not a carried accumulator" % out["serial_rule"] if (out["exact"] and not ser) else None
    return out


def input_verdict(inp):
    """(verdict per arm, runs per arm, proof-drop flag) for one pass input."""
    res = {}
    a1 = run(inp, "a", 1)
    if a1 and a1.get("status") == "done" and (a1.get("test_acc") or 0) < 0.95:
        return {"a": {"runs": {1: a1}, "verdict": "DROP at proof (a s1 %s < 0.95)" % a1.get("test_acc")},
                "ctrl": {"runs": {1: run(inp, "ctrl", 1)}, "verdict": "descriptive"}}, True
    for arm in ARMS:
        runs = {s_: run(inp, arm, s_) for s_ in SEEDS[arm]}
        v = S152.verdict(arm, runs)
        res[arm] = {"runs": runs, "verdict": v, "headline": S152.headline(runs) if v == "PASS" else None}
    return res, False


def fails(rv, dropped):
    return dropped or rv["a"]["verdict"].startswith("FAIL")


def passes(rv):
    return any(rv.get(arm, {}).get("verdict") == "PASS" for arm in ("a", "bearly"))


def main():
    out = {"inputs": {}}
    g = S152.run("a", 1)
    gate = "pending" if not g or g.get("status") != "done" else ("open" if (g.get("test_acc") or 0) >= 0.95 else "closed (§152 a s1 %s < 0.95)" % g.get("test_acc"))
    out["gate_s152_a_s1"] = gate; print("GATE (§152 a s1 >= 0.95):", gate)
    V = {}
    for inp in ("visible", "pure"):
        rv, dropped = input_verdict(inp); V[inp] = (rv, dropped); out["inputs"][inp] = rv
        for arm, x in rv.items():
            for s_, y in x["runs"].items():
                print(inp, arm, "s%d" % s_, y and {k: y.get(k) for k in ("status", "test_acc", "serial", "serial_rule", "pass", "gate_step", "ext_4x5", "compute", "reading",
                                                                         "error")})
            print("VERDICT", inp, arm, x["verdict"], "| HEADLINE", x.get("headline"))
    # predictions (descriptive checks of the pre-stated expectations)
    s152_rec = [((S152.run("a", s_) or {}).get("serial_shares") or {}).get("recompute") for s_ in (1, 2, 3, 4)]
    s152_rec = sorted(x for x in s152_rec if x is not None)
    vis_rec = sorted(x for x in (((y or {}).get("serial_shares") or {}).get("recompute") for y in V["visible"][0]["a"]["runs"].values()) if x is not None)
    if s152_rec and vis_rec:
        m1, m2 = vis_rec[len(vis_rec) // 2], s152_rec[len(s152_rec) // 2]
        out["prediction_i"] = {"visible_recompute_median": m1, "s152_a_recompute_median": m2, "holds": m1 > m2}
        print("PREDICTION (i) VISIBLE recompute > §152 A:", out["prediction_i"])
    pe = sorted(x for x in ((y or {}).get("ext_4x5") for y in V["pure"][0]["a"]["runs"].values()) if x is not None)
    if pe:
        out["prediction_ii"] = {"pure_ext_4x5": pe, "holds": all(x < 0.5 for x in pe)}
        print("PREDICTION (ii) PURE ext_4x5 fails (< 0.5):", out["prediction_ii"])
    # readings
    (rvv, dv), (rvp, dp) = V["visible"], V["pure"]
    s152_pass = any(S152.verdict(arm, {s_: S152.run(arm, s_) for s_ in S152.SEEDS[arm]}) == "PASS" for arm in ("a", "bearly"))
    rd = []
    if passes(rvv):
        rd.append("VISIBLE PASS: item-agnostic passes carry the computation")
    if passes(rvp):
        rd.append("PURE PASS: the memory alone holds problem and state across passes (the strongest C1 form)")
    if fails(rvp, dp) and passes(rvv):
        rd.append("PURE fails, VISIBLE passes: operand recall is the block (operand_recall_pass1 / bj_recall_by_pass locate it)")
    if fails(rvp, dp) and fails(rvv, dv) and s152_pass:
        rd.append("both fail while §152 passes: the designer's per-pass input was doing the work")
    for inp, (rv, dr) in V.items():
        hb = ((rv.get("bearly") or {}).get("headline") or {}).get("LATENT_BEATS_TEXT")
        if not dr and (rv.get("bearly") or {}).get("verdict") == "PASS" and hb:
            rd.append("%s B-EARLY PASS + LATENT BEATS TEXT (final-answer only after CoT-guided training, item-agnostic passes)" % inp.upper())
    out["reading"] = "; ".join(rd) if rd else "pending"
    print("READING:", out["reading"])
    # stage 2 (REPEAT), only if PURE passes at 4x4
    if passes(rvp):
        rr, drr = input_verdict("repeat"); out["inputs"]["repeat"] = rr
        ok = [y for x in ("a", "bearly") for y in (rr.get(x) or {}).get("runs", {}).values() if y and y.get("pass")]
        c2 = bool(ok) and all((y.get("ext_4x5") or 0) >= 0.9 for y in ok)
        noop = bool(ok) and all(all(((y.get("noop") or {}).get(k) or {}).get("test_acc", 0) >= 0.99 for k in ("K6", "K8")) for y in ok)
        out["stage2"] = {"verdicts": {arm: (rr.get(arm) or {}).get("verdict") for arm in ARMS}, "C2_ext_4x5": c2, "noop_stable": noop}
        print("STAGE 2 (REPEAT):", out["stage2"])
    else:
        out["stage2"] = "not run (PURE has not passed at 4x4)"
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s153.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
