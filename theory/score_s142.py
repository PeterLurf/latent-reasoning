"""THEORY §142 scoring (rules fixed in §142; written before any §142 implementation or run; Theory is the only scorer).
MULT-READ on multiplier-major multiplication (fsq120 --family mult_row --task 4x4): segment j = [EOT, carried code, a's digits, b_j, position j],
carried state S_j = a x (b_1 .. b_j); carrier NOISE-64 (--d_q 64 --no_round --dither 0.25; no COPY, no restarts); curriculum L_max 1 / 2 / 3 / 4
at 0 / 2k / 5k / 9k (--lmax_steps 2000,5000,9000).
Arms (dirs boxes/runs_fsq/s142_{arm}_4x4_s{1..4}):
  a       READ: the shared answer decoder also decodes S_j from code_j at every hand-off j < L (targets = the TC-M teacher's own generated S_j,
          label_source "teacher_generated"; C4 as reworded); loss = CE_final + mean over hand-offs of CE_j (each over the same 8 answer-format
          tokens); 48k steps. Removal gate: L_max = 4 and greedy final-answer exact >= 0.95 on a fixed 512-item TRAIN subset at L 4 (checked every
          1k) -> gate.pt + s142.gate_step.
  bearly  primary removal arm: forked from a's gate.pt (same seed); every fork row is 4x4 (--fixed_L 4: no prefix problems); reader targets removed
          earliest-first (S_1 at +0, S_2 at +4k, S_3 at +8k), final answer only from +8k to +20k; AdamW reset + 200-step LR re-warm-up at each
          removal; 10% of rows preview the next removal.
  blate   descriptive: the same fork, removing S_3, then S_2, then S_1.
  ctrl    descriptive control (s1 only, only after a s1's proof): a without the reader (FINAL only), same carrier, curriculum and 48k.
PASS(arm): >= 3/4 seeds with test_acc >= 0.99 on the 1,000 published 4x4 items, SERIAL (donor code after segment 2; donor >= 0.90 and
recipient - chance < 0.10; exact ground truth: donor S_2 + a_rec x (b_rec,3 b_rec,4) x 100) and contractive (133-1's first-3 form, rho_k3 <= 0.28,
from the on-host --contraction_only --fp32 file). FAIL: >= 2 decided seeds not passing. DROP at proof: a s1 test_acc < 0.95 stops the section.
B arms: a seed whose a run never opened the gate counts as not passing ("not reached"); PASS is read on the fork's final (final-answer-only) model.
Descriptive: SERIAL before (fork_start) and after removal, b-prefix / operand-carry consistency under the donor swap, retention of each removed
S_j (acc_by_length at every stage), reader accuracy per hand-off, ridge probes (S_j, b-prefix) on code_j, ext_4x8_low8 (S_j mod 10^8 at j = 5-8; note 142-1).
    python theory/score_s142.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
ARMS = ("a", "bearly", "blate", "ctrl")
SEEDS = {"a": (1, 2, 3, 4), "bearly": (1, 2, 3, 4), "blate": (1, 2, 3, 4), "ctrl": (1,)}
BASE = {"family": "mult_row", "task": "4x4", "d_q": 64, "no_round": True, "dither": 0.25, "lmax_steps": "2000,5000,9000"}
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
    n = 20000 if arm in ("bearly", "blate") else 48000
    k = "fork_steps" if arm in ("bearly", "blate") else "steps"
    if int(a.get(k, 0) or 0) != n or float(a.get("sched_scale", 1.0)) != 1.0 or str(a.get("device", "")).startswith("cpu") or a.get("mem_check") or a.get("train_n"):
        SKIPPED.append(d); return "smoke"
    exp = dict(BASE, reader=(arm != "ctrl"))
    if arm in ("bearly", "blate"):
        exp.update({"removal": arm[1:], "removal_every": 4000, "rewarm": 200, "fork_from": "boxes/runs_fsq/s142_a_4x4_s%d/gate.pt" % seed})
    bad = {k_: a.get(k_) for k_, v in exp.items() if a.get(k_) != v}
    if float(a.get("copy_frac") or 0) != 0:
        bad["copy_frac"] = a.get("copy_frac")
    if arm in ("a", "ctrl") and a.get("removal"):
        bad["removal"] = a.get("removal")
    if arm != "ctrl" and a.get("label_source") != "teacher_generated":                 # C4 as reworded: the teacher's outputs, never gold S_j
        bad["label_source"] = a.get("label_source")
    if a.get("stop_plateau"):
        bad["stop_plateau"] = True
    if int(a.get("fixed_L") or 0) != (4 if arm in ("bearly", "blate") else 0):        # B: every fork row is 4x4 (no prefix problems re-supervising S_j)
        bad["fixed_L"] = a.get("fixed_L")
    if bad:
        ERRORS.append((d, bad)); return "args mismatch %s" % bad
    return None


def run(arm, seed):
    d = "s142_%s_4x4_s%d" % (arm, seed)
    if arm in ("bearly", "blate"):
        ra = load("s142_a_4x4_s%d" % seed)
        if ra is not None and (ra.get("s142") or {}).get("gate_step") is None:
            return {"status": "not reached", "pass": False}                             # a finished without opening the gate: B is not run
    r = load(d)
    if r is None:
        return None
    b = bad_args(d, arm, seed)
    if b:
        return {"status": "error", "error": b}
    s = r.get("s142") or {}; dn = (r.get("donor") or {}).get("pos2") or {}
    out = {"status": "done", "test_acc": r.get("test_acc"), "serial": serial(r.get("donor")), "rho": rho_k3(load(d, "eval_contraction_fp32.json")),
           "acc_by_length": r.get("acc_by_length"), "train_acc": r.get("train_acc"), "n_nonfinite": r.get("n_nonfinite"),
           "donor": {k: dn.get(k) for k in ("donor_consistent", "recipient_consistent", "chance_overlap", "bprefix_consistent", "operands_consistent")},
           "probe_r2": s.get("probe_r2"), "ext_4x8_low8": r.get("ext_4x8_low8"), "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver")}
    if arm == "a":
        out.update({"gate_step": s.get("gate_step"), "gate_train_exact": s.get("gate_train_exact"),
                    "teacher_vs_gold_mismatch": s.get("teacher_vs_gold_mismatch_share"), "teacher_invalid": s.get("teacher_invalid_share")})
    if arm in ("bearly", "blate"):
        fs = s.get("fork_start") or {}
        out["serial_before"] = serial(fs.get("donor")); out["test_acc_before"] = fs.get("test_acc")
        out["retention"] = [{"step": st.get("step"), "removed": st.get("removed"), "acc_by_length": st.get("acc_by_length"),
                             "test_acc": st.get("test_acc"), "serial": serial(st.get("donor"))} for st in (s.get("stages") or [])]
    out["exact"] = (out["test_acc"] or 0) >= 0.99
    out["pass"] = None if out["rho"] is None else bool(out["exact"] and out["serial"] and out["rho"] <= 0.28)
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


def main():
    res = {"arms": {}}
    a1 = run("a", 1)
    if a1 and a1.get("status") == "done" and (a1.get("test_acc") or 0) < 0.95:
        res["verdict"] = "DROP at proof (a s1 test_acc %s < 0.95): the section stops; reading 'not acquired within budget'" % a1.get("test_acc")
        res["arms"]["a"] = {"runs": {1: a1}}
        print("PROOF:", res["verdict"]); print("a s1", {k: a1.get(k) for k in ("test_acc", "acc_by_length", "serial", "rho", "gate_step", "donor")})
    else:
        for arm in ARMS:
            runs = {s: run(arm, s) for s in SEEDS[arm]}
            v = verdict(arm, runs); res["arms"][arm] = {"runs": runs, "verdict": v}
            for s, x in runs.items():
                print(arm, "s%d" % s, x and {k: x.get(k) for k in ("status", "test_acc", "serial", "rho", "pass", "gate_step", "serial_before", "reading", "error")})
                for st in (x or {}).get("retention") or []:
                    print("   retention", st)
            print("VERDICT", arm, v)
        va = res["arms"]["a"]["verdict"]; vb = res["arms"]["bearly"]["verdict"]
        if va == "PASS":
            claim = ("A PASS: a per-step reader of the running sum makes multiplication learnable through a Markov latent code (C4 as reworded; unforced C1 "
                     "by SERIAL). ")
            if vb == "PASS":
                claim += "B-EARLY PASS: exact multiplication survives ICoT-style removal to final-answer only (not strict C4: trained through CoT)."
            elif vb.startswith("FAIL"):
                claim += "B-EARLY FAIL: removing the per-step targets loses the skill (retention says which S_j goes first)."
            else:
                claim += "B-EARLY " + vb + "."
            c1 = (res["arms"]["ctrl"]["runs"].get(1) or {}).get("test_acc")
            if c1 is not None:
                claim += (" CTRL (no reader) %.3f: %s" % (c1, "the reader is not what made A learn (carrier / budget)" if c1 >= 0.95
                                                           else "the reader was needed at this budget"))
        elif va.startswith("FAIL"):
            claim = "A FAIL: not acquired within budget (about 3.8 epochs, against ICoT-SI's much longer training); not read as impossibility."
        else:
            claim = "pending (A %s)" % va
        res["verdict"] = claim; print("READING:", claim)
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s142.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
