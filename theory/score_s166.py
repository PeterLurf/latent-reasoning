"""THEORY §166 scoring (rules fixed in §166; written before any §166 implementation or run; Theory is the only scorer).
STATE-COVERAGE: short post-training of existing models with wider initial-state coverage (Buitrago Ruiz & Gu 2507.02782), against a same-length plain
post-train (the control for "more steps"). 1,000 steps, AdamW lr 2e-5 constant after 50 warmup, bs 64; half the chains start from the covered state.
  S5 (§162 LONG s2's model, l_max 32): SP = state passing (memory from another random chain's final memory, start text = its final state) vs PLAIN.
  MULT (§161 MIX s1 / s2 / s3's models): NOISE = Gaussian memory init (per-head norm matched to the trained mid-chain state) vs PLAIN.
COVERAGE-HELPS(S5): SP long_acc_512 >= 0.90 AND PLAIN long_acc_512 < 0.50, with SP final accuracy >= 0.99 at every length <= 32; else 'covered range only' if SP >= 0.90 at 48 / 64.
Covered states come from the current model (state_source current); SP skips segment 0 for passed chains (Audit).
COVERAGE-HELPS(MULT): NOISE mean ext_4x16 (3 models) >= 0.50 AND PLAIN mean < 0.30, with NOISE mean 4x4 >= 0.99 and mean ext_4x8 >= 0.90.
Dirs boxes/runs_fsq/s166_s5_{sp,plain}_s2 and s166_mult_{noise,plain}_s{1,2,3}.
    python theory/score_s166.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
PT = {"pt_steps": 1000, "pt_lr": 2e-5, "init_p": 0.5, "state_source": "current"}
ERRORS = []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def cell(d, mode, src):
    r = load(d); a = load(d, "args.json") or {}
    if r is None:
        return {"status": "running" if a else "open"}
    exp = dict(PT, init_mode=mode, post_train_from=src)
    if mode == "zero":                                                       # PLAIN: init_p / state_source have no effect
        exp.pop("init_p"); exp.pop("state_source")
    bad = {k: a.get(k) for k, v in exp.items() if a.get(k) != v}
    if bad:
        ERRORS.append((d, bad)); return {"status": "error", "error": bad}
    return dict(r, status="done")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    res = {}
    sp, pl = cell("s166_s5_sp_s2", "pass", "s162_long_s2"), cell("s166_s5_plain_s2", "zero", "s162_long_s2")
    if sp["status"] == "done" and pl["status"] == "done":
        fl = sp.get("final_acc_by_length") or {}
        kept = bool(fl) and all((fl.get(str(L)) or 0) >= 0.99 for L in (1, 2, 4, 8, 16, 32))
        helps = bool(kept and (sp.get("long_acc") or 0) >= 0.90 and (pl.get("long_acc") or 0) < 0.50)
        ext = sp.get("acc_by_length_ext") or {}; cov = bool((ext.get("48") or 0) >= 0.90 and (ext.get("64") or 0) >= 0.90)
        rs5 = ("COVERAGE-HELPS (S5)" if helps else ("in-distribution not kept" if not kept else
               ("covered range only (SP >= 0.90 at 48 / 64, not at 512)" if cov else "not shown (drift is not coverage: read mem_norm_by_pass)")))
        res["s5"] = {"sp": {k: sp.get(k) for k in ("long_acc", "acc_2048", "acc_by_length_ext")}, "plain": {k: pl.get(k) for k in ("long_acc", "acc_2048", "acc_by_length_ext")}, "kept": kept, "reading": rs5}
    else:
        res["s5"] = {"reading": "pending", "status": (sp["status"], pl["status"])}
    print("S5:", res["s5"])
    nz = {s: cell("s166_mult_noise_s%d" % s, "noise", "s161_mix_4x4_s%d" % s) for s in (1, 2, 3)}
    pn = {s: cell("s166_mult_plain_s%d" % s, "zero", "s161_mix_4x4_s%d" % s) for s in (1, 2, 3)}
    if all(v["status"] == "done" for v in list(nz.values()) + list(pn.values())):
        g = lambda v, k: ((v.get(k) or {}).get("test_acc") if k != "test_acc" else v.get("test_acc"))  # noqa: E731
        n16, p16 = mean([g(v, "ext_4x16") for v in nz.values()]), mean([g(v, "ext_4x16") for v in pn.values()])
        n44, n48 = mean([g(v, "test_acc") for v in nz.values()]), mean([g(v, "ext_4x8") for v in nz.values()])
        kept = bool((n44 or 0) >= 0.99 and (n48 or 0) >= 0.90)
        helps = bool(kept and (n16 or 0) >= 0.50 and (p16 if p16 is not None else 1) < 0.30)
        rm = "COVERAGE-HELPS (MULT)" if helps else ("4x4 / 4x8 not kept" if not kept else "not shown (drift is not coverage: read mem_norm_by_pass)")
        res["mult"] = {"noise_mean_4x16": n16, "plain_mean_4x16": p16, "noise_mean_4x4": n44, "noise_mean_4x8": n48, "kept": kept, "reading": rm}
    else:
        res["mult"] = {"reading": "pending", "status": {**{"noise_s%d" % s: v["status"] for s, v in nz.items()}, **{"plain_s%d" % s: v["status"] for s, v in pn.items()}}}
    print("MULT:", res["mult"])
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s166.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
