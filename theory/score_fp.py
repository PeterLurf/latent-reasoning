"""THEORY §270 FULL-PERM scorer (FP_v1 = docs/research_fullperm_2026-11-04/FP_v1.md; Theory is the only scorer; committed before output).
Runs fp_s1 / fp_s2 = of_s1 / of_s2 args.json + --perm_scope all (all statements permuted, question last; chain + ask items), from bun_sX_init, 8k.
Binding per seed:
  FULL-ORDER: eval_full_perm by_d[d].acc >= 0.85 for d 2-4 (fresh seed family 27010, K' 8).
  RETAIN-ORDER: eval_test_perm by_d[d].acc_kprime['8'] >= 0.90 and eval_late_premise by_d[d].acc >= 0.85, d 2-4.
  RETAIN: eval_acc d 1-4 >= 0.98; NOT-DECODER by_d['4'].acc_kprime['2'] <= 0.20; SWAP4; LP5-D4 (as §263).
  C2-RETAIN: eval_acc d 6 / d 7 >= of_sX (same seed) - 0.05; c2 rows NEAR-SEED (sigma_seed 0.090, Note 269-2).
  C6: eval_long_k128 d 1-6: acc['64'], acc['128'] >= acc['8'] - 0.02.
  CONTENT: eval_insert_ho fam['H-CHAINSRC-OP'] acc8 >= 0.85 and min(by_pos) >= 0.80.
  STABLE (score_ao).
Every binding cell carries a Wilson 95% CI and NEAR (CI contains the bar); letters are scored on raw values.
Descriptive: full_perm d 5-6, by_flag splits, the of-vs-fp paired d 4 diff (of_sX eval-only full_perm), shuffle_graded, name_swap, bun d 6 / 7.
Censoring: without --final a missing binding file is pending; with --final it is FAIL: VOID (after the one VOID-RERUN).
Label FULL-ORDER-C1 if all pass, else 'FAIL: <criteria>'. Per arm FULL-ORDER-C1 k/2.
    python -B theory/score_fp.py [--fixtures] [--final]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory"))
import score_ao as AO  # noqa: E402

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_fp.json")
RUNS = {"fp_s1": ("of_s1", "bun_s1"), "fp_s2": ("of_s2", "bun_s2")}
SIGMA_SEED = 0.090   # Note 269-1 / 269-2: d 6 seed spread
BIND = ("eval_acc", "eval_full_perm", "eval_test_perm", "eval_late_premise", "eval_swap_kd", "eval_lp5h", "eval_long_k128", "eval_insert_ho")


def jl(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def wilson(p, n, z=1.96):
    n = n or 1
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return (c - h, c + h)


def cell(value, bar, n=1000, ge=True):
    """A binding cell: raw-value letter, Wilson 95% CI, NEAR when the CI contains the bar."""
    lo, hi = wilson(value, n)
    return {"value": value, "bar": bar, "ok": value >= bar if ge else value <= bar, "ci95": (lo, hi), "near": lo <= bar <= hi}


def c2_row(value, bar, n=1000):
    lab = "NEAR-SEED" if abs(value - bar) < SIGMA_SEED else ("PASS" if value >= bar else "FAIL")
    return {"value": value, "bar": bar, "margin": value - bar, "ci95": wilson(value, n), "label": lab}


def seed(runs, run, ref, bun, final=False):
    f = {x: jl(os.path.join(runs, run, x + ".json")) for x in BIND}
    rf = jl(os.path.join(runs, ref, "eval_acc.json"))
    if any(v is None for v in list(f.values()) + [rf]):
        miss = [k for k, v in f.items() if v is None] + ([ref + "/eval_acc"] if rf is None else [])
        return {"label": "FAIL: VOID" if final else "pending", "missing": miss}
    a = f["eval_acc"]["by_d"]; fp = f["eval_full_perm"]["by_d"]; tp = f["eval_test_perm"]["by_d"]; lp = f["eval_late_premise"]["by_d"]
    sw = f["eval_swap_kd"]["by_d"]["4"]["swap_mid"]["4"]; h = f["eval_lp5h"]["by_d"]["4"]; L = f["eval_long_k128"]["by_d"]; r = rf["by_d"]
    n_of = lambda blk, d: blk[d].get("n", 1000)  # noqa: E731
    cells = {}
    for d in ("2", "3", "4"):
        cells["full_perm_d" + d] = cell(fp[d]["acc"], 0.85, n_of(fp, d))
        cells["test_perm_d" + d] = cell(tp[d]["acc_kprime"]["8"], 0.90, n_of(tp, d))
        cells["late_premise_d" + d] = cell(lp[d]["acc"], 0.85, n_of(lp, d))
    for d in ("1", "2", "3", "4"):
        cells["test2_d" + d] = cell(a[d]["acc"], 0.98, n_of(a, d))
    cells["not_decoder_d4"] = cell(a["4"]["acc_kprime"]["2"], 0.20, n_of(a, "4"), ge=False)
    # Audit G-D MED-1: every proportion bar gets a cell (Wilson CI + NEAR); LP5-D4 (a difference of two accuracies) stays a raw boolean
    sw_n = f["eval_swap_kd"]["by_d"]["4"].get("n_pairs", 1000)   # insert_ho / long_k128 record no per-cell n: those cells use n 1000 (disclosed, Note 270-1)
    cells["swap4_follow"] = cell(sw["follow_U"], max(0.50, sw["floor"] + 0.30), sw_n)
    co = f["eval_insert_ho"]["fam"]["H-CHAINSRC-OP"]
    cells["content_acc8"] = cell(co["acc8"], 0.85, co.get("n", 1000))
    for k_, v_ in co["by_pos"].items():
        cells["content_pos" + str(k_)] = cell(v_, 0.80, co.get("n_by_pos", {}).get(k_, 1000) if isinstance(co.get("n_by_pos"), dict) else 1000)
    for d in ("1", "2", "3", "4", "5", "6"):
        for k_ in ("64", "128"):
            cells["c6_d%s_k%s" % (d, k_)] = cell(L[d]["acc"][k_], L[d]["acc"]["8"] - 0.02, L[d].get("n", 1000))
    full = all(cells["full_perm_d" + d]["ok"] for d in ("2", "3", "4"))
    rord = all(cells[k + d]["ok"] for k in ("test_perm_d", "late_premise_d") for d in ("2", "3", "4"))
    retain = (all(cells["test2_d" + d]["ok"] for d in ("1", "2", "3", "4")) and cells["not_decoder_d4"]["ok"] and cells["swap4_follow"]["ok"]
              and h["intact"]["4"] - h["mean_at_t"]["4"] >= 0.30)
    c2 = a["6"]["acc"] >= r["6"]["acc"] - 0.05 and a["7"]["acc"] >= r["7"]["acc"] - 0.05
    c6 = all(cells["c6_d%s_k%s" % (d, k)]["ok"] for d in ("1", "2", "3", "4", "5", "6") for k in ("64", "128"))
    content = all(v["ok"] for k, v in cells.items() if k.startswith("content_"))
    sok, smsg = AO.stable(runs, run)
    if sok is None:   # Audit SCORER_BUGS MED-1: a missing / blank log is pending, never STABLE
        return {"label": "FAIL: VOID" if final else "pending", "stable": (sok, smsg)}
    crit = {"FULL-ORDER": full, "RETAIN-ORDER": rord, "RETAIN": retain, "C2-RETAIN": c2, "C6": c6, "CONTENT": content, "STABLE": sok is not False}
    fail = [k for k, v in crit.items() if not v]
    c2rows = {"d6": c2_row(a["6"]["acc"], r["6"]["acc"] - 0.05, n_of(a, "6")), "d7": c2_row(a["7"]["acc"], r["7"]["acc"] - 0.05, n_of(a, "7"))}
    ofp = jl(os.path.join(runs, ref, "eval_full_perm.json")); bn = jl(os.path.join(runs, bun, "eval_acc.json"))
    paired = {d: fp[d]["acc"] - ofp["by_d"][d]["acc"] for d in fp if ofp and d in ofp.get("by_d", {})} if ofp else None
    desc_missing = ([ref + "/eval_full_perm"] if ofp is None else []) + ([bun + "/eval_acc"] if bn is None else [])   # Audit LOW (b): reported, never binding
    return {"label": "FULL-ORDER-C1" if not fail else "FAIL: " + ",".join(fail), "crit": crit, "cells": cells,
            "near": sorted(k for k, v in cells.items() if v["near"]), "c2_rows": c2rows,
            "c2_claim": "NEAR-SEED" if any(v["label"] == "NEAR-SEED" for v in c2rows.values()) else ("PASS" if c2 else "FAIL"),
            "full_perm_desc": {d: {k: v for k, v in fp[d].items() if k != "items"} for d in fp},
            "paired_vs_of_desc": paired, "desc_missing": desc_missing, "of_d6": r["6"]["acc"], "of_d7": r["7"]["acc"],
            "bun_d6_desc": bn["by_d"]["6"]["acc"] if bn else None, "bun_d7_desc": bn["by_d"]["7"]["acc"] if bn else None,
            "graded_desc": jl(os.path.join(runs, run, "eval_shuffle_graded.json")), "stable": (sok, smsg)}


def score(runs=R, final=False):
    o = {"section": 270, "final": final, "seeds": {r: seed(runs, r, ref, bun, final) for r, (ref, bun) in RUNS.items()}}
    labs = [v["label"] for v in o["seeds"].values()]
    dm = sorted({m for v in o["seeds"].values() for m in v.get("desc_missing", [])})
    if final and dm:
        o["desc_missing"] = dm; print("NOTE (--final): descriptive inputs missing, P2 / P6 not readable:", ", ".join(dm))
    o["verdict"] = "pending" if "pending" in labs else "FULL-ORDER-C1 %d/2 (%s); C2 claim %s" % (labs.count("FULL-ORDER-C1"), " / ".join(labs),
                                                                                            "/".join(v.get("c2_claim", "?") for v in o["seeds"].values())); return o


def fixtures():
    res = []

    def world(fpv=0.95, fp2=None, tp=0.95, lp=0.9, t2=1.0, t2d3=None, k2=0.03, sw=0.9, lp5=0.1, d6=0.95, d7=0.85, hold=1.0, co=0.95, pos=0.9,
              nolog=False, nofp=False, of6=0.97, of7=0.86, bun6=0.70, bun7=0.60, fail_s2=False, swfloor=0.1, nanlog=False, noofacc=False):
        d = tempfile.mkdtemp()
        for run, (ref, bun) in RUNS.items():
            for rr, v6, v7 in ((ref, of6, of7), (bun, bun6, bun7)):
                os.makedirs(os.path.join(d, rr))
                if not (noofacc and rr == ref):
                    json.dump({"by_d": {"6": {"acc": v6}, "7": {"acc": v7}}}, open(os.path.join(d, rr, "eval_acc.json"), "w"))
            json.dump({"by_d": {x: {"acc": 0.45} for x in ("2", "3", "4")}}, open(os.path.join(d, ref, "eval_full_perm.json"), "w"))
            p = os.path.join(d, run); os.makedirs(p)
            fv = 0.5 if (fail_s2 and run == "fp_s2") else fpv
            json.dump({"by_d": {**{x: {"acc": (t2d3 if (x == "3" and t2d3 is not None) else t2), "acc_kprime": {"2": k2 if x == "4" else 1.0}} for x in ("1", "2", "3", "4")},
                                "6": {"acc": d6}, "7": {"acc": d7}}}, open(os.path.join(p, "eval_acc.json"), "w"))
            if not nofp:
                json.dump({"by_d": {x: {"acc": (fp2 if (x == "2" and fp2 is not None) else fv), "n": 1000} for x in ("2", "3", "4")}}, open(os.path.join(p, "eval_full_perm.json"), "w"))
            json.dump({"by_d": {x: {"acc_kprime": {"8": tp}} for x in ("2", "3", "4")}}, open(os.path.join(p, "eval_test_perm.json"), "w"))
            json.dump({"by_d": {x: {"acc": lp} for x in ("2", "3", "4")}}, open(os.path.join(p, "eval_late_premise.json"), "w"))
            json.dump({"by_d": {"4": {"swap_mid": {"4": {"follow_U": sw, "floor": swfloor}}}}}, open(os.path.join(p, "eval_swap_kd.json"), "w"))
            json.dump({"by_d": {"4": {"intact": {"4": 1.0}, "mean_at_t": {"4": lp5}}}}, open(os.path.join(p, "eval_lp5h.json"), "w"))
            json.dump({"by_d": {str(x): {"acc": {"8": 1.0, "64": 1.0, "128": hold}} for x in range(1, 7)}}, open(os.path.join(p, "eval_long_k128.json"), "w"))
            json.dump({"fam": {"H-CHAINSRC-OP": {"acc8": co, "by_pos": {"1": pos, "2": 0.95}}}}, open(os.path.join(p, "eval_insert_ho.json"), "w"))
            json.dump({"n_nonfinite": 0}, open(os.path.join(p, "train_summary.json"), "w"))
            with open(os.devnull if nolog else os.path.join(p, "log.jsonl"), "w") as fh:
                for i in range(1, 81):
                    fh.write(json.dumps({"step": i * 100, "loss": 5.0 if (nanlog and i > 40) else 0.01}) + chr(10))
        return d
    cases = (("all", {}, {}, "FULL-ORDER-C1 2/2"),
             ("full_perm d4 0.84", {"fpv": 0.84}, {}, "FAIL: FULL-ORDER"),
             ("only d2 below", {"fp2": 0.80}, {}, "FAIL: FULL-ORDER"),
             ("TEST-PERM 0.89", {"tp": 0.89}, {}, "FAIL: RETAIN-ORDER"),
             ("LATE-PREMISE 0.84", {"lp": 0.84}, {}, "FAIL: RETAIN-ORDER"),
             ("TEST2 d3 0.97", {"t2d3": 0.97}, {}, "FAIL: RETAIN"),
             ("K'2 0.6", {"k2": 0.6}, {}, "FAIL: RETAIN"),
             ("SWAP4", {"sw": 0.2}, {}, "FAIL: RETAIN"),
             ("LP5", {"lp5": 0.9}, {}, "FAIL: RETAIN"),
             ("C2 far below of (both rows; Note 269-6: any NEAR row makes the claim NEAR-SEED)", {"d6": 0.70, "d7": 0.60}, {}, "C2 claim FAIL/FAIL"),
             ("C2 within sigma_seed", {"d6": 0.95}, {}, "C2 claim NEAR-SEED/NEAR-SEED"),
             ("C2 bar reads of not bun", {"d6": 0.80, "d7": 0.70}, {}, "FAIL: C2-RETAIN"),
             ("C6", {"hold": 0.5}, {}, "FAIL: C6"),
             ("CONTENT acc8", {"co": 0.6}, {}, "FAIL: CONTENT"),
             ("CONTENT by_pos", {"pos": 0.7}, {}, "FAIL: CONTENT"),
             ("blank log is pending (MED-1)", {"nolog": True}, {}, "pending"),
             ("missing full_perm is pending", {"nofp": True}, {}, "pending"),
             ("missing full_perm --final is VOID", {"nofp": True}, {"final": True}, "FULL-ORDER-C1 0/2 (FAIL: VOID / FAIL: VOID)"),
             ("one seed fails", {"fail_s2": True}, {}, "FULL-ORDER-C1 1/2"),
             ("STABLE fails (unrecovered loss jump)", {"nanlog": True}, {}, "STABLE"),
             ("SWAP4 floor + 0.30 clause", {"sw": 0.75, "swfloor": 0.5}, {}, "FAIL: RETAIN"),
             ("missing of eval_acc is pending", {"noofacc": True}, {}, "pending"))
    for name, kw, sk, want in cases:
        o = score(world(**kw), **sk); res.append((name, want in o["verdict"], o["verdict"]))
    o = score(world(fpv=0.86)); near = o["seeds"]["fp_s1"]
    res.append(("d4 0.86 n1000 PASS letter + NEAR flag", near["label"] == "FULL-ORDER-C1" and "full_perm_d4" in near["near"], near["label"] + " near=" + ",".join(near["near"])))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); ap.add_argument("--final", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(final=a.final); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
