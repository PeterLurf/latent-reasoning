"""THEORY §176 scoring (rules fixed in §176; written before any §176 implementation or run; Theory is the only scorer).
RAND-K: §169's INJECT on §161's 4 x n mix (48k, RTX 3090, fast_add), two training-method arms:
  RK   = per batch K = 1 + Poisson(e^tau), tau ~ N(log 8 - 1/8, 1/2), cap 32, independent of n; full backprop where memory allows (bptt_last recorded);
  PROG = per batch k1 ~ U{0..8} no-grad passes then k2 ~ U{1..8} grad passes (Bansal 2202.05826), answer decoded after k1 + k2.
Evaluated at K' in {0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64}; fixed-K reference = §169 K8 (its eval.json + post-hoc eval_s176_ref.json at 24 / 32 / 48 / 64).
EXACT(run): 4x4 >= 0.99 at K' = 8. THINK-NECESSARY(run): §169's form at K = 8.
MORE-THINKING(run, informative): ext_4x8 at 16 >= at 8 + 0.05 and 4x4 at 16 >= at 8 - 0.01 (inside RK / PROG's training range: not C2).
NO-OVERTHINKING(run): 4x4 and ext_4x8 at K' = 24 and 32 each >= their value at 8 - 0.01.
C2-BEYOND(run): ext_4x8 at 64 >= at 32 - 0.01 and >= at 8 + 0.05.
C2-RECIPE(arm): C2-BEYOND and NO-OVERTHINKING in >= 2 / 3 of the arm's seeds while fixed K8's C2-BEYOND holds in < 2 of its seeds.
Descriptive: across-seed SD of ext_4x8 at 16 / 32 / 64 per arm vs fixed K8.
Dirs boxes/runs_fsq/s176_{rk,prog}_s{seed}.
    python theory/score_s176.py
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s169 as S169  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
ARMS = {"rk": {"rand_k": "8,0.5,32"}, "prog": {"prog": "8,8"}}
ERRORS = []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def curve_rules(bk, e8):
    g = lambda dct, k: dct.get(str(k))  # noqa: E731
    more = None if None in (g(e8, 8), g(e8, 16), g(bk, 8), g(bk, 16)) else bool(g(e8, 16) >= g(e8, 8) + 0.05 and g(bk, 16) >= g(bk, 8) - 0.01)
    noov = None
    if None not in (g(e8, 8), g(bk, 8), g(e8, 24), g(e8, 32), g(bk, 24), g(bk, 32)):
        noov = all(g(d, k) >= g(d, 8) - 0.01 for d in (bk, e8) for k in (24, 32))
    c2 = None if None in (g(e8, 8), g(e8, 32), g(e8, 64)) else bool(g(e8, 64) >= g(e8, 32) - 0.01 and g(e8, 64) >= g(e8, 8) + 0.05)
    return more, noov, c2


def run(arm, s):
    d = "s176_%s_s%d" % (arm, s); r = load(d); a = load(d, "args.json") or {}
    if r is None:
        return {"status": "running" if a else "open"}
    if a.get("mem_check") or a.get("train_n") or str(a.get("device", "")).startswith("cpu"):
        return {"status": "smoke"}
    base = dict(S169.BASE, think_k=8, **ARMS[arm])
    bad = {k: a.get(k) for k, v in base.items() if a.get(k) != v}
    if bad:
        ERRORS.append((d, bad)); return {"status": "error", "error": bad}
    x = r.get("s169") or {}; bk = x.get("acc_by_kprime") or {}; e8 = x.get("ext_4x8_by_kprime") or {}
    t = bk.get("8", r.get("test_acc")); a0, a1 = bk.get("0"), bk.get("1")
    rem = bool(t and a0 is not None and a1 is not None and a0 <= 0.5 * t and a1 <= 0.5 * t)
    sk = [t - v for v in (x.get("skip_pass") or {}).values() if isinstance(v, (int, float))] if t is not None else []
    more, noov, c2 = curve_rules(bk, e8)
    return {"status": "done", "test_4x4_k8": t, "exact": bool((t or 0) >= 0.99), "think_necessary": bool(rem and sk and max(sk) >= 0.2 and min(sk) <= 0.05),
            "more_thinking": more, "no_overthinking": noov, "c2_beyond": c2, "ext_4x8_by_kprime": e8, "acc_by_kprime": bk, "bptt_last": a.get("bptt_last"),
            "acc_by_size_kprime": x.get("acc_by_size_kprime"), "k_hist": (r.get("s176") or {}).get("k_hist"), "gpu": r.get("train_gpu")}


def ref_run(s):
    v = S169.run("k8", s)
    if v.get("status") != "done":
        return v
    p = load("s169_k8_4x4_s%d" % s, "eval_s176_ref.json") or {}
    bk = dict(v.get("acc_by_kprime") or {}, **(p.get("acc_by_kprime") or {})); e8 = dict(v.get("ext_4x8_by_kprime") or {}, **(p.get("ext_4x8_by_kprime") or {}))
    more, noov, c2 = curve_rules(bk, e8)
    return {"status": "done", "has_posthoc": bool(p), "more_thinking": more, "no_overthinking": noov, "c2_beyond": c2, "ext_4x8_by_kprime": e8}


def sd(runs, k):
    xs = [v["ext_4x8_by_kprime"].get(str(k)) for v in runs if v.get("status") == "done" and v["ext_4x8_by_kprime"].get(str(k)) is not None]
    return statistics.stdev(xs) if len(xs) >= 2 else None


def main():
    res = {}
    ref = {s: ref_run(s) for s in (1, 2, 3)}; rdone = [v for v in ref.values() if v.get("status") == "done"]
    ref_c2 = sum(1 for v in rdone if v.get("c2_beyond")); ref_ready = bool(rdone) and all(v.get("has_posthoc") for v in rdone)
    res["fixed_k8"] = {"runs": ref, "c2_beyond": "%d / %d" % (ref_c2, len(rdone)), "posthoc_ready": ref_ready, "sd_ext_4x8": {k: sd(rdone, k) for k in (16, 32, 64)}}
    lines = []
    for arm in ARMS:
        runs = {s: run(arm, s) for s in (1, 2, 3)}; done = [v for v in runs.values() if v["status"] == "done"]; s1 = runs[1]
        n_ok = sum(1 for v in done if v["c2_beyond"] and v["no_overthinking"]); n_more = sum(1 for v in done if v["more_thinking"])
        res[arm] = {"runs": runs, "c2_and_noov": n_ok, "more_thinking": n_more, "sd_ext_4x8": {k: sd(done, k) for k in (16, 32, 64)}}
        if s1["status"] != "done" or not ref_ready:
            t = "%s: pending (s1 %s; fixed-K8 post-hoc curve ready %s)" % (arm, s1["status"], ref_ready)
        elif not s1["exact"]:
            t = "%s: slot 1 not EXACT at K' = 8 (%s): the training method costs in-distribution accuracy" % (arm, s1["test_4x4_k8"])
        elif len(done) < 3 and n_ok < 2 and len(done) - n_ok < 2:
            t = "%s: pending (%d / 3; C2-BEYOND + NO-OVERTHINKING %d)" % (arm, len(done), n_ok)
        elif n_ok >= 2 and ref_c2 < 2:
            t = "%s: C2-RECIPE (C2-BEYOND + NO-OVERTHINKING %d / %d; fixed K8 C2-BEYOND %d / %d)" % (arm, n_ok, len(done), ref_c2, len(rdone))
        elif n_ok >= 2:
            t = "%s: C2-BEYOND %d / %d but fixed K8 too (%d / %d): the sampling is not needed for C2 here" % (arm, n_ok, len(done), ref_c2, len(rdone))
        else:
            t = "%s: not C2 (C2-BEYOND + NO-OVERTHINKING %d / %d; MORE-THINKING at 16, informative, %d)" % (arm, n_ok, len(done), n_more)
        lines.append(t)
        print(arm, {s: {k: v.get(k) for k in ("status", "test_4x4_k8", "more_thinking", "no_overthinking", "c2_beyond")} for s, v in runs.items() if v["status"] != "open"})
    res["reading"] = lines
    for ln in lines:
        print("READING:", ln)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s176.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
