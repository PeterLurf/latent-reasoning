"""THEORY §174 scoring (rules fixed in §174; written before any §174 implementation or run; Theory is the only scorer).
MULTI-TASK: one INJECT model (§169's harness exactly, K8, FINAL only, 48k, RTX 3090) on a 50 / 50 per-batch mix of §161's 4 x n multiplication (n 2..5) and §170's
nested expressions (n 1..8), no task token.
MATCH(run): multiplication 4x4 >= §169 K8 seed-mean - 0.02 and ext_4x8 >= its seed-mean - 0.05; nested accuracy at every n in n*..8 >= §170 K8's seed-mean at
that n - 0.02. THINK-NECESSARY(run) per task: §169's form on 4x4 (K' = 0, 1 each <= 0.5 x acc(K), with step-specific skip damage); 168-1's above-chance form
(chance 1/11) on the nested pooled range, with step-specific skip damage. TASK-AGNOSTIC(arm): MATCH and both THINK-NECESSARY in >= 2 / 3 seeds.
Gate: read only if §169 K8 and §170 K8 are each EXACT / DEEP at slot 1. §170's blinding applies (n* from score_s170.N_STAR_FIXED).
Budget control (Audit): a MATCH miss reads BUDGET if the run meets the same margins against the single-task runs' step-24k evals (eval_24k.json, matched
examples per task), INTERFERENCE if it falls below them; NEAR-MATCH (descriptive) = a miss <= 0.05 at 4x4 / pooled.
Dirs boxes/runs_fsq/s174_mt_k8_s{seed}.
    python theory/score_s174.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s169 as S169  # noqa: E402
import score_s170 as S170  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
K = 8
BASE = {"arm": "inject", "task": "mix", "think_k": K, "steps": 48000, "mem_write": "top", "write_mix_start": 2000, "write_mix_end": 6000}
ERRORS = []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def match_against(t, e, ab, ns_, ref):
    mm = None if ref["m_t"] is None or ref["m_e"] is None or t is None or e is None else bool(t >= ref["m_t"] - 0.02 and e >= ref["m_e"] - 0.05)
    mn = None
    if ref["n_by_n"]:
        vals = [(S170.acc(ab, n), ref["n_by_n"].get(n)) for n in range(ns_, 9)]
        mn = None if any(v is None or w is None for v, w in vals) else all(v >= w - 0.02 for v, w in vals)
    return mm, mn


def run(s, ns_, ref, ref24):
    d = "s174_mt_k8_s%d" % s; r = load(d); a = load(d, "args.json") or {}
    if r is None:
        return {"status": "running" if a else "open"}
    if a.get("mem_check") or a.get("train_n") or str(a.get("device", "")).startswith("cpu"):
        return {"status": "smoke"}
    bad = {k: a.get(k) for k, v in BASE.items() if a.get(k) != v}
    if bad:
        ERRORS.append((d, bad)); return {"status": "error", "error": bad}
    # multiplication (§169 fields)
    x9 = r.get("s169") or {}; bk = x9.get("acc_by_kprime") or {}
    t = r.get("test_acc"); e = (r.get("ext_4x8") or {}).get("test_acc")
    a0, a1 = bk.get("0"), bk.get("1")
    rem_m = bool(t and a0 is not None and a1 is not None and a0 <= 0.5 * t and a1 <= 0.5 * t)
    skm = [t - v for v in (x9.get("skip_pass") or {}).values() if isinstance(v, (int, float))] if t is not None else []
    tn_m = bool(rem_m and skm and max(skm) >= 0.2 and min(skm) <= 0.05)
    # nested (§170 fields)
    ab = r.get("acc_by_n") or {}; x0 = r.get("s170") or {}
    pk = S170.pooled(ab, ns_); p0 = S170.pooled(S170.kseries(x0, 0), ns_); p1 = S170.pooled(S170.kseries(x0, 1), ns_); CH = S170.CH
    rem_n = bool(pk is not None and p0 is not None and p1 is not None and (p0 - CH) <= 0.5 * (pk - CH) and (p1 - CH) <= 0.5 * (pk - CH))
    skn = [pk - v for v in (S170.pooled(w, ns_) for w in (x0.get("skip_pass") or {}).values() if isinstance(w, dict)) if v is not None] if pk is not None else []
    tn_n = bool(rem_n and skn and max(skn) >= 0.2 and min(skn) <= 0.05)
    match_m, match_n = match_against(t, e, ab, ns_, ref)
    match = None if match_m is None or match_n is None else bool(match_m and match_n)
    b_m, b_n = match_against(t, e, ab, ns_, ref24)
    miss = {}
    if match_m is False:
        miss["mult"] = "BUDGET" if b_m else ("INTERFERENCE" if b_m is False else "no 24k reference")
    if match_n is False:
        miss["nest"] = "BUDGET" if b_n else ("INTERFERENCE" if b_n is False else "no 24k reference")
    near = {"mult": None if t is None or ref["m_t"] is None else bool(t >= ref["m_t"] - 0.05),
            "nest": None if pk is None or not ref["n_by_n"] else bool(pk >= (sum(ref["n_by_n"].values()) / len(ref["n_by_n"])) - 0.05)}
    return {"status": "done", "mult_4x4": t, "mult_ext_4x8": e, "nest_pooled": pk, "match_mult": match_m, "match_nest": match_n, "match": match,
            "miss_reading": miss, "near_match": near, "tn_mult": tn_m, "tn_nest": tn_n, "task_agnostic": bool(match and tn_m and tn_n), "s174": r.get("s174"), "gpu": r.get("train_gpu")}


def main():
    res = {}
    ns_ = S170.N_STAR_FIXED
    if ns_ is None:
        res["reading"] = "blinded / pending (§170 n* note not committed)"; print("READING:", res["reading"])
    else:
        i9 = {s: S169.run("k8", s) for s in (1, 2, 3)}; i0 = {s: S170.inject_run("k8", s, ns_) for s in (1, 2, 3)}
        g9 = i9[1].get("exact") if i9[1].get("status") == "done" else None; g0 = i0[1].get("deep") if i0[1].get("status") == "done" else None
        d9 = [v for v in i9.values() if v.get("status") == "done"]; d0 = [v for v in i0.values() if v.get("status") == "done"]
        ref = {"m_t": mean([v["test_4x4"] for v in d9]), "m_e": mean([v["ext_4x8"] for v in d9]),
               "n_by_n": {n: mean([S170.acc(v["acc_by_n"], n) for v in d0]) for n in range(ns_, 9)} if d0 else {}}
        e9 = [load("s169_k8_4x4_s%d" % s, "eval_24k.json") for s in (1, 2, 3)]; e9 = [x for x in e9 if x]
        e0 = [load("s170_k8_s%d" % s, "eval_24k.json") for s in (1, 2, 3)]; e0 = [x for x in e0 if x]
        ref24 = {"m_t": mean([x.get("test_acc") for x in e9]), "m_e": mean([(x.get("ext_4x8") or {}).get("test_acc") for x in e9]),
                 "n_by_n": {n: mean([S170.acc(x.get("acc_by_n"), n) for x in e0]) for n in range(ns_, 9)} if e0 else {}}
        res["gate"] = {"s169_k8_s1_exact": g9, "s170_k8_s1_deep": g0}; res["reference"] = ref; res["reference_24k"] = ref24
        if g9 is None or g0 is None:
            res["reading"] = "pending (single-task slot-1 references)"
        elif not (g9 and g0):
            res["reading"] = "gate closed (a single-task K8 reference failed slot 1): not launched / not read"
        else:
            runs = {s: run(s, ns_, ref, ref24) for s in (1, 2, 3)}; done = [v for v in runs.values() if v["status"] == "done"]
            n_ta = sum(1 for v in done if v["task_agnostic"]); res["runs"] = runs
            if n_ta >= 2:
                res["reading"] = "TASK-AGNOSTIC (%d / %d): one model, one harness, two serial tasks, no flag, no loss against single-task models" % (n_ta, len(done))
            elif len(done) - n_ta >= 2:
                mm = sum(1 for v in done if v["match_mult"] is False); mn = sum(1 for v in done if v["match_nest"] is False)
                res["reading"] = "not TASK-AGNOSTIC (%d / %d); MATCH fails: mult %d, nest %d (miss readings %s); THINK-NECESSARY mult %d, nest %d" % (
                    n_ta, len(done), mm, mn, [v["miss_reading"] for v in done], sum(1 for v in done if v["tn_mult"]), sum(1 for v in done if v["tn_nest"]))
            else:
                res["reading"] = "pending (%d / 3 seeds)" % len(done)
            print({s: {k: v.get(k) for k in ("status", "mult_4x4", "mult_ext_4x8", "nest_pooled", "match", "tn_mult", "tn_nest")} for s, v in runs.items() if v["status"] != "open"})
        print("READING:", res["reading"])
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s174.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
