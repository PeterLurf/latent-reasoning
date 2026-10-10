"""THEORY §182 scoring (rules fixed in §182; written before any §182 implementation or run; Theory is the only scorer).
STEP-AUX-HOLD: §175 exactly with the aux weight held at 1 to 48k (NOFADE, fixed K 8); NOFADE + K = n + U{1..4} per item, masked (WINDOW); WINDOW + hold aux
targets e_{n+1} on passes t > n (HOLD). Pooled = n 3..8 (n* = 3); chance 1/11. K' = 24 / 32 descriptive beside STABLE (2x WINDOW / HOLD's max K).
DEEP(cell) = acc >= 0.90 at every n 3..8 at K' = 8. THINK-NECESSARY = 170-4 removal-only vs K' = 8 (K' = 0, 1), 170-5 guard. STABLE (C6) = pooled K'=16 >=
pooled K'=8 - 0.02. A DEEP or STABLE cell opens s2; its claim then needs 2 / 2 seeds. n 9-12 descriptive (not C2).
Dirs boxes/runs_fsq/s182_{nofade,window,hold}_s{seed}.
    python theory/score_s182.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s170 as S170  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
NS = 3                                                                      # n* (note 170-3)
CH = 1.0 / 11
EPS = 1e-9
S175 = dict(S170.IBASE, aux_step=1.0)
CELLS = {"nofade": dict(S175, think_k=8, aux_fade="48000,48000", k_window="", aux_hold=False),
         "window": dict(S175, think_k=12, aux_fade="48000,48000", k_window="1,4", kmatch_impl="masked", aux_hold=False),
         "hold": dict(S175, think_k=12, aux_fade="48000,48000", k_window="1,4", kmatch_impl="masked", aux_hold=True)}
ERRORS = []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def run(cell, s):
    d = "s182_%s_s%d" % (cell, s); r = load(d); a = load(d, "args.json") or {}
    if r is None:
        return {"status": "running" if a else "open"}
    if a.get("mem_check") or a.get("train_n") or str(a.get("device", "")).startswith("cpu"):
        return {"status": "smoke"}
    bad = {k: a.get(k) for k, v in CELLS[cell].items() if str(a.get(k)) != str(v)}
    if bad:
        ERRORS.append((d, bad)); return {"status": "error", "error": bad}
    x = r.get("s170") or {}
    k8, k16 = S170.kseries(x, 8), S170.kseries(x, 16)
    p8, p16 = S170.pooled(k8, NS), S170.pooled(k16, NS); p0, p1 = S170.pooled(S170.kseries(x, 0), NS), S170.pooled(S170.kseries(x, 1), NS)
    out = {"status": "done", "gpu": r.get("train_gpu"), "pooled_k8": p8, "pooled_k16": p16, "pooled_k24": S170.pooled(S170.kseries(x, 24), NS), "pooled_k32": S170.pooled(S170.kseries(x, 32), NS),
           "by_kprime": {k: {n: v.get(str(n)) for n in range(1, 13)} for k, v in (x.get("acc_by_n_kprime") or {}).items() if isinstance(v, dict)},
           "mean_9_12": {kp: S170.mean([S170.acc(S170.kseries(x, kp), n) for n in range(9, 13)]) for kp in (8, 16, 24, 32)},
           "aux_acc_by_pass": (r.get("s175") or {}).get("aux_acc_by_pass"), "handle": {t: (v or {}).get("handle") for t, v in ((r.get("s175") or {}).get("aux_acc_by_role") or {}).items()},
           "path_indep": (r.get("s182") or {}).get("path_indep")}
    if p8 is None or p16 is None or p0 is None or p1 is None:
        out["status"] = "incomplete"; out["missing"] = {"K'8": p8 is None, "K'16": p16 is None, "K'0": p0 is None, "K'1": p1 is None}; return out
    rem = (p0 - CH) <= 0.5 * (p8 - CH) + EPS and (p1 - CH) <= 0.5 * (p8 - CH) + EPS
    out.update({"deep": all((S170.acc(k8, n) or 0) >= 0.90 - EPS for n in range(NS, 9)),
                "think_necessary": None if p8 - CH < 0.1 else bool(rem),          # note 170-5: n/a near chance
                "stable": bool(p16 >= p8 - 0.02 - EPS)})
    return out


def verdict(cell, runs):
    done = [v for v in runs.values() if v.get("status") == "done"]
    if not done or runs[1].get("status") != "done":
        return None, "pending (s1 %s)" % runs[1].get("status")
    s2 = runs[2].get("status") == "done"; opens = runs[1]["deep"] or runs[1]["stable"]
    if opens and not s2:
        return None, "slot 1: DEEP %s, STABLE %s (s2 opens; the claim needs 2 / 2)" % (runs[1]["deep"], runs[1]["stable"])
    use = done if s2 else [runs[1]]
    return {"deep": all(v["deep"] for v in use), "stable": all(v["stable"] for v in use), "n": len(use)}, None


def main():
    res = {c: {s: run(c, s) for s in (1, 2)} for c in CELLS}
    vs = {}
    for c in CELLS:
        v, msg = verdict(c, res[c]); vs[c] = v
        res[c]["verdict"] = v or msg
        print(c, v or msg, {s: {k: res[c][s].get(k) for k in ("status", "pooled_k8", "pooled_k16", "deep", "stable", "think_necessary")} for s in (1, 2) if res[c][s].get("status") != "open"})
    nf, wd, hd = vs["nofade"], vs["window"], vs["hold"]
    lines = []
    if nf is not None:
        lines.append("NOFADE DEEP: §175's decay was the fade" if nf["deep"] else "NOFADE not DEEP: the decay is intrinsic to per-step error at this budget")
        if nf["stable"]:
            lines.append("NOFADE STABLE: training to the end with the aux suffices for stability")
    if nf is not None and wd is not None and wd["stable"] and not nf["stable"]:
        lines.append("WINDOW STABLE, NOFADE not: the sampled step-matched pass count gives stability")
    if wd is not None and hd is not None:
        if hd["stable"] and not wd["stable"]:
            lines.append("HOLD STABLE, WINDOW not: the hold target itself gives stability (C6)")
        elif hd["stable"] and wd["stable"]:
            lines.append("WINDOW and HOLD both STABLE: the window suffices; hold adds nothing to stability")
        elif not hd["stable"]:
            lines.append("HOLD not STABLE: hold targets do not fix overthinking here" + (" (and they cost the window's stability)" if wd["stable"] else ""))
    for a_, b_ in (("window", "nofade"), ("hold", "window")):
        pa, pb = res[a_][1].get("pooled_k8"), res[b_][1].get("pooled_k8")
        if pa is not None and pb is not None:
            lines.append("%s - %s at K'=8 (s1): %+.3f" % (a_.upper(), b_.upper(), pa - pb))
    for c in ("window", "hold"):
        v = vs[c]
        if v is not None and v["stable"]:
            lines.append("%s K'=24 (2x max K) pooled %s vs K'=8 %s" % (c.upper(), res[c][1].get("pooled_k24"), res[c][1].get("pooled_k8")))
    res["reading"] = "; ".join(lines) if lines else "pending"
    print("READING:", res["reading"])
    if ERRORS:
        print("ERRORS:", ERRORS)
    res["errors"] = [str(e) for e in ERRORS]
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s182.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
