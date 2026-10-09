"""THEORY §175 scoring (rules fixed in §175; written before any §175 implementation or run; Theory is the only scorer).
STEP-AUX: §170's INJECT K8 exactly, plus a read-only aux decoder ([EOT, " Step", " t", " :"] reading M_t, no problem text, never fed back) trained to emit
Feng's CoT step t (e_{t+1}) after pass t; loss = answer CE + lambda x mean aux CE over passes 1..#steps, lambda 1.0 fading to 0 over 36k-48k; dropped at test.
MORE-THINKING contrast (descriptive): mean n 9-12 at K' = 2K with vs without aux.
Within training: §170's DEEP / THINK-NECESSARY / SERIAL-NEST per run (pooled n* <= n <= 8). AUX-HELPS-DEPTH(arm): mean n 9-12 >= §170 K8's seed-mean + 0.10
in >= 2 / 3 seeds; AUX-HURTS-DEPTH(arm): <= reference - 0.10 in >= 2 / 3. READOUT (C7): aux exact match >= 0.90 at every pass t <= #steps (n <= 8;
aux_acc_by_pass). §170's blinding applies (n* from score_s170.N_STAR_FIXED). Dirs boxes/runs_fsq/s175_auxk8_s{seed}.
    python theory/score_s175.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s170 as S170  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
K = 8
BASE = dict(S170.IBASE, think_k=K, aux_step=1.0, aux_fade="36000,48000")
ERRORS = []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def run(s, ns_):
    d = "s175_auxk8_s%d" % s; r = load(d); a = load(d, "args.json") or {}
    if r is None:
        return {"status": "running" if a else "open"}
    if a.get("mem_check") or a.get("train_n") or str(a.get("device", "")).startswith("cpu"):
        return {"status": "smoke"}
    bad = {k: a.get(k) for k, v in BASE.items() if a.get(k) != v}
    if bad:
        ERRORS.append((d, bad)); return {"status": "error", "error": bad}
    ab = r.get("acc_by_n") or {}; x = r.get("s170") or {}; CH = S170.CH
    pk = S170.pooled(ab, ns_); p0 = S170.pooled(S170.kseries(x, 0), ns_); p1 = S170.pooled(S170.kseries(x, 1), ns_)
    rem = bool(pk is not None and p0 is not None and p1 is not None and (p0 - CH) <= 0.5 * (pk - CH) and (p1 - CH) <= 0.5 * (pk - CH))
    sk = [S170.pooled(v, ns_) for v in (x.get("skip_pass") or {}).values() if isinstance(v, dict)]
    drops = [pk - v for v in sk if v is not None] if pk is not None else []
    ser = x.get("serial") or {}; nul = x.get("serial_null")
    if isinstance(nul, dict):                                               # the run writes {t, hybrid}
        nul = nul.get("hybrid")
    gaps = {t: v.get("hybrid") - nul for t, v in ser.items() if isinstance(v, dict) and v.get("hybrid") is not None and nul is not None and 1 <= int(t) < K}
    ax = (r.get("s175") or {}).get("aux_acc_by_pass") or {}
    return {"status": "done", "pooled": pk, "deep": all((S170.acc(ab, n) or 0) >= 0.90 for n in range(ns_, 9)),
            "think_necessary": bool(rem and drops and max(drops) >= 0.2 and min(drops) <= 0.05), "serial_nest": bool(gaps and max(gaps.values()) >= 0.5),
            "mean_9_12": S170.mean([S170.acc(ab, n) for n in range(9, 13)]), "aux_acc_by_pass": ax,
            "mean_9_12_at_2K": S170.mean([S170.acc(S170.kseries(x, 2 * K), n) for n in range(9, 13)]),
            "readout": bool(ax) and all(float(v) >= 0.90 for v in ax.values()), "aux_acc_by_pass_ext": (r.get("s175") or {}).get("aux_acc_by_pass_ext"),
            "aux_acc_by_role": (r.get("s175") or {}).get("aux_acc_by_role"), "step_recover": (r.get("s175") or {}).get("step_recover"),
            "gpu": r.get("train_gpu")}


def main():
    res = {}
    ns_ = S170.N_STAR_FIXED
    if ns_ is None:
        res["reading"] = "blinded / pending (§170 n* note not committed)"
    else:
        refs = [S170.inject_run("k8", s, ns_) for s in (1, 2, 3)]; refs = [v for v in refs if v.get("status") == "done"]
        ref = S170.mean([v.get("mean_9_12") for v in refs]); res["reference_final_only_9_12"] = ref
        res["reference_final_only_9_12_at_2K"] = S170.mean([v.get("mean_9_12_at_2K") for v in refs])
        res["reference_step_recover"] = {s_: ((S170.load("s170_k8_s%d" % s_) or {}).get("s170") or {}).get("step_recover") for s_ in (1, 2, 3)}
        runs = {s: run(s, ns_) for s in (1, 2, 3)}; done = [v for v in runs.values() if v["status"] == "done"]; res["runs"] = runs
        if ref is None or not done:
            res["reading"] = "pending (FINAL-only reference %s; %d STEP-AUX seeds)" % (ref, len(done))
        else:
            helps = sum(1 for v in done if v["mean_9_12"] is not None and v["mean_9_12"] >= ref + 0.10)
            hurts = sum(1 for v in done if v["mean_9_12"] is not None and v["mean_9_12"] <= ref - 0.10)
            cnt = lambda k: sum(1 for v in done if v.get(k))  # noqa: E731
            tag = "AUX-HELPS-DEPTH" if helps >= 2 else ("AUX-HURTS-DEPTH" if hurts >= 2 else ("neither (%d helps / %d hurts of %d)" % (helps, hurts, len(done))
                                                                                             if len(done) == 3 else "pending (%d / 3)" % len(done)))
            res["reading"] = "%s vs FINAL-only 9-12 %.3f; DEEP %d, THINK-NECESSARY %d, SERIAL-NEST %d, READOUT %d of %d" % (
                tag, ref, cnt("deep"), cnt("think_necessary"), cnt("serial_nest"), cnt("readout"), len(done))
            print({s: {k: v.get(k) for k in ("status", "pooled", "mean_9_12", "deep", "think_necessary", "readout")} for s, v in runs.items() if v["status"] != "open"})
    print("READING:", res["reading"])
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s175.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
