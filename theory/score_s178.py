"""THEORY §178 scoring (rules fixed in §178; written before any §178 implementation or run; Theory is the only scorer).
FROM-SCRATCH: §170's nested task and INJECT harness on a randomly initialised small GPT-2 (6 layers, d 384, 6 heads; DeltaMem-R 3 + 3 heads after block 3),
96k steps (eval at 48k descriptive); arms DIRECT-scratch, K0-scratch, K8-scratch. Pooled = n 3..8 (§170's n* = 3); chance 1/11.
USES-PASSES(run) = K8-scratch pooled >= K0-scratch s1 pooled + 0.15 AND K8-scratch pooled >= 1/11 + 0.15 (chance floor) AND removal (168-1: (acc(K') - 1/11) <= 0.5 x (acc(K8) - 1/11) for K' = 0, 1, pooled)
AND 173-1 delta cos at passes 2, 3, 4 each <= the arm's own cos at init - 0.1 (eval_s177_dir.json).
PRETRAINING-OR-CAPACITY = USES-PASSES from scratch (>= 2 / 2 seeds once s2 runs; slot 1 descriptive) while pretrained §170 K8's gap over K0 (< 0.15) holds;
then the conditional matched pair (s178_m12_{k0,k8}: GPT-2's exact 12 / 768 / 12 config at random init) decides PRETRAINING-BLOCKS (it USES-PASSES) vs CAPACITY.
The init cosine is the random trunk's (fresh memory, gamma 1).
HARNESS-BLOCKS = K8-scratch pooled < K0-scratch pooled + 0.05 with removal failing. Else mixed.
Dirs boxes/runs_fsq/s178_{direct,k0,k8}_s{seed}.
    python theory/score_s178.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s170 as S170  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
CH = 1.0 / 11
BASE = {"task": "nested", "scratch": "6,384,6", "steps": 96000, "lr": 0.0005}
ARMS = {"direct": dict(BASE, arm="direct"), "k0": dict(BASE, arm="inject", think_k=0), "k8": dict(BASE, arm="inject", think_k=8),
        "m12_k0": dict(BASE, scratch="12,768,12", arm="inject", think_k=0), "m12_k8": dict(BASE, scratch="12,768,12", arm="inject", think_k=8)}
ERRORS = []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def cell(arm, s):
    d = "s178_%s_s%d" % (arm, s); r = load(d); a = load(d, "args.json") or {}
    if r is None:
        return {"status": "running" if a else "open"}
    bad = {k: a.get(k) for k, v in ARMS[arm].items() if str(a.get(k)) != str(v)}
    if bad:
        ERRORS.append((d, bad)); return {"status": "error", "error": bad}
    ab = r.get("acc_by_n") or {}; x = r.get("s170") or {}; kp = x.get("acc_by_n_kprime") or {}
    out = {"status": "done", "pooled": S170.pooled(ab, 3), "acc_by_n": ab, "gpu": r.get("train_gpu"),
           "pooled_by_kprime": {k: S170.pooled(v, 3) for k, v in kp.items() if isinstance(v, dict)}}
    if arm in ("k8", "m12_k8"):
        pk = out["pooled"]; p0 = S170.pooled(kp.get("0") or {}, 3); p1 = S170.pooled(kp.get("1") or {}, 3)
        out["removal"] = bool(pk is not None and p0 is not None and p1 is not None and (p0 - CH) <= 0.5 * (pk - CH) and (p1 - CH) <= 0.5 * (pk - CH))
        dr = load(d, "eval_s177_dir.json") or {}; dc = dr.get("delta_cos") or {}; ci = ((dr.get("cos_at_init") or {}).get("delta")) or {}
        pairs = [(dc.get(t), ci.get(t)) for t in ("2", "3", "4")]
        out["delta_cos_vs_init"] = pairs
        out["moved_from_init"] = None if any(a_ is None or b_ is None for a_, b_ in pairs) else all(a_ <= b_ - 0.1 for a_, b_ in pairs)
    return out


def uses(v, k0p):
    if v.get("status") != "done" or k0p is None or v.get("pooled") is None:
        return None
    gap = v["pooled"] - k0p; v["gap_over_k0"] = gap
    v["above_chance_floor"] = v["pooled"] >= CH + 0.15
    v["uses_passes"] = bool(gap >= 0.15 and v["above_chance_floor"] and v["removal"] and v.get("moved_from_init"))
    v["harness_blocks"] = bool(gap < 0.05 and not v["removal"])
    return v["uses_passes"]


def main():
    res = {a: {s: cell(a, s) for s in (1, 2)} for a in ARMS}
    k0 = res["k0"][1]; k0p = k0.get("pooled") if k0.get("status") == "done" else None
    pre = {"k8_pooled": 0.3513, "k0_pooled": 0.3023}; pre_gap = pre["k8_pooled"] - pre["k0_pooled"]   # §170 slot 1 (pretrained)
    res["pretrained_gap_s170"] = pre_gap
    lines = []
    for s, v in res["k8"].items():
        uses(v, k0p)
    done = [v for v in res["k8"].values() if v.get("status") == "done" and "uses_passes" in v]
    if k0p is None or not done:
        rd = "pending (K0-scratch s1 %s; K8-scratch done %d)" % (k0.get("status"), len(done))
    else:
        nu = sum(1 for v in done if v["uses_passes"]); nh = sum(1 for v in done if v["harness_blocks"])
        if k0p < CH + 0.1 and all((v.get("pooled") or 0) < CH + 0.1 for v in done):   # note 178-1
            res["reading"] = rd = "UNINFORMATIVE: the scratch model does not learn nested at 96k (K0 %.3f, K8 %s pooled, chance %.3f)" % (
                k0p, [round(v["pooled"], 3) for v in done], CH)
            print("READING:", rd); json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s178.json"), "w", newline=chr(10)), indent=1, default=float)
            return
        tag = "slot 1, descriptive" if len(done) < 2 else "%d / %d seeds" % (len(done), len(done))
        if nu == len(done) and pre_gap < 0.15:
            rd = "PRETRAINING-OR-CAPACITY (%s): from scratch the passes are used (gap %s); pretrained §170 K8's gap is %.3f" % (tag, [round(v["gap_over_k0"], 3) for v in done], pre_gap)
            mk0 = res["m12_k0"][1]; mk0p = mk0.get("pooled") if mk0.get("status") == "done" else None
            mdone = [v for v in res["m12_k8"].values() if uses(v, mk0p) is not None]
            if mdone:
                rd += "; matched 12-layer pair: %s" % ("PRETRAINING-BLOCKS (it uses the passes)" if all(v["uses_passes"] for v in mdone)
                                                     else "CAPACITY (it does not)" if not any(v["uses_passes"] for v in mdone) else "mixed")
            else:
                rd += "; the conditional matched 12-layer pair decides PRETRAINING-BLOCKS vs CAPACITY (pending)"
        elif nh == len(done):
            rd = "HARNESS-BLOCKS (%s): from scratch K8 ~ K0 (gap %s), removal fails" % (tag, [round(v["gap_over_k0"], 3) for v in done])
        else:
            rd = "mixed (%s): USES-PASSES %d, HARNESS-BLOCKS %d of %d" % (tag, nu, nh, len(done))
    res["reading"] = rd; print("READING:", rd)
    for a in ARMS:
        print(a, {s: {k: v.get(k) for k in ("status", "pooled", "removal", "moved_from_init", "gap_over_k0", "above_chance_floor")} for s, v in res[a].items() if v.get("status") != "open"})
        for s, v in res[a].items():
            if v.get("status") == "done":
                print("   per n", a, s, {n: round(x, 3) for n, x in (v.get("acc_by_n") or {}).items() if int(n) <= 12})
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s178.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
