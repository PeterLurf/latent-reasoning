"""THEORY §160 scoring (rules fixed in §160; written before any §160 implementation or run; Theory is the only scorer).
R-ROWS: §152 A's recipe (multrow, b-first, the teacher S_j reader, 48k, the gate) with DeltaMem-R (--mem_write top, write_mix 2000 / 6000); arms ar (reader) and
ctrlr (no reader). Runs only if §155 R s2 >= 0.99. PASS(arm): §152's rule (>= 3/4 >= 0.99 + SERIAL); EXACT(ar s1) >= 0.99 gates s2-s4. Headline: §152's LATENT BEATS
TEXT vs TC-M, ICoT-SI beside it (expected: 0.46x TC-M FLOPs / 0.21x passes; 1.44x ICoT-SI FLOPs / 0.92x passes, stated in §160).
Dirs boxes/runs_fsq/s160_{ar,ctrlr}_4x4_s{seed}.
    python theory/score_s160.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s152 as S152  # noqa: E402
import score_s155 as S155  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
BASE = dict(S152.BASE, seg_order="b_first", mem_write="top", write_mix_start=2000, write_mix_end=6000)
ERRORS = []


def trigger(r, blk):
    """155-1 fallback trigger with 160-1's floor (descriptive readout; the block-9-write variant is due if it fires): write-row spikes > 100x the
    preceding-1k median AND > 1.0 after write_mix_start (the module's write_grad_spikes_floor list), or any non-finite step."""
    sp = ((r.get(blk) or {}).get("write_grad_spikes_floor")) or []; nf = int(r.get("n_nonfinite") or 0)
    return {"floor_spikes": len(sp), "n_nonfinite": nf, "fires": bool(sp or nf)}


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def cell(arm, seed):
    d = "s160_%s_4x4_s%d" % (arm, seed); r = load(d); a = load(d, "args.json") or {}
    if r is None:
        return None
    if int(a.get("steps", 0) or 0) != 48000 or a.get("mem_check") or a.get("train_n"):
        return {"status": "smoke"}
    exp = dict(BASE, reader=(arm == "ar"))
    bad = {k: a.get(k) for k, v in exp.items() if a.get(k) != v}
    if bad:
        ERRORS.append((d, bad)); return {"status": "error", "error": bad}
    c = r.get("compute") or {}; s = r.get("s152") or {}
    return {"status": "done", "test_acc": r.get("test_acc"), "acc_by_length": r.get("acc_by_length"), "serial": S152.serial(r.get("donor")),
            "flops": c.get("flops"), "passes": c.get("sequential_passes"), "pass": None, "gate_step": s.get("gate_step"), "gpu": r.get("train_gpu"),
            "compute": {"flops": c.get("flops"), "passes": c.get("sequential_passes")}, "fallback_trigger": trigger(r, "s160")}


def main():
    g = S155.slot("r", 2)["accepted"]
    cond = "pending (§155 R s2 not read)" if not g else ("RUN (R s2 %s >= 0.99)" % g["test_acc"] if (g["test_acc"] or 0) >= 0.99 else "NOT RUN (R s2 %s < 0.99)" % g["test_acc"])
    print("CONDITION:", cond); res = {"condition": cond}
    for arm in ("ar", "ctrlr"):
        runs = {s_: cell(arm, s_) for s_ in (1, 2, 3, 4)}
        for x in runs.values():
            if x and x.get("status") == "done":
                x["pass"] = bool((x["test_acc"] or 0) >= 0.99 and x["serial"])
        v = S152.verdict("a", runs) if arm == "ar" else ("descriptive: test_acc %s" % ((runs[1] or {}).get("test_acc")) if runs[1] else "pending")
        hl = S152.headline(runs) if v == "PASS" else None
        res[arm] = {"runs": runs, "verdict": v, "headline": hl}
        print(arm, v, {k: x and {kk: x.get(kk) for kk in ("status", "test_acc", "acc_by_length", "serial", "flops", "passes", "fallback_trigger", "error")} for k, x in runs.items()})
    a1 = (res["ar"]["runs"][1] or {}); c1 = (res["ctrlr"]["runs"][1] or {})
    rd = []
    if a1.get("status") == "done":
        rd.append("A-R s1 %s: %s" % (a1["test_acc"], "write-after-read fixes §152's row-4 carry (stored computed results used at read time)" if (a1["test_acc"] or 0) >= 0.99
                  else "rows overload a pass even with stored results" if (a1["test_acc"] or 0) < 0.95 else "between the bars"))
        if a1.get("flops"):
            rd.append("compute %.3g FLOPs (%.2fx TC-M, %.2fx ICoT-SI), %s passes" % (a1["flops"], a1["flops"] / S152.TCM["flops"], a1["flops"] / S152.ICOT["flops"], a1["passes"]))
    if c1.get("status") == "done" and (c1.get("test_acc") or 0) >= 0.99 and (a1.get("test_acc") or 0) >= 0.99:
        rd.append("CTRL-R also >= 0.99: the reader is no longer needed")
    res["reading"] = "; ".join(rd) if rd else "pending"; print("READING:", res["reading"])
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s160.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
