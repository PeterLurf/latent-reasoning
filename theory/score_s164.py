"""THEORY §164 scoring (rules fixed in §164; written before any §164 implementation or run; Theory is the only scorer).
MIX-BASELINES: like-for-like baselines for §161 MIX, trained on the same 4 x n mix (n ~ U{2..5} per batch, the same operand generator, 24k, RTX 3090,
seeds 1-2) and tested on the 4x4 bench plus §161's ext items (the same seeds): TC-MIX (TC-M's text CoT, mult98 --arm tc) and DIRECT-MIX (the direct
answer, mult98 --arm nc).
LATENT BEYOND TEXT ON LENGTH = §161 MIX LENGTH-GENERAL (score_s161) AND TC-MIX's mean ext_4x8 (s1, s2) < 0.90. TC-MIX mean ext_4x8 >= 0.90 -> "the shape
mix, not latency, gives length; the latent claim is limited to C3". DIRECT-MIX (descriptive): the measured single-pass baseline on the same data.
C3 at 4x8: each arm's FLOPs / passes at 4x8 beside MIX's (compute_pair at 4x8).
Note 164-1 (Audit, before any §164 scoring): the ANCHORED read governs - eval_s164_anchor.json (decode budget = schedule length + 50, the answer read from
the model's own first "####"); a cell without it is "awaiting anchored re-eval"; the position-locked eval.json values are descriptive only.
Dirs boxes/runs_fsq/s164_{tcmix,directmix}_4x4_s{1,2}.
    python theory/score_s164.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s161 as S161  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
ARMS = {"tcmix": "tc", "directmix": "nc"}
BASE = {"task": "4x4", "shape_mix": "2,3,4,5", "steps": 24000}
EXT = ("ext_4x6", "ext_4x7", "ext_4x8", "ext_4x16")
ERRORS = []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def run(arm, s):
    d = "s164_%s_4x4_s%d" % (arm, s); r = load(d); a = load(d, "args.json") or {}
    if r is None:
        return {"status": "running" if a else "open"}
    if a.get("mem_check") or a.get("train_n") or str(a.get("device", "")).startswith("cpu"):
        return {"status": "smoke"}
    bad = {k: a.get(k) for k, v in dict(BASE, arm=ARMS[arm]).items() if a.get(k) != v}
    if bad:
        ERRORS.append((d, bad)); return {"status": "error", "error": bad}
    c4, c8 = r.get("compute") or {}, r.get("compute_4x8") or {}
    an = load(d, "eval_s164_anchor.json")                                      # note 164-1: the anchored read governs
    if an is None:
        return {"status": "awaiting anchored re-eval", "position_locked": {"test_4x4": r.get("acc_test"), "ext": {k: (r.get(k) or {}).get("test_acc") for k in EXT}}}
    return {"status": "done", "test_4x4": an.get("acc_test"), "ext": {k: (an.get(k) or {}).get("test_acc") for k in EXT},
            "ext_4x8_per_column": ((an.get("ext_4x8") or {}).get("acc_by_column") or {}).get("per_column"), "cot_len_mismatch_share": an.get("cot_len_mismatch_share"),
            "position_locked": {"test_4x4": r.get("acc_test"), "ext": {k: (r.get(k) or {}).get("test_acc") for k in EXT}},
            "compute_4x4": {"flops": c4.get("flops"), "passes": c4.get("sequential_passes")}, "compute_4x8": {"flops": c8.get("flops"), "passes": c8.get("sequential_passes")},
            "gpu": r.get("train_gpu", r.get("gpu"))}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    res = {}
    for arm in ARMS:
        runs = {s: run(arm, s) for s in (1, 2)}; done = [v for v in runs.values() if v.get("status") == "done"]
        res[arm] = {"runs": runs, "mean_test_4x4": mean([v["test_4x4"] for v in done]) if done else None,
                    "mean_ext_4x8": mean([v["ext"]["ext_4x8"] for v in done]) if len(done) == 2 else None, "n_done": len(done)}
        print(arm, "| mean 4x4", res[arm]["mean_test_4x4"], "| mean ext_4x8", res[arm]["mean_ext_4x8"], "|",
              {s: {k: v.get(k) for k in ("status", "test_4x4", "ext", "compute_4x8", "error")} for s, v in runs.items() if v.get("status") != "open"})
    sl = {k: S161.slot("mix", k) for k in (1, 2, 3)}; acc = [v["accepted"] for v in sl.values() if v["accepted"]]
    n_lg = sum(1 for a in acc if a["length_general"]); mix_lg = n_lg >= 2
    tm = res["tcmix"]["mean_ext_4x8"]
    if tm is None:
        rd = "pending (TC-MIX %d / 2 seeds)" % res["tcmix"]["n_done"]
    elif not mix_lg:
        rd = "no claim: §161 MIX is not LENGTH-GENERAL (%d / %d seeds)" % (n_lg, len(acc))
    elif tm < 0.90:
        rd = "LATENT BEYOND TEXT ON LENGTH: MIX length-general (%d / %d) where TC-MIX's mean ext_4x8 is %.3f < 0.90" % (n_lg, len(acc), tm)
    else:
        rd = "the shape mix, not latency, gives length (TC-MIX mean ext_4x8 %.3f >= 0.90): the latent claim is limited to C3" % tm
    dm = res["directmix"]["mean_ext_4x8"]
    res["single_pass"] = None if dm is None else ("the single pass fails at 4x8 on the same data (mean %.3f)" % dm if dm < 0.90 else "the single pass also reaches 4x8 (mean %.3f)" % dm)
    res["reading"] = rd; print("READING:", rd, "| single pass (DIRECT-MIX):", res["single_pass"])
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s164.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
