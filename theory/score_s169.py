"""THEORY §169 scoring (rules fixed in §169; written before any §169 implementation or run; Theory is the only scorer).
INJECT: §155 R's DeltaMem-R with the FULL problem text (the bench's reversed-digit format) re-presented in every think pass ([EOT, problem, " ."], fresh
cache, pass-start read, write after read), K think passes, then the decoder [EOT, problem, " ="] -> the 4 + n answer digits (reads the memory, writes
nothing). K0 = the same-harness single pass. Trained FINAL-only on §161's 4 x n mix (n 2..5), 48k, RTX 3090; arms K16 / K8 / K0.
EXACT(run): 4x4 bench >= 0.99 at K' = K; slot 1 EXACT opens s2 / s3 (K16 / K8). LENGTH-GENERAL(arm): >= 2 / 3 seeds EXACT with ext_4x8 >= 0.90.
THINK-NECESSARY(run) (C1): 4x4 accuracy at K' = 0 and K' = 1 each <= 0.5 x accuracy at K, credited only with step-specific skip damage (some t drop >= 0.2
and some t drop <= 0.05); the C1 claim also needs K0 slot 1 < 0.95 (same-harness single pass) and §164 DIRECT-MIX mean 4x4 < 0.95.
MORE-THINKING (C2 past the trained K; descriptive reading): ext_4x8 at K' = 2K >= ext_4x8 at K + 0.05.
Dirs boxes/runs_fsq/s169_{k16,k8,k0}_4x4_s{seed}.
    python theory/score_s169.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s164 as S164  # noqa: E402  (DIRECT-MIX: the single pass on the same data)

R = os.path.join(ROOT, "boxes", "runs_fsq")
ARMS = {"k16": 16, "k8": 8, "k0": 0}
BASE = {"arm": "inject", "shape_mix": "2,3,4,5", "steps": 48000, "mem_write": "top", "write_mix_start": 2000, "write_mix_end": 6000}
ERRORS = []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def run(arm, s):
    K = ARMS[arm]; d = "s169_%s_4x4_s%d" % (arm, s); r = load(d); a = load(d, "args.json") or {}
    if r is None:
        return {"status": "running" if a else "open"}
    if a.get("mem_check") or a.get("train_n") or str(a.get("device", "")).startswith("cpu"):
        return {"status": "smoke"}
    bad = {k: a.get(k) for k, v in dict(BASE, think_k=K).items() if str(a.get(k)) != str(v)}   # type-tolerant: newer code writes think_k as a string
    if bad:
        ERRORS.append((d, bad)); return {"status": "error", "error": bad}
    x = r.get("s169") or {}; bk = x.get("acc_by_kprime") or {}; e8 = x.get("ext_4x8_by_kprime") or {}
    t = r.get("test_acc"); e = (r.get("ext_4x8") or {}).get("test_acc")
    a0, a1 = bk.get("0"), bk.get("1")
    rem = bool(K > 0 and t and a0 is not None and a1 is not None and a0 <= 0.5 * t and a1 <= 0.5 * t)
    sk = {k_: v_ for k_, v_ in (x.get("skip_pass") or {}).items() if isinstance(v_, (int, float))}
    drops = [t - v_ for v_ in sk.values()] if t is not None else []
    step_specific = bool(drops and max(drops) >= 0.2 and min(drops) <= 0.05)
    more = None if K == 0 or e8.get(str(K)) is None or e8.get(str(2 * K)) is None else bool(e8[str(2 * K)] >= e8[str(K)] + 0.05)
    return {"status": "done", "test_4x4": t, "exact": bool((t or 0) >= 0.99), "ext_4x8": e, "length_general": bool((t or 0) >= 0.99 and (e or 0) >= 0.90),
            "ext": {k: (r.get(k) or {}).get("test_acc") for k in ("ext_4x6", "ext_4x7", "ext_4x8", "ext_4x16")}, "acc_by_kprime": bk, "ext_4x8_by_kprime": e8,
            "think_necessary": bool(rem if arm == "k16" else (rem and step_specific)),   # note 170-4: K16 (autonomous passes) is removal only
            "removal_drop": rem, "skip_step_specific": step_specific, "more_thinking": more,
            "graded": None if K == 0 else (bk.get(str(K // 2)), bk.get(str(K))),
            "acc_by_size_kprime": x.get("acc_by_size_kprime"), "inprog_by_pass": x.get("inprog_by_pass"), "compute": r.get("compute"), "compute_4x8": x.get("compute_4x8"),
            "gpu": r.get("train_gpu")}


def main():
    res = {}
    for arm in ARMS:
        runs = {s: run(arm, s) for s in (1, 2, 3)}; done = [v for v in runs.values() if v.get("status") == "done"]
        s1 = runs[1]; gate = None if s1.get("status") != "done" else s1["exact"]
        n_lg = sum(1 for v in done if v["length_general"])
        if arm == "k0":
            v = "pending" if gate is None else ("same-harness single pass EXACT (%s)" % s1["test_4x4"] if gate else "same-harness single pass fails (%s)" % s1["test_4x4"])
        elif gate is False:
            v = "gate closed (s1 %s < 0.99)" % s1["test_4x4"]
        elif n_lg >= 2:
            v = "LENGTH-GENERAL"
        elif len(done) - n_lg >= 2:
            v = "NOT LENGTH-GENERAL"
        else:
            v = "pending (%d / 3 seeds)" % len(done)
        res[arm] = {"runs": runs, "gate": gate, "verdict": v, "think_necessary_all": bool(done) and all(x["think_necessary"] for x in done)}
        print(arm, "|", v, "|", {s: {k: x.get(k) for k in ("status", "test_4x4", "ext_4x8", "acc_by_kprime", "think_necessary", "more_thinking", "error")}
                                 for s, x in runs.items() if x.get("status") != "open"})
    dm = [S164.run("directmix", s_) for s_ in (1, 2)]; dm = [v_["test_4x4"] for v_ in dm if v_.get("status") == "done" and v_.get("test_4x4") is not None]
    dmm = sum(dm) / len(dm) if len(dm) == 2 else None; res["direct_mix_mean_4x4"] = dmm
    k0 = res["k0"]["gate"]
    good = [a for a in ("k16", "k8") if res[a]["verdict"] == "LENGTH-GENERAL" and res[a]["think_necessary_all"]]
    exact_arms = [a for a in ("k16", "k8") if res[a]["gate"]]
    if k0 is None or any(res[a]["gate"] is None for a in ("k16", "k8")):
        rd = "pending"
    elif k0:
        rd = "the same-harness single pass (K0) is exact: think passes are not needed, no C1 claim"
    elif good and (dmm is None or dmm >= 0.95):
        rd = "pending (§164 DIRECT-MIX)" if dmm is None else "no C1 claim: DIRECT-MIX solves 4x4 (%.3f)" % dmm
    elif good:
        rd = ("task-agnostic latent computation: %s exact, length-general, think passes necessary with step-specific damage, while the single pass "
              "(K0 and DIRECT-MIX %.3f) fails; with the input re-shown every pass, what crosses the passes is computed partial results" % (good, dmm))
    elif exact_arms and all(not res[a]["verdict"].startswith("pending") for a in exact_arms):
        rd = "task-agnostic exactness (%s) without length generalisation (or without step-specific think necessity)" % exact_arms
    elif not exact_arms:
        rd = "neither K arm is exact: input re-presentation alone does not suffice at 48k"
    else:
        rd = "pending"
    res["reading"] = rd; print("READING:", rd)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s169.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
