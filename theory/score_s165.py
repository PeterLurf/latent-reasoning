"""THEORY §165 scoring (rules fixed in §165; written before any §165 implementation or run; Theory is the only scorer).
FIXED-CHUNK: §155 R's DeltaMem-R fed the PLAIN problem text (the bench's own input format) in fixed chunks of 4 tokens per pass, then K think passes on a
constant filler segment, then a read-only answer decoder; trained FINAL-only on §161's 4 x n mix (n 2..5), 48k, RTX 3090. Arms K16, K8 and K0 (no think
passes: the single-pass baseline on the same data and harness).
EXACT(run): 4x4 bench >= 0.99 at K' = K. Slot 1 EXACT opens s2 / s3 for that arm. LENGTH-GENERAL(arm): >= 2 / 3 seeds EXACT with ext_4x8 >= 0.90 (K' = K).
THINK-NECESSARY(run) (C1, the north star's removal test): 4x4 accuracy at K' = 0 and at K' = 1 each <= 0.5 x the accuracy at K' = K, credited only with
step-specific skip damage (skip_pass: some t with a drop >= 0.2 and some t with a drop <= 0.05; Audit: removal alone could be distribution shift).
C1 also needs §164 DIRECT-MIX (the measured single pass, same data) mean 4x4 < 0.95 and K0 (the no-think baseline) slot 1 < 0.95. MORE-THINKING (C2 past the trained K; descriptive reading): ext_4x8 at K' = 2K >= ext_4x8 at K + 0.05.
Dirs boxes/runs_fsq/s165_{k16,k8,k0,k16c1}_4x4_s{seed} (k16c1: chunk size 1, slot 1 only, descriptive).
    python theory/score_s165.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s164 as S164  # noqa: E402  (DIRECT-MIX: the measured single-pass baseline on the same data)

R = os.path.join(ROOT, "boxes", "runs_fsq")
ARMS = {"k16": (16, 4), "k8": (8, 4), "k0": (0, 4), "k16c1": (16, 1)}           # (think K, chunk size); k16c1 = Literature's chunk-size check (s1, descriptive)
BASE = {"arm": "fixedchunk", "shape_mix": "2,3,4,5", "steps": 48000, "mem_write": "top", "write_mix_start": 2000, "write_mix_end": 6000}
ERRORS = []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def run(arm, s):
    K, ck = ARMS[arm]; d = "s165_%s_4x4_s%d" % (arm, s); r = load(d); a = load(d, "args.json") or {}
    if r is None:
        return {"status": "running" if a else "open"}
    if a.get("mem_check") or a.get("train_n") or str(a.get("device", "")).startswith("cpu"):
        return {"status": "smoke"}
    bad = {k: a.get(k) for k, v in dict(BASE, think_k=K, chunk_k=ck).items() if a.get(k) != v}
    if bad:
        ERRORS.append((d, bad)); return {"status": "error", "error": bad}
    x = r.get("s165") or {}; bk = x.get("acc_by_kprime") or {}; e8 = x.get("ext_4x8_by_kprime") or {}
    t = r.get("test_acc"); e = (r.get("ext_4x8") or {}).get("test_acc")
    a0, a1 = bk.get("0"), bk.get("1")
    rem = bool(K > 0 and t and a0 is not None and a1 is not None and a0 <= 0.5 * t and a1 <= 0.5 * t)
    sk = {k_: v_ for k_, v_ in (x.get("skip_pass") or {}).items() if isinstance(v_, (int, float))}
    drops = [t - v_ for v_ in sk.values()] if t is not None else []
    step_specific = bool(drops and max(drops) >= 0.2 and min(drops) <= 0.05)
    nec = bool(rem and step_specific)
    more = None if K == 0 or e8.get(str(K)) is None or e8.get(str(2 * K)) is None else bool(e8[str(2 * K)] >= e8[str(K)] + 0.05)
    return {"status": "done", "test_4x4": t, "exact": bool((t or 0) >= 0.99), "ext_4x8": e, "length_general": bool((t or 0) >= 0.99 and (e or 0) >= 0.90),
            "ext": {k: (r.get(k) or {}).get("test_acc") for k in ("ext_4x6", "ext_4x7", "ext_4x8", "ext_4x16")}, "acc_by_kprime": bk, "ext_4x8_by_kprime": e8,
            "think_necessary": nec, "removal_drop": rem, "skip_step_specific": step_specific, "more_thinking": more, "digit_emergence": x.get("digit_emergence"), "skip_pass": x.get("skip_pass"),
            "compute": r.get("compute"), "compute_4x8": x.get("compute_4x8"), "gpu": r.get("train_gpu")}


def main():
    res = {}
    for arm in ARMS:
        runs = {s: run(arm, s) for s in (1, 2, 3)}; done = [v for v in runs.values() if v.get("status") == "done"]
        s1 = runs[1]; gate = None if s1.get("status") != "done" else s1["exact"]
        n_lg = sum(1 for v in done if v["length_general"])
        if arm == "k16c1":                                                   # chunk-size check (Literature 9878803c): slot 1, descriptive
            v = "pending" if gate is None else "chunk 1: 4x4 %s, ext_4x8 %s (descriptive)" % (s1["test_4x4"], s1["ext_4x8"])
        elif arm == "k0":
            v = "pending" if gate is None else ("single pass EXACT (%s)" % s1["test_4x4"] if gate else "single pass fails (%s)" % s1["test_4x4"])
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
    elif good and dmm is None:
        rd = "pending (§164 DIRECT-MIX, the single-pass baseline, not read)"
    elif good and dmm >= 0.95:
        rd = "no C1 claim: the single pass (§164 DIRECT-MIX mean 4x4 %.3f) already solves the problem on this data" % dmm
    elif k0:
        rd = "the plain-text problem fits a single pass on this data (K0 exact): think passes are not needed, no C1 claim"
    elif good:
        rd = ("the memory learns its own schedule from the standard reversed-digit text: task-agnostic segmentation (%s: exact, length-general, think "
              "passes necessary with step-specific damage; single pass %.3f and K0 fail)" % (good, dmm))
    elif exact_arms and all(not res[a]["verdict"].startswith("pending") for a in exact_arms):
        rd = "task-agnostic exactness (%s) without length generalisation: length needs the harness's schedule" % exact_arms
    elif not exact_arms:
        rd = "neither K arm is exact on plain text: the harness's schedule is doing work; the claim stays 'structured-input latent computation'"
    else:
        rd = "pending"
    res["reading"] = rd; print("READING:", rd)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s165.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
