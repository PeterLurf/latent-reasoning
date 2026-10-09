"""THEORY §171 scoring (rules fixed in §171; written before any §171 implementation or run; Theory is the only scorer).
READ-SLOTS: the problem text (the bench's reversed-digit format) is encoded once by GPT-2 with no memory read; each of K think passes runs M = 8 learned
slot tokens that attend to the input's KV in every layer (and causally to themselves; fresh slot cache per pass), read §155 R's DeltaMem-R at pass start
and write it from the slot positions; the decoder [" ="] after the input's KV -> the 4 + n answer digits (reads the memory, writes nothing, attends to no
slot). K0 = DIRECT by construction (§164 DIRECT-MIX). Trained FINAL-only on §161's 4 x n mix (n 2..5), 48k, RTX 3090; arm K8 (slot 1 EXACT opens s2 / s3).
EXACT(run): 4x4 >= 0.99 at K' = K. LENGTH-GENERAL(arm): >= 2 / 3 seeds EXACT with ext_4x8 >= 0.90.
THINK-NECESSARY(run) (C1): 4x4 accuracy at K' = 0 and K' = 1 each <= 0.5 x accuracy at K, credited only with step-specific skip damage (some t drop >= 0.2
and some t drop <= 0.05); the C1 claim also needs §164 DIRECT-MIX mean 4x4 < 0.95.
CHEAPER-EQUAL vs INJECT (§169 K8, same data): seed-mean 4x4 and ext_4x8 each >= §169 K8's seed-mean - 0.01 at fewer FLOPs per item (compute.flops at 4x4,
compute_4x8.flops at 4x8); read only with >= 2 seeds in both arms. MORE-THINKING (descriptive): ext_4x8 at 2K >= at K + 0.05.
BLIND (note 171-1, descriptive): the decoder sees only [EOT, " ="] + memory; EXACT-BLIND (4x4 >= 0.99) = the memory carries the whole answer.
Dirs boxes/runs_fsq/s171_k8_4x4_s{seed}; s171_k8blind_4x4_s1.
    python theory/score_s171.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s164 as S164  # noqa: E402  (DIRECT-MIX = this harness's K0)
import score_s169 as S169  # noqa: E402  (INJECT K8, the head-to-head arm)

R = os.path.join(ROOT, "boxes", "runs_fsq")
K = 8
BASE = {"arm": "readslots", "think_k": K, "n_slots": 8, "shape_mix": "2,3,4,5", "steps": 48000, "mem_write": "top", "write_mix_start": 2000, "write_mix_end": 6000}
ERRORS = []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def flops(c):
    if isinstance(c, dict):
        return c.get("flops")
    return c if isinstance(c, (int, float)) else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def run(s, d=None, base=None):
    d = d or "s171_k8_4x4_s%d" % s; base = base or BASE; r = load(d); a = load(d, "args.json") or {}
    if r is None:
        return {"status": "running" if a else "open"}
    if a.get("mem_check") or a.get("train_n") or str(a.get("device", "")).startswith("cpu"):
        return {"status": "smoke"}
    bad = {k: a.get(k) for k, v in base.items() if a.get(k) != v}
    if bad:
        ERRORS.append((d, bad)); return {"status": "error", "error": bad}
    x = r.get("s171") or {}; bk = x.get("acc_by_kprime") or {}; e8 = x.get("ext_4x8_by_kprime") or {}
    t = r.get("test_acc"); e = (r.get("ext_4x8") or {}).get("test_acc")
    a0, a1 = bk.get("0"), bk.get("1")
    rem = bool(t and a0 is not None and a1 is not None and a0 <= 0.5 * t and a1 <= 0.5 * t)
    sk = {k_: v_ for k_, v_ in (x.get("skip_pass") or {}).items() if isinstance(v_, (int, float))}
    drops = [t - v_ for v_ in sk.values()] if t is not None else []
    step_specific = bool(drops and max(drops) >= 0.2 and min(drops) <= 0.05)
    more = None if e8.get(str(K)) is None or e8.get(str(2 * K)) is None else bool(e8[str(2 * K)] >= e8[str(K)] + 0.05)
    return {"status": "done", "test_4x4": t, "exact": bool((t or 0) >= 0.99), "ext_4x8": e, "length_general": bool((t or 0) >= 0.99 and (e or 0) >= 0.90),
            "ext": {k: (r.get(k) or {}).get("test_acc") for k in ("ext_4x6", "ext_4x7", "ext_4x8", "ext_4x16")}, "acc_by_kprime": bk, "ext_4x8_by_kprime": e8,
            "think_necessary": bool(rem and step_specific), "removal_drop": rem, "skip_step_specific": step_specific, "more_thinking": more,
            "read_order_rho": x.get("read_order_rho"), "read_entropy": x.get("read_entropy"), "flops_4x4": flops(r.get("compute")), "flops_4x8": flops(x.get("compute_4x8") or r.get("compute_4x8")),
            "gpu": r.get("train_gpu")}


def main():
    res = {}
    runs = {s: run(s) for s in (1, 2, 3)}; done = [v for v in runs.values() if v.get("status") == "done"]
    s1 = runs[1]; gate = None if s1.get("status") != "done" else s1["exact"]
    n_lg = sum(1 for v in done if v["length_general"])
    if gate is False:
        v = "gate closed (s1 %s < 0.99)" % s1["test_4x4"]
    elif n_lg >= 2:
        v = "LENGTH-GENERAL"
    elif len(done) - n_lg >= 2:
        v = "NOT LENGTH-GENERAL"
    else:
        v = "pending (%d / 3 seeds)" % len(done)
    tn = bool(done) and all(x["think_necessary"] for x in done)
    res["k8"] = {"runs": runs, "gate": gate, "verdict": v, "think_necessary_all": tn}
    print("k8 |", v, "|", {s: {k: x.get(k) for k in ("status", "test_4x4", "ext_4x8", "acc_by_kprime", "think_necessary", "more_thinking", "read_order_rho", "error")}
                          for s, x in runs.items() if x.get("status") != "open"})
    dm = [S164.run("directmix", s_) for s_ in (1, 2)]; dm = [v_["test_4x4"] for v_ in dm if v_.get("status") == "done" and v_.get("test_4x4") is not None]
    dmm = sum(dm) / len(dm) if len(dm) == 2 else None; res["direct_mix_mean_4x4"] = dmm
    inj = [S169.run("k8", s_) for s_ in (1, 2, 3)]; inj = [v_ for v_ in inj if v_.get("status") == "done"]
    ce = None
    if len(done) >= 2 and len(inj) >= 2:
        rs = {"t": mean([x["test_4x4"] for x in done]), "e": mean([x["ext_4x8"] for x in done]), "f4": mean([x["flops_4x4"] for x in done]), "f8": mean([x["flops_4x8"] for x in done])}
        ij = {"t": mean([x["test_4x4"] for x in inj]), "e": mean([x["ext_4x8"] for x in inj]),
              "f4": mean([flops(x.get("compute")) for x in inj]), "f8": mean([flops(x.get("compute_4x8")) for x in inj])}
        ok = all(rs[k] is not None and ij[k] is not None for k in rs)
        ce = {"readslots": rs, "inject_k8": ij, "cheaper_equal": None if not ok else bool(rs["t"] >= ij["t"] - 0.01 and rs["e"] >= ij["e"] - 0.01
                                                                                          and rs["f4"] < ij["f4"] and rs["f8"] < ij["f8"])}
    res["head_to_head"] = ce
    bl = run(1, "s171_k8blind_4x4_s1", dict(BASE, blind_decoder=True)); res["blind"] = bl
    if bl.get("status") == "done":
        print("BLIND (descriptive) | 4x4 %s, ext_4x8 %s, EXACT-BLIND %s" % (bl["test_4x4"], bl["ext_4x8"], bl["exact"]))
    if gate is None or v.startswith("pending"):
        rd = "pending"
    elif gate is False:
        rd = "READ-SLOTS K8 not exact at slot 1"
    elif v == "LENGTH-GENERAL" and tn and dmm is not None and dmm < 0.95:
        rd = "a model-directed read carries serial computation across passes (EXACT, LENGTH-GENERAL, THINK-NECESSARY; DIRECT-MIX %.3f)" % dmm
    elif tn:
        rd = "EXACT and THINK-NECESSARY, %s" % v
    else:
        rd = "%s without step-specific think necessity (the slots may solve it within one pass)" % v
    if ce and ce.get("cheaper_equal") is not None:
        rd += "; CHEAPER-EQUAL vs INJECT K8: %s" % ce["cheaper_equal"]
    res["reading"] = rd; print("READING:", rd)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s171.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
