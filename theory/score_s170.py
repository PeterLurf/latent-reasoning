"""THEORY §170 scoring (rules fixed in §170; written before any §170 implementation or run; Theory is the only scorer).
NESTED: Feng 2305.15408's Arithmetic(n, 11) (backward generation, brackets only where precedence / left associativity need them), trained on n ~ U{1..8}
on the fly, tested on 1,000 items per n in 1..16 (chance 1/11). Arms (GPT-2 small, 48k, bs 64, RTX 3090): DIRECT [EOT, expr, " ="] -> answer (s1 / s2);
INJECT K8 / K16 / K0 (§169's harness exactly: think pass [EOT, expr, " ."], decoder [EOT, expr, " ="] reads the memory and writes nothing);
TEXT-COT in Feng's format (leftmost handle per step), read anchored (164-1).
Boundary: per DIRECT seed n*_s = the smallest n in 1..16 with accuracy < 0.90 (a seed < 0.90 at n = 1 did not train: dropped); n* = the lower ("boundary
uncertain" if they differ); fixed in a dated note from DIRECT alone. Until N_STAR_FIXED below is set by that note's commit, only DIRECT is scored (blinding).
If n* >= 7 or DIRECT passes through n = 8: no C1 claim from the 1..8 range.
Pooled = items at n* <= n <= 8 (equal counts per n, so the mean over n).
DEEP(run): accuracy >= 0.90 at every n in n*..8 at K' = K. THINK-NECESSARY(run): (acc(K') - 1/11) <= 0.5 x (acc(K) - 1/11) for K' = 0 and 1 (pooled),
credited only with step-specific skip damage (some t drops pooled accuracy >= 0.2, some <= 0.05). SERIAL-NEST(run): max over think passes t < K of (hybrid share - null) >= 0.5,
reported with its t and beside the donor (whole-answer transplant) share.
C1(arm): DEEP, THINK-NECESSARY and SERIAL-NEST each in >= 2 / 3 seeds, with K0 slot 1 pooled < 0.90. DEPTH-GENERAL(arm): mean n 9-12 >= 0.80 in >= 2 / 3
seeds. MORE-THINKING (descriptive): mean 9-12 at 2K >= at K + 0.05. BEYOND TEXT: TEXT-COT mean (2 seeds) at 9-12 < 0.80 while the arm is DEPTH-GENERAL.
AT-LEAST-TEXT-COT (C3): per n <= 8, the arm's seed mean >= TEXT-COT's seed mean - 0.01 at lower compute_by_n.
Dirs boxes/runs_fsq/s170_{direct,k8,k16,k0,cot}_s{seed}.
    python theory/score_s170.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
N_STAR_FIXED = 3  # note 170-3 (from DIRECT alone: s1 3, s2 3); was None = blinded, DIRECT only
CH = 1.0 / 11
NS = list(range(1, 17))
KARMS = {"k8": 8, "k16": 16, "k0": 0}
BASE = {"task": "nested", "n_max_train": 8, "steps": 48000}
IBASE = dict(BASE, arm="inject", mem_write="top", write_mix_start=2000, write_mix_end=6000)
ERRORS = []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def check(d, a, want):
    bad = {k: a.get(k) for k, v in want.items() if a.get(k) != v}
    if bad:
        ERRORS.append((d, bad))
    return bad


def acc(ab, n):
    v = (ab or {}).get(str(n))
    return None if v is None else float(v)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def flops_n(v, n):
    """compute_by_n[n] is {flops_trunk, sequential_passes} (boxes.core.Meter, trunk; the same convention for every arm)."""
    c = (v.get("compute_by_n") or {}).get(str(n))
    return c.get("flops_trunk") if isinstance(c, dict) else c


def pooled(ab, lo, hi=8):
    vs = [acc(ab, n) for n in range(lo, hi + 1)]
    return None if any(v is None for v in vs) else sum(vs) / len(vs)


def direct_seed(s):
    d = "s170_direct_s%d" % s; r = load(d); a = load(d, "args.json") or {}
    if r is None:
        return {"status": "running" if a else "open"}
    if check(d, a, dict(BASE, arm="direct")):
        return {"status": "error"}
    ab = r.get("acc_by_n") or {}
    if (acc(ab, 1) or 0) < 0.90:
        return {"status": "dropped (did not train)", "acc_by_n": ab}
    fail = [n for n in NS if acc(ab, n) is not None and acc(ab, n) < 0.90]
    return {"status": "done", "acc_by_n": ab, "n_star": fail[0] if fail else None, "acc_by_depth": r.get("acc_by_depth"), "gpu": r.get("train_gpu")}


def boundary():
    seeds = {s: direct_seed(s) for s in (1, 2)}
    st = [v["status"] for v in seeds.values()]
    if any(x in ("open", "running") for x in st):
        return seeds, {"status": "pending"}
    kept = [v for v in seeds.values() if v["status"] == "done"]
    if not kept:
        return seeds, {"status": "no boundary (no DIRECT seed trained)"}
    ns = [v["n_star"] for v in kept]; fin = [n for n in ns if n is not None]
    ns_ = min(fin) if fin else None
    return seeds, {"status": "fixed", "n_star": ns_, "per_seed": ns, "boundary_uncertain": len(set(ns)) > 1,
                   "no_c1_range": ns_ is None or ns_ >= 7}


def kseries(x, kp):
    return ((x.get("acc_by_n_kprime") or {}).get(str(kp))) or {}


def inject_run(arm, s, ns_):
    K = KARMS[arm]; d = "s170_%s_s%d" % (arm, s); r = load(d); a = load(d, "args.json") or {}
    if r is None:
        return {"status": "running" if a else "open"}
    if a.get("mem_check") or a.get("train_n") or str(a.get("device", "")).startswith("cpu"):
        return {"status": "smoke"}
    if check(d, a, dict(IBASE, think_k=K)):
        return {"status": "error"}
    ab = r.get("acc_by_n") or {}; x = r.get("s170") or {}
    out = {"status": "done", "acc_by_n": ab, "pooled": pooled(ab, ns_), "gpu": r.get("train_gpu"), "compute_by_n": r.get("compute_by_n")}
    out["deep"] = all((acc(ab, n) or 0) >= 0.90 for n in range(ns_, 9))
    out["mean_9_12"] = mean([acc(ab, n) for n in range(9, 13)])
    out["depth_general"] = bool(out["mean_9_12"] is not None and out["mean_9_12"] >= 0.80)
    if K == 0:
        return out
    pk = out["pooled"]; p0 = pooled(kseries(x, 0), ns_); p1 = pooled(kseries(x, 1), ns_)
    rem = bool(pk is not None and p0 is not None and p1 is not None and (p0 - CH) <= 0.5 * (pk - CH) and (p1 - CH) <= 0.5 * (pk - CH))
    sk = {t: pooled(v, ns_) for t, v in (x.get("skip_pass") or {}).items() if isinstance(v, dict)}
    drops = [pk - v for v in sk.values() if v is not None] if pk is not None else []
    step_specific = bool(drops and max(drops) >= 0.2 and min(drops) <= 0.05)
    ser = x.get("serial") or {}; nul = x.get("serial_null")
    gaps = {t: v.get("hybrid") - nul for t, v in ser.items() if isinstance(v, dict) and v.get("hybrid") is not None and nul is not None and 1 <= int(t) < K}
    t_best = max(gaps, key=gaps.get) if gaps else None
    donor_share = {t: v.get("donor") for t, v in ser.items() if isinstance(v, dict)}
    m2 = mean([acc(kseries(x, 2 * K), n) for n in range(9, 13)])
    out.update({"pooled_k0": p0, "pooled_k1": p1, "removal": rem, "skip_pooled": sk, "skip_step_specific": step_specific,
                "think_necessary": bool(rem and step_specific), "serial": ser, "serial_null": nul, "serial_gap_by_t": gaps,
                "serial_nest": bool(gaps and max(gaps.values()) >= 0.5), "serial_best_t": t_best, "serial_best_gap": gaps.get(t_best),
                "serial_donor_share": donor_share, "n_serial_items": x.get("n_serial_items"),
                "mean_9_12_at_2K": m2, "more_thinking": None if m2 is None or out["mean_9_12"] is None else bool(m2 >= out["mean_9_12"] + 0.05)})
    return out


def cot_seed(s):
    d = "s170_cot_s%d" % s; r = load(d); a = load(d, "args.json") or {}
    if r is None:
        return {"status": "running" if a else "open"}
    if check(d, a, dict(BASE, arm="textcot")):
        return {"status": "error"}
    ab = r.get("acc_by_n") or {}
    return {"status": "done", "acc_by_n": ab, "mean_9_12": mean([acc(ab, n) for n in range(9, 13)]), "compute_by_n": r.get("compute_by_n"),
            "cot_tokens_by_n": r.get("cot_tokens_by_n")}


def main():
    res = {}
    seeds, b = boundary(); res["direct"] = {"seeds": seeds, "boundary": b}
    print("DIRECT |", b, "|", {s: v.get("acc_by_n") for s, v in seeds.items() if v.get("acc_by_n")})
    if N_STAR_FIXED is None:
        print("BLINDED: N_STAR_FIXED unset (the boundary note is not committed); INJECT / TEXT-COT not read")
        res["reading"] = "blinded"
    elif b.get("status") != "fixed" or b.get("n_star") != N_STAR_FIXED:
        print("ERROR: N_STAR_FIXED %s != DIRECT's n* %s" % (N_STAR_FIXED, b.get("n_star"))); res["reading"] = "error"
    elif b["no_c1_range"]:
        res["reading"] = "no C1 claim from the 1..8 range (n* %s): a dated note sets a deeper range" % b["n_star"]; print("READING:", res["reading"])
    else:
        ns_ = N_STAR_FIXED
        cots = {s: cot_seed(s) for s in (1, 2)}; cd = [v for v in cots.values() if v["status"] == "done"]
        cot_mean = {n: mean([acc(v["acc_by_n"], n) for v in cd]) for n in NS} if len(cd) == 2 else None
        cot_912 = mean([v["mean_9_12"] for v in cd]) if len(cd) == 2 else None
        cot_comp = {n: mean([flops_n(v, n) for v in cd]) for n in range(1, 9)} if len(cd) == 2 else None
        res["cot"] = {"seeds": cots, "mean_9_12": cot_912}
        print("TEXT-COT |", {s: (v["status"], v.get("mean_9_12")) for s, v in cots.items()})
        for arm in KARMS:
            runs = {s: inject_run(arm, s, ns_) for s in (1, 2, 3)}; done = [v for v in runs.values() if v["status"] == "done"]
            s1 = runs[1]; rd = {"runs": runs}
            if arm == "k0":
                rd["k0_pooled"] = s1.get("pooled"); rd["k0_fails"] = None if s1.get("pooled") is None else bool(s1["pooled"] < 0.90)
            else:
                cnt = lambda key: sum(1 for v in done if v.get(key))  # noqa: E731
                rd.update({"n_done": len(done), "deep": cnt("deep"), "think_necessary": cnt("think_necessary"), "serial_nest": cnt("serial_nest"),
                           "depth_general": cnt("depth_general"), "slot1_deep": s1.get("deep")})
                if len(done) >= 2:
                    am = {n: mean([acc(v["acc_by_n"], n) for v in done]) for n in range(1, 9)}
                    acomp = {n: mean([flops_n(v, n) for v in done]) for n in range(1, 9)}
                    if cot_mean:
                        rd["at_least_text_cot_n"] = [n for n in range(1, 9) if am[n] is not None and cot_mean[n] is not None and am[n] >= cot_mean[n] - 0.01
                                                     and acomp.get(n) is not None and cot_comp.get(n) is not None and acomp[n] < cot_comp[n]]
            res[arm] = rd
            print(arm, "|", {k: v for k, v in rd.items() if k != "runs"}, "|",
                  {s: {k: v.get(k) for k in ("status", "pooled", "deep", "think_necessary", "serial_nest", "serial_best_t", "mean_9_12")} for s, v in runs.items() if v["status"] != "open"})
        k0f = res["k0"].get("k0_fails"); lines = []
        for arm in ("k8", "k16"):
            rd = res[arm]
            if rd["slot1_deep"] is False:
                lines.append("%s: slot 1 not DEEP (gate closed)" % arm); continue
            if rd["n_done"] < 3 and not (rd["deep"] >= 2 or rd["n_done"] - rd["deep"] >= 2):
                lines.append("%s: pending (%d / 3 seeds)" % (arm, rd["n_done"])); continue
            c1 = rd["deep"] >= 2 and rd["think_necessary"] >= 2 and rd["serial_nest"] >= 2 and k0f is True
            dg = rd["depth_general"] >= 2
            t = "%s: DEEP %d, THINK-NECESSARY %d, SERIAL-NEST %d, DEPTH-GENERAL %d / %d; K0 fails %s" % (arm, rd["deep"], rd["think_necessary"], rd["serial_nest"],
                                                                                                    rd["depth_general"], rd["n_done"], k0f)
            if c1:
                t += " -> C1 (beyond a measured single pass, n* %d)" % ns_
            elif rd["deep"] >= 2 and rd["think_necessary"] < 2:
                t += " -> DEEP without THINK-NECESSARY (a single-pass solution DIRECT did not find)"
            elif rd["deep"] < 2:
                t += " -> not DEEP"
            if dg:
                t += "; DEPTH-GENERAL" + ("" if cot_912 is None else ("; BEYOND TEXT (CoT 9-12 %.3f)" % cot_912 if cot_912 < 0.80 else "; not beyond text (CoT 9-12 %.3f)" % cot_912))
            if rd.get("at_least_text_cot_n") is not None:
                t += "; AT-LEAST-TEXT-COT at n %s" % rd["at_least_text_cot_n"]
            lines.append(t)
        res["reading"] = lines
        for ln in lines:
            print("READING:", ln)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s170.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
