"""THEORY §177 scoring (rules fixed in §177; written before any §177 implementation or run; Theory is the only scorer). Cloud cells are gated on the trigger
(§169 K8 and §170 K8 both fail); local proto177_* previews are read by no rule.
Arms: (a) tidx = PASS-INDEX on §169's INJECT (mult mix, K 8); (d) fadetidx = SCHEDULE-FADE + PASS-INDEX (K = 4n; INJECT's decoder throughout; SCAFFOLD gate:
  s177.acc_scaffold_12k (4x4 at p = 0, 12k) >= 0.90, else uninformative; missing -> pending - notes 177-2 / 177-3); rules §169's at the arm's rule K:
  EXACT 4x4 >= 0.99; LENGTH-GENERAL(arm) >= 2 / 3 EXACT with ext_4x8 >= 0.90; THINK-NECESSARY K' 0 / 1 each <= 0.5 x acc(K) with step-specific skip damage;
  C1 also needs §164 DIRECT-MIX < 0.95.
(f) kmatch = STEP-MATCHED K on §170 nested (K = n + U{0..2}); rules at K' = n + 1: HIT = DEEP (>= 0.90 at every n 3..8) + THINK-NECESSARY (168-1 form,
  chance 1/11, K' 0 / 1 vs n + 1, step-specific skip damage), labelled by DISTINCT-PASSES (note 177-1: a qualifier, not a conjunct); BEYOND = mean n 9-12
  >= 0.80; FAIL = pooled 3..8 <= DIRECT
  pooled + 0.05 with delta cos >= 0.96 at passes 2-4 ("open (under-trained)" if train_loss_tail.second_half < 0.98 x first_half); two FAIL seeds close
  fixed / oracle-count K only.
(g) curpois = CURRICULUM + POISSON K (gated on an (f) HIT): HIT at K' = 10 (DEEP + THINK-NECESSARY vs 10; DISTINCT-PASSES labels it); BEYOND at K' = 16.
MOVES-MECHANISM (note 173-1, descriptive) from eval_s177_dir.json: delta cos < 0.9 at passes 2, 3, 4 AND a step probe >= 0.5 (mult: some t >= 2, c >= 2;
nested: some t >= 2).
Dirs boxes/runs_fsq/s177_{tidx,fadetidx,kmatch,curpois}_s{seed}.
    python theory/score_s177.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s164 as S164  # noqa: E402
import score_s169 as S169  # noqa: E402
import score_s170 as S170  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
CH = 1.0 / 11
MULT_ARMS = {"tidx": dict(S169.BASE, think_k=8, pass_index="sin_gain"), "fadetidx": dict(S169.BASE, think_k="4n", pass_index="sin_gain", fade="12000,36000")}   # note 177-4
NEST_ARMS = {"kmatch": dict(S170.IBASE, k_window="0,2"), "curpois": dict(S170.IBASE, k_poisson="6,2,10", curriculum=0.95)}
NEST_RULE_K = {"kmatch": ("n+1", "n+1"), "curpois": ("10", "16")}   # (HIT K', BEYOND K')
ERRORS = []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def header(d, base):
    r = load(d); a = load(d, "args.json") or {}
    if r is None:
        return None, {"status": "running" if a else "open"}
    if a.get("mem_check") or a.get("train_n") or str(a.get("device", "")).startswith("cpu"):
        return None, {"status": "smoke"}
    bad = {k: a.get(k) for k, v in base.items() if a.get(k) != v}
    if bad:
        ERRORS.append((d, bad)); return None, {"status": "error", "error": bad}
    return r, {"status": "done", "gpu": r.get("train_gpu")}


def mechanism(d, kind):
    x = load(d, "eval_s177_dir.json")
    if not x:
        return {"moves_mechanism": None}
    dc = x.get("delta_cos") or {}
    vals = [dc.get(t) for t in ("2", "3", "4")]
    distinct = None if any(v is None for v in vals) else all(v < 0.9 for v in vals)
    sp = x.get("step_probe_delta") or {}
    if kind == "mult":
        content = any(float(acc) >= 0.5 for t, cs in sp.items() if int(t) >= 2 and isinstance(cs, dict) for c, acc in cs.items() if int(c) >= 2 and acc is not None)
    else:
        content = any(acc is not None and float(acc) >= 0.5 for t, acc in sp.items() if int(t) >= 2)
    return {"distinct_passes": distinct, "per_step_content": bool(content) if sp else None, "delta_cos_2_4": vals,
            "moves_mechanism": None if distinct is None or not sp else bool(distinct and content), "cos_at_init": x.get("cos_at_init")}


def mult_run(arm, s):
    d = "s177_%s_s%d" % (arm, s); r, out = header(d, MULT_ARMS[arm])
    if r is None:
        return out
    x = r.get("s177") or {}; bk = x.get("acc_by_kprime") or {}; e8 = x.get("ext_4x8_by_kprime") or {}
    t = bk.get("K", r.get("test_acc")); e = e8.get("K", (r.get("ext_4x8") or {}).get("test_acc")); a0, a1 = bk.get("0"), bk.get("1")
    rem = bool(t and a0 is not None and a1 is not None and a0 <= 0.5 * t and a1 <= 0.5 * t)
    sk = [t - v for v in (x.get("skip_pass") or {}).values() if isinstance(v, (int, float))] if t is not None else []
    out.update({"test_4x4": t, "ext_4x8": e, "exact": bool((t or 0) >= 0.99), "length_general": bool((t or 0) >= 0.99 and (e or 0) >= 0.90),
                "think_necessary": bool(rem and sk and max(sk) >= 0.2 and min(sk) <= 0.05), "acc_by_p": x.get("acc_by_p"), "acc_scaffold_12k": x.get("acc_scaffold_12k"),
                "more_thinking": None if e8.get("2K") is None or e is None else bool(e8["2K"] >= e + 0.05)})
    out.update(mechanism(d, "mult")); return out


def nest_run(arm, s, direct_pooled):
    d = "s177_%s_s%d" % (arm, s); r, out = header(d, NEST_ARMS[arm])
    if r is None:
        return out
    kh, kb = NEST_RULE_K[arm]; x = r.get("s177") or {}; kp = x.get("acc_by_n_kprime") or {}
    ab = kp.get(kh) or (r.get("acc_by_n") if kh == "n+1" else None) or {}
    pk = S170.pooled(ab, 3); p0 = S170.pooled(kp.get("0") or {}, 3); p1 = S170.pooled(kp.get("1") or {}, 3)
    rem = bool(pk is not None and p0 is not None and p1 is not None and (p0 - CH) <= 0.5 * (pk - CH) and (p1 - CH) <= 0.5 * (pk - CH))
    sk = [S170.pooled(v, 3) for v in (x.get("skip_pass") or {}).values() if isinstance(v, dict)]
    drops = [pk - v for v in sk if v is not None] if pk is not None else []
    tn = bool(rem and drops and max(drops) >= 0.2 and min(drops) <= 0.05)
    deep = all((S170.acc(ab, n) or 0) >= 0.90 for n in range(3, 9))
    mech = mechanism(d, "nest")
    hit = bool(deep and tn)   # note 177-1: DISTINCT-PASSES qualifies a HIT, it is not a conjunct
    abb = kp.get(kb) or (ab if kb == kh else {})
    m912 = mean([S170.acc(abb, n) for n in range(9, 13)])
    tail = x.get("train_loss_tail") or {}
    under = None if not tail else bool(tail.get("second_half") is not None and tail.get("first_half") and tail["second_half"] < 0.98 * tail["first_half"])
    dc = mech.get("delta_cos_2_4") or [None]
    fail = bool(pk is not None and direct_pooled is not None and pk <= direct_pooled + 0.05 and all(v is not None and v >= 0.96 for v in dc))
    out.update({"pooled_3_8": pk, "deep": deep, "think_necessary": tn, "hit": hit,
                "hit_label": None if not hit else {True: "HIT (passes distinct)", False: "HIT (passes not distinct)"}.get(mech.get("distinct_passes"), "HIT (passes unknown)"), "beyond": bool(m912 is not None and m912 >= 0.80), "mean_9_12": m912,
                "fail": fail, "under_trained": under, "compute_by_n": r.get("compute_by_n")}); out.update(mech)
    return out


def main():
    res = {}
    dm = [S164.run("directmix", s_) for s_ in (1, 2)]; dm = [v_["test_4x4"] for v_ in dm if v_.get("status") == "done" and v_.get("test_4x4") is not None]
    dmm = sum(dm) / len(dm) if len(dm) == 2 else None; res["direct_mix_mean_4x4"] = dmm
    dseeds = [S170.direct_seed(s) for s in (1, 2)]; dpool = mean([S170.pooled(v.get("acc_by_n") or {}, 3) for v in dseeds if v.get("status") == "done"])
    res["direct_pooled_3_8"] = dpool
    lines = []
    for arm in MULT_ARMS:
        runs = {s: mult_run(arm, s) for s in (1, 2, 3)}; done = [v for v in runs.values() if v["status"] == "done"]; res[arm] = {"runs": runs}
        if runs[1]["status"] != "done":
            lines.append("%s: pending (s1 %s)" % (arm, runs[1]["status"])); continue
        if arm == "fadetidx":                                               # notes 177-2 / 177-3: SCAFFOLD gate on s177.acc_scaffold_12k
            sc = runs[1].get("acc_scaffold_12k")
            if sc is None:
                lines.append("%s: pending (scaffold field acc_scaffold_12k missing)" % arm); continue
            if sc < 0.90:
                lines.append("%s: uninformative: the scaffold did not train (4x4 at p = 0, 12k: %s)" % (arm, sc)); continue
        if not runs[1]["exact"]:
            lines.append("%s: slot 1 not EXACT (%s); MOVES-MECHANISM %s" % (arm, runs[1]["test_4x4"], runs[1].get("moves_mechanism"))); continue
        n_lg = sum(1 for v in done if v["length_general"]); n_tn = sum(1 for v in done if v["think_necessary"])
        c1 = n_lg >= 2 and n_tn >= 2 and dmm is not None and dmm < 0.95
        lines.append("%s: EXACT s1; LENGTH-GENERAL %d, THINK-NECESSARY %d of %d%s" % (arm, n_lg, n_tn, len(done), " -> C1 (DIRECT-MIX %.3f)" % dmm if c1 else ""))
    for arm in NEST_ARMS:
        runs = {s: nest_run(arm, s, dpool) for s in (1, 2, 3)}; done = [v for v in runs.values() if v["status"] == "done"]; res[arm] = {"runs": runs}
        if not done:
            lines.append("%s: pending" % arm); continue
        hits = sum(1 for v in done if v["hit"]); fails = [v for v in done if v["fail"]]
        closed = [v for v in fails if v.get("under_trained") is False]
        t = "%s: HIT %d, BEYOND %d, FAIL %d of %d" % (arm, hits, sum(1 for v in done if v["beyond"]), len(fails), len(done))
        if arm == "kmatch" and len(closed) >= 2:
            t += " -> fixed / oracle-count K closed (the curriculum / long-training ingredient stays open)"
        elif any(v.get("under_trained") for v in fails):
            t += " (a FAIL with loss still falling reads open (under-trained))"
        lines.append(t)
    res["reading"] = lines
    for ln in lines:
        print("READING:", ln)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s177.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
