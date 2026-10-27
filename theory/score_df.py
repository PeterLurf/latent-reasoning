"""THEORY §238 DAG-FREE scorer (docs/research_dagfree_2026-10-27/DF_v1.md; Theory is the only scorer).
Runs: LAT-AW df_plaw4_s1 (init rcb_plaw4_s1) / df_plaw4_s2 (init rcb_plaw4_s2), CD-AN df_pcd4_s1 (init rcb_pcd4_s1).
Refs (eval-only, DF sets): s228_pdir2c_s1_df + _df_bs50 (CUE-ONLY bar, min over the two); step 0: rcb_plaw4_s1_df / rcb_plaw4_s2_df (ws, acc, opclass, swap_prod).
Per LAT seed, binding: LP0 (seed + CD-AN); PROD-ACC (TEST2-DF d 3-4 items with >= 1 mul_s: LAT >= 0.80 and >= CD-AN + 0.25 on the same items);
  CHAIN (WS >= 0.50 at passes 2..d, d 3-4); OP-ALL+ (pass 3 on TEST2-DF d 3-4 and pass 4 on OPCLASS-DF: every class >= 0.80, mul_s at pass 4 split j 1 / 2;
  any class n < 50 -> VOID); PROD-SWAP (follow_U >= 0.50 and >= floor + 0.30); CUE-ONLY guard d 3-4.
Labels (first match): VOID; PASS [ZERO-SHOT if the step-0 init already has mul_s >= 0.80 at passes 3 and 4 and passes PROD-SWAP]; READ2 (pass named;
  partial if min mul_s >= 0.55); CAP; SWAP-FAIL; MERGED; PASS4-LAG; GATE-OTHER. Overall: PASS (both) / SPLIT / FAIL / VOID (both VOID).
    python -B theory/score_df.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
import numpy as np
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory")); sys.path.insert(0, _HR)
import score_aw4 as A4  # noqa: E402

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_df.json")
LATS = {"df_plaw4_s1": "rcb_plaw4_s1_df", "df_plaw4_s2": "rcb_plaw4_s2_df"}     # run -> its step-0 ref
CD, DIRS = "df_pcd4_s1", ("s228_pdir2c_s1_df", "s228_pdir2c_s1_df_bs50")
REG = ("add_c", "sub_c", "mul_c", "add_v", "sub_v", "mul_v", "add_s_j0", "add_s_j1+")


def regen():
    """-> (TEST2-DF {sig: item}, OPCLASS-DF {sig: item}) drawn as the build does (DF_v1 sec. 2); None if the build has not exported the seeds."""
    import boxes.inject_ow as IO
    import boxes.ow_words as OW
    if not (hasattr(IO, "SEED_TEST2_DF") and hasattr(IO, "SEED_OPCLASS_DF")):
        return None
    kw = {**IO.OWKW, "mul_s_p": 2}
    s = OW.Stream(IO.SEED_TEST2_DF, frames=OW.FRAME_TRAIN, **kw); t2 = {it["sig"]: it for d in IO.TEST_DS for it in (s.draw(d) for _ in range(IO.TEST_N))}
    s = OW.Stream(IO.SEED_OPCLASS_DF, frames=OW.FRAME_TRAIN, **kw); oc = {it["sig"]: it for it in (s.draw(4) for _ in range(3000))}
    return t2, oc


def op_class(it, t, split_mul_j=False):
    op = it["ops"][t - 1]; j = it["src_j"][t - 1]
    if op == "add_s":
        return "add_s_j0" if j == 0 else "add_s_j1+"
    if op == "mul_s":
        return "mul_s_j%d" % j if split_mul_j else "mul_s"
    return op


def _cls_acc(items, rg, t, split):
    if any(x not in rg for x in items["sig"]):
        return None
    V = np.array(items["v"]); ok = np.array(items["pred_by_pass"][str(t)]) == V[:, t - 1]
    cl = np.array([op_class(rg[x], t, split) for x in items["sig"]])
    return {c: {"acc": float(ok[cl == c].mean()), "n": int((cl == c).sum())} for c in sorted(set(cl.tolist()))}


def opall(run, runs, t2, oc):
    """-> (classes dict {'p3': {...}, 'p4': {...}}, err)."""
    w = A4.jl(os.path.join(runs, run, "eval_ws.json")); o = A4.jl(os.path.join(runs, run, "eval_opclass.json"))
    if w is None or o is None:
        return None, "missing ws / opclass"
    p3 = {}
    for d in ("3", "4"):
        r = _cls_acc(w["items"][d], t2, 3, False)
        if r is None:
            return None, "TEST2-DF sig mismatch"
        for c, v in r.items():
            a = p3.setdefault(c, [0.0, 0]); a[0] += v["acc"] * v["n"]; a[1] += v["n"]
    p3 = {c: {"acc": a[0] / a[1], "n": a[1]} for c, a in p3.items()}
    p4 = _cls_acc(o["items"], oc, 4, True)
    if p4 is None:
        return None, "OPCLASS-DF sig mismatch"
    return {"p3": p3, "p4": p4}, None


def prod_acc(run, runs, t2):
    tot, ok = 0, 0
    for d in ("3", "4"):
        a = A4.jl(os.path.join(runs, run, "eval_acc.json")); it = a["items"][d]; km = a["by_d"][d].get("k_main", 8)
        pr = np.array(it["pred"]); pr = pr[:, km] if pr.ndim == 2 else pr; ans = np.array(it["ans"])
        m = np.array([("mul_s" in t2[x]["ops"]) if x in t2 else False for x in it["sig"]])
        tot += int(m.sum()); ok += int((pr == ans)[m].sum())
    return (ok / tot if tot else None), tot


def mul_ok(cls):
    p3 = cls["p3"].get("mul_s", {"acc": 0}); p4 = [v for c, v in cls["p4"].items() if c.startswith("mul_s")]
    return p3["acc"] >= 0.80 and bool(p4) and all(v["acc"] >= 0.80 for v in p4), min([p3["acc"]] + [v["acc"] for v in p4])


def swap_ok(sp):
    return sp["follow_U"] >= 0.50 and sp["follow_U"] >= sp["floor"] + 0.30


def seed_readings(runs, lat, ref0, t2, oc):
    old = (A4.LAT, A4.CD, A4.DIR)
    A4.LAT, A4.CD, A4.DIR = lat, CD, DIRS[0]
    try:
        o = A4.score(runs)
    finally:
        A4.LAT, A4.CD, A4.DIR = old
    o["run"] = lat
    if o["verdict"].startswith("pending"):
        o["label"] = "pending"; return o
    need = [(dr, "eval_acc") for dr in DIRS] + [(lat, "eval_opclass"), (lat, "eval_swap_prod"), (ref0, "eval_ws"), (ref0, "eval_opclass"), (ref0, "eval_swap_prod")]
    miss = ["%s/%s" % k for k in need if A4.jl(os.path.join(runs, k[0], k[1] + ".json")) is None]
    if miss:
        o["label"] = "pending"; o["verdict"] = "pending (missing %s)" % miss; return o
    co = A4.jl(os.path.join(runs, lat, "eval_cue_only.json"))["by_d"]
    bars = {d: max(co[d]["majority"], min(A4.jl(os.path.join(runs, dr, "eval_acc.json"))["by_d"][d]["acc"] for dr in DIRS)) + 0.05 for d in A4.DS}
    guard = all(co[d]["answer_acc"] <= bars[d] for d in A4.DS); o["cue_only_bars"] = bars
    cls, err = opall(lat, runs, t2, oc); cls0, err0 = opall(ref0, runs, t2, oc)
    if err or err0:
        o["label"] = "VOID"; o["void_why"] = err or ("step 0: " + err0); return o
    o["op_all"] = cls; o["op_all_step0"] = cls0
    small = [(p, c) for p in ("p3", "p4") for c, v in cls[p].items() if v["n"] < 50]
    if small:
        o["label"] = "VOID"; o["void_why"] = "class n < 50: %s" % small; return o
    if not guard:
        o["label"] = "VOID"; o["void_why"] = "CUE-ONLY d 3-4"; return o
    pa, npr = prod_acc(lat, runs, t2); pc, _ = prod_acc(CD, runs, t2)
    sp = A4.jl(os.path.join(runs, lat, "eval_swap_prod.json")); sp0 = A4.jl(os.path.join(runs, ref0, "eval_swap_prod.json"))
    mok, mmin = mul_ok(cls); m0ok, m0min = mul_ok(cls0)
    others = min(v["acc"] for p in ("p3", "p4") for c, v in cls[p].items() if not c.startswith("mul_s"))
    g = {"LP0": o["LP0"]["lat"][0] and o["LP0"]["cd"][0], "PROD-ACC": pa is not None and pa >= 0.80 and pa >= (pc or 0) + 0.25, "CHAIN": o["CHAIN"],
         "OP-ALL+": mok and others >= 0.80, "PROD-SWAP": swap_ok(sp)}
    merged = {}
    w = A4.jl(os.path.join(runs, lat, "eval_ws.json"))["items"]
    for d in ("3", "4"):
        V = np.array(w[d]["v"]); merged[d] = {str(t): float((np.array(w[d]["pred_by_pass"][str(t)]) == V[:, t]).mean()) for t in range(1, int(d))}
    o.update({"gates": g, "prod_acc": {"lat": pa, "cd": pc, "n": npr}, "mul_min": mmin, "mul_min_step0": m0min, "others_min": others, "swap_prod": sp, "swap_prod_step0": sp0,
              "partner": "MEM" if sp["follow_dfin"] >= sp["follow_cb"] else "TEXT", "merged_rows": merged})
    zero = m0ok and swap_ok(sp0)
    if all(g.values()):
        o["label"] = "PASS" + (" (ZERO-SHOT)" if zero else "")
    elif not mok and others >= 0.80:
        p3m = cls["p3"].get("mul_s", {"acc": 0})["acc"]; where = "pass 3" if p3m < 0.80 else "pass 4"
        o["label"] = "READ2 (%s; %s)" % (where, "partial" if mmin >= 0.55 else "none")
    elif others < 0.80:
        o["label"] = "CAP"
    elif not g["PROD-SWAP"] and g["PROD-ACC"]:
        o["label"] = "SWAP-FAIL"
    elif any(v >= 0.50 for r in merged.values() for v in r.values()):
        o["label"] = "MERGED"
    elif not g["CHAIN"] and o["ws_rows"]["4"]["4"] < 0.50:
        o["label"] = "PASS4-LAG"
    else:
        o["label"] = "GATE-OTHER (%s)" % ",".join(k for k, v in g.items() if not v)
    return o


def score(runs=R, sets=None):
    sets = sets if sets is not None else regen()
    o = {"section": 238, "registered": "DF_v1.md", "runs": list(LATS) + [CD]}
    if sets is None:
        o["verdict"] = "pending (inject_ow DF seeds not exported)"; return o
    t2, oc = sets
    seeds = {lat: seed_readings(runs, lat, ref0, t2, oc) for lat, ref0 in LATS.items()}; o["seeds"] = seeds
    labs = [s["label"] for s in seeds.values()]
    if "pending" in labs:
        o["verdict"] = "pending (%s)" % "; ".join(s["verdict"] for s in seeds.values() if s["label"] == "pending"); return o
    npass = sum(lb.startswith("PASS") for lb in labs)
    v = "DF PASS" if npass == 2 else "DF SPLIT" if npass == 1 else "DF VOID" if all(lb == "VOID" for lb in labs) else "DF FAIL"
    o["verdict"] = "%s | %s" % (v, " / ".join("%s: %s" % (k[-2:], s["label"]) for k, s in seeds.items())); return o


def fixtures():
    res = []; rng0 = np.random.default_rng(0)
    REGOPS = ["add_c", "sub_c", "mul_c", "add_v", "sub_v", "mul_v", "add_s"]

    def mk_items(prefix, d, n, mul_every):
        items, sig = [], []
        for i in range(n):
            ops = [REGOPS[(i + k) % 7] for k in range(d)]; src = [((1 if i % 2 else 0) if op == "add_s" else None) for op in ops]
            for t in range(3, d + 1):
                if (i + t) % mul_every == 0:
                    ops[t - 1] = "mul_s"; src[t - 1] = 1 + (i % (t - 2))
            items.append({"ops": ops, "src_j": src}); sig.append("%s%d_%d" % (prefix, d, i))
        return sig, items

    def write_ws(path, base, mul_bad, p4=1.0):
        ws = {"by_d": {}, "items": {}}
        for dd, (V, sig, items) in base.items():
            pv = {}
            for t in range(1, 9):
                pr = V[:, t - 1].copy()
                if int(dd) >= 3 and t in (3, 4):
                    bad = np.array([len(it["ops"]) >= t and it["ops"][t - 1] == "mul_s" for it in items]) & (rng0.random(len(items)) < mul_bad)
                    pr[bad] = (pr[bad] + 1) % 11
                if dd == "4" and t == 4:
                    b2 = rng0.random(len(items)) > p4; pr[b2] = (pr[b2] + 1) % 11
                pv[str(t)] = pr.tolist()
            ws["items"][dd] = {"sig": sig, "pred_by_pass": pv, "v": V.tolist()}
            ws["by_d"][dd] = {str(t): float((np.array(pv[str(t)]) == V[:, t - 1]).mean()) for t in range(1, 9)}
        json.dump(ws, open(path, "w")); return ws

    def world(cfg=None, step0_mul_bad=0.9, step0_follow=0.2, co=0.1):
        cfg = cfg or {}; d = tempfile.mkdtemp(); t2, oc = {}, {}
        base = {}
        for dd in (1, 2, 3, 4):
            sig, items = mk_items("t", dd, 1000, 4)
            V = rng0.integers(0, 11, size=(1000, 8)); base[str(dd)] = (V, sig, items); t2.update(zip(sig, items))
        osig, oitems = mk_items("o", 4, 3000, 4); oc.update(zip(osig, oitems)); OV = rng0.integers(0, 11, size=(3000, 8))
        runs_cfg = [(lat, cfg.get(lat, {})) for lat in LATS] + [(r0, {"mul_bad": step0_mul_bad, "follow": step0_follow}) for r0 in LATS.values()]
        for run, c in runs_cfg:
            c = {"mul_bad": 0.0, "follow": 0.8, "p4": 1.0, "prod_wrong": False, **c}
            p = os.path.join(d, run); os.makedirs(p)
            ws = write_ws(os.path.join(p, "eval_ws.json"), base, c["mul_bad"], c["p4"])
            pv4 = OV[:, 3].copy(); bad = np.array([it["ops"][3] == "mul_s" for it in oitems]) & (rng0.random(3000) < c["mul_bad"]); pv4[bad] = (pv4[bad] + 1) % 11
            json.dump({"items": {"sig": osig, "pred_by_pass": {"4": pv4.tolist()}, "v": OV.tolist()}}, open(os.path.join(p, "eval_opclass.json"), "w"))
            json.dump({"n_pairs": 1000, "follow_dfin": c["follow"] * 0.7, "follow_cb": c["follow"] * 0.3, "floor": 0.15, "follow_U": c["follow"], "stay_own": 0.05}, open(os.path.join(p, "eval_swap_prod.json"), "w"))
            acc_items = {}
            for dd in ("1", "2", "3", "4"):
                V, sig, items = base[dd]; ans = V[:, int(dd) - 1]; pr = ans.copy()
                if c["prod_wrong"]:
                    m = np.array(["mul_s" in it["ops"] for it in items]); pr[m] = (pr[m] + 1) % 11
                acc_items[dd] = {"sig": sig, "ans": ans.tolist(), "pred": [[int(x)] * 9 for x in pr]}
            json.dump({"by_d": {dd: {"acc": 0.95, "k_main": 8} for dd in ("1", "2", "3", "4")}, "items": acc_items}, open(os.path.join(p, "eval_acc.json"), "w"))
            if run in LATS:
                with open(os.path.join(p, "log.jsonl"), "w") as fh:
                    for i in range(1, 161):
                        fh.write(json.dumps({"step": i * 100, "gn_preclip_mean100": 1.0 + 0.1 * (i % 3)}) + "\n")
                json.dump({"by_d": {dd: {"intact": {dd: 0.9}, "mean_at_t": {dd: 0.2}} for dd in A4.DS}}, open(os.path.join(p, "eval_lp5h.json"), "w"))
                json.dump({"by_d": {dd: {"intact": {dd: 0.9}, "mean_at_1": {dd: 0.1}} for dd in ("2",) + A4.DS}}, open(os.path.join(p, "eval_lp5.json"), "w"))
                json.dump({"by_d": {dd: {"answer_acc": co, "majority": 0.1} for dd in ("2",) + A4.DS}}, open(os.path.join(p, "eval_cue_only.json"), "w"))
                json.dump({"by_d": {dd: {"all": {dd: 0.4}} for dd in ("2",) + A4.DS}}, open(os.path.join(p, "eval_cue_wrong.json"), "w"))
                json.dump({"by_d": {"3": {"t_mid": 1, "swap_mid": {"3": {"follow_U": 0.8, "floor": 0.2}}}, "4": {"t_mid": 2, "swap_mid": {"4": {"follow_U": 0.6, "floor": 0.2}}}}}, open(os.path.join(p, "eval_swap_kd.json"), "w"))
        for run, a in ((CD, 0.1), (DIRS[0], 0.1), (DIRS[1], 0.1)):
            p = os.path.join(d, run); os.makedirs(p, exist_ok=True)
            items = {dd: {"sig": base[dd][1], "ans": base[dd][0][:, int(dd) - 1].tolist(), "pred": ((base[dd][0][:, int(dd) - 1] + 1) % 11).tolist()} for dd in ("1", "2", "3", "4")}
            json.dump({"by_d": {dd: {"acc": a} for dd in ("1", "2", "3", "4")}, "items": items}, open(os.path.join(p, "eval_acc.json"), "w"))
            if run == CD:
                with open(os.path.join(p, "log.jsonl"), "w") as fh:
                    for i in range(1, 161):
                        fh.write(json.dumps({"step": i * 100, "gn_preclip_mean100": 1.0 + 0.1 * (i % 3)}) + "\n")
        return d, (t2, oc)

    d, st = world(); o = score(d, st); res.append(("products chained, step 0 not -> DF PASS", o["verdict"].startswith("DF PASS") and "ZERO-SHOT" not in o["verdict"], o["verdict"]))
    d, st = world(step0_mul_bad=0.0, step0_follow=0.8); o = score(d, st); res.append(("step 0 already does it -> PASS (ZERO-SHOT)", o["verdict"].count("ZERO-SHOT") == 2, o["verdict"]))
    d, st = world({"df_plaw4_s2": {"mul_bad": 0.9, "prod_wrong": True}}); o = score(d, st)
    res.append(("s2 products fail at pass 3 -> SPLIT, READ2 (pass 3; none)", o["verdict"].startswith("DF SPLIT") and "s2: READ2 (pass 3; none)" in o["verdict"], o["verdict"]))
    d, st = world({lat: {"follow": 0.3} for lat in LATS}); o = score(d, st); res.append(("swap not followed -> SWAP-FAIL", o["verdict"].count("SWAP-FAIL") == 2, o["verdict"]))
    d, st = world({"df_plaw4_s1": {"prod_wrong": True}}); o = score(d, st); res.append(("products wrong at answer only -> GATE-OTHER (PROD-ACC)", "s1: GATE-OTHER (PROD-ACC)" in o["verdict"], o["verdict"]))
    d, st = world(co=0.5); o = score(d, st); res.append(("cue alone -> VOID", o["verdict"].startswith("DF VOID"), o["verdict"]))
    d, st = world(); st = (st[0], {k: v for k, v in st[1].items() if k != "o4_0"}); o = score(d, st); res.append(("opclass sig missing -> VOID", o["verdict"].startswith("DF VOID"), o["verdict"]))
    d, st = world(); os.remove(os.path.join(d, "rcb_plaw4_s1_df", "eval_swap_prod.json")); o = score(d, st); res.append(("missing step-0 part -> pending", o["verdict"].startswith("pending"), o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", newline=chr(10)), indent=1, default=str)
