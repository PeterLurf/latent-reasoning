"""THEORY §236 REG-CHAIN scorer (docs/research_regchain_2026-10-27/RC_v1.md; Theory is the only scorer).
Runs: LAT-AW rc_plaw4_s1 / rc_plaw4_s2, CD-AN rc_pcd4_s1; CUE-ONLY bar from s228_pdir2c_s1_reg + _reg_bs50 (plain DIRECT, REG eval), read at the MIN over the two (Note 236-4).
Per LAT seed, binding: LP0 (seed + CD-AN); ACC (d 3 >= 0.60, d 4 >= 0.50, each >= CD-AN + 0.25); CHAIN (WS >= 0.50 at passes 2..d, d 3-4);
  OP-ALL (pass-3 WS pooled d 3-4 >= 0.80 for every op class: 6 regular + add_s j=0 + add_s j>=1; any class n < 50 -> VOID); LP5-4 (pooled >= 0.30);
  SWAP-MID d 3 t 1 (follow_U >= 0.50 and >= floor + 0.30); CUE-ONLY guard d 3-4.
Seed labels (first match): VOID; PASS; CAP (none / partial); MERGED (pass t < d prints v_{t+1} >= 0.50); PASS4-LAG; CHAIN-OTHER; GATE-OTHER.
Overall: PASS (2 PASS) / SPLIT (1 PASS) / FAIL (no PASS, >= 1 non-VOID) / VOID (both VOID).
Ops come from TEST2-REG regenerated here (inject_ow.SEED_TEST2_REG, OW.Stream(..., mul_s_p=0)), joined by sig.
    python -B theory/score_rc.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
import numpy as np
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory")); sys.path.insert(0, _HR)
import score_aw4 as A4  # noqa: E402

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_rc.json")
LATS, CD, DIRR = ("rc_plaw4_s1", "rc_plaw4_s2"), "rc_pcd4_s1", "s228_pdir2c_s1_reg"
DIR_VARIANTS = (DIRR, DIRR + "_bs50")
CLASSES = ("add_c", "sub_c", "mul_c", "add_v", "sub_v", "mul_v", "add_s_j0", "add_s_j1+")


def regen_reg():
    """TEST2-REG as build_sets draws it (one Stream, d in TEST_DS order) -> {sig: raw item}; None if the build has not exported it yet."""
    import boxes.inject_ow as IO
    import boxes.ow_words as OW
    if not hasattr(IO, "SEED_TEST2_REG"):
        return None
    s = OW.Stream(IO.SEED_TEST2_REG, frames=OW.FRAME_TRAIN, **{**IO.OWKW, "mul_s_p": 0.0})
    return {it["sig"]: it for d in IO.TEST_DS for it in (s.draw(d) for _ in range(IO.TEST_N))}


def op_class(it, t):
    op = it["ops"][t - 1]
    if op == "add_s":
        return "add_s_j0" if it["src_j"][t - 1] == 0 else "add_s_j1+"
    return op


def seed_readings(runs, lat, regen):
    old = (A4.LAT, A4.CD, A4.DIR)
    A4.LAT, A4.CD, A4.DIR = lat, CD, DIRR
    try:
        o = A4.score(runs)
    finally:
        A4.LAT, A4.CD, A4.DIR = old
    o["run"] = lat
    if o["verdict"].startswith("pending"):
        o["label"] = "pending"; return o
    # Note 236-4: the DIRECT bar is read at the MIN over the reference's eval_bs variants (stricter guard); each variant must exist
    co = A4.jl(os.path.join(runs, lat, "eval_cue_only.json"))["by_d"]; bars = {}
    for d in A4.DS:
        accs = []
        for dr in DIR_VARIANTS:
            j = A4.jl(os.path.join(runs, dr, "eval_acc.json"))
            if j is None:
                o["label"] = "pending"; o["verdict"] = "pending (missing %s/eval_acc)" % dr; return o
            accs.append(j["by_d"][d]["acc"])
        bars[d] = max(co[d]["majority"], min(accs)) + 0.05
    o["cue_only_bars"] = bars; o["CUE_ONLY_ok"] = all(co[d]["answer_acc"] <= bars[d] for d in A4.DS)
    w = A4.jl(os.path.join(runs, lat, "eval_ws.json"))["items"]
    ok3, cls3, merged = [], [], {}
    for d in ("3", "4"):
        it = w[d]; miss = sum(x not in regen for x in it["sig"])
        if miss:
            o["label"] = "VOID"; o["void_why"] = "TEST2-REG sig mismatch %d" % miss; return o
        V = np.array(it["v"]); p3 = np.array(it["pred_by_pass"]["3"])
        ok3 += (p3 == V[:, 2]).tolist(); cls3 += [op_class(regen[x], 3) for x in it["sig"]]
        merged[d] = {str(t): float((np.array(it["pred_by_pass"][str(t)]) == V[:, t]).mean()) for t in range(1, int(d))}  # v[t] = values[t + 1]
    ok3, cls3 = np.array(ok3), np.array(cls3)
    per = {c: {"acc": float(ok3[cls3 == c].mean()) if (cls3 == c).any() else None, "n": int((cls3 == c).sum())} for c in CLASSES}
    o["op_all"] = per; o["merged_rows"] = merged
    sm = A4.jl(os.path.join(runs, lat, "eval_swap_kd.json"))["by_d"]["3"]
    s1 = sm["swap_mid"]["3"]; o["swap_mid_d3_t1"] = s1; o["swap_mid_d3_t2"] = sm.get("swap_mid_t2")
    small = [c for c, v in per.items() if v["n"] < 50]
    if small:
        o["label"] = "VOID"; o["void_why"] = "op class n < 50: %s" % small; return o
    if not o["CUE_ONLY_ok"]:
        o["label"] = "VOID"; o["void_why"] = "CUE-ONLY d 3-4"; return o
    mn = min(v["acc"] for v in per.values())
    g = {"LP0": o["LP0"]["lat"][0] and o["LP0"]["cd"][0], "ACC": o["LP4"], "CHAIN": o["CHAIN"], "OP-ALL": mn >= 0.80, "LP5-4": o["LP5_4"],
         "SWAP-MID": s1["follow_U"] >= 0.50 and s1["follow_U"] >= s1["floor"] + 0.30}
    o["gates"] = g; o["op_all_min"] = mn
    if all(g.values()):
        o["label"] = "PASS"
    elif not g["OP-ALL"]:
        o["label"] = "CAP (%s)" % ("partial" if mn >= 0.55 else "none")
    elif any(v >= 0.50 for r in merged.values() for v in r.values()):
        o["label"] = "MERGED"
    elif not g["CHAIN"] and o["ws_rows"]["4"]["4"] < 0.50 and all(v >= 0.50 for d in ("3", "4") for t, v in o["ws_rows"][d].items() if not (d == "4" and t == "4")):
        o["label"] = "PASS4-LAG"
    elif not g["CHAIN"]:
        o["label"] = "CHAIN-OTHER"
    else:
        o["label"] = "GATE-OTHER (%s)" % ",".join(k for k, v in g.items() if not v)
    return o


def score(runs=R, regen=None):
    regen = regen if regen is not None else regen_reg()
    o = {"section": 236, "registered": "RC_v1.md", "runs": list(LATS) + [CD, DIRR]}
    if regen is None:
        o["verdict"] = "pending (inject_ow.SEED_TEST2_REG not exported)"; return o
    seeds = {lat: seed_readings(runs, lat, regen) for lat in LATS}; o["seeds"] = seeds
    labs = [s["label"] for s in seeds.values()]
    if "pending" in labs:
        o["verdict"] = "pending (%s)" % "; ".join(s["verdict"] for s in seeds.values() if s["label"] == "pending"); return o
    npass = labs.count("PASS")
    if npass == 2:
        v = "REG-CHAIN PASS"
    elif npass == 1:
        v = "REG-CHAIN SPLIT"
    elif all(lb == "VOID" for lb in labs):
        v = "REG-CHAIN VOID"
    else:
        v = "REG-CHAIN FAIL"
    o["verdict"] = "%s | %s" % (v, " / ".join("%s: %s" % (k[-2:], s["label"]) for k, s in seeds.items()))
    return o


def fixtures():
    res = []
    rng0 = np.random.default_rng(0)

    def world(seedcfg=None, co=0.1, dir100=0.1, dir50=0.1):
        """seedcfg: {lat: dict(p3_bad_class=None|cls, p3_bad=0..1, p4=1.0, merge=False, follow=0.8, lat4=0.85)}"""
        seedcfg = seedcfg or {}
        d = tempfile.mkdtemp(); regen = {}
        cls_cycle = ["add_c", "sub_c", "mul_c", "add_v", "sub_v", "mul_v", "add_s"]
        base = {}
        for dd in ("1", "2", "3", "4"):
            n = 1000; V = rng0.integers(0, 11, size=(n, 8)); sig = ["%s_%d" % (dd, i) for i in range(n)]
            items = []
            for i in range(n):
                ops = [cls_cycle[(i + k) % 7] for k in range(int(dd))]
                src = [(1 if (i % 2) else 0) if op == "add_s" else None for op in ops]
                items.append({"ops": ops, "src_j": src}); regen[sig[i]] = items[-1]
            base[dd] = (V, sig, items)
        for run, accs in [(lat, {"2": 1.0, "3": 0.95, "4": (seedcfg.get(lat) or {}).get("lat4", 0.85)}) for lat in LATS] + [(CD, {"2": 0.1, "3": 0.3, "4": 0.1}), (DIRR, {"2": dir100, "3": dir100, "4": dir100}), (DIRR + "_bs50", {"2": dir50, "3": dir50, "4": dir50})]:
            p = os.path.join(d, run); os.makedirs(p)
            json.dump({"by_d": {k: {"acc": v} for k, v in accs.items()}}, open(os.path.join(p, "eval_acc.json"), "w"))
            with open(os.path.join(p, "log.jsonl"), "w") as fh:
                for i in range(1, 161):
                    fh.write(json.dumps({"step": i * 100, "gn_preclip_mean100": 1.0 + 0.1 * (i % 3)}) + "\n")
        for lat in LATS:
            c = {"p3_bad_class": None, "p3_bad": 0.0, "p4": 1.0, "merge": False, "follow": 0.8}; c.update(seedcfg.get(lat) or {})
            p = os.path.join(d, lat); ws = {"by_d": {}, "items": {}}
            for dd, (V, sig, items) in base.items():
                pv = {}
                for t in range(1, 9):
                    pr = V[:, t - 1].copy()
                    if c["merge"] and dd == "3" and t == 2:
                        pr = V[:, 2].copy()
                    if dd in ("3", "4") and t == 3 and c["p3_bad_class"]:
                        bad = np.array([op_class(it, 3) == c["p3_bad_class"] for it in items]) & (rng0.random(len(items)) < c["p3_bad"])
                        pr[bad] = (pr[bad] + 1) % 11
                    if dd == "4" and t == 4:
                        bad = rng0.random(len(items)) > c["p4"]; pr[bad] = (pr[bad] + 1) % 11
                    pv[str(t)] = pr.tolist()
                ws["items"][dd] = {"sig": sig, "pred_by_pass": pv, "v": V.tolist()}
                ws["by_d"][dd] = {str(t): float((np.array(pv[str(t)]) == V[:, t - 1]).mean()) for t in range(1, 9)}
            json.dump(ws, open(os.path.join(p, "eval_ws.json"), "w"))
            json.dump({"by_d": {dd: {"intact": {dd: 0.9}, "mean_at_t": {dd: 0.2}} for dd in A4.DS}}, open(os.path.join(p, "eval_lp5h.json"), "w"))
            json.dump({"by_d": {dd: {"intact": {dd: 0.9}, "mean_at_1": {dd: 0.1}} for dd in ("2",) + A4.DS}}, open(os.path.join(p, "eval_lp5.json"), "w"))
            json.dump({"by_d": {dd: {"answer_acc": co, "majority": 0.1} for dd in ("2",) + A4.DS}}, open(os.path.join(p, "eval_cue_only.json"), "w"))
            json.dump({"by_d": {dd: {"all": {dd: 0.4}} for dd in ("2",) + A4.DS}}, open(os.path.join(p, "eval_cue_wrong.json"), "w"))
            json.dump({"by_d": {"3": {"t_mid": 1, "swap_mid": {"3": {"follow_U": c["follow"], "floor": 0.2}}, "swap_mid_t2": {"follow_U": 0.7, "floor": 0.2}},
                                "4": {"t_mid": 2, "swap_mid": {"4": {"follow_U": 0.6, "floor": 0.2}}}}}, open(os.path.join(p, "eval_swap_kd.json"), "w"))
        return d, regen

    d, rg = world(); o = score(d, rg); res.append(("both chain -> PASS", o["verdict"].startswith("REG-CHAIN PASS"), o["verdict"]))
    d, rg = world({"rc_plaw4_s2": {"p3_bad_class": "add_s_j1+", "p3_bad": 0.9}}); o = score(d, rg)
    res.append(("s2 fails add_s j>=1 at pass 3 -> SPLIT, s2 CAP (none)", o["verdict"].startswith("REG-CHAIN SPLIT") and "s2: CAP (none)" in o["verdict"], o["verdict"]))
    d, rg = world({"rc_plaw4_s1": {"p3_bad_class": "mul_c", "p3_bad": 0.3}, "rc_plaw4_s2": {"p4": 0.1}}); o = score(d, rg)
    res.append(("s1 CAP partial, s2 PASS4-LAG -> FAIL", o["verdict"].startswith("REG-CHAIN FAIL") and "s1: CAP (partial)" in o["verdict"] and "s2: PASS4-LAG" in o["verdict"], o["verdict"]))
    d, rg = world({lat: {"merge": True} for lat in LATS}); o = score(d, rg); res.append(("pass 2 prints v_3 -> MERGED", o["verdict"].count("MERGED") == 2, o["verdict"]))
    d, rg = world({"rc_plaw4_s1": {"follow": 0.4}}); o = score(d, rg); res.append(("swap not followed -> GATE-OTHER (SWAP-MID)", "s1: GATE-OTHER (SWAP-MID)" in o["verdict"], o["verdict"]))
    d, rg = world(co=0.5); o = score(d, rg); res.append(("cue alone -> both VOID", o["verdict"].startswith("REG-CHAIN VOID"), o["verdict"]))
    d, rg = world(co=0.17, dir100=0.13, dir50=0.11); o = score(d, rg)
    res.append(("bar at min over bs variants (0.16 < 0.17 < 0.18) -> VOID", o["verdict"].startswith("REG-CHAIN VOID"), o["verdict"]))
    d, rg = world(); import shutil; shutil.rmtree(os.path.join(d, DIRR + "_bs50")); o = score(d, rg); res.append(("missing bs50 ref -> pending", o["verdict"].startswith("pending"), o["verdict"]))
    d, rg = world(); rg = {k: v for k, v in rg.items() if k != "4_0"}; o = score(d, rg); res.append(("sig missing -> seed VOID", o["verdict"].startswith("REG-CHAIN VOID"), o["verdict"]))
    d, rg = world(); os.remove(os.path.join(d, LATS[1], "eval_lp5h.json")); o = score(d, rg); res.append(("missing part -> pending", o["verdict"].startswith("pending"), o["verdict"]))
    d, rg = world(); rg2 = {k: {"ops": ["add_c"] * len(v["ops"]), "src_j": [None] * len(v["ops"])} for k, v in rg.items()}; o = score(d, rg2)
    res.append(("op class n < 50 -> VOID", o["verdict"].startswith("REG-CHAIN VOID"), o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true")
    ap.add_argument("--s237", action="store_true", help="§237 REG-BOOT: the same readings on rcb_plaw4_s1 / s2 + rcb_pcd4_s1, to score_rc_s237.json")
    a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    out = OUT
    if a.s237:
        LATS, CD = ("rcb_plaw4_s1", "rcb_plaw4_s2"), "rcb_pcd4_s1"; out = OUT.replace(".json", "_s237.json")
    o = score(); o["section"] = 237 if a.s237 else 236; print(o["verdict"])
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(o, open(out, "w", newline=chr(10)), indent=1, default=str)
