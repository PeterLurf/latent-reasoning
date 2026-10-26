"""THEORY §235 D4-DECOR scorer (docs/research_d4chain_2026-10-26/D4_v1.md; Theory is the only scorer).
Runs: LAT-AW d4d_plaw4_s1, CD-AN d4d_pcd4_s1 (one RTX 3090 each); plain DIRECT s228_pdir2c_s1 (CUE-ONLY bar only).
Binding: score_aw4's P-4 gates (LP0 each arm, LP4, CHAIN, LP5-4, CUE-ONLY guard) on these runs, plus
  OP-READ: TEST2 d 4 pass-3 WS on items whose step-3 op != mul_s >= 0.80 (ops from TEST2 regenerated here, joined by sig; any sig missing -> VOID).
PASS = score_aw4 PASS & OP-READ. FAIL labels, first match: VOID; D4-REGRESS (LAT d 3 acc < 0.95); D4-CAP (OP-READ < 0.80; 'none' < 0.55 else
  'partial'); D4-PASS4 (OP-READ ok, CHAIN fails at d 4 pass 4); D4-OTHER.
Descriptive: WS by (pass, op_t) on d 3-4; pass-3 S_2*S_1 mod 11 shortcut rate on wrong non-mul_s d 4 items (P-4: 0.505).
    python -B theory/score_d4.py [--fixtures] [--ref]   (--ref: the same readings on P-4's aw_plaw4_s1 / aw_pcd4_s1, to theory/results/score_d4_ref.json)
"""
import argparse, json, os, sys, tempfile
import numpy as np
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory")); sys.path.insert(0, _HR)
import score_aw4 as A4  # noqa: E402

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_d4.json")
LAT, CD = "d4d_plaw4_s1", "d4d_pcd4_s1"
P = 11


def test2_by_sig():
    """TEST2 regenerated exactly as boxes/inject_ow.build_sets draws it (one Stream, d in TEST_DS order). -> {sig: raw item}."""
    import boxes.inject_ow as IO
    import boxes.ow_words as OW
    s = OW.Stream(IO.SEED_TEST2, frames=OW.FRAME_TRAIN, **IO.OWKW)
    return {it["sig"]: it for d in IO.TEST_DS for it in (s.draw(d) for _ in range(IO.TEST_N))}


def by_op(runs, lat, regen):
    w = A4.jl(os.path.join(runs, lat, "eval_ws.json"))
    if w is None:
        return None
    out = {}
    for d in ("3", "4"):
        it = w["items"][d]; miss = [x for x in it["sig"] if x not in regen]
        if miss:
            return {"sig_missing": len(miss)}
        ops = [regen[x]["ops"] for x in it["sig"]]; V = np.array(it["v"])
        rows = {}
        for t in range(1, int(d) + 1):
            pr = np.array(it["pred_by_pass"][str(t)]); ok = pr == V[:, t - 1]; o_t = np.array([o[t - 1] for o in ops])
            rows[str(t)] = {op: {"acc": float(ok[o_t == op].mean()), "n": int((o_t == op).sum())} for op in sorted(set(o_t.tolist()))}
            rows[str(t)]["non_mul_s"] = {"acc": float(ok[o_t != "mul_s"].mean()) if (o_t != "mul_s").any() else None, "n": int((o_t != "mul_s").sum())}
        out[d] = rows
    it = w["items"]["4"]; V = np.array(it["v"]); p3 = np.array(it["pred_by_pass"]["3"]); o3 = np.array([regen[x]["ops"][2] for x in it["sig"]])
    wrong = (p3 != V[:, 2]) & (o3 != "mul_s"); cf = (V[:, 1] * V[:, 0]) % P
    out["shortcut_rate_d4_p3"] = {"rate": float((p3[wrong] == cf[wrong]).mean()) if wrong.any() else None, "n_wrong_non_mul_s": int(wrong.sum())}
    out["OP_READ_value"] = out["4"]["3"]["non_mul_s"]["acc"]
    return out


def score(runs=R, lat=LAT, cd=CD, regen=None):
    old = (A4.LAT, A4.CD)
    A4.LAT, A4.CD = lat, cd
    try:
        o = A4.score(runs)
    finally:
        A4.LAT, A4.CD = old
    o.update({"section": 235, "stage": "D4-DECOR", "registered": "D4_v1.md", "runs": [lat, cd]})
    if o["verdict"].startswith("pending"):
        return o
    bo = by_op(runs, lat, regen if regen is not None else test2_by_sig()); o["by_op"] = bo
    if bo is None:
        o["verdict"] = "pending (missing eval_ws)"; return o
    if "sig_missing" in bo:
        o["verdict"] = "VOID (TEST2 sig mismatch: %d)" % bo["sig_missing"]; return o
    opr = bo["OP_READ_value"]; o["OP_READ"] = opr is not None and opr >= 0.80
    if o["verdict"].startswith("VOID"):
        return o
    base_pass = o["verdict"].startswith("P-4 PASS")
    if base_pass and o["OP_READ"]:
        o["verdict"] = "D4 PASS" + o["verdict"][len("P-4 PASS"):]; return o
    fails = [] if base_pass else o["verdict"][len("P-4 FAIL ("):-1].split(",")
    if not o["OP_READ"]:
        fails.append("OP-READ")
    if o["acc"]["3"]["lat"] < 0.95:
        lab = "D4-REGRESS"
    elif not o["OP_READ"]:
        lab = "D4-CAP (%s)" % ("none" if (opr or 0) < 0.55 else "partial")
    elif "CHAIN" in fails and o["ws_rows"]["4"]["4"] < 0.50:
        lab = "D4-PASS4"
    else:
        lab = "D4-OTHER"
    o["label"] = lab; o["verdict"] = "D4 FAIL (%s) | %s" % (",".join(fails), lab)
    return o


def fixtures():
    res = []
    ops_mix = ["mul_s"] * 460 + ["add_c"] * 540

    def world(p3_non=0.95, p4=0.95, lat3=1.0, drop_sig=False, **kw):
        d = A4_world(lat3=lat3, **kw); regen = {}
        rng = np.random.default_rng(0)
        for dd in ("1", "2", "3", "4"):
            n = 1000; V = rng.integers(0, P, size=(n, 8)); sig = ["%s_%d" % (dd, i) for i in range(n)]
            ops = [["add_c"] * int(dd) for _ in range(n)]
            if dd == "3":
                ops = [["add_c", "add_c", "mul_s"] for _ in range(n)]
            if dd == "4":
                ops = [["add_c", "add_c", ops_mix[i], "add_c"] for i in range(n)]
            pv = {}
            for t in range(1, 9):
                pr = V[:, t - 1].copy()
                if dd == "4" and t == 3:
                    non = np.array([o[2] != "mul_s" for o in ops]); bad = non & (rng.random(n) > p3_non); pr[bad] = (V[bad, 1] * V[bad, 0]) % P
                    pr[bad & (pr == V[:, 2])] = (pr[bad & (pr == V[:, 2])] + 1) % P
                if dd == "4" and t == 4:
                    bad = rng.random(n) > p4; pr[bad] = (pr[bad] + 1) % P
                pv[str(t)] = pr.tolist()
            ws_items = json.load(open(os.path.join(d, LAT, "eval_ws.json")))
            ws_items.setdefault("items", {})[dd] = {"sig": sig, "ans": V[:, int(dd) - 1].tolist(), "pred_by_pass": pv, "v": V.tolist()}
            ws_items["by_d"][dd] = {str(t): float((np.array(pv[str(t)]) == V[:, t - 1]).mean()) for t in range(1, 9)}
            json.dump(ws_items, open(os.path.join(d, LAT, "eval_ws.json"), "w"))
            for i, s_ in enumerate(sig):
                if not (drop_sig and dd == "4" and i == 0):
                    regen[s_] = {"ops": ops[i]}
        return d, regen

    def A4_world(lat3=1.0, **kw):
        d = tempfile.mkdtemp()
        for run, accs in ((LAT, {"2": 1.0, "3": lat3, "4": kw.get("lat4", 0.85)}), (CD, {"2": 0.1, "3": 0.3, "4": 0.1}), (A4.DIR, {"2": 0.1, "3": 0.1, "4": 0.1})):
            p = os.path.join(d, run); os.makedirs(p)
            json.dump({"by_d": {k: {"acc": v} for k, v in accs.items()}}, open(os.path.join(p, "eval_acc.json"), "w"))
            with open(os.path.join(p, "log.jsonl"), "w") as fh:
                for i in range(1, 161):
                    fh.write(json.dumps({"step": i * 100, "gn_preclip_mean100": 1.0 + 0.1 * (i % 3)}) + "\n")
        p = os.path.join(d, LAT)
        json.dump({"by_d": {}}, open(os.path.join(p, "eval_ws.json"), "w"))
        json.dump({"by_d": {dd: {"intact": {dd: 0.9}, "mean_at_t": {dd: 0.2}} for dd in A4.DS}}, open(os.path.join(p, "eval_lp5h.json"), "w"))
        json.dump({"by_d": {dd: {"intact": {dd: 0.9}, "mean_at_1": {dd: 0.1}} for dd in ("2",) + A4.DS}}, open(os.path.join(p, "eval_lp5.json"), "w"))
        json.dump({"by_d": {dd: {"answer_acc": kw.get("co", 0.1), "majority": 0.1} for dd in ("2",) + A4.DS}}, open(os.path.join(p, "eval_cue_only.json"), "w"))
        json.dump({"by_d": {dd: {"all": {dd: 0.4}} for dd in ("2",) + A4.DS}}, open(os.path.join(p, "eval_cue_wrong.json"), "w"))
        json.dump({"by_d": {dd: {"t_mid": 1 if dd == "3" else 2, "n_pairs": 1000, "swap_k": {}, "swap_mid": {}} for dd in A4.DS}}, open(os.path.join(p, "eval_swap_kd.json"), "w"))
        return d

    d, rg = world(); o = score(d, regen=rg); res.append(("op read + chain -> D4 PASS", o["verdict"].startswith("D4 PASS"), o["verdict"]))
    d, rg = world(p3_non=0.0); o = score(d, regen=rg)
    res.append(("shortcut kept -> D4-CAP (none), shortcut rate ~1", "D4-CAP (none)" in o["verdict"] and o["by_op"]["shortcut_rate_d4_p3"]["rate"] > 0.85, o["verdict"]))
    d, rg = world(p3_non=0.65); o = score(d, regen=rg); res.append(("non-mul_s 0.65 -> D4-CAP (partial)", "D4-CAP (partial)" in o["verdict"], o["verdict"]))
    d, rg = world(p4=0.1); o = score(d, regen=rg); res.append(("pass 4 lags -> D4-PASS4", o["verdict"].endswith("D4-PASS4") and "CHAIN" in o["verdict"], o["verdict"]))
    d, rg = world(lat3=0.9, p3_non=0.0); o = score(d, regen=rg); res.append(("d3 0.90 + shortcut -> D4-REGRESS first", o["verdict"].endswith("D4-REGRESS"), o["verdict"]))
    d, rg = world(lat3=0.9); o = score(d, regen=rg); res.append(("d3 0.90 is a FAIL label only -> D4 PASS", o["verdict"].startswith("D4 PASS"), o["verdict"]))
    d, rg = world(drop_sig=True); o = score(d, regen=rg); res.append(("sig missing -> VOID", o["verdict"].startswith("VOID (TEST2"), o["verdict"]))
    d, rg = world(co=0.5); o = score(d, regen=rg); res.append(("cue alone -> VOID", o["verdict"].startswith("VOID"), o["verdict"]))
    d, rg = world(); os.remove(os.path.join(d, LAT, "eval_lp5h.json")); o = score(d, regen=rg); res.append(("missing -> pending", o["verdict"].startswith("pending"), o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); ap.add_argument("--ref", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(lat="aw_plaw4_s1", cd="aw_pcd4_s1") if a.ref else score()
    print(o["verdict"], "OP-READ %s" % (o.get("by_op") or {}).get("OP_READ_value"))
    out = OUT.replace(".json", "_ref.json") if a.ref else OUT
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(o, open(out, "w", newline=chr(10)), indent=1, default=str)
