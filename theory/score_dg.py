"""THEORY §269 DIAGNOSTIC BUNDLE scorer (DG_v1 in THEORY §269; Theory is the only scorer; committed before output; descriptive labels that route backlog lanes).
Five eval parts on bun_s1 (+ bun_s2 confirmation), each its own queue line / JSON:
 eval_triage.json    I5  {"n_fail": N, "n_ok_ctrl": M, "fail": {"text_fix": k1, "state_fix": k2, "both": k3, "neither": k4},
                          "ctrl": {"text_keep": a, "state_keep": b}}   (TEST2 d 8 items whose pass-7 WS is wrong with passes 1-6 right;
                          TEXT = pass 7 run on the item re-rendered as its first 7 steps (S_8's sentence removed, question S_7) with the full run's S_6;
                          STATE = pass 7 on the full text with S_6 from the shortened run; fixed = pass-7 readout == v_7; ctrl = the same on pass-7-correct items)
 eval_keys.json      MC1 {"by_d": {d: {"add"|"delta": {"cos_same_ent": x, "cos_same_val": y, "cos_all": z, "n_pairs": n}}}}   (':' write keys, passes 1..d,
                          per family averaged over heads; same_ent = pairs (items differ) with the same cue entity and different values; same_val = same value,
                          different entity; all = random pairs)
 eval_openvocab.json CG5 {"by_fam": {"base"|"multitoken"|"overlap"|"nounphrase": {"by_d": {d: {"acc": x, "n": n}}}}}   (TEST2 d 1-4 re-rendered; acc at K' 8)
 eval_readuse.json   I2  {"by_d": {d: {"intact": x, "zero": {cls: acc}, "mean": {cls: acc}}}}  cls in COLON / CUE / STEP / OTHEROPS / REST (d 4, d 8; K' 8)
 eval_alphahold.json D12+TH4 {"delta": {"alpha_comp": [6], "alpha_hold": [6], "beta_gt1_share": x}, "add": {"wnorm_comp": x, "wnorm_hold": y,
                          "readnorm_slope_rel": z}}  (TEST2 d 1-6, STD cue, K' 32; comp = passes t <= d, hold = t >= d + 3; per-head medians;
                          readnorm_slope_rel = OLS slope of the additive ':' read norm over t in [d+3, 32] / its mean, pooled)
Labels (pre-stated; descriptive; per seed, s2 confirms):
 I5:  over fixed failing items, share_text = (text_fix + both) / n_fail, share_state = (state_fix + both) / n_fail.
      SELECTION if share_text >= 0.5 and share_state < 0.5; MEMORY if share_state >= 0.5 and share_text < 0.5; JOINT if both >= 0.5;
      COMPUTATION if neither / n_fail >= 0.5; else MIXED. VALID only if ctrl text_keep and state_keep >= 0.95 (interventions not destructive).
 MC1: per family: CONSTANT if cos_all >= 0.9; else NAME if same_ent - same_val >= 0.10; VALUE if same_val - same_ent >= 0.10; else MIXED (read at d 4).
 CG5: OPEN if multitoken, overlap and nounphrase each >= base - 0.05 pooled over d 1-4; CLOSED if multitoken <= 0.5; else PARTIAL.
 I2:  at d 4, drop(cls) = intact - min(zero, mean). COLON-ONLY if drop(COLON) >= 0.5 and every other class drop <= 0.05;
      SENTENCE-USE if drop(STEP) >= 0.2 or drop(OTHEROPS) >= 0.2; else MIXED.
 D12: delta GATED-USED if some head has alpha_comp <= 0.95 and alpha_hold >= alpha_comp + 0.03; ALPHA-INERT if every head alpha_comp >= 0.99; else MIXED.
      additive FREEZE if wnorm_hold <= 0.01 wnorm_comp; ECHO if > 0.1 wnorm_comp; else PARTIAL.
 TH4: BOUNDED if |readnorm_slope_rel| <= 0.01 per pass; else DRIFT.
    python -B theory/score_dg.py [--fixtures]
"""
import argparse, json, math, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_dg.json")
RUNS = ("bun_s1", "bun_s2")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 1.0)
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n); return (c - h, c + h)


def triage(e):
    f = e["fail"]; n = e["n_fail"]
    if not n:
        return {"label": "NO-FAILURES"}
    st = (f["text_fix"] + f["both"]) / n; ss = (f["state_fix"] + f["both"]) / n; sn = f["neither"] / n
    valid = e["ctrl"]["text_keep"] >= 0.95 and e["ctrl"]["state_keep"] >= 0.95
    lab = ("JOINT" if st >= 0.5 and ss >= 0.5 else "SELECTION" if st >= 0.5 else "MEMORY" if ss >= 0.5 else "COMPUTATION" if sn >= 0.5 else "MIXED")
    return {"label": lab if valid else "INVALID (" + lab + ")", "share_text": st, "share_state": ss, "share_neither": sn,
            "ci_text": wilson(f["text_fix"] + f["both"], n), "ci_state": wilson(f["state_fix"] + f["both"], n), "n_fail": n}


def keys(e):
    out = {}
    for fam in ("add", "delta"):
        k = e["by_d"]["4"][fam]
        out[fam] = ("CONSTANT" if k["cos_all"] >= 0.9 else "NAME" if k["cos_same_ent"] - k["cos_same_val"] >= 0.10
                    else "VALUE" if k["cos_same_val"] - k["cos_same_ent"] >= 0.10 else "MIXED")
    return out


def openvocab(e):
    pool = {f: sum(e["by_fam"][f]["by_d"][d]["acc"] * e["by_fam"][f]["by_d"][d]["n"] for d in ("1", "2", "3", "4")) /
            sum(e["by_fam"][f]["by_d"][d]["n"] for d in ("1", "2", "3", "4")) for f in ("base", "multitoken", "overlap", "nounphrase")}
    lab = ("OPEN" if all(pool[f] >= pool["base"] - 0.05 for f in ("multitoken", "overlap", "nounphrase")) else
           "CLOSED" if pool["multitoken"] <= 0.5 else "PARTIAL")
    return {"label": lab, "pooled": pool}


def readuse(e):
    r = e["by_d"]["4"]; drop = {c: r["intact"] - min(r["zero"][c], r["mean"][c]) for c in r["zero"]}
    others = [c for c in drop if c != "COLON"]
    lab = ("COLON-ONLY" if drop["COLON"] >= 0.5 and all(drop[c] <= 0.05 for c in others) else
           "SENTENCE-USE" if drop.get("STEP", 0) >= 0.2 or drop.get("OTHEROPS", 0) >= 0.2 else "MIXED")
    return {"label": lab, "drop_d4": drop}


def alphahold(e):
    dl = e["delta"]; ad = e["add"]
    gated = any(c <= 0.95 and h >= c + 0.03 for c, h in zip(dl["alpha_comp"], dl["alpha_hold"]))
    inert = all(c >= 0.99 for c in dl["alpha_comp"])
    dlab = "GATED-USED" if gated else "ALPHA-INERT" if inert else "MIXED"
    alab = "FREEZE" if ad["wnorm_hold"] <= 0.01 * ad["wnorm_comp"] else "ECHO" if ad["wnorm_hold"] > 0.1 * ad["wnorm_comp"] else "PARTIAL"
    th4 = "BOUNDED" if abs(ad["readnorm_slope_rel"]) <= 0.01 else "DRIFT"
    return {"delta": dlab, "additive": alab, "TH4": th4, "beta_gt1_share": dl.get("beta_gt1_share")}


PARTS = {"triage": ("eval_triage.json", triage), "keys": ("eval_keys.json", keys), "openvocab": ("eval_openvocab.json", openvocab),
         "readuse": ("eval_readuse.json", readuse), "alphahold": ("eval_alphahold.json", alphahold)}


def score(runs=R):
    o = {"section": 269, "seeds": {}}
    for run in RUNS:
        o["seeds"][run] = {}
        for part, (fn, f) in PARTS.items():
            e = jl(os.path.join(runs, run, fn)); o["seeds"][run][part] = "pending" if e is None else f(e)
    s1 = o["seeds"]["bun_s1"]
    o["verdict"] = "; ".join("%s %s" % (p, s1[p] if isinstance(s1[p], str) else s1[p].get("label", s1[p])) for p in PARTS); return o


def fixtures():
    res = []

    def chk(name, got, want):
        res.append((name, got == want, got))
    base_tr = {"n_fail": 100, "n_ok_ctrl": 500, "fail": {"text_fix": 60, "state_fix": 5, "both": 5, "neither": 30}, "ctrl": {"text_keep": 0.98, "state_keep": 0.99}}
    chk("triage selection", triage(base_tr)["label"], "SELECTION")
    chk("triage memory", triage({**base_tr, "fail": {"text_fix": 5, "state_fix": 60, "both": 5, "neither": 30}})["label"], "MEMORY")
    chk("triage computation", triage({**base_tr, "fail": {"text_fix": 10, "state_fix": 10, "both": 5, "neither": 75}})["label"], "COMPUTATION")
    chk("triage joint", triage({**base_tr, "fail": {"text_fix": 10, "state_fix": 10, "both": 50, "neither": 30}})["label"], "JOINT")
    chk("triage invalid", triage({**base_tr, "ctrl": {"text_keep": 0.8, "state_keep": 0.99}})["label"], "INVALID (SELECTION)")
    k = lambda a, b, c: {"cos_same_ent": a, "cos_same_val": b, "cos_all": c, "n_pairs": 1000}
    chk("keys value/constant", keys({"by_d": {"4": {"add": k(0.95, 0.95, 0.95), "delta": k(0.3, 0.6, 0.2)}}}), {"add": "CONSTANT", "delta": "VALUE"})
    chk("keys name", keys({"by_d": {"4": {"add": k(0.5, 0.45, 0.4), "delta": k(0.7, 0.3, 0.2)}}}), {"add": "MIXED", "delta": "NAME"})
    ov = lambda m, o_, n_: {"by_fam": {f: {"by_d": {str(d): {"acc": v, "n": 100} for d in range(1, 5)}} for f, v in (("base", 0.99), ("multitoken", m), ("overlap", o_), ("nounphrase", n_))}}
    chk("openvocab open", openvocab(ov(0.97, 0.96, 0.95))["label"], "OPEN")
    chk("openvocab closed", openvocab(ov(0.3, 0.9, 0.9))["label"], "CLOSED")
    chk("openvocab partial", openvocab(ov(0.8, 0.9, 0.9))["label"], "PARTIAL")
    ru = lambda colon, step, other: {"by_d": {"4": {"intact": 1.0, "zero": {"COLON": 1 - colon, "CUE": 1.0, "STEP": 1 - step, "OTHEROPS": 1 - other, "REST": 1.0},
                                                     "mean": {"COLON": 1 - colon, "CUE": 1.0, "STEP": 1 - step, "OTHEROPS": 1 - other, "REST": 1.0}}}}
    chk("readuse colon-only", readuse(ru(0.9, 0.02, 0.0))["label"], "COLON-ONLY")
    chk("readuse sentence", readuse(ru(0.9, 0.4, 0.0))["label"], "SENTENCE-USE")
    ah = lambda comp, hold, wh, slope: {"delta": {"alpha_comp": comp, "alpha_hold": hold, "beta_gt1_share": 0.1}, "add": {"wnorm_comp": 1.0, "wnorm_hold": wh, "readnorm_slope_rel": slope}}
    chk("alphahold gated/freeze/bounded", alphahold(ah([0.9] + [0.995] * 5, [0.97] + [0.996] * 5, 0.005, 0.001)), {"delta": "GATED-USED", "additive": "FREEZE", "TH4": "BOUNDED", "beta_gt1_share": 0.1})
    chk("alphahold inert/echo/drift", alphahold(ah([0.995] * 6, [0.996] * 6, 0.5, 0.05)), {"delta": "ALPHA-INERT", "additive": "ECHO", "TH4": "DRIFT", "beta_gt1_share": 0.1})
    d = tempfile.mkdtemp()
    for run in RUNS:
        os.makedirs(os.path.join(d, run)); json.dump(base_tr, open(os.path.join(d, run, "eval_triage.json"), "w"))
    o = score(d); chk("score pending parts", ("triage SELECTION" in o["verdict"]) and ("keys pending" in o["verdict"]), True)
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], "->", r[2])
    k_ = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k_, len(res))); return k_ == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
