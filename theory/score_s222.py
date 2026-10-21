"""§222 PROBE-S scorer, part 1 (PS_v1.md at 7174fc3b): the m rule, CERT(S1) / CERT(S2) and the choice rule. Part 2 (A1 readings: PROBE-VALID-S with the
per-t K2 guard, PROBE-GAIN-S, SERIAL-ONSET-S, VB-C1-L-S over score_s218's non-probe components) is added before any A1 probe output exists.
Files (scratch/gate_s222/):
  m_rule.json      {meta, k2: {"t0"/"t1": {"1.5"/"2"/"3": {anti_lin_logreg, cexor_rule, ...}}}, m, flag}
  cert_untrained.json / cert_ref.json   (S1)   and   cert2_untrained.json / cert2_ref.json   (S2, only if S1 fails)
      {meta {card, dtype, tf32, m, dirs_seed, pool_stream, batches, streams{...}, steps, ...},
       arms {"lin_t{t}" / "cexor_t{t}" / "noplant_t{t}" / "shuffled_t{t}": {curve [[step, acc]], LATE, steps_run, extended}, "probe0": {test_acc, n_test, ...}},
       guards (S2 only) {serial_oracle {slope, span, gain_min, ok}, selfcomp {probe1_untrained}}}
Rules (PS_v1 §2-§3): m = smallest of 1.5/2/3 with anti_lin_logreg >= 0.985 and cexor_rule >= 0.985 at t0 AND t1 (else 3, flagged).
CERT(S): on both models, LATE(lin_t) >= 0.90 and LATE(cexor_t) >= 0.90 at t 0/1/2; noplant_t and shuffled_t LATE <= 0.55; probe0 test_acc in 0.50 +- 0.015;
card 4060, fp32, tf32 off, m == m_rule.m, dirs_seed 22750, pool 22700, batches 22701, the pinned streams. S2 also: SERIAL-ORACLE (slope >= 0.8, span >= 4,
gain >= 0.40 in every candidate band) and SELFCOMP (PROBE2(1) on the untrained V_1 <= 0.55). Choice: S1 > S2 > NO CERTIFIABLE PROBE.
    python theory/score_s222.py [--fixtures]
"""
import argparse, json, os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
G = os.path.join(ROOT, "scratch", "gate_s222")
OUT = os.path.join(ROOT, "theory", "results", "score_s222.json")
MS = ("1.5", "2", "3")
STREAMS = {"lin_z": [22751, 22752], "xor_coin": [22753, 22754], "xor_signs": [22755, 22756], "noplant_coin": 22757, "shuffle_perm": 22758}
EPS = 1e-12
CARD_CLASS = "3090"        # Note 222-1: every §222 cert / A1 file on one cloud class, RTX 3090 (fp32)
ACC_TOL = 0.01             # Note 222-1: ACC cross-card check


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def m_rule(f):
    """-> (m, flag, problems)."""
    if f is None:
        return None, None, ["m_rule.json missing (pending)"]
    m = None
    for x in MS:
        if all(f["k2"][t][x]["anti_lin_logreg"] >= 0.985 - EPS and f["k2"][t][x]["cexor_rule"] >= 0.985 - EPS for t in ("t0", "t1")):
            m = x; break
    flag = None if m else "ceiling below 0.985"
    m = m or "3"
    prob = [] if str(f.get("m")) == m and f.get("flag") == flag else ["file m/flag %s/%s != recomputed %s/%s" % (f.get("m"), f.get("flag"), m, flag)]
    return m, flag, prob


def cert_model(c, m, s2=False):
    """-> (ok, reasons, table) for one model's cert file."""
    if c is None:
        return None, ["missing (pending)"], {}
    why, tab = [], {}
    me = c["meta"]
    if CARD_CLASS not in str(me.get("card")) or me.get("dtype") != "fp32" or me.get("tf32") not in (False, "false", None):
        why.append("card/dtype/tf32 %s/%s/%s" % (me.get("card"), me.get("dtype"), me.get("tf32")))
    if str(me.get("m")) != str(m):
        why.append("m %s != m_rule %s" % (me.get("m"), m))
    for k, v in (("dirs_seed", 22750), ("pool_stream", 22700), ("batches", 22701)):
        if me.get(k) != v:
            why.append("%s %s != %s" % (k, me.get(k), v))
    if (me.get("streams") or {}) != STREAMS:
        why.append("streams != pinned (E1)")
    A = c["arms"]
    for t in (0, 1, 2):
        for kind, ok in (("lin", lambda x: x >= 0.90 - EPS), ("cexor", lambda x: x >= 0.90 - EPS), ("noplant", lambda x: x <= 0.55 + EPS), ("shuffled", lambda x: x <= 0.55 + EPS)):
            a = A.get("%s_t%d" % (kind, t))
            if a is None:
                why.append("%s_t%d missing" % (kind, t)); continue
            tab["%s_t%d" % (kind, t)] = a["LATE"]
            if not ok(a["LATE"]):
                why.append("%s_t%d LATE %.3f" % (kind, t, a["LATE"]))
    p0 = A.get("probe0")
    if p0 is None:
        why.append("probe0 missing")
    else:
        tab["probe0"] = p0["test_acc"]
        if abs(p0["test_acc"] - 0.50) > 0.015 + EPS:
            why.append("PROBE(0) test %.4f outside 0.50 +- 0.015" % p0["test_acc"])
    if s2:
        g = c.get("guards") or {}
        so, sc = g.get("serial_oracle") or {}, g.get("selfcomp") or {}
        if not (so.get("slope", 0) >= 0.8 - EPS and so.get("span", 0) >= 4 and so.get("gain_min", 0) >= 0.40 - EPS):
            why.append("S2 SERIAL-ORACLE %s" % so)
        if not (sc.get("probe1_untrained", 1) <= 0.55 + EPS):
            why.append("S2 SELFCOMP %s" % sc)
    return not why, why, tab


def score(gdir=G, with_a1=True):
    out = {"section": 222, "registered": "PS_v1.md at 7174fc3b"}
    m, flag, mp = m_rule(jl(os.path.join(gdir, "m_rule.json")))
    out["m_rule"] = {"m": m, "flag": flag, "problems": mp}
    if mp and "pending" in mp[0]:
        out["choice"] = "pending (m_rule.json)"; return out
    res = {}
    for S, pre, s2 in (("S1", "cert_", False), ("S2", "cert2_", True)):
        r = {mod: cert_model(jl(os.path.join(gdir, "%s%s.json" % (pre, mod))), m, s2) for mod in ("untrained", "ref")}
        res[S] = {mod: {"ok": v[0], "why": v[1], "LATE": v[2]} for mod, v in r.items()}
        res[S]["CERT"] = (None if any(v[0] is None for v in r.values()) else all(v[0] for v in r.values()))
    out["cert"] = res
    if mp:
        out["choice"] = "VOID (m_rule.json inconsistent: %s)" % mp
    elif res["S1"]["CERT"] is None:
        out["choice"] = "pending (S1 certification)"
    elif res["S1"]["CERT"]:
        out["choice"] = "S1"
    elif res["S2"]["CERT"] is None:
        out["choice"] = "pending (S1 failed; S2 certification)"
    elif res["S2"]["CERT"]:
        out["choice"] = "S2"
    else:
        out["choice"] = "NO CERTIFIABLE PROBE (A1's probe readings stay §218's UNREADABLE)"
    out["flag"] = flag
    if with_a1:
        out["A1"] = score_a1(out["choice"]) if out["choice"] in ("S1", "S2") else {"status": "UNREAD / pending (%s)" % out["choice"]}
    return out


# ------------------------------------------------------------------ part 2: A1 readings (PS_v1 §4, E2)
def valid_s(ev):
    """PROBE-VALID-S(t), t 0..2: plant_s lin and cexor acc >= 0.90 and the per-t K2 guard (anti_lin >= 0.985, cexor rule >= 0.97). -> {t: (bool, why)}"""
    out = {}
    for t in (0, 1, 2):
        ps, kg, why = (ev.get("plant_s") or {}).get(str(t)), (ev.get("k2_guard") or {}).get(str(t)), []
        if ps is None or kg is None:
            out[t] = (False, "plant_s / k2_guard t%d missing" % t); continue
        if not (ps["lin"]["acc"] >= 0.90 - EPS and ps["cexor"]["acc"] >= 0.90 - EPS):
            why.append("PLANT-S lin %.3f / cexor %.3f < 0.90" % (ps["lin"]["acc"], ps["cexor"]["acc"]))
        if not (kg["anti_lin"] >= 0.985 - EPS and kg["cexor"] >= 0.97 - EPS):
            why.append("plant at ceiling on A1 (K2 lin %.4f / cexor %.4f)" % (kg["anti_lin"], kg["cexor"]))
        out[t] = (not why, "; ".join(why) or None)
    return out


def readings_s(ev, P_acc, c1, l0, S):
    """PROBE-GAIN-S and SERIAL-ONSET-S on A1 with the selected S (score_s218's formulas; PROBE(t) from the §222 file)."""
    import score_s218 as s218
    o = {}; pr = s218.p_probe(ev); vs = valid_s(ev)
    wp = s218.bad_part(P_acc, "acc") or s218.bad_part(pr, "probe")
    if (ev.get("meta") or {}).get("set") != S:
        wp = "eval_lvb_s222 set %s != selected %s" % ((ev.get("meta") or {}).get("set"), S)
    band = (c1 or {}).get("band") or []
    w = ("C1 band unreadable: %s" % (c1 or {}).get("why") if not (c1 or {}).get("readable") else
         "EMPTY-C1-BAND (band %s)" % band if len(band) < 2 else wp)
    gain = accp = p1 = p12 = None
    if w is None:
        accp = s218.mean(s218.acc_items(P_acc, band)); p1 = s218.probe_pooled(pr, 1, band); p12 = s218.probe_pooled(pr, 12, band); gain = accp - p1
        if not vs[1][0]:
            w = "PROBE-INVALID-S(1) (%s)" % vs[1][1]
        elif p12 < accp - 0.05 - EPS:
            w = "PROBE-INVALID (final state on the band: PROBE(12) %.4f < ACC %.4f - 0.05)" % (p12, accp)
    o["PROBE_GAIN_S"] = s218.rd(s218.ge(gain, 0.35), w, gain=gain, band=band, acc_band=accp, probe1_band=p1, probe12_band=p12)
    ON, slope, never = {}, None, []
    w = wp or ("G-L0-failed band %s" % s218.failed_in(l0, range(4, 9)) if s218.failed_in(l0, range(4, 9)) else None)
    if w is None:
        for D in range(4, 9):
            vals = {t: pr["acc"].get(t, {}).get(D) for t in s218.PROBE_GRID}
            ON[D] = next((t for i, t in enumerate(s218.PROBE_GRID) if all(s218.ge(vals[u], 0.90) for u in s218.PROBE_GRID[i:])), None)
        never = [D for D in ON if ON[D] is None]
        if not never:
            slope = s218.ols(list(ON), [ON[D] for D in ON])
        inv = [t for t in (0, 1, 2) if not vs[t][0]]
        p12_48, acc48 = s218.probe_pooled(pr, 12, range(4, 9)), s218.mean(s218.acc_items(P_acc, range(4, 9)))
        if inv:
            w = "PROBE-INVALID-S(%s)" % inv
        elif p12_48 < acc48 - 0.05 - EPS:
            w = "PROBE-INVALID (final state D 4-8: PROBE(12) %.4f < ACC %.4f - 0.05)" % (p12_48, acc48)
    o["SERIAL_ONSET_S"] = s218.rd(bool(ON) and not never and s218.ge(slope, 0.6) and ON[8] - ON[4] >= 3, w, ON=ON, slope=slope,
                                  never_reached=never, span_8_4=(ON[8] - ON[4]) if ON and not never else None)
    o["PROBE_VALID_S"] = {t: {"valid": v[0], "why": v[1]} for t, v in vs.items()}
    o["DIAG_ALIGN"] = ev.get("diag_align")                                   # descriptive (read only with the probe parts)
    return o


def acc_crosscard(P4060, P3090):
    """Note 222-1: max |acc_d2(3090) - acc_d2(4060)| over D 3..12; > ACC_TOL -> PROBE-GAIN-S uses the 3090 ACC (decided by this check only)."""
    ds = [D for D in range(3, 13) if P4060["a"].get(D) is not None and P3090["a"].get(D) is not None]
    mx = max(abs(P3090["a"][D] - P4060["a"][D]) for D in ds) if ds else None
    use3090 = mx is not None and mx > ACC_TOL + EPS
    return (P3090 if use3090 else P4060), {"max_abs_delta": mx, "use_3090_acc": use3090, "flag": "ACC from the 3090 (cross-card > %.2f)" % ACC_TOL if use3090 else None}


NONPROBE = ("DEEP_L", "SERIAL_STEP_L", "CF_L", "NO_CLOCK_L", "NO_PRECOMPUTE_L")


def score_a1(choice):
    """VB-C1-L-S on A1: score_s218's non-probe components (as read) + PROBE-GAIN-S + SERIAL-ONSET-S."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import score_s218 as s218
    ev = s218.jl(s218.CELL["a1"], "eval_lvb_s222.json")                  # checked FIRST: score_s218 (which reads DIRECT / A1) runs only once §222's A1 file exists
    if ev is None:
        return {"status": "pending (eval_lvb_s222.json missing)"}
    out = s218.score(); a1 = out["cells"]["a1"]
    if a1.get("status") not in ("SCORED", "FAIL"):
        return {"status": "pending (score_s218 A1: %s)" % a1.get("status")}
    if CARD_CLASS not in str((ev.get("meta") or {}).get("card")):
        return {"status": "UNREAD (eval_lvb_s222 card %s, Note 222-1 requires %s)" % ((ev.get("meta") or {}).get("card"), CARD_CLASS)}
    P_acc = s218.p_acc(s218.jl(s218.CELL["a1"], "eval_lvb.json") or {})
    acc3090 = s218.jl(s218.CELL["a1"], "eval_lvb_s222_acc3090.json")
    if acc3090 is None:
        return {"status": "pending (eval_lvb_s222_acc3090.json missing: Note 222-1 ACC cross-card check)"}
    P_acc, xcard = acc_crosscard(P_acc, s218.p_acc(acc3090))
    rs = readings_s(ev, P_acc, a1.get("C1"), out.get("G_L0"), choice); rs["ACC_CROSSCARD"] = xcard
    comp = {c: a1["READINGS"][c]["met"] for c in NONPROBE}
    comp["PROBE_GAIN_S"] = rs["PROBE_GAIN_S"]["met"]; comp["SERIAL_ONSET_S"] = rs["SERIAL_ONSET_S"]["met"]
    vb = all(comp.values()) and not a1.get("censored")
    return {"status": "SCORED", "set": choice, "VB_C1_L_S": vb, "components": comp, "readings": rs, "censored": a1.get("censored"),
            "provenance": "§218 non-probe readings as score_s218 reads them; PROBE(t) from eval_lvb_s222.json under %s" % choice,
            "wording": ("under a certified probe (%s; lin and content-equivariant xor plants certified on two reference models and on A1; plant strength by an "
                        "oracle rule), ... one seed; a successor reading to §218, whose VB-C1-L is not met by a control-design flaw; never C2 / C4" % choice) if vb else None}


# ------------------------------------------------------------------ fixtures
def _mrule(vals):
    k2 = {t: {x: {"anti_lin_logreg": vals[t][x][0], "cexor_rule": vals[t][x][1]} for x in MS} for t in ("t0", "t1")}
    m = None
    for x in MS:
        if all(k2[t][x]["anti_lin_logreg"] >= 0.985 and k2[t][x]["cexor_rule"] >= 0.985 for t in ("t0", "t1")):
            m = x; break
    return {"k2": k2, "m": m or "3", "flag": None if m else "ceiling below 0.985"}


def _cert(m, lin=0.97, cex=0.95, nop=0.50, shf=0.51, p0=0.50, card="NVIDIA GeForce RTX 3090", s2guards=None, drop=None):
    arms = {}
    for t in (0, 1, 2):
        for k, v in (("lin", lin), ("cexor", cex), ("noplant", nop), ("shuffled", shf)):
            arms["%s_t%d" % (k, t)] = {"LATE": v, "curve": [], "steps_run": 10000, "extended": False}
    arms["probe0"] = {"test_acc": p0, "n_test": 10000}
    if drop:
        arms.pop(drop)
    c = {"meta": {"card": card, "dtype": "fp32", "tf32": False, "m": m, "dirs_seed": 22750, "pool_stream": 22700, "batches": 22701, "streams": dict(STREAMS)}, "arms": arms}
    if s2guards is not None:
        c["guards"] = s2guards
    return c


def fixtures():
    import tempfile
    res = []
    good = {t: {"1.5": (0.99, 0.92), "2": (0.998, 0.977), "3": (1.0, 0.999)} for t in ("t0", "t1")}

    def run(files):
        d = tempfile.mkdtemp()
        for k, v in files.items():
            json.dump(v, open(os.path.join(d, k + ".json"), "w"))
        return score(d, with_a1=False)
    mr = _mrule(good)
    res.append(("m rule picks 3 (cexor < 0.985 at 1.5 / 2)", mr["m"] == "3", mr["m"]))
    o = run({"m_rule": mr, "cert_untrained": _cert("3"), "cert_ref": _cert("3")}); res.append(("S1 certifies", o["choice"] == "S1", o["choice"]))
    o = run({"m_rule": mr, "cert_untrained": _cert("3"), "cert_ref": _cert("3", cex=0.88)}); res.append(("S1 fails on ref cexor -> pending S2", o["choice"].startswith("pending (S1 failed"), o["choice"]))
    g2 = {"serial_oracle": {"slope": 0.9, "span": 5, "gain_min": 0.45}, "selfcomp": {"probe1_untrained": 0.51}}
    o = run({"m_rule": mr, "cert_untrained": _cert("3"), "cert_ref": _cert("3", cex=0.88), "cert2_untrained": _cert("3", s2guards=g2), "cert2_ref": _cert("3", s2guards=g2)})
    res.append(("S1 fail, S2 certifies", o["choice"] == "S2", o["choice"]))
    o = run({"m_rule": mr, "cert_untrained": _cert("3"), "cert_ref": _cert("3", cex=0.88), "cert2_untrained": _cert("3", s2guards=g2),
             "cert2_ref": _cert("3", s2guards={"serial_oracle": {"slope": 0.5, "span": 2, "gain_min": 0.3}, "selfcomp": {"probe1_untrained": 0.51}})})
    res.append(("S2 guard fails -> NO CERTIFIABLE PROBE", o["choice"].startswith("NO CERTIFIABLE"), o["choice"]))
    o = run({"m_rule": mr, "cert_untrained": _cert("3", nop=0.6), "cert_ref": _cert("3")}); res.append(("NOPLANT > 0.55 fails S1", o["cert"]["S1"]["CERT"] is False, o["cert"]["S1"]["untrained"]["why"]))
    o = run({"m_rule": mr, "cert_untrained": _cert("3", p0=0.52), "cert_ref": _cert("3")}); res.append(("PROBE(0) outside band fails", o["cert"]["S1"]["CERT"] is False, o["cert"]["S1"]["untrained"]["why"]))
    o = run({"m_rule": mr, "cert_untrained": _cert("3", card="NVIDIA GeForce RTX 4060"), "cert_ref": _cert("3")}); res.append(("wrong card (4060) fails", o["cert"]["S1"]["CERT"] is False, o["cert"]["S1"]["untrained"]["why"]))
    o = run({"m_rule": mr, "cert_untrained": _cert("2"), "cert_ref": _cert("2")}); res.append(("cert at wrong m fails", o["cert"]["S1"]["CERT"] is False, o["cert"]["S1"]["untrained"]["why"][:1]))
    bad = dict(mr); bad["m"] = "2"; o = run({"m_rule": bad, "cert_untrained": _cert("2"), "cert_ref": _cert("2")}); res.append(("m_rule file inconsistent -> VOID", o["choice"].startswith("VOID"), o["choice"]))
    low = _mrule({t: {"1.5": (0.9, 0.8), "2": (0.95, 0.9), "3": (0.98, 0.96)} for t in ("t0", "t1")}); res.append(("no m qualifies -> 3 flagged", bool(low["m"] == "3" and low["flag"]), low["flag"]))
    o = run({"m_rule": mr, "cert_untrained": _cert("3")}); res.append(("ref missing -> pending", o["choice"].startswith("pending (S1"), o["choice"]))
    o = run({}); res.append(("m_rule missing -> pending", o["choice"].startswith("pending (m_rule"), o["choice"]))
    o = run({"m_rule": mr, "cert_untrained": _cert("3", drop="shuffled_t2"), "cert_ref": _cert("3")}); res.append(("missing arm fails", o["cert"]["S1"]["CERT"] is False, o["cert"]["S1"]["untrained"]["why"]))
    # part 2: readings_s on synthetic A1 inputs (score_s218 helpers only; no real file is read)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import score_s218 as s218
    Ds = list(range(3, 13))
    def ev222(prob, lin=0.95, cex=0.93, k2l=0.99, k2c=0.98, sset="S1"):
        pr = {}
        for t_ in s218.PROBE_GRID:
            ab, it = {}, {}
            for D in Ds:
                k = int(round(prob(t_, D) * 100)); lst = [1] * k + [0] * (100 - k); ab[str(D)] = sum(lst) / 100; it[str(D)] = lst
            pr[str(t_)] = {"acc_by_D": ab, "items": it, "steps_run": 10000, "curve": [[8000, 0.9], [10000, 0.9]]}
        return {"meta": {"set": sset}, "probe": pr, "plant_s": {str(t_): {"lin": {"acc": lin}, "cexor": {"acc": cex}} for t_ in (0, 1, 2)},
                "k2_guard": {str(t_): {"anti_lin": k2l, "cexor": k2c} for t_ in (0, 1, 2)}}
    P_acc = {"void": None, "missing": [], "items": {str(D): {str(D + 2): [1] * 97 + [0] * 3} for D in Ds}}
    c1 = {"readable": True, "band": [5, 6, 7, 8]}
    serial = lambda t_, D: 0.97 if t_ >= D else 0.5
    pre = lambda t_, D: 0.97 if t_ >= 1 else 0.5
    r = readings_s(ev222(serial), P_acc, c1, None, "S1"); res.append(("serial model: GAIN-S + ONSET-S met", bool(r["PROBE_GAIN_S"]["met"] and r["SERIAL_ONSET_S"]["met"]), (r["PROBE_GAIN_S"]["status"], r["SERIAL_ONSET_S"]["status"])))
    r = readings_s(ev222(pre), P_acc, c1, None, "S1"); res.append(("precompute model: GAIN-S not met", r["PROBE_GAIN_S"]["status"] == "NOT MET", r["PROBE_GAIN_S"]["status"]))
    r = readings_s(ev222(serial, cex=0.85), P_acc, c1, None, "S1"); res.append(("PLANT-S cexor < 0.90 -> UNREADABLE", r["PROBE_GAIN_S"]["status"].startswith("UNREADABLE (PROBE-INVALID-S"), r["PROBE_GAIN_S"]["status"]))
    r = readings_s(ev222(serial, k2c=0.95), P_acc, c1, None, "S1"); res.append(("K2 ceiling guard -> UNREADABLE", "plant at ceiling" in r["PROBE_GAIN_S"]["status"], r["PROBE_GAIN_S"]["status"]))
    r = readings_s(ev222(serial, sset="S2"), P_acc, c1, None, "S1"); res.append(("file set != selected -> UNREADABLE", r["PROBE_GAIN_S"]["status"].startswith("UNREADABLE"), r["PROBE_GAIN_S"]["status"]))
    r = readings_s(ev222(serial), P_acc, {"readable": True, "band": [8]}, None, "S1"); res.append(("C1 band < 2 depths -> UNREADABLE", r["PROBE_GAIN_S"]["status"].startswith("UNREADABLE (EMPTY-C1-BAND"), r["PROBE_GAIN_S"]["status"]))
    Pa = {"a": {D: 0.95 for D in range(3, 13)}}; Pb = {"a": {D: 0.95 + (0.02 if D == 7 else 0.0) for D in range(3, 13)}}
    use, x = acc_crosscard(Pa, Pb); res.append(("ACC cross-card > 0.01 -> 3090 ACC used", use is Pb and x["use_3090_acc"], x))
    Pc = {"a": {D: 0.955 for D in range(3, 13)}}; use, x = acc_crosscard(Pa, Pc); res.append(("ACC cross-card <= 0.01 -> 4060 ACC kept", use is Pa and not x["use_3090_acc"], x))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(r[1] for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["choice"], o.get("m_rule"))
    json.dump(o, open(OUT, "w", newline=chr(10)), indent=1, default=str)
