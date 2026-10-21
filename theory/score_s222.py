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
    if "4060" not in str(me.get("card")) or me.get("dtype") != "fp32" or me.get("tf32") not in (False, "false", None):
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


def score(gdir=G):
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
    return out


# ------------------------------------------------------------------ fixtures
def _mrule(vals):
    k2 = {t: {x: {"anti_lin_logreg": vals[t][x][0], "cexor_rule": vals[t][x][1]} for x in MS} for t in ("t0", "t1")}
    m = None
    for x in MS:
        if all(k2[t][x]["anti_lin_logreg"] >= 0.985 and k2[t][x]["cexor_rule"] >= 0.985 for t in ("t0", "t1")):
            m = x; break
    return {"k2": k2, "m": m or "3", "flag": None if m else "ceiling below 0.985"}


def _cert(m, lin=0.97, cex=0.95, nop=0.50, shf=0.51, p0=0.50, card="NVIDIA GeForce RTX 4060", s2guards=None, drop=None):
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
        return score(d)
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
    o = run({"m_rule": mr, "cert_untrained": _cert("3", card="NVIDIA GeForce RTX 3070"), "cert_ref": _cert("3")}); res.append(("wrong card fails", o["cert"]["S1"]["CERT"] is False, o["cert"]["S1"]["untrained"]["why"]))
    o = run({"m_rule": mr, "cert_untrained": _cert("2"), "cert_ref": _cert("2")}); res.append(("cert at wrong m fails", o["cert"]["S1"]["CERT"] is False, o["cert"]["S1"]["untrained"]["why"][:1]))
    bad = dict(mr); bad["m"] = "2"; o = run({"m_rule": bad, "cert_untrained": _cert("2"), "cert_ref": _cert("2")}); res.append(("m_rule file inconsistent -> VOID", o["choice"].startswith("VOID"), o["choice"]))
    low = _mrule({t: {"1.5": (0.9, 0.8), "2": (0.95, 0.9), "3": (0.98, 0.96)} for t in ("t0", "t1")}); res.append(("no m qualifies -> 3 flagged", bool(low["m"] == "3" and low["flag"]), low["flag"]))
    o = run({"m_rule": mr, "cert_untrained": _cert("3")}); res.append(("ref missing -> pending", o["choice"].startswith("pending (S1"), o["choice"]))
    o = run({}); res.append(("m_rule missing -> pending", o["choice"].startswith("pending (m_rule"), o["choice"]))
    o = run({"m_rule": mr, "cert_untrained": _cert("3", drop="shuffled_t2"), "cert_ref": _cert("3")}); res.append(("missing arm fails", o["cert"]["S1"]["CERT"] is False, o["cert"]["S1"]["untrained"]["why"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(r[1] for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["choice"], o.get("m_rule"))
    json.dump(o, open(OUT, "w", newline=chr(10)), indent=1, default=str)
