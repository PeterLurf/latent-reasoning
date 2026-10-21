"""§223 PORT-STATE scorer (PS2_v1.md at 54f63381 + Addenda A / B; Notes 223-1 / 223-2). Written and fixture-checked before any §223 output.
Cells (boxes/runs_fsq/): s223_direct_s1 (DIRECT), s223_ps_s1 (PS = M1 + M2), s223_m1_s1 (M1-only), s223_m2_s1 (M2-only); each with eval_ps.json (layout:
Experiments' final contract, Note 223-2), log.jsonl, args.json.
Readings:
  band (from DIRECT acc_by_d): n* = max d in 1..8 with acc > 0.50 (0 if none); C1 band = {d in max(3, n*+1)..8 : DIRECT acc <= 0.50}; < 3 depths -> G-W0 FAIL
    (every latent reading UNREAD).
  DECODE-VAL (M1 cells): pooled value_acc over band d, t 2..d (weighted by n_value_tokens) >= 0.80, and z-shuffle pooled <= floor pooled + 0.05.
  COPY-CONTROL (M1 cells): dprime pooled (band, t >= 2) <= floor + 0.05 AND dpost pooled >= 0.80 (else UNREADABLE = not met).
  PORT-FOLLOW (pairs in band): follow = mean(prefix == donor) >= floor_rate + 0.30, floor_rate = mean(floor == donor); n_kept >= 1,000; LEAK-SEG
    (leak == donor) <= 0.25; MARKER-SWAP (marker == donor) <= floor_rate + 0.05; self_bitwise.
  BUDGET-FRONT (v1.1): band observations, t and b* centred within d; tie-aware Spearman <= -0.6 and OLS slope <= -0.4; n_obs >= 1,000 from >= 300 pairs.
  ANSWER-ONSET (M1 cells): answer_onset acc <= floor_d + 0.05 at every band d and t <= d - 2.
  PORT-STATE = DECODE-VAL & COPY-CONTROL & PORT-FOLLOW & BUDGET-FRONT & ANSWER-ONSET.
Labels (precedence): UNREAD > M1-NOT-MET > PORT-STATE > PORT-RELAY (follow, front fails) > PORT-PRECOMPUTE (follow, onset fails) > CARRIER
  (cf == pred rate >= cf floor + 0.30) > CONTEXT-BOUND (follow < floor + 0.10) > MIXED. M2-only has no decode head: decode / copy / onset / PORT-STATE
  are N/A and its label is drawn from PORT-RELAY / FOLLOW-FRONT / CARRIER / CONTEXT-BOUND / MIXED (descriptive attribution).
Addendum B: entry step (first 1k-bin mean num_ce (number-token CE) < 0.5); |entry(PS) - entry(ablation)| > 5,000 -> that attribution CONFOUNDED. 'Needs both' is descriptive.
Censoring: log short of 24,000 steps (dead), a missing file -> pending; G-W0 failure -> UNREAD.
    python theory/score_s223.py [--fixtures]
"""
import argparse, json, math, os, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = os.path.join(ROOT, "boxes", "runs_fsq")
OUT = os.path.join(ROOT, "theory", "results", "score_s223.json")
CELLS = {"direct": "s223_direct_s1", "ps": "s223_ps_s1", "m1": "s223_m1_s1", "m2": "s223_m2_s1"}
M1_CELLS = ("ps", "m1")
STEPS = 24000
EPS = 1e-12


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def log_info(path):
    try:
        rows = [json.loads(l) for l in open(path) if l.strip()]
    except OSError:
        return None
    last = max((r.get("step", 0) for r in rows), default=0)
    bins = {}
    for r in rows:
        if r.get("num_ce") is not None:                                   # the NUMBER-token CE (what CurGate reads; Experiments' trainer field)
            bins.setdefault(int(r["step"]) // 1000, []).append(float(r["num_ce"]))
    entry = next((1000 * (b + 1) for b in sorted(bins) if np.mean(bins[b]) < 0.5), None)
    last_bin = float(np.mean(bins[max(bins)])) if bins else None
    return {"last_step": last, "entry_step": entry, "last_bin_num_ce": last_bin}


def band_from_direct(ev):
    a = {int(d): v for d, v in ev["acc"]["acc_by_d"].items()}
    nstar = max((d for d in range(1, 9) if a.get(d, 0) > 0.50 + EPS), default=0)
    band = [d for d in range(max(3, nstar + 1), 9) if a.get(d) is not None and a[d] <= 0.50 + EPS]
    return nstar, band


def gw0(ev, info):
    """G-W0 (ii) with Note 223-6's guards: DIRECT NO-ENTRY (entry None or last num_ce 1k bin >= 0.5) or DIRECT acc(d = 1) < 0.80 -> UNREADABLE
    (band []); else the band rule, which needs >= 3 depths."""
    nstar, band = band_from_direct(ev); a1 = ev["acc"]["acc_by_d"].get("1")
    why = []
    if info is None or info.get("entry_step") is None or info.get("last_bin_num_ce") is None or info["last_bin_num_ce"] >= 0.5:
        why.append("DIRECT NO-ENTRY")
    if a1 is None or a1 < 0.80:
        why.append("DIRECT acc(d=1) %s < 0.80" % a1)
    if why:
        return {"nstar": nstar, "band": [], "band_rule": band, "readable": False, "ok": False, "reason": "UNREADABLE: " + "; ".join(why)}
    return {"nstar": nstar, "band": band, "readable": True, "ok": len(band) >= 3, "reason": "band %s" % band}


def pooled(tab, band, tmin=2, w=None):
    num = den = 0.0
    for d in band:
        for t in range(tmin, d + 1):
            v = tab.get(str(d), {}).get(str(t))
            if v is None:
                continue
            ww = (w or {}).get(str(d), {}).get(str(t), 1) if w else 1
            num += v * ww; den += ww
    return num / den if den else None


def avgrank(a):
    a = np.asarray(a, dtype=float); o = np.argsort(a, kind="mergesort"); r = np.empty(len(a)); r[o] = np.arange(len(a), dtype=float)
    for v in np.unique(a):
        m = a == v; r[m] = r[m].mean()
    return r


def spearman(x, y):
    rx, ry = avgrank(x), avgrank(y)
    return float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 0 and np.std(ry) > 0 else 0.0


def front(bf, band):
    obs = [o for o in bf["items"] if int(o[0]) in band]
    pairs = {(int(o[0]), int(o[1])) for o in obs}
    if len(obs) < 1000 or len(pairs) < 300:
        return {"ok": None, "why": "UNREADABLE (n_obs %d, pairs %d < 1000 / 300)" % (len(obs), len(pairs)), "n_obs": len(obs), "n_pairs": len(pairs)}
    T, B = [], []
    for d in band:
        ob = [o for o in obs if int(o[0]) == d]
        if not ob:
            continue
        t = np.array([o[2] for o in ob], float); b = np.array([o[3] for o in ob], float)
        T.append(t - t.mean()); B.append(b - b.mean())
    T, B = np.concatenate(T), np.concatenate(B)
    rho = spearman(T, B); slope = float(np.polyfit(T, B, 1)[0]) if np.std(T) > 0 else 0.0
    flat_b = float(np.mean([o[3] for o in obs]))
    return {"ok": bool(rho <= -0.6 + EPS and slope <= -0.4 + EPS), "rho": rho, "slope": slope, "n_obs": len(obs), "n_pairs": len(pairs), "mean_bstar": flat_b}


def swaps(sw, band):
    it = [x for d in band for x in sw["items"].get(str(d), [])]
    n = len(it)
    if n == 0:
        return {"n": 0}
    rate = lambda f: sum(1 for x in it if f(x)) / n   # noqa: E731
    return {"n": n, "follow": rate(lambda x: x["prefix"] == x["donor"]), "floor": rate(lambda x: x["floor"] == x["donor"]),
            "cf": rate(lambda x: x["prefix"] == x["cf"]), "cf_floor": rate(lambda x: x["floor"] == x["cf"]), "own": rate(lambda x: x["prefix"] == x["own"]),
            "leak": rate(lambda x: x["leak"] == x["donor"]), "marker": rate(lambda x: x["marker"] == x["donor"]),
            "state_full": rate(lambda x: x["state_full"] == x["donor"]), "cache_full": rate(lambda x: x["cache_full"] == x["donor"])}


def score_cell(k, ev, info, band):
    o = {"cell": CELLS[k]}
    if ev is None or info is None:
        o["status"] = "pending (eval_ps.json / log.jsonl missing)"; return o
    if info["last_step"] < STEPS:
        o.update(status="FAIL", label="UNREAD (dead: log at %d < %d; rule b)" % (info["last_step"], STEPS)); return o
    if len(band) < 3:
        o.update(status="FAIL", label="UNREAD (G-W0 band < 3 depths)"); return o
    o["entry_step"] = info["entry_step"]
    sw = swaps(ev["swaps"], band); o["SWAPS"] = sw
    follow_bar = sw["n"] >= 1000 and sw["follow"] >= sw["floor"] + 0.30 - EPS
    port_follow = bool(follow_bar and sw["leak"] <= 0.25 + EPS and sw["marker"] <= sw["floor"] + 0.05 + EPS and ev["swaps"].get("self_bitwise"))
    o["PORT_FOLLOW"] = port_follow if sw["n"] >= 1000 else None
    fr = front(ev["budget_front"], band); o["BUDGET_FRONT"] = fr
    m1 = k in M1_CELLS
    if m1:
        dc, cc, ao = ev["decode"], ev["copy_control"], ev["answer_onset"]
        w = dc.get("n_value_tokens")
        val, zs, fl = pooled(dc["value_acc"], band, w=w), pooled(dc["zshuffle_value_acc"], band, w=w), pooled(dc["value_floor"], band, w=w)
        decode_val = bool(val is not None and val >= 0.80 - EPS and zs is not None and zs <= fl + 0.05 + EPS)
        dp, dq, cfl = pooled(cc["dprime_value_acc"], band), pooled(cc["dpost_value_acc"], band), pooled(cc["value_floor"], band)
        copy_ok = None if (dq is None or dq < 0.80 - EPS) else bool(dp <= cfl + 0.05 + EPS)
        onset = all(ao["acc"][str(d)][str(t)] <= ao["floor"][str(d)] + 0.05 + EPS for d in band for t in range(1, d - 1))
        o.update(DECODE_VAL={"met": decode_val, "value": val, "zshuffle": zs, "floor": fl},
                 COPY_CONTROL={"met": copy_ok, "dprime": dp, "dpost": dq, "floor": cfl,
                               "why": "UNREADABLE (D_post < 0.80: under-powered)" if copy_ok is None else None},
                 ANSWER_ONSET=onset)
        state = bool(decode_val and copy_ok and port_follow and fr["ok"] and onset)
    carrier = sw["n"] > 0 and sw["cf"] >= sw["cf_floor"] + 0.30 - EPS
    if m1 and not o["DECODE_VAL"]["met"]:
        lab = "M1-NOT-MET"
    elif m1 and state:
        lab = "PORT-STATE"
    elif port_follow and fr["ok"] is False:
        lab = "PORT-RELAY"
    elif m1 and port_follow and not o["ANSWER_ONSET"]:
        lab = "PORT-PRECOMPUTE"
    elif not m1 and port_follow and fr["ok"]:
        lab = "FOLLOW-FRONT (no decode head; descriptive)"
    elif carrier:
        lab = "CARRIER"
    elif sw["n"] > 0 and sw["follow"] < sw["floor"] + 0.10:
        lab = "CONTEXT-BOUND"
    else:
        lab = "MIXED"
    if sw["n"] < 1000 and lab not in ("M1-NOT-MET",):
        lab = "UNREAD (< 1,000 exposed pairs)" if lab in ("PORT-STATE", "PORT-RELAY", "PORT-PRECOMPUTE") else lab
    o.update(status="SCORED", label=lab, wording="one seed, templated word task; never GSM8K / C4")
    return o


def score(runs=R):
    out = {"section": 223, "registered": "PS2_v1.md at 54f63381"}
    evd = jl(os.path.join(runs, CELLS["direct"], "eval_ps.json"))
    if evd is None:
        out["band"] = "pending (DIRECT eval_ps.json)"; return out
    g = gw0(evd, log_info(os.path.join(runs, CELLS["direct"], "log.jsonl"))); out["G_W0"] = g    # Note 223-6 guards
    nstar, band = g["nstar"], g["band"]; out["nstar"], out["band"] = nstar, band
    out["G_W0_band_ok"] = g["ok"]
    cells = {}
    for k in ("ps", "m1", "m2"):
        cells[k] = score_cell(k, jl(os.path.join(runs, CELLS[k], "eval_ps.json")), log_info(os.path.join(runs, CELLS[k], "log.jsonl")), band)
    out["cells"] = cells
    attr = {}
    ps = cells["ps"]
    for k in ("m1", "m2"):
        e1, e2 = ps.get("entry_step"), cells[k].get("entry_step")
        conf = (e1 is None) != (e2 is None) or (e1 is not None and e2 is not None and abs(e1 - e2) > 5000)
        attr[k] = {"entry_ps": e1, "entry_ablation": e2, "CONFOUNDED": bool(conf)}
    out["attribution"] = attr
    out["needs_both"] = (ps.get("label") == "PORT-STATE" and cells["m1"].get("label") != "PORT-STATE" and not cells["m2"].get("label", "").startswith("FOLLOW-FRONT")) \
        if ps.get("status") == "SCORED" else None
    out["needs_both_note"] = "descriptive only (one seed; Addendum B)"
    return out


# ------------------------------------------------------------------ fixtures
def _ev(m1=True, follow=0.5, floor=0.05, cf=0.1, leak=0.05, marker=0.05, n_per_d=300, front_kind="serial", val=0.9, zs=0.1, vfl=0.1,
        dprime=0.1, dpost=0.9, onset_hi=False, direct_acc=None, seed=0):
    rng = np.random.default_rng(seed); ev = {"meta": {}}
    ev["acc"] = {"acc_by_d": {str(d): (direct_acc or {}).get(d, 0.3) for d in range(1, 13)}}
    items = {}
    for d in range(3, 9):
        lst = []
        for i in range(n_per_d):
            own, donor, cfv = 1, 2, 3
            u = rng.random()
            pref = donor if u < follow else (cfv if u < follow + cf else own)
            lst.append({"pair": i, "own": own, "donor": donor, "cf": cfv, "cf_b": 4, "plain": own, "prefix": pref,
                        "floor": donor if rng.random() < floor else 9, "floor_donor_pair": i, "leak": donor if rng.random() < leak else own,
                        "marker": donor if rng.random() < marker else own, "self": own, "state_full": donor, "cache_full": donor})
        items[str(d)] = lst
    ev["swaps"] = {"items": items, "self_bitwise": True}
    bf = []
    for d in range(3, 9):
        for p in range(0, n_per_d, 3):
            for t in range(1, d):
                extra = int(rng.geometric(0.7) - 1)
                b = (d - t + extra) if front_kind == "serial" else (d + extra)
                bf.append([d, p, t, b, True])
    ev["budget_front"] = {"items": bf}
    if m1:
        tab = lambda v: {str(d): {str(t): v for t in range(1, d + 1)} for d in range(1, 13)}   # noqa: E731
        ev["decode"] = {"value_acc": tab(val), "zshuffle_value_acc": tab(zs), "value_floor": tab(vfl), "n_value_tokens": tab(10)}
        ev["copy_control"] = {"dprime_value_acc": tab(dprime), "dpost_value_acc": tab(dpost), "value_floor": tab(vfl)}
        ev["answer_onset"] = {"acc": tab(0.5 if onset_hi else 0.1), "floor": {str(d): 0.1 for d in range(1, 13)}}
    return ev


def fixtures():
    import tempfile
    res = []
    band = [4, 5, 6, 7, 8]; info = {"last_step": 24000, "entry_step": 5000}
    def lab(**kw):
        k = kw.pop("cell", "ps"); return score_cell(k, _ev(m1=k in M1_CELLS, **kw), info, band)["label"]
    res.append(("PORT-STATE", lab() == "PORT-STATE", lab()))
    res.append(("relay (flat front) -> PORT-RELAY", lab(front_kind="relay") == "PORT-RELAY", lab(front_kind="relay")))
    res.append(("precompute (onset fails) -> PORT-PRECOMPUTE", lab(onset_hi=True) == "PORT-PRECOMPUTE", lab(onset_hi=True)))
    res.append(("decode fails -> M1-NOT-MET", lab(val=0.5) == "M1-NOT-MET", lab(val=0.5)))
    res.append(("zshuffle high -> M1-NOT-MET", lab(zs=0.5) == "M1-NOT-MET", lab(zs=0.5)))
    o = score_cell("ps", _ev(dpost=0.6), info, band); res.append(("copy under-powered -> not PORT-STATE", o["label"] != "PORT-STATE" and o["COPY_CONTROL"]["met"] is None, o["label"]))
    o = score_cell("ps", _ev(dprime=0.5), info, band); res.append(("copy control fails -> not PORT-STATE", o["label"] != "PORT-STATE", o["label"]))
    res.append(("carrier", lab(follow=0.05, cf=0.6) == "CARRIER", lab(follow=0.05, cf=0.6)))
    res.append(("context-bound", lab(follow=0.06, cf=0.05) == "CONTEXT-BOUND", lab(follow=0.06, cf=0.05)))
    res.append(("leak > 0.25 blocks PORT-STATE", lab(leak=0.4) != "PORT-STATE", lab(leak=0.4)))
    res.append(("marker leak blocks PORT-STATE", lab(marker=0.3) != "PORT-STATE", lab(marker=0.3)))
    res.append(("M2-only follow+front -> FOLLOW-FRONT", lab(cell="m2").startswith("FOLLOW-FRONT"), lab(cell="m2")))
    o = score_cell("ps", _ev(n_per_d=150), info, band); res.append(("< 1000 pairs -> no PORT-STATE", o["label"] != "PORT-STATE", o["label"]))
    o = score_cell("ps", _ev(), {"last_step": 20000, "entry_step": 5000}, band); res.append(("dead -> UNREAD", o["label"].startswith("UNREAD (dead"), o["label"]))
    o = score_cell("ps", _ev(), info, [7, 8]); res.append(("band < 3 -> UNREAD", o["label"].startswith("UNREAD (G-W0"), o["label"]))
    n, b = band_from_direct({"acc": {"acc_by_d": {"1": .9, "2": .8, "3": .6, "4": .45, "5": .3, "6": .2, "7": .15, "8": .1}}})
    res.append(("band from DIRECT", n == 3 and b == [4, 5, 6, 7, 8], (n, b)))
    flat = {"acc": {"acc_by_d": {str(k): .1 for k in range(1, 9)}}}; ok_info = {"entry_step": 3000, "last_bin_num_ce": 0.2}
    g = gw0(flat, ok_info); res.append(("223-6: chance DIRECT (n*=0) -> UNREADABLE, not band 3-8", not g["readable"] and g["band"] == [] and g["band_rule"] == [3, 4, 5, 6, 7, 8], g["reason"]))
    good = {"acc": {"acc_by_d": {"1": .9, "2": .8, "3": .6, "4": .45, "5": .3, "6": .2, "7": .15, "8": .1}}}
    g = gw0(good, {"entry_step": None, "last_bin_num_ce": 0.6}); res.append(("223-6: DIRECT NO-ENTRY -> UNREADABLE", not g["readable"] and "NO-ENTRY" in g["reason"], g["reason"]))
    g = gw0(good, {"entry_step": 3000, "last_bin_num_ce": 0.55}); res.append(("223-6: last bin >= 0.5 -> UNREADABLE", not g["readable"], g["reason"]))
    low1 = {"acc": {"acc_by_d": {"1": .7, "2": .6, "3": .45, "4": .4, "5": .3, "6": .2, "7": .15, "8": .1}}}
    g = gw0(low1, ok_info); res.append(("223-6: acc(d=1) 0.70 -> UNREADABLE", not g["readable"] and "d=1" in g["reason"], g["reason"]))
    g = gw0(good, ok_info); res.append(("223-6: good DIRECT -> band 4-8 ok", g["readable"] and g["ok"] and g["band"] == [4, 5, 6, 7, 8], g["reason"]))
    # end-to-end with confound
    d = tempfile.mkdtemp()
    def put(cell, ev, entry):
        os.makedirs(os.path.join(d, CELLS[cell]), exist_ok=True); json.dump(ev, open(os.path.join(d, CELLS[cell], "eval_ps.json"), "w"))
        with open(os.path.join(d, CELLS[cell], "log.jsonl"), "w") as fh:
            for s in range(100, 24001, 100):
                fh.write(json.dumps({"step": s, "ans_ce": 9.9, "num_ce": 0.2 if s > entry else 1.0}) + "\n")
    put("direct", _ev(m1=False, direct_acc={1: .9, 2: .8, 3: .6, 4: .45, 5: .3, 6: .2, 7: .15, 8: .1}), 3000)
    put("ps", _ev(), 4000); put("m1", _ev(follow=0.05, cf=0.6), 4500); put("m2", _ev(m1=False, follow=0.05, cf=0.05), 12000)
    out = score(d)
    res.append(("e2e: PS PORT-STATE, needs_both, m2 CONFOUNDED", out["cells"]["ps"]["label"] == "PORT-STATE" and out["needs_both"] is True and out["attribution"]["m2"]["CONFOUNDED"] and not out["attribution"]["m1"]["CONFOUNDED"], (out["cells"]["ps"]["label"], out["needs_both"], out["attribution"])))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(json.dumps({k: o.get(k) for k in ("nstar", "band")}), {k: v.get("label", v.get("status")) for k, v in (o.get("cells") or {}).items()})
    json.dump(o, open(OUT, "w", newline=chr(10)), indent=1, default=str)
