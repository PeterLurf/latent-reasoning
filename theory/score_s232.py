"""THEORY §232 HOLD-C1 scorer (registered text docs/research_holdc1_2026-10-24/HC_v1.md at ee6828ac + Note 232-1; Theory is the only scorer).
Inputs per model m (Stage A, one RTX 3090, fp32, bs 32): boxes/runs_fsq/<m>_c1/{eval_vb.json, eval_d3hist.json, eval_c1hold.json}; HOLD from the HC
model's own eval_khorizon.json (boxes/runs_fsq/<m>/). Roles: s1 BASE s213_d3t_s1, CTRL s215_ftctrl_s1, HC s230_fhftfix_s1; s2 BASE s216_d3t_s2,
CTRL s224_ftctrl_s2, HC s224_fhft_s2 (s3 cells added only if run: s232_d3t_s3 / s232_ftctrl_s3 / s232_fhftfix_s3).
VB-C1 per model = score_s211's definitions without its 48k-dead / entry rules (fine-tunes are 12k): DEEP (acc_n2 n 3..8 >= 0.90), PROBE-GAIN (valid and
  acc - max(PROBE(0), PROBE(1)) >= 0.50), SERIAL-ONSET (score_s211.onset_reading), SERIAL-STEP (vswap n_pairs >= 100, follow_ii >= 0.6, own_ii <= 0.3),
  NO-CLOCK (stall >= 0.9).
C1-KEPT (HC vs BASE and vs CTRL; HC_v1 sec. 3 margins): PROBE-GAIN >= ref - 0.03; |PROBE(2) - ref| <= 0.03; |PROBE(t) - ref| <= 0.05 for t in {3, 4, 6};
  follow_ii >= ref - 0.02; own_ii <= ref + 0.02; stall >= ref - 0.02; DEEP min acc_n2 n 3..8 >= ref - 0.02; |mean-onset slope - ref| <= 0.10 (slope of the
  per-height MEAN onset, heights with >= 30 items); |mean onset - ref| <= 0.5 passes. A reference that is unreadable (missing / c1hold gate fail) -> C1-KEPT
  reads on the other with a flag; both unreadable -> UNREAD.
HOLD: H(K') = mean n 1..8 of eval_khorizon acc.intact[K'][n] >= 0.95 at every K' in 24..128. DEPTH: mean acc_n2 n 9..11 >= CTRL - 0.03. NOLOSS: acc_n2
  n 1..8 >= 0.99, stall >= CTRL - 0.02, acc_k['0'][n] n 1..16 and acc_k['1'][n] n 2..16 <= CTRL + 0.05.
SHIFT (COEXIST-SHIFTED trigger): carry_likeness drop vs CTRL >= 0.10 OR OLD-SWAP old_follow >= 0.10 (HC); counts as COEXIST if follow_ii and the
  c1hold v0off_K follow stay within 0.02 of CTRL.
Labels per seed: COEXIST / COEXIST-SHIFTED / C1-HARMED (a decisive statistic outside margin vs CTRL) / HOLD-FAIL / NOT-VB-C1 / UNREAD.
  FAR-CHANGED (descriptive) = SHIFT with C1 kept. Claim: COEXIST(-SHIFTED) in s1 and s2 (2 / 3 with s3). SPLIT -> s3 trigger (as also both passing).
    python -B theory/score_s232.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
import numpy as np

_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory"))
import score_s211 as S211  # noqa: E402

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_s232.json")
SEEDS = {"s1": {"base": "s213_d3t_s1", "ctrl": "s215_ftctrl_s1", "hc": "s230_fhftfix_s1"},
         "s2": {"base": "s216_d3t_s2", "ctrl": "s224_ftctrl_s2", "hc": "s224_fhft_s2"},
         "s3": {"base": "s232_d3t_s3", "ctrl": "s232_ftctrl_s3", "hc": "s232_fhftfix_s3"}}
KS = [24, 32, 48, 64, 96, 128]
NS = range(3, 9)
INF = float("inf")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def mean_onset(ev):
    pr, meta = ev["probe"], ev["test_meta"]; ns, hs = meta["n"], meta["height"]; byh, allo = {}, []
    for i, n in enumerate(ns):
        if not 3 <= n <= 8:
            continue
        o = INF
        for t in S211.T_SET:
            ts = [u for u in S211.T_SET if t <= u <= n + 4]
            if ts and all(pr[str(u)]["items"][i] for u in ts):
                o = t; break
        if o < INF:
            byh.setdefault(hs[i], []).append(o); allo.append(o)
    hk = sorted(h for h in byh if len(byh[h]) >= 30)
    sl = S211.ols_slope(hk, [float(np.mean(byh[h])) for h in hk]) if len(hk) >= 2 else None
    return sl, (float(np.mean(allo)) if allo else None)


def model(runs, m):
    d = os.path.join(runs, m + "_c1"); ev, dh, ch = jl(os.path.join(d, "eval_vb.json")), jl(os.path.join(d, "eval_d3hist.json")), jl(os.path.join(d, "eval_c1hold.json"))
    o = {"name": m, "missing": [f for f, v in (("eval_vb.json", ev), ("eval_d3hist.json", dh), ("eval_c1hold.json", ch)) if v is None], "cards": []}
    if o["missing"]:
        return o
    o["cards"] = [ev.get("card"), dh.get("card"), ch.get("card")]
    c = ch["c1hold"]; o["gate_ok"] = bool(c.get("gate", {}).get("pass"))
    a2 = ev["acc_n2"]; acc = S211.pooled(a2); pr = ev["probe"]; p = {t: S211.pooled(pr[str(t)]["acc_by_n"]) for t in S211.T_SET}
    valid = p[12] is not None and p[12] >= acc - 0.05
    vs = ev["vswap"]; st = ev["stall"]["recovery"]; on = S211.onset_reading(ev) if valid else {"SERIAL_ONSET": False}
    sl, mo = mean_onset(ev)
    o.update({"DEEP": all(a2[str(n)] >= 0.90 for n in NS), "deep_min": min(a2[str(n)] for n in NS), "PROBE": p, "PROBE_VALID": valid,
              "pgain": acc - max(p[0], p[1]), "SERIAL_ONSET": on["SERIAL_ONSET"], "n_pairs": vs.get("n_pairs", 0), "follow_ii": vs.get("follow_ii"),
              "own_ii": vs.get("own_ii"), "stall": st, "onset_slope_mean": sl, "onset_mean": mo, "carry": ev.get("carry_likeness"),
              "old_follow": dh.get("swap", {}).get("old_follow"), "v0off_follow": c["arms"]["v0off_K"]["follow"] if "arms" in c else c.get("v0off_K", {}).get("follow"),
              "acc18": min(a2[str(n)] for n in range(1, 9)), "depth": float(np.mean([a2[str(n)] for n in (9, 10, 11)])),
              "k0": {n: ev["acc_k"]["0"][str(n)] for n in range(1, 17)}, "k1": {n: ev["acc_k"]["1"][str(n)] for n in range(2, 17)}})
    o["VB_C1"] = bool(o["DEEP"] and valid and o["pgain"] >= 0.50 and o["SERIAL_ONSET"] and o["n_pairs"] >= 100 and o["follow_ii"] >= 0.6 and o["own_ii"] <= 0.3 and st >= 0.9)
    return o


def kept(h, r):
    """decisive C1 statistics of HC within margin of reference r -> (ok, failures)."""
    f = []
    if h["pgain"] < r["pgain"] - 0.03:
        f.append("PROBE-GAIN")
    if abs(h["PROBE"][2] - r["PROBE"][2]) > 0.03:
        f.append("PROBE(2)")
    for t in (3, 4, 6):
        if abs(h["PROBE"][t] - r["PROBE"][t]) > 0.05:
            f.append("PROBE(%d)" % t)
    if h["follow_ii"] < r["follow_ii"] - 0.02:
        f.append("follow_ii")
    if h["own_ii"] > r["own_ii"] + 0.02:
        f.append("own_ii")
    if h["stall"] < r["stall"] - 0.02:
        f.append("STALL")
    if h["deep_min"] < r["deep_min"] - 0.02:
        f.append("DEEP")
    if h["onset_slope_mean"] is None or r["onset_slope_mean"] is None or abs(h["onset_slope_mean"] - r["onset_slope_mean"]) > 0.10:
        f.append("onset_slope")
    if h["onset_mean"] is None or r["onset_mean"] is None or abs(h["onset_mean"] - r["onset_mean"]) > 0.5:
        f.append("onset_mean")
    return not f, f


def hold(runs, m):
    k = jl(os.path.join(runs, m, "eval_khorizon.json"))
    if k is None or not all(str(x) in k["acc"]["intact"] for x in KS):
        return None
    return {x: float(np.mean([k["acc"]["intact"][str(x)][str(n)] for n in range(1, 9)])) for x in KS}


def score_seed(runs, s):
    C = SEEDS[s]; M = {k: model(runs, v) for k, v in C.items()}
    out = {"models": {k: {kk: vv for kk, vv in v.items() if kk not in ("k0", "k1")} for k, v in M.items()}}
    h, b, c = M["hc"], M["base"], M["ctrl"]
    if h["missing"] or not h.get("gate_ok"):
        out["label"] = "UNREAD (HC: %s)" % (h["missing"] or "c1hold gate fail"); return out, M
    refs = {k: M[k] for k in ("base", "ctrl") if not M[k]["missing"] and M[k].get("gate_ok")}
    if not refs:
        out["label"] = "UNREAD (no readable reference)"; return out, M
    H = hold(runs, C["hc"]); out["H"] = H
    HOLD = H is not None and all(v >= 0.95 for v in H.values())
    kres = {k: kept(h, r) for k, r in refs.items()}; out["C1_KEPT_by_ref"] = {k: v[1] for k, v in kres.items()}
    C1K = all(v[0] for v in kres.values()); flag = [] if len(refs) == 2 else ["C1-KEPT on %s only" % list(refs)]
    if "ctrl" in refs:
        DEPTH = h["depth"] >= c["depth"] - 0.03
        c1f = all(h["k0"][n] <= c["k0"][n] + 0.05 for n in h["k0"]) and all(h["k1"][n] <= c["k1"][n] + 0.05 for n in h["k1"])
        NOLOSS = h["acc18"] >= 0.99 and h["stall"] >= c["stall"] - 0.02 and c1f
        shift = (h["carry"] is not None and c["carry"] is not None and c["carry"] - h["carry"] >= 0.10) or (h["old_follow"] is not None and h["old_follow"] >= 0.10)
        shift_ok = h["follow_ii"] >= c["follow_ii"] - 0.02 and h["v0off_follow"] is not None and c["v0off_follow"] is not None and h["v0off_follow"] >= c["v0off_follow"] - 0.02
        harmed = not kres["ctrl"][0]
    else:
        DEPTH = NOLOSS = None; shift = shift_ok = False; harmed = not C1K; flag.append("no CTRL: DEPTH / NOLOSS / SHIFT unreadable")
    out.update({"HOLD": HOLD, "VB_C1_hc": h["VB_C1"], "C1_KEPT": C1K, "DEPTH": DEPTH, "NOLOSS": NOLOSS, "SHIFT": shift, "flags": flag,
                "depth_hc_ctrl": (h["depth"], c["depth"]) if "ctrl" in refs else None})
    if not HOLD:
        lab = "HOLD-FAIL"
    elif harmed:
        lab = "C1-HARMED (%s)" % ",".join(kres["ctrl"][1] if "ctrl" in refs else sum((v[1] for v in kres.values()), []))
    elif not h["VB_C1"]:
        lab = "NOT-VB-C1"
    elif C1K and DEPTH and NOLOSS:
        lab = ("COEXIST-SHIFTED" if shift_ok else "C1-HARMED (shift without follow / v0off kept)") if shift else "COEXIST"
    else:
        lab = "NOT COEXIST (%s)" % ",".join(k for k, v in (("C1-KEPT", C1K), ("DEPTH", DEPTH), ("NOLOSS", NOLOSS)) if not v)
    out["label"] = lab; out["FAR_CHANGED"] = bool(shift and C1K)
    return out, M


def score(runs=R):
    res = {"section": 232, "registered": "HC_v1.md @ ee6828ac + 232-1", "seeds": {}}; cards = []
    seeds = ["s1", "s2"] + (["s3"] if os.path.isdir(os.path.join(runs, SEEDS["s3"]["hc"] + "_c1")) else [])
    for s in seeds:
        o, M = score_seed(runs, s); res["seeds"][s] = o; cards += [x for v in M.values() for x in v.get("cards", []) if x]
    cls = sorted(set(str(c) for c in cards)); res["card_classes"] = cls
    if cls and (len(cls) > 1 or "3090" not in cls[0]):
        res["verdict"] = "VOID (card %s)" % cls; return res
    labs = [res["seeds"][s]["label"] for s in seeds]
    if any(l.startswith("UNREAD") for l in labs[:2]) and len(seeds) == 2:
        res["verdict"] = "UNREAD (%s)" % labs; return res
    ok = [l.startswith("COEXIST") for l in labs]
    if len(seeds) == 2:
        res["verdict"] = ("COEXIST (2/2; s3 trigger)" if all(ok) else ("SPLIT (s3 trigger)" if any(ok) else "NOT COEXIST (both seeds)")) + " | " + " / ".join(labs)
    else:
        res["verdict"] = ("COEXIST (%d/3)" % sum(ok) if sum(ok) >= 2 else "NOT COEXIST (%d/3)" % sum(ok)) + " | " + " / ".join(labs)
    return res


# ------------------------------------------------------------------ fixtures
def _mk(runs, m, *, pgain=0.6, p2=0.2, p346=0.5, fol=0.95, own=0.02, stall=1.0, deep=0.97, slope=1.0, carry=0.83, old=0.01, v0=0.9, hold128=0.99,
        acc18=1.0, depth=0.6, card="NVIDIA GeForce RTX 3090", gate=True, kho=True):
    d = os.path.join(runs, m + "_c1"); os.makedirs(d, exist_ok=True); N = 1200
    rng = np.random.default_rng(abs(hash(m)) % 2 ** 31)
    ns = [3 + (i % 6) for i in range(N)]; hs = [n - 2 for n in ns]                 # height grows with n (1..6)
    ons = [min([t for t in S211.T_SET if t >= slope * h] or [12]) for h in hs]
    a2 = {str(n): (acc18 if n <= 8 else depth) for n in range(1, 17)}
    for n in NS:
        a2[str(n)] = max(deep, a2[str(n)]) if acc18 >= deep else deep
    accp = float(np.mean([a2[str(n)] for n in NS]))
    pr = {}
    for t in S211.T_SET:
        v = {0: accp - pgain - 0.05, 1: accp - pgain, 2: p2, 3: p346, 4: p346, 6: p346, 8: accp, 12: accp}[t]
        items = [1 if t >= ons[i] else 0 for i in range(N)]
        pr[str(t)] = {"acc_by_n": {str(n): v for n in range(1, 17)}, "items": items}
    ev = {"card": card, "acc_n2": a2, "acc_k": {"0": {str(n): 0.1 for n in range(1, 17)}, "1": {str(n): 0.2 for n in range(1, 17)}},
          "probe": pr, "test_meta": {"n": ns, "height": hs}, "vswap": {"n_pairs": 500, "follow_ii": fol, "own_ii": own}, "stall": {"recovery": stall},
          "carry_likeness": carry}
    json.dump(ev, open(os.path.join(d, "eval_vb.json"), "w"))
    json.dump({"card": card, "swap": {"old_follow": old}}, open(os.path.join(d, "eval_d3hist.json"), "w"))
    json.dump({"card": card, "c1hold": {"gate": {"pass": gate}, "arms": {"v0off_K": {"follow": v0}}}}, open(os.path.join(d, "eval_c1hold.json"), "w"))
    if kho:
        os.makedirs(os.path.join(runs, m), exist_ok=True)
        json.dump({"acc": {"intact": {str(k): {str(n): (1.0 if k < 128 else hold128) for n in range(1, 9)} for k in KS}}}, open(os.path.join(runs, m, "eval_khorizon.json"), "w"))


def _world(over=None):
    over = over or {}; d = tempfile.mkdtemp()
    for s in ("s1", "s2"):
        for role, m in SEEDS[s].items():
            _mk(d, m, **over.get((s, role), {}))
    return d


def fixtures():
    res = []
    o = score(_world()); res.append(("identical stats, HC holds -> COEXIST 2/2", o["verdict"].startswith("COEXIST (2/2"), o["verdict"]))
    o = score(_world({("s1", "hc"): {"fol": 0.85}})); res.append(("s1 HC follow_ii -0.10 -> C1-HARMED, SPLIT", "C1-HARMED" in o["verdict"] and o["verdict"].startswith("SPLIT"), o["verdict"]))
    o = score(_world({("s2", "hc"): {"hold128": 0.9}})); res.append(("s2 HC hold 0.90 -> HOLD-FAIL, SPLIT", "HOLD-FAIL" in o["verdict"], o["verdict"]))
    o = score(_world({("s1", "hc"): {"carry": 0.6}, ("s2", "hc"): {"carry": 0.69}})); res.append(("carry drop >= 0.10, follow / v0off kept -> COEXIST-SHIFTED", o["verdict"].count("COEXIST-SHIFTED") == 2, o["verdict"]))
    o = score(_world({("s1", "hc"): {"carry": 0.6, "v0": 0.8}})); res.append(("shift with v0off dropped -> C1-HARMED", "C1-HARMED (shift" in o["verdict"], o["verdict"]))
    o = score(_world({("s1", "hc"): {"p346": 0.6}})); res.append(("PROBE(3-6) +0.10 -> C1-HARMED", "C1-HARMED" in o["verdict"], o["verdict"]))
    o = score(_world({("s1", "hc"): {"depth": 0.55}})); res.append(("DEPTH -0.05 vs CTRL -> NOT COEXIST", "NOT COEXIST (DEPTH" in o["verdict"], o["verdict"]))
    o = score(_world({("s1", "hc"): {"gate": False}})); res.append(("HC c1hold gate fail -> UNREAD", o["verdict"].startswith("UNREAD"), o["verdict"]))
    o = score(_world({("s1", "base"): {"gate": False}})); res.append(("BASE unreadable -> C1-KEPT on ctrl only, flagged", "C1-KEPT on" in str(o["seeds"]["s1"]["flags"]), o["seeds"]["s1"]["flags"]))
    o = score(_world({("s2", "ctrl"): {"card": "NVIDIA GeForce RTX 4060"}})); res.append(("mixed card -> VOID", o["verdict"].startswith("VOID"), o["verdict"]))
    o = score(_world({("s1", "hc"): {"slope": 1.5}})); res.append(("onset slope +0.5 -> C1-HARMED", "C1-HARMED" in o["verdict"], o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", newline=chr(10)), indent=1, default=str)
