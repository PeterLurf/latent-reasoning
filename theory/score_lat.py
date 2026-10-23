"""THEORY §228 LAT-OW scorer (registered text docs/research_ordered_words_2026-10-23/LAT_v1.md at e78d7211; field paths per
LAT_build_s228.md "Eval output layout" (c4950323); Theory is the only scorer).

Stage 1 cells (boxes/runs_fsq): DIRECT s228_dir_s1, LAT s228_lat_s1 (names fixed here; Experiments uses them). Pilots: score_pilot(dir_run, lat_run, mode 2|4).
Band B (G-W0): d in 2..8 with DIRECT eval_acc by_d[d].acc <= 0.50 AND not still rising (OLS slope of DIRECT's last 4 d-specific 1k-bin accs <= +0.02 per
  bin); guards: DIRECT d = 1 acc >= 0.80 and entry_d1 not None; identical d-sequence (log 1k rows ihash equal); B needs >= 3 depths, >= 2 of them >= 4.
R1: pooled over B items, acc_LAT - acc_DIR >= 0.30 and acc_LAT >= 0.60.
R2 (binding mean-state, LAT_v1 sec. 2): pooled over B, acc_LAT - acc_MEAN >= 0.30 and acc_MEAN <= acc_DIR + 0.10, with acc_MEAN = eval_mean_s
  items[d].pooled (the TEST2-pooled mean of S_8; Theory's ruling 10-23, the per-d mean is descriptive). K'1 on d >= 3 vs DIRECT, K'0-nobias: descriptive.
R3: STATE-SWAP at t* on pairs with d >= 4 and d in B; target U = {ca, cb}, keeping only pairs with cb not in {own, dfin} (and ca not in {own, dfin});
  follow = state_swap[t*] pred in U. floor = max(redraw floor (share of valid floor_preds in their U), permutation floor (state_swap preds shuffled within
  d across pairs, rng 22980, 200 permutations, mean share in U)). Pass: N >= 2,000, follow_U >= floor + 0.30, follow_U >= 0.50, self_bitwise.state true.
C1-LANG = R1 & R2 & R3; DEEP-NOT-SERIAL = R1 & R2 & not R3; NO-GAIN = not R1; censors: dead (train_summary last_step < args steps), n_nonfinite > 0,
  missing file (pending; FAIL at --final), band < 3 depths -> UNREAD (G-W0).
    python -B theory/score_lat.py [--final] [--fixtures]
"""
import argparse, json, os, sys, tempfile
import numpy as np

_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_lat.json")
CELL = {"dir": "s228_dir_s1", "lat": "s228_lat_s1"}
FILES = {"dir": ("eval_acc.json", "log.jsonl", "train_summary.json", "args.json"),
         "lat": ("eval_acc.json", "eval_mean_s.json", "eval_swaps.json", "log.jsonl", "train_summary.json", "args.json")}


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def rows(p):
    try:
        return [json.loads(l) for l in open(p) if l.strip()]
    except OSError:
        return None


def load(runs, k):
    d = os.path.join(runs, CELL[k]); o = {"missing": []}
    for f in FILES[k]:
        v = rows(os.path.join(d, f)) if f.endswith(".jsonl") else jl(os.path.join(d, f))
        if v is None:
            o["missing"].append(f)
        o[f.split(".")[0]] = v
    return o


def acc_items(ev, d, k):
    """per-item correctness at K' = k (inject: pred is a list over K' 0..14; direct: an int)."""
    it = ev["items"][str(d)]; pr = [p[k] if isinstance(p, list) else p for p in it["pred"]]
    return [int(a == b) for a, b in zip(it["ans"], pr)]


def main_acc(ev, d):
    """K' = think_k accuracy per item: by_d acc is the registered value; per-item from pred (list -> index think_k recorded as by_d['k_main'] if present)."""
    it = ev["items"][str(d)]; ans = it["ans"]; pr = it["pred"]; kk = ev["by_d"][str(d)].get("k_main")
    if pr and isinstance(pr[0], list):
        kk = kk if kk is not None else 8
        pr = [p[kk] for p in pr]
    return [int(a == b) for a, b in zip(ans, pr)]


def rising(log, d):
    b = [r["by_d"][str(d)]["acc"] for r in (log or []) if isinstance(r.get("by_d"), dict) and str(d) in r["by_d"] and r["by_d"][str(d)].get("acc") is not None]
    b = b[-4:]
    if len(b) < 4:
        return None
    return float(np.polyfit(range(4), b, 1)[0]) > 0.02


def band(D):
    ev = D["eval_acc"]; ts = D["train_summary"]
    a1 = ev["by_d"]["1"]["acc"]; why = []
    if a1 < 0.80:
        why.append("DIRECT d=1 %.3f < 0.80" % a1)
    if ts.get("entry_d1") is None:
        why.append("DIRECT NO-ENTRY")
    B = [d for d in range(2, 9) if str(d) in ev["by_d"] and ev["by_d"][str(d)]["acc"] <= 0.50 and rising(D["log"], d) is not True]
    ok = not why and len(B) >= 3 and sum(d >= 4 for d in B) >= 2
    return B, ok, why


def r3(sw, B, rng_seed=22980):
    per, allf, allfl, perm_hits, n = {}, [], [], [], 0
    rng = np.random.default_rng(rng_seed)
    for d in [d for d in B if d >= 4]:
        prs = [p for p in sw["items"].get(str(d), []) if p["cb"] not in (p["own"], p["dfin"]) and p["ca"] not in (p["own"], p["dfin"])]
        if not prs:
            continue
        U = [{p["ca"], p["cb"]} for p in prs]; sp = [p["state_swap"][str(p["t"])] for p in prs]
        f = [int(x in u) for x, u in zip(sp, U)]
        fl = [int(fp in u) for p, u in zip(prs, U) for fp in p["floor_preds"] if fp != -1]
        perm = [float(np.mean([x in u for x, u in zip(rng.permutation(sp), U)])) for _ in range(200)]
        per[d] = {"n": len(prs), "follow_U": float(np.mean(f)), "redraw_floor": float(np.mean(fl)) if fl else None, "perm_floor": float(np.mean(perm))}
        allf += f; allfl += fl; perm_hits.append((len(prs), float(np.mean(perm)))); n += len(prs)
    if n == 0:
        return {"n": 0, "pass": False, "per_d": per}
    fu = float(np.mean(allf)); rf = float(np.mean(allfl)) if allfl else 0.0; pf = sum(k * v for k, v in perm_hits) / n; floor = max(rf, pf)
    ok = n >= 2000 and fu >= floor + 0.30 and fu >= 0.50 and bool(sw.get("self_bitwise", {}).get("state"))
    return {"n": n, "follow_U": fu, "redraw_floor": rf, "perm_floor": pf, "floor": floor, "self_bitwise_state": sw.get("self_bitwise", {}).get("state"),
            "pass": ok, "per_d": per}


def score(runs=R, final=False):
    out = {"section": 228, "registered": "LAT_v1.md @ e78d7211"}
    D, L = load(runs, "dir"), load(runs, "lat")
    miss = [(CELL["dir"], m) for m in D["missing"]] + [(CELL["lat"], m) for m in L["missing"]]
    if miss:
        out["verdict"] = ("FAIL (rule b: missing %s)" if final else "pending (missing %s)") % miss; return out
    for k, X in (("dir", D), ("lat", L)):
        ts, ar = X["train_summary"], X["args"]
        if ts.get("last_step", 0) < int(ar.get("steps", 0)):
            out["verdict"] = "FAIL (dead %s)" % CELL[k]; return out
        if ts.get("n_nonfinite", 0) > 0:
            out["verdict"] = "FAIL (non-finite %s)" % CELL[k]; return out
    ih = lambda lg: [r.get("ihash") for r in lg if "ihash" in r]  # noqa: E731
    if ih(D["log"]) != ih(L["log"]):
        out["verdict"] = "VOID (d-sequence differs: exposure not matched)"; return out
    B, ok, why = band(D); out["band"] = B; out["G_W0"] = {"ok": ok, "why": why}
    if not ok:
        out["verdict"] = "UNREAD (G-W0: %s)" % (why or "band %s" % B); return out
    ld, ll = [], []; lm = []
    for d in B:
        ld += main_acc(D["eval_acc"], d); ll += main_acc(L["eval_acc"], d)
        it = L["eval_acc"]["items"][str(d)]; lm += [int(a == b) for a, b in zip(it["ans"], L["eval_mean_s"]["items"][str(d)]["pooled"])]
    aD, aL, aM = float(np.mean(ld)), float(np.mean(ll)), float(np.mean(lm))
    R1 = aL - aD >= 0.30 and aL >= 0.60
    R2 = aL - aM >= 0.30 and aM <= aD + 0.10
    k1 = {}
    for d in [d for d in B if d >= 3]:
        k1[d] = {"lat_k1": float(np.mean(acc_items(L["eval_acc"], d, 1))), "dir": float(np.mean(main_acc(D["eval_acc"], d)))}
    R3 = r3(L["eval_swaps"], B)
    out.update({"acc_dir": aD, "acc_lat": aL, "acc_mean_state": aM, "R1": R1, "R2": R2, "R3": R3, "k1_desc": k1,
                "mean_state_per_d_desc": {d: L["eval_mean_s"]["by_d"][str(d)].get("per_d") for d in B}})
    out["verdict"] = "C1-LANG" if (R1 and R2 and R3["pass"]) else ("DEEP-NOT-SERIAL" if (R1 and R2) else ("NO-GAIN" if not R1 else "R1 only (R2 fail: state not required)"))
    return out


def score_pilot(runs, dir_run, lat_run, mode):
    """LP0-LP3 (mode 2) / LP4 (mode 4), LAT_v1 sec. 5 + draft sec. 7."""
    D = {f: (rows if f.endswith(".jsonl") else jl)(os.path.join(runs, dir_run, f)) for f in ("eval_acc.json", "log.jsonl")}
    L = {f: (rows if f.endswith(".jsonl") else jl)(os.path.join(runs, lat_run, f)) for f in ("eval_acc.json", "log.jsonl")}
    if any(v is None for v in list(D.values()) + list(L.values())):
        return {"verdict": "pending"}
    lg = L["log.jsonl"]; g = [r["gn_preclip_mean100"] for r in lg if r.get("gn_preclip_mean100") is not None][1:]
    lp0 = all(np.isfinite(x) and x <= 10 for x in g)
    k1 = [r for r in lg if isinstance(r.get("by_d"), dict)]
    lp1 = bool(k1) and k1[-1]["by_d"]["1"]["acc"] >= 0.90
    av = k1[-1].get("aux_vacc_by_pass") if k1 else None                    # inject_ow: {pass: value-token acc}, keyed "1", "2", ...
    lp2 = bool(av) and float(av.get("1", 0.0)) >= 0.80
    a = lambda X, d: float(np.mean(main_acc(X["eval_acc.json"], d)))  # noqa: E731
    o = {"LP0": lp0, "LP1": lp1, "LP2": lp2}
    if mode == 2:
        o["LP3"] = a(L, 2) >= 0.70 and a(L, 2) >= a(D, 2) + 0.30; o["acc_d2"] = {"lat": a(L, 2), "dir": a(D, 2)}
        o["verdict"] = "PASS" if all(o[k] for k in ("LP0", "LP1", "LP2", "LP3")) else "FAIL"
    else:
        o["LP4"] = a(L, 3) >= 0.60 and a(L, 4) >= 0.50 and a(L, 3) >= a(D, 3) + 0.25 and a(L, 4) >= a(D, 4) + 0.25
        o["acc"] = {d: {"lat": a(L, d), "dir": a(D, d)} for d in (2, 3, 4)}
        o["verdict"] = "PASS" if all(o[k] for k in ("LP0", "LP4")) else "FAIL"
    return o


# ------------------------------------------------------------------ fixtures
def _world(kind):
    """kind: hop (a hop-per-pass serial model), relay (state not used: precompute from the question), nogain, bandfail."""
    d0 = tempfile.mkdtemp(); rng = np.random.default_rng(0); N = 400
    dir_acc = {1: 0.99, 2: 0.25, 3: 0.12, 4: 0.10, 5: 0.10, 6: 0.09, 7: 0.09, 8: 0.09}
    if kind == "bandfail":
        dir_acc.update({2: 0.95, 3: 0.9, 4: 0.85, 5: 0.8, 6: 0.8, 7: 0.2, 8: 0.1})
    lat_acc = {d: (0.98 if kind != "nogain" else dir_acc[d] + 0.05) for d in range(1, 9)}
    mean_acc = {d: (0.11 if kind in ("hop", "nogain", "bandfail") else 0.95) for d in range(1, 9)}
    def items(acc_by_d, listy):
        it = {}
        for d in range(1, 9):
            ans = [int(x) for x in rng.integers(0, 11, N)]; ok = rng.random(N) < acc_by_d[d]
            pr = [a if o else (a + 1) % 11 for a, o in zip(ans, ok)]
            if listy:
                pr = [[p] * 15 for p in pr]
                for row, a in zip(pr, ans):
                    row[1] = a if (kind == "relay") else (a + 1) % 11                 # K'1: relay solves at one pass; hop doesn't (d >= 3)
            it[str(d)] = {"sig": list(range(N)), "ans": ans, "pred": pr, "L": [10] * N, "s": [3] * N}
        return it
    for cell, acc, listy in (("s228_dir_s1", dir_acc, False), ("s228_lat_s1", lat_acc, True)):
        p = os.path.join(d0, cell); os.makedirs(p)
        it = items(acc, listy)
        json.dump({"by_d": {str(d): {"acc": float(np.mean([a == (q[8] if listy else q) for a, q in zip(it[str(d)]["ans"], it[str(d)]["pred"])])), "k_main": 8 if listy else None}
                            for d in range(1, 9)}, "items": it}, open(os.path.join(p, "eval_acc.json"), "w"))
        json.dump({"last_step": 48000, "n_nonfinite": 0, "entry_d1": 900}, open(os.path.join(p, "train_summary.json"), "w"))
        json.dump({"steps": 48000}, open(os.path.join(p, "args.json"), "w"))
        with open(os.path.join(p, "log.jsonl"), "w") as fh:
            for b in range(1, 49):
                fh.write(json.dumps({"bin": b, "ihash": "h%d" % b, "by_d": {str(d): {"acc": acc[d], "steps": 120} for d in range(1, 9)}}) + "\n")
        if listy:
            ms = {}
            for d in range(1, 9):
                ans = it[str(d)]["ans"]; ok = rng.random(N) < mean_acc[d]
                ms[str(d)] = {"pooled": [a if o else (a + 1) % 11 for a, o in zip(ans, ok)], "per_d": []}
            json.dump({"by_d": {str(d): {"pooled": mean_acc[d], "per_d": mean_acc[d]} for d in range(1, 9)}, "items": ms}, open(os.path.join(p, "eval_mean_s.json"), "w"))
            sw = {}
            for d in range(4, 9):
                prs = []
                for i in range(500):
                    own, dfin, ca, cb = 0, 1, 2 + (i % 3), 5 + (i % 4)
                    t = d // 2 + d % 2
                    follow = (kind == "hop" and rng.random() < 0.8)
                    pred = ca if follow else own
                    prs.append({"t": t, "own": own, "dfin": dfin, "ca": ca, "cb": cb, "state_swap": {str(t): pred}, "floor_preds": [int(rng.integers(0, 11)) for _ in range(3)]})
                sw[str(d)] = prs
            json.dump({"items": sw, "self_bitwise": {"state": True}}, open(os.path.join(p, "eval_swaps.json"), "w"))
    return d0


def fixtures():
    res = []
    o = score(_world("hop")); res.append(("hop-per-pass stub -> C1-LANG (R2 passes via mean-state; R3 follows)", o["verdict"] == "C1-LANG" and o["R2"], (o["verdict"], o.get("acc_mean_state"), o.get("R3", {}).get("follow_U"))))
    o = score(_world("relay")); res.append(("relay / precompute stub -> R2 fails, not C1", (not o["R2"]) and o["verdict"] != "C1-LANG", (o["verdict"], o.get("acc_mean_state"))))
    o = score(_world("nogain")); res.append(("no gain -> NO-GAIN", o["verdict"] == "NO-GAIN", o["verdict"]))
    o = score(_world("bandfail")); res.append(("DIRECT solves d<=6 -> band [7, 8] < 3 depths -> UNREAD", o["verdict"].startswith("UNREAD"), (o["verdict"], o.get("band"))))
    d = _world("hop"); os.remove(os.path.join(d, "s228_lat_s1", "eval_swaps.json"))
    o = score(d); res.append(("missing -> pending", o["verdict"].startswith("pending"), o["verdict"]))
    o = score(d, final=True); res.append(("missing at --final -> FAIL", o["verdict"].startswith("FAIL (rule b"), o["verdict"]))
    d = _world("hop"); lg = os.path.join(d, "s228_lat_s1", "log.jsonl"); L = [json.loads(l) for l in open(lg)]; L[3]["ihash"] = "x"
    open(lg, "w").write("\n".join(json.dumps(r) for r in L) + "\n")
    o = score(d); res.append(("d-sequence differs -> VOID", o["verdict"].startswith("VOID"), o["verdict"]))
    d = _world("hop"); p = os.path.join(d, "s228_lat_s1", "train_summary.json"); json.dump({"last_step": 30000, "n_nonfinite": 0, "entry_d1": 900}, open(p, "w"))
    o = score(d); res.append(("dead LAT -> FAIL", o["verdict"].startswith("FAIL (dead"), o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--final", action="store_true"); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(final=a.final); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", newline=chr(10)), indent=1, default=str)
