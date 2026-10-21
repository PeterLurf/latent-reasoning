"""Note 216-14 IDLE-HORIZON scorer ($0 eval-only grid on the six closed §215 / §216 models; decides whether a round-3 lever is needed; no §215 / §216
reading changes).

File contract: boxes/runs_fsq/<run>/eval_khorizon.json (boxes/vbank_diag.part_khorizon) =
  {"acc": {arm: {K': {n: float}}}, "items": {arm: {n: {K': [argmax answer]}}, "y": {n: [int]}, "idx": {n: [int]}}, "ks": [K'], "k_max": int,
   "gate_IH0": {"intact_k24_vs_eval_vb": {n: {"equal": bool, "khorizon", "eval_vb"}}, "k24_vs_eval_hist24": {arm: {n: share}}, "bitwise": bool,
                "min_share_vs_hist24": float}, "card": str, "eval_bs": int}
G-IH0: bitwise, else (every |intact K' 24 - eval_vb acc_k['24']| <= 0.002 and min share vs eval_hist24 >= 0.998) with the card recorded; else UNREADABLE.
Censor: k_max < 48 in a self model -> UNREADABLE. Missing file -> pending (215-7).
Reading (lineages: s1 self FT-SELF, R2-s1 vs ctrl FT-CTRL; s2 self R1-s2 vs ctrl D3-T s2). Delta(m, n) = intact a(n, 24) - a(n, Kmax), Kmax = the largest
read K' (64 unless censored); SE(m, n) = the paired (McNemar) SE of that difference over the n's items.
  H_DRIFT iff some self model and n in 1..3 has Delta(self) - Delta(ctrl) >= 0.02 and Delta(self) >= 2 SE(self) ("tentative" if only n = 1 qualifies);
  H_RED iff every self model has intact a >= 0.99 at every n 3..8 and read K', and Delta(self) - Delta(ctrl) < 0.02 at every n 1..3;
  NEITHER otherwise.
Descriptive: per model the four arms' a(n, K') for n 1..8; BANK = self_ablate - bank_ablate_slva and STATE = bank_ablate_slva per (n, K'); R1-s2's n 1
K' 24 intact errors cross-tabbed with bank_ablate_slva correctness.
    python theory/score_s216_ih.py [--fixtures]
"""
import argparse, json, math, os, sys, tempfile
import numpy as np

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "boxes", "runs_fsq")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "score_s216_ih.json")
LINEAGES = {"s1": {"ctrl": "s215_ftctrl_s1", "self": ["s215_ftself_s1", "s216_r2_s1"]}, "s2": {"ctrl": "s216_d3t_s2", "self": ["s216_r1_s2"]}}
ALL = ["s213_d3t_s1", "s215_ftctrl_s1", "s215_ftself_s1", "s216_r2_s1", "s216_d3t_s2", "s216_r1_s2"]
ARMS = ["intact", "bank_ablate_slva", "self_ablate", "other_ablate"]
DRIFT, RED_ACC, TOL_VB, TOL_SH = 0.02, 0.99, 0.002, 0.998


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def gate(ev):
    g = ev.get("gate_IH0") or {}
    if g.get("bitwise") is True:
        return True, "bitwise"
    vb = g.get("intact_k24_vs_eval_vb") or {}
    d = [abs(v["khorizon"] - v["eval_vb"]) for v in vb.values() if isinstance(v, dict) and v.get("eval_vb") is not None and v.get("khorizon") is not None]
    sh = g.get("min_share_vs_hist24")
    if d and max(d) <= TOL_VB and sh is not None and sh >= TOL_SH:
        return True, "agreement (max |d| %.4f, min share %.4f, card %s)" % (max(d), sh, ev.get("card"))
    return False, "G-IH0 failed (max |d| %s, min share %s)" % (max(d) if d else None, sh)


def ok_vec(ev, arm, n, k):
    a = np.asarray(ev["items"][arm][str(n)][str(k)]); y = np.asarray(ev["items"]["y"][str(n)]); return (a == y).astype(np.int64)


def delta(ev, n):
    km = max(int(k) for k in ev["ks"]); o24, ok = ok_vec(ev, "intact", n, 24), ok_vec(ev, "intact", n, km); N = len(o24)
    b = int(((o24 == 1) & (ok == 0)).sum()); c = int(((o24 == 0) & (ok == 1)).sum()); d = (b - c) / N
    se = math.sqrt(max(0.0, (b + c) - (b - c) ** 2 / N)) / N
    return d, se, km


def describe(ev):
    out = {"acc": ev.get("acc"), "k_max": ev.get("k_max"), "card": ev.get("card")}
    acc = ev.get("acc") or {}
    if "self_ablate" in acc and "bank_ablate_slva" in acc:
        out["BANK"] = {k: {n: acc["self_ablate"][k][n] - acc["bank_ablate_slva"][k][n] for n in acc["self_ablate"][k]} for k in acc["self_ablate"]}
        out["STATE"] = acc["bank_ablate_slva"]
    return out


def score(runs=RUNS):
    out = {"note": "216-14", "models": {}}
    E = {}
    for r in ALL:
        ev = jl(os.path.join(runs, r, "eval_khorizon.json"))
        if ev is None:
            out["models"][r] = {"status": "pending (eval_khorizon.json)"}; continue
        okg, why = gate(ev)
        out["models"][r] = {"gate": why, **describe(ev)}
        if not okg:
            out["models"][r]["status"] = "UNREADABLE"; continue
        E[r] = ev; out["models"][r]["status"] = "READ"
    need = [LINEAGES[l]["ctrl"] for l in LINEAGES] + [s for l in LINEAGES for s in LINEAGES[l]["self"]]
    if any(out["models"][r]["status"].startswith("pending") for r in need):
        out["verdict"] = "pending"; return out
    if any(out["models"][r]["status"] == "UNREADABLE" for r in need):
        out["verdict"] = "UNREADABLE (G-IH0)"; return out
    if any(E[s]["k_max"] < 48 for l in LINEAGES for s in LINEAGES[l]["self"]):
        out["verdict"] = "UNREADABLE (k_max < 48)"; return out
    rows, drift_hits, red = [], [], True
    for l, L in LINEAGES.items():
        c = E[L["ctrl"]]
        for s in L["self"]:
            e = E[s]
            for n in (1, 2, 3):
                ds, ses, km = delta(e, n); dc, sec, _ = delta(c, n); exc = ds - dc
                hit = exc >= DRIFT and ds >= 2 * ses
                rows.append({"lineage": l, "self": s, "n": n, "delta_self": ds, "se_self": ses, "delta_ctrl": dc, "excess": exc, "kmax": km, "drift": hit})
                if hit:
                    drift_hits.append((s, n))
                if exc >= DRIFT:
                    red = False
            for n in range(3, 9):
                for k in e["ks"]:
                    if e["acc"]["intact"][str(k)][str(n)] < RED_ACC:
                        red = False
    out["rows"] = rows
    if drift_hits:
        out["verdict"] = "H_DRIFT" + (" (tentative: n = 1 only)" if all(n == 1 for _, n in drift_hits) else ""); out["drift_hits"] = drift_hits
    elif red:
        out["verdict"] = "H_RED"
    else:
        out["verdict"] = "NEITHER"
    r1 = E.get("s216_r1_s2")
    if r1 is not None:
        i, b = ok_vec(r1, "intact", 1, 24), ok_vec(r1, "bank_ablate_slva", 1, 24)
        out["r1s2_n1_k24_errors"] = {"n_err": int((i == 0).sum()), "err_and_state_correct": int(((i == 0) & (b == 1)).sum()), "err_and_state_wrong": int(((i == 0) & (b == 0)).sum())}
    return out


# ------------------------------------------------------------------ fixtures
def _ev(acc_fn, kmax=64, bitwise=True, N=1000, seed=0):
    rng = np.random.default_rng(seed); ks = [k for k in (24, 32, 48, 64) if k <= kmax]
    it = {"y": {str(n): [0] * N for n in range(1, 9)}}; acc = {}
    base = {n: rng.random(N) for n in range(1, 9)}                         # one latent per item: monotone decay keeps items nested across K'
    for arm in ARMS:
        it[arm] = {}; acc[arm] = {str(k): {} for k in ks}
        for n in range(1, 9):
            it[arm][str(n)] = {}
            for k in ks:
                p = acc_fn(arm, n, k); ok = base[n] < p; it[arm][str(n)][str(k)] = [0 if o else 1 for o in ok]; acc[arm][str(k)][str(n)] = float(ok.mean())
    return {"acc": acc, "items": it, "ks": ks, "k_max": kmax, "card": "fixture", "gate_IH0": {"bitwise": bitwise, "intact_k24_vs_eval_vb": {}, "min_share_vs_hist24": 0.5}}


def _put(d, evs):
    for r in ALL:
        if evs.get(r) is not None:
            os.makedirs(os.path.join(d, r), exist_ok=True); json.dump(evs[r], open(os.path.join(d, r, "eval_khorizon.json"), "w"))


def fixtures():
    flat = lambda arm, n, k: 0.999 if arm == "intact" else 0.6  # noqa: E731
    def decay(rate, ns=(1, 2, 3)):
        return lambda arm, n, k: (0.999 - (rate * (k - 24) if n in ns else 0.0)) if arm == "intact" else 0.6
    res = []
    def run(evs):
        d = tempfile.mkdtemp(); _put(d, evs); return score(d)
    base = {r: _ev(flat, seed=i) for i, r in enumerate(ALL)}
    o = run(base); res.append(("flat everywhere -> H_RED", o["verdict"] == "H_RED", o["verdict"]))
    e = dict(base); e["s215_ftself_s1"] = _ev(decay(0.002), seed=9)
    o = run(e); res.append(("FT-SELF decays 0.08 at n 1-3 -> H_DRIFT", o["verdict"] == "H_DRIFT", o["verdict"]))
    e = dict(base); e["s215_ftself_s1"] = _ev(decay(0.002), seed=9); e["s215_ftctrl_s1"] = _ev(decay(0.002), seed=10)
    o = run(e); res.append(("self and ctrl decay alike -> not H_DRIFT", not o["verdict"].startswith("H_DRIFT"), o["verdict"]))
    e = dict(base); e["s216_r1_s2"] = _ev(decay(0.002, ns=(1,)), seed=11)
    o = run(e); res.append(("n 1 only -> H_DRIFT tentative", o["verdict"] == "H_DRIFT (tentative: n = 1 only)", o["verdict"]))
    e = dict(base); e["s216_r2_s1"] = _ev(lambda arm, n, k: (0.97 if n == 5 else 0.999) if arm == "intact" else 0.6, seed=12)
    o = run(e); res.append(("level offset < 0.99 at n 5, no trend -> NEITHER", o["verdict"] == "NEITHER", o["verdict"]))
    e = dict(base); e["s215_ftself_s1"] = _ev(flat, bitwise=False, seed=13)
    o = run(e); res.append(("G-IH0 fail -> UNREADABLE", o["verdict"].startswith("UNREADABLE (G-IH0"), o["verdict"]))
    e = dict(base); e["s216_r1_s2"] = _ev(flat, kmax=32, seed=14)
    o = run(e); res.append(("k_max 32 -> UNREADABLE", o["verdict"] == "UNREADABLE (k_max < 48)", o["verdict"]))
    e = dict(base); e["s216_r2_s1"] = None
    o = run(e); res.append(("missing file -> pending", o["verdict"] == "pending", o["verdict"]))
    e = dict(base); e["s215_ftself_s1"] = _ev(decay(0.0001), seed=15)
    o = run(e); res.append(("tiny decay 0.004 -> H_RED", o["verdict"] == "H_RED", o["verdict"]))
    e = dict(base); e["s215_ftself_s1"] = _ev(decay(0.002), kmax=48, seed=16)
    o = run(e); res.append(("k_max 48 censor, decay 0.048 -> H_DRIFT", o["verdict"] == "H_DRIFT" and all(r_["kmax"] == 48 for r_ in o["rows"] if r_["self"] == "s215_ftself_s1"), o["verdict"]))
    o = run(base); res.append(("R1-s2 n1 crosstab present", "r1s2_n1_k24_errors" in o, o.get("r1s2_n1_k24_errors")))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", newline=chr(10)), indent=1, default=str)
