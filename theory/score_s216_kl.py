"""Note 216-10 KL-DRIFT scorer ($0 side row on the closed §215 / §216 models; descriptive; no §215 / §216 reading changes).

File contract: boxes/runs_fsq/<run>/eval_kl_base.json =
  {"meta": {"card": str, "dtype": "fp32", "eval_bs": 32, "base_run": str, "items_sha256": str},
   "gate": {"acc_n2": {"3".."8": float}, "acc_k24": {"3".."8": float}},          # the model's intact 4060 values
   "items": [{"n": int, "idx": int, "y": int, "lp_r": {"n2": [11], "24": [11]}, "lp_b": {"n2": [11], "24": [11]}}]}   # log-probs over the 11 value classes
Gate: |gate - committed eval_vb (acc_n2, acc_k['24'])| <= 0.005 at every n 3..8, else UNREADABLE. Missing file -> pending (215-7).
Reading (s1, n 3..6, K' 24; item bootstrap 10,000, seed 21610, one resample of item indices shared by the two models since items are the same test items):
  OUTPUT-DRIFT iff ratio KL_SELF / KL_CTRL >= 2 with CI95 low > 1 and KL_R2 >= KL_SELF; INTERNAL-ONLY iff CI95 contains 1 or ratio < 1.25; else MIXED.
    python theory/score_s216_kl.py [--fixtures]
"""
import argparse, json, math, os, sys
import numpy as np

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "boxes", "runs_fsq")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "score_s216_kl.json")
MODELS = {"s1": {"base": "s213_d3t_s1", "ctrl": "s215_ftctrl_s1", "self": "s215_ftself_s1", "r2": "s216_r2_s1"},
          "s2": {"base": "s216_d3t_s2", "r1": "s216_r1_s2"}}
STAB = {"s213_d3t_s1": 0.782, "s215_ftctrl_s1": 0.696, "s215_ftself_s1": 0.489, "s216_r2_s1": 0.437, "s216_d3t_s2": 0.684, "s216_r1_s2": 0.489}
BINS = {"n3_6": range(3, 7), "n9_16": range(9, 17), "n1_2": range(1, 3)}
TOL, NBOOT, SEED = 0.005, 10000, 21610


def jl(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def kl_item(lp_b, lp_r):
    """KL(p_b || p_r) from log-probs; renormalised over the 11 classes (log-sum-exp) so a writer's restriction to NUM_IDS is exact."""
    a, b = np.asarray(lp_b, dtype=np.float64), np.asarray(lp_r, dtype=np.float64)
    a = a - np.logaddexp.reduce(a); b = b - np.logaddexp.reduce(b)
    return float(max(0.0, np.sum(np.exp(a) * (a - b))))


def gate(f, ev):
    if ev is None:
        return False, "eval_vb.json missing"
    bad = [(n, k) for n in range(3, 9) for k, ref in (("acc_n2", ev["acc_n2"][str(n)]), ("acc_k24", ev["acc_k"]["24"][str(n)]))
           if abs(f["gate"][k][str(n)] - ref) > TOL + 1e-12]
    return (not bad), ("ok" if not bad else "off by > %.3f at %s" % (TOL, bad))


def per_model(run, runs=RUNS):
    f = jl(os.path.join(runs, run, "eval_kl_base.json"))
    if f is None:
        return {"run": run, "status": "pending (missing eval_kl_base.json)"}
    ok, why = gate(f, jl(os.path.join(runs, run, "eval_vb.json")))
    if not ok:
        return {"run": run, "status": "UNREADABLE (gate: %s)" % why}
    kl = {"n2": {}, "24": {}}; agree = {"n2": {}, "24": {}}; acc = {"n2": {}, "24": {}}
    for it in f["items"]:
        for K in ("n2", "24"):
            kl[K][it["idx"]] = (it["n"], kl_item(it["lp_b"][K], it["lp_r"][K]))
            agree[K][it["idx"]] = int(np.argmax(it["lp_b"][K]) == np.argmax(it["lp_r"][K])); acc[K][it["idx"]] = int(np.argmax(it["lp_r"][K]) == it["y"])
    summ = {K: {b: {"kl": float(np.mean([v for (n, v) in kl[K].values() if n in rg])) if any(n in rg for (n, _) in kl[K].values()) else None,
                    "agree": float(np.mean([agree[K][i] for i, (n, _) in kl[K].items() if n in rg])) if any(n in rg for (n, _) in kl[K].values()) else None,
                    "acc": float(np.mean([acc[K][i] for i, (n, _) in kl[K].items() if n in rg])) if any(n in rg for (n, _) in kl[K].values()) else None}
                for b, rg in BINS.items()} for K in ("n2", "24")}
    return {"run": run, "status": "ok", "base": f["meta"].get("base_run"), "summary": summ, "_kl": kl}


def ratio_boot(klA, klB, rg, seed=SEED, nboot=NBOOT):
    """ratio of mean KL (A / B) over shared item idx in bin rg; paired item bootstrap."""
    idx = sorted(i for i, (n, _) in klA.items() if n in rg and i in klB)
    a = np.array([klA[i][1] for i in idx]); b = np.array([klB[i][1] for i in idx])
    if not len(idx) or b.mean() <= 0:
        return None
    rng = np.random.default_rng(seed); R = np.empty(nboot)
    for k in range(nboot):
        s = rng.integers(0, len(idx), len(idx)); R[k] = a[s].mean() / max(b[s].mean(), 1e-300)
    return {"ratio": float(a.mean() / b.mean()), "ci95": [float(np.quantile(R, 0.025)), float(np.quantile(R, 0.975))], "n_items": len(idx)}


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else None


def score(runs=RUNS):
    out = {"note": "216-10", "models": {}}
    for lin, ms in MODELS.items():
        for role, run in ms.items():
            if role != "base":
                out["models"][run] = per_model(run, runs)
    s1 = [out["models"][MODELS["s1"][r]] for r in ("ctrl", "self", "r2")]
    if any(m["status"] != "ok" for m in s1):
        st = [m["status"] for m in s1]
        out["reading"] = {"label": "pending" if all(not s.startswith("UNREADABLE") for s in st) else "FAILURE (UNREADABLE model; rule b)", "status": st}
    else:
        ctrl, slf, r2 = s1; rg = BINS["n3_6"]
        rb = ratio_boot(slf["_kl"]["24"], ctrl["_kl"]["24"], rg)
        kS, kR = slf["summary"]["24"]["n3_6"]["kl"], r2["summary"]["24"]["n3_6"]["kl"]
        if rb is None:
            lab = "UNDEFINED (KL_CTRL = 0)"
        elif rb["ratio"] >= 2 and rb["ci95"][0] > 1 and kR >= kS:
            lab = "OUTPUT-DRIFT"
        elif (rb["ci95"][0] <= 1 <= rb["ci95"][1]) or rb["ratio"] < 1.25:
            lab = "INTERNAL-ONLY"
        else:
            lab = "MIXED"
        dS = [STAB[MODELS["s1"]["base"]] - STAB[m["run"]] for m in s1]
        out["reading"] = {"label": lab, "ratio_self_ctrl_k24_n3_6": rb, "KL_k24_n3_6": {m["run"]: m["summary"]["24"]["n3_6"]["kl"] for m in s1},
                          "spearman_dSTAB_KL": spearman(dS, [m["summary"]["24"]["n3_6"]["kl"] for m in s1]),
                          "wording": "one FT pair, one seed: an instance"}
    for m in out["models"].values():
        m.pop("_kl", None)
    return out


def _fixture(tmp, run, base_lp, r_lp_fn, n_items=50, gate_off=0.0):
    os.makedirs(os.path.join(tmp, run), exist_ok=True); items = []; i = 0
    for n in list(range(1, 7)) + list(range(9, 17)):
        for _ in range(n_items):
            lb = list(base_lp); items.append({"n": n, "idx": i, "y": 0, "lp_b": {"n2": lb, "24": lb}, "lp_r": {"n2": r_lp_fn(i), "24": r_lp_fn(i)}}); i += 1
    g = {k: {str(n): 1.0 + (gate_off if n == 5 else 0.0) for n in range(3, 9)} for k in ("acc_n2", "acc_k24")}
    json.dump({"meta": {"base_run": "x", "card": "fixture", "dtype": "fp32", "eval_bs": 32}, "gate": g, "items": items}, open(os.path.join(tmp, run, "eval_kl_base.json"), "w"))
    json.dump({"acc_n2": {str(n): 1.0 for n in range(1, 17)}, "acc_k": {"24": {str(n): 1.0 for n in range(1, 17)}}}, open(os.path.join(tmp, run, "eval_vb.json"), "w"))


def fixtures():
    import tempfile
    rng = np.random.default_rng(0); base = np.log(np.full(11, 0.01)); base[0] = math.log(0.9); base = base - np.logaddexp.reduce(base)
    def shift(s):
        def f(i):
            v = base.copy(); v[0] += -s * (1 + 0.1 * ((i * 7919) % 13) / 13); v = v - np.logaddexp.reduce(v); return list(v)
        return f
    checks = []
    def case(name, s_ctrl, s_self, s_r2, expect, gate_off=0.0, drop=None):
        tmp = tempfile.mkdtemp()
        for run, s in ((MODELS["s1"]["ctrl"], s_ctrl), (MODELS["s1"]["self"], s_self), (MODELS["s1"]["r2"], s_r2), (MODELS["s2"]["r1"], s_self)):
            if run != drop:
                _fixture(tmp, run, base, shift(s), gate_off=gate_off if run == MODELS["s1"]["self"] else 0.0)
        got = score(tmp)["reading"]["label"]; checks.append((name, got.startswith(expect), got))
    case("drift", 0.1, 1.0, 1.5, "OUTPUT-DRIFT")
    case("internal (equal)", 0.5, 0.5, 0.6, "INTERNAL-ONLY")
    case("internal (ratio<1.25)", 0.5, 0.55, 2.0, "INTERNAL-ONLY")
    case("mixed (R2 below SELF)", 0.1, 1.0, 0.5, "MIXED")
    case("gate fail -> rule b", 0.1, 1.0, 1.5, "FAILURE", gate_off=0.01)
    case("missing -> pending", 0.1, 1.0, 1.5, "pending", drop=MODELS["s1"]["r2"])
    assert kl_item(list(base), list(base)) == 0.0
    v = base.copy(); v2 = list(v + 3.0); checks.append(("renormalised (shift-invariant)", abs(kl_item(list(base), v2)) < 1e-12, kl_item(list(base), v2)))
    for c in checks:
        print("FIXTURE", "PASS" if c[1] else "FAIL", c[0], c[2])
    n_ok = sum(c[1] for c in checks); print("fixtures %d/%d" % (n_ok, len(checks))); return n_ok == len(checks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(json.dumps(o["reading"], default=str)[:2000])
    json.dump(o, open(OUT, "w", newline=chr(10)), indent=1, default=str)
