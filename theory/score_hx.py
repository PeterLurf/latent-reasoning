"""THEORY Note 257-2: L1 HALT-EXTRAP (C5) + L2 HOLD-DIS (C6) on the §256 C2 models (eval-only on a host; committed before output; Theory is the only scorer).
Runs dis_s1 / dis_s2. Files:
  eval_halthead.json (§248 procedure on the REG stream: head fit on d U{1..4} items, features = the decoder's answer-position hidden at K' 0..8,
    y = [pred(K') == pred(8)], halt at the first K' with p >= 0.90 else 8; TEST2 d 1..7): per d {acc_k8, acc_halt, mean_khalt, khalt_hist, never_halt_n, n}; head.val_acc.
  Censoring (Audit 257-2): a never-halting item counts as halting at K' 8 (its K' 8 answer in acc_halt, 8 in mean_khalt), as the eval does; the never-halt share and the
  event-only mean are reported beside it, as is mean_khalt - (d - 1) (the decoder adds one hop, so the efficient halt is d - 1).
  eval_long_k128.json (§240 run_long): by_d[d].acc[K'] for K' in 8 / 64 / 128, d 1..6; by_d[d].s_norm[K'] if present (||S|| ratio K' 128 / 8 reported, descriptive).
L1 per seed (VOID if head val_acc < 0.80): IN-RANGE d 1-4 = §247 C5 gates (acc_halt >= acc_k8 - 0.02; mean_khalt increasing; mean_khalt <= d + 1);
  EXTRAP d 5-7: acc_halt >= acc_k8 - 0.02 each d, and mean_khalt <= d each d, and mean_khalt strictly increasing over d 1..7.
  Labels: C5-EXTRAP (IN-RANGE + EXTRAP) / C5-IN-RANGE (IN-RANGE only) / NO-SIGNAL.
L2 per seed: HOLD if every d 1..6 has acc[64] and acc[128] >= acc[8] - 0.02; else DRIFT (the failing d named).
    python -B theory/score_hx.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_hx.json")
RUNS = ("dis_s1", "dis_s2")
RUNS_DCH = ("dch_s1", "dch_s2")   # Note 259-1: the same L1 / L2 rules on the §257 models (the bundle init)


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def halt(e):
    if e is None:
        return {"label": "pending"}
    if (e.get("head") or {}).get("val_acc", 1.0) < 0.80:
        return {"label": "VOID", "why": "head val_acc < 0.80"}
    g = lambda d, k: e[str(d)][k]  # noqa: E731
    ok_acc = lambda ds: all(g(d, "acc_halt") >= g(d, "acc_k8") - 0.02 for d in ds)  # noqa: E731
    inc = lambda ds: all(g(a, "mean_khalt") < g(b, "mean_khalt") for a, b in zip(ds, ds[1:]))  # noqa: E731
    inr = ok_acc(range(1, 5)) and inc(list(range(1, 5))) and all(g(d, "mean_khalt") <= d + 1 for d in range(1, 5))
    ext = ok_acc(range(5, 8)) and all(g(d, "mean_khalt") <= d for d in range(5, 8)) and inc(list(range(1, 8)))
    lab = "C5-EXTRAP" if (inr and ext) else "C5-IN-RANGE" if inr else "NO-SIGNAL"
    def extra(d):
        r = e[str(d)]; n = r.get("n"); nh = r.get("never_halt_n", 0); hist = r.get("khalt_hist") or {}
        ev = None
        if n and hist and n > nh:
            ev = (sum(int(k) * v for k, v in hist.items()) - 8 * nh) / (n - nh)
        return {"never_halt_share": (nh / n) if n else None, "event_only_mean": ev, "offset_vs_d_minus_1": r["mean_khalt"] - (d - 1)}
    return {"label": lab, "by_d": {d: {**{k: g(d, k) for k in ("acc_k8", "acc_halt", "mean_khalt")}, **extra(d)} for d in range(1, 8)}}


def hold(e):
    if e is None:
        return {"label": "pending"}
    b = e["by_d"]; bad = [d for d in ("1", "2", "3", "4", "5", "6") if any(b[d]["acc"][k] < b[d]["acc"]["8"] - 0.02 for k in ("64", "128"))]
    ratio = {d: (b[d]["s_norm"]["128"] / b[d]["s_norm"]["8"]) for d in b if isinstance(b[d].get("s_norm"), dict) and b[d]["s_norm"].get("8")}
    return {"label": "HOLD" if not bad else "DRIFT", "failing_d": bad, "acc": {d: b[d]["acc"] for d in b}, "s_norm_ratio_128_8": ratio}


def score(runs=R, which=RUNS):
    o = {"halt": {r: halt(jl(os.path.join(runs, r, "eval_halthead.json"))) for r in which},
         "hold": {r: hold(jl(os.path.join(runs, r, "eval_long_k128.json"))) for r in which}}
    o["verdict"] = "L1 %s | L2 %s" % (" / ".join(o["halt"][r]["label"] for r in which), " / ".join(o["hold"][r]["label"] for r in which)); return o


def fixtures():
    res = []

    def world(acc_drop=0.0, k_ext=None, drift=False, val=0.95, never=0.0):
        d = tempfile.mkdtemp()
        for r in RUNS:
            p = os.path.join(d, r); os.makedirs(p)
            hh = {"head": {"val_acc": val}}
            for dd in range(1, 8):
                kh = dd - 0.8 if (k_ext is None or dd < 5) else k_ext
                if never and dd >= 5:
                    kh = (1 - never) * kh + never * 8.0
                hh[str(dd)] = {"acc_k8": 1.0, "acc_halt": 1.0 - (acc_drop if dd >= 5 else 0), "mean_khalt": kh, "n": 1000, "never_halt_n": int(never * 1000) if dd >= 5 else 0,
                               "khalt_hist": {str(max(0, dd - 1)): 1000 - (int(never * 1000) if dd >= 5 else 0), "8": int(never * 1000) if dd >= 5 else 0}}
            json.dump(hh, open(os.path.join(p, "eval_halthead.json"), "w"))
            json.dump({"by_d": {str(dd): {"acc": {"8": 1.0, "64": 0.9 if (drift and dd == 6) else 1.0, "128": 1.0}} for dd in range(1, 7)}}, open(os.path.join(p, "eval_long_k128.json"), "w"))
        return d
    for name, kw, want in (("extrap + hold", {}, "L1 C5-EXTRAP / C5-EXTRAP | L2 HOLD / HOLD"), ("acc lost out of range", {"acc_drop": 0.1}, "L1 C5-IN-RANGE / C5-IN-RANGE"),
                           ("halts late", {"k_ext": 8.0}, "L1 C5-IN-RANGE / C5-IN-RANGE"), ("drift", {"drift": True}, "L2 DRIFT / DRIFT"), ("void head", {"val": 0.7}, "L1 VOID / VOID"), ("30% never halt", {"never": 0.3}, "L1 C5-IN-RANGE / C5-IN-RANGE")):
        o = score(world(**kw)); res.append((name, want in o["verdict"], o["verdict"]))
    o = score(tempfile.mkdtemp()); res.append(("pending", "pending" in o["verdict"], o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); ap.add_argument("--dch", action="store_true", help="Note 259-1: dch_s1 / dch_s2"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(which=RUNS_DCH if a.dch else RUNS); print(o["verdict"])
    out = OUT.replace(".json", "_dch.json") if a.dch else OUT
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(o, open(out, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
