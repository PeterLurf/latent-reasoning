"""THEORY Note 257-2: L1 HALT-EXTRAP (C5) + L2 HOLD-DIS (C6) on the §256 C2 models (eval-only on a host; committed before output; Theory is the only scorer).
Runs dis_s1 / dis_s2. Files:
  eval_halthead.json (§248 procedure on the REG stream: head fit on d U{1..4} items, features = the decoder's answer-position hidden at K' 0..8,
    y = [pred(K') == pred(8)], halt at the first K' with p >= 0.90 else 8; TEST2 d 1..7): per d {acc_k8, acc_halt, mean_khalt}; head.val_acc.
  eval_long_k128.json (§240 run_long): by_d[d].acc[K'] for K' in 8 / 64 / 128, d 1..6.
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
    return {"label": lab, "by_d": {d: {k: g(d, k) for k in ("acc_k8", "acc_halt", "mean_khalt")} for d in range(1, 8)}}


def hold(e):
    if e is None:
        return {"label": "pending"}
    b = e["by_d"]; bad = [d for d in ("1", "2", "3", "4", "5", "6") if any(b[d]["acc"][k] < b[d]["acc"]["8"] - 0.02 for k in ("64", "128"))]
    return {"label": "HOLD" if not bad else "DRIFT", "failing_d": bad, "acc": {d: b[d]["acc"] for d in b}}


def score(runs=R):
    o = {"halt": {r: halt(jl(os.path.join(runs, r, "eval_halthead.json"))) for r in RUNS},
         "hold": {r: hold(jl(os.path.join(runs, r, "eval_long_k128.json"))) for r in RUNS}}
    o["verdict"] = "L1 %s | L2 %s" % (" / ".join(o["halt"][r]["label"] for r in RUNS), " / ".join(o["hold"][r]["label"] for r in RUNS)); return o


def fixtures():
    res = []

    def world(acc_drop=0.0, k_ext=None, drift=False, val=0.95):
        d = tempfile.mkdtemp()
        for r in RUNS:
            p = os.path.join(d, r); os.makedirs(p)
            hh = {"head": {"val_acc": val}}
            for dd in range(1, 8):
                kh = dd - 0.8 if (k_ext is None or dd < 5) else k_ext
                hh[str(dd)] = {"acc_k8": 1.0, "acc_halt": 1.0 - (acc_drop if dd >= 5 else 0), "mean_khalt": kh}
            json.dump(hh, open(os.path.join(p, "eval_halthead.json"), "w"))
            json.dump({"by_d": {str(dd): {"acc": {"8": 1.0, "64": 0.9 if (drift and dd == 6) else 1.0, "128": 1.0}} for dd in range(1, 7)}}, open(os.path.join(p, "eval_long_k128.json"), "w"))
        return d
    for name, kw, want in (("extrap + hold", {}, "L1 C5-EXTRAP / C5-EXTRAP | L2 HOLD / HOLD"), ("acc lost out of range", {"acc_drop": 0.1}, "L1 C5-IN-RANGE / C5-IN-RANGE"),
                           ("halts late", {"k_ext": 8.0}, "L1 C5-IN-RANGE / C5-IN-RANGE"), ("drift", {"drift": True}, "L2 DRIFT / DRIFT"), ("void head", {"val": 0.7}, "L1 VOID / VOID")):
        o = score(world(**kw)); res.append((name, want in o["verdict"], o["verdict"]))
    o = score(tempfile.mkdtemp()); res.append(("pending", "pending" in o["verdict"], o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
