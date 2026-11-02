"""THEORY Note 259-7 C5-STABLE ($0; training-free answer-stability halting on stored per-item predictions; Theory is the only scorer; committed before computing).
Source: bun_s1 / bun_s2 eval_halthead2.json items per d 1..7: pred[K'] for K' 0..14 (the decoder's answer after K' passes), ans (the label).
Rule r in {1, 2}: halt at the first K' with pred[K'] == pred[K'+1] == ... == pred[K'+r] (r further passes unchanged); answer = pred[K']; khalt = K'; passes_run = K' + r.
  Censoring: no such K' with K' + r <= 14 -> khalt 14 with pred[14] (never-halt counted). acc_k8 = mean(pred[8] == ans).
Per seed and r: score_hx's C5 rule on khalt (IN-RANGE d 1-4; EXTRAP d 5-7: acc_halt >= acc_k8 - 0.02, mean_khalt <= d, strictly increasing over d 1-7). passes_run - (d - 1) reported.
  --fresh (Note 259-11): r = 3 only, on eval_stab_fresh.json (a new TEST draw; same layout, items pred / ans per d).
    python -B theory/score_c5c.py [--fresh] [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory"))
import score_hx as HX  # noqa: E402

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_c5c.json")
RUNS = ("bun_s1", "bun_s2")
KMAX = 14


def halt_items(pred, r):
    for k in range(0, KMAX - r + 1):
        if all(pred[k + j] == pred[k] for j in range(1, r + 1)):
            return k, False
    return KMAX, True


def per_d(items, r):
    n = len(items["pred"]); acc8 = ah = kh = pr = nh = 0
    for p, y in zip(items["pred"], items["ans"]):
        k, never = halt_items(p, r)
        acc8 += p[8] == y; ah += p[k] == y; kh += k; pr += min(k + r, KMAX); nh += never
    return {"n": n, "acc_k8": acc8 / n, "acc_halt": ah / n, "mean_khalt": kh / n, "mean_passes_run": pr / n, "never_halt_n": nh}


def score(runs=R, fname="eval_halthead2.json", rs=(1, 2)):
    o = {"runs": {}}
    for run in RUNS:
        try:
            h = json.load(open(os.path.join(runs, run, fname)))
        except (OSError, ValueError):
            o["runs"][run] = "pending"; continue
        o["runs"][run] = {}
        for r in rs:
            e = {"head": {"val_acc": 1.0}}
            for d in range(1, 8):
                e[str(d)] = per_d(h[str(d)]["items"], r)
            res = HX.halt(e); res["passes_minus_eff"] = {d: e[str(d)]["mean_passes_run"] - (d - 1) for d in range(1, 8)}
            o["runs"][run]["r%d" % r] = res
    if any(v == "pending" for v in o["runs"].values()):
        o["verdict"] = "pending"; return o
    o["verdict"] = " | ".join("r%d %s" % (r, " / ".join(o["runs"][run]["r%d" % r]["label"] for run in RUNS)) for r in rs); return o


def fixtures():
    res = []

    def world(stable_from_offset=0, flicker=False):
        d = tempfile.mkdtemp()
        for run in RUNS:
            p = os.path.join(d, run); os.makedirs(p); h = {}
            for dd in range(1, 8):
                preds, ans = [], []
                for i in range(50):
                    y = (i + dd) % 11; s0 = max(0, dd - 1 + stable_from_offset)
                    pr = [((y + 3 + k) % 11) if k < s0 else y for k in range(KMAX + 1)]
                    if flicker and dd >= 5 and s0 >= 2:
                        pr[s0 - 2] = pr[s0 - 1]          # an early coincidental repeat
                    preds.append(pr); ans.append(y)
                h[str(dd)] = {"items": {"pred": preds, "ans": ans}}
            json.dump(h, open(os.path.join(p, "eval_halthead2.json"), "w"))
        return d
    o = score(world()); res.append(("clean stability -> r1 and r2 EXTRAP", o["verdict"] == "r1 C5-EXTRAP / C5-EXTRAP | r2 C5-EXTRAP / C5-EXTRAP", o["verdict"]))
    o = score(world(flicker=True)); res.append(("coincidental early repeat breaks r1, not r2", o["verdict"].startswith("r1 C5-IN-RANGE") and "r2 C5-EXTRAP" in o["verdict"], o["verdict"]))
    w = world(flicker=True)
    for run in RUNS:
        os.rename(os.path.join(w, run, "eval_halthead2.json"), os.path.join(w, run, "eval_stab_fresh.json"))
    o = score(w, fname="eval_stab_fresh.json", rs=(3,)); res.append(("--fresh r3 on eval_stab_fresh.json", o["verdict"] == "r3 C5-EXTRAP / C5-EXTRAP", o["verdict"]))
    o = score(tempfile.mkdtemp()); res.append(("pending", o["verdict"] == "pending", o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); ap.add_argument("--fresh", action="store_true", help="Note 259-11: r = 3 on eval_stab_fresh.json")
    a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(fname="eval_stab_fresh.json", rs=(3,)) if a.fresh else score(); print(o["verdict"])
    out = OUT.replace(".json", "_fresh.json") if a.fresh else OUT
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(o, open(out, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
