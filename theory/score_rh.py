"""THEORY §247 READ-HALT scorer (docs/research_readhalt_2026-10-29/RH_v1.md; $0; Theory is the only scorer).
Per model (lh_fh_s1 / lh_fh_s2):
C7 from eval_c7.json cells d3t1, d3t2, d4t2, d4t3: ROUND-TRIP (pooled rt_agree >= 0.95, each cell >= 0.90); EDIT (pooled follow >= 0.50 and >= floor + 0.30,
  each cell >= 0.40). Labels C7-PASS / ROUND-TRIP-ONLY / REJECTED (/ VOID if a cell n < 900).
C5 from eval_halt.json d 1..4: HALT-ACC (acc_halt >= acc_k8 - 0.02 each d); HARDER-MORE (mean_khalt strictly increasing d 1..4); SAVING (mean_khalt <= d + 1).
  Labels C5-READOUT / NO-SIGNAL / COSTLY. Overall per criterion: shared label, else MIXED.
    python -B theory/score_rh.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_rh.json")
MODELS = ("lh_fh_s1", "lh_fh_s2")
CELLS = ("d3t1", "d3t2", "d4t2", "d4t3")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def c7(e):
    if e is None or any(c not in e for c in CELLS):
        return {"label": "pending"}
    if any(e[c]["n"] < 900 for c in CELLS):
        return {"label": "VOID", "why": "cell n < 900"}
    n = sum(e[c]["n"] for c in CELLS)
    rt = sum(e[c]["rt_agree"] * e[c]["n"] for c in CELLS) / n
    fo = sum(e[c]["edit_follow"] * e[c]["n"] for c in CELLS) / n
    fl = sum(e[c]["floor"] * e[c]["n"] for c in CELLS) / n
    rt_ok = rt >= 0.95 and all(e[c]["rt_agree"] >= 0.90 for c in CELLS)
    ed_ok = fo >= 0.50 and fo >= fl + 0.30 and all(e[c]["edit_follow"] >= 0.40 for c in CELLS)
    lab = "C7-PASS" if (rt_ok and ed_ok) else "ROUND-TRIP-ONLY" if rt_ok else "REJECTED"
    return {"label": lab, "rt_pooled": rt, "edit_pooled": fo, "floor_pooled": fl, "per_cell": {c: {k: e[c].get(k) for k in ("rt_agree", "edit_follow", "floor", "ctrl_aug_all", "oracle_rt")} for c in CELLS}}


def c5(e):
    if e is None or any(str(d) not in e for d in (1, 2, 3, 4)):
        return {"label": "pending"}
    h = {d: e[str(d)] for d in (1, 2, 3, 4)}
    acc_ok = all(h[d]["acc_halt"] >= h[d]["acc_k8"] - 0.02 for d in h)
    k = [h[d]["mean_khalt"] for d in (1, 2, 3, 4)]
    harder = all(k[i] < k[i + 1] for i in range(3))
    saving = all(h[d]["mean_khalt"] <= d + 1 for d in h)
    lab = "COSTLY" if not acc_ok else "NO-SIGNAL" if not harder else "C5-READOUT" if saving else "NO-SAVING"
    return {"label": lab, "mean_khalt": dict(zip((1, 2, 3, 4), k)), "acc": {d: (h[d]["acc_halt"], h[d]["acc_k8"]) for d in h},
            "desc_d5_6": {d: e.get(str(d)) and {x: e[str(d)].get(x) for x in ("acc_halt", "acc_k8", "mean_khalt")} for d in (5, 6)}}


def score(runs=R):
    o = {"section": 247, "registered": "RH_v1.md", "models": {}}
    for m in MODELS:
        o["models"][m] = {"C7": c7(jl(os.path.join(runs, m, "eval_c7.json"))), "C5": c5(jl(os.path.join(runs, m, "eval_halt.json")))}
    out = []
    for crit in ("C7", "C5"):
        labs = [o["models"][m][crit]["label"] for m in MODELS]
        v = "pending" if "pending" in labs else labs[0] if len(set(labs)) == 1 else "MIXED"
        o[crit] = v; out.append("%s %s (%s)" % (crit, v, " / ".join(labs)))
    o["verdict"] = " | ".join(out); return o


def fixtures():
    res = []

    def world(rt=0.98, fo=0.8, fl=0.09, n=1000, kh=(0.0, 1.0, 2.0, 3.0), acc_h=1.0, per=None):
        d = tempfile.mkdtemp()
        for m in MODELS:
            g = dict(rt=rt, fo=fo, fl=fl, n=n, kh=kh, acc_h=acc_h); g.update((per or {}).get(m, {}))
            os.makedirs(os.path.join(d, m))
            json.dump({c: {"n": g["n"], "intact_acc": 1.0, "rt_agree": g["rt"], "edit_follow": g["fo"], "floor": g["fl"]} for c in CELLS}, open(os.path.join(d, m, "eval_c7.json"), "w"))
            json.dump({str(dd): {"acc_k8": 1.0, "acc_halt": g["acc_h"], "mean_khalt": g["kh"][dd - 1] if dd <= 4 else 4.5} for dd in range(1, 7)}, open(os.path.join(d, m, "eval_halt.json"), "w"))
        return d
    o = score(world()); res.append(("readable + halts -> C7-PASS, C5-READOUT", o["C7"] == "C7-PASS" and o["C5"] == "C5-READOUT", o["verdict"]))
    o = score(world(rt=0.6)); res.append(("round trip fails -> REJECTED", o["C7"] == "REJECTED", o["verdict"]))
    o = score(world(fo=0.2)); res.append(("edit not followed -> ROUND-TRIP-ONLY", o["C7"] == "ROUND-TRIP-ONLY", o["verdict"]))
    o = score(world(n=500)); res.append(("small cell -> VOID", o["C7"] == "VOID", o["verdict"]))
    o = score(world(kh=(2.0, 2.0, 2.0, 2.0))); res.append(("flat halting -> NO-SIGNAL", o["C5"] == "NO-SIGNAL", o["verdict"]))
    o = score(world(acc_h=0.9)); res.append(("early halts cost acc -> COSTLY", o["C5"] == "COSTLY", o["verdict"]))
    o = score(world(per={"lh_fh_s2": {"rt": 0.6}})); res.append(("models disagree -> MIXED", o["C7"] == "MIXED", o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
