"""THEORY §242 NORM-COUNTER scorer (docs/research_normctr_2026-10-28/NC_v1.md; $0 causal test; Theory is the only scorer).
Per model (lh_fh_s1 / lh_fh_s2), eval_normctr.json arms intact_d4 / intact_d6 / down_d6 / up_d4 / harm_d4 {acc, ws {t}, n}:
HARMLESS acc(harm_d4) >= 0.90 (else INCONCLUSIVE); DOWN acc(down_d6) - acc(intact_d6) >= 0.15 and ws5 gain >= 0.15; UP acc(intact_d4) - acc(up_d4) >= 0.15.
Labels: NORM-COUNTER / PARTIAL-DOWN / PARTIAL-UP / NOT-NORM / INCONCLUSIVE; overall the shared label, else MIXED.
    python -B theory/score_nc.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_nc.json")
MODELS = ("lh_fh_s1", "lh_fh_s2")
ARMS = ("intact_d4", "intact_d6", "down_d6", "up_d4", "harm_d4")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def one(e):
    if e is None or any(a not in e for a in ARMS):
        return {"label": "pending"}
    acc = {a: e[a]["acc"] for a in ARMS}
    harmless = acc["harm_d4"] >= 0.90
    dws = e["down_d6"]["ws"]["5"] - e["intact_d6"]["ws"]["5"]
    down = acc["down_d6"] - acc["intact_d6"] >= 0.15 and dws >= 0.15
    up = acc["intact_d4"] - acc["up_d4"] >= 0.15
    lab = ("INCONCLUSIVE" if not harmless else "NORM-COUNTER" if (down and up) else "PARTIAL-DOWN" if down else "PARTIAL-UP" if up else "NOT-NORM")
    return {"label": lab, "acc": acc, "ws5_gain_down": dws, "down": down, "up": up, "harmless": harmless}


def score(runs=R):
    o = {"section": 242, "registered": "NC_v1.md", "models": {m: one(jl(os.path.join(runs, m, "eval_normctr.json"))) for m in MODELS}}
    labs = [v["label"] for v in o["models"].values()]
    if "pending" in labs:
        o["verdict"] = "pending"; return o
    o["verdict"] = "NORM-COUNTER %s | %s" % (labs[0] if len(set(labs)) == 1 else "MIXED", " / ".join("%s: %s" % (m[-2:], v["label"]) for m, v in o["models"].items()))
    return o


def fixtures():
    res = []

    def world(i4=1.0, i6=0.3, dn=0.3, up=1.0, hm=1.0, ws_i6=0.5, ws_dn=None, per=None):
        d = tempfile.mkdtemp()
        for m in MODELS:
            v = dict(i4=i4, i6=i6, dn=dn, up=up, hm=hm, ws_i6=ws_i6, ws_dn=ws_dn); v.update((per or {}).get(m, {}))
            wd = v["ws_dn"] if v["ws_dn"] is not None else v["ws_i6"]
            arms = {"intact_d4": {"acc": v["i4"], "ws": {"5": 1.0}}, "intact_d6": {"acc": v["i6"], "ws": {"5": v["ws_i6"]}}, "down_d6": {"acc": v["dn"], "ws": {"5": wd}},
                    "up_d4": {"acc": v["up"], "ws": {"5": 1.0}}, "harm_d4": {"acc": v["hm"], "ws": {"5": 1.0}}}
            os.makedirs(os.path.join(d, m)); json.dump(arms, open(os.path.join(d, m, "eval_normctr.json"), "w"))
        return d
    o = score(world(dn=0.7, ws_dn=0.9, up=0.5)); res.append(("down helps, up hurts -> NORM-COUNTER", o["verdict"].startswith("NORM-COUNTER NORM-COUNTER"), o["verdict"]))
    o = score(world()); res.append(("no effect -> NOT-NORM", o["verdict"].startswith("NORM-COUNTER NOT-NORM"), o["verdict"]))
    o = score(world(hm=0.5)); res.append(("rescale breaks reads -> INCONCLUSIVE", o["verdict"].startswith("NORM-COUNTER INCONCLUSIVE"), o["verdict"]))
    o = score(world(dn=0.7, ws_dn=0.9)); res.append(("down only -> PARTIAL-DOWN", o["verdict"].startswith("NORM-COUNTER PARTIAL-DOWN"), o["verdict"]))
    o = score(world(dn=0.7, ws_dn=0.55)); res.append(("acc up but ws5 not -> not DOWN", "PARTIAL-DOWN" not in o["verdict"], o["verdict"]))
    o = score(world(per={"lh_fh_s2": {"up": 0.5}})); res.append(("models disagree -> MIXED", o["verdict"].startswith("NORM-COUNTER MIXED"), o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
