"""THEORY §249 FRAME-TRANSFER scorer (docs/research_frametransfer_2026-10-30/FT_v1.md; $0; Theory is the only scorer).
Per model (lh_fh_s1 / lh_fh_s2), on TRANSFER-DF (frames 6-7, never trained) from eval_transfer.json {by_d: {d: {acc, ws {t}}}, cue_only {d: acc, majority}}
and CD-AN df_pcd4_s1/eval_transfer.json (acc): T-ACC (d 2-4 acc >= 0.80 and >= CD-AN_transfer + 0.25); T-CHAIN (WS >= 0.50 at every pass 2..d, d 3-4);
T-GUARD (cue-only <= max(majority, CD-AN) + 0.05 at d 2-4, else VOID). Labels VOID / TRANSFER / ACC-ONLY / NO-TRANSFER; overall both, else MIXED.
    python -B theory/score_ft.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_ft.json")
MODELS, CD = ("lh_fh_s1", "lh_fh_s2"), "df_pcd4_s1"


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def one(e, c):
    if e is None or c is None:
        return {"label": "pending"}
    b, cb = e["by_d"], c["by_d"]
    co = e.get("cue_only") or {}
    guard = all(co.get(d, {}).get("acc", 0) <= max(co.get(d, {}).get("majority", 0), cb[d]["acc"]) + 0.05 for d in ("2", "3", "4"))
    acc_ok = all(b[d]["acc"] >= 0.80 and b[d]["acc"] >= cb[d]["acc"] + 0.25 for d in ("2", "3", "4"))
    chain = all(b[d]["ws"][str(t)] >= 0.50 for d in ("3", "4") for t in range(2, int(d) + 1))
    lab = "VOID" if not guard else "TRANSFER" if (acc_ok and chain) else "ACC-ONLY" if acc_ok else "NO-TRANSFER"
    return {"label": lab, "acc": {d: b[d]["acc"] for d in b}, "cd_acc": {d: cb[d]["acc"] for d in cb}, "ws": {d: b[d].get("ws") for d in ("3", "4")}, "guard": guard}


def score(runs=R):
    c = jl(os.path.join(runs, CD, "eval_transfer.json"))
    o = {"section": 249, "registered": "FT_v1.md", "models": {m: one(jl(os.path.join(runs, m, "eval_transfer.json")), c) for m in MODELS}}
    labs = [v["label"] for v in o["models"].values()]
    o["verdict"] = "FRAME-TRANSFER %s (%s)" % ("pending" if "pending" in labs else labs[0] if len(set(labs)) == 1 else "MIXED", " / ".join(labs)); return o


def fixtures():
    res = []

    def world(acc=0.95, ws=0.95, cd=0.1, co=0.1, per=None):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, CD)); json.dump({"by_d": {str(x): {"acc": cd} for x in range(1, 7)}}, open(os.path.join(d, CD, "eval_transfer.json"), "w"))
        for m in MODELS:
            g = dict(acc=acc, ws=ws, co=co); g.update((per or {}).get(m, {}))
            os.makedirs(os.path.join(d, m))
            json.dump({"by_d": {str(x): {"acc": g["acc"], "ws": {str(t): g["ws"] for t in range(1, 9)}} for x in range(1, 7)},
                       "cue_only": {str(x): {"acc": g["co"], "majority": 0.1} for x in range(2, 5)}}, open(os.path.join(d, m, "eval_transfer.json"), "w"))
        return d
    o = score(world()); res.append(("transfers -> TRANSFER", o["verdict"].startswith("FRAME-TRANSFER TRANSFER"), o["verdict"]))
    o = score(world(acc=0.5)); res.append(("low acc -> NO-TRANSFER", "NO-TRANSFER (" in o["verdict"], o["verdict"]))
    o = score(world(ws=0.3)); res.append(("acc but no chain -> ACC-ONLY", "ACC-ONLY (" in o["verdict"], o["verdict"]))
    o = score(world(co=0.5)); res.append(("cue alone -> VOID", "VOID (" in o["verdict"], o["verdict"]))
    o = score(world(per={"lh_fh_s2": {"acc": 0.5}})); res.append(("disagree -> MIXED", "MIXED" in o["verdict"], o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
