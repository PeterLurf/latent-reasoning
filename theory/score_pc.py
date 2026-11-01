"""THEORY §254 P2P-CHAIN scorer (Note 253-4 / §254; Theory is the only scorer). One cell p2pc_s1 = aw_plaw2_s1's args + init_trunk s252_askperm_s1 + perm_p 0 (12k, d <= 2).
Readings: log.jsonl final aux_vacc_by_d_pass['2']['2']; eval_ws.json by_d['2']['2'] (pass-2 ':' readout, TEST2 d 2); eval_acc.json by_d['2'].acc.
Labels: PERM-BLOCKER (FORMED: aux >= 0.90 and ws >= 0.90 and TEST2 d 2 >= 0.90) / INIT-BLOCKER (NOT-FORMED: aux < 0.50 and ws < 0.50) / INCONCLUSIVE (otherwise, incl. measures disagreeing) / pending.
    python -B theory/score_pc.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq"); RUN = "p2pc_s1"
OUT = os.path.join(_HR, "theory", "results", "score_pc.json")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def score(runs=R):
    p = os.path.join(runs, RUN); a = jl(os.path.join(p, "eval_acc.json")); w = jl(os.path.join(p, "eval_ws.json")); aux = None
    try:
        for ln in open(os.path.join(p, "log.jsonl")):
            x = json.loads(ln)
            if "aux_vacc_by_d_pass" in x and "2" in x["aux_vacc_by_d_pass"]:
                aux = x["aux_vacc_by_d_pass"]["2"]["2"]
    except OSError:
        pass
    if a is None or w is None or aux is None:
        return {"verdict": "pending"}
    ws = (w.get("by_d", w))["2"]["2"]; acc = a["by_d"]["2"]["acc"]
    lab = ("PERM-BLOCKER" if (aux >= 0.90 and ws >= 0.90 and acc >= 0.90) else "INIT-BLOCKER" if (aux < 0.50 and ws < 0.50) else "INCONCLUSIVE")
    return {"verdict": lab, "aux_d2_p2": aux, "ws_d2_p2": ws, "test2_d2": acc}


def fixtures():
    res = []

    def world(aux, ws, acc):
        d = tempfile.mkdtemp(); p = os.path.join(d, RUN); os.makedirs(p)
        json.dump({"by_d": {"2": {"acc": acc}}}, open(os.path.join(p, "eval_acc.json"), "w"))
        json.dump({"by_d": {"2": {"1": 1.0, "2": ws}}}, open(os.path.join(p, "eval_ws.json"), "w"))
        with open(os.path.join(p, "log.jsonl"), "w") as fh:
            fh.write(json.dumps({"step": 6000, "aux_vacc_by_d_pass": {"2": {"1": 1.0, "2": 0.2}}}) + "\n")
            fh.write(json.dumps({"step": 12000, "aux_vacc_by_d_pass": {"2": {"1": 1.0, "2": aux}}}) + "\n")
        return d
    for name, args, want in (("formed", (0.99, 0.99, 1.0), "PERM-BLOCKER"), ("not formed", (0.15, 0.02, 1.0), "INIT-BLOCKER"),
                             ("middle", (0.7, 0.7, 1.0), "INCONCLUSIVE"), ("disagree", (0.95, 0.3, 1.0), "INCONCLUSIVE"), ("final record used", (0.95, 0.95, 0.95), "PERM-BLOCKER")):
        o = score(world(*args)); res.append((name, o["verdict"] == want, o["verdict"]))
    o = score(tempfile.mkdtemp()); res.append(("missing -> pending", o["verdict"] == "pending", o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1)
