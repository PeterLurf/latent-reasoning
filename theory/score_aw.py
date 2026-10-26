"""THEORY §234 ANSWER-WRITE scorer, P-2 pilot (registered text docs/research_answerpos_2026-10-25/AW_v1.md at b51151b1 + Notes 234-1 / 234-2; field paths
per Experiments' inject_aw_eval layout (0a6f6c17); Theory is the only scorer).
Runs: LAT-AW aw_plaw2_s1, CD-AN aw_pcd2_s1 (same RTX 3090), plain DIRECT s228_pdir2c_s1 (CUE-ONLY bar only).
P-2 gates = score_st's (LP0, LP1, LP2 on d = 2, LP3 vs CD-AN, LP5, MEAN-STATE, CUE-ONLY guard, CUE-WRONG label), with the run names above, plus
  STATE-USE (binding): eval_swap_k follow_U >= 0.50 and >= floor + 0.30 (n_pairs >= 1,000).
P-2 PASS = score_st's PASS conditions & STATE-USE & not VOID. Labels on a pass: C1-TOLD / plain C1 (CUE-WRONG), plus DEC-STEP2 (d = 2 cannot show chaining; P-4 must).
Failure diagnosis (AW_v1 sec. 3, descriptive): end-of-run eval_probe u1|v1 < 0.80 -> AW-SOURCE-DRIFT; u1|v1 >= 0.80 and read1|v1 < 0.80 -> AW-NOT-STORED;
  LP2 pass but LP3 or LP5 fail -> AW-WEAK-READ. WS-READOUT (eval_ws by_d) reported.
    python -B theory/score_aw.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory"))
import score_st as ST  # noqa: E402

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_aw.json")
LAT, CD, DIR = "aw_plaw2_s1", "aw_pcd2_s1", "s228_pdir2c_s1"


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def score(runs=R):
    old = (ST.LAT, ST.CD, ST.DIR)
    ST.LAT, ST.CD, ST.DIR = LAT, CD, DIR
    try:
        o = ST.score(runs)
    finally:
        ST.LAT, ST.CD, ST.DIR = old
    o.update({"section": 234, "registered": "AW_v1.md @ b51151b1 + 234-1 / 234-2"})
    sk, pr, ws = jl(os.path.join(runs, LAT, "eval_swap_k.json")), jl(os.path.join(runs, LAT, "eval_probe.json")), jl(os.path.join(runs, LAT, "eval_ws.json"))
    if o["verdict"].startswith("pending") or sk is None or pr is None:
        o["verdict"] = o["verdict"] if o["verdict"].startswith("pending") else "pending (missing %s)" % [n for n, v in (("swap_k", sk), ("probe", pr)) if v is None]
        return o
    su = sk.get("n_pairs", 0) >= 1000 and sk["follow_U"] >= 0.50 and sk["follow_U"] >= sk["floor"] + 0.30
    u1 = pr["probes"]["u1|v1"]["test_acc"]; rd = pr["probes"]["read1|v1"]["test_acc"]
    diag = []
    if u1 < 0.80:
        diag.append("AW-SOURCE-DRIFT")
    elif rd < 0.80:
        diag.append("AW-NOT-STORED")
    if o.get("LP2") and not (o.get("LP3") and o.get("LP5")):
        diag.append("AW-WEAK-READ")
    o.update({"STATE_USE": su, "swap_k": {k: sk.get(k) for k in ("n_pairs", "follow_U", "floor", "follow_dfin", "follow_ca", "stay_own")},
              "probe_end": {"u1|v1": u1, "read1|v1": rd, "l0|v1": pr["probes"]["l0|v1"]["test_acc"]}, "diagnosis": diag,
              "ws_readout": (ws or {}).get("by_d")})
    if o["verdict"].startswith("VOID"):
        return o
    if o["verdict"].startswith("P-2 PASS"):
        if su:
            o["verdict"] = o["verdict"].replace(")", "; DEC-STEP2: chaining untested at d = 2)")
        else:
            o["verdict"] = "P-2 FAIL (STATE-USE: follow_U %.3f, floor %.3f)" % (sk["follow_U"], sk["floor"])
    elif not su:
        o["verdict"] = o["verdict"].replace("P-2 FAIL (", "P-2 FAIL (STATE-USE,")
    if diag and "FAIL" in o["verdict"]:
        o["verdict"] += " | " + ",".join(diag)
    return o


def fixtures():
    res = []
    def world(**kw):
        sk = dict(n_pairs=1000, follow_U=0.8, floor=0.2); sk.update({k[3:]: v for k, v in kw.items() if k.startswith("sk_")})
        pr = dict(u1=1.0, rd=1.0); pr.update({k[3:]: v for k, v in kw.items() if k.startswith("pr_")})
        st = {k: v for k, v in kw.items() if not k.startswith(("sk_", "pr_"))}
        d = ST._world(**st)
        for a, b in ((ST.LAT, LAT), (ST.CD, CD)):
            os.replace(os.path.join(d, a), os.path.join(d, b))
        json.dump(sk, open(os.path.join(d, LAT, "eval_swap_k.json"), "w"))
        json.dump({"probes": {"u1|v1": {"test_acc": pr["u1"]}, "read1|v1": {"test_acc": pr["rd"]}, "l0|v1": {"test_acc": 0.09}}}, open(os.path.join(d, LAT, "eval_probe.json"), "w"))
        json.dump({"by_d": {"2": {"1": 0.9, "2": 0.9}}}, open(os.path.join(d, LAT, "eval_ws.json"), "w"))
        return d
    o = score(world()); res.append(("serial stub -> P-2 PASS (C1-TOLD; DEC-STEP2)", o["verdict"].startswith("P-2 PASS (C1-TOLD") and "DEC-STEP2" in o["verdict"], o["verdict"]))
    o = score(world(sk_follow_U=0.3)); res.append(("swap not followed -> FAIL STATE-USE", "STATE-USE" in o["verdict"], o["verdict"]))
    o = score(world(sk_follow_U=0.55, sk_floor=0.4)); res.append(("follow 0.55 over floor 0.40 -> FAIL STATE-USE", "STATE-USE" in o["verdict"], o["verdict"]))
    o = score(world(lat=0.1, lp2=0.3, pr_u1=0.5)); res.append(("source drifted -> FAIL + AW-SOURCE-DRIFT", "AW-SOURCE-DRIFT" in o["verdict"], o["verdict"]))
    o = score(world(lat=0.1, lp2=0.3, pr_rd=0.3)); res.append(("not stored -> AW-NOT-STORED", "AW-NOT-STORED" in o["verdict"], o["verdict"]))
    o = score(world(lat=0.1, mean1=0.8)); res.append(("LP2 ok, LP3 / LP5 fail -> AW-WEAK-READ", "AW-WEAK-READ" in o["verdict"], o["verdict"]))
    o = score(world(co=0.5)); res.append(("cue alone answers -> VOID", o["verdict"].startswith("VOID"), o["verdict"]))
    d = world(); os.remove(os.path.join(d, LAT, "eval_swap_k.json")); o = score(d); res.append(("missing swap_k -> pending", o["verdict"].startswith("pending"), o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", newline=chr(10)), indent=1, default=str)
