"""THEORY §268 LINEAGE-SEED scorer (X7; Theory is the only scorer; committed before output).
A fresh, independent seed (3) of bun's whole lineage, every stage with the s1 stage's args (args_ref) except seed / init paths, RTX 3090:
  s233_askname_s3 (8k) -> aw_plaw2_s3 (P-2, 12k) -> rcb_plaw4_s3 (16k) -> dis_s3 (16k) -> dch_s3 (8k) -> bun_s3 (8k).
Stage gates (stop on the first failure; label BREAKS-AT-<stage>):
  askname   eval_acc by_d['1'].acc >= 0.98
  aw_plaw2  eval_acc d 2 >= 0.95 AND eval_ws by_d['2']['2'] (pass-2 ':' readout == v_2) >= 0.90 (gate on the mechanism)
  rcb_plaw4 eval_acc d 1-4 >= 0.95
  dis       eval_acc d 1-4 >= 0.95
  dch       eval_acc d 1-4 >= 0.98
Final binding on bun_s3 (vs bun_s1 / bun_s2, the §259 bars):
  RETAIN    eval_acc d 1-4 >= 0.98; NOT-DECODER by_d['4'].acc_kprime['2'] <= 0.20; SWAP4 (swap_kd d 4 swap_mid t 4: follow_U >= 0.50 and >= floor + 0.30);
            LP5-D4 (lp5h d 4: intact - mean_at_t >= 0.30)
  C2        d 6 >= mean(bun d 6) - 0.05 and d 7 >= mean(bun d 7) - 0.05
  C6        eval_long_k128 d 1-6: acc[64], acc[128] >= acc[8] - 0.02
  CONTENT   eval_insert_ho H-CHAINSRC-OP acc8 >= 0.85 and min(by_pos) >= 0.80
Label: LINEAGE-REPLICATES if all pass; LINEAGE-PARTIAL (RETAIN passes, others fail: listed); BREAKS-AT-<stage>. Wilson 95% CIs + NEAR on d 6 / d 7.
    python -B theory/score_ls.py [--fixtures]
"""
import argparse, json, math, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_ls.json")
STAGES = ("s233_askname_s3", "aw_plaw2_s3", "rcb_plaw4_s3", "dis_s3", "dch_s3")
FINAL, REF = "bun_s3", ("bun_s1", "bun_s2")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n); return (c - h, c + h)


def near(p, n, bar):
    return abs(p - bar) <= 2 * math.sqrt(max(p * (1 - p), 1e-12) / n)


def stage_ok(runs, st):
    a = jl(os.path.join(runs, st, "eval_acc.json"))
    if a is None:
        return None
    A = a["by_d"]
    if st.startswith("s233_askname"):
        return A["1"]["acc"] >= 0.98
    if st.startswith("aw_plaw2"):
        w = jl(os.path.join(runs, st, "eval_ws.json"))
        if w is None:
            return None
        return A["2"]["acc"] >= 0.95 and w["by_d"]["2"]["2"] >= 0.90
    thr = 0.98 if st.startswith("dch") else 0.95
    return all(A[d]["acc"] >= thr for d in ("1", "2", "3", "4"))


def score(runs=R):
    o = {"section": 268, "stages": {}}
    for st in STAGES:
        ok = stage_ok(runs, st); o["stages"][st] = ok
        if ok is None:
            o["verdict"] = "pending (%s)" % st; return o
        if not ok:
            o["verdict"] = "BREAKS-AT-%s" % st.split("_s3")[0]; return o
    f = {x: jl(os.path.join(runs, FINAL, x + ".json")) for x in ("eval_acc", "eval_swap_kd", "eval_lp5h", "eval_long_k128", "eval_insert_ho")}
    rf = [jl(os.path.join(runs, r, "eval_acc.json")) for r in REF]
    if any(v is None for v in list(f.values()) + rf):
        o["verdict"] = "pending (bun_s3)"; return o
    a = f["eval_acc"]["by_d"]; m6 = sum(r["by_d"]["6"]["acc"] for r in rf) / 2; m7 = sum(r["by_d"]["7"]["acc"] for r in rf) / 2
    sw = f["eval_swap_kd"]["by_d"]["4"]["swap_mid"]["4"]; h = f["eval_lp5h"]["by_d"]["4"]; L = f["eval_long_k128"]["by_d"]; co = f["eval_insert_ho"]["fam"]["H-CHAINSRC-OP"]
    crit = {"RETAIN": (all(a[d]["acc"] >= 0.98 for d in ("1", "2", "3", "4")) and a["4"]["acc_kprime"]["2"] <= 0.20 and sw["follow_U"] >= 0.50
                       and sw["follow_U"] >= sw["floor"] + 0.30 and h["intact"]["4"] - h["mean_at_t"]["4"] >= 0.30),
            "C2": a["6"]["acc"] >= m6 - 0.05 and a["7"]["acc"] >= m7 - 0.05,
            "C6": all(L[d]["acc"][k] >= L[d]["acc"]["8"] - 0.02 for d in ("1", "2", "3", "4", "5", "6") for k in ("64", "128")),
            "CONTENT": co["acc8"] >= 0.85 and min(co["by_pos"].values()) >= 0.80}
    fail = [k for k, v in crit.items() if not v]
    o.update({"crit": crit, "d6": a["6"]["acc"], "d7": a["7"]["acc"], "bar6": m6 - 0.05, "bar7": m7 - 0.05,
              "ci": {"d6": wilson(a["6"]["acc"], 1000), "d7": wilson(a["7"]["acc"], 1000)},
              "near": {"d6": near(a["6"]["acc"], 1000, m6 - 0.05), "d7": near(a["7"]["acc"], 1000, m7 - 0.05)}})
    o["verdict"] = "LINEAGE-REPLICATES" if not fail else ("LINEAGE-PARTIAL (FAIL: %s)" % ",".join(fail) if crit["RETAIN"] else "LINEAGE-PARTIAL (FAIL: %s)" % ",".join(fail))
    return o


def fixtures():
    res = []

    def world(ask=1.0, ws2=0.98, rcb=0.99, dch=0.99, d6=0.95, d7=0.84, k2=0.03, hold=1.0, co=0.95, missing=None):
        dd = tempfile.mkdtemp()

        def put(run, name, obj):
            os.makedirs(os.path.join(dd, run), exist_ok=True); json.dump(obj, open(os.path.join(dd, run, name), "w"))
        for st in STAGES:
            if st == missing:
                continue
            v = {"s233_askname_s3": ask, "rcb_plaw4_s3": rcb, "dch_s3": dch}.get(st, 0.99)
            put(st, "eval_acc.json", {"by_d": {str(d): {"acc": v} for d in range(1, 5)}})
            if st == "aw_plaw2_s3":
                put(st, "eval_ws.json", {"by_d": {"2": {"1": 1.0, "2": ws2}}})
        for r in REF:
            put(r, "eval_acc.json", {"by_d": {"6": {"acc": 0.964}, "7": {"acc": 0.855}}})
        put(FINAL, "eval_acc.json", {"by_d": {**{str(x): {"acc": 1.0, "acc_kprime": {"2": k2}} for x in range(1, 5)}, "6": {"acc": d6}, "7": {"acc": d7}}})
        put(FINAL, "eval_swap_kd.json", {"by_d": {"4": {"swap_mid": {"4": {"follow_U": 0.9, "floor": 0.1}}}}})
        put(FINAL, "eval_lp5h.json", {"by_d": {"4": {"intact": {"4": 1.0}, "mean_at_t": {"4": 0.1}}}})
        put(FINAL, "eval_long_k128.json", {"by_d": {str(x): {"acc": {"8": 1.0, "64": 1.0, "128": hold}} for x in range(1, 7)}})
        put(FINAL, "eval_insert_ho.json", {"fam": {"H-CHAINSRC-OP": {"acc8": co, "by_pos": {"1": 0.9}}}})
        return dd
    for name, kw, want in (("replicates", {}, "LINEAGE-REPLICATES"), ("askname breaks", {"ask": 0.9}, "BREAKS-AT-s233_askname"),
                           ("P-2 write not formed", {"ws2": 0.3}, "BREAKS-AT-aw_plaw2"), ("rcb breaks", {"rcb": 0.9}, "BREAKS-AT-rcb_plaw4"),
                           ("dch breaks", {"dch": 0.97}, "BREAKS-AT-dch"), ("c2 fails", {"d6": 0.85}, "LINEAGE-PARTIAL (FAIL: C2)"),
                           ("decoder", {"k2": 0.5}, "FAIL: RETAIN"), ("c6", {"hold": 0.5}, "FAIL: C6"), ("content", {"co": 0.6}, "FAIL: CONTENT"),
                           ("pending stage", {"missing": "dis_s3"}, "pending (dis_s3)")):
        o = score(world(**kw)); res.append((name, want in o["verdict"], o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], "->", r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
