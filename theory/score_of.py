"""THEORY §263 ORDER-FREE scorer (OF_v1 in THEORY §263; Theory is the only scorer; committed before output).
Runs of_s1 / of_s2 = bun_s1 / bun_s2 + perm-then-insert (perm_p 0.5) + 25% one-hop ':'-CE co-training (the §262 CTRL recipe, no NK), 8k.
Binding per seed:
  ORDER-FREE: eval_test_perm by_d[d].acc_kprime['8'] >= 0.90 for d 2-4 (full permutation);
              eval_late_premise by_d[d].acc >= 0.85 for d 2-4 (S_1's sentence placed last among the op sentences, others in chain order; Audit L1a).
  RETAIN: eval_acc d 1-4 >= 0.98; NOT-DECODER by_d['4'].acc_kprime['2'] <= 0.20; SWAP4; LP5-D4.
  C2: eval_acc d 6 >= bun d 6 - 0.05 and d 7 >= bun d 7 - 0.05 (same seed; Audit L2).
  C6: eval_long_k128 d 1-6: acc['64'], acc['128'] >= acc['8'] - 0.02.
  STABLE (score_ao).
Descriptive: eval_name_swap (cue-name swap follow rate vs floor; Audit L1b), eval_shuffle_graded (independent-only vs full shuffle), insert families, C5 r = 3.
Label ORDER-FREE-C1 if all pass, else 'FAIL: <criteria>'. Per arm ORDER-FREE-C1 k/2.
    python -B theory/score_of.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory"))
import score_ao as AO  # noqa: E402

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_of.json")
RUNS = {"of_s1": "bun_s1", "of_s2": "bun_s2"}


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def seed(runs, run, ref):
    f = {x: jl(os.path.join(runs, run, x + ".json")) for x in ("eval_acc", "eval_test_perm", "eval_late_premise", "eval_swap_kd", "eval_lp5h", "eval_long_k128")}
    rf = jl(os.path.join(runs, ref, "eval_acc.json"))
    if any(v is None for v in list(f.values()) + [rf]):
        return {"label": "pending"}
    a = f["eval_acc"]["by_d"]; tp = f["eval_test_perm"]["by_d"]; lp = f["eval_late_premise"]["by_d"]
    sw = f["eval_swap_kd"]["by_d"]["4"]["swap_mid"]["4"]; h = f["eval_lp5h"]["by_d"]["4"]; L = f["eval_long_k128"]["by_d"]; r = rf["by_d"]
    of = all(tp[d]["acc_kprime"]["8"] >= 0.90 for d in ("2", "3", "4")) and all(lp[d]["acc"] >= 0.85 for d in ("2", "3", "4"))
    retain = (all(a[d]["acc"] >= 0.98 for d in ("1", "2", "3", "4")) and a["4"]["acc_kprime"]["2"] <= 0.20 and sw["follow_U"] >= 0.50 and sw["follow_U"] >= sw["floor"] + 0.30
              and h["intact"]["4"] - h["mean_at_t"]["4"] >= 0.30)
    c2 = a["6"]["acc"] >= r["6"]["acc"] - 0.05 and a["7"]["acc"] >= r["7"]["acc"] - 0.05
    c6 = all(L[d]["acc"][k] >= L[d]["acc"]["8"] - 0.02 for d in ("1", "2", "3", "4", "5", "6") for k in ("64", "128"))
    sok, smsg = AO.stable(runs, run)
    crit = {"ORDER-FREE": of, "RETAIN": retain, "C2": c2, "C6": c6, "STABLE": sok is not False}
    fail = [k for k, v in crit.items() if not v]
    ns = jl(os.path.join(runs, run, "eval_name_swap.json")); gr = jl(os.path.join(runs, run, "eval_shuffle_graded.json"))
    return {"label": "ORDER-FREE-C1" if not fail else "FAIL: " + ",".join(fail), "crit": crit,
            "test_perm": {d: tp[d]["acc_kprime"]["8"] for d in ("2", "3", "4")}, "late_premise": {d: lp[d]["acc"] for d in ("2", "3", "4")},
            "d6": a["6"]["acc"], "d7": a["7"]["acc"], "bun_d6": r["6"]["acc"], "bun_d7": r["7"]["acc"], "name_swap_desc": ns, "graded_desc": gr, "stable": (sok, smsg)}


def score(runs=R):
    o = {"section": 263, "seeds": {r: seed(runs, r, ref) for r, ref in RUNS.items()}}
    labs = [v["label"] for v in o["seeds"].values()]
    o["verdict"] = "pending" if "pending" in labs else "ORDER-FREE-C1 %d/2 (%s)" % (labs.count("ORDER-FREE-C1"), " / ".join(labs)); return o


def fixtures():
    res = []

    def world(tp=0.95, lp=0.9, t2=1.0, k2=0.03, sw=0.9, d6=0.95, d7=0.85, hold=1.0):
        d = tempfile.mkdtemp()
        for run, ref in RUNS.items():
            os.makedirs(os.path.join(d, ref)); json.dump({"by_d": {"6": {"acc": 0.97}, "7": {"acc": 0.86}}}, open(os.path.join(d, ref, "eval_acc.json"), "w"))
            p = os.path.join(d, run); os.makedirs(p)
            json.dump({"by_d": {**{x: {"acc": t2, "acc_kprime": {"2": k2 if x == "4" else 1.0}} for x in ("1", "2", "3", "4")}, "6": {"acc": d6}, "7": {"acc": d7}}}, open(os.path.join(p, "eval_acc.json"), "w"))
            json.dump({"by_d": {x: {"acc_kprime": {"8": tp}} for x in ("2", "3", "4")}}, open(os.path.join(p, "eval_test_perm.json"), "w"))
            json.dump({"by_d": {x: {"acc": lp} for x in ("2", "3", "4")}}, open(os.path.join(p, "eval_late_premise.json"), "w"))
            json.dump({"by_d": {"4": {"swap_mid": {"4": {"follow_U": sw, "floor": 0.1}}}}}, open(os.path.join(p, "eval_swap_kd.json"), "w"))
            json.dump({"by_d": {"4": {"intact": {"4": 1.0}, "mean_at_t": {"4": 0.1}}}}, open(os.path.join(p, "eval_lp5h.json"), "w"))
            json.dump({"by_d": {str(x): {"acc": {"8": 1.0, "64": 1.0, "128": hold}} for x in range(1, 7)}}, open(os.path.join(p, "eval_long_k128.json"), "w"))
            json.dump({"n_nonfinite": 0}, open(os.path.join(p, "train_summary.json"), "w"))
            with open(os.path.join(p, "log.jsonl"), "w") as fh:
                for i in range(1, 81):
                    fh.write(json.dumps({"step": i * 100, "loss": 0.01}) + chr(10))
        return d
    for name, kw, want in (("all", {}, "ORDER-FREE-C1 2/2"), ("perm fails", {"tp": 0.6}, "FAIL: ORDER-FREE"), ("late premise fails", {"lp": 0.6}, "FAIL: ORDER-FREE"),
                           ("decoder", {"k2": 0.6}, "FAIL: RETAIN"), ("swap", {"sw": 0.2}, "FAIL: RETAIN"), ("c2 d6", {"d6": 0.85}, "FAIL: C2"), ("c2 d7", {"d7": 0.7}, "FAIL: C2"),
                           ("c6", {"hold": 0.5}, "FAIL: C6")):
        o = score(world(**kw)); res.append((name, want in o["verdict"], o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
