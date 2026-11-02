"""THEORY §260 GENERIC-BUN full-stage scorer (Notes 260-1 / 260-2; Theory is the only scorer; committed before the full launch).
Runs gb_s1 / gb_s2 = bun_s1 / bun_s2 + --cue generic, 8k (the same build as the pilot; all evals under the generic cue).
Binding per seed:
  C1 plan-free: eval_acc d 1-4 >= 0.95; NOT-DECODER by_d['4'].acc_kprime['2'] <= 0.20; SWAP4 (eval_swap_kd by_d['4'].swap_mid['4'] follow_U >= 0.50 and >= floor + 0.30);
     LP5-D4 (eval_lp5h d 4 intact['4'] - mean_at_t['4'] >= 0.30); PLAN-FREE assert passed on the host (gates_pass.json key 'plan_free' true).
  CONTENT: eval_insert_ho fam['H-CHAINSRC-OP'].acc8 >= 0.85.
  C2: eval_acc by_d['6'].acc >= 0.80.
  C6: eval_long_k128 d 1-4: acc['64'] >= acc['8'] - 0.02.
  STABLE (score_ao). (CUE-ONLY is vacuous under a generic cue (REVIEW_NS #3) and is not used.)
Label: PLAN-FREE-C1 if all pass, else 'FAIL: <criteria>'. Per arm: PLAN-FREE-C1 k/2.
    python -B theory/score_gbf.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory"))
import score_ao as AO  # noqa: E402

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_gbf.json")
RUNS = ("gb_s1", "gb_s2")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def seed(runs, run):
    f = {x: jl(os.path.join(runs, run, x + ".json")) for x in ("eval_acc", "eval_swap_kd", "eval_lp5h", "eval_insert_ho", "eval_long_k128", "gates_pass")}
    if any(v is None for v in f.values()):
        return {"label": "pending"}
    a = f["eval_acc"]["by_d"]; sw = f["eval_swap_kd"]["by_d"]["4"]["swap_mid"]["4"]; h = f["eval_lp5h"]["by_d"]["4"]; L = f["eval_long_k128"]["by_d"]
    c1 = (all(a[d]["acc"] >= 0.95 for d in ("1", "2", "3", "4")) and a["4"]["acc_kprime"]["2"] <= 0.20 and sw["follow_U"] >= 0.50 and sw["follow_U"] >= sw["floor"] + 0.30
          and h["intact"]["4"] - h["mean_at_t"]["4"] >= 0.30 and f["gates_pass"].get("plan_free") is True)
    content = f["eval_insert_ho"]["fam"]["H-CHAINSRC-OP"]["acc8"] >= 0.85
    c2 = a["6"]["acc"] >= 0.80
    c6 = all(L[d]["acc"]["64"] >= L[d]["acc"]["8"] - 0.02 for d in ("1", "2", "3", "4"))
    sok, smsg = AO.stable(runs, run)
    crit = {"C1": c1, "CONTENT": content, "C2": c2, "C6": c6, "STABLE": sok is not False}
    fail = [k for k, v in crit.items() if not v]
    return {"label": "PLAN-FREE-C1" if not fail else "FAIL: " + ",".join(fail), "crit": crit, "d4_k2": a["4"]["acc_kprime"]["2"], "d6": a["6"]["acc"],
            "chainsrc_op": f["eval_insert_ho"]["fam"]["H-CHAINSRC-OP"]["acc8"], "swap4": sw, "stable": (sok, smsg)}


def score(runs=R):
    o = {"section": 260, "seeds": {r: seed(runs, r) for r in RUNS}}
    labs = [v["label"] for v in o["seeds"].values()]
    o["verdict"] = "pending" if "pending" in labs else "PLAN-FREE-C1 %d/2 (%s)" % (labs.count("PLAN-FREE-C1"), " / ".join(labs)); return o


def fixtures():
    res = []

    def world(t2=1.0, k2=0.03, sw=0.9, co=0.95, d6=0.9, hold=1.0, pf=True):
        d = tempfile.mkdtemp()
        for r in RUNS:
            p = os.path.join(d, r); os.makedirs(p)
            json.dump({"by_d": {**{x: {"acc": t2, "acc_kprime": {"2": k2 if x == "4" else 1.0}} for x in ("1", "2", "3", "4")}, "6": {"acc": d6}}}, open(os.path.join(p, "eval_acc.json"), "w"))
            json.dump({"by_d": {"4": {"swap_mid": {"4": {"follow_U": sw, "floor": 0.1}}}}}, open(os.path.join(p, "eval_swap_kd.json"), "w"))
            json.dump({"by_d": {"4": {"intact": {"4": 1.0}, "mean_at_t": {"4": 0.1}}}}, open(os.path.join(p, "eval_lp5h.json"), "w"))
            json.dump({"fam": {"H-CHAINSRC-OP": {"acc8": co, "by_pos": {"1": co}}}}, open(os.path.join(p, "eval_insert_ho.json"), "w"))
            json.dump({"by_d": {x: {"acc": {"8": 1.0, "64": hold, "128": hold}} for x in ("1", "2", "3", "4")}}, open(os.path.join(p, "eval_long_k128.json"), "w"))
            json.dump({"plan_free": pf}, open(os.path.join(p, "gates_pass.json"), "w"))
            json.dump({"n_nonfinite": 0}, open(os.path.join(p, "train_summary.json"), "w"))
            with open(os.path.join(p, "log.jsonl"), "w") as fh:
                for i in range(1, 81):
                    fh.write(json.dumps({"step": i * 100, "loss": 0.01}) + chr(10))
        return d
    for name, kw, want in (("all", {}, "PLAN-FREE-C1 2/2"), ("decoder", {"k2": 0.6}, "FAIL: C1"), ("swap", {"sw": 0.2}, "FAIL: C1"), ("assert", {"pf": False}, "FAIL: C1"),
                           ("content", {"co": 0.6}, "FAIL: CONTENT"), ("c2", {"d6": 0.5}, "FAIL: C2"), ("c6", {"hold": 0.5}, "FAIL: C6")):
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
