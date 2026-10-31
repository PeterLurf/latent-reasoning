"""THEORY §253 LOOKUP-REBUILD scorer (docs/research_lookuprebuild_2026-10-31/LB_v1.md; Theory is the only scorer).
Step-0 gate: lb_step0/eval_test_perm.json by_d['2'].ws['1'] >= 0.95 (the askperm init's pass 1 on full TEST-PERM d 2, wrap included).
Stage 1 (P2P, d <= 2): p2p_s1 / p2p_s2 (askperm init) + p2p_old_s1 (askname init, attribution, descriptive). Per seed:
  NOT-FORMED if eval_acc by_d['2'].acc < 0.90; POSITION-REEMERGED if eval_test_perm by_d['2'].acc_kprime['8'] < 0.80 or ws['1'] < 0.80; else SEED-OK.
Stage 2 (LBR, REG d <= 4): lbr_s1 / lbr_s2. Binding per seed:
  TEST-PERM: by_d[d].acc_kprime['8'] >= 0.90 (d 2-4); NOT-DECODER by_d['4'].acc_kprime['2'] <= 0.20; LP5P mean_at_2_d4.intact['8'] - mean_at_t['8'] >= 0.30;
  RETAIN (TEST2): eval_acc d 1-4 >= 0.98; SWAP4 eval_swap_kd by_d['4'].swap_mid['4'] follow_U >= 0.50 and >= floor + 0.30; LP5-D4 eval_lp5h d 4 intact['4'] - mean_at_t['4'] >= 0.30;
  guards: CUE-ONLY d 3-4 (score_ae rule); STABLE (score_ao rule).
Labels (first match): VOID / UNSTABLE / NOT-FORMED / DECODER / LOOKUP-C1 / LOOKUP-NOCAUSAL / PARTIAL / ORDER-ONLY. Per arm: PASS (both LOOKUP-C1) / SPLIT / FAIL.
    python -B theory/score_lb.py [--stage step0|1|2] [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory"))
import score_ao as AO  # noqa: E402

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_lb_%s.json")
S1, S1_OLD, S2 = ("p2p_s1", "p2p_s2"), "p2p_old_s1", ("lbr_s1", "lbr_s2")
DIRS = ("s228_pdir2c_s1_reg", "s228_pdir2c_s1_reg_bs50")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def step0(runs=R):
    tp = jl(os.path.join(runs, "lb_step0", "eval_test_perm.json"))
    if tp is None:
        return {"verdict": "pending"}
    v = tp["by_d"]["2"]["ws"]["1"]
    return {"ws_d2_p1": v, "verdict": "STEP0-OK" if v >= 0.95 else "STEP0-FAIL"}


def stage1_seed(runs, run):
    a = jl(os.path.join(runs, run, "eval_acc.json")); tp = jl(os.path.join(runs, run, "eval_test_perm.json"))
    if a is None or tp is None:
        return {"label": "pending"}
    acc2 = a["by_d"]["2"]["acc"]; p8 = tp["by_d"]["2"]["acc_kprime"]["8"]; w1 = tp["by_d"]["2"]["ws"]["1"]
    lab = "NOT-FORMED" if acc2 < 0.90 else "POSITION-REEMERGED" if (p8 < 0.80 or w1 < 0.80) else "SEED-OK"
    return {"label": lab, "test2_d2": acc2, "perm_d2_k8": p8, "perm_ws_d2_p1": w1}


def stage1(runs=R):
    o = {"seeds": {r: stage1_seed(runs, r) for r in S1}, "attribution_old": stage1_seed(runs, S1_OLD)}
    labs = [v["label"] for v in o["seeds"].values()]
    o["verdict"] = ("pending" if "pending" in labs else "NO-SEED" if "SEED-OK" not in labs else "SEED %d/2" % labs.count("SEED-OK")) + " | " + " / ".join(labs)
    return o


def stage2_seed(runs, run):
    f = {x: jl(os.path.join(runs, run, x + ".json")) for x in ("eval_acc", "eval_test_perm", "eval_swap_kd", "eval_lp5h", "eval_cue_only")}
    dirs = [jl(os.path.join(runs, d, "eval_acc.json")) for d in DIRS]
    if any(v is None for v in list(f.values()) + dirs):
        return {"label": "pending"}
    a, tp = f["eval_acc"]["by_d"], f["eval_test_perm"]
    co = f["eval_cue_only"]["by_d"]
    guard = all(co[d]["answer_acc"] <= max(co[d]["majority"], min(x["by_d"][d]["acc"] for x in dirs)) + 0.05 for d in ("3", "4"))
    sok, smsg = AO.stable(runs, run)
    perm = {d: tp["by_d"][d]["acc_kprime"]["8"] for d in ("2", "3", "4")}
    nodec = tp["by_d"]["4"]["acc_kprime"]["2"] <= 0.20
    m2 = tp["mean_at_2_d4"]; lp5p = m2["intact"]["8"] - m2["mean_at_t"]["8"]
    sw = f["eval_swap_kd"]["by_d"]["4"]["swap_mid"]["4"]; swap4 = sw["follow_U"] >= 0.50 and sw["follow_U"] >= sw["floor"] + 0.30
    h = f["eval_lp5h"]["by_d"]["4"]; lp5 = h["intact"]["4"] - h["mean_at_t"]["4"]
    retain = all(a[d]["acc"] >= 0.98 for d in ("1", "2", "3", "4")) and swap4 and lp5 >= 0.30
    permok = all(v >= 0.90 for v in perm.values()) and lp5p >= 0.30
    lab = ("VOID" if not guard else "UNSTABLE" if sok is False else "NOT-FORMED" if a["2"]["acc"] < 0.90 else "DECODER" if not nodec else
           "LOOKUP-C1" if (permok and retain) else "LOOKUP-NOCAUSAL" if permok else "PARTIAL" if perm["2"] >= 0.90 else "ORDER-ONLY")
    return {"label": lab, "perm_k8": perm, "nodec_k2": tp["by_d"]["4"]["acc_kprime"]["2"], "lp5p": lp5p, "retain": retain, "swap4": sw, "lp5_test2": lp5, "stable": (sok, smsg)}


def stage2(runs=R):
    o = {"seeds": {r: stage2_seed(runs, r) for r in S2}}
    labs = [v["label"] for v in o["seeds"].values()]
    o["verdict"] = ("pending" if "pending" in labs else "PASS" if labs.count("LOOKUP-C1") == 2 else "SPLIT" if "LOOKUP-C1" in labs else "FAIL") + " | " + " / ".join(labs)
    return o


def fixtures():
    res = []

    def world(perm=0.95, k2=0.05, lp5p=0.6, t2=1.0, sw=0.9, lp5=0.8, co=0.1, s1_acc=0.95, s1_p=0.9, s1_w=0.9, ws0=0.98):
        d = tempfile.mkdtemp()
        for run in DIRS:
            os.makedirs(os.path.join(d, run)); json.dump({"by_d": {x: {"acc": 0.09} for x in ("2", "3", "4")}}, open(os.path.join(d, run, "eval_acc.json"), "w"))
        os.makedirs(os.path.join(d, "lb_step0")); json.dump({"by_d": {"2": {"ws": {"1": ws0}}}}, open(os.path.join(d, "lb_step0", "eval_test_perm.json"), "w"))
        for run in S1 + (S1_OLD,):
            p = os.path.join(d, run); os.makedirs(p)
            json.dump({"by_d": {"2": {"acc": s1_acc}}}, open(os.path.join(p, "eval_acc.json"), "w"))
            json.dump({"by_d": {"2": {"acc_kprime": {"8": s1_p}, "ws": {"1": s1_w}}}}, open(os.path.join(p, "eval_test_perm.json"), "w"))
        for run in S2:
            p = os.path.join(d, run); os.makedirs(p)
            json.dump({"by_d": {x: {"acc": t2} for x in ("1", "2", "3", "4")}}, open(os.path.join(p, "eval_acc.json"), "w"))
            json.dump({"by_d": {"2": {"acc_kprime": {"8": perm}}, "3": {"acc_kprime": {"8": perm}}, "4": {"acc_kprime": {"8": perm, "2": k2}}},
                       "mean_at_2_d4": {"intact": {"8": 1.0}, "mean_at_t": {"8": 1.0 - lp5p}}}, open(os.path.join(p, "eval_test_perm.json"), "w"))
            json.dump({"by_d": {"4": {"swap_mid": {"4": {"follow_U": sw, "floor": 0.1}}}}}, open(os.path.join(p, "eval_swap_kd.json"), "w"))
            json.dump({"by_d": {"4": {"intact": {"4": 1.0}, "mean_at_t": {"4": 1.0 - lp5}}}}, open(os.path.join(p, "eval_lp5h.json"), "w"))
            json.dump({"by_d": {x: {"answer_acc": co, "majority": 0.1} for x in ("3", "4")}}, open(os.path.join(p, "eval_cue_only.json"), "w"))
            json.dump({"n_nonfinite": 0}, open(os.path.join(p, "train_summary.json"), "w"))
            with open(os.path.join(p, "log.jsonl"), "w") as fh:
                for i in range(1, 161):
                    fh.write(json.dumps({"step": i * 100, "loss": 0.01 if i > 20 else 2.0}) + "\n")
        return d
    w = world(); res.append(("all pass", stage2(w)["verdict"].startswith("PASS") and stage1(w)["verdict"].startswith("SEED 2/2") and step0(w)["verdict"] == "STEP0-OK", stage2(w)["verdict"]))
    o = stage2(world(perm=0.5)); res.append(("perm fails -> ORDER-ONLY", "ORDER-ONLY" in o["verdict"] and o["verdict"].startswith("FAIL"), o["verdict"]))
    o = stage2(world(sw=0.3)); res.append(("swap fails -> LOOKUP-NOCAUSAL", "LOOKUP-NOCAUSAL" in o["verdict"], o["verdict"]))
    o = stage2(world(k2=0.6)); res.append(("decoder does 2 steps -> DECODER", "DECODER" in o["verdict"], o["verdict"]))
    o = stage2(world(co=0.6)); res.append(("cue-only -> VOID", "VOID" in o["verdict"], o["verdict"]))
    o = stage2(world(t2=0.5)); res.append(("TEST2 collapse -> NOT-FORMED", "NOT-FORMED" in o["verdict"], o["verdict"]))
    o = stage1(world(s1_w=0.5)); res.append(("stage1 ws fails -> POSITION-REEMERGED", "NO-SEED" in o["verdict"] and "POSITION-REEMERGED" in o["verdict"], o["verdict"]))
    o = stage1(world(s1_acc=0.5)); res.append(("stage1 not formed", "NOT-FORMED" in o["verdict"], o["verdict"]))
    o = step0(world(ws0=0.9)); res.append(("step0 fail", o["verdict"] == "STEP0-FAIL", o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); ap.add_argument("--stage", default="2", choices=("step0", "1", "2")); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = {"step0": step0, "1": stage1, "2": stage2}[a.stage](); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT % a.stage, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
