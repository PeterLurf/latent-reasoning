"""THEORY §256 DISTRACT scorer (docs/research_distract_2026-11-01/DS_v1.md; Theory is the only scorer; committed before any output).
Runs dis_s1 / dis_s2 (rcb_plaw4_s{1,2} + --distract_p 1, 16k). Comparators rcb_plaw4_s{1,2} (eval_acc d 5-6 as stored).
Files per run (host-built): eval_insert_ho.json {"fam": {F: {"acc8": p, "by_pos": {pos: p}}}, "indist": {cond: {"acc8": p}}}, F in H-OP, H-CHAINSRC, H-GIVPOS (binding),
H-EXTEND, H-LIVE, K3 (descriptive); eval_ws_deep.json {"by_d": {"5": {pass: p}, "6": {...}}}; eval_acc / eval_swap_kd / eval_lp5h / eval_cue_only (as §251).
Binding per seed:
  HELD-OUT: each of H-OP, H-CHAINSRC, H-GIVPOS: acc8 >= 0.90 and min(by_pos) >= 0.80.
  RETAIN: eval_acc d 1-4 >= 0.98; NOT-DECODER eval_acc by_d['4'].acc_kprime['2'] <= 0.20 (Audit 256-1); SWAP4 (by_d['4'].swap_mid['4'] follow_U >= 0.50 and >= floor + 0.30); LP5-D4 (eval_lp5h d 4 intact['4'] - mean_at_t['4'] >= 0.30).
  C2: d5 >= 0.80 (retention); AMBITION d6 >= 0.65 and >= rcb same seed + 0.10; SELECTION-CREDIT: eval_ws_deep d 6 passes 1-4 all >= 0.90.
  Guards: CUE-ONLY d 3-4 (score_ae rule); STABLE (score_ao rule).
Labels (first match): VOID / UNSTABLE / NOT-CAUSAL / SELECTOR+C2 (HELD-OUT + C2 d5 + AMBITION + SELECTION-CREDIT) / SELECTOR (HELD-OUT) /
  C2-ONLY (AMBITION without HELD-OUT) / TRAINED-INSERT-ONLY (indist acc8 >= 0.90 in every cond) / FAIL. Per arm: label counts.
    python -B theory/score_ds.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory"))
import score_ao as AO  # noqa: E402

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_ds.json")
RUNS = {"dis_s1": "rcb_plaw4_s1", "dis_s2": "rcb_plaw4_s2"}
BIND = ("H-OP", "H-CHAINSRC", "H-GIVPOS")
DIRS = ("s228_pdir2c_s1_reg", "s228_pdir2c_s1_reg_bs50")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def seed(runs, run, ref):
    f = {x: jl(os.path.join(runs, run, x + ".json")) for x in ("eval_insert_ho", "eval_ws_deep", "eval_acc", "eval_swap_kd", "eval_lp5h", "eval_cue_only")}
    rf = jl(os.path.join(runs, ref, "eval_acc.json")); dirs = [jl(os.path.join(runs, d, "eval_acc.json")) for d in DIRS]
    if any(v is None for v in list(f.values()) + [rf] + dirs):
        return {"label": "pending"}
    a = f["eval_acc"]["by_d"]; ho = f["eval_insert_ho"]; co = f["eval_cue_only"]["by_d"]
    guard = all(co[d]["answer_acc"] <= max(co[d]["majority"], min(x["by_d"][d]["acc"] for x in dirs)) + 0.05 for d in ("3", "4"))
    sok, smsg = AO.stable(runs, run)
    fam = {F: ho["fam"][F]["acc8"] >= 0.90 and min(ho["fam"][F]["by_pos"].values()) >= 0.80 for F in BIND}
    held = all(fam.values())
    sw = f["eval_swap_kd"]["by_d"]["4"]["swap_mid"]["4"]; h = f["eval_lp5h"]["by_d"]["4"]
    nodec = a["4"]["acc_kprime"]["2"] <= 0.20
    retain = all(a[d]["acc"] >= 0.98 for d in ("1", "2", "3", "4")) and nodec and sw["follow_U"] >= 0.50 and sw["follow_U"] >= sw["floor"] + 0.30 and h["intact"]["4"] - h["mean_at_t"]["4"] >= 0.30
    d5ok = a["5"]["acc"] >= 0.80; d6 = a["6"]["acc"]; d6ref = rf["by_d"]["6"]["acc"]
    amb = d6 >= 0.65 and d6 >= d6ref + 0.10
    ws6 = f["eval_ws_deep"]["by_d"]["6"]; credit = all(ws6[str(t)] >= 0.90 for t in (1, 2, 3, 4))
    indist = all(v["acc8"] >= 0.90 for v in ho["indist"].values())
    lab = ("VOID" if not guard else "UNSTABLE" if sok is False else "NOT-CAUSAL" if not retain else
           "SELECTOR+C2" if (held and d5ok and amb and credit) else "SELECTOR" if held else "C2-ONLY" if amb else
           "TRAINED-INSERT-ONLY" if indist else "FAIL")
    return {"label": lab, "families": fam, "retain": retain, "nodec_k2": a["4"]["acc_kprime"]["2"], "d5": a["5"]["acc"], "d6": d6, "d6_ref": d6ref, "ws6_p1_4": [ws6[str(t)] for t in (1, 2, 3, 4)],
            "credit": credit, "indist": indist, "stable": (sok, smsg)}


def score(runs=R):
    o = {"section": 256, "seeds": {r: seed(runs, r, ref) for r, ref in RUNS.items()}}
    labs = [v["label"] for v in o["seeds"].values()]
    if "pending" in labs:
        o["verdict"] = "pending"; return o
    sel = sum(l_ in ("SELECTOR", "SELECTOR+C2") for l_ in labs); c2 = sum(l_ in ("SELECTOR+C2", "C2-ONLY") for l_ in labs)
    o["verdict"] = "SELECTOR %d/2, C2 %d/2 (%s)" % (sel, c2, " / ".join(labs)); return o


def fixtures():
    res = []

    def world(fam=0.95, pos=0.9, d6=0.8, ws=0.95, sw=0.9, t2=1.0, co=0.1, ind=0.95, d5=0.9, k2=0.05):
        d = tempfile.mkdtemp()
        for run in DIRS:
            os.makedirs(os.path.join(d, run)); json.dump({"by_d": {x: {"acc": 0.09} for x in ("3", "4")}}, open(os.path.join(d, run, "eval_acc.json"), "w"))
        for ref in RUNS.values():
            os.makedirs(os.path.join(d, ref)); json.dump({"by_d": {"6": {"acc": 0.35}}}, open(os.path.join(d, ref, "eval_acc.json"), "w"))
        for run in RUNS:
            p = os.path.join(d, run); os.makedirs(p)
            json.dump({"fam": {F: {"acc8": fam, "by_pos": {"before": pos, "after": pos}} for F in BIND + ("H-EXTEND",)}, "indist": {"before_k2": {"acc8": ind}}},
                      open(os.path.join(p, "eval_insert_ho.json"), "w"))
            json.dump({"by_d": {"6": {str(t): ws for t in range(1, 9)}}}, open(os.path.join(p, "eval_ws_deep.json"), "w"))
            json.dump({"by_d": {**{x: {"acc": t2, "acc_kprime": {"2": k2 if x == "4" else 1.0}} for x in ("1", "2", "3", "4")}, "5": {"acc": d5}, "6": {"acc": d6}}}, open(os.path.join(p, "eval_acc.json"), "w"))
            json.dump({"by_d": {"4": {"swap_mid": {"4": {"follow_U": sw, "floor": 0.1}}}}}, open(os.path.join(p, "eval_swap_kd.json"), "w"))
            json.dump({"by_d": {"4": {"intact": {"4": 1.0}, "mean_at_t": {"4": 0.1}}}}, open(os.path.join(p, "eval_lp5h.json"), "w"))
            json.dump({"by_d": {x: {"answer_acc": co, "majority": 0.1} for x in ("3", "4")}}, open(os.path.join(p, "eval_cue_only.json"), "w"))
            json.dump({"n_nonfinite": 0}, open(os.path.join(p, "train_summary.json"), "w"))
            with open(os.path.join(p, "log.jsonl"), "w") as fh:
                for i in range(1, 161):
                    fh.write(json.dumps({"step": i * 100, "loss": 0.01}) + "\n")
        return d
    for name, kw, want in (("all", {}, "SELECTOR 2/2, C2 2/2"), ("no C2", {"d6": 0.4}, "SELECTOR 2/2, C2 0/2"), ("no credit", {"ws": 0.5}, "SELECTOR 2/2, C2 0/2"),
                           ("held-out fail, C2", {"fam": 0.5}, "SELECTOR 0/2, C2 2/2"), ("filter only", {"fam": 0.5, "d6": 0.3}, "TRAINED-INSERT-ONLY"),
                           ("not causal", {"sw": 0.2}, "NOT-CAUSAL"), ("decoder shortcut K'2 0.6", {"k2": 0.6}, "NOT-CAUSAL"), ("cue-only", {"co": 0.6}, "VOID"), ("pos min fails", {"pos": 0.7, "d6": 0.3}, "TRAINED-INSERT-ONLY")):
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
