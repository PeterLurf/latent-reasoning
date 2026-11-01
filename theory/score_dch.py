"""THEORY §257 DISTRACT-CHAIN scorer (docs/research_distractchain_2026-11-01/DC2_v1.md; derived from score_ds §256; Theory is the only scorer; committed before any output).
Runs dch_s1 / dch_s2 (8k FT of dis_s1 / dis_s2 with chain-reading and given-before-op add_c inserts added to the §256 mix).
Files per run (host-built): eval_insert_ho.json {"fam": {F: {"acc8": p, "by_pos": {pos: p}}}, ...}; eval_acc / eval_swap_kd / eval_lp5h / eval_cue_only (as §251).
Binding per seed:
  HELD-OUT (never trained): H-CHAINSRC-OP (sub_c / mul_c chain-reading inserts) and H-LIVE: each acc8 >= 0.90 and min(by_pos) >= 0.80.
  TRAINED-TYPE: H-CHAINSRC and H-GIVPOS acc8 >= 0.90.
  RETAIN: eval_acc d 1-4 >= 0.98; NOT-DECODER by_d['4'].acc_kprime['2'] <= 0.20; SWAP4; LP5-D4.
  C2 retention: eval_acc by_d['6'].acc >= 0.90.
  Guards: CUE-ONLY d 3-4 (score_ae rule); STABLE (score_ao rule).
Labels (first match): VOID / UNSTABLE / NOT-CAUSAL / C2-LOST / CONTENT-SELECTOR (HELD-OUT + TRAINED-TYPE) / TRAINED-TYPE-ONLY / FAIL. Per arm: CONTENT-SELECTOR k/2.
    python -B theory/score_dch.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory"))
import score_ao as AO  # noqa: E402

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_dch.json")
RUNS = {"dch_s1": "dis_s1", "dch_s2": "dis_s2"}
BIND = ("H-CHAINSRC-OP", "H-LIVE")
TRAINED = ("H-CHAINSRC", "H-GIVPOS")
DIRS = ("s228_pdir2c_s1_reg", "s228_pdir2c_s1_reg_bs50")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def seed(runs, run, ref):
    f = {x: jl(os.path.join(runs, run, x + ".json")) for x in ("eval_insert_ho", "eval_acc", "eval_swap_kd", "eval_lp5h", "eval_cue_only")}
    rf = {}; dirs = [jl(os.path.join(runs, d, "eval_acc.json")) for d in DIRS]
    if any(v is None for v in list(f.values()) + [rf] + dirs):
        return {"label": "pending"}
    a = f["eval_acc"]["by_d"]; ho = f["eval_insert_ho"]; co = f["eval_cue_only"]["by_d"]
    guard = all(co[d]["answer_acc"] <= max(co[d]["majority"], min(x["by_d"][d]["acc"] for x in dirs)) + 0.05 for d in ("3", "4"))
    sok, smsg = AO.stable(runs, run)
    fam = {F: ho["fam"][F]["acc8"] >= 0.90 and min(ho["fam"][F]["by_pos"].values()) >= 0.80 for F in BIND}
    held = all(fam.values()); trained = all(ho["fam"][F]["acc8"] >= 0.90 for F in TRAINED)
    sw = f["eval_swap_kd"]["by_d"]["4"]["swap_mid"]["4"]; h = f["eval_lp5h"]["by_d"]["4"]
    nodec = a["4"]["acc_kprime"]["2"] <= 0.20
    retain = all(a[d]["acc"] >= 0.98 for d in ("1", "2", "3", "4")) and nodec and sw["follow_U"] >= 0.50 and sw["follow_U"] >= sw["floor"] + 0.30 and h["intact"]["4"] - h["mean_at_t"]["4"] >= 0.30
    d6 = a["6"]["acc"]; c2 = d6 >= 0.90
    lab = ("VOID" if not guard else "UNSTABLE" if sok is False else "NOT-CAUSAL" if not retain else "C2-LOST" if not c2 else
           "CONTENT-SELECTOR" if (held and trained) else "TRAINED-TYPE-ONLY" if trained else "FAIL")
    return {"label": lab, "families": fam, "trained_type": trained, "retain": retain, "nodec_k2": a["4"]["acc_kprime"]["2"], "d6": d6, "stable": (sok, smsg)}


def score(runs=R):
    o = {"section": 257, "seeds": {r: seed(runs, r, ref) for r, ref in RUNS.items()}}
    labs = [v["label"] for v in o["seeds"].values()]
    if "pending" in labs:
        o["verdict"] = "pending"; return o
    o["verdict"] = "CONTENT-SELECTOR %d/2 (%s)" % (labs.count("CONTENT-SELECTOR"), " / ".join(labs)); return o


def fixtures():
    res = []

    def world(ho=0.95, pos=0.9, tr=0.95, d6=0.95, sw=0.9, t2=1.0, co=0.1, k2=0.05):
        d = tempfile.mkdtemp()
        for run in DIRS:
            os.makedirs(os.path.join(d, run)); json.dump({"by_d": {x: {"acc": 0.09} for x in ("3", "4")}}, open(os.path.join(d, run, "eval_acc.json"), "w"))
        for run in RUNS:
            p = os.path.join(d, run); os.makedirs(p)
            fam = {F: {"acc8": ho, "by_pos": {"a": pos, "b": pos}} for F in BIND}; fam.update({F: {"acc8": tr, "by_pos": {"a": tr}} for F in TRAINED})
            json.dump({"fam": fam, "indist": {}}, open(os.path.join(p, "eval_insert_ho.json"), "w"))
            json.dump({"by_d": {**{x: {"acc": t2, "acc_kprime": {"2": k2 if x == "4" else 1.0}} for x in ("1", "2", "3", "4")}, "6": {"acc": d6}}}, open(os.path.join(p, "eval_acc.json"), "w"))
            json.dump({"by_d": {"4": {"swap_mid": {"4": {"follow_U": sw, "floor": 0.1}}}}}, open(os.path.join(p, "eval_swap_kd.json"), "w"))
            json.dump({"by_d": {"4": {"intact": {"4": 1.0}, "mean_at_t": {"4": 0.1}}}}, open(os.path.join(p, "eval_lp5h.json"), "w"))
            json.dump({"by_d": {x: {"answer_acc": co, "majority": 0.1} for x in ("3", "4")}}, open(os.path.join(p, "eval_cue_only.json"), "w"))
            json.dump({"n_nonfinite": 0}, open(os.path.join(p, "train_summary.json"), "w"))
            with open(os.path.join(p, "log.jsonl"), "w") as fh:
                for i in range(1, 81):
                    fh.write(json.dumps({"step": i * 100, "loss": 0.01}) + chr(10))
        return d
    for name, kw, want in (("all", {}, "CONTENT-SELECTOR 2/2"), ("held-out fails", {"ho": 0.6}, "TRAINED-TYPE-ONLY"), ("pos min fails", {"pos": 0.7}, "TRAINED-TYPE-ONLY"),
                           ("nothing", {"ho": 0.6, "tr": 0.6}, "FAIL"), ("C2 lost", {"d6": 0.8}, "C2-LOST"), ("decoder shortcut", {"k2": 0.6}, "NOT-CAUSAL"),
                           ("swap fails", {"sw": 0.2}, "NOT-CAUSAL"), ("cue-only", {"co": 0.6}, "VOID")):
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
