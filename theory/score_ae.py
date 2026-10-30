"""THEORY §251 ANSWER-EXTEND scorer (docs/research_answerextend_2026-10-30/AE_v1.md; Theory is the only scorer).
Arms (init aw_plaw2_s1, §237 recipe, REG law, d ~ U{1..4}, 16k final): AEX aex_s1 / aex_s2 (--aux_d_max 2) and AEX0 aex0_s1 / aex0_s2 (--aux_fade 0,0: per-pass targets
only in the seed). CTRL rcb_plaw4_s{1,2} re-evaluated under the current eval module (rcb_plaw4_s{1,2}_re). CD-AN rcb_pcd4_s1; DIRECT bars s228_pdir2c_s1_reg (+ _bs50).
Per arm per seed, binding (d 4-centred; d 3 can pass via passes 1-2 + the decoder's last step):
  ACC4: d 4 acc (K' 8) >= 0.50 and >= CD + 0.25; NOT-DECODER: d 4 acc at K' 2 <= 0.20; SWAP4: SWAP-MID d 4 t 2 follow_U >= 0.50 and >= floor + 0.30;
  LP5-D4: lp5h d 4 drop >= 0.30; CUE-ONLY guard d 3-4 (bar = max(majority, min DIRECT bs-variant) + 0.05); STABLE (score_ao rule).
Labels (first match): VOID / UNSTABLE / EXTEND / EXTEND-D3-ONLY (d 3 acc >= 0.60 and SWAP-MID d 3 t 2 >= 0.50 / floor + 0.30, d 4 gates fail) / NOT-EXTEND.
Per arm overall: EXTEND (both) / SPLIT / FAIL. Descriptive: d 3 gates, WS, the learning curve (train answer acc by d), SURVIVAL vs CTRL.
    python -B theory/score_ae.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory"))
import score_ao as AO  # noqa: E402  (stable rule)

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_ae.json")
ARMS = {"AEX": ("aex_s1", "aex_s2"), "AEX0": ("aex0_s1", "aex0_s2")}
CTRL = ("rcb_plaw4_s1_re", "rcb_plaw4_s2_re")
CD, DIRS = "rcb_pcd4_s1", ("s228_pdir2c_s1_reg", "s228_pdir2c_s1_reg_bs50")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def sw_ok(r):
    return r is not None and r["follow_U"] >= 0.50 and r["follow_U"] >= r["floor"] + 0.30


def one(runs, run):
    g = {x: jl(os.path.join(runs, run, x + ".json")) for x in ("eval_acc", "eval_swap_kd", "eval_lp5h", "eval_cue_only")}
    cd = jl(os.path.join(runs, CD, "eval_acc.json")); dirs = [jl(os.path.join(runs, d, "eval_acc.json")) for d in DIRS]
    if any(v is None for v in list(g.values()) + [cd] + dirs):
        return {"label": "pending"}
    a = g["eval_acc"]["by_d"]; acc = lambda d, k=None: a[d]["acc"] if k is None else a[d]["acc_kprime"][str(k)]  # noqa: E731
    co = g["eval_cue_only"]["by_d"]
    guard = all(co[d]["answer_acc"] <= max(co[d]["majority"], min(x["by_d"][d]["acc"] for x in dirs)) + 0.05 for d in ("3", "4"))
    sk = g["eval_swap_kd"]["by_d"]; s4 = sk["4"]["swap_mid"].get("4"); s3 = (sk["3"].get("swap_mid_t2") or {}).get("3")
    h = g["eval_lp5h"]["by_d"]; lp4 = h["4"]["intact"]["4"] - h["4"]["mean_at_t"]["4"]
    gates = {"ACC4": acc("4") >= 0.50 and acc("4") >= cd["by_d"]["4"]["acc"] + 0.25, "NOT-DECODER": acc("4", 2) <= 0.20, "SWAP4": sw_ok(s4), "LP5-D4": lp4 >= 0.30}
    d3 = acc("3") >= 0.60 and sw_ok(s3)
    sok, smsg = AO.stable(runs, run)
    lab = ("VOID" if not guard else "UNSTABLE" if sok is False else "EXTEND" if all(gates.values()) else "EXTEND-D3-ONLY" if d3 else "NOT-EXTEND")
    return {"label": lab, "gates": gates, "d3_desc": d3, "acc": {d: acc(d) for d in ("2", "3", "4")}, "acc4_k2": acc("4", 2), "swap4": s4, "swap3_t2": s3, "lp5_d4": lp4, "stable": (sok, smsg)}


def score(runs=R):
    o = {"section": 251, "registered": "AE_v1.md", "arms": {}, "ctrl": {c: one(runs, c) for c in CTRL}}
    out = []
    for arm, rs in ARMS.items():
        seeds = {r: one(runs, r) for r in rs}; o["arms"][arm] = seeds
        labs = [v["label"] for v in seeds.values()]
        v = "pending" if "pending" in labs else "EXTEND" if labs.count("EXTEND") == 2 else "SPLIT" if "EXTEND" in labs else "FAIL"
        o["arms"][arm + "_overall"] = v; out.append("%s %s (%s)" % (arm, v, " / ".join(labs)))
    o["verdict"] = " | ".join(out); return o


def fixtures():
    res = []

    def world(acc4=0.9, k2=0.1, f4=0.8, lp4=0.8, co=0.1, acc3=1.0, f3=0.9, per=None):
        d = tempfile.mkdtemp()
        for run in (CD,) + DIRS:
            os.makedirs(os.path.join(d, run)); json.dump({"by_d": {x: {"acc": 0.09} for x in ("2", "3", "4")}}, open(os.path.join(d, run, "eval_acc.json"), "w"))
        for run in ARMS["AEX"] + ARMS["AEX0"] + CTRL:
            g = dict(acc4=acc4, k2=k2, f4=f4, lp4=lp4, co=co, acc3=acc3, f3=f3); g.update((per or {}).get(run, {}))
            p = os.path.join(d, run); os.makedirs(p)
            json.dump({"by_d": {"2": {"acc": 1.0, "acc_kprime": {"2": 1.0}}, "3": {"acc": g["acc3"], "acc_kprime": {"2": 1.0}}, "4": {"acc": g["acc4"], "acc_kprime": {"2": g["k2"]}}}}, open(os.path.join(p, "eval_acc.json"), "w"))
            json.dump({"by_d": {"3": {"swap_mid": {"3": {"follow_U": 0.9, "floor": 0.1}}, "swap_mid_t2": {"3": {"follow_U": g["f3"], "floor": 0.1}}},
                                "4": {"swap_mid": {"4": {"follow_U": g["f4"], "floor": 0.15}}}}}, open(os.path.join(p, "eval_swap_kd.json"), "w"))
            json.dump({"by_d": {x: {"intact": {x: 1.0}, "mean_at_t": {x: 1.0 - g["lp4"]}} for x in ("3", "4")}}, open(os.path.join(p, "eval_lp5h.json"), "w"))
            json.dump({"by_d": {x: {"answer_acc": g["co"], "majority": 0.1} for x in ("2", "3", "4")}}, open(os.path.join(p, "eval_cue_only.json"), "w"))
            json.dump({"n_nonfinite": 0}, open(os.path.join(p, "train_summary.json"), "w"))
            with open(os.path.join(p, "log.jsonl"), "w") as fh:
                for i in range(1, 161):
                    fh.write(json.dumps({"step": i * 100, "loss": 0.01 if i > 20 else 2.0}) + "\n")
        return d
    o = score(world()); res.append(("both arms extend -> EXTEND", o["verdict"] == "AEX EXTEND (EXTEND / EXTEND) | AEX0 EXTEND (EXTEND / EXTEND)", o["verdict"]))
    o = score(world(acc4=0.3)); res.append(("d4 fails, d3 ok -> EXTEND-D3-ONLY", "EXTEND-D3-ONLY" in o["verdict"] and "FAIL" in o["verdict"], o["verdict"]))
    o = score(world(k2=0.6)); res.append(("decoder does 2 steps -> not EXTEND", "AEX FAIL" in o["verdict"], o["verdict"]))
    o = score(world(f4=0.3)); res.append(("d4 swap not followed -> not EXTEND", "AEX FAIL" in o["verdict"], o["verdict"]))
    o = score(world(co=0.5)); res.append(("cue alone -> VOID", "VOID" in o["verdict"], o["verdict"]))
    o = score(world(acc4=0.3, acc3=0.3)); res.append(("nothing -> NOT-EXTEND", "NOT-EXTEND" in o["verdict"], o["verdict"]))
    o = score(world(per={"aex0_s1": {"acc4": 0.3}, "aex0_s2": {"acc4": 0.3}})); res.append(("AEX extends, AEX0 not", o["verdict"].startswith("AEX EXTEND") and "AEX0 FAIL" in o["verdict"], o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
