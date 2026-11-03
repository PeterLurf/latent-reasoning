"""THEORY §259 BUNDLE scorer (docs/research_bundle_2026-11-01/BU_v1.md + Note 259-1 / 259-2; Theory is the only scorer; committed before launch).
Runs: bun_s1 / bun_s2 (dch_s<k> + --hold_inc 1.6 --hold_log, keeping the §257 insert mix, 8k, hold_start 500), bunc_s1 / bunc_s2 (the same without the penalty).
Refs: dch_s1 / dch_s2 (eval_acc for C2 retention).
Binding per BUN seed:
  C1 RETAIN: eval_acc d 1-4 >= 0.98; NOT-DECODER by_d['4'].acc_kprime['2'] <= 0.20; SWAP4; LP5-D4; CUE-ONLY (score_ae rule).
  C2 retention: d 6 >= dch d 6 - 0.05 and d 7 >= dch d 7 - 0.05; eval_ws_deep by_d['6'] passes 5 and 6 >= 0.90.
  C6: eval_long_k128 d 1-6: acc[64], acc[128] >= acc[8] - 0.02 (score_hx.hold); norm guard one-sided vs bunc (score_lh.norm_guard, GAMED iff > 1.25);
      LP0-W: gn_plain windows with step > 1000 <= max(2 x bunc post-1k max, 10) (score_fd.lp0). SEED-VAR (both seeds): |acc128_s1 - acc128_s2| <= 0.05 at every d 1-6 (reported; fails C6 for both if broken).
  C5: score_hx.halt on eval_halthead (C5-EXTRAP required).
  STABLE (score_ao).
Label: BUNDLE-PASS (scoped) if all pass; else 'FAIL: <failing criteria>'. Per arm: BUNDLE k/n.
    python -B theory/score_bu.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory"))
import score_ao as AO  # noqa: E402
import score_lh as LH  # noqa: E402
import score_fd as FD  # noqa: E402
import score_hx as HX  # noqa: E402

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_bu.json")
SEEDS = {"s1": ("bun_s1", "bunc_s1", "dch_s1"), "s2": ("bun_s2", "bunc_s2", "dch_s2")}
DIRS = ("s228_pdir2c_s1_reg", "s228_pdir2c_s1_reg_bs50")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def seed(runs, bun, ctrl, ref):
    f = {x: jl(os.path.join(runs, bun, x + ".json")) for x in ("eval_acc", "eval_swap_kd", "eval_lp5h", "eval_cue_only", "eval_ws_deep", "eval_long_k128", "eval_halthead")}
    rf = jl(os.path.join(runs, ref, "eval_acc.json")); dirs = [jl(os.path.join(runs, d, "eval_acc.json")) for d in DIRS]
    if any(v is None for v in list(f.values()) + [rf] + dirs) or not os.path.isdir(os.path.join(runs, ctrl)):
        return {"label": "pending"}
    a = f["eval_acc"]["by_d"]; sw = f["eval_swap_kd"]["by_d"]["4"]["swap_mid"]["4"]; h = f["eval_lp5h"]["by_d"]["4"]; co = f["eval_cue_only"]["by_d"]
    c1 = (all(a[d]["acc"] >= 0.98 for d in ("1", "2", "3", "4")) and a["4"]["acc_kprime"]["2"] <= 0.20 and sw["follow_U"] >= 0.50 and sw["follow_U"] >= sw["floor"] + 0.30
          and h["intact"]["4"] - h["mean_at_t"]["4"] >= 0.30
          and all(co[d]["answer_acc"] <= max(co[d]["majority"], min(x["by_d"][d]["acc"] for x in dirs)) + 0.05 for d in ("3", "4")))
    r = rf["by_d"]; w6 = f["eval_ws_deep"]["by_d"]["6"]
    c2 = a["6"]["acc"] >= r["6"]["acc"] - 0.05 and a["7"]["acc"] >= r["7"]["acc"] - 0.05 and w6["5"] >= 0.90 and w6["6"] >= 0.90
    hl = HX.hold(f["eval_long_k128"]); ng_ok, ng = LH.norm_guard(runs, bun, ctrl); lp_ok, lp = FD.lp0(runs, bun, ctrl)
    if ng_ok is None or lp_ok is None:   # Audit SCORER_BUGS MED-2: a missing / short bunc log is pending, not FAIL: C6
        return {"label": "pending", "norm_guard": (ng_ok, ng), "lp0": (lp_ok, lp)}
    c6 = hl["label"] == "HOLD" and ng_ok is True and lp_ok is True
    c5r = HX.halt(f["eval_halthead"]); c5 = c5r["label"] == "C5-EXTRAP"
    sok, smsg = AO.stable(runs, bun)
    if sok is None:   # Audit SCORER_BUGS MED-1: a missing / blank log is pending, never STABLE
        return {"label": "pending", "stable": (sok, smsg)}
    st = sok is not False
    crit = {"C1": c1, "C2": c2, "C6": c6, "C5": c5, "STABLE": st}
    fail = [k for k, v in crit.items() if not v]
    bl = jl(os.path.join(runs, ctrl, "eval_long_k128.json"))     # Note 259-3: BUN-CTRL hold, descriptive (attribution of a C6 HOLD to the penalty)
    bunc_hold = HX.hold(bl)["label"] if bl is not None else None
    return {"label": "BUNDLE-PASS (scoped)" if not fail else "FAIL: " + ",".join(fail), "crit": crit, "hold": hl, "bunc_hold_desc": bunc_hold, "norm_guard": (ng_ok, ng), "lp0": (lp_ok, lp),
            "c5": c5r.get("label"), "d6": a["6"]["acc"], "d7": a["7"]["acc"], "acc128": {d: f["eval_long_k128"]["by_d"][d]["acc"]["128"] for d in f["eval_long_k128"]["by_d"]}}


def score(runs=R):
    o = {"section": 259, "seeds": {}}
    for sd, (bun, ctrl, ref) in SEEDS.items():
        if os.path.isdir(os.path.join(runs, bun)):
            o["seeds"][sd] = seed(runs, bun, ctrl, ref)
    labs = {k: v["label"] for k, v in o["seeds"].items()}
    if not labs or "pending" in labs.values():
        o["verdict"] = "pending"; return o
    if len(labs) == 2:   # SEED-VAR on acc128 (C6)
        a1, a2 = o["seeds"]["s1"]["acc128"], o["seeds"]["s2"]["acc128"]
        sv = all(abs(a1[d] - a2[d]) <= 0.05 for d in a1)
        o["seed_var_ok"] = sv
        if not sv:
            for k in o["seeds"]:
                v = o["seeds"][k]; v["crit"]["C6"] = False
                fail = [c for c, ok in v["crit"].items() if not ok]; v["label"] = "FAIL: " + ",".join(fail)
    n = sum(v["label"].startswith("BUNDLE-PASS") for v in o["seeds"].values())
    o["verdict"] = "BUNDLE %d/%d (%s)" % (n, len(o["seeds"]), " / ".join(v["label"] for v in o["seeds"].values())); return o


def fixtures():
    res = []

    def world(t2=1.0, k2=0.05, d6=0.97, w56=0.95, a128=1.0, ng=1.0, gn=1.0, c5k=None, a128_s2=None):
        d = tempfile.mkdtemp()
        for run in DIRS:
            os.makedirs(os.path.join(d, run)); json.dump({"by_d": {x: {"acc": 0.09} for x in ("3", "4")}}, open(os.path.join(d, run, "eval_acc.json"), "w"))
        for sd, (bun, ctrl, ref) in SEEDS.items():
            os.makedirs(os.path.join(d, ref)); json.dump({"by_d": {"6": {"acc": 0.97}, "7": {"acc": 0.90}}}, open(os.path.join(d, ref, "eval_acc.json"), "w"))
            for run, isb in ((bun, True), (ctrl, False)):
                p = os.path.join(d, run); os.makedirs(p)
                with open(os.path.join(p, "log.jsonl"), "w") as fh:
                    for i in range(1, 81):
                        r = {"step": i * 100, "loss": 0.01, "gn_plain_mean100": (gn if isb else 1.0)}
                        if i % 10 == 0:
                            r["bin"] = i // 10; r["hold"] = {"norm_guard": {"s8": (ng if isb else 1.0), "sd2": 1.0}}
                        fh.write(json.dumps(r) + chr(10))
                if not isb:
                    continue
                json.dump({"by_d": {**{x: {"acc": t2, "acc_kprime": {"2": k2 if x == "4" else 1.0}} for x in ("1", "2", "3", "4")}, "6": {"acc": d6}, "7": {"acc": 0.9}}},
                          open(os.path.join(p, "eval_acc.json"), "w"))
                json.dump({"by_d": {"4": {"swap_mid": {"4": {"follow_U": 0.9, "floor": 0.1}}}}}, open(os.path.join(p, "eval_swap_kd.json"), "w"))
                json.dump({"by_d": {"4": {"intact": {"4": 1.0}, "mean_at_t": {"4": 0.1}}}}, open(os.path.join(p, "eval_lp5h.json"), "w"))
                json.dump({"by_d": {x: {"answer_acc": 0.1, "majority": 0.1} for x in ("3", "4")}}, open(os.path.join(p, "eval_cue_only.json"), "w"))
                json.dump({"by_d": {"6": {str(t): (w56 if t in (5, 6) else 1.0) for t in range(1, 9)}}}, open(os.path.join(p, "eval_ws_deep.json"), "w"))
                aa = a128_s2 if (a128_s2 is not None and sd == "s2") else a128
                json.dump({"by_d": {str(x): {"acc": {"8": 1.0, "64": 1.0, "128": aa}} for x in range(1, 7)}}, open(os.path.join(p, "eval_long_k128.json"), "w"))
                hh = {"head": {"val_acc": 0.95}}
                for x in range(1, 8):
                    hh[str(x)] = {"acc_k8": 1.0, "acc_halt": 1.0, "mean_khalt": (x - 0.8) if (c5k is None or x < 5) else c5k}
                json.dump(hh, open(os.path.join(p, "eval_halthead.json"), "w"))
                json.dump({"n_nonfinite": 0}, open(os.path.join(p, "train_summary.json"), "w"))
        return d
    cases = (("all", {}, "BUNDLE 2/2"), ("C1 decoder", {"k2": 0.6}, "C1"), ("C2 lost", {"d6": 0.8}, "C2"), ("C2 passes 5-6 off", {"w56": 0.5}, "C2"),
             ("C6 drift", {"a128": 0.5}, "C6"), ("C6 gamed", {"ng": 1.4}, "C6"), ("C6 lp0", {"gn": 30.0}, "C6"), ("C5 late halt", {"c5k": 8.0}, "C5"),
             ("seed var", {"a128_s2": 0.9}, "C6"))
    for name, kw, want in cases:
        o = score(world(**kw)); ok = o["verdict"].startswith("BUNDLE 2/2") if want == "BUNDLE 2/2" else (want in o["verdict"] and "BUNDLE 0/2" in o["verdict"])
        res.append((name, ok, o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
