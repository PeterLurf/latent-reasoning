"""THEORY §258 LARGER-BASE DISTRACT stage scorer (docs/research_largebase2_2026-11-01/LB2_v1.md; Theory is the only scorer; committed before the first gate read).
Runs (boxes/runs_fsq): bridge_s1 (small, aw_plaw2_s1 -> REG d<=4 + distract); askm_lr3e5 / askm_lr1e4 (S1); p2m_s1 (S2); dism_s1 (S3). Refs dis_s1 / dis_s2.
  B:  RETAIN (TEST2 d 1-4 >= 0.98, NOT-DECODER by_d['4'].acc_kprime['2'] <= 0.20, SWAP4, LP5-D4) and d 6 >= 0.90 -> BRIDGE-OK else BRIDGE-FAIL (S3 in two stages).
  S1: eval_st_classes.json {"pooled": {cls: {"acc": p}}} classes i-iv >= 0.98 and eval_acc by_d['1'].acc >= 0.98; winner = passing arm with higher mean class acc (tie: lr 3e-5).
  S2 (Note 258-4): 6k WS descriptive; 12k SEED-OK (TEST2 d 2 >= 0.90 and WS d 2 p 2 >= 0.90) / EXTEND (0.50 <= ws < 0.90, or < 0.50 rising >= 0.10 from 6k; +6k once, cap $1.5) / NO-SEED.
  S3: RETAIN (as B, + CUE-ONLY + STABLE); HOLDS d 6 >= 0.90 and d 7 >= 0.80 and eval_ws_deep by_d['6'] passes 1-6 >= 0.90;
      IMPROVES d 8 >= 0.70 or d 7 >= 0.95. Labels MEDIUM-C2-IMPROVES / MEDIUM-C2-HOLDS / MEDIUM-C1-ONLY / FAIL (PILOT, 1 seed).
    python -B theory/score_lb2.py [--stage B|S1|S2|S3] [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory"))
import score_ao as AO  # noqa: E402

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_lb2_%s.json")
DIRS = ("s228_pdir2c_s1_reg", "s228_pdir2c_s1_reg_bs50")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def retain(runs, run, cue=False):
    f = {x: jl(os.path.join(runs, run, x + ".json")) for x in ("eval_acc", "eval_swap_kd", "eval_lp5h") + (("eval_cue_only",) if cue else ())}
    if any(v is None for v in f.values()):
        return None, None
    a = f["eval_acc"]["by_d"]; sw = f["eval_swap_kd"]["by_d"]["4"]["swap_mid"]["4"]; h = f["eval_lp5h"]["by_d"]["4"]
    ok = (all(a[d]["acc"] >= 0.98 for d in ("1", "2", "3", "4")) and a["4"]["acc_kprime"]["2"] <= 0.20 and sw["follow_U"] >= 0.50 and sw["follow_U"] >= sw["floor"] + 0.30
          and h["intact"]["4"] - h["mean_at_t"]["4"] >= 0.30)
    if cue:
        dirs = [jl(os.path.join(runs, d, "eval_acc.json")) for d in DIRS]; co = f["eval_cue_only"]["by_d"]
        if any(x is None for x in dirs):
            return None, None
        ok = ok and all(co[d]["answer_acc"] <= max(co[d]["majority"], min(x["by_d"][d]["acc"] for x in dirs)) + 0.05 for d in ("3", "4"))
    return ok, a


def stage_b(runs=R):
    ok, a = retain(runs, "bridge_s1")
    if ok is None:
        return {"verdict": "pending"}
    return {"verdict": "BRIDGE-OK" if (ok and a["6"]["acc"] >= 0.90) else "BRIDGE-FAIL", "retain": ok, "d6": a["6"]["acc"], "d7": a.get("7", {}).get("acc")}


def stage_s1(runs=R):
    res = {}
    for arm in ("askm_lr3e5", "askm_lr1e4"):
        c = jl(os.path.join(runs, arm, "eval_st_classes.json")); a = jl(os.path.join(runs, arm, "eval_acc.json"))
        if c is None or a is None:
            return {"verdict": "pending"}
        cls = c["pooled"]; m = sum(cls[k]["acc"] for k in ("i", "ii", "iii", "iv")) / 4
        res[arm] = {"pass": all(cls[k]["acc"] >= 0.98 for k in ("i", "ii", "iii", "iv")) and a["by_d"]["1"]["acc"] >= 0.98, "mean": m}
    ok = [k for k, v in res.items() if v["pass"]]
    if not ok:
        return {"verdict": "S1-FAIL", "arms": res}
    win = max(ok, key=lambda k: (res[k]["mean"], k == "askm_lr3e5"))
    return {"verdict": "S1-OK winner %s" % win, "arms": res}


def ws_d2p2(runs, run):
    w = jl(os.path.join(runs, run, "eval_ws.json"))
    return None if w is None else (w.get("by_d", w))["2"]["2"]


def stage_s2(runs=R):
    """Note 258-4: 6k read (p2m_s1_ws6k/eval_ws.json) is DESCRIPTIVE only (no early park). At 12k (p2m_s1): SEED-OK if TEST2 d 2 >= 0.90 and WS d 2 pass 2 >= 0.90;
    EXTEND (one +6k, cap $1.5, run p2m_s1_ext) if 0.50 <= ws12 < 0.90, or ws12 < 0.50 with ws12 - ws6 >= 0.10 (rising); else NO-SEED.
    After an extension the same SEED-OK gate is applied to p2m_s1_ext (18k), final (no second extension)."""
    a = jl(os.path.join(runs, "p2m_s1", "eval_acc.json")); ws12 = ws_d2p2(runs, "p2m_s1"); ws6 = ws_d2p2(runs, "p2m_s1_ws6k")
    if a is None or ws12 is None:
        return {"verdict": "pending", "ws6k": ws6}
    acc = a["by_d"]["2"]["acc"]
    if acc >= 0.90 and ws12 >= 0.90:
        return {"verdict": "SEED-OK", "test2_d2": acc, "ws_d2_p2": ws12, "ws6k": ws6}
    ext = jl(os.path.join(runs, "p2m_s1_ext", "eval_acc.json")); wse = ws_d2p2(runs, "p2m_s1_ext")
    rising = ws6 is not None and ws12 - ws6 >= 0.10
    if 0.50 <= ws12 < 0.90 or (ws12 < 0.50 and rising):
        if ext is None or wse is None:
            return {"verdict": "EXTEND (+6k once, cap $1.5)", "test2_d2": acc, "ws_d2_p2": ws12, "ws6k": ws6}
        acc_e = ext["by_d"]["2"]["acc"]
        return {"verdict": "SEED-OK (after extension)" if (acc_e >= 0.90 and wse >= 0.90) else "NO-SEED (after extension)", "test2_d2_18k": acc_e, "ws_d2_p2_18k": wse, "ws12k": ws12, "ws6k": ws6}
    return {"verdict": "NO-SEED", "test2_d2": acc, "ws_d2_p2": ws12, "ws6k": ws6}


def stage_s3(runs=R):
    ok, a = retain(runs, "dism_s1", cue=True); w = jl(os.path.join(runs, "dism_s1", "eval_ws_deep.json"))
    if ok is None or w is None:
        return {"verdict": "pending"}
    sok, smsg = AO.stable(runs, "dism_s1"); ok = ok and sok is not False
    w6 = w["by_d"]["6"]; holds = a["6"]["acc"] >= 0.90 and a["7"]["acc"] >= 0.80 and all(w6[str(t)] >= 0.90 for t in range(1, 7))
    imp = a["8"]["acc"] >= 0.70 or a["7"]["acc"] >= 0.95
    lab = ("FAIL" if not ok else "MEDIUM-C2-IMPROVES" if (holds and imp) else "MEDIUM-C2-HOLDS" if holds else "MEDIUM-C1-ONLY")
    return {"verdict": lab + " (PILOT, 1 seed)", "retain": ok, "d6": a["6"]["acc"], "d7": a["7"]["acc"], "d8": a["8"]["acc"], "ws6": [w6[str(t)] for t in range(1, 7)], "stable": (sok, smsg)}


def fixtures():
    res = []

    def world(t2=1.0, k2=0.05, sw=0.9, d6=0.95, d7=0.85, d8=0.5, ws=0.95, cls=0.99, ws2=0.95, co=0.1, ws6=None, ext=None):
        d = tempfile.mkdtemp()
        for run in DIRS:
            os.makedirs(os.path.join(d, run)); json.dump({"by_d": {x: {"acc": 0.09} for x in ("3", "4")}}, open(os.path.join(d, run, "eval_acc.json"), "w"))
        acc = {"by_d": {**{x: {"acc": t2, "acc_kprime": {"2": k2 if x == "4" else 1.0}} for x in ("1", "2", "3", "4")}, "6": {"acc": d6}, "7": {"acc": d7}, "8": {"acc": d8}}}
        for run in ("bridge_s1", "dism_s1", "p2m_s1", "askm_lr3e5", "askm_lr1e4"):
            p = os.path.join(d, run); os.makedirs(p)
            json.dump(acc, open(os.path.join(p, "eval_acc.json"), "w"))
            json.dump({"by_d": {"4": {"swap_mid": {"4": {"follow_U": sw, "floor": 0.1}}}}}, open(os.path.join(p, "eval_swap_kd.json"), "w"))
            json.dump({"by_d": {"4": {"intact": {"4": 1.0}, "mean_at_t": {"4": 0.1}}}}, open(os.path.join(p, "eval_lp5h.json"), "w"))
            json.dump({"by_d": {x: {"answer_acc": co, "majority": 0.1} for x in ("3", "4")}}, open(os.path.join(p, "eval_cue_only.json"), "w"))
            json.dump({"by_d": {"6": {str(t): ws for t in range(1, 9)}}}, open(os.path.join(p, "eval_ws_deep.json"), "w"))
            json.dump({"by_d": {"2": {"1": 1.0, "2": ws2}}}, open(os.path.join(p, "eval_ws.json"), "w"))
            json.dump({"pooled": {k: {"acc": cls} for k in ("i", "ii", "iii", "iv")}}, open(os.path.join(p, "eval_st_classes.json"), "w"))
            json.dump({"n_nonfinite": 0}, open(os.path.join(p, "train_summary.json"), "w"))
            with open(os.path.join(p, "log.jsonl"), "w") as fh:
                for i in range(1, 81):
                    fh.write(json.dumps({"step": i * 100, "loss": 0.01}) + chr(10))
        if ws6 is not None:
            os.makedirs(os.path.join(d, "p2m_s1_ws6k")); json.dump({"by_d": {"2": {"2": ws6}}}, open(os.path.join(d, "p2m_s1_ws6k", "eval_ws.json"), "w"))
        if ext is not None:
            q = os.path.join(d, "p2m_s1_ext"); os.makedirs(q)
            json.dump(acc, open(os.path.join(q, "eval_acc.json"), "w")); json.dump({"by_d": {"2": {"2": ext}}}, open(os.path.join(q, "eval_ws.json"), "w"))
        return d
    cases = (("B ok", stage_b, {}, "BRIDGE-OK"), ("B d6 low", stage_b, {"d6": 0.7}, "BRIDGE-FAIL"), ("S1 ok", stage_s1, {}, "S1-OK winner askm_lr3e5"),
             ("S1 fail", stage_s1, {"cls": 0.9}, "S1-FAIL"), ("S2 ok", stage_s2, {}, "SEED-OK"), ("S2 write missing", stage_s2, {"ws2": 0.02}, "NO-SEED"),
             ("S2 mid -> extend", stage_s2, {"ws2": 0.7}, "EXTEND"), ("S2 low but rising -> extend", stage_s2, {"ws2": 0.3, "ws6": 0.1}, "EXTEND"),
             ("S2 extension ok", stage_s2, {"ws2": 0.7, "ext": 0.95}, "SEED-OK (after extension)"), ("S2 extension fails", stage_s2, {"ws2": 0.7, "ext": 0.6}, "NO-SEED (after extension)"),
             ("S3 holds", stage_s3, {}, "MEDIUM-C2-HOLDS"), ("S3 improves", stage_s3, {"d8": 0.75}, "MEDIUM-C2-IMPROVES"),
             ("S3 c1 only", stage_s3, {"d6": 0.5}, "MEDIUM-C1-ONLY"), ("S3 decoder", stage_s3, {"k2": 0.6}, "FAIL"))
    for name, fn, kw, want in cases:
        o = fn(world(**kw)); res.append((name, o["verdict"].startswith(want), o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); ap.add_argument("--stage", default="S3", choices=("B", "S1", "S2", "S3")); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = {"B": stage_b, "S1": stage_s1, "S2": stage_s2, "S3": stage_s3}[a.stage](); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT % a.stage, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
