"""THEORY §234 ANSWER-WRITE P-4 scorer (Note 234-5 + its amendment; field paths per Experiments' P-4 layout (5357efd0); Theory is the only scorer).
Runs: LAT-AW aw_plaw4_s1, CD-AN aw_pcd4_s1 (same RTX 3090); plain DIRECT s228_pdir2c_s1 (CUE-ONLY bar only).
Binding (each read on TEST2):
  LP0 (each arm; score_st.lp0 on resume-deduplicated logs).
  LP4: LAT d = 3 >= 0.60 and d = 4 >= 0.50, each >= CD-AN + 0.25.
  CHAIN: eval_ws by_d[d][t] >= 0.50 at every pass t = 2..d, for d = 3 and 4 (WS reads v_min(t, d)).
  LP5-4: eval_lp5h, item-weighted pool (items[d].sig counts) over d 3-4 of intact[str(d)] - mean_at_t[str(d)] >= 0.30.
  CUE-ONLY guard on d 3-4: answer_acc <= max(majority, plain DIRECT d) + 0.05 at each d, else VOID.
P-4 PASS = LP0 (both) & LP4 & CHAIN & LP5-4 & not VOID.
Descriptive: MEAN-AT-1 (eval_lp5 d 3-4), CUE-WRONG drop per d (C1-TOLD vs plain C1 label on a pass), SWAP-K / SWAP-MID (eval_swap_kd; d 3 at t 1, d 4 at t 2).
    python -B theory/score_aw4.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory"))
import score_st as ST  # noqa: E402

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_aw4.json")
LAT, CD, DIR = "aw_plaw4_s1", "aw_pcd4_s1", "s228_pdir2c_s1"
DS = ("3", "4")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def score(runs=R):
    o = {"section": 234, "stage": "P-4", "registered": "Note 234-5 (+ amendment)"}
    need = {("lat", f): jl(os.path.join(runs, LAT, f + ".json")) for f in ("eval_acc", "eval_ws", "eval_lp5h", "eval_lp5", "eval_cue_only", "eval_cue_wrong", "eval_swap_kd")}
    need[("cd", "eval_acc")] = jl(os.path.join(runs, CD, "eval_acc.json")); need[("dir", "eval_acc")] = jl(os.path.join(runs, DIR, "eval_acc.json"))
    lgl, lgc = ST.dedupe(ST.rows(os.path.join(runs, LAT, "log.jsonl"))), ST.dedupe(ST.rows(os.path.join(runs, CD, "log.jsonl")))
    miss = ["%s/%s" % k for k, v in need.items() if v is None] + [n for n, v in (("lat log", lgl), ("cd log", lgc)) if v is None]
    if miss:
        o["verdict"] = "pending (missing %s)" % miss; return o
    A = lambda k, d: need[(k, "eval_acc")]["by_d"][d]["acc"]  # noqa: E731
    l0a, l0c = ST.lp0(lgl), ST.lp0(lgc)
    lp4 = A("lat", "3") >= 0.60 and A("lat", "4") >= 0.50 and A("lat", "3") >= A("cd", "3") + 0.25 and A("lat", "4") >= A("cd", "4") + 0.25
    ws = need[("lat", "eval_ws")]["by_d"]
    chain_rows = {d: {str(t): ws[d][str(t)] for t in range(2, int(d) + 1)} for d in DS}
    chain = all(v >= 0.50 for d in DS for v in chain_rows[d].values())
    h = need[("lat", "eval_lp5h")]["by_d"]; drops = {d: h[d]["intact"][d] - h[d]["mean_at_t"][d] for d in DS}
    nh = {d: len(((need[("lat", "eval_lp5h")].get("items") or {}).get(d) or {}).get("sig") or []) or 1 for d in DS}  # item-weighted pool
    pooled = sum(drops[d] * nh[d] for d in DS) / sum(nh.values()); lp54 = pooled >= 0.30
    co = need[("lat", "eval_cue_only")]["by_d"]; guard = all(co[d]["answer_acc"] <= max(co[d]["majority"], A("dir", d)) + 0.05 for d in DS)
    l5 = need[("lat", "eval_lp5")]["by_d"]; cw = need[("lat", "eval_cue_wrong")]["by_d"]
    cwd = {d: l5[d]["intact"][d] - cw[d]["all"][d] for d in DS}
    sw = need[("lat", "eval_swap_kd")]["by_d"]
    o.update({"LP0": {"lat": l0a, "cd": l0c}, "LP4": lp4, "acc": {d: {"lat": A("lat", d), "cd": A("cd", d), "dir": A("dir", d)} for d in ("2", "3", "4")},
              "CHAIN": chain, "ws_rows": chain_rows, "LP5_4": lp54, "lp5h_drops": drops, "lp5h_pooled": pooled, "lp5h_n": nh, "CUE_ONLY_ok": guard,
              "MEAN_AT_1_desc": {d: l5[d]["intact"][d] - l5[d]["mean_at_1"][d] for d in DS}, "CUE_WRONG_drop": cwd,
              "swap_desc": {d: {"t_mid": sw[d].get("t_mid"), "n_pairs": sw[d].get("n_pairs"), "swap_k": sw[d].get("swap_k"), "swap_mid": sw[d].get("swap_mid")} for d in DS}})
    if not guard:
        o["verdict"] = "VOID (CUE-ONLY guard on d 3-4)"; return o
    ok = l0a[0] and l0c[0] and lp4 and chain and lp54
    fails = [k for k, v in (("LP0-lat", l0a[0]), ("LP0-cd", l0c[0]), ("LP4", lp4), ("CHAIN", chain), ("LP5-4", lp54)) if not v]
    lab = "C1-TOLD" if min(cwd.values()) >= 0.20 else "plain C1 (cue unused at some d)"
    o["verdict"] = ("P-4 PASS (%s)" % lab) if ok else ("P-4 FAIL (%s)" % ",".join(fails))
    return o


def fixtures():
    res = []
    def world(lat3=0.9, lat4=0.85, cd=0.1, ws=0.95, mean_t=0.2, co=0.1, cwd=0.5):
        d = tempfile.mkdtemp()
        for run, accs in ((LAT, {"2": 1.0, "3": lat3, "4": lat4}), (CD, {"2": cd, "3": cd, "4": cd}), (DIR, {"2": 0.1, "3": 0.1, "4": 0.1})):
            p = os.path.join(d, run); os.makedirs(p)
            json.dump({"by_d": {k: {"acc": v} for k, v in accs.items()}}, open(os.path.join(p, "eval_acc.json"), "w"))
            with open(os.path.join(p, "log.jsonl"), "w") as fh:
                for i in range(1, 161):
                    fh.write(json.dumps({"step": i * 100, "gn_preclip_mean100": 1.0 + 0.1 * (i % 3)}) + "\n")
        p = os.path.join(d, LAT)
        json.dump({"by_d": {dd: {str(t): ws for t in range(1, 9)} for dd in ("1", "2", "3", "4")}}, open(os.path.join(p, "eval_ws.json"), "w"))
        json.dump({"by_d": {dd: {"intact": {dd: 0.9}, "mean_at_t": {dd: mean_t}} for dd in DS}}, open(os.path.join(p, "eval_lp5h.json"), "w"))
        json.dump({"by_d": {dd: {"intact": {dd: 0.9}, "mean_at_1": {dd: 0.1}} for dd in ("2",) + DS}}, open(os.path.join(p, "eval_lp5.json"), "w"))
        json.dump({"by_d": {dd: {"answer_acc": co, "majority": 0.1} for dd in ("2",) + DS}}, open(os.path.join(p, "eval_cue_only.json"), "w"))
        json.dump({"by_d": {dd: {"all": {dd: 0.9 - cwd}} for dd in ("2",) + DS}}, open(os.path.join(p, "eval_cue_wrong.json"), "w"))
        json.dump({"by_d": {dd: {"t_mid": 1 if dd == "3" else 2, "n_pairs": 1000, "swap_k": {}, "swap_mid": {}} for dd in DS}}, open(os.path.join(p, "eval_swap_kd.json"), "w"))
        return d
    o = score(world()); res.append(("chaining stub -> P-4 PASS (C1-TOLD)", o["verdict"] == "P-4 PASS (C1-TOLD)", o["verdict"]))
    o = score(world(lat3=0.5)); res.append(("d3 0.5 -> FAIL LP4", "LP4" in o["verdict"], o["verdict"]))
    o = score(world(ws=0.3)); res.append(("WS 0.3 at later passes -> FAIL CHAIN", "CHAIN" in o["verdict"], o["verdict"]))
    o = score(world(mean_t=0.8)); res.append(("mid-state mean harmless -> FAIL LP5-4", "LP5-4" in o["verdict"], o["verdict"]))
    o = score(world(co=0.5)); res.append(("cue alone answers -> VOID", o["verdict"].startswith("VOID"), o["verdict"]))
    o = score(world(cwd=0.05)); res.append(("cue unused -> PASS plain C1", "plain C1" in o["verdict"], o["verdict"]))
    d = world(); os.remove(os.path.join(d, LAT, "eval_lp5h.json")); o = score(d); res.append(("missing -> pending", o["verdict"].startswith("pending"), o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", newline=chr(10)), indent=1, default=str)
