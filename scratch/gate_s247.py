"""§247 READ-HALT gates (RH_v1 §3 + Note 247-1) -> scratch/gate_s247/<gate>.json (incremental; --redo re-runs).
  G1 (4060; lh_fh_s1 / lh_fh_s2, one process each; intact / path checks only - no registered re-encode / edit reading is computed):
     rh_run intact at d 3 / d 4 (K' = d) == answers_by_kprime[:, d] and its WS argmax == ws_readout per item (exact); the EMPTY augmentation at each cell's t (rh_input(it, t, ''))
     == intact per item (path check, RH_v1 §3); rh_input(it, t, '') == think_cues[t - 1] for every item / pass; the halt runner's preds at K' 0..8 (d 1..6) == answers_by_kprime[:, 0..8]
     (exact); rh_answer(raw, t, gold v_t) == the generator's answer for every TEST2-DF item d 1..6 and every t <= d.
  G2 (4060): the §243 eval module (boxes/_inject_aw_eval_s243.py) vs the new one on parts ws, opclass, cue_only (lh_fh_s1): equal bar sec / module_sha256.
  G3 (4060): full-scale smoke of scratch/s247_rh.py on df_plaw4_s1 (NOT a §247 model) into scratch/_tmp/s247: the fields score_rh reads (+ never_halt_n, per-item lists);
     score_rh.c7 / c5 run (labels not stored, no value printed).
    python -B scratch/gate_s247.py [--gates G1,G2,G3] [--redo]"""
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
OUT = os.path.join(ROOT, "scratch", "gate_s247"); TMP = os.path.join(ROOT, "scratch", "_tmp", "s247"); RUNS = os.path.join(ROOT, "boxes", "runs_fsq")

G1_CODE = r'''
import sys, json, numpy as np
sys.path.insert(0, ROOT_)
from boxes import inject_aw_eval as E, inject_ow as IO, inject_aw as AW
run = RUN_; dev = "cuda:0"; a, m, g = E.load(run, dev); s = IO.build_sets(mul_s_p=E.law(a)); T = E.tokd(s, a, range(1, 7)); o = {}
o["rh_input_empty_eq_cues"] = all(E.rh_input(it, t, "") == it["think_cues"][t - 1] for d in (3, 4) for it in T[d] for t in range(1, d + 1))
o["rh_answer_gold_ok"] = all(E.rh_answer(it["raw"], t, it["raw"]["values"][t]) == it["ans"] for d in range(1, 7) for it in T[d] for t in range(1, d + 1))
for d in (3, 4):
    ai, wi, _ = E.rh_run(m, T[d], dev, 250, d, lambda i, it, t: it["think_cues"][t - 1]); Q = E.answers_by_kprime(m, g, T[d], d, dev, 250)
    _, pv = AW.ws_readout(m, {d: T[d]}, dev, d, 250, per_item=True)
    rec = {"ans_eq_abk": bool((ai == Q[:, d]).all()), "ws_eq_readout": all(wi[t].tolist() == pv[str(d)][str(t)] for t in range(1, d + 1))}
    for name, dd, t in E.RH_CELLS:
        if dd == d:
            ae = E.rh_run(m, T[d], dev, 250, d, lambda i, it, tt, t=t: E.rh_input(it, tt, "") if tt == t else it["think_cues"][tt - 1])[0]
            rec["empty_aug_eq_intact_" + name] = bool((ae == ai).all())
    o["d%d" % d] = rec
for d in range(1, 7):
    _, _, P = E.rh_run(m, T[d], dev, 250, 8, lambda i, it, t: E.seq_std(it, t), probs=True); Q = E.answers_by_kprime(m, g, T[d], 8, dev, 250)
    o["halt_pred_eq_abk_d%d" % d] = bool((P.argmax(-1) == Q[:, :9]).all())
print("G1JSON " + json.dumps(o))
'''


def dump(name, o):
    os.makedirs(OUT, exist_ok=True); json.dump(o, open(os.path.join(OUT, name + ".json"), "w", newline="\n"), indent=1); print(name, "PASS" if o.get("pass") else "FAIL", flush=True)
    return o


def g1():
    o = {}
    for r in ("lh_fh_s1", "lh_fh_s2"):
        code = G1_CODE.replace("ROOT_", repr(ROOT)).replace("RUN_", repr(os.path.join(RUNS, r)))
        p = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True); ln = [x for x in p.stdout.splitlines() if x.startswith("G1JSON ")]
        o[r] = json.loads(ln[-1][7:]) if ln else {"rc": p.returncode, "stderr": p.stderr[-2000:]}
    ok = lambda v: (v.get("rh_input_empty_eq_cues") and v.get("rh_answer_gold_ok") and all(all(x for x in v["d%d" % d].values()) for d in (3, 4))  # noqa: E731
                    and all(v.get("halt_pred_eq_abk_d%d" % d) for d in range(1, 7)))
    o["pass"] = all("d3" in o[r] and ok(o[r]) for r in ("lh_fh_s1", "lh_fh_s2")); return dump("G1", o)


def tmp_run(src, name):
    d = os.path.join(TMP, name); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for f in ("args.json", "train_summary.json"):
        shutil.copy(os.path.join(RUNS, src, f), d)
    os.link(os.path.join(RUNS, src, "model.pt"), os.path.join(d, "model.pt")); return d


def g2():
    parts = "ws,opclass,cue_only"; res = {}
    for mod, name in (("boxes._inject_aw_eval_s243", "g2_old"), ("boxes.inject_aw_eval", "g2_new")):
        d = tmp_run("lh_fh_s1", name)
        res[name] = (d, subprocess.run([sys.executable, "-B", "-m", mod, "--run", d, "--parts", parts, "--eval_bs", "250", "--device", "cuda:0"]).returncode)
    eq = {}
    for p_ in parts.split(","):
        x = [json.load(open(os.path.join(res[n][0], "eval_%s.json" % p_))) for n in ("g2_old", "g2_new")]
        for y in x:
            y.pop("sec", None); y.pop("module_sha256", None)
        eq[p_] = x[0] == x[1]
    for n in res:
        os.remove(os.path.join(res[n][0], "model.pt"))
    return dump("G2", {"rc": {k: v[1] for k, v in res.items()}, "equal": eq, "pass": all(v[1] == 0 for v in res.values()) and all(eq.values())})


def g3():
    d = tmp_run("df_plaw4_s1", "g3_smoke")
    rc = subprocess.run([sys.executable, "-B", "scratch/s247_rh.py", "--run", "df_plaw4_s1", "--src_dir", d, "--out", d, "--eval_bs", "250", "--device", "cuda:0"]).returncode
    os.remove(os.path.join(d, "model.pt")); o = {"rc": rc}
    fc, fh = os.path.join(d, "eval_c7.json"), os.path.join(d, "eval_halt.json")
    if rc == 0 and os.path.exists(fc) and os.path.exists(fh):
        c, h = json.load(open(fc)), json.load(open(fh)); cells = ("d3t1", "d3t2", "d4t2", "d4t3")
        need_c = ("n", "intact_acc", "rt_agree", "edit_follow", "floor", "ctrl_aug_all", "oracle_rt"); need_ci = ("sig", "x_hat", "x_edit", "a_edit", "ans_intact", "ans_rt", "ans_edit")
        o["c7_fields_ok"] = all(all(k in c[x] for k in need_c) and all(len(c[x]["items"][k]) == c[x]["n"] for k in need_ci) for x in cells)
        o["c7_n"] = {x: c[x]["n"] for x in cells}; o["x_edit_ne_x_hat"] = all(a != b for x in cells for a, b in zip(c[x]["items"]["x_edit"], c[x]["items"]["x_hat"]))
        o["halt_fields_ok"] = all(all(k in h[str(dd)] for k in ("acc_k8", "acc_halt", "mean_khalt", "khalt_hist", "never_halt_n")) and len(h[str(dd)]["items"]["maxprob"][0]) == 9
                                  for dd in range(1, 7))
        o["meta"] = {k: c.get(k) for k in ("eval_gpu", "eval_bs", "eval_dtype", "law", "set_law", "think_k", "peak_gib", "sec")}; o["halt_sec"] = h.get("sec")
        sys.path.insert(0, os.path.join(ROOT, "theory")); import score_rh as SR
        o["score_rh_runs"] = SR.c7(c)["label"] in ("C7-PASS", "ROUND-TRIP-ONLY", "REJECTED", "VOID") and SR.c5(h)["label"] in ("COSTLY", "NO-SIGNAL", "NO-SAVING", "C5-READOUT")
        o["pass"] = bool(o["c7_fields_ok"] and o["x_edit_ne_x_hat"] and o["halt_fields_ok"] and o["score_rh_runs"] and all(v == 1000 for v in o["c7_n"].values()))
    else:
        o["pass"] = False
    return dump("G3", o)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gates", default="G1,G2,G3"); ap.add_argument("--redo", action="store_true"); a = ap.parse_args()
    for gname in a.gates.split(","):
        if os.path.exists(os.path.join(OUT, gname + ".json")) and not a.redo:
            print(gname, "exists, skipped"); continue
        {"G1": g1, "G2": g2, "G3": g3}[gname]()
