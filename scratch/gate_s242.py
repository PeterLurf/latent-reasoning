"""§242 NORM-COUNTER gates (NC_v1 §4 + Note 242-1) -> scratch/gate_s242/<gate>.json (incremental; --redo re-runs).
  G1 (4060; lh_fh_s1 / lh_fh_s2, one process each; only intact / identity arms - no registered intervention arm is run): flags-off - nc_run intact d 4 / d 6 answers ==
     answers_by_kprime[:, 8] and WS argmax == ws_readout per item (same card, exact); identity_d6 (scale 1.0, same path) == intact_d6 per item (exact); vs the host
     eval_acc (RTX 3090) at K' 8, d 4 / d 6: per-item agreement >= 0.99 and |acc diff| <= 0.005 (cross-card fp32).
  G2 (4060): old module (boxes/_inject_aw_eval_s241.py) vs new on parts ws, opclass, cue_only (lh_fh_s1): outputs equal bar sec / module_sha256.
  G3 (4060): full-scale smoke of scratch/s242_normctr.py on df_plaw4_s1 (NOT a §242 model) into scratch/_tmp/s242: the fields score_nc reads, identity_equal_items,
     post_relerr < 1e-4 on every scaled arm, mean_scale; score_nc.one runs (label not stored, no accuracy printed).
    python -B scratch/gate_s242.py [--gates G1,G2,G3] [--redo]"""
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
OUT = os.path.join(ROOT, "scratch", "gate_s242"); TMP = os.path.join(ROOT, "scratch", "_tmp", "s242"); RUNS = os.path.join(ROOT, "boxes", "runs_fsq")

G1_CODE = r'''
import sys, json, numpy as np
sys.path.insert(0, ROOT_)
from boxes import inject_aw_eval as E, inject_ow as IO, inject_aw as AW
run = RUN_; dev = "cuda:0"; a, m, g = E.load(run, dev); s = IO.build_sets(mul_s_p=E.law(a)); T = E.tokd(s, a, [4, 6]); o = {}
acc_h = json.load(open(run + "/eval_acc.json"))
for d in (4, 6):
    r = E.nc_run(m, T[d], dev, 250, 8); Q = E.answers_by_kprime(m, g, T[d], 8, dev, 250); _, pv = AW.ws_readout(m, {d: T[d]}, dev, 8, 250, per_item=True)
    y = np.array([it["ans"] for it in T[d]]); ph = np.array(acc_h["items"][str(d)]["pred"])[:, 8]
    assert acc_h["items"][str(d)]["sig"] == [it["sig"] for it in T[d]], "eval_acc item order"
    rec = {"ans_eq_abk": bool((r["ans"] == Q[:, 8]).all()), "ws_eq_readout": all(r["ws"][t].tolist() == pv[str(d)][str(t)] for t in range(1, 9)),
           "acc_local": float((r["ans"] == y).mean()), "acc_host": acc_h["by_d"][str(d)]["acc_kprime"]["8"], "agree_host": float((r["ans"] == ph).mean())}
    if d == 6:
        ri = E.nc_run(m, T[d], dev, 250, 8, (4, "one", "all"), r["norms"])
        rec["identity_eq"] = bool((ri["ans"] == r["ans"]).all() and all((ri["ws"][t] == r["ws"][t]).all() for t in range(1, 9))); rec["identity_scale_all_one"] = bool((ri["scale"] == 1.0).all())
    o[str(d)] = rec
print("G1JSON " + json.dumps(o))
'''


def dump(name, o):
    os.makedirs(OUT, exist_ok=True); json.dump(o, open(os.path.join(OUT, name + ".json"), "w", newline="\n"), indent=1); print(name, "PASS" if o.get("pass") else "FAIL", flush=True)
    return o


def g1():
    o = {}
    for r in ("lh_fh_s1", "lh_fh_s2"):
        code = G1_CODE.replace("ROOT_", repr(ROOT)).replace("RUN_", repr(os.path.join(RUNS, r)))
        p = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True); line = [x for x in p.stdout.splitlines() if x.startswith("G1JSON ")]
        o[r] = json.loads(line[-1][7:]) if line else {"rc": p.returncode, "stderr": p.stderr[-2000:]}
    ok = lambda v: all(v[d]["ans_eq_abk"] and v[d]["ws_eq_readout"] and v[d]["agree_host"] >= 0.99 and abs(v[d]["acc_local"] - v[d]["acc_host"]) <= 0.005 for d in ("4", "6")) \
        and v["6"]["identity_eq"] and v["6"]["identity_scale_all_one"]
    o["pass"] = all("4" in o[r] and ok(o[r]) for r in ("lh_fh_s1", "lh_fh_s2")); return dump("G1", o)


def tmp_run(src, name):
    d = os.path.join(TMP, name); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for f in ("args.json", "train_summary.json"):
        shutil.copy(os.path.join(RUNS, src, f), d)
    os.link(os.path.join(RUNS, src, "model.pt"), os.path.join(d, "model.pt")); return d


def g2():
    parts = "ws,opclass,cue_only"; res = {}
    for mod, name in (("boxes._inject_aw_eval_s241", "g2_old"), ("boxes.inject_aw_eval", "g2_new")):
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
    rc = subprocess.run([sys.executable, "-B", "scratch/s242_normctr.py", "--run", "df_plaw4_s1", "--src_dir", d, "--out", d, "--eval_bs", "250", "--device", "cuda:0"]).returncode
    os.remove(os.path.join(d, "model.pt")); f = os.path.join(d, "eval_normctr.json"); o = {"rc": rc}
    if rc == 0 and os.path.exists(f):
        e = json.load(open(f)); arms = ("intact_d4", "intact_d6", "identity_d6", "down_d6", "down_add_d6", "up_d4", "harm_d4")
        o["arms_present"] = all(k in e for k in arms); o["fields_ok"] = all(all(k in e[x] for k in ("acc", "ws", "n", "mean_scale")) and e[x]["n"] == 1000 for x in arms)
        o["ws_has_5"] = all("5" in e[x]["ws"] for x in arms); o["identity_equal_items"] = e["identity_equal_items"]
        o["identity_acc_ws_equal"] = e["identity_d6"]["acc"] == e["intact_d6"]["acc"] and e["identity_d6"]["ws"] == e["intact_d6"]["ws"]
        o["post_relerr"] = {x: e[x]["post_relerr"] for x in arms if e[x]["edit"] and e[x]["edit"][1] != "one"}
        o["head_norms_shape"] = {dd: sorted(e["head_norms"][dd], key=int) for dd in e["head_norms"]}
        o["meta"] = {k: e.get(k) for k in ("eval_gpu", "eval_bs", "eval_dtype", "law", "set_law", "think_k", "peak_gib", "sec")}
        sys.path.insert(0, os.path.join(ROOT, "theory")); import score_nc as SN; lab = SN.one(e)["label"]
        o["score_nc_one_runs"] = lab in ("NORM-COUNTER", "PARTIAL-DOWN", "PARTIAL-UP", "NOT-NORM", "INCONCLUSIVE")
        o["pass"] = bool(o["arms_present"] and o["fields_ok"] and o["ws_has_5"] and o["identity_equal_items"] and o["identity_acc_ws_equal"]
                         and all(v < 1e-4 for v in o["post_relerr"].values()) and o["score_nc_one_runs"] and set(o["head_norms_shape"]) == {"4", "6"})
    else:
        o["pass"] = False
    return dump("G3", o)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gates", default="G1,G2,G3"); ap.add_argument("--redo", action="store_true"); a = ap.parse_args()
    for gname in a.gates.split(","):
        if os.path.exists(os.path.join(OUT, gname + ".json")) and not a.redo:
            print(gname, "exists, skipped"); continue
        {"G1": g1, "G2": g2, "G3": g3}[gname]()
