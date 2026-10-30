"""§248 HALT-HEAD gates (HH_v1 §4 + Note 248-1) -> scratch/gate_s248/<gate>.json (incremental; --redo re-runs).
  G1 (4060; lh_fh_s1 / lh_fh_s2, one process each; dumps only - no head is fit, no halting reading computed):
     hh_dump preds at K' 0..8 == answers_by_kprime[:, 0..8] on TEST2-DF d 1..6 (exact); the K' 8 preds == eval_acc's (host RTX 3090) K' 8 per item at d 1..6 (HH_v1 §4 gate;
     the agreement is reported); Note 248-1 (2): for every K' 0..8 the feature of a K'-truncated rollout (hh_dump K = K') == the full rollout's at K' (np.array_equal) on
     TEST2-DF d 4 and d 6 (first 250 each); Note 248-1 (1): the fit stream (as part_halthead draws it) has 0 sigs in TEST2-DF.
  G2 (4060): the §247 eval module (boxes/_inject_aw_eval_s247.py) vs the new one on parts ws, opclass, cue_only (lh_fh_s1): equal bar sec / module_sha256.
  G3 (4060): full-scale smoke of scratch/s248_hh.py on df_plaw4_s1 (NOT a §248 model) into scratch/_tmp/s248: the fields score_rh --hh reads (d 1..6 + head + never_halt_n,
     per-item lists of 9); score_rh.c5 runs on it (label not stored, no value printed).
    python -B scratch/gate_s248.py [--gates G1,G2,G3] [--redo]"""
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
OUT = os.path.join(ROOT, "scratch", "gate_s248"); TMP = os.path.join(ROOT, "scratch", "_tmp", "s248"); RUNS = os.path.join(ROOT, "boxes", "runs_fsq")

G1_CODE = r'''
import sys, json, numpy as np
sys.path.insert(0, ROOT_)
from boxes import inject_aw_eval as E, inject_ow as IO, inject_aw as AW
import boxes.ow_words as OW
run = RUN_; dev = "cuda:0"; a, m, g = E.load(run, dev); s = IO.build_sets(mul_s_p=E.law(a)); T = E.tokd(s, a, range(1, 7)); o = {}
acc_h = json.load(open(run + "/eval_acc.json")); full = {}
for d in range(1, 7):
    H, P = E.hh_dump(m, T[d], dev, 250); Q = E.answers_by_kprime(m, g, T[d], 8, dev, 250); full[d] = H
    ph = np.array(acc_h["items"][str(d)]["pred"])[:, 8]; assert acc_h["items"][str(d)]["sig"] == [it["sig"] for it in T[d]]
    o["d%d" % d] = {"pred_eq_abk": bool((P == Q[:, :9]).all()), "k8_eq_eval_acc": bool((P[:, 8] == ph).all()), "k8_agree_eval_acc": float((P[:, 8] == ph).mean())}
tr = {}
for d in (4, 6):
    sub = T[d][:250]; r = {}
    for k in range(0, 9):
        Hk, _ = E.hh_dump(m, sub, dev, 250, K=k); r[str(k)] = bool(np.array_equal(Hk[:, k], full[d][:250, k]))
    tr["d%d" % d] = r
o["truncated_eq_full"] = tr
rng = np.random.default_rng(E.SEED_HH_FIT); s_ = OW.Stream(E.SEED_HH_FIT + 1, excl=s["excl"], frames=OW.FRAME_TRAIN, mul_s_p=E.law(a), **IO.OWKW); ds = rng.integers(1, 5, size=E.HH_N)
fit = [s_.draw(int(d)) for d in ds]; t2 = {it["sig"] for d in s["test2"] for it in s["test2"][d]}
o["fit_in_test2"] = sum(it["sig"] in t2 for it in fit); o["fit_in_excl"] = sum(it["sig"] in s["excl"] for it in fit); o["fit_d_counts"] = {str(d): int((ds == d).sum()) for d in range(1, 5)}
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
    ok = lambda v: (all(v["d%d" % d]["pred_eq_abk"] and v["d%d" % d]["k8_eq_eval_acc"] for d in range(1, 7))  # noqa: E731
                    and all(all(x.values()) for x in v["truncated_eq_full"].values()) and v["fit_in_test2"] == 0 and v["fit_in_excl"] == 0)
    o["pass"] = all("d1" in o[r] and ok(o[r]) for r in ("lh_fh_s1", "lh_fh_s2")); return dump("G1", o)


def tmp_run(src, name):
    d = os.path.join(TMP, name); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for f in ("args.json", "train_summary.json"):
        shutil.copy(os.path.join(RUNS, src, f), d)
    os.link(os.path.join(RUNS, src, "model.pt"), os.path.join(d, "model.pt")); return d


def g2():
    parts = "ws,opclass,cue_only"; res = {}
    for mod, name in (("boxes._inject_aw_eval_s247", "g2_old"), ("boxes.inject_aw_eval", "g2_new")):
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
    rc = subprocess.run([sys.executable, "-B", "scratch/s248_hh.py", "--run", "df_plaw4_s1", "--src_dir", d, "--out", d, "--eval_bs", "250", "--device", "cuda:0"]).returncode
    os.remove(os.path.join(d, "model.pt")); o = {"rc": rc}; f = os.path.join(d, "eval_halthead.json")
    if rc == 0 and os.path.exists(f):
        h = json.load(open(f))
        o["d_fields_ok"] = all(all(k in h[str(dd)] for k in ("acc_k8", "acc_halt", "mean_khalt", "khalt_hist", "never_halt_n")) and len(h[str(dd)]["items"]["p"][0]) == 9
                               and h[str(dd)]["n"] == 1000 for dd in range(1, 7))
        o["head_keys"] = sorted(h["head"]); o["head_ok"] = all(k in h["head"] for k in ("train_acc", "val_acc", "val_auc")) and h["head"]["val_rows"] == 1200 * 9
        o["meta"] = {k: h.get(k) for k in ("eval_gpu", "eval_bs", "eval_dtype", "law", "set_law", "think_k", "peak_gib", "sec")}
        sys.path.insert(0, os.path.join(ROOT, "theory")); import score_rh as SR
        o["score_rh_c5_runs"] = SR.c5(h)["label"] in ("COSTLY", "NO-SIGNAL", "NO-SAVING", "C5-READOUT")
        o["pass"] = bool(o["d_fields_ok"] and o["head_ok"] and o["score_rh_c5_runs"])
    else:
        o["pass"] = False
    return dump("G3", o)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gates", default="G1,G2,G3"); ap.add_argument("--redo", action="store_true"); a = ap.parse_args()
    for gname in a.gates.split(","):
        if os.path.exists(os.path.join(OUT, gname + ".json")) and not a.redo:
            print(gname, "exists, skipped"); continue
        {"G1": g1, "G2": g2, "G3": g3}[gname]()
