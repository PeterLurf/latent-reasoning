"""§241 LEN-STEP gates (LS_v1 §2 + Note 241-1) -> scratch/gate_s241/<gate>.json (incremental; a gate with a json is skipped unless --redo).
  G0 (CPU, no model): cells build for lh_fh_s1 / lh_fh_s2 args: identical cells for both; deterministic (two builds); re-render identity (e = 0 render of every A
     item == the item; drop render of every n_dis-0 C-pool item == the item); program identity asserted inside rerender_dis (+ the independent solver); the B / C
     length gate; D count; A / C overlap with TEST2-DF and the DF training exclusion (counted).
  G1 (4060, separate processes): flags-off bitwise - parts ws, opclass, cue_only on lh_fh_s1 with the HEAD module (boxes/_inject_aw_eval_s240.py) and the new
     module, outputs equal bar sec / module_sha256.
  G2 (4060): passes(K' 8) == answers_by_kprime[:, 8] on TEST2-DF d 4 (first 250) of lh_fh_s1.
  G3 (4060): full-scale smoke of scratch/s241_lenstep.py on df_plaw4_s1 (NOT a §241 model) into scratch/_tmp/s241_smoke: field paths score_ls reads + build fields;
     no accuracy value is printed.
    python -B scratch/gate_s241.py [--gates G0,G1,G2,G3] [--redo]"""
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
OUT = os.path.join(ROOT, "scratch", "gate_s241"); TMP = os.path.join(ROOT, "scratch", "_tmp", "s241")
RUNS = os.path.join(ROOT, "boxes", "runs_fsq")


def dump(name, o):
    os.makedirs(OUT, exist_ok=True); json.dump(o, open(os.path.join(OUT, name + ".json"), "w", newline="\n"), indent=1); print(name, "PASS" if o.get("pass") else "FAIL", flush=True)
    return o


def g0():
    import numpy as np
    from boxes import inject_aw_eval as E
    from boxes import inject_ow as IO
    a1 = json.load(open(os.path.join(RUNS, "lh_fh_s1", "args.json"))); a2 = json.load(open(os.path.join(RUNS, "lh_fh_s2", "args.json")))
    c1, i1 = E.lenstep_cells(a1); c1b, _ = E.lenstep_cells(a1); c2, i2 = E.lenstep_cells(a2)
    sig = lambda c: {k: [it["sig"] for it in v[0]] + v[1] for k, v in c.items()}
    det, same = sig(c1) == sig(c1b), (sig(c1) == sig(c2) and i1 == i2)
    A = c1["A"][0]; kw = dict(frames=E.OW.FRAME_TRAIN, mul_s_p=2, **IO.OWKW)
    rng = np.random.default_rng(0); id0 = all(E.rerender_dis(rng, it, 0) == it for it in A)
    sC = E.OW.Stream(E.SEED_LS_C, **kw); Cpool = [sC.draw(6) for _ in range(E.LS_DPOOL)]; z = [it for it in Cpool if it["n_distractors"] == 0]
    idd = all(E.rerender_dis(None, it, drop=True) == it for it in z)
    B = c1["B"][0]; e = i1["e"]; prog_same = all(b["values"] == x["values"] and b["ops"] == x["ops"] and b["src_j"] == x["src_j"] and b["answer"] == x["answer"]
                                                and b["n_distractors"] == x["n_distractors"] + e for b, x in zip(B, A))
    sets = IO.build_sets(mul_s_p=2); t2 = {it["sig"] for d in sets["test2"] for it in sets["test2"][d]}; ex = sets["excl"]
    ov = {k: {"test2": sum(s_ in t2 for s_ in v[1]), "excl": sum(s_ in ex for s_ in v[1])} for k, v in c1.items()}
    lens = {k: (round(float(np.mean([E.ls_len(a1, it) for it in v[0]])), 2), round(float(np.percentile([E.ls_len(a1, it) for it in v[0]], 90)), 2), len(v[0])) for k, v in c1.items()}
    gate = abs(lens["B"][0] - lens["C"][0]) <= 5 and abs(lens["B"][1] - lens["C"][1]) <= 10
    o = {"deterministic": det, "same_cells_s1_s2": same, "rerender_e0_identity": id0, "rerender_drop_identity_ndis0": idd, "n_ndis0_checked": len(z),
         "B_same_programs_plus_e": prog_same, "length_gate": gate, "len_mean_p90_n": lens, "build": i1, "overlap_prog_sigs": ov,
         "pass": bool(det and same and id0 and idd and prog_same and gate and len(z) > 0)}
    return dump("G0", o)


def tmp_run(src, name):
    d = os.path.join(TMP, name); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for f in ("args.json", "train_summary.json"):
        shutil.copy(os.path.join(RUNS, src, f), d)
    os.link(os.path.join(RUNS, src, "model.pt"), os.path.join(d, "model.pt")); return d


def g1():
    parts = "ws,opclass,cue_only"; res = {}
    for mod, name in (("boxes._inject_aw_eval_s240", "g1_old"), ("boxes.inject_aw_eval", "g1_new")):
        d = tmp_run("lh_fh_s1", name)
        rc = subprocess.run([sys.executable, "-B", "-m", mod, "--run", d, "--parts", parts, "--eval_bs", "250", "--device", "cuda:0"]).returncode
        res[name] = (d, rc)
    eq = {}
    for p_ in parts.split(","):
        x = [json.load(open(os.path.join(res[n][0], "eval_%s.json" % p_))) for n in ("g1_old", "g1_new")]
        for y in x:
            y.pop("sec", None); y.pop("module_sha256", None)
        eq[p_] = x[0] == x[1]
    o = {"rc": {k: v[1] for k, v in res.items()}, "equal": eq, "pass": all(v[1] == 0 for v in res.values()) and all(eq.values())}
    for n in res:
        os.remove(os.path.join(res[n][0], "model.pt"))
    return dump("G1", o)


def g2():
    code = ("import sys, json, numpy as np; sys.path.insert(0, %r); from boxes import inject_aw_eval as E, inject_ow as IO; a, m, g = E.load(%r, 'cuda:0'); "
            "s = IO.build_sets(mul_s_p=2); T = E.tokd(s, a, [4])[4][:250]; P, _ = E.passes(m, T, 'cuda:0', 250, {8}); Q = E.answers_by_kprime(m, g, T, 8, 'cuda:0', 250); "
            "print(json.dumps({'n': len(T), 'equal': bool((P[8] == Q[:, 8]).all())}))") % (ROOT, os.path.join(RUNS, "lh_fh_s1"))
    r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True); last = (r.stdout.strip().splitlines() or ["{}"])[-1]
    try:
        v = json.loads(last)
    except ValueError:
        v = {"stderr": r.stderr[-2000:]}
    return dump("G2", {**v, "rc": r.returncode, "pass": r.returncode == 0 and v.get("equal") is True})


def g3():
    d = tmp_run("df_plaw4_s1", "g3_smoke")
    rc = subprocess.run([sys.executable, "-B", "scratch/s241_lenstep.py", "--run", "df_plaw4_s1", "--src_dir", d, "--out", d, "--eval_bs", "250", "--device", "cuda:0"]).returncode
    os.remove(os.path.join(d, "model.pt")); f = os.path.join(d, "eval_lenstep.json"); o = {"rc": rc}
    if rc == 0 and os.path.exists(f):
        e = json.load(open(f)); c = e["cells"]; need = ("n", "tok_mean", "tok_p90", "acc", "ws", "items")
        o["cells"] = sorted(c); o["fields_ok"] = all(all(k in c[x] for k in need) for x in c)
        o["items_ok"] = all(len(c[x]["items"]["prog_sig"]) == len(c[x]["items"]["ok"]) == c[x]["n"] and set(c[x]["items"]["ok"]) <= {0, 1} for x in c)
        o["ws_passes"] = {x: sorted(c[x]["ws"], key=int) for x in c}; o["AB_pairs"] = len(set(c["A"]["items"]["prog_sig"]) & set(c["B"]["items"]["prog_sig"]))
        o["DC_pairs"] = len(set(c["D"]["items"]["prog_sig"]) & set(c["C"]["items"]["prog_sig"])) if "D" in c else None
        o["meta"] = {k: e.get(k) for k in ("eval_gpu", "eval_bs", "eval_dtype", "law", "think_k", "peak_gib", "sec")}
        sys.path.insert(0, os.path.join(ROOT, "theory")); import score_ls as SL; lab = SL.one(e)["label"]
        o["score_ls_one_runs"] = lab in ("LENGTH", "STEPS", "BOTH", "NEITHER")   # the label itself is not stored (df_plaw4_s1 is not a §241 model)
        o["pass"] = bool(o["fields_ok"] and o["items_ok"] and o["AB_pairs"] == c["A"]["n"] and o["score_ls_one_runs"])
    else:
        o["pass"] = False
    return dump("G3", o)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gates", default="G0,G1,G2,G3"); ap.add_argument("--redo", action="store_true"); a = ap.parse_args()
    for gname in a.gates.split(","):
        if os.path.exists(os.path.join(OUT, gname + ".json")) and not a.redo:
            print(gname, "exists, skipped"); continue
        {"G0": g0, "G1": g1, "G2": g2, "G3": g3}[gname]()
