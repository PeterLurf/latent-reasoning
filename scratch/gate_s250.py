"""§250 FRAME-DIV Stage 0 build gates (FD_v1 §2-3) -> scratch/gate_s250/<gate>.json (incremental; --redo re-runs). The pre-edit stack is a git worktree of HEAD
(scratch/_tmp/wt_s250; the commit before any §250 module edit) with prosqa/ckpt/gpt2_pretrained.bin hard-linked; model files are passed as absolute paths.
  G1  (a) every pre-edit set sha unchanged: scratch/gate_s250/G1_before.json (taken BEFORE the edit) vs a fresh snapshot (build_sets 0/1/2 every old key incl. transfer
      195f6756, test2, pairs2, floors2, s226, s227, set_law; NW / OW test_set and pair_candidates); excl grows by exactly the 32,000 frame-set sigs per law.
      (b) training-stream identity: OW.Stream(SEED_TRAIN, build_sets(law)['excl'], FRAME_TRAIN, law) 20,000 draws (d cycling 1..4), sigs, old stack vs new, laws 0 / 1 / 2.
      (c) full-stack trainbit: the worktree (old inject_aw / inject_ow / ow_words / nested_words) vs the working tree, det wrap, fp32, micro 8, sched 0.02, stop_at 30, at the
      §240 FH flags (init df_plaw4_s1, hold_inc 1.6) and at plain DF (no hold): ckpt model / opt / rngs / stream equal, logs equal bar sec.
  G2  FD_v1 §1: FRAMES[op][8:12] / [12:16] == fd_v1_templates.json; add_s / mul_s aliased; the novelty gate (max slot-masked trigram Jaccard of each held-out template vs the
      same op's frames 0-11 <= 0.20, the check script's mask); OW.solve == answer on 5,400 items (frames (8-11), (12-15), (0-7); laws 1 / 2; d 1-6 x 150; seeds 777 + law);
      NW.selfcheck().
  G3  per-frame test sets: frameset<f>_sha present for f 8..15 (laws 0 / 1 / 2), every item's frames == {f}, 1,000 per d 1..4, all sigs in excl, disjoint from test2 / transfer.
  G4  pre-declared sha change: --args_only at the §240 FH flags vs scratch/s240_args_frozen_lh_fh_s1.json -> the diff (module / nested_words / inject_ow / eval shas, the new
      frameset shas, excl_n, train_frames ''); python processes at gate time (none may import nested_words besides this gate).
  G5  (4060) the §249 eval module (boxes/_inject_aw_eval_s249.py) vs the new one on parts ws, opclass, cue_only (lh_fh_s1): equal bar sec / module_sha256.
  G6  (4060) full-scale smoke of scratch/s250_frames.py on df_plaw4_s1 (NOT a Stage-0 model): fields {frame 8..15: {d 1..4: n, acc, ws, wrap_any_share}}.
    python -B scratch/gate_s250.py [--gates G1,G2,G3,G4,G5,G6] [--redo]"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
OUT = ROOT + "/scratch/gate_s250"; TMP = ROOT + "/scratch/_tmp/s250"; WT = ROOT + "/scratch/_tmp/wt_s250"; RUNS = ROOT + "/boxes/runs_fsq"
FH = ("--arm inject --think_k 8 --aux_step 1 --aux_fade 48000,48000 --aux_reweight --fast_add --cache_lower --d_max 4 --steps 8000 --cue step --write_src answer "
      "--init_trunk D:/Latent/boxes/runs_fsq/s233_askname_s1/model.pt --init_sha c00f7ec0bd7977fe --write_mix_start 0 --write_mix_end 0 --mul_s_p 2 "
      "--init_model D:/Latent/boxes/runs_fsq/df_plaw4_s1/model.pt --init_model_sha ccf13a50a717fc2a").split()


def dump(name, o):
    os.makedirs(OUT, exist_ok=True); json.dump(o, open(os.path.join(OUT, name + ".json"), "w", newline="\n"), indent=1, default=str); print(name, "PASS" if o.get("pass") else "FAIL", flush=True)
    return o


def worktree():
    if not os.path.isdir(WT):
        base = subprocess.run(["git", "log", "-1", "--format=%H", "--", "boxes/nested_words.py"], capture_output=True, text=True).stdout.strip()
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        subprocess.run(["git", "worktree", "add", "--detach", WT, head], check=True)
        os.makedirs(WT + "/prosqa/ckpt", exist_ok=True); os.link(ROOT + "/prosqa/ckpt/gpt2_pretrained.bin", WT + "/prosqa/ckpt/gpt2_pretrained.bin")
    return WT


STREAM_CODE = r'''
import sys, json, hashlib, os
os.chdir(WT_); sys.path.insert(0, WT_)
import boxes.inject_ow as IO, boxes.ow_words as OW
o = {}
for law in (0, 1, 2):
    s = IO.build_sets(mul_s_p=law); kw = {} if law == 1 else {"mul_s_p": law}
    st = OW.Stream(OW.SEED_TRAIN, s["excl"], frames=OW.FRAME_TRAIN, **IO.OWKW, **kw); h = hashlib.sha256()
    for i in range(20000):
        h.update(st.draw(1 + i % 4)["sig"].encode())
    o[str(law)] = h.hexdigest()
print("STREAMJSON " + json.dumps(o))
'''


def stream_sigs(root):
    p = subprocess.run([sys.executable, "-B", "-c", STREAM_CODE.replace("WT_", repr(root))], capture_output=True, text=True)
    ln = [x for x in p.stdout.splitlines() if x.startswith("STREAMJSON ")]; return json.loads(ln[-1][11:]) if ln else {"err": p.stderr[-1500:]}


def g1():
    import gate_s228 as G8
    subprocess.run([sys.executable, "-B", "scratch/s250_build/sha_snapshot.py", "after"], check=True)
    b, a = json.load(open(OUT + "/G1_before.json")), json.load(open(OUT + "/G1_after.json")); o = {"sets": {}}
    for law in ("0", "1", "2"):
        bs, as_ = b["build_sets"][law]["shas"], a["build_sets"][law]["shas"]
        o["sets"][law] = {"old_keys_equal": all(as_.get(k) == v for k, v in bs.items()), "new_keys": sorted(set(as_) - set(bs)),
                          "excl_growth": a["build_sets"][law]["excl_n"] - b["build_sets"][law]["excl_n"]}
    o["nw_ow_sets_equal"] = all(a[k] == b[k] for k in ("NW_test_set", "NW_pair_candidates", "OW_test_set", "OW_pair_candidates"))
    wt = worktree(); old, new = stream_sigs(wt), stream_sigs(ROOT); o["stream_old"], o["stream_new"] = old, new; o["stream_equal"] = old == new and "err" not in old
    tb = {}
    for tag, ex in (("fh", ["--hold_inc", "1.6"]), ("plain", [])):
        com = FH + ex + ["--fp32", "--micro", "8", "--sched_scale", "0.02", "--ckpt_every", "30", "--log_every", "10", "--stop_at", "30", "--device", "cuda:0"]
        da, db = TMP + "/tb_old_" + tag, TMP + "/tb_new_" + tag
        for d_, cwd in ((da, wt), (db, ROOT)):
            G8.fresh(d_); r = subprocess.run([sys.executable, "-B", "-m", "scratch.s222_det_wrap", "boxes/inject_aw.py"] + com + ["--out", d_], cwd=cwd,
                                             stdout=open(d_ + "/stdout.log", "w"), stderr=subprocess.STDOUT)
            tb.setdefault(tag, {}).setdefault("rc", []).append(r.returncode)
        if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
            tb[tag].update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=("model", "opt", "rng_d", "np_rng", "torch_rng", "cuda_rng", "stream_rng", "stream_stats", "gn_log")))
            st = lambda p: [{k: v for k, v in json.loads(l_).items() if k != "sec"} for l_ in open(p)]  # noqa: E731
            tb[tag]["log_equal"] = st(da + "/log.jsonl") == st(db + "/log.jsonl")
        tb[tag]["pass"] = tb[tag]["rc"] == [9, 9] and tb[tag].get("n_mismatch") == 0 and bool(tb[tag].get("log_equal"))
    o["trainbit"] = tb
    o["pass"] = (all(v["old_keys_equal"] and v["excl_growth"] == 32000 and len(v["new_keys"]) == 8 for v in o["sets"].values()) and o["nw_ow_sets_equal"]
                 and o["stream_equal"] and all(v["pass"] for v in tb.values()))
    return dump("G1", o)


def mask(t):
    return re.sub(r"\{(\w)\}", lambda m: "X" if m.group(1) in "ABC" else ("N" if m.group(1) == "c" else "U"), t).lower().replace(".", " .").replace(",", " ,").replace("'s", " 's").split()


def grams(t, n=3):
    w = mask(t); return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def g2():
    import boxes.nested_words as NW
    import boxes.ow_words as OW
    T = json.load(open("docs/research_framediv_2026-10-30/fd_v1_templates.json", encoding="utf-8")); o = {}
    o["templates_equal"] = all(NW.FRAMES[op][8:12] == T["TRAIN_NEW"][op] and NW.FRAMES[op][12:16] == T["HELD"][op] and len(NW.FRAMES[op]) == 16 for op in T["HELD"])
    o["alias_ok"] = NW.FRAMES["add_s"] is NW.FRAMES["add_v"] and NW.FRAMES["mul_s"] is NW.FRAMES["mul_v"]; o["frame_all"] = list(NW.FRAME_ALL); o["frame_train"] = list(NW.FRAME_TRAIN)
    jw = []
    for op in T["HELD"]:
        refs = NW.FRAMES[op][:12]
        for k, t in enumerate(T["HELD"][op]):
            g = grams(t); jw.append(max(len(g & grams(s)) / max(1, len(g | grams(s))) for s in refs))
    o["novelty_max_jaccard"] = round(max(jw), 4)
    bad = tot = 0
    for frs in ((8, 9, 10, 11), (12, 13, 14, 15), tuple(range(8))):
        for law in (1, 2):
            s = OW.Stream(777 + law, frames=frs, order_mode="topo", mul_s_p=law)
            for d in range(1, 7):
                for _ in range(150):
                    it = s.draw(d); tot += 1; bad += OW.solve(it["text"]) != it["answer"]
    o["solve"] = [bad, tot]
    try:
        r = NW.selfcheck(); o["nw_selfcheck"] = r if isinstance(r, (dict, list, str, int, float, bool, type(None))) else str(r); o["nw_selfcheck_ok"] = True
    except AssertionError as e:
        o["nw_selfcheck_ok"] = False; o["nw_selfcheck"] = repr(e)
    o["pass"] = o["templates_equal"] and o["alias_ok"] and o["frame_all"] == list(range(8)) and o["frame_train"] == list(range(6)) and o["novelty_max_jaccard"] <= 0.20 \
        and o["solve"] == [0, 5400] and o["nw_selfcheck_ok"]
    return dump("G2", o)


def g3():
    import boxes.inject_ow as IO
    o = {}
    for law in (0, 1, 2):
        s = IO.build_sets(mul_s_p=law); fs = s["framesets"]; t2 = {it["sig"] for d in s["test2"] for it in s["test2"][d]}; tr = {it["sig"] for d in s["transfer"] for it in s["transfer"][d]}
        sig = [it["sig"] for f in fs for d in fs[f] for it in fs[f][d]]
        o[str(law)] = {"frames": sorted(fs), "sha_keys": sorted(k for k in s["shas"] if k.startswith("frameset")), "n_per_d": {str(f): [len(fs[f][d]) for d in IO.FRAMESET_DS] for f in fs},
                       "frames_ok": all(set(it["frames"]) == {f} for f in fs for d in fs[f] for it in fs[f][d]), "all_in_excl": all(x in s["excl"] for x in sig),
                       "in_test2": sum(x in t2 for x in sig), "in_transfer": sum(x in tr for x in sig), "n_unique": len(set(sig)), "n": len(sig),
                       "shas": {k: v[:12] for k, v in s["shas"].items() if k.startswith("frameset")}}
    o["pass"] = all(v["frames"] == list(range(8, 16)) and len(v["sha_keys"]) == 8 and v["frames_ok"] and v["all_in_excl"] and v["in_test2"] == 0 and v["in_transfer"] == 0
                    and all(n == [1000] * 4 for n in v["n_per_d"].values()) for v in (o["0"], o["1"], o["2"]))
    return dump("G3", o)


def g4():
    import gate_s228 as G8
    import s228_args_check as C
    d = G8.fresh(TMP + "/args_fh"); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + FH + ["--hold_inc", "1.6", "--auto_micro", "--ckpt_every", "1000", "--eval_parts",
                                                 "acc,ws,opclass,swap_prod,swap_kd,lp5h,cue_only", "--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
    new = json.load(open(d + "/args.json")); ref = json.load(open(ROOT + "/scratch/s240_args_frozen_lh_fh_s1.json")); dif = C.diff(ref, new)
    o = {"rc": rc, "diff_keys": sorted(dif), "diff": {k: (v if not k.endswith("sha256") and not k.endswith("_sha") else [str(v[0])[:12], str(v[1])[:12]]) for k, v in dif.items() if k != "pair_stats"}}
    ps = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { $_.CommandLine }"], capture_output=True, text=True).stdout
    o["python_processes"] = [l_.strip()[:160] for l_ in ps.splitlines() if l_.strip()]
    expected = {"module_sha256", "nested_words_sha256", "inject_ow_sha256", "eval_module_sha256", "excl_n", "train_frames", "init_trunk", "init_model", "keep_final_ckpt"} | {"frameset%d_sha" % f for f in range(8, 16)}
    o["unexpected_keys"] = sorted(set(dif) - expected)
    o["pass"] = rc == 0 and not o["unexpected_keys"]; return dump("G4", o)


def tmp_run(src, name):
    d = os.path.join(TMP, name); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for f in ("args.json", "train_summary.json"):
        shutil.copy(os.path.join(RUNS, src, f), d)
    os.link(os.path.join(RUNS, src, "model.pt"), os.path.join(d, "model.pt")); return d


def g5():
    parts = "ws,opclass,cue_only"; res = {}
    for mod, name in (("boxes._inject_aw_eval_s249", "g5_old"), ("boxes.inject_aw_eval", "g5_new")):
        d = tmp_run("lh_fh_s1", name)
        res[name] = (d, subprocess.run([sys.executable, "-B", "-m", mod, "--run", d, "--parts", parts, "--eval_bs", "250", "--device", "cuda:0"]).returncode)
    eq = {}
    for p_ in parts.split(","):
        x = [json.load(open(os.path.join(res[n][0], "eval_%s.json" % p_))) for n in ("g5_old", "g5_new")]
        for y in x:
            y.pop("sec", None); y.pop("module_sha256", None)
        eq[p_] = x[0] == x[1]
    for n in res:
        os.remove(os.path.join(res[n][0], "model.pt"))
    return dump("G5", {"rc": {k: v[1] for k, v in res.items()}, "equal": eq, "pass": all(v[1] == 0 for v in res.values()) and all(eq.values())})


def g6():
    d = tmp_run("df_plaw4_s1", "g6_smoke")
    rc = subprocess.run([sys.executable, "-B", "scratch/s250_frames.py", "--run", "df_plaw4_s1", "--src_dir", d, "--out", d, "--eval_bs", "250", "--device", "cuda:0"]).returncode
    os.remove(os.path.join(d, "model.pt")); o = {"rc": rc}; f = os.path.join(d, "eval_frames.json")
    if rc == 0 and os.path.exists(f):
        e = json.load(open(f)); fr = [str(x) for x in range(8, 16)]
        o["frames"] = sorted(k for k in e if k.isdigit()); o["fields_ok"] = all(all(set(e[x][str(dd)]) >= {"n", "acc", "ws", "wrap_any_share"} and e[x][str(dd)]["n"] == 1000
                                                                                  for dd in range(1, 5)) for x in fr)
        o["meta"] = {k: e.get(k) for k in ("eval_gpu", "eval_bs", "eval_dtype", "law", "set_law", "think_k", "peak_gib", "sec")}
        o["pass"] = rc == 0 and o["frames"] == sorted(fr) and o["fields_ok"]
    else:
        o["pass"] = False
    return dump("G6", o)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gates", default="G1,G2,G3,G4,G5,G6"); ap.add_argument("--redo", action="store_true"); a = ap.parse_args()
    os.makedirs(TMP, exist_ok=True)
    for gname in a.gates.split(","):
        if os.path.exists(os.path.join(OUT, gname + ".json")) and not a.redo:
            print(gname, "exists, skipped"); continue
        {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6}[gname]()
