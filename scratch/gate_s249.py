"""§249 FRAME-TRANSFER gates (FT_v1 §2 + Note 249-1) -> scratch/gate_s249/<gate>.json (incremental; --redo re-runs).
  G0 (CPU) Note 249-1 (1): frames 6-7 never in a training stream of the lh_fh lineage (lh_fh_s1/s2 <- df_plaw4_s1/s2 <- rcb_plaw4_s1/s2 <- aw_plaw2_s1 <- s233_askname_s1 <- GPT-2):
     per run, its trainer (matched by module_sha256) builds the training stream with frames=OW.FRAME_TRAIN, its ow_words version (matched by ow_words_sha256) has
     FRAME_TRAIN == (0..5), and a replay of that stream (that ow_words version, the trainer's kwargs and law, no EXCL) over 20,000 draws (d cycling 1..d_max) uses frames 0..5
     only; FT_v1 §2 gate: the transfer set's sha (printed) == build_sets' transfer_sha == each model's args.json transfer_sha; every transfer item's frames are in {6, 7}.
  G2 (4060): the §248 eval module (boxes/_inject_aw_eval_s248.py) vs the new one on parts ws, opclass, cue_only (lh_fh_s1): equal bar sec / module_sha256.
  G3 (4060): full-scale smokes of scratch/s249_ft.py on df_plaw4_s1 (LAT) and rcb_pcd4_s1 (CD; neither a §249 model) into scratch/_tmp/s249: the fields score_ft reads
     (by_d acc / ws, cue_only acc / majority, CD by_d acc) + by_frame; score_ft.one runs on (LAT, CD) (label not stored, no value printed).
    python -B scratch/gate_s249.py [--gates G0,G2,G3] [--redo]"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
OUT = os.path.join(ROOT, "scratch", "gate_s249"); TMP = os.path.join(ROOT, "scratch", "_tmp", "s249"); RUNS = os.path.join(ROOT, "boxes", "runs_fsq")
LINEAGE = ("lh_fh_s1", "lh_fh_s2", "df_plaw4_s1", "df_plaw4_s2", "rcb_plaw4_s1", "rcb_plaw4_s2", "aw_plaw2_s1", "s233_askname_s1")


def dump(name, o):
    os.makedirs(OUT, exist_ok=True); json.dump(o, open(os.path.join(OUT, name + ".json"), "w", newline="\n"), indent=1, default=str); print(name, "PASS" if o.get("pass") else "FAIL", flush=True)
    return o


def g0():
    import numpy as np
    files = {}
    for p in sorted(os.listdir("boxes")):
        if p.endswith(".py"):
            files.setdefault(hashlib.sha256(open("boxes/" + p, "rb").read()).hexdigest(), []).append("boxes/" + p)
    o = {"runs": {}}
    for r in LINEAGE:
        a = json.load(open(os.path.join(RUNS, r, "args.json"))); tr = files.get(a.get("module_sha256"), [None])[0]; owf = files.get(a.get("ow_words_sha256"), [None])[0]
        src = open(tr, encoding="utf-8").read() if tr else ""; line = [l_.strip() for l_ in src.splitlines() if "OW.Stream(OW.SEED_TRAIN" in l_]
        rec = {"trainer": tr, "ow_words": owf, "train_stream_line": line, "frames_arg_is_FRAME_TRAIN": bool(line) and all("frames=OW.FRAME_TRAIN" in l_ for l_ in line)}
        if owf:
            spec = importlib.util.spec_from_file_location("_owv_%s" % r, owf); OWv = importlib.util.module_from_spec(spec); spec.loader.exec_module(OWv)
            rec["FRAME_TRAIN"] = list(OWv.FRAME_TRAIN); kw = {"order_mode": "topo", "dag": True}
            if line and "mul_s_p=a.mul_s_p" in line[0]:
                kw.update(mul_s_p3=a.get("mul_s_p3", 1.0) if a.get("mul_s_p3") is not None else 1.0, mul_s_p=a.get("mul_s_p") if a.get("mul_s_p") is not None else 1)
            s_ = OWv.Stream(OWv.SEED_TRAIN, frames=OWv.FRAME_TRAIN, **kw); fr = set(); dm = int(a.get("d_max") or 4)
            for i in range(20000):
                fr.update(int(x) for x in s_.draw(1 + i % dm)["frames"])
            rec["replay_draws"] = 20000; rec["replay_frames"] = sorted(fr); rec["replay_kw"] = kw
        rec["ok"] = bool(rec["frames_arg_is_FRAME_TRAIN"] and rec.get("FRAME_TRAIN") == list(range(6)) and rec.get("replay_frames") and set(rec["replay_frames"]) <= set(range(6)))
        o["runs"][r] = rec
    import boxes.inject_ow as IO
    import boxes.ow_words as OW
    s = IO.build_sets(mul_s_p=2); sha = OW.set_sha([it for d in IO.TEST_DS for it in s["transfer"][d]])
    o["transfer_sha"] = sha; o["transfer_sha_eq_build_sets"] = sha == s["shas"]["transfer_sha"]
    o["transfer_sha_eq_args"] = {r: json.load(open(os.path.join(RUNS, r, "args.json"))).get("transfer_sha") == sha for r in ("lh_fh_s1", "lh_fh_s2", "df_pcd4_s1")}
    o["transfer_frames_in_6_7"] = all(set(it["frames"]) <= {6, 7} for d in s["transfer"] for it in s["transfer"][d])
    print("TRANSFER SHA", sha, flush=True)
    o["pass"] = (all(v["ok"] for v in o["runs"].values()) and o["transfer_sha_eq_build_sets"] and all(o["transfer_sha_eq_args"].values()) and o["transfer_frames_in_6_7"])
    return dump("G0", o)


def tmp_run(src, name):
    d = os.path.join(TMP, name); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for f in ("args.json", "train_summary.json"):
        if os.path.exists(os.path.join(RUNS, src, f)):
            shutil.copy(os.path.join(RUNS, src, f), d)
    os.link(os.path.join(RUNS, src, "model.pt"), os.path.join(d, "model.pt")); return d


def g2():
    parts = "ws,opclass,cue_only"; res = {}
    for mod, name in (("boxes._inject_aw_eval_s248", "g2_old"), ("boxes.inject_aw_eval", "g2_new")):
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
    o = {}; ev = {}
    for src, tag, bs in (("df_plaw4_s1", "lat", "250"), ("rcb_pcd4_s1", "cd", "100")):   # CD: bs 100 (the direct trunk on the cued text OOMed the 8 GB 4060 at 250)
        d = tmp_run(src, "g3_" + tag)
        rc = subprocess.run([sys.executable, "-B", "scratch/s249_ft.py", "--run", src, "--src_dir", d, "--out", d, "--eval_bs", bs, "--device", "cuda:0"]).returncode
        os.remove(os.path.join(d, "model.pt")); f = os.path.join(d, "eval_transfer.json"); o[tag] = {"rc": rc}
        if rc == 0 and os.path.exists(f):
            e = json.load(open(f)); ev[tag] = e; o[tag]["ds"] = sorted(e["by_d"], key=int); o[tag]["n"] = {k: v["n"] for k, v in e["by_d"].items()}
            o[tag]["meta"] = {k: e.get(k) for k in ("eval_gpu", "eval_bs", "eval_dtype", "law", "set_law", "think_k", "arm", "peak_gib", "sec", "transfer_sha")}
            o[tag]["by_frame_keys"] = sorted(e["by_frame"]); o[tag]["by_frame_n_d3"] = {f: e["by_frame"][f]["3"]["n"] for f in e["by_frame"]}
    lat, cd = ev.get("lat"), ev.get("cd")
    o["lat_fields_ok"] = bool(lat) and all("ws" in lat["by_d"][d] and "acc" in lat["by_d"][d] for d in lat["by_d"]) and all(k in lat["cue_only"] for k in ("2", "3", "4")) \
        and all(set(v) >= {"acc", "majority"} for v in lat["cue_only"].values()) and all("ws" in lat["by_frame"]["6"][d] for d in lat["by_frame"]["6"])
    o["cd_fields_ok"] = bool(cd) and all("acc" in cd["by_d"][d] for d in cd["by_d"]) and "cue_only" not in cd
    if lat and cd:
        sys.path.insert(0, os.path.join(ROOT, "theory")); import score_ft as SF
        o["score_ft_one_runs"] = SF.one(lat, cd)["label"] in ("VOID", "TRANSFER", "ACC-ONLY", "NO-TRANSFER")
    o["pass"] = bool(o.get("lat_fields_ok") and o.get("cd_fields_ok") and o.get("score_ft_one_runs") and all(o[t]["rc"] == 0 for t in ("lat", "cd"))
                     and all(v == 1000 for t in ("lat", "cd") for v in o[t]["n"].values()))
    return dump("G3", o)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gates", default="G0,G2,G3"); ap.add_argument("--redo", action="store_true"); a = ap.parse_args()
    for gname in a.gates.split(","):
        if os.path.exists(os.path.join(OUT, gname + ".json")) and not a.redo:
            print(gname, "exists, skipped"); continue
        {"G0": g0, "G2": g2, "G3": g3}[gname]()
