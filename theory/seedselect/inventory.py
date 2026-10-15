"""seedselect inventory (THEORY_SEEDSELECT_DRAFT_2026-10-14, task 1): run metadata and OUTCOME LABELS only.

BLIND: no early-window value (loss / answer_ce / aux_ce / write_grad_norm before any detector cutoff, or any function of
them) is stored or printed. The only quantities read from log files are: record counts, field names, step counts,
duplicate / resume markers, and these outcome labels:
  nested (DeltaMem, args task == "nested"): theory/basin_entry.py entry(0.5), entry(0.1), and the last bin mean.
  fsq120 (args has "family"): 123-3's 2k plateau FLAG and 131's FLAG4 (booleans only; nearest logged steps; phase 1, as
      score_s131.py), and eval.json long_acc (exact = long_acc >= 0.99, §131's target).
  replay130 / replay_x (theory/results/replay*/*.jsonl, 2k replays): FLAG (123-3). replay130's FLAG / BASIN are taken
      from score_s130_*.json (published) and FLAG is recomputed here as a cross-check; replay_x's FLAG is computed here.
Recipes are normalised with each module's argparse defaults (older args.json files lack newer keys) and inert sub-options
(e.g. hinge_eps_frac without --hinge) are dropped. Duplicate-trajectory CANDIDATES (same seed, card class and recipe in two
dirs or harnesses) are listed only; identity is tested by gate G2 of seedselect.py, not here.
Outputs: theory/seedselect/inventory.json (metadata), theory/seedselect/labels.json (labels and panel roles only).
    python theory/seedselect/inventory.py
"""
import json
import os
import sys
from collections import Counter, defaultdict

try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, "boxes", "runs_fsq")
OUT = os.path.join(ROOT, "theory", "seedselect")
sys.path.insert(0, os.path.join(ROOT, "theory"))
import basin_entry as BE  # noqa: E402

NOT_RECIPE = {
    "seed", "out", "device", "resume", "resume_2k", "from_args", "eval_only", "eval_at", "eval_bs", "cot_eval_bs", "per_n",
    "serial_pool", "mem_budget", "mem_check", "mcheck_only", "time_steps", "kprimes", "excl_survive_by_n", "format",
    "aux_decoder", "ckpt_every", "log_every", "path_indep", "scratch", "cert_cap", "cert_only", "cert_time", "contract_n",
    "contraction_only", "growth_only", "no_cert", "donor_n", "ext_n", "margin_n", "test_len", "test_n", "data_dir",
    "stop_after", "stop_plateau", "save_at_flag", "check_marker", "probe127", "prefix_only", "probe_only", "clock_only",
    "grow_force_smoke", "micro_from", "train_gpu", "train_driver", "torch_version", "cuda_version", "cudnn_version",
    "python_version", "image_tag", "hostprobe_sps", "keep_ckpt_steps", "ablate704_only"}
NUMERICS = {"micro", "auto_micro", "fast_add", "kmatch_impl", "fp32", "tanh_fp32", "ckpt_passes", "cache_lower", "probe_loss_every"}
NESTED_DEF = {"think_k": 0, "steps": 48000, "bs": 64, "lr": 1e-4, "n_max_train": 8, "mem_write": "top", "write_mix_start": 2000,
              "write_mix_end": 6000, "anneal": "", "mem_layer": 6, "anneal_gate": "", "csr": "", "grokfast": "", "gf_alpha": 0.98,
              "gf_lamb": 2.0, "aux_step": 0.0, "aux_fade": "36000,48000", "add_norm": "", "excl_min_n": 1, "warmup": 100,
              "k_window": "", "sched_scale": 1.0, "aux_nopass": False, "aux_reweight": False, "aux_hold": False, "n_slots": 8}
FSQ_DEF = {"anneal": False, "carrier_ln": False, "copy_frac": 0.0, "d_q": 8, "dither": 0.0, "fsq_dither": 0.0, "grow": None,
           "grow_noise": 0.0, "grow_staged": False, "grow_trigger": None, "levels": 8, "lr_new": 1.0, "nested": False,
           "no_round": False, "p_hard_end": 4000, "p_hard_start": 2000, "raw_h": False, "redraw_head": False, "skip_dec": False,
           "soft_beta": 0.0, "std": False, "std2": False, "tsd": None, "tsd_layer": None, "tsd_tau": 0.1, "tsd_teacher": None,
           "tsd_w": 1.0, "var_est": "ema", "var_res": False, "var_w": 1.0, "grow2": 0, "grow2_min_step": 6000,
           "grow2_lr_scale": 1.0, "hinge": False, "hinge_eps_frac": 0.1, "hinge_rows": 0.25, "sum_mod": 10000, "lmax_fixed": 0,
           "sched_scale": 1.0, "steps": 24000, "bs": 64, "lr": 1e-4}


def normalise(fam, a):
    rec = {k: v for k, v in a.items() if k not in NOT_RECIPE and k not in NUMERICS}
    if fam == "nested":
        for k, v in NESTED_DEF.items():
            rec.setdefault(k, v)
        if not rec.get("grokfast"):
            rec.pop("gf_alpha", None); rec.pop("gf_lamb", None)
        if rec.get("arm") != "readslots":
            rec.pop("n_slots", None)
    elif fam == "fsq120":
        for k, v in FSQ_DEF.items():
            rec.setdefault(k, v)
        if not rec.get("hinge"):
            rec.pop("hinge_eps_frac", None); rec.pop("hinge_rows", None)
        if not rec.get("var_res"):
            rec.pop("var_est", None); rec.pop("var_w", None)
        if rec.get("tsd") is None:
            for k in ("tsd_layer", "tsd_tau", "tsd_teacher", "tsd_w"):
                rec.pop(k, None)
        if not rec.get("grow2"):
            rec.pop("grow2_min_step", None); rec.pop("grow2_lr_scale", None)
        if rec.get("grow") is None:
            for k in ("grow_noise", "grow_staged", "grow_trigger"):
                rec.pop(k, None)
        if rec.get("family") != "sum4":
            rec.pop("sum_mod", None)
        if rec.get("no_round"):
            rec.pop("fsq_dither", None)
        else:
            rec.pop("dither", None)
    return rec


def gpu_short(s):
    if not s:
        return None
    s = str(s)
    for tag in ["3090 Ti", "3090", "4090", "4060 Ti", "4060", "3070", "2070", "2080 Ti", "A4000", "A10", "A5000", "A6000", "L4"]:
        if tag in s:
            return tag.replace(" ", "").lower()
    return s


def suffix_gpu(name):
    for tag in ["_3090", "_4090", "_4060ti", "_4060", "_3070", "_2070", "_2080ti", "_a4000"]:
        if name.endswith(tag) or (tag + "_") in name or (tag + ".") in name:
            return tag[1:]
    return None


def log_meta(path):
    n = 0; loss_steps = []; fields = set(); resumed = 0; stopped = None
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        n += 1
        if "resumed" in r or r.get("event") == "resumed":
            resumed += 1
        if r.get("stopped"):
            stopped = str(r.get("stopped"))
        if "step" in r and "loss" in r and r["loss"] is not None:
            loss_steps.append(int(r["step"]))
            fields |= set(r.keys())
    c = Counter(loss_steps); st = sorted(c)
    gaps = Counter(b - a for a, b in zip(st, st[1:]))
    return {"records": n, "loss_records": len(loss_steps), "unique_steps": len(c),
            "dup_steps": sum(1 for v in c.values() if v > 1), "max_step": max(c) if c else 0, "min_step": min(c) if c else 0,
            "log_interval": gaps.most_common(1)[0][0] if gaps else None,
            "resumed_records": resumed, "stopped": stopped, "fields": sorted(fields)}


def flags_from(path, phase_filter=True):
    """123-3 FLAG and 131 FLAG4 as booleans (nearest logged steps; last occurrence of a step wins). No value is returned."""
    L = {}
    for line in open(path):
        try:
            x = json.loads(line)
        except Exception:
            continue
        if "loss" in x and "step" in x and x["loss"] is not None and (not phase_filter or x.get("phase", 1) == 1):
            L[int(x["step"])] = float(x["loss"])
    if not L:
        return None, None
    g = lambda t: L[min(L, key=lambda s: abs(s - t))]  # noqa: E731
    f2 = bool(g(2000) >= 0.9 * g(1000) and g(2000) >= 0.25 * g(100)) if max(L) >= 2000 else None
    f4 = bool(g(4000) >= 0.9 * g(2600) and g(4000) >= 0.25 * g(100)) if max(L) >= 4000 else None
    return f2, f4


def main():
    runs = {}; voids = []
    for d in sorted(os.listdir(R)):
        p = os.path.join(R, d)
        if not os.path.isdir(p):
            continue
        if ".void" in d:
            voids.append(d); continue
        ap, lp = os.path.join(p, "args.json"), os.path.join(p, "log.jsonl")
        if not (os.path.exists(ap) and os.path.exists(lp)):
            continue
        try:
            a = json.load(open(ap))
        except Exception:
            continue
        ev = {}
        if os.path.exists(os.path.join(p, "eval.json")):
            try:
                ev = json.load(open(os.path.join(p, "eval.json")))
            except Exception:
                ev = {}
        fam = "nested" if a.get("task") == "nested" else ("fsq120" if "family" in a else ("other:" + str(a.get("task"))))
        g_train = gpu_short(a.get("train_gpu") or ev.get("train_gpu")); g_src = "train_gpu"
        if not g_train and suffix_gpu(d):
            g_train, g_src = suffix_gpu(d), "dir suffix"
        if not g_train:
            g_src = "unknown (eval gpu %s)" % gpu_short(ev.get("gpu"))
        m = log_meta(lp)
        r = {"family": fam, "arm": a.get("arm"), "sub": a.get("family"), "seed": a.get("seed"), "steps": a.get("steps"),
             "gpu": g_train, "gpu_src": g_src, "driver": a.get("train_driver"), "harness": "fsq120.py" if fam == "fsq120" else ("nested170.py" if fam == "nested" else None),
             "numerics": {k: a.get(k) for k in NUMERICS if k in a}, "recipe": normalise(fam, a), "log": m}
        lab = {}
        if fam == "nested":
            bm = BE.bins(BE.losses(d))
            lab = {"entry_0.5": BE.entry(d, 0.5), "entry_0.1": BE.entry(d, 0.1),
                   "last_bin": round(bm[max(bm)], 4) if bm else None, "last_bin_end": 1000 * (max(bm) + 1) if bm else None,
                   "complete": bool(a.get("steps") and m["max_step"] >= a["steps"])}
            lab["entered"] = (lab["entry_0.5"] is not None) if lab["complete"] else None
        elif fam == "fsq120":
            f2, f4 = flags_from(lp)
            la = ev.get("long_acc")
            lab = {"flag2k": f2, "flag4k": f4, "long_acc": la, "exact": (None if la is None else bool(la >= 0.99)),
                   "complete": bool(a.get("steps") and m["max_step"] >= a["steps"]), "stopped": m["stopped"]}
        r["labels"] = lab
        runs[d] = r

    groups = defaultdict(list)
    for d, rr in runs.items():
        groups[(rr["family"], json.dumps(rr["recipe"], sort_keys=True, default=str))].append(d)
    base = {"nested": "s185_rms_s1", "fsq120": "s120_s5_final_s1"}
    G = []
    for i, ((fam, key), ds) in enumerate(sorted(groups.items(), key=lambda kv: (kv[0][0], sorted(kv[1])[0]))):
        rec = json.loads(key); b = runs.get(base.get(fam, ""), {}).get("recipe", {})
        diff = {k: rec.get(k, "<absent>") for k in sorted(set(rec) | set(b)) if json.dumps(rec.get(k, "<absent>"), default=str) != json.dumps(b.get(k, "<absent>"), default=str)}
        gid = "%s/%03d" % (fam, i)
        G.append({"gid": gid, "family": fam, "runs": sorted(ds), "recipe_diff_vs_base": diff, "base": base.get(fam)})
        for d in ds:
            runs[d]["gid"] = gid

    # 2k replay panels (scratch/replay_basin.py: fsq120 FINAL S5, logs every 50 steps)
    fit = json.load(open(os.path.join(ROOT, "theory", "results", "score_s130_fit.json")))
    test = json.load(open(os.path.join(ROOT, "theory", "results", "score_s130_test.json")))
    pub = {}
    for s, v in fit["seeds"].items():
        pub[("4060", int(s))] = {"block": "FIT", "flag": v["flag"], "basin": v["basin"]}
    for blk, card in [("CROSS (3070)", "3070"), ("SAME (4060)", "4060")]:
        for s, v in test["dip|flag"][blk]["seeds"].items():
            pub[(card, int(s))] = {"block": blk.split()[0], "flag": v["flag"], "basin": v["basin"]}
    replay = {}
    for sub in ("replay130", "replay_x"):
        RP = os.path.join(ROOT, "theory", "results", sub)
        for f in sorted(os.listdir(RP)):
            if not f.endswith(".jsonl") or ".void" in f:
                continue
            card, s = f[:-6].split("_s")
            path = os.path.join(RP, f); m = log_meta(path)
            cfg = {}
            for line in open(path):
                x = json.loads(line)
                if "config" in x:
                    cfg = x["config"]; break
            f2, _ = flags_from(path, phase_filter=False)
            pb = pub.get((card, int(s)), {}) if sub == "replay130" else {}
            replay["%s/%s_s%s" % (sub, card, s)] = {
                "panel": sub, "card": card, "seed": int(s), "amp": cfg.get("amp"), "torch": cfg.get("torch"),
                "block": pb.get("block"), "flag2k": pb.get("flag", f2) if sub == "replay130" else f2, "basin": pb.get("basin"),
                "flag_recomputed_agrees": (f2 == pb.get("flag")) if pb else None,
                "complete": m["max_step"] >= 2000, "log": {k: m[k] for k in ("records", "loss_records", "unique_steps", "dup_steps", "max_step", "log_interval")}}

    # duplicate-trajectory candidates: same seed + card class + normalised recipe, different dir / harness (not tested here)
    FINAL = runs["s120_s5_final_s1"]["recipe"]
    cand = defaultdict(list)
    for d, rr in runs.items():
        if rr["family"] == "fsq120" and rr["recipe"] == FINAL:
            cand[(rr["seed"], rr["gpu"])].append(d)
        elif rr["family"] == "nested":
            cand[("nested", rr["gid"], rr["seed"], rr["gpu"])].append(d)
    amp_of = {"2070": "fp32"}
    for k, v in replay.items():
        cand[(v["seed"], v["card"])].append(k)
    dup_cand = {str(k): sorted(v) for k, v in cand.items() if len(v) > 1}

    json.dump({"runs": runs, "groups": G, "voids": voids, "replays": replay, "duplicate_candidates": dup_cand},
              open(os.path.join(OUT, "inventory.json"), "w", newline="\n"), indent=1, default=str)

    labs = {}
    for d, rr in runs.items():
        if rr["family"] in ("nested", "fsq120"):
            labs[d] = dict(family=rr["family"], gid=rr["gid"], seed=rr["seed"], gpu=rr["gpu"], harness=rr["harness"],
                           max_step=rr["log"]["max_step"], log_interval=rr["log"]["log_interval"], **rr["labels"])
    for k, v in replay.items():
        labs[k] = {"family": "fsq120", "gid": "replay(FINAL S5)", "panel": v["panel"], "seed": v["seed"], "gpu": v["card"],
                   "amp": v["amp"], "harness": "replay_basin.py", "block": v["block"], "flag2k": v["flag2k"], "basin": v["basin"],
                   "max_step": v["log"]["max_step"], "log_interval": v["log"]["log_interval"]}
    json.dump({"_note": "OUTCOME LABELS ONLY (no early-window values). See docs/THEORY_SEEDSELECT_DRAFT_2026-10-14.md section 1.",
               "labels": labs, "duplicate_candidates": dup_cand},
              open(os.path.join(OUT, "labels.json"), "w", newline="\n"), indent=1, default=str)

    for gr in G:
        if gr["family"] not in ("nested", "fsq120"):
            continue
        print("\n== %s  n=%d  diff_vs_%s=%s" % (gr["gid"], len(gr["runs"]), gr["base"], json.dumps(gr["recipe_diff_vs_base"], default=str)[:260]))
        for d in gr["runs"]:
            rr = runs[d]; lg = rr["log"]
            print("   %-36s s=%-4s gpu=%-7s max=%-6s int=%-4s dup=%d res=%d %s| %s" % (
                d, rr["seed"], rr["gpu"], lg["max_step"], lg["log_interval"], lg["dup_steps"], lg["resumed_records"],
                "" if not lg["stopped"] else "STOP ", json.dumps(rr["labels"], default=str)))
    print("\nreplays:", Counter((v["panel"], v["card"], v["amp"]) for v in replay.values()))
    print("replay130 flag recompute agreement:", Counter(v["flag_recomputed_agrees"] for v in replay.values() if v["panel"] == "replay130"))
    print("duplicate candidates:", len(dup_cand))
    print("voids:", len(voids))


if __name__ == "__main__":
    main()
