"""§216 SD2 pool builder (B1; SD2_v1.md at e9ade2ec §2.1-§2.2, §2.4; from scratch/s215_build_pool.py). Local 3070 (cuda:1), fp32, bs 32.
  --lineage s1|s2 --round 1|2 (s1 round 2; s2 round 1 and round 2), two stages:
  --stage gates: the exclusion set (below) -> the gold patch -> each window's gate items (its own rng, n order, 2,000 per n, redrawn on an exclusion hit,
     syms + n only) -> <gate dir>/gate_items.json {windows {W: {seed, ns, items}}, collide_n}. Then scratch/s216_gate2.py scores them.
  --stage pool: reads <gate dir>/eval_gate.json (each window's PASSING labeller: the round's model, else the pre-stated base fallback; a window with no
     passing labeller -> no pool, exit 3), the exclusion set + this build's gate items, the gold patch, then per n (n order) 4,096-draw chunks labelled
     by that window's labeller (selfdepth.label_chains), kept = KEEP and CV, exclusion hits and within-pool duplicates redrawn (counted per set), until
     the per-n target; one interleaved order (default_rng(pool seed + 1).permutation) -> <pool dir>/<pool file> (keys exactly {syms, n, chain_label,
     steps}) + pool_meta.json.
  Exclusion (§2.2 item 2, item 9) by exact rendered syms: test_set(1000, n_max 24); SERIAL (rng 17300, 2,200 x generate(8)); the 213-37 fresh items
     (part_chainlabel_fresh's draw: rng 21337, n 9..24 sequentially, 1,000 per n, excluding test_set(1000) + SERIAL); GATE-1 (s215_gate1); every s216
     gate_items.json and pool file that exists at build time (this build's own gates are added in stage pool); the round-1 pool s215_pool/pool_r1.jsonl.
  The labeller's chain is the model's own decode only (no simulator call); gold functions raise from the patch on.
    python scratch/s216_build_pool.py --lineage s1 --round 2 --stage gates --device cuda:1"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402
import boxes.vbank_diag as VD  # noqa: E402
import boxes.vbank_eval as VE  # noqa: E402
import boxes.selfdepth as SD  # noqa: E402

R = "boxes/runs_fsq/"
R2N = {**{n: 37_000 for n in (9, 10, 11, 12)}, **{n: 74_000 for n in (13, 14, 15, 16)}}
CFG = {   # SD2_v1 §2.1 / §2.2 items 1, 9 / §2.4
    ("s1", 2): {"labeller": R + "s215_ftself_s1", "base": R + "s213_d3t_s1", "pool_seed": 22600, "per_n": R2N, "pool_dir": R + "s216_pool_s1", "pool_file": "pool_r2.jsonl",
                "gate_dir": R + "s216_gate2_s1", "windows": {"W2": (22660, (13, 14, 15, 16), "GATE_2"), "W1": (22661, (9, 10, 11, 12), "GATE_1p")}},
    ("s2", 1): {"labeller": R + "s216_d3t_s2", "base": None, "pool_seed": 22700, "per_n": {n: 110_000 for n in (9, 10, 11, 12)}, "pool_dir": R + "s216_pool_r1_s2",
                "pool_file": "pool_r1.jsonl", "gate_dir": R + "s216_gate1_s2", "windows": {"W1": (22760, (9, 10, 11, 12), "GATE_1")}},
    ("s2", 2): {"labeller": R + "s216_r1_s2", "base": R + "s216_d3t_s2", "pool_seed": 22800, "per_n": R2N, "pool_dir": R + "s216_pool_r2_s2", "pool_file": "pool_r2.jsonl",
                "gate_dir": R + "s216_gate2_s2", "windows": {"W2": (22860, (13, 14, 15, 16), "GATE_2"), "W1": (22861, (9, 10, 11, 12), "GATE_1p")}},
}
GATE_PER_N = 2000


def patch_gold():
    def boom(*a, **k):
        raise RuntimeError("G-S0: a gold function was called in the pool builder")
    for mod, nm in ((N, "apply"), (N, "value"), (N, "eval_tokens"), (N, "cot_steps"), (VB, "recode_targets"), (VB, "rec_targets"), (VD, "reduce_k")):
        setattr(mod, nm, boom)


def fresh_213_37(n_lo=9, n_hi=24, n_items=1000):
    """part_chainlabel_fresh's item draw, extended sequentially to n 24 (E3): rng 21337, excluding test_set(1000) (main's TE) + SERIAL."""
    excl = {tuple(it["syms"]) for it in N.test_set(1000)}; r_sp = np.random.default_rng(17300)
    for _ in range(2200):
        excl.add(tuple(N.render(N.generate(r_sp, 8)[0])))
    rng = np.random.default_rng(21337); out = set()
    for n in range(n_lo, n_hi + 1):
        got = 0
        while got < n_items:
            r_, _ = N.generate(rng, n); s_ = tuple(N.render(r_))
            if s_ in excl:
                continue
            out.add(s_); got += 1
    return out


def exclusion_sets(cfg, with_own_gates):
    """{name: set of syms tuples} per §2.2 item 2 / 9; every s216 gate / pool file that exists now is included (this build's own pool never is)."""
    S = {"test_n24": {tuple(it["syms"]) for it in N.test_set(1000, n_max=24)}}
    r_sp = np.random.default_rng(17300); S["serial"] = {tuple(N.render(N.generate(r_sp, 8)[0])) for _ in range(2200)}
    S["s213_37_n9_24"] = fresh_213_37()
    S["gate1_s215"] = {tuple(it["syms"]) for it in json.load(open(R + "s215_gate1/gate_items.json"))["items"]}
    S["pool_r1_s215"] = set()
    with open(R + "s215_pool/pool_r1.jsonl", encoding="utf-8") as f:
        for ln in f:
            S["pool_r1_s215"].add(tuple(json.loads(ln)["syms"]))
    own_g = os.path.normpath(os.path.join(cfg["gate_dir"], "gate_items.json")); own_p = os.path.normpath(os.path.join(cfg["pool_dir"], cfg["pool_file"]))
    for g in sorted(x for x in glob.glob(R + "s216_gate*/gate_items.json") if "_smoke" not in x):   # smoke dirs never count
        if os.path.normpath(g) == own_g and not with_own_gates:
            continue
        S["gates:" + g] = {tuple(it["syms"]) for w in json.load(open(g))["windows"].values() for it in w["items"]}
    for p in sorted(x for x in glob.glob(R + "s216_pool*/pool_r*.jsonl") if "_smoke" not in x):
        if os.path.normpath(p) == own_p:
            continue
        S["pool:" + p] = set()
        with open(p, encoding="utf-8") as f:
            for ln in f:
                S["pool:" + p].add(tuple(json.loads(ln)["syms"]))
    return S


def sha256_file(p):
    return SD.file_sha256(p)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lineage", required=True, choices=["s1", "s2"]); ap.add_argument("--round", type=int, required=True, choices=[1, 2])
    ap.add_argument("--stage", required=True, choices=["gates", "pool"]); ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--scale", type=float, default=1.0, help="smokes only: multiplies every per-n target and the gate per-n count")
    ap.add_argument("--out_suffix", default="", help="smokes only: appended to the pool and gate dirs"); a = ap.parse_args(); dev = a.device
    cfg = dict(CFG[(a.lineage, a.round)]); cfg["pool_dir"] += a.out_suffix; cfg["gate_dir"] += a.out_suffix
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; t0 = time.time()
    os.makedirs(cfg["pool_dir"], exist_ok=True); os.makedirs(cfg["gate_dir"], exist_ok=True)
    S = exclusion_sets(cfg, with_own_gates=(a.stage == "pool")); allx = set().union(*S.values())
    print("EXCL", {k: len(v) for k, v in S.items()}, "union", len(allx), "%.0f s" % (time.time() - t0), flush=True)
    patch_gold()                                                            # G-S0: no gold function from here on
    if a.stage == "gates":
        gp = int(GATE_PER_N * a.scale); out = {"windows": {}, "collide_n": {}, "per_n": gp}
        for w, (seed, ns, gname) in cfg["windows"].items():
            rg = np.random.default_rng(seed); items = []
            for n in ns:
                got = 0; out["collide_n"]["%s_%d" % (w, n)] = 0
                while got < gp:
                    r_, _ = N.generate(rg, n); s_ = N.render(r_); del r_
                    if tuple(s_) in allx:
                        out["collide_n"]["%s_%d" % (w, n)] += 1; continue
                    assert SD.n_of(s_) == n; items.append({"syms": s_, "n": n}); allx.add(tuple(s_)); got += 1
            out["windows"][w] = {"seed": seed, "ns": list(ns), "gate": gname, "items": items}
        out["excl"] = {k: len(v) for k, v in S.items()}
        json.dump(out, open(os.path.join(cfg["gate_dir"], "gate_items.json"), "w", newline="\n"))
        print("GATE_ITEMS", {w: len(v["items"]) for w, v in out["windows"].items()}, out["collide_n"], "%.0f s" % (time.time() - t0), flush=True); return
    eg = json.load(open(os.path.join(cfg["gate_dir"], "eval_gate.json"))); wl = eg["window_labeller"]
    if any(wl.get(w) is None for w in cfg["windows"]):
        print("NO_POOL: a window has no passing labeller", wl, flush=True); sys.exit(3)
    models = {p: VE.load(p, dev)[0] for p in sorted(set(wl.values()))}
    win_of = {n: w for w, (_, ns, _g) in cfg["windows"].items() for n in ns}
    rp = np.random.default_rng(cfg["pool_seed"]); kept, seen = [], set(); M = {k: {} for k in ("draws", "dup", "labelled", "dropped", "keep_all", "at_target", "T_max")}
    hits = {k: {} for k in S}
    for n in sorted(cfg["per_n"]):
        tgt = int(cfg["per_n"][n] * a.scale); lab_m = models[wl[win_of[n]]]; kn = nlab = draws = dup = drop = 0; tmax = 0
        for k in S:
            hits[k][str(n)] = 0
        while kn < tgt:
            batch = []
            while len(batch) < 4096:
                r_, _ = N.generate(rp, n); s_ = N.render(r_); del r_; draws += 1; ts = tuple(s_)
                if ts in allx:
                    for k, v in S.items():
                        hits[k][str(n)] += int(ts in v)
                    continue
                if ts in seen:
                    dup += 1; continue
                assert SD.n_of(s_) == n; seen.add(ts); batch.append(s_)
            lab = SD.label_chains(lab_m, batch, dev, bs=32)
            for q, (s_, o) in enumerate(zip(batch, lab)):
                if o["keep"] and o["cv"] and kn >= tgt:
                    drop += 1
                if o["keep"] and o["cv"] and kn < tgt:
                    kept.append({"syms": s_, "n": n, "chain_label": o["label"], "steps": o["steps"]}); kn += 1; tmax = max(tmax, len(s_) + 2)
                    if kn == tgt:
                        M["at_target"][str(n)] = {"draws": draws, "labelled": nlab + q + 1, "keep": kn / (nlab + q + 1)}
            nlab += len(batch)
        M["draws"][str(n)] = draws; M["dup"][str(n)] = dup; M["labelled"][str(n)] = nlab; M["dropped"][str(n)] = drop; M["keep_all"][str(n)] = (kn + drop) / nlab
        M["T_max"][str(n)] = tmax
        print("POOL n", n, "window", win_of[n], "kept", kn, "labelled", nlab, "keep %.4f" % M["keep_all"][str(n)], "draws", draws, "excl_hits",
              sum(h[str(n)] for h in hits.values()), "dup", dup, "%.0f s" % (time.time() - t0), flush=True)
    perm = np.random.default_rng(cfg["pool_seed"] + 1).permutation(len(kept)); fp = os.path.join(cfg["pool_dir"], cfg["pool_file"])
    with open(fp, "w", encoding="utf-8", newline="\n") as f:
        for i in perm:
            r = kept[int(i)]; assert set(r) == SD.REC_KEYS; f.write(json.dumps(r, separators=(",", ":")) + "\n")
    meta = {"lineage": a.lineage, "round": a.round, "seed": cfg["pool_seed"], "perm_seed": cfg["pool_seed"] + 1, "per_n": {str(n): int(v * a.scale) for n, v in cfg["per_n"].items()},
            "ns": sorted(cfg["per_n"]), "n_records": len(kept), "draws_n": M["draws"], "dup_n": M["dup"], "labelled_n": M["labelled"], "dropped_extra_n": M["dropped"],
            "keep_all": M["keep_all"], "keep_all_def": "kept-eligible (KEEP and CV) / all labelled, incl. the last chunk's overshoot",
            "keep_to_target": {k: v["keep"] for k, v in M["at_target"].items()}, "at_target": M["at_target"], "T_max_by_n": M["T_max"],
            "windows": {w: {"seed": s_, "ns": list(ns), "gate": g_, "labeller": wl[w], "labeller_sha256": sha256_file(os.path.join(wl[w], "model.pt"))} for w, (s_, ns, g_) in cfg["windows"].items()},
            "labeller": {w: wl[w] for w in cfg["windows"]}, "excl": {k: {"size": len(v), "hits_by_n": hits[k]} for k, v in S.items()},
            "collide_n": {str(n): sum(h[str(n)] for h in hits.values()) for n in cfg["per_n"]}, "gate_dir": cfg["gate_dir"], "sha256": sha256_file(fp),
            "builder_commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip(), "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else dev,
            "dtype": "fp32", "bs": 32, "cap": "n + 4", "filter": "KEEP and COMPOSE-VALID", "scale": a.scale, "sec": round(time.time() - t0)}
    json.dump(meta, open(os.path.join(cfg["pool_dir"], "pool_meta.json"), "w", newline="\n"), indent=1); print("DONE", meta["n_records"], meta["sha256"][:16], meta["sec"], flush=True)


if __name__ == "__main__":
    main()
