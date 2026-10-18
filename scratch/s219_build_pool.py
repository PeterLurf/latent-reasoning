"""§219 TEXT-SELF pool builder (B1; TS_v1.md at 146a25d0 §2.2, §2.4, §2.6 B1; from scratch/s216_build_pool.py). Local RTX 3070, fp32, tf32 off,
label batch 128. Two stages per round (textself.ROUNDS):
  --stage gates: the exclusion set (textself.exclusion_sets: test_set(1000, n_max 30), SERIAL, 213-37, every latent gate / on-disk latent pool, the
     precondition streams 23180 / 23187, the G-T18 items 23195, every s219 gate / pool already built) -> the gold patch -> each window's gate items (its
     own rng, n order, 2,000 per n, redrawn on an exclusion hit; syms + n only) -> <gate dir>/gate_items.json. No model. Then scratch/s219_gate.py.
  --stage pool: G-T15: reads <gate dir>/eval_gate.json's window_labeller (a window with none -> no pool, exit 3); round 2 also needs
     s219_t1_s1/eval_text.json (the seal, V3); the exclusion set + this round's gate items; the gold patch; per n (n order) 4,096-draw chunks labelled by
     the window's labeller (TF-GREEDY REROOT; --gen_windows W: per-step KV-cache generation instead, the G-T16 fallback), kept = KEEP-T and CV-T,
     exclusion hits and within-pool duplicates redrawn (counted per set), until the per-n target; one interleaved order (default_rng(perm seed)
     .permutation) -> <pool dir>/<pool file> (keys exactly {syms, n, chain_label, values}) + pool_meta.json.
  The labeller's chain is the model's own argmax only; gold functions raise from the patch on (G-T0).
    python -B scratch/s219_build_pool.py --round 1 --stage gates
    python -B scratch/s219_build_pool.py --round 1 --stage pool --device cuda:1
  Smokes: --out_root scratch/_tmp/s219/<tag>/ (pool / gate dirs re-rooted there) --scale 0.01 [--labeller_override <run dir>]."""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.textself as TS  # noqa: E402

N170_SHA = TS.file_sha256("boxes/nested170.py")


SMOKE_SEED_OFFSET = 100_000                                                 # smokes (--out_root) never draw a registered gate / pool stream


def cfg_for(rnd, out_root):
    cfg = dict(TS.ROUNDS[rnd]); cfg["windows"] = dict(cfg["windows"])
    if out_root:
        for k in ("pool_dir", "gate_dir"):
            cfg[k] = os.path.join(out_root, os.path.basename(cfg[k])).replace("\\", "/")
        for k in ("pool_seed", "perm_seed", "audit_seed", "kept_seed"):
            cfg[k] += SMOKE_SEED_OFFSET
        cfg["windows"] = {w: (s_ + SMOKE_SEED_OFFSET, ns, g_) for w, (s_, ns, g_) in cfg["windows"].items()}
    return cfg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--round", type=int, required=True, choices=[1, 2]); ap.add_argument("--stage", required=True, choices=["gates", "pool"])
    ap.add_argument("--device", default="cuda:1"); ap.add_argument("--label_bs", type=int, default=TS.LABEL_BS)
    ap.add_argument("--gen_windows", default="", help="G-T16 fail: these windows are labelled by per-step REROOT KV-cache generation (Theory notes the cost first)")
    ap.add_argument("--out_root", default="", help="smokes only: re-root the pool / gate dirs (under scratch/)"); ap.add_argument("--scale", type=float, default=1.0, help="smokes only")
    ap.add_argument("--labeller_override", default="", help="smokes only: one labeller run dir for every window"); a = ap.parse_args(); dev = a.device
    if a.out_root or a.scale != 1.0 or a.labeller_override:
        assert a.out_root and "scratch" in a.out_root.replace("\\", "/").split("/"), "smoke options need --out_root under scratch/"
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; t0 = time.time()
    t_start = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds")   # the seal-order field
    cfg = cfg_for(a.round, a.out_root); os.makedirs(cfg["pool_dir"], exist_ok=True); os.makedirs(cfg["gate_dir"], exist_ok=True)
    pool_fp = os.path.join(cfg["pool_dir"], cfg["pool_file"]); gate_fp = os.path.join(cfg["gate_dir"], "gate_items.json")
    if a.stage == "pool" and a.round == 2 and not a.out_root:
        assert os.path.exists(os.path.join(TS.ROUNDS[1]["cell"], "eval_text.json")), "§219 V3: the T2 pool only after s219_t1_s1/eval_text.json"
    S, smeta = TS.exclusion_sets(own_pool=pool_fp, own_gate=gate_fp, with_own_gate=(a.stage == "pool"))
    if a.stage == "pool" and a.out_root:                                    # a smoke's own gate items are not globbed (they live under scratch/)
        S["gates:own"] = TS._gate_syms(gate_fp)
    allx = set().union(*S.values())
    print("EXCL", {k: len(v) for k, v in S.items()}, "union", len(allx), "%.0f s" % (time.time() - t0), flush=True)
    TS.patch_gold()                                                         # G-T0: no gold function from here on
    if a.stage == "gates":
        gp = int(TS.GATE_PER_N * a.scale); out = {"windows": {}, "collide_n": {}, "per_n": gp, "round": a.round}
        for w, (seed, ns, gname) in cfg["windows"].items():
            rg = np.random.default_rng(seed); items = []
            for n in ns:
                got = 0; out["collide_n"]["%s_%d" % (w, n)] = 0
                while got < gp:
                    r_, _ = N.generate(rg, n); s_ = N.render(r_); del r_
                    if tuple(s_) in allx:
                        out["collide_n"]["%s_%d" % (w, n)] += 1; continue
                    assert TS.n_of(s_) == n; items.append({"syms": s_, "n": n}); allx.add(tuple(s_)); got += 1
            out["windows"][w] = {"seed": seed, "ns": list(ns), "gate": gname, "items": items}
        out.update({"excl": {k: len(v) for k, v in S.items()}, "excl_meta": smeta, "sec": round(time.time() - t0), "t_start": t_start})
        json.dump(out, open(gate_fp, "w", newline="\n"))
        print("GATE_ITEMS", {w: len(v["items"]) for w, v in out["windows"].items()}, out["collide_n"], "%.0f s" % (time.time() - t0), flush=True); return
    eg = json.load(open(os.path.join(cfg["gate_dir"], "eval_gate.json"))); wl = eg["window_labeller"]
    if a.labeller_override:
        wl = {w: a.labeller_override for w in cfg["windows"]}
    if any(wl.get(w) is None for w in cfg["windows"]):
        print("NO_POOL: a window has no passing labeller", wl, flush=True); sys.exit(3)
    models = {p: TS.load_model(os.path.join(p, "model.pt"), dev) for p in sorted(set(wl.values()))}
    gen_w = set(x for x in a.gen_windows.split(",") if x)
    win_of = {n: w for w, (_, ns, _) in cfg["windows"].items() for n in ns}
    rp = np.random.default_rng(cfg["pool_seed"]); kept, seen = [], set(); M = {k: {} for k in ("draws", "dup", "labelled", "dropped", "keep_all", "at_target", "T_max", "L0_max", "sec")}
    hits = {k: {} for k in S}
    for n in sorted(cfg["per_n"]):
        tgt = max(1, int(cfg["per_n"][n] * a.scale)); g = models[wl[win_of[n]]][0]; kn = nlab = draws = dup = drop = 0; tmax = l0 = 0; tn = time.time()
        for k in S:
            hits[k][str(n)] = 0
        while kn < tgt:
            batch = []
            while len(batch) < (4096 if a.scale == 1.0 else 64):           # smokes: 64-draw chunks
                r_, _ = N.generate(rp, n); s_ = N.render(r_); del r_; draws += 1; ts = tuple(s_)
                if ts in allx:
                    for k, v in S.items():
                        hits[k][str(n)] += int(ts in v)
                    continue
                if ts in seen:
                    dup += 1; continue
                assert TS.n_of(s_) == n; seen.add(ts); batch.append(s_)
            lab = TS.gen_reroot(g, batch) if win_of[n] in gen_w else TS.label_reroot(g, batch, a.label_bs)
            for q, (s_, o) in enumerate(zip(batch, lab)):
                if o["keep"] and kn >= tgt:
                    drop += 1
                if o["keep"] and kn < tgt:
                    kept.append(TS.record(s_, o["values"])); kn += 1; tmax = max(tmax, len(s_) + 2 + TS.c_struct(s_)); l0 = max(l0, len(s_))
                    if kn == tgt:
                        M["at_target"][str(n)] = {"draws": draws, "labelled": nlab + q + 1, "keep": kn / (nlab + q + 1)}
            nlab += len(batch)
        M["draws"][str(n)] = draws; M["dup"][str(n)] = dup; M["labelled"][str(n)] = nlab; M["dropped"][str(n)] = drop; M["keep_all"][str(n)] = (kn + drop) / nlab
        M["T_max"][str(n)] = tmax; M["L0_max"][str(n)] = l0; M["sec"][str(n)] = round(time.time() - tn)
        print("POOL n", n, "window", win_of[n], "kept", kn, "labelled", nlab, "keep %.4f" % M["keep_all"][str(n)], "draws", draws, "excl_hits",
              sum(h[str(n)] for h in hits.values()), "dup", dup, "%.0f s" % (time.time() - t0), flush=True)
    perm = np.random.default_rng(cfg["perm_seed"]).permutation(len(kept))
    with open(pool_fp, "w", encoding="utf-8", newline="\n") as f:
        for i in perm:
            r = kept[int(i)]; assert set(r) == TS.REC_KEYS; f.write(json.dumps(r, separators=(",", ":")) + "\n")
    try:
        bc = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:                                                       # noqa: BLE001
        bc = None
    meta = {"round": a.round, "lineage": "s1", "seeds": {"pool": cfg["pool_seed"], "perm": cfg["perm_seed"]}, "per_n": {str(n): max(1, int(v * a.scale)) for n, v in cfg["per_n"].items()},
            "ns": sorted(cfg["per_n"]), "n_records": len(kept), "draws_n": M["draws"], "dup_n": M["dup"], "labelled_n": M["labelled"], "dropped_extra_n": M["dropped"],
            "keep_n": M["keep_all"], "keep_def": "kept-eligible (KEEP-T and CV-T) / all labelled, incl. the last chunk's overshoot", "keep_to_target": {k: v["keep"] for k, v in M["at_target"].items()},
            "at_target": M["at_target"], "T_max_by_n": M["T_max"], "T_max_def": "max P + C over the kept records of n (P = len(syms) + 2, C = c_struct)", "L0_max_by_n": M["L0_max"],
            "sec_by_n": M["sec"], "windows": {w: {"seed": s_, "ns": list(ns), "gate": g_, "labeller": wl[w], "labeller_sha256": models[wl[w]][1],
                                                  "impl": "gen_reroot" if w in gen_w else "label_reroot (TF-GREEDY)"} for w, (s_, ns, g_) in cfg["windows"].items()},
            "labeller": {w: wl[w] for w in cfg["windows"]}, "labeller_sha256": {w: models[wl[w]][1] for w in cfg["windows"]},
            "excl": {k: {"size": len(v), "hits_by_n": hits[k]} for k, v in S.items()}, "excl_meta": smeta, "precondition_stream_sha": smeta["precondition_stream_sha"],
            "collide_n": {str(n): sum(h[str(n)] for h in hits.values()) for n in cfg["per_n"]}, "gate_dir": cfg["gate_dir"], "sha256": TS.file_sha256(pool_fp),
            "builder_commit": bc, "nested170_sha256": N170_SHA, "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else dev, "dtype": "fp32", "tf32": False,
            "label_bs": a.label_bs, "filter": "KEEP-T and CV-T", "collision_bound": "deleted latent pools not regenerated (Addendum 1 item 4): P(render = s) <= (1/11) 40^-n per draw",
            "scale": a.scale, "sec": round(time.time() - t0), "t_start": t_start}
    json.dump(meta, open(os.path.join(cfg["pool_dir"], "pool_meta.json"), "w", newline="\n"), indent=1); print("DONE", meta["n_records"], meta["sha256"][:16], meta["sec"], flush=True)


if __name__ == "__main__":
    main()
