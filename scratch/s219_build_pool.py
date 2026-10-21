"""§219 TEXT-SELF pool builder (B1; TS_v1.md at 146a25d0 §2.2, §2.4, §2.6 B1; from scratch/s216_build_pool.py). Local RTX 3070, fp32, tf32 off,
label batch 128. Two stages per round (textself.ROUNDS):
  --stage gates: the exclusion set (textself.exclusion_sets: test_set(1000, n_max 30), SERIAL, 213-37, every latent gate / on-disk latent pool, the
     precondition streams 23180 / 23187, the G-T18 items 23195, every s219 gate / pool already built) -> the gold patch -> each window's gate items (its
     own rng, n order, 2,000 per n, redrawn on an exclusion hit; syms + n only) -> <gate dir>/gate_items.json. No model. Then scratch/s219_gate.py.
  --stage pool: G-T15: reads <gate dir>/eval_gate.json's window_labeller (a window with none -> no pool, exit 3); round 2 also needs
     s219_t1_s1/eval_text.json (the seal, V3); the exclusion set + this round's gate items; the gold patch; per n (n order) 4,096-draw chunks labelled by
     the window's labeller (TF-GREEDY REROOT only if eval_gate's G-T16 EQUIV for THAT labeller and window PASSed - equiv[W] for the primary,
     fallback[W].equiv for a BASE-LABELLED window; a failed or missing EQUIV, or --gen_windows W, labels the window by per-step KV-cache generation,
     TS_v1 §7 G-T16), the label settings asserted against eval_gate.label_settings (label_bs 128, fp32, tf32 off, the SAME card; G-T13 item 2),
     kept = KEEP-T and CV-T,
     exclusion hits and within-pool duplicates redrawn (counted per set), until the per-n target; one interleaved order (default_rng(perm seed)
     .permutation) -> <pool dir>/<pool file> (keys exactly {syms, n, chain_label, values}) + pool_meta.json.
  The labeller's chain is the model's own argmax only; gold functions raise from the patch on (G-T0).
  RETENTION GUARD (G-T13 item 3): round 2 (both stages) asserts that T1's pool exists, matches its pool_meta sha256 and is fully in T2's exclusion set
  (and T1's gate items too); it refuses to run otherwise. Keep s219_pool_t1_s1/ until T2's pool is built AND T2's eval_textself CPU parts are scored.
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


def window_impl(eg, cfg, wl, gen_w, label_bs, card_here, override=False):
    """G-T13 items 1 / 2 (TS_v1 §7 G-T16, §2.2): per window, the labeller's source (primary / BASE-LABELLED fallback), its label settings asserted against
    eval_gate.label_settings (label_bs 128, fp32, tf32 off, the gate's label card == this pool's card), and the labelling implementation: TF-GREEDY
    only if THAT labeller's EQUIV for the window PASSed, else per-step KV-cache generation. -> (generation windows, {W: reason})."""
    gen_w = set(gen_w); why = {}
    for w in cfg["windows"]:
        src = "primary" if (override or wl[w] == eg.get("labeller")) else "fallback"
        fbw = (eg.get("fallback") or {}).get(w) or {}
        assert src == "primary" or fbw.get("labeller") == wl[w], ("window labeller is neither the primary nor the window's fallback", w)
        ls = (eg.get("label_settings") or {}).get(src)
        assert isinstance(ls, dict), ("§219 G-T13 item 2: eval_gate.json has no label_settings for the %s labeller (re-run s219_gate.py --stage score)" % src)
        assert ls.get("label_bs") == label_bs == TS.LABEL_BS and ls.get("dtype") == "fp32" and ls.get("tf32") is False, ("label settings differ from the gate's", ls)
        assert ls.get("card") == card_here, ("§219 G-T13 item 2 (§218 labeller rule): the pool's card must be the gate's label card", ls.get("card"), card_here)
        eq = (eg.get("equiv") or {}).get(w) if src == "primary" else fbw.get("equiv")
        if w in gen_w:
            why[w] = "generation (--gen_windows)"
        elif not (isinstance(eq, dict) and eq.get("PASS") is True):
            gen_w.add(w); why[w] = "generation (G-T16 EQUIV %s for the %s labeller)" % ("FAILED" if isinstance(eq, dict) else "missing", src)
        else:
            why[w] = "TF-GREEDY (G-T16 EQUIV PASS, %s labeller)" % src
    return gen_w, why


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
    assert a.label_bs == TS.LABEL_BS, "§219 §2.2: label batch 128 (fixed)"
    S, smeta = TS.exclusion_sets(own_pool=pool_fp, own_gate=gate_fp, with_own_gate=(a.stage == "pool"))
    if a.stage == "pool" and a.out_root:                                    # a smoke's own gate items are not globbed (they live under scratch/)
        S["gates:own"] = TS._gate_syms(gate_fp)
    if a.round == 2:                                                        # G-T13 item 3: T1's pool / gates must be in T2's exclusions (never silently dropped)
        c1 = cfg_for(1, a.out_root); p1 = os.path.join(c1["pool_dir"], c1["pool_file"]).replace("\\", "/"); g1 = os.path.join(c1["gate_dir"], "gate_items.json").replace("\\", "/")
        assert os.path.exists(p1), ("§219 G-T13 item 3: T1's pool %s is missing; T2's gates / pool need it in the exclusion set. Restore it (pool_meta sha) "
                                    "before building T2 - RETENTION: keep T1's pool until T2's pool is built and T2's eval_textself CPU parts are scored" % p1)
        m1 = json.load(open(os.path.join(c1["pool_dir"], "pool_meta.json"))); assert TS.file_sha256(p1) == m1["sha256"], ("T1 pool sha != its pool_meta", p1)
        assert os.path.exists(g1), ("T1's gate items missing", g1)
        if a.out_root:                                                      # smokes: scratch trees are not globbed
            S["pool:" + p1] = TS._pool_syms(p1); S["gates:" + g1] = TS._gate_syms(g1)
        assert ("pool:" + p1) in S and len(S["pool:" + p1]) == m1["n_records"], ("T1's pool is not (fully) in T2's exclusion set", p1)
        assert ("gates:" + g1) in S, ("T1's gate items are not in T2's exclusion set", g1)
        print("T1 POOL IN EXCLUSION", p1, len(S["pool:" + p1]), "sha ok", flush=True)
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
    card_here = torch.cuda.get_device_name(dev) if dev.startswith("cuda") else dev
    gen_w, impl_why = window_impl(eg, cfg, wl, set(x for x in a.gen_windows.split(",") if x), a.label_bs, card_here, bool(a.labeller_override))
    for w in cfg["windows"]:
        print("WINDOW", w, "labeller", wl[w], "->", impl_why[w], flush=True)
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
                                                  "impl": "gen_reroot" if w in gen_w else "label_reroot (TF-GREEDY)", "impl_why": impl_why[w]}
                        for w, (s_, ns, g_) in cfg["windows"].items()},
            "labeller": {w: wl[w] for w in cfg["windows"]}, "labeller_sha256": {w: models[wl[w]][1] for w in cfg["windows"]},
            "excl": {k: {"size": len(v), "hits_by_n": hits[k]} for k, v in S.items()}, "excl_meta": smeta, "precondition_stream_sha": smeta["precondition_stream_sha"],
            "collide_n": {str(n): sum(h[str(n)] for h in hits.values()) for n in cfg["per_n"]}, "gate_dir": cfg["gate_dir"], "sha256": TS.file_sha256(pool_fp),
            "builder_commit": bc, "nested170_sha256": N170_SHA, "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else dev, "dtype": "fp32", "tf32": False,
            "label_bs": a.label_bs, "filter": "KEEP-T and CV-T", "collision_bound": "deleted latent pools not regenerated (Addendum 1 item 4): P(render = s) <= (1/11) 40^-n per draw",
            "scale": a.scale, "sec": round(time.time() - t0), "t_start": t_start}
    json.dump(meta, open(os.path.join(cfg["pool_dir"], "pool_meta.json"), "w", newline="\n"), indent=1); print("DONE", meta["n_records"], meta["sha256"][:16], meta["sec"], flush=True)


if __name__ == "__main__":
    main()
