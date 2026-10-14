"""SVR seed-stability draft G-pre (docs/THEORY_SEEDSTAB2_DRAFT_2026-10-14.md §4.1, 0c005392; Theory contract 2026-10-14, descriptive, eval-only):
nested170's step_recover (note 170-2: a multinomial logistic on r_t = _mem_read(h6 at the decoder's " =" position, S_t), standardised, L2 1e-3, LBFGS 200;
fit 4,000 train_items(rng 17021, n <= 8, the run's exclusion) / test 1,000 test items n <= 8 (rng 17022)) recomputed on three feature sets:
  ALL   : the 768-d read (nested170._mem_read = cat([g_add read_add (h_add heads), g_del read_del (h_del heads)]).flatten -> dims [0, h_add*64) are
          the ADDITIVE heads, [h_add*64, (h_add+h_del)*64) the gated-DELTA heads; head counts read from the model's own mem.h_add / mem.h_del)
  DELTA : the delta heads' slice;  ADD : the additive heads' slice.
The reads are computed ONCE (nested170.dec_reads, the run's K passes) and the three probes fit on slices of the same arrays. Also, for t >= 2, the test
accuracy of each probe split by the reduction height h (the height in the ORIGINAL tree of the node reduced at step t; height(leaf) = 0, op = 1 + max).
The run's exclusion set is rebuilt as nested170.main does (test items n >= excl_min_n + the 2,200-item n = 8 SERIAL pool, rng 17300), so train_items draws
the same 4,000 fit items. DeltaMem flags: FAST_ADD from args.json, CHUNK_DELTA 0. Autocast bf16 on a cuda card with capability >= 8 (the original eval:
3090 bf16), else fp32; --bs default = the run's eval_bs (250).
GATE: ALL must reproduce eval.json s175.step_recover at t = 1..4 within 0.01 (max |diff| reported; card / dtype recorded).
-> <run>/eval_svr_gpre.json
    python scratch/svr_gpre.py <run> [device cuda:1] [--bs N] [--memcheck]"""
import copy
import json
import os
import sys
import time

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.halt173 as H  # noqa: E402
import boxes.nested170 as N  # noqa: E402


def probe(Xf, yf, Xs, ys, dev, wd=1e-3, iters=200):
    """nested170.logreg_acc verbatim, returning the test predictions too (for the height split)."""
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    Xf_ = torch.tensor((Xf - mu) / sd, device=dev, dtype=torch.float32); Xs_ = torch.tensor((Xs - mu) / sd, device=dev, dtype=torch.float32)
    yf_ = torch.tensor(yf, device=dev, dtype=torch.long); ys_ = torch.tensor(ys, device=dev, dtype=torch.long)
    with torch.enable_grad(), torch.autocast("cuda", enabled=False):
        W = torch.zeros(Xf.shape[1], N.P, device=dev, requires_grad=True); b = torch.zeros(N.P, device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

        def clos():
            opt.zero_grad(); l_ = Fn.cross_entropy(Xf_ @ W + b, yf_) + wd * (W ** 2).sum(); l_.backward(); return l_
        opt.step(clos)
    ok = ((Xs_ @ W + b).argmax(1) == ys_).cpu().numpy()
    return float(torch.tensor(ok, dtype=torch.float32).mean()), ok


def red_heights(root):
    """heights (original tree) of the nodes reduced in nested170.reductions' order (leftmost ready handle)."""
    nd = copy.deepcopy(root); hgt = {}

    def seth(x):
        if x.op is None:
            hgt[id(x)] = 0; return 0
        hgt[id(x)] = 1 + max(seth(x.l), seth(x.r)); return hgt[id(x)]
    seth(nd); out = []
    while nd.op is not None:
        path = N._leftmost_ready(nd); tgt = nd
        for p_ in path:
            tgt = getattr(tgt, p_)
        out.append(hgt[id(tgt)]); tgt.v = N.apply(tgt.op, tgt.l.v, tgt.r.v); tgt.op = tgt.l = tgt.r = None
    return out


def main():
    run = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "cuda:1"; rd = os.path.join(ROOT, "boxes", "runs_fsq", run)
    kind, m, a = H.load_model(rd, dev); assert kind == "nested", kind
    bs = int(sys.argv[sys.argv.index("--bs") + 1]) if "--bs" in sys.argv else int(a.get("eval_bs") or 250)
    DM.FAST_ADD = bool(a.get("fast_add")); DM.CHUNK_DELTA = 0; K = int(a["think_k"]); t0 = time.time()
    amp = None if (not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    ac = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
    TE = N.test_set(int(a.get("per_n") or 1000)); EXCL = {tuple(it["syms"]) for it in TE if it["n"] >= int(a.get("excl_min_n", 1))}
    r_sp = np.random.default_rng(17300)
    for _ in range(int(a.get("serial_pool") or 2200)):
        r__, _t = N.generate(r_sp, 8); EXCL.add(tuple(N.render(r__)))
    tr = N.train_items(np.random.default_rng(17021), 4000, 8, EXCL); pool = [it for it in TE if it["n"] <= 8]
    te = [pool[i] for i in np.random.default_rng(17022).permutation(len(pool))[:1000]]
    if "--memcheck" in sys.argv:
        torch.cuda.reset_peak_memory_stats(dev); N.dec_reads(m, te[:bs], K, dev, ac, bs)
        print(run, "memcheck peak %.2f GiB at bs %d K %d" % (torch.cuda.max_memory_allocated(dev) / 2 ** 30, bs, K), flush=True); return
    Rf, Rs = N.dec_reads(m, tr, K, dev, ac, bs), N.dec_reads(m, te, K, dev, ac, bs)
    vf = [N.reductions(it["root"]) for it in tr]; vs = [N.reductions(it["root"]) for it in te]; hs = [red_heights(it["root"]) for it in te]
    HA, HD = int(m.mem.h_add), int(m.mem.h_del); DH = Rf.shape[2] // (HA + HD); assert DH * (HA + HD) == Rf.shape[2]
    sl = {"ALL": slice(0, Rf.shape[2]), "DELTA": slice(HA * DH, (HA + HD) * DH), "ADD": slice(0, HA * DH)}
    res = {k_: {} for k_ in sl}; by_h = {k_: {} for k_ in sl}; nn = {}
    for t in range(1, K + 1):
        fi = [i for i, v in enumerate(vf) if len(v) >= t]; si = [i for i, v in enumerate(vs) if len(v) >= t]
        if len(fi) < 100 or len(si) < 20:
            continue
        yf = np.array([vf[i][t - 1] for i in fi]); ys = np.array([vs[i][t - 1] for i in si]); ht = np.array([hs[i][t - 1] for i in si]); nn[str(t)] = {"fit": len(fi), "test": len(si)}
        for k_, s_ in sl.items():
            acc, ok = probe(Rf[t - 1, fi][:, s_], yf, Rs[t - 1, si][:, s_], ys, dev); res[k_][str(t)] = acc
            if t >= 2:
                by_h[k_][str(t)] = {str(h_): {"acc": float(ok[ht == h_].mean()), "n": int((ht == h_).sum())} for h_ in sorted(set(ht.tolist()))}
        print(run, "t", t, {k_: round(res[k_][str(t)], 4) for k_ in sl}, flush=True)
    ref = json.load(open(os.path.join(rd, "eval.json"))).get("s175", {}).get("step_recover") or {}
    dif = {t: abs(res["ALL"][t] - ref[t]) for t in ("1", "2", "3", "4") if t in ref and t in res["ALL"]}
    gate = {"ref": {t: ref.get(t) for t in ("1", "2", "3", "4")}, "abs_diff": dif, "max_abs_diff": max(dif.values()) if dif else None,
            "pass": bool(len(dif) == 4 and max(dif.values()) <= 0.01)}
    out = {"step_recover": res, "by_height": by_h, "n": nn, "gate_all_vs_eval": gate, "head_slices": {"ADD": [0, HA * DH], "DELTA": [HA * DH, (HA + HD) * DH], "h_add": HA, "h_del": HD, "dh": DH},
           "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "dtype": str(amp), "bs": bs, "K": K, "fast_add": DM.FAST_ADD, "sec": round(time.time() - t0),
           "orig_eval": {"eval_dtype": json.load(open(os.path.join(rd, "eval.json"))).get("eval_dtype"), "eval_bs": a.get("eval_bs")},
           "def": "SVR G-pre (THEORY_SEEDSTAB2 draft §4.1): nested170.step_recover on the ALL / DELTA / ADD slices of _mem_read's 768-d read; by_height = test acc split by the "
                  "original-tree height of the node reduced at step t (t >= 2)"}
    json.dump(out, open(os.path.join(rd, "eval_svr_gpre.json"), "w", newline="\n"), indent=1); print(run, "GATE", gate, flush=True)


if __name__ == "__main__":
    main()
