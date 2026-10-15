"""§205 IIT-SERIAL post-landing eval-only pass (note 205-1 contract; local, $0). -> <run>/eval_iit.json
  iit_eval.tau{1,2,3}: the training loss's own pairing (boxes/iit_serial.make_pairs, every eligible n 3-8 test item, rng [20520, tau]); the whole state
      after pass tau swapped from the donor x_d (tau passes on x_d, passes tau+1..K and the decoder on x); on CLEAN-ccc pairs (recipient x, donor x_d and
      the hybrid item all answered correctly without a swap, K' = K): hybrid_ccc = P(answer = y_hyb), own = P(answer = y), donor = P(answer = y_d), n_ccc;
      tau 1 is the trained pairing, tau 2 / 3 untrained. By n beside.
  siit_eval: the SIIT text x' (h's operands redrawn, y(x') != y) after S_1 of x: passes 2..K and the decoder on x'; keep_ccc = P(answer = y) on pairs whose
      x and x' are both answered correctly clean; n_ccc.
  g2_pass1_deep_probe: nested170's step_recover protocol (svr_gpre: the decoder-position read, standardised multinomial logistic, L2 1e-3, LBFGS 200; fit
      4,000 train_items rng 17021 with the run's exclusion, test 1,000 test items rng 17022, n <= 8) refitted on S_1's read for each target v_t (t >= 2),
      read on test targets whose reduction height h >= 4, pooled over t (by t beside).
Runner = scratch/dasused.Runner (bf16 autocast trunk on a cc >= 8 card, fp32 memory branch); DM.FAST_ADD from args.json, CHUNK_DELTA from args (0).
    python scratch/s205_eval_iit.py <run name> [--device cuda:1] [--bs 250]"""
import argparse
import json
import os
import sys
import time

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scratch")); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.halt173 as H  # noqa: E402
import boxes.iit_serial as IS  # noqa: E402
import boxes.nested170 as N  # noqa: E402
import dasused as DU  # noqa: E402
import svr_gpre as SG  # noqa: E402


def item(root, n):
    return {"syms": N.render(root), "root": root, "n": n, "t": N.value(root)}


def chunks(R_, rec, src, tau, K):
    """answers of `rec` with the state after pass tau replaced by `src`'s state after tau passes (whole state)."""
    out = []
    for i in range(0, len(rec), R_.bs):
        out.append(R_.answers(rec[i:i + R_.bs], K, override=(tau, R_.states(src[i:i + R_.bs], tau))))
    return np.concatenate(out) if out else np.zeros(0, dtype=int)


def shares(got, y, yh, yd, mask, nn):
    k = int(mask.sum())
    o = {"n_ccc": k, "hybrid_ccc": float((got == yh)[mask].mean()) if k else None, "own": float((got == y)[mask].mean()) if k else None,
         "donor": float((got == yd)[mask].mean()) if k else None}
    o["by_n"] = {str(n_): {"n": int((mask & (nn == n_)).sum()), "hybrid": float((got == yh)[mask & (nn == n_)].mean()) if (mask & (nn == n_)).any() else None}
                 for n_ in sorted(set(nn.tolist()))}
    return o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--device", default="cuda:1"); ap.add_argument("--bs", type=int, default=0)
    ap.add_argument("--model", default="model.pt"); ap.add_argument("--out", default="eval_iit.json"); a = ap.parse_args(); dev = a.device; T0 = time.time()
    rd = os.path.join(ROOT, "boxes", "runs_fsq", a.run); kind, m, ra = H.load_model(rd, dev) if a.model == "model.pt" else (None, None, None)
    if a.model != "model.pt":                                                # e.g. ckpt_model_8000.pt: build as load_model, then load that state
        import shutil, tempfile  # noqa: E401
        td = tempfile.mkdtemp(dir=os.path.join(ROOT, "scratch", "_tmp")); shutil.copy(os.path.join(rd, "args.json"), td)
        os.link(os.path.join(rd, a.model), os.path.join(td, "model.pt")) if os.name != "nt" else shutil.copy(os.path.join(rd, a.model), os.path.join(td, "model.pt"))
        kind, m, ra = H.load_model(td, dev); os.remove(os.path.join(td, "model.pt"))
    assert kind == "nested", kind
    DM.FAST_ADD = bool(ra.get("fast_add")); DM.CHUNK_DELTA = int(ra.get("chunk_delta") or 0); DM.READ_CLAMP = None; K = int(ra["think_k"])
    bs = a.bs or int(ra.get("eval_bs") or 250); R_ = DU.Runner(m, dev, bs); TE = N.test_set(int(ra.get("per_n") or 1000)); TEn = [it for it in TE if 3 <= it["n"] <= 8]
    out = {"run": a.run, "model": a.model, "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "dtype": DU.DTYPE if R_.amp is not None else "fp32",
           "bs": bs, "K": K, "fast_add": DM.FAST_ADD, "chunk_delta": DM.CHUNK_DELTA, "iit_eval": {}}
    for tau in (1, 2, 3):
        hyb = []; rec, don, yh, sx, yy, cnt = IS.make_pairs(TEn, np.random.default_rng([20520, tau]), tau, 0, 10 ** 9, hyb_out=hyb)
        hyb_items = [item(h, it["n"]) for h, it in zip(hyb, rec)]; yd = np.array([d["t"] for d in don]); nn = np.array([it["n"] for it in rec])
        assert all(hi["t"] == y_ for hi, y_ in zip(hyb_items, yh)), "hybrid item value != y_hyb"
        cx, cd, ch = R_.answers(rec, K), R_.answers(don, K), R_.answers(hyb_items, K); ccc = (cx == yy) & (cd == yd) & (ch == yh)
        got = chunks(R_, rec, don, tau, K); out["iit_eval"]["tau%d" % tau] = {**shares(got, yy, yh, yd, ccc, nn), "n_pairs": len(rec), "counts": cnt,
                                                                             "all": {"hybrid": float((got == yh).mean()), "own": float((got == yy).mean()), "n": len(rec)}}
        if tau == 1:                                                         # SIIT on the same tau-1 recipients
            ys = np.array([s["t"] for s in sx]); cs = R_.answers(sx, K); okp = (cx == yy) & (cs == ys); gs = chunks(R_, sx, rec, 1, K)
            out["siit_eval"] = {"keep_ccc": float((gs == yy)[okp].mean()) if okp.any() else None, "n_ccc": int(okp.sum()),
                                "follow_text_ccc": float((gs == ys)[okp].mean()) if okp.any() else None, "n_pairs": len(sx)}
        print("tau", tau, {k_: out["iit_eval"]["tau%d" % tau][k_] for k_ in ("hybrid_ccc", "own", "donor", "n_ccc")}, round(time.time() - T0), flush=True)
    # g2: step_recover protocol on S_1's read, targets v_t (t >= 2) with reduction height >= 4
    amp = R_.amp; ac = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
    EXCL = {tuple(it["syms"]) for it in TE if it["n"] >= int(ra.get("excl_min_n", 1))}; r_sp = np.random.default_rng(17300)
    for _ in range(int(ra.get("serial_pool") or 2200)):
        r__, _t = N.generate(r_sp, 8); EXCL.add(tuple(N.render(r__)))
    tr = N.train_items(np.random.default_rng(17021), 4000, 8, EXCL); pool = [it for it in TE if it["n"] <= 8]; te = [pool[i] for i in np.random.default_rng(17022).permutation(len(pool))[:1000]]
    Rf, Rs = N.dec_reads(m, tr, K, dev, ac, bs), N.dec_reads(m, te, K, dev, ac, bs)
    vf = [N.reductions(it["root"]) for it in tr]; vs = [N.reductions(it["root"]) for it in te]; hs = [SG.red_heights(it["root"]) for it in te]; okall = []; by_t = {}
    for t in range(2, K + 1):
        fi = [i for i, v in enumerate(vf) if len(v) >= t]; si = [i for i, v in enumerate(vs) if len(v) >= t and hs[i][t - 1] >= 4]
        if len(fi) < 100 or not si:
            continue
        acc, ok = SG.probe(Rf[0, fi], np.array([vf[i][t - 1] for i in fi]), Rs[0, si], np.array([vs[i][t - 1] for i in si]), dev)
        by_t[str(t)] = {"acc": float(acc), "n": len(si)}; okall += list(np.asarray(ok, bool))
    out["g2_pass1_deep_probe"] = float(np.mean(okall)) if okall else None
    out["g2_detail"] = {"by_t": by_t, "n": len(okall), "def": "step_recover protocol (svr_gpre.probe) fitted on S_1's decoder-position read per target t >= 2; test targets with h >= 4; pooled"}
    out["sec"] = round(time.time() - T0)
    json.dump(out, open(os.path.join(rd, a.out), "w", newline="\n"), indent=1); print("EVAL_IIT", {k_: out["iit_eval"][k_]["hybrid_ccc"] for k_ in out["iit_eval"]}, out["siit_eval"]["keep_ccc"], out["g2_pass1_deep_probe"], out["sec"], flush=True)


if __name__ == "__main__":
    main()
