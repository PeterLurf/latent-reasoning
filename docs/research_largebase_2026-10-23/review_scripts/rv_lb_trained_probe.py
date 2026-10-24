"""REVIEW_LB check T ($0, CPU, 2 threads): where does pass-1 binding get lost in the TRAINED §228 model (boxes/runs_fsq/s228_plat2_s1, closed)?

D3 of LB_draft says pass-1 storage is limited by how linearly present v_1 is in the write source. On GPT-2 small's trained §228 model we can test
that link directly, without the frozen-probe proxy:
  U_EAGER  v_1 from the write source u = (1-gamma) LN(h6) + gamma LN_w(h_top) at the last token of S_1's defining sentence (pass 1, S_0 = 0)
  U_LAST   v_1 from u at the last think token ("?")
  U_MEAN   v_1 from the token-mean of u (masked)
  S1_ALL   v_1 from the pass-1 state S_1 (each 64x64 head flattened, random-projected 4096 -> 128, 12 heads -> 1536 features)
  S1_ADD / S1_DEL  the additive / delta heads only
  AUX      the model's own teacher-forced pass-1 aux value-token acc on the same items (training stat: 0.51)
Probe: multinomial logistic regression (standardised, L2, LBFGS), fit / val / test split; one fixed feature set each (no layer search).
Items: TOPO, frames 0-5, d alternating 1 / 2, stream seed 23121 (unused by any registered set). Not a blinded file; §228 is closed (229-1).
Usage: python -B rv_lb_trained_probe.py [--n 3000] [--out rv_lb_trained_probe.json]
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(2)
import boxes.ow_words as OW  # noqa: E402
import boxes.inject_ow as IO  # noqa: E402
import boxes.inject_ow_eval as IOE  # noqa: E402
import boxes.deltamem as DM  # noqa: E402

RUN = os.path.join(ROOT, "boxes", "runs_fsq", "s228_plat2_s1")
P = 11


def logreg(Xf, yf, Xs_list, wd=1e-3, iters=300):
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    Xf_ = torch.tensor((Xf - mu) / sd, dtype=torch.float32); yf_ = torch.tensor(yf, dtype=torch.long)
    W = torch.zeros(Xf.shape[1], P, requires_grad=True); b = torch.zeros(P, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def clos():
        opt.zero_grad(); l_ = F.cross_entropy(Xf_ @ W + b, yf_) + wd * (W ** 2).sum(); l_.backward(); return l_
    opt.step(clos)
    with torch.no_grad():
        return [(torch.tensor((Xs - mu) / sd, dtype=torch.float32) @ W + b).argmax(1).numpy() for Xs in Xs_list]


def probe(X, y, nf, nv, dd, wds=(1e-4, 1e-3, 1e-2)):
    """dd: per-item depth; test accuracy is also reported by d (d1 / d2)."""
    fit, val, te = slice(0, nf), slice(nf, nf + nv), slice(nf + nv, None); best = None
    for wd in wds:                                                      # L2 chosen on val, read once on test
        pv, pt = logreg(X[fit], y[fit], [X[val], X[te]], wd)
        va = float((pv == y[val]).mean()); ta = float((pt == y[te]).mean())
        if best is None or va > best["val"]:
            best = {"wd": wd, "val": va, "test": ta, "test_d1": float((pt == y[te])[dd[te] == 1].mean()) if (dd[te] == 1).any() else None,
                    "test_d2": float((pt == y[te])[dd[te] == 2].mean()) if (dd[te] == 2).any() else None}
    best["majority_test"] = float(np.mean(y[te] == np.bincount(y[fit], minlength=P).argmax())); return best


def eager_pos(it):
    """token index (in think = [EOT] + enc(' ' + text)) of the last token of S_1's defining sentence (lb_bind_probe's rule)."""
    import re
    text = it["text"]; sp = it["names"]["spine"]; later = sp[2:]
    sents = [(m.start(), m.end()) for m in re.finditer(r"[^.?]*[.?]", text)]
    def has(n, a, b):
        return re.search(r"\b%s\b" % re.escape(n), text[a:b]) is not None
    cand = [(a, b) for a, b in sents if has(sp[1], a, b) and has(sp[0], a, b) and not any(has(n, a, b) for n in later) and "How many" not in text[a:b]]
    assert len(cand) == 1
    return len(IO.ENC.encode(" " + text[:cand[0][1]]))                  # +1 (EOT) -1 (last token) = len(prefix tokens)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=3000); ap.add_argument("--bs", type=int, default=50)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "rv_lb_trained_probe.json")); a = ap.parse_args(argv)
    t0 = time.time(); args, m, g = IOE.load(RUN, "cpu"); mem = m.mem
    st = OW.Stream(23121, frames=OW.FRAME_TRAIN, order_mode="topo", dag=True); raw = [st.draw(1 + (i % 2)) for i in range(a.n)]
    items = [IO.tok_ow(it, "value") for it in raw]
    R = torch.randn(4096, 128, generator=torch.Generator().manual_seed(5)) / 128 ** 0.5
    feats = {k: [] for k in ("U_EAGER", "U_LAST", "U_MEAN", "S1_ALL")}; y = []; aux_ok = []; dot_ok = 0
    with torch.no_grad():
        for i in range(0, a.n, a.bs):
            ch, rw = items[i:i + a.bs], raw[i:i + a.bs]; x, msk = m.think_inputs(ch, "cpu"); B = len(ch)
            cap = {}; _, S1 = DM.gpt_forward_mem_r(g, g.embed(x), mem, mem.zero_state(B, "cpu"), cap=cap, mask=msk)
            u = (1 - mem.gamma) * mem.ln(cap["h6"].float()) + mem.gamma * mem.ln_w(cap["hw"].float())
            for r, (it, ri) in enumerate(zip(ch, rw)):
                p = eager_pos(ri); dot_ok += int(IO.ENC.decode([it["think"][p]]).endswith("."))
                L = len(it["think"]); feats["U_EAGER"].append(u[r, p].numpy()); feats["U_LAST"].append(u[r, L - 1].numpy())
                feats["U_MEAN"].append(u[r, :L].mean(0).numpy()); y.append(int(ri["values"][1]))
            hs = S1[:, :12].reshape(B, 12, 4096) @ R; feats["S1_ALL"].extend(list(hs.reshape(B, -1).numpy()))
            lg, ya = m.aux_logits(S1, ch, 1, "cpu"); pr = lg.argmax(-1)
            for r, it in enumerate(ch):
                vi = it["vidx"][0]; aux_ok.append(int(pr[r, vi] == ya[r, vi]))
    y = np.array(y); X = {k: np.stack(v) for k, v in feats.items()}
    X["S1_ADD"] = X["S1_ALL"][:, :6 * 128]; X["S1_DEL"] = X["S1_ALL"][:, 6 * 128:]
    nf, nv = int(0.6 * a.n), int(0.15 * a.n); te = slice(nf + nv, None); d_te = np.array([it["d"] for it in raw])[te]
    res = {"run": "s228_plat2_s1", "gamma": float(mem.gamma), "fade": float(mem.fade), "n": a.n, "n_fit": nf, "n_val": nv, "n_test": a.n - nf - nv,
           "eager_pos_is_period_share": dot_ok / a.n, "probes": {}}
    aux_ok = np.array(aux_ok); res["AUX_pass1_value_acc"] = {"all": float(aux_ok.mean()), "test": float(aux_ok[te].mean()),
                                                              "d1": float(aux_ok[np.array([it["d"] for it in raw]) == 1].mean()), "d2": float(aux_ok[np.array([it["d"] for it in raw]) == 2].mean())}
    dd = np.array([it["d"] for it in raw]); i2 = np.where(dd == 2)[0]; n2 = len(i2); f2, v2 = int(0.6 * n2), int(0.15 * n2)
    for k, Xk in X.items():
        res["probes"][k] = probe(Xk, y, nf, nv, dd)
        res["probes"][k + "_fit_d2_only"] = probe(Xk[i2], y[i2], f2, v2, dd[i2])   # fit / val / test on d = 2 items only
    res["sec"] = round(time.time() - t0, 1); res["test_d_counts"] = {"1": int((d_te == 1).sum()), "2": int((d_te == 2).sum())}
    with open(a.out, "w", newline="\n") as f:
        json.dump(res, f, indent=1); f.write("\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
