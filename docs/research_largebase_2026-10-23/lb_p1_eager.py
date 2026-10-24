"""§231 P1' EAGER probe ($0; CPU fp32, <= 3 threads, Idle priority; spec frozen in Note 231-1 at c3fcdcbb; Theory scores). Adapted from
review_scripts/rv_lb_trained_probe.py to FROZEN bases (no INJECT model).

Question: at the last token of S_1's defining sentence (the EAGER position), does the frozen base's write source linearly carry v_1?
  Qwen2.5-0.5B (revision 060db649..., HF, fp32 CPU), input [<|endoftext|> 151643] + tok(" " + text) (lb_bind_probe's format):
    Q_RMS12   RMSNorm_12(h_12): layers[12].input_layernorm's output (forward hook)
    Q_H24     h_24 before the final norm: model.norm's input (forward hook; HF's last hidden_states entry is post-norm)
    Q_L0      control (a): the embedding states (hidden_states[0]) at the same position
  GPT-2 small (boxes s5.load_gpt2, the harness weights), input [EOT] + enc(" " + text) (inject_ow's think input):
    G_LN6     control (b): LN(h_6) with block 6's ln_1 (DeltaMem's write-source LN at init)
    G_HTOP    control (b): h_top = the residual after block 11, before ln_f
    G_L0      the embedding states (token + position) at the same position (descriptive)
Items: TOPO, frames 0-5, dag on, stream seed 23122 (unused by any registered set): d = 2 N 3,000 then d = 3 N 1,500 (same stream, that order).
Split per set by item sig: sha1(sig) mod 100 -> < 60 fit, < 80 val, else test.
Targets: d2 V1 (binding: v_1 at S_1's sentence end); d3 V1 and V2 at S_1's sentence end, V2S2 = v_2 at S_2's sentence end (descriptive).
S_1's defining sentence = the unique sentence with S_1's and S_0's names and no later spine name (not the question); S_2's = the unique sentence
with S_2's name, no later spine name, not the question (items where it is not unique are dropped from V2S2 only, counted).
Probe: multinomial logreg (11 classes), standardised on fit, L2 wd chosen on val from {1e-4, 1e-3, 1e-2}, LBFGS 300; test read once at that wd.
Chance: the test-set majority share (max class share on test) and the fit-majority's test acc, and 1 / 11. Per-item test predictions stored.
    python -B docs/research_largebase_2026-10-23/lb_p1_eager.py [--smoke] [--model Qwen/Qwen2.5-0.5B --revision 060db649...]"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "3"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(3)
import boxes.ow_words as OW  # noqa: E402

SEED = 23122; P = 11; QWEN = "Qwen/Qwen2.5-0.5B"; QBOS = 151643


def sentences(text):
    return [(m.start(), m.end()) for m in re.finditer(r"[^.?]*[.?]", text)]


def has(text, n, a, b):
    return re.search(r"\b%s\b" % re.escape(n), text[a:b]) is not None


def sent_end(it, k):
    """char end (exclusive) of S_k's defining sentence (k = 1: S_1 and S_0 named, no later spine name; k = 2: S_2 named, no later name), else None."""
    text = it["text"]; sp = it["names"]["spine"]; later = sp[k + 1:]
    cand = [(a, b) for a, b in sentences(text) if has(text, sp[k], a, b) and (k != 1 or has(text, sp[0], a, b))
            and not any(has(text, n, a, b) for n in later) and "How many" not in text[a:b]]
    return cand[0][1] if len(cand) == 1 else None


def split_of(sig):
    h = int(hashlib.sha1(sig.encode()).hexdigest()[:8], 16) % 100
    return "fit" if h < 60 else ("val" if h < 80 else "test")


@torch.no_grad()
def qwen_feats(tok, m, items, ends, bs):
    """-> {Q_RMS12, Q_H24, Q_L0: (n, D)} at the token whose char span holds the last char of " " + text[:end] (offset lookup, +1 for the BOS)."""
    cap = {}
    h1 = m.model.layers[12].input_layernorm.register_forward_hook(lambda mod, i, o: cap.__setitem__("rms12", o))
    h2 = m.model.norm.register_forward_hook(lambda mod, i, o: cap.__setitem__("h24", i[0]))
    out = {"Q_RMS12": [], "Q_H24": [], "Q_L0": []}
    try:
        for s in range(0, len(items), bs):
            ch, en = items[s:s + bs], ends[s:s + bs]; encs = [tok(" " + it["text"], return_offsets_mapping=True, add_special_tokens=False) for it in ch]
            ids = [[QBOS] + e["input_ids"] for e in encs]; T = max(len(q) for q in ids)
            x = torch.tensor([q + [QBOS] * (T - len(q)) for q in ids]); am = torch.tensor([[1] * len(q) + [0] * (T - len(q)) for q in ids])
            hs = m(input_ids=x, attention_mask=am, output_hidden_states=True).hidden_states   # right padding + causal: real rows unaffected
            for r, (e, c) in enumerate(zip(encs, en)):
                hit = [j for j, (a_, b_) in enumerate(e["offset_mapping"]) if a_ <= c < b_]   # c = the char index of the sentence's last char in " " + text
                assert len(hit) == 1, ("offset lookup", c); j = hit[0] + 1
                out["Q_RMS12"].append(cap["rms12"][r, j].float().numpy()); out["Q_H24"].append(cap["h24"][r, j].float().numpy()); out["Q_L0"].append(hs[0][r, j].float().numpy())
    finally:
        h1.remove(); h2.remove()
    return {k: np.stack(v) for k, v in out.items()}


@torch.no_grad()
def gpt2_feats(g, items, ends, bs):
    """-> {G_LN6, G_HTOP, G_L0} at index len(enc(" " + text[:end])) of [EOT] + enc(" " + text) (rv_lb_trained_probe.eager_pos's rule)."""
    import boxes.deltamem as DM
    import boxes.inject_ow as IO
    out = {"G_LN6": [], "G_HTOP": [], "G_L0": []}
    for s in range(0, len(items), bs):
        ch, en = items[s:s + bs], ends[s:s + bs]; th = [[IO.EOT] + IO.ENC.encode(" " + it["text"]) for it in ch]; T = max(len(q) for q in th)
        x = torch.tensor([q + [IO.EOT] * (T - len(q)) for q in th]); e0 = g.embed(x); B, T_, D = e0.shape; hd = D // g.H
        h0 = e0 + DM.wpe_slice(g, T_)[None]; h6 = DM.gpt_lower(g, e0, 6); h = h6
        for i in range(6, g.L):
            h = DM._block(g, i, h, hd)
        ln6 = F.layer_norm(h6, (D,), g.w("h.6.ln_1.weight"), g.w("h.6.ln_1.bias"))
        for r, (it, c) in enumerate(zip(ch, en)):
            j = len(IO.ENC.encode(" " + it["text"][:c]))                    # +1 (EOT) - 1 (last token)
            assert IO.ENC.decode([th[r][j]]).endswith("."), ("GPT-2 EAGER position is not the sentence's last token", it["sig"])
            out["G_LN6"].append(ln6[r, j].float().numpy()); out["G_HTOP"].append(h[r, j].float().numpy()); out["G_L0"].append(h0[r, j].float().numpy())
    return {k: np.stack(v) for k, v in out.items()}


def logreg(Xf, yf, Xs_list, wd, iters=300):
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    Xf_ = torch.tensor((Xf - mu) / sd, dtype=torch.float32); yf_ = torch.tensor(yf, dtype=torch.long)
    W = torch.zeros(Xf.shape[1], P, requires_grad=True); b = torch.zeros(P, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def clos():
        opt.zero_grad(); l_ = F.cross_entropy(Xf_ @ W + b, yf_) + wd * (W ** 2).sum(); l_.backward(); return l_
    opt.step(clos)
    with torch.no_grad():
        return [(torch.tensor((Xs - mu) / sd, dtype=torch.float32) @ W + b).argmax(1).numpy() for Xs in Xs_list]


def probe(X, y, sp, sigs):
    f, v, t = sp == "fit", sp == "val", sp == "test"; best = None
    for wd in (1e-4, 1e-3, 1e-2):
        pv, pt = logreg(X[f], y[f], [X[v], X[t]], wd); va = float((pv == y[v]).mean())
        if best is None or va > best["val_acc"]:
            best = {"wd": wd, "val_acc": va, "test_acc": float((pt == y[t]).mean()), "pred": pt}
    yt = y[t]; maj_fit = int(np.bincount(y[f], minlength=P).argmax())
    return {"wd": best["wd"], "val_acc": best["val_acc"], "test_acc": best["test_acc"], "test_max_share": float(np.bincount(yt, minlength=P).max() / len(yt)),
            "fit_majority_test_acc": float(np.mean(yt == maj_fit)), "uniform": 1 / P, "n": {"fit": int(f.sum()), "val": int(v.sum()), "test": int(t.sum())},
            "test_items": {"sig": [s for s, k in zip(sigs, t) if k], "label": yt.tolist(), "pred": best["pred"].tolist()}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=QWEN); ap.add_argument("--revision", default="060db6499f32faf8b98477b0a26969ef7d8b9987")
    ap.add_argument("--n2", type=int, default=3000); ap.add_argument("--n3", type=int, default=1500); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="N 40 / 20; prints shapes and counts only, writes nothing")
    ap.add_argument("--out", default=os.path.join(HERE, "lb_p1_eager.json")); a = ap.parse_args()
    if a.smoke:
        a.n2, a.n3 = 40, 20
    t0 = time.time(); st = OW.Stream(SEED, frames=OW.FRAME_TRAIN, order_mode="topo", dag=True)
    I2 = [st.draw(2) for _ in range(a.n2)]; I3 = [st.draw(3) for _ in range(a.n3)]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    qm = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.float32).eval()
    rev_full = getattr(qm.config, "_commit_hash", None)
    assert len(qm.model.layers) == 24, len(qm.model.layers)
    from boxes import s5
    g = s5.load_gpt2("cpu").eval()
    res = {"spec": "Note 231-1 (c3fcdcbb)", "seed": SEED, "qwen": {"model": a.model, "revision_arg": a.revision, "commit_hash": rev_full}, "sets": {}, "probes": {}}
    sets = {"d2": (I2, {"V1": 1}), "d3": (I3, {"V1": 1, "V2": 2})}
    for nm, (items, tg) in sets.items():
        e1 = [sent_end(it, 1) for it in items]; assert all(x is not None for x in e1), "S_1's defining sentence not unique"
        F_ = {**qwen_feats(tok, qm, items, [x - 1 + 1 for x in e1], a.bs), **gpt2_feats(g, items, e1, a.bs)}   # " " + text shifts chars by 1: last char = end - 1 + 1
        sigs = [it["sig"] for it in items]; sp = np.array([split_of(s) for s in sigs]); res["sets"][nm] = {"n": len(items), "split": {k: int((sp == k).sum()) for k in ("fit", "val", "test")}}
        for tn, k in tg.items():
            y = np.array([int(it["values"][k]) for it in items])
            for fn, X in F_.items():
                if a.smoke:
                    print(nm, tn, fn, X.shape, flush=True); continue
                res["probes"]["%s_%s_%s" % (nm, tn, fn)] = probe(X, y, sp, sigs)
        if nm == "d3":                                                      # V2 at S_2's own sentence end (descriptive)
            e2 = [sent_end(it, 2) for it in items]; keep = [i for i, x in enumerate(e2) if x is not None]; res["sets"]["d3"]["v2s2_kept"] = len(keep)
            sub = [items[i] for i in keep]; F2 = {**qwen_feats(tok, qm, sub, [e2[i] for i in keep], a.bs), **gpt2_feats(g, sub, [e2[i] for i in keep], a.bs)}
            y2 = np.array([int(it["values"][2]) for it in sub]); sg2 = [it["sig"] for it in sub]; sp2 = np.array([split_of(s) for s in sg2])
            for fn, X in F2.items():
                if a.smoke:
                    print("d3 V2S2", fn, X.shape, flush=True); continue
                res["probes"]["d3_V2S2_%s" % fn] = probe(X, y2, sp2, sg2)
    res["sec"] = round(time.time() - t0, 1); res["threads"] = torch.get_num_threads()
    res["features"] = {"Q_RMS12": "Qwen layers[12].input_layernorm output", "Q_H24": "Qwen model.norm input (h_24 pre-final-norm)", "Q_L0": "Qwen hidden_states[0] (embeddings)",
                       "G_LN6": "GPT-2 LN(h_6) with h.6.ln_1", "G_HTOP": "GPT-2 residual after block 11, pre ln_f", "G_L0": "GPT-2 token + position embedding"}
    if a.smoke:
        print("SMOKE ok", res["sets"], res["qwen"], round(time.time() - t0, 1), "s", flush=True); return
    with open(a.out, "w", newline="\n") as f:
        json.dump(res, f, indent=1); f.write("\n")
    print("P1' done", a.out, res["sec"], "s", flush=True)


if __name__ == "__main__":
    main()
