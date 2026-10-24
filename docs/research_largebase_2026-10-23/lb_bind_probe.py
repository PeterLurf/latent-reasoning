"""LB L0 BIND-PROBE ($0; frozen base, forward only; registered in LB_draft.md §5). Not run for a reading at drafting time: only --smoke was run
(tiny N, prints shapes / counts, never an accuracy).

Question: does the frozen base already carry the name -> value binding that INJECT's pass-1 write must store (the §228 / §229 failure:
pass-1 aux value acc 0.51 / 0.55)? Three linear-probe targets on TOPO items (d ~ {1, 2} alternating, train frames 0-5, stream seed 23111):
  OPD    (binding, lookup only; BINDING PREDICTION): at the first mention of S_0's name inside S_1's defining sentence, decode S_0's value
         (stated earlier as "<S_0> has <v> <u>."). 11 classes.
  EAGER  (descriptive): at the last token of S_1's defining sentence, decode v_1 (one operation mod 11 on bound values).
  ANSQ   (descriptive, d = 1 only): at the final "?" token, decode the answer (= v_1).
Features: every hidden state 0..L (HF output_hidden_states) at that position, input [BOS] + " " + text (BOS = the harness's EOT analogue:
GPT-2 <|endoftext|> 50256, Qwen2.5 <|endoftext|> 151643). Probe: multinomial logistic regression (standardised, L2 1e-3, LBFGS 200), fit on
N_FIT items, layer chosen on N_VAL, read on N_TEST (the chosen layer only; the per-layer test curve is stored but descriptive).
S_1's defining sentence = the unique sentence containing S_1's and S_0's names and no later spine name (asserted unique).
Usage: python -B lb_bind_probe.py --model <hf dir or id> --revision <sha> --name qwen25_05b --device cuda:0 --out scratch/gate_lb/bind_<name>.json
       python -B lb_bind_probe.py --smoke ... (N 24 / 8 / 8, no accuracies printed or written)
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.ow_words as OW  # noqa: E402

torch.set_num_threads(2)
SEED_PROBE = 23111                       # unused by any registered OW set (22900-22997; lb_tok_check uses 23101)
N_FIT, N_VAL, N_TEST = 3000, 500, 1000
P = 11


def sentences(text):
    return [(m.start(), m.end()) for m in re.finditer(r"[^.?]*[.?]", text)]


def name_spans(text, name, lo, hi):
    return [(m.start(), m.end()) for m in re.finditer(r"\b%s\b" % re.escape(name), text[lo:hi])]


def targets(it):
    """-> dict of char positions (end char, exclusive) and labels in it['text'] for OPD / EAGER / ANSQ."""
    text = it["text"]; sp = it["names"]["spine"]; later = sp[2:]
    cand = [(a, b) for a, b in sentences(text) if name_spans(text, sp[1], a, b) and name_spans(text, sp[0], a, b) and not any(name_spans(text, n, a, b) for n in later)
            and "How many" not in text[a:b]]
    assert len(cand) == 1, ("S_1's defining sentence not unique", it["sig"], len(cand))
    a, b = cand[0]; s0 = name_spans(text, sp[0], a, b)[0]
    out = {"OPD": (a + s0[1], int(it["values"][0])), "EAGER": (b, int(it["values"][1]))}
    if it["d"] == 1:
        out["ANSQ"] = (len(text), int(it["answer"]))
    return out


def load(model, revision, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    m = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=dtype).to(device).eval()
    return tok, m


def bos_id(tok):
    for t in ("<|endoftext|>",):
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i != tok.unk_token_id:
            return i
    return tok.bos_token_id


@torch.no_grad()
def features(tok, m, items, device, bs=16):
    """-> {target: (X (n, L+1, D) float32, y (n,))}; positions = the token whose char span contains the target's last char."""
    B = bos_id(tok); feats = {k: ([], []) for k in ("OPD", "EAGER", "ANSQ")}
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]; encs = [tok(" " + it["text"], return_offsets_mapping=True, add_special_tokens=False) for it in chunk]
        ids = [[B] + e["input_ids"] for e in encs]; T = max(len(x) for x in ids); pad = B
        x = torch.tensor([q + [pad] * (T - len(q)) for q in ids], device=device); am = torch.tensor([[1] * len(q) + [0] * (T - len(q)) for q in ids], device=device)
        hs = m(input_ids=x, attention_mask=am, output_hidden_states=True).hidden_states       # L+1 x (b, T, D); right padding, causal: unaffected
        H = torch.stack(hs, 1).float().cpu().numpy()                                           # (b, L+1, T, D)
        for r, (it, e) in enumerate(zip(chunk, encs)):
            for k, (cpos, lab) in targets(it).items():
                c = cpos + 1 - 1                                                               # char index of the target's last char in " " + text
                hit = [j for j, (s_, e_) in enumerate(e["offset_mapping"]) if s_ <= c < e_]
                assert len(hit) == 1, ("offset lookup", it["sig"], k)
                feats[k][0].append(H[r, :, hit[0] + 1]); feats[k][1].append(lab)              # +1: the BOS token
    return {k: (np.stack(v[0]), np.array(v[1])) for k, v in feats.items() if v[0]}


def logreg(Xf, yf, Xs_list, dev, wd=1e-3, iters=200):
    import torch.nn.functional as F_
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    Xf_ = torch.tensor((Xf - mu) / sd, device=dev, dtype=torch.float32); yf_ = torch.tensor(yf, device=dev, dtype=torch.long)
    W = torch.zeros(Xf.shape[1], P, device=dev, requires_grad=True); b = torch.zeros(P, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def clos():
        opt.zero_grad(); l_ = F_.cross_entropy(Xf_ @ W + b, yf_) + wd * (W ** 2).sum(); l_.backward(); return l_
    with torch.enable_grad():
        opt.step(clos)
    with torch.no_grad():
        return [(torch.tensor((Xs - mu) / sd, device=dev, dtype=torch.float32) @ W + b).argmax(1).cpu().numpy() for Xs in Xs_list]


def run_probe(F, n_fit, n_val, dev):
    X, y = F; L1 = X.shape[1]; fit, val, te = slice(0, n_fit), slice(n_fit, n_fit + n_val), slice(n_fit + n_val, None)
    val_acc, test_acc = [], []
    for l in range(L1):
        pv, pt = logreg(X[fit, l], y[fit], [X[val, l], X[te, l]], dev)
        val_acc.append(float((pv == y[val]).mean())); test_acc.append(float((pt == y[te]).mean()))
    best = int(np.argmax(val_acc[1:]) + 1)                                                     # layer >= 1 (0 = embeddings)
    maj = float(np.mean(y[te] == np.bincount(y[fit], minlength=P).argmax()))
    return {"layer_sel": best, "test_acc_sel": test_acc[best], "val_acc": val_acc, "test_acc_by_layer": test_acc, "majority_test": maj,
            "n_fit": int(len(y[fit])), "n_val": int(len(y[val])), "n_test": int(len(y[te]))}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--revision", default=None); ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"]); ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args(argv)
    nf, nv, nt = (24, 8, 8) if a.smoke else (N_FIT, N_VAL, N_TEST)
    st = OW.Stream(SEED_PROBE, frames=OW.FRAME_TRAIN, order_mode="topo", dag=True); items = [st.draw(1 + (i % 2)) for i in range(nf + nv + nt)]
    t0 = time.time(); tok, m = load(a.model, a.revision, a.device, getattr(torch, a.dtype)); F = features(tok, m, items, a.device)
    res = {"name": a.name, "model": a.model, "revision": a.revision, "dtype": a.dtype, "seed": SEED_PROBE, "n_layers_plus_emb": int(F["OPD"][0].shape[1]),
           "items_sha": OW.set_sha(items), "module_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(), "smoke": a.smoke, "probes": {}}
    for k in ("OPD", "EAGER", "ANSQ"):
        if k not in F:
            continue
        X, y = F[k]
        if k == "ANSQ":                                                                        # d = 1 items only (about half)
            n1 = len(y); f_ = int(n1 * nf / (nf + nv + nt)); v_ = int(n1 * nv / (nf + nv + nt)); res["probes"][k] = run_probe((X, y), f_, v_, a.device)
        else:
            res["probes"][k] = run_probe((X, y), nf, nv, a.device)
    res["sec"] = round(time.time() - t0, 1)
    if a.smoke:
        print("SMOKE OK", a.name, {k: (F[k][0].shape, int(F[k][1].min()), int(F[k][1].max())) for k in F},
              "sec", res["sec"])                                                               # no accuracy printed (pre-registration)
        for k in res["probes"]:
            for f_ in ("layer_sel", "test_acc_sel", "val_acc", "test_acc_by_layer", "majority_test"):
                res["probes"][k][f_] = "withheld (smoke)"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="\n") as f:
        json.dump(res, f, indent=1); f.write("\n")


if __name__ == "__main__":
    main()
