"""THEORY §139 DISTILL-2 teacher probe (683318e2, note 139-1; $0, local): can a linear readout recover the running sum S_j from the
TC-M teacher's hidden states? Teacher = §98's text-CoT model (boxes/runs_mult/m98_tc_4x4_s1, GPT-2, acc_test 1.0).
* 4,096 train problems (bench 4x4 split, deduped against the 1,000 test items incl. commuted pairs; fixed draw, seed 13900); the teacher
  greedy-generates its own CoT + answer (mult98's tc decode: 56 tokens at the format's known counts).
* Teacher-decoded S_j at the format's fixed positions: S_1 = P_1 (step 1), S_2 / S_3 = the digits inside "( .. )" of steps 2 / 3,
  S_4 = the answer digits after " ####". Rows whose decoded S_j is not all digits are dropped (counted).
* Hidden = the residual stream after each block (teacher135.hidden_layers: h0 .. h12, pre ln_f) at the LAST token of each S_j text, on
  the teacher's own sequence (question + its generated text).
* Probes: per layer, one multinomial logistic readout per digit position (S_j zero-padded to 8 digits, LSB first; 10 classes), pooled
  over j, fit on the teacher-decoded digits of half the problems, scored on the other half (problem-level split).
teacher_probe.json: n, dropped, teacher_format_valid, teacher_sj_vs_gold_exact, heldout_exact_by_layer (all 8 digits = the teacher's S_j),
chosen_layer (argmax, ties -> the later layer), gate (>= 0.95), probe_vs_teacher_digit_agree, probe_vs_gold_exact (chosen layer,
held-out), label_source "teacher_generated"; probe_L<chosen>.pt (W, b, feature mean / std) for arm b's targets.
    python scratch/teacher_probe139.py --device cuda:0 --out boxes/runs_fsq/s139_teacher_probe"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BOXES_NB", "5")
try:                                                                        # local hard memory cap
    from boxes import memcap  # noqa: E402
except ImportError:
    memcap = None
from boxes import s5  # noqa: E402
from boxes import mult98 as M  # noqa: E402
from boxes import multcol as MC  # noqa: E402
from boxes.teacher135 import hidden_layers  # noqa: E402

SJ_POS = {1: (0, 5), 2: (13, 19), 3: (29, 36), 4: (48, 56)}               # [start, end) of S_j's digits in the 56 generated tokens
GEN_LEN = 56


def gold_sj(A, B, j):
    """S_j = a x (b_1 .. b_j) (b LSB first) -> 8 digits LSB first."""
    a = MC.to_int(A); b = MC.to_int(B[:, :j]); v = a * b
    return np.stack([(v // 10 ** i) % 10 for i in range(8)], 1)


def fit_probe(Xf, Yf, Xs, steps=400, lr=0.05, wd=1e-4, dev="cpu", seed=0):
    """8 digit heads x 10 classes, full-batch Adam on standardised features. Yf (N, 8). Returns (pred (Ns, 8), probs (Ns, 8, 10), state)."""
    torch.manual_seed(seed); Xf = torch.tensor(Xf, device=dev, dtype=torch.float32); Xs = torch.tensor(Xs, device=dev, dtype=torch.float32)
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6; Xf = (Xf - mu) / sd; Xs = (Xs - mu) / sd; Y = torch.tensor(Yf, device=dev)
    W = torch.zeros(Xf.shape[1], 80, device=dev, requires_grad=True); b = torch.zeros(80, device=dev, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad(); lg = (Xf @ W + b).view(-1, 8, 10); loss = F.cross_entropy(lg.reshape(-1, 10), Y.reshape(-1)); loss.backward(); opt.step()
    with torch.no_grad():
        P = torch.softmax((Xs @ W + b).view(-1, 8, 10), -1)
    return P.argmax(-1).cpu().numpy(), P.cpu().numpy(), {"W": W.detach().cpu(), "b": b.detach().cpu(), "mu": mu.cpu(), "sd": sd.cpu()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--teacher", default="boxes/runs_mult/m98_tc_4x4_s1"); ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    ap.add_argument("--train_limit", type=int, default=0, help="smoke only"); ap.add_argument("--diag", action="store_true", help="per-j / per-digit / fit-set / per-j probes / long fits -> teacher_probe_diag.json")
    a = ap.parse_args(); dev = a.device; os.makedirs(a.out, exist_ok=True); t0 = time.time(); memcap and memcap.apply(dev)
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    D = M.load(os.path.join(ROOT, "bench", "data", "mult_train", "4x4_train.txt"), limit=a.train_limit or None)
    teA, teB = MC.load_bench(os.path.join(ROOT, "bench", "data", "mult_4x4_test.txt"))
    A, B, rem = MC.dedup(D["A"], D["B"], teA, teB); rng = np.random.default_rng(13900); idx = rng.permutation(len(A))[:a.n]; A, B = A[idx], B[idx]; N = len(A)
    Dq = {"A": A, "B": B}
    g = s5.load_gpt2(dev); g.load_state_dict(torch.load(os.path.join(ROOT, a.teacher, "model.pt"), map_location=dev)); g.eval()
    gen = np.zeros((N, GEN_LEN), np.int64); H = None
    for s in range(0, N, a.bs):
        ii = np.arange(s, min(N, s + a.bs)); q = torch.tensor(M.question(Dq, ii), device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            last, past = M.seq_forward(g, q, 0, None); out = M.seq_greedy(g, last, past, GEN_LEN)
            ids = torch.cat([q, out], 1); hs = hidden_layers(g, ids)       # 13 x (b, T, 768)
        gen[ii] = out.cpu().numpy(); Lq = q.shape[1]
        pos = torch.tensor([Lq + SJ_POS[j][1] - 1 for j in (1, 2, 3, 4)], device=dev)
        Hb = torch.stack([h[:, pos].float() for h in hs], 0).cpu().numpy().astype(np.float16)   # (13, b, 4, 768)
        H = Hb if H is None else np.concatenate([H, Hb], 1)
    dig = {int(t): d for d, t in enumerate(M.DIGIT_IDS)}
    dec = np.full((N, 4, 8), -1); ok_fmt = np.ones(N, bool)
    for j in (1, 2, 3, 4):
        st, en = SJ_POS[j]
        for i in range(en - st):
            dec[:, j - 1, i] = [dig.get(int(t), -1) for t in gen[:, st + i]]
        dec[:, j - 1, en - st:] = 0                                         # zero-pad to 8 digits
    valid = (dec >= 0).all(2)                                               # (N, 4) per-j validity
    gold = np.stack([gold_sj(A, B, j) for j in (1, 2, 3, 4)], 1)            # (N, 4, 8)
    h = N // 2; fit_rows = [(i, j) for i in range(h) for j in range(4) if valid[i, j]]; sc_rows = [(i, j) for i in range(h, N) for j in range(4) if valid[i, j]]
    fi, fj = map(np.array, zip(*fit_rows)); si, sj = map(np.array, zip(*sc_rows))
    res = {"n": N, "teacher": a.teacher, "train_dedup_removed": rem, "dropped_rows": int((~valid).sum()), "label_source": "teacher_generated",
           "teacher_format_valid": float(valid.all(1).mean()), "teacher_sj_vs_gold_exact": float(((dec == gold).all(2) & valid).mean()),
           "heldout_exact_by_layer": {}, "eval_dtype": str(amp), "device": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"}
    best = None
    for L in range(H.shape[0]):
        Xf, Xs = H[L, fi, fj].astype(np.float32), H[L, si, sj].astype(np.float32)
        pred, probs, stt = fit_probe(Xf, dec[fi, fj], Xs, dev=dev)
        ex = float((pred == dec[si, sj]).all(1).mean()); res["heldout_exact_by_layer"][str(L)] = round(ex, 4)
        if best is None or ex >= best[0]:                                  # ties -> the later layer
            best = (ex, L, pred, stt)
        print("layer", L, "heldout_exact", round(ex, 4), flush=True)
    ex, L, pred, stt = best
    res.update({"chosen_layer": L, "chosen_heldout_exact": ex, "gate": bool(ex >= 0.95),
                "probe_vs_teacher_digit_agree": float((pred == dec[si, sj]).mean()), "probe_vs_gold_exact": float((pred == gold[si, sj]).all(1).mean()),
                "heldout_rows": int(len(si)), "fit_rows": int(len(fi)), "sec": round(time.time() - t0)})
    torch.save({**stt, "layer": L, "digits": 8, "classes": 10}, os.path.join(a.out, "probe_L%d.pt" % L))
    if a.diag:                                                              # diagnostics (gate failed): per-j / per-digit, fit-set accuracy, per-j probes, longer fits
        dg = {"chosen_layer": L, "per_j_exact": {}, "per_digit_agree": [round(float(v), 4) for v in (pred == dec[si, sj]).mean(0)]}
        for j in range(4):
            mk = sj == j; dg["per_j_exact"][str(j + 1)] = round(float((pred[mk] == dec[si[mk], j]).all(1).mean()), 4)
        pf, _, _ = fit_probe(H[L, fi, fj].astype(np.float32), dec[fi, fj], H[L, fi, fj].astype(np.float32), dev=dev); dg["fit_set_exact"] = round(float((pf == dec[fi, fj]).all(1).mean()), 4)
        dg["per_j_probe_exact_by_layer"] = {}; dg["long_fit_exact_by_layer"] = {}
        for L2 in range(4, H.shape[0]):
            pj_ = {}
            for j in range(4):
                f_, s_ = fi[fj == j], si[sj == j]; p2, _, _ = fit_probe(H[L2, f_, j].astype(np.float32), dec[f_, j], H[L2, s_, j].astype(np.float32), dev=dev)
                pj_[str(j + 1)] = round(float((p2 == dec[s_, j]).all(1).mean()), 4)
            dg["per_j_probe_exact_by_layer"][str(L2)] = pj_
            p3, _, _ = fit_probe(H[L2, fi, fj].astype(np.float32), dec[fi, fj], H[L2, si, sj].astype(np.float32), steps=2000, lr=0.02, wd=1e-5, dev=dev)
            dg["long_fit_exact_by_layer"][str(L2)] = round(float((p3 == dec[si, sj]).all(1).mean()), 4)
            print("diag layer", L2, pj_, dg["long_fit_exact_by_layer"][str(L2)], flush=True)
        json.dump(dg, open(os.path.join(a.out, "teacher_probe_diag.json"), "w", newline="\n"), indent=1)
    json.dump(res, open(os.path.join(a.out, "teacher_probe.json"), "w", newline="\n"), indent=1); print("PROBE", json.dumps({k: res[k] for k in ("chosen_layer", "chosen_heldout_exact", "gate", "probe_vs_teacher_digit_agree", "probe_vs_gold_exact", "teacher_sj_vs_gold_exact")}), flush=True)


if __name__ == "__main__":
    main()
