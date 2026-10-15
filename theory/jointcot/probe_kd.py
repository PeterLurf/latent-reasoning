"""JOINT-COT-L1 coupling probes (CPU, <= 3 threads, idle priority; reads D:\\Latent read-only, writes only next to this script).

For docs/THEORY_JOINTCOT_DRAFT_2026-10-14.md section 3 / 4 (descriptive, POST-HOC; no run is selected on it).
Formats (boxes/gsm8k_data.encode; C's TEXT-COT sequence is pre + cot + a1 + EOS):
  T1 (teacher, full CoT)        = pre + lines[0..m-1] + a1        coupling position = the "###" token
  T2 (teacher, final step drop) = pre + lines[0..m-2] + a1  (m>=2) coupling position = the "###" token
  S0 (student decoder, init)    = [EOT] + pre + a1                coupling position = the "###" token
S0 is the student's decoder exactly at init (W_o = 0, fade = 0: deltamem.read_fixed adds 0), i.e. the memory-free direct path (K'0).
States: block outputs l = 1..12 (pre ln_f), computed with deltamem._block (gate: final ln_f output == GPT2KV.forward).
Per model (raw GPT-2; §200 C = boxes/runs_fsq/gsm_txt_aug_s1/model.pt), per format T1 / T2, per layer:
  sigma_T (std over items x dims, mean over batches of 64), L1_init = mean |z_T - h_S0| / sigma_T, L1_const = the best item-independent
  constant (per-dim median of z_T, fit on train), L1_lin = a ridge map h_S0 -> z_T (fit on train, scored on valid), R2_lin, top5_share.
  ans_top1: the first answer token (a1[1]) by the LM head on ln_f(z) at layer 12 for T1 / T2 / S0 / the ridge imitation.
COUNTER (per-pass coupling option): T1 states at the last token of lines t = 1..3 (m >= 3 items), layers 7..12; share of the pass-varying
  part delta(t) = z(t) - mean_t z(t) explained by its item-independent mean c(t) (a pass counter).
LEAK (m = 1): T2 for m = 1 is pre + a1, the student's direct input without EOT; L1(T2, S0) per layer.
"""
import ctypes
import json
import os
import sys
import time

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True; os.environ["BOXES_NB"] = "5"
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)   # IDLE_PRIORITY_CLASS
except Exception:
    pass
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

torch.set_num_threads(3)
ROOT = r"D:\Latent"; sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
import boxes.gsm8k_data as G  # noqa: E402
from boxes.prosqa168s2 import ANS, ENC, EOT, pad  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__)); T0 = time.time()
N_TR = int(os.environ.get("N_TR", 600)); N_VA = int(os.environ.get("N_VA", 500)); BS = 16; LAYERS = list(range(1, 13))


def log(*a):
    s = " ".join(str(x) for x in a); print("[%6.0fs] %s" % (time.time() - T0, s), flush=True)
    with open(os.path.join(OUT, "probe_kd.log"), "a", encoding="utf-8", newline="\n") as f:
        f.write(s + "\n")


def build(r):
    pre = ENC.encode(r["q"] + "\n"); a1 = ENC.encode("### " + r["ans"]); assert a1[0] == ANS and len(a1) >= 2
    L = [ENC.encode("<<%s>>\n" % e) for e in r["eqs"]]; m = len(L); cot = [t for l_ in L for t in l_]
    assert cot == G.encode(r, ENC, 50256)["cot"]
    T1 = pre + cot + a1; p1 = len(pre) + len(cot)
    ends = [len(pre) + sum(len(l_) for l_ in L[:t]) - 1 for t in range(1, m + 1)]
    T2 = pre + [t for l_ in L[:-1] for t in l_] + a1 if m >= 1 else None; p2 = len(T2) - len(a1) if T2 is not None else None
    S0 = [EOT] + pre + a1; ps = 1 + len(pre)
    assert T1[p1] == ANS and S0[ps] == ANS and (T2 is None or T2[p2] == ANS)
    return {"m": m, "T1": (T1, [p1] + ends[:3]), "T2": (T2, [p2]) if T2 is not None else None, "S0": (S0, [ps]), "y1": a1[1]}


@torch.no_grad()
def states(g, seqpos):
    """block outputs (pre ln_f) at the given positions: list of (n_pos, 12, 768) float32 arrays."""
    hd = 768 // g.H; out = []
    for i in range(0, len(seqpos), BS):
        ch = seqpos[i:i + BS]; ids_, _ = pad([s for s, _ in ch]); x = torch.tensor(ids_)
        h = g.embed(x) + DM.wpe_slice(g, x.shape[1])[None]; col = [[] for _ in ch]
        for li in range(g.L):
            h = DM._block(g, li, h, hd)
            for r, (_, p) in enumerate(ch):
                col[r].append(h[r, p])                                      # (n_pos, D)
        if i == 0:                                                          # gate: the collected forward is GPT2KV.forward
            ref = g(g.embed(x), None)[0]; mine = Fn.layer_norm(h, (768,), g.w("ln_f.weight"), g.w("ln_f.bias"))
            states.gate = max(getattr(states, "gate", 0.0), float((ref - mine).abs().max()))
        out += [torch.stack(c_, 1).float().numpy() for c_ in col]           # (n_pos, 12, D)
    return out


def ans_top1(g, Z, y):
    with torch.no_grad():
        z = torch.tensor(Z); lg = g.logits(Fn.layer_norm(z, (768,), g.w("ln_f.weight"), g.w("ln_f.bias")))
        return float((lg.argmax(-1).numpy() == np.array(y)).mean())


def sig_batch(Z):
    return float(np.mean([Z[i:i + 64].std() for i in range(0, len(Z), 64) if len(Z[i:i + 64]) >= 8]))


def ridge(Xtr, Ytr, Xva, lam_grid=(1e-1, 1e0, 1e1, 1e2, 1e3)):
    mu_x, mu_y = Xtr.mean(0), Ytr.mean(0); sx = Xtr.std(0) + 1e-6; A = (Xtr - mu_x) / sx; B = Ytr - mu_y
    n_in = int(0.8 * len(A)); best = None
    for lam in lam_grid:                                                    # lambda by an inner split (squared error)
        W = np.linalg.solve(A[:n_in].T @ A[:n_in] + lam * n_in * np.eye(A.shape[1]) / 100.0, A[:n_in].T @ B[:n_in])
        e = float(((A[n_in:] @ W - B[n_in:]) ** 2).mean())
        if best is None or e < best[1]:
            best = (lam, e)
    W = np.linalg.solve(A.T @ A + best[0] * len(A) * np.eye(A.shape[1]) / 100.0, A.T @ B)
    return ((Xva - mu_x) / sx) @ W + mu_y, best[0]


def main():
    open(os.path.join(OUT, "probe_kd.log"), "w").close()
    Rtr, _ = G.load_aug("train"); Rva, _ = G.load_aug("valid", filter_bad=False); rng = np.random.default_rng(31416)
    m_tr = np.array([len(r["eqs"]) for r in Rtr]); res = {"train_m_share": {str(k): float((m_tr == k).mean()) for k in range(0, 9)}}
    tr = [build(Rtr[i]) for i in rng.permutation(len(Rtr))[:4 * N_TR]]; tr = [b for b in tr if b["m"] >= 2][:N_TR]
    va_all = [build(r) for r in Rva[:N_VA]]; va = [b for b in va_all if b["m"] >= 2]; va1 = [b for b in va_all if b["m"] == 1]
    va3 = [b for b in va if b["m"] >= 3]
    res.update({"n_train_m2": len(tr), "n_valid_m2": len(va), "n_valid_m1": len(va1), "n_valid_m3": len(va3)})
    log("items", res["n_train_m2"], res["n_valid_m2"], res["n_valid_m1"], res["n_valid_m3"])
    models = {"raw": None, "C": os.path.join(ROOT, "boxes", "runs_fsq", "gsm_txt_aug_s1", "model.pt")}
    for mn, path in models.items():
        g = s5.load_gpt2("cpu")
        if path:
            g.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        g.eval(); R = {}
        Z = {}
        for split, items in (("tr", tr), ("va", va)):
            for fm in ("T1", "T2", "S0"):
                st = states(g, [b[fm] for b in items]); Z[(split, fm)] = np.stack([s[0] for s in st])   # (n, 12, D) at "###"
                if fm == "T1" and split == "va":
                    Z["ends"] = [s[1:] for s, b in zip(st, items)]                                     # line-end states
            log(mn, split, "done")
        st1 = states(g, [b["T2"] for b in va1]); st0 = states(g, [b["S0"] for b in va1])
        y_va = [b["y1"] for b in va]
        R["gate_max_abs"] = states.gate
        for fm in ("T1", "T2"):
            per = {}
            for l in LAYERS:
                zt_tr, zt_va = Z[("tr", fm)][:, l - 1], Z[("va", fm)][:, l - 1]; hs_tr, hs_va = Z[("tr", "S0")][:, l - 1], Z[("va", "S0")][:, l - 1]
                sg = sig_batch(zt_va); l1i = float(np.abs(zt_va - hs_va).mean() / sg)
                med = np.median(zt_tr, 0); l1c = float(np.abs(zt_va - med).mean() / sg)
                zh, lam = ridge(hs_tr, zt_tr, hs_va); l1l = float(np.abs(zt_va - zh).mean() / sg)
                r2 = float(1 - ((zt_va - zh) ** 2).sum() / ((zt_va - zt_tr.mean(0)) ** 2).sum())
                big = np.argsort(-np.abs(zt_va).mean(0))[:5]; d_ = np.abs(zt_va - hs_va); top5 = float(d_[:, big].sum() / d_.sum())
                per[str(l)] = {"sigma_T": sg, "sigma_S0": sig_batch(hs_va), "L1_init": l1i, "L1_const": l1c, "L1_lin": l1l, "R2_lin": r2, "lam": lam,
                               "top5_share": top5, "top5_dims": [int(v) for v in big]}
                if l >= 7:
                    per[str(l)]["logitlens_top1_teacher"] = ans_top1(g, zt_va, y_va)
                if l == 12:
                    per[str(l)].update({"ans_top1_teacher": ans_top1(g, zt_va, y_va), "ans_top1_S0": ans_top1(g, hs_va, y_va),
                                        "ans_top1_lin_imitation": ans_top1(g, zh.astype(np.float32), y_va)})
                log(mn, fm, l, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in per[str(l)].items() if k != "top5_dims"}))
            R[fm] = per
        cnt = {}
        for l in range(7, 13):                                              # COUNTER: line-end states t = 1..3, m >= 3 items
            E = np.stack([np.stack([e[t][l - 1] for t in range(3)]) for e, b in zip(Z["ends"], va) if b["m"] >= 3])   # (n, 3, D)
            dl = E - E.mean(1, keepdims=True); c = dl.mean(0, keepdims=True); sg = sig_batch(E.reshape(-1, 768))
            cnt[str(l)] = {"R2_counter": float(1 - ((dl - c) ** 2).sum() / (dl ** 2).sum()), "L1_passvar": float(np.abs(dl).mean() / sg),
                           "L1_passvar_after_counter": float(np.abs(dl - c).mean() / sg), "n": int(E.shape[0])}
            log(mn, "counter", l, json.dumps(cnt[str(l)]))
        R["COUNTER"] = cnt
        lk = {}
        for l in LAYERS:                                                     # LEAK: m = 1, T2 = pre + a1 vs the student's [EOT] + pre + a1
            a = np.stack([s[0][l - 1] for s in st1]); b = np.stack([s[0][l - 1] for s in st0]); sg = sig_batch(a)
            lk[str(l)] = {"L1_T2_vs_S0": float(np.abs(a - b).mean() / sg)}
        R["LEAK_m1"] = lk; R["LEAK_m1"]["n"] = len(va1)
        R["ans_top1_T2_m1"] = ans_top1(g, np.stack([s[0][11] for s in st1]), [b["y1"] for b in va1]) if va1 else None
        R["ans_top1_S0_m1"] = ans_top1(g, np.stack([s[0][11] for s in st0]), [b["y1"] for b in va1]) if va1 else None
        res[mn] = R; log(mn, "model done")
        json.dump(res, open(os.path.join(OUT, "probe_kd.json"), "w", newline="\n"), indent=1)
    res["sec"] = round(time.time() - T0); json.dump(res, open(os.path.join(OUT, "probe_kd.json"), "w", newline="\n"), indent=1); log("DONE")


if __name__ == "__main__":
    main()
