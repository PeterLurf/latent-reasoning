"""STEPFACTOR D5b (Audit's RMS question): the residual pass clock in §200 A's memory reads. CPU fp32, A's FINAL model.pt (boxes/runs_fsq/gsm_lat_aug_s1),
200 Aug-valid items (rng 20214; 100 fit / 100 test items for the probes). DATA, POST-HOC (A has landed). Read-only on the repo; output clock_A.json here.
Reads are taken at the 3 positions of the NOPASS aux prompt [EOT, ' Step', ' :'] (queries item- and pass-independent: blocks 0-5 on a fixed prompt), on
M_t after t = 1..8 think passes, in two regimes: 'trained' (A's buffers: gamma 1, fade 1) and 'open' (gamma forced to 0: memory-independent writes, the
first 2k steps of any run). Per regime and t:
  rel_del(t)      = ||od(t+1) - od(t)|| / ||od(t)||     (delta read S_del q_hat, post-gain, all 6 heads x 64 at a position; mean over items x positions)
  rel_add(t)      = the same for the raw additive read phi(q)^T S_add (post-gain)
  rel_add_rms(t)  = the same after a synthetic per-head RMS normalisation (deltamem add_read 'rms' with gain 1: oa * rsqrt(mean(oa^2) + 1e-6))
  spread_*(t)     = mean_i ||r_i(t) - mean_j r_j(t)|| / mean_i ||r_i(t)||  (item-to-item spread, the noise a clock must beat)
Probes (pass index t from the read alone; multinomial logistic, L2 chosen by grouped 5-fold CV on the fit items; nearest-centroid beside it; features
standardised on the fit items): t in 1..4 and 1..8, on (a) delta read, (b) raw additive, (c) RMS-normalised additive, (d) RMS additive + delta (what a
NOPASS + RMS decoder could see); plus consecutive-pair (t vs t+1) accuracy from the 1..4 probe."""
import json
import os
import sys
import time

import numpy as np
import psutil

try:
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
except Exception:
    pass
os.environ["BOXES_NB"] = "5"
sys.path.insert(0, "D:/Latent")
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

torch.set_num_threads(3)
from boxes import s5  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
import boxes.gsm8k_data as G  # noqa: E402
from boxes.prosqa168s2 import COLON_ID, ENC, EOS, EOT, STEP_ID, ProsMem  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); RUN = "D:/Latent/boxes/runs_fsq/gsm_lat_aug_s1"; DEV = "cpu"; TMAX = 8; BS = 25; N_ITEMS = 200
DM.FAST_ADD = True; DM.CHUNK_DELTA = 64


@torch.no_grad()
def reads(m, S, q_pos):
    """post-gain additive / delta reads of states S (B, 12, 64, 64) at the fixed prompt queries -> oa, od (B, P, 6, 64)."""
    mem = m.mem; ha = mem.h_add
    pq = Fn.elu(q_pos[:, :ha]) + 1; oa = torch.exp(mem.s_add) * torch.einsum("phk,bhkv->bphv", pq, S[:, :ha].float())
    od = torch.exp(mem.s_del) * torch.einsum("bhvk,phk->bphv", S[:, ha:ha + mem.h_del].float(), Fn.normalize(q_pos[:, ha:], dim=-1))
    return oa, od


def rms(oa):
    return oa * torch.rsqrt(oa.pow(2).mean(-1, keepdim=True) + 1e-6)


def rel(R):
    """R: list over t of (B, P, H, D) -> rel change t -> t+1 (mean / median over items x positions) and item spread per t."""
    out, spread = {}, {}
    for t in range(len(R) - 1):
        a, b = R[t].flatten(2), R[t + 1].flatten(2); x = (b - a).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-30)
        out[str(t + 1)] = [round(float(x.mean()), 5), round(float(x.median()), 5)]
    for t in range(len(R)):
        a = R[t].flatten(2); spread[str(t + 1)] = round(float((a - a.mean(0, keepdim=True)).norm(dim=-1).mean() / a.norm(dim=-1).mean()), 5)
    return out, spread


def logreg(Xtr, ytr, Xte, lam, ncls):
    W = torch.zeros(Xtr.shape[1], ncls, requires_grad=True); b = torch.zeros(ncls, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=300, line_search_fn="strong_wolfe")

    def cl():
        opt.zero_grad(); L = Fn.cross_entropy(Xtr @ W + b, ytr) + lam * (W ** 2).sum(); L.backward(); return L
    opt.step(cl)
    with torch.no_grad():
        return Xte @ W + b


def probe(F, items_fit, items_te, ts):
    """F: dict t -> (N_items, d) features; classes = ts. Grouped CV on fit items for lambda; returns test acc, pair accs, centroid acc."""
    def stack(idx):
        X = torch.cat([F[t][idx] for t in ts]); y = torch.cat([torch.full((len(idx),), k, dtype=torch.long) for k in range(len(ts))]); return X, y
    Xf, yf = stack(items_fit); mu, sd = Xf.mean(0), Xf.std(0).clamp_min(1e-8); Xf = (Xf - mu) / sd
    Xt, yt = stack(items_te); Xt = (Xt - mu) / sd
    folds = np.array_split(np.arange(len(items_fit)), 5); best = None
    for lam in (1e-4, 1e-3, 1e-2, 1e-1):
        acc = []
        for f in folds:
            tr = np.setdiff1d(np.arange(len(items_fit)), f)
            Xa, ya = stack(items_fit[tr]); Xb, yb = stack(items_fit[f]); Xa, Xb = (Xa - mu) / sd, (Xb - mu) / sd
            acc.append(float((logreg(Xa, ya, Xb, lam, len(ts)).argmax(-1) == yb).float().mean()))
        if best is None or np.mean(acc) > best[1]:
            best = (lam, float(np.mean(acc)))
    lg = logreg(Xf, yf, Xt, best[0], len(ts)); pr = lg.argmax(-1); acc = float((pr == yt).float().mean())
    C = torch.stack([Xf[yf == k].mean(0) for k in range(len(ts))]); pc = torch.cdist(Xt, C).argmin(-1); cacc = float((pc == yt).float().mean())
    pairs = {}
    for k in range(len(ts) - 1):                                            # consecutive pair t_k vs t_{k+1}: binary decision on the two classes' logits
        mk = (yt == k) | (yt == k + 1)
        if mk.any():
            dec = torch.where(lg[mk, k + 1] > lg[mk, k], k + 1, k); pairs["%d_vs_%d" % (ts[k], ts[k + 1])] = round(float((dec == yt[mk]).float().mean()), 4)
    conf = np.zeros((len(ts), len(ts)), int)
    for a_, p_ in zip(yt.tolist(), pr.tolist()):
        conf[a_, p_] += 1
    return {"acc": round(acc, 4), "chance": round(1 / len(ts), 4), "lambda": best[0], "cv_acc": round(best[1], 4), "centroid_acc": round(cacc, 4),
            "pair_acc_binary": pairs, "confusion_rows_true": conf.tolist()}


if __name__ == "__main__":
    T0 = time.time(); g = s5.load_gpt2(DEV); m = ProsMem(g).to(DEV)
    m.load_state_dict(torch.load(os.path.join(RUN, "model.pt"), map_location=DEV, weights_only=False)); m.eval()
    Rva, _ = G.load_aug("valid", filter_bad=False); idx = np.random.default_rng(20214).permutation(len(Rva))[:N_ITEMS]
    D = [G.encode(Rva[i], ENC, EOS) for i in idx]
    pre = torch.tensor([[EOT, STEP_ID, COLON_ID]], device=DEV)
    with torch.no_grad():
        x6 = DM.gpt_lower(g, g.embed(pre), m.mem.layer)[0]; a = m.mem.ln(x6.float()); q = m.mem.qkv(a)[:, :m.mem.D].view(3, m.mem.h_add + m.mem.h_del, DM.DH)
    res = {"model": RUN + "/model.pt", "dtype": "cpu fp32", "n_items": N_ITEMS, "g_add": float(torch.exp(m.mem.s_add)), "g_del": float(torch.exp(m.mem.s_del)),
           "trained_gamma": float(m.mem.gamma), "trained_fade": float(m.mem.fade), "prompt": "[EOT, ' Step', ' :'] (NOPASS), 3 read positions"}
    rng_split = np.random.default_rng(20215).permutation(N_ITEMS); fit, te = np.sort(rng_split[:N_ITEMS // 2]), np.sort(rng_split[N_ITEMS // 2:])
    for regime, gam in (("trained", float(m.mem.gamma)), ("open", 0.0)):
        m.mem.gamma.fill_(gam); OA, OD = [[] for _ in range(TMAX)], [[] for _ in range(TMAX)]
        for i in range(0, len(D), BS):
            ch = D[i:i + BS]; x, msk = m.think_inputs(ch, DEV)
            with torch.no_grad():
                hl = m.lower(x); S = m.mem.zero_state(len(ch), DEV)
                for t in range(TMAX):
                    S = m.write(S, x, msk, hl); oa, od = reads(m, S, q); OA[t].append(oa); OD[t].append(od)
        OA = [torch.cat(v) for v in OA]; OD = [torch.cat(v) for v in OD]; OR = [rms(v) for v in OA]
        r_del, s_del = rel(OD); r_add, s_add = rel(OA); r_rms, s_rms = rel(OR)
        nrm = {str(t + 1): [round(float(OA[t].flatten(2).norm(dim=-1).mean()), 6), round(float(OD[t].flatten(2).norm(dim=-1).mean()), 5)] for t in range(TMAX)}
        feats = {"delta": {t + 1: OD[t].flatten(1) for t in range(TMAX)}, "add_raw": {t + 1: OA[t].flatten(1) for t in range(TMAX)},
                 "add_rms": {t + 1: OR[t].flatten(1) for t in range(TMAX)}, "add_rms+delta": {t + 1: torch.cat([OR[t].flatten(1), OD[t].flatten(1)], 1) for t in range(TMAX)}}
        pr = {}
        for nm, F in feats.items():
            pr[nm] = {"t1-4": probe(F, fit, te, [1, 2, 3, 4]), "t1-8": probe(F, fit, te, list(range(1, 9)))}
            print(regime, nm, pr[nm]["t1-4"]["acc"], pr[nm]["t1-8"]["acc"], round(time.time() - T0), flush=True)
        res[regime] = {"gamma": gam, "rel_change_delta": r_del, "rel_change_add_raw": r_add, "rel_change_add_rms": r_rms, "spread_delta": s_del,
                       "spread_add_raw": s_add, "spread_add_rms": s_rms, "norm_add_del_by_t": nrm, "probe": pr}
        print(regime, "rel del", r_del, "| rel add", r_add, "| rel add rms", r_rms, flush=True)
    res["sec"] = round(time.time() - T0)
    json.dump(res, open(os.path.join(HERE, "clock_A.json"), "w", newline="\n"), indent=1); print("DONE", res["sec"], flush=True)
