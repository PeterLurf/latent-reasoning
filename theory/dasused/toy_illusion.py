"""DAS-USED draft (docs/THEORY_DASUSED_DRAFT_2026-10-14.md §3, §6) CPU sanity check: do the proposed controls separate a USED variable from Makelov's
dormant-pathway illusion (2311.17030) in a small recurrent toy, and does the exact-energy control reach low target energies (the §203 G-use flaw)?

State s in R^2048 after the write pass: blocks A (2-d), C (2-d), B (2-d), N (18-d, var 1), F (2,022-d filler, var 0.005). Value v in Z_7 by the circle code
e(v) = (cos 2 pi v / 7, sin 2 pi v / 7). Then 3 carry passes s <- M s + noise (identity on A / C / B / F, 0.8 on N + fresh N noise), then a readout with
the late operand w (from the 'text'): logits_y = kappa <z, e(y - w)>, y = v + w (mod 7).
  USED      A = e(v), C = 0.1 e(v) (decodable, unused);  z = P_A s + gamma P_B s           (gamma 30: a dormant high-gain route)
  UNUSED    A = 0,    C = 0.1 e(v) (decodable, unused);  z = e(v_text) + gamma P_B s      (the state is not used: the answer is recomputed from text)
  USED-LOW  A = 0,    C = 0.1 e(v) (USED, low energy);  z = 10 P_C s + gamma P_B s
B carries no information and has variance 1e-10 (dormant). Pairs: donor v_d != v_r, w_d != w_r; counterfactual y' = v_d + w_r.
DAS: R = QR(W).Q in R^{2048 x k}, patch at pass 1 s <- s_r + R R^T (s_d - s_r), CE to y', Adam, 3 random restarts, best by validation; IIA on held-out pairs.
Controls (the draft's): 5 Haar and 5 exact-energy random subspaces (beta-bisection, |E - E*| / E* reported; and the G-use Sigma^beta beta in {0.5, 1, 2}
best-of-200 for comparison), shuffled targets, clean decodability (linear / kNN) of R^T s, necessity by resample / mean / zero ablation, var share,
off-support share omega and the on-support refit (R inside the eigenvectors with lambda > 1e-6 lambda_max of the clean covariance).
-> theory/dasused/toy_illusion.json      python theory/dasused/toy_illusion.py"""
import ctypes
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "3")
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)   # IDLE_PRIORITY_CLASS
except Exception:
    pass
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

torch.set_num_threads(3); torch.set_default_dtype(torch.float64)
P, D, KAPPA, GAMMA, SIG_C = 7, 2048, 8.0, 30.0, 0.1
A_, C_, B_, N_, F_ = slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 24), slice(24, D)
CODE = torch.stack([torch.cos(2 * np.pi * torch.arange(P) / P), torch.sin(2 * np.pi * torch.arange(P) / P)], 1)   # (P, 2)


def states(v, gen, model):
    n = len(v); s = torch.zeros(n, D); e = CODE[v]
    if model == "USED":
        s[:, A_] = e + 0.05 * torch.randn(n, 2, generator=gen)
    s[:, C_] = SIG_C * e + 0.005 * torch.randn(n, 2, generator=gen)
    s[:, B_] = 1e-5 * torch.randn(n, 2, generator=gen); s[:, N_] = torch.randn(n, 18, generator=gen); s[:, F_] = 0.005 ** 0.5 * torch.randn(n, D - 24, generator=gen)
    return s


def readout(s, v_text, w, gen, model):
    for _ in range(3):                                                      # carry passes (the recipient's own noise)
        s = s.clone(); s[:, N_] = 0.8 * s[:, N_] + 0.6 * torch.randn(len(s), 18, generator=gen)
    z = GAMMA * s[:, B_]
    z = z + (s[:, A_] if model == "USED" else CODE[v_text] if model == "UNUSED" else s[:, C_] / SIG_C)
    return KAPPA * torch.einsum("bi,bpi->bp", z, CODE[(torch.arange(P)[None, :] - w[:, None]) % P])   # logit of y = <z, e(y - w)>


def make_pairs(n, seed, model):
    g = torch.Generator().manual_seed(seed); vr = torch.randint(0, P, (n,), generator=g); vd = (vr + torch.randint(1, P, (n,), generator=g)) % P
    wr = torch.randint(0, P, (n,), generator=g); wd = (wr + torch.randint(1, P, (n,), generator=g)) % P
    return {"vr": vr, "vd": vd, "wr": wr, "wd": wd, "sr": states(vr, g, model), "sd": states(vd, g, model), "y": (vr + wr) % P, "ycf": (vd + wr) % P, "gen_seed": seed + 1}


def run_patch(pr, R, model, target="ycf", mode="swap", other=None, mu=None):
    g = torch.Generator().manual_seed(pr["gen_seed"]); s = pr["sr"]
    if R is not None:
        if mode == "swap":
            s = s + (pr["sd"] - s) @ R @ R.T
        elif mode == "resample":
            s = s + (other - s) @ R @ R.T
        elif mode == "mean":
            s = s - (s - mu) @ R @ R.T
        elif mode == "zero":
            s = s - s @ R @ R.T
    lg = readout(s, pr["vr"], pr["wr"], g, model); return lg


def acc(lg, y):
    return float((lg.argmax(1) == y).double().mean())


def inits(tr, k, basis, gen, warm=None):
    """the draft §7 fit inits: random, the top-k PCs of s_d - s_r (das167), the top-k class-mean directions of s by v (the decodable directions), and
    an optional warm start (the on-support refit starts at the projection of R*); all in the basis' coordinates."""
    X = tr["sr"] if basis is None else tr["sr"] @ basis; dX = (tr["sd"] - tr["sr"]) if basis is None else (tr["sd"] - tr["sr"]) @ basis; m = X.shape[1]
    cm = torch.stack([X[tr["vr"] == c].mean(0) for c in range(P)]); cm = cm - cm.mean(0)
    out = [torch.randn(m, k, generator=gen), torch.linalg.svd(dX[:4000], full_matrices=False)[2][:k].T.clone(),
           torch.linalg.svd(cm, full_matrices=False)[2][:min(k, P - 1)].T.clone()]
    if out[-1].shape[1] < k:
        out[-1] = torch.cat([out[-1], torch.randn(m, k - out[-1].shape[1], generator=gen) * 1e-3], 1)
    if warm is not None:
        out.append((warm if basis is None else basis.T @ warm).clone())
    return out


def fit(tr, va, k, model, basis=None, seed=0, steps=1500, lr=0.03, shuffle=None, warm=None):
    best = (-1, None); gen0 = torch.Generator().manual_seed(1000 * seed + 17 * k)
    for rs, W0 in enumerate(inits(tr, k, basis, gen0, warm)):
        gen = torch.Generator().manual_seed(1000 * seed + 17 * k + rs)
        W = W0.clone().requires_grad_(True); opt = torch.optim.Adam([W], lr=lr); yt = tr["ycf"] if shuffle is None else shuffle
        for st in range(steps):
            idx = torch.randint(0, len(yt), (512,), generator=gen); Q = torch.linalg.qr(W).Q; R = Q if basis is None else basis @ Q
            sub = {kk: (vv[idx] if torch.is_tensor(vv) else vv) for kk, vv in tr.items()}; sub["gen_seed"] = int(torch.randint(0, 10 ** 6, (1,), generator=gen))
            loss = Fn.cross_entropy(run_patch(sub, R, model), yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            Q = torch.linalg.qr(W).Q; R = Q if basis is None else basis @ Q; v_ = acc(run_patch(va, R, model), va["ycf"] if shuffle is None else va["ycf_sh"])
        if v_ > best[0]:
            best = (v_, R.detach())
    return best[1], best[0]


def orth(M):
    return torch.linalg.qr(M).Q


def energy(R, Sig):
    return float(torch.trace(R.T @ Sig @ R))


def exact_energy_controls(Sig, k, Estar, n, seed):
    ev, U = torch.linalg.eigh(Sig); ev = ev.clamp_min(0) + 1e-12 * float(ev.max()); out = []; gen = torch.Generator().manual_seed(seed); tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1; G = torch.randn(D, k, generator=gen); UG = U.T @ G; Rb = lambda b: orth(U @ ((ev ** (b / 2))[:, None] * UG))  # noqa: E731
        lo, hi = -12.0, 12.0; Elo, Ehi = energy(Rb(lo), Sig), energy(Rb(hi), Sig)
        if not (Elo < Estar < Ehi):
            continue
        for _ in range(80):
            mid = (lo + hi) / 2; Em = energy(Rb(mid), Sig)
            if abs(Em - Estar) / Estar < 1e-4:
                break
            lo, hi = (mid, hi) if Em < Estar else (lo, mid)
        out.append(Rb(mid))
    return out


def guse_style_controls(Sig, k, Estar, seed):
    """§203 G-use's construction: span of k Gaussian vectors with covariance Sigma^beta, beta in (0.5, 1, 2); the 5 of 200 closest in energy."""
    ev, U = torch.linalg.eigh(Sig); ev = ev.clamp_min(0); gen = torch.Generator().manual_seed(seed); c = []
    for j in range(200):
        be = (0.5, 1.0, 2.0)[j % 3]; R = orth(U @ ((ev ** (be / 2))[:, None] * (U.T @ torch.randn(D, k, generator=gen)))); c.append((abs(energy(R, Sig) - Estar) / Estar, R))
    return sorted(c, key=lambda x: x[0])[:5]


def probe(Xf, yf, Xs, ys):
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-9; Xf_, Xs_ = (Xf - mu) / sd, (Xs - mu) / sd
    W = torch.zeros(Xf.shape[1], P, requires_grad=True); b = torch.zeros(P, requires_grad=True); opt = torch.optim.LBFGS([W, b], max_iter=200, line_search_fn="strong_wolfe")

    def clos():
        opt.zero_grad(); l_ = Fn.cross_entropy(Xf_ @ W + b, yf) + 1e-3 * (W ** 2).sum(); l_.backward(); return l_
    opt.step(clos); lin = float(((Xs_ @ W + b).argmax(1) == ys).double().mean())
    d2 = torch.cdist(Xs_, Xf_); nn_ = d2.topk(15, largest=False).indices; vote = torch.zeros(len(Xs), P).scatter_add_(1, yf[nn_], torch.ones_like(nn_, dtype=torch.float64))
    return lin, float((vote.argmax(1) == ys).double().mean())


def label(r):
    """the draft §3.5 rule (toy instance): IIA* = held-out IIA at the best k; the illusion test is the on-support refit (omega is descriptive)."""
    ctrl = max(r["haar_iia_max"], r["energy_iia_max"]); iia = r["iia"]
    if iia >= 0.70 and iia - max(ctrl, r["b0"]) >= 0.50 and r["shuffled_iia"] <= 0.20:
        if r["iia_on_support"] >= iia - 0.15:
            return "UR-FOUND" + ("" if r["necessity_resample_on_support_excess"] >= 0.10 else "-REDUNDANT")
        if r["iia_on_support"] < iia - 0.30:
            return "ILLUSION-SUSPECT"
        return "PARTIAL"
    return "NONE-FOUND" if iia < 0.30 else "PARTIAL"


def main():
    t0 = time.time(); out = {"def": __doc__.split("->")[0].strip(), "models": {}}
    for model in ("USED", "UNUSED", "USED-LOW"):
        tr, va, te = make_pairs(20000, 11, model), make_pairs(2000, 12, model), make_pairs(4000, 13, model)
        for d_ in (va, te):
            d_["ycf_sh"] = d_["ycf"][torch.randperm(len(d_["ycf"]), generator=torch.Generator().manual_seed(5))]
        clean = states(torch.randint(0, P, (20000,), generator=torch.Generator().manual_seed(21)), torch.Generator().manual_seed(22), model)
        mu = clean.mean(0); Sig = torch.cov(clean.T); ev, U = torch.linalg.eigh(Sig); supp = U[:, ev > 1e-6 * ev.max()]
        b0 = acc(run_patch(te, None, model), te["ycf"]); base = acc(run_patch(te, None, model), te["y"]); res = {"base_acc": base, "b0": b0, "support_dim": int(supp.shape[1]), "by_k": {}}
        best = None
        for k in (1, 2, 4):
            R, vbest = fit(tr, va, k, model); iia = acc(run_patch(te, R, model), te["ycf"]); res["by_k"][str(k)] = {"iia": iia, "val": vbest}
            if best is None or iia > best[1] + 1e-9:
                best = (k, iia, R)
        k, iia, R = best; Estar = energy(R, Sig); dlt = (te["sd"] - te["sr"]) @ R @ R.T; off = dlt - dlt @ supp @ supp.T
        omega = float((off ** 2).sum() / (dlt ** 2).sum())
        Rproj = orth(supp @ (supp.T @ R)); iia_proj = acc(run_patch(te, Rproj, model), te["ycf"])   # the split test without refit (R_par of Prop D4)
        Ron, _ = fit(tr, va, k, model, basis=supp, seed=1, warm=R); iia_on = acc(run_patch(te, Ron, model), te["ycf"])
        gH = torch.Generator().manual_seed(31); haar = [orth(torch.randn(D, k, generator=gH)) for _ in range(5)]
        ens = exact_energy_controls(Sig, k, Estar, 5, 41); gus = guse_style_controls(Sig, k, Estar, 51)
        sh_tr = tr["ycf"][torch.randperm(len(tr["ycf"]), generator=torch.Generator().manual_seed(6))]; Rsh, _ = fit(tr, va, k, model, shuffle=sh_tr, seed=2)
        other = te["sr"][torch.randperm(len(te["sr"]), generator=torch.Generator().manual_seed(7))]
        nec = lambda R_, md: base - acc(run_patch(te, R_, model, mode=md, other=other, mu=mu), te["y"])  # noqa: E731
        Xf = clean[:10000] @ R; yf = torch.randint(0, P, (20000,), generator=torch.Generator().manual_seed(21))[:10000]
        cl2 = states(torch.randint(0, P, (4000,), generator=torch.Generator().manual_seed(23)), torch.Generator().manual_seed(24), model)
        lin, knn = probe(Xf, yf, cl2 @ R, torch.randint(0, P, (4000,), generator=torch.Generator().manual_seed(23)))
        haar_nec = float(np.mean([nec(R_, "resample") for R_ in haar]))
        res.update({"best_k": k, "iia": iia, "E_star": Estar, "var_share": Estar / float(torch.trace(Sig)), "omega": omega, "iia_proj_support": iia_proj, "iia_on_support": iia_on,
                    "haar_iia": [acc(run_patch(te, R_, model), te["ycf"]) for R_ in haar], "energy_iia": [acc(run_patch(te, R_, model), te["ycf"]) for R_ in ens],
                    "energy_rel_err": [abs(energy(R_, Sig) - Estar) / Estar for R_ in ens],
                    "guse_style_rel_err_best5": [float(c_[0]) for c_ in gus],
                    "shuffled_iia": acc(run_patch(te, Rsh, model), te["ycf_sh"]),
                    "decode_linear": lin, "decode_knn": knn,
                    "necessity_resample": nec(R, "resample"), "necessity_mean": nec(R, "mean"), "necessity_zero": nec(R, "zero"),
                    "necessity_resample_on_support": nec(Ron, "resample"), "necessity_resample_haar_mean": haar_nec,
                    "mass_in_blocks": {b_: float((R[sl] ** 2).sum() / k) for b_, sl in (("A", A_), ("C", C_), ("B", B_), ("N", N_), ("F", F_))}})
        res["haar_iia_max"], res["energy_iia_max"] = max(res["haar_iia"]), max(res["energy_iia"])
        res["necessity_resample_on_support_excess"] = res["necessity_resample_on_support"] - haar_nec
        res["label"] = label(res); res["label_without_support_split"] = "UR-FOUND" if (iia >= 0.7 and iia - max(res["haar_iia_max"], res["energy_iia_max"], b0) >= 0.5
                                                                                      and res["shuffled_iia"] <= 0.2 and res["necessity_resample"] - haar_nec >= 0.10) else "other"
        res["makelov_low_var_flag"] = bool(res["var_share"] < 0.01)
        out["models"][model] = res; print(model, json.dumps({k_: v_ for k_, v_ in res.items() if k_ not in ("by_k",)}), flush=True)
    out["sec"] = round(time.time() - t0)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "toy_illusion.json"), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
