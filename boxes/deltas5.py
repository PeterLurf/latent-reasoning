"""THEORY §162 DELTA-S5 (9f7d4793; scorer theory/score_s162.py): §155 R's write-after-read memory on fsq120's S5 Family, plus a single-pass baseline.
--arm direct (s162_direct_s{seed}): GPT-2 small with the whole chain in one context, [start-state text (6), the items' tokens (6 each), " ="] ->
    the 6-token state text (teacher-forced CE in training, greedy at test). Random start, L ~ U(1, L_max), L_max 2 / 4 / 8 / 16 / 32 / 64 at
    0 / 2k / 5k / 9k / 14k / 18k. eval.json acc_by_length {"1", "2", "4", ..., "64" (+ "128", descriptive)}: final-state exact on the 2,000 test
    prefixes (bench S5=512.csv, identity start) of each length.
--arm delta (s162_{long,short,long64}_s{seed}): §155 R's DeltaMem (6 additive + 6 gated-delta heads after block 6, --mem_write top: pass-start
    reads, writes from u = (1 - gamma) LN_6(h_6) + gamma LN_w(h_top), gamma 0 -> 1 over write_mix 2000 -> 6000; W_o fade-in over 500 steps;
    g_add calibrated at step 0 as note 147-2) as the only cross-segment channel. Segment 0 = [EOT, start-state text]; segment t = [EOT, item t's
    6 tokens], each a fresh GPT-2 pass; the answer decoder = a final [EOT] segment (+ the first 5 answer tokens, teacher-forced) that only READS
    the memory (no write) -> the 6-token state text. FINAL only, the same curriculum capped at --l_max.
    eval.json: final_acc_by_length {"1".."l_max"}, long_acc (cumulative all positions at 512, identity start, as §128), acc_2048 (cumulative,
    500 chains of S5=2048.csv), donor {posA: {donor_consistent, recipient_consistent, chance_overlap}} (whole-memory swap on length-l chains,
    l = min(8, l_max), at up to three evenly spaced items), s155 (recurrence incl. rho_loop, write_grad_norm, write_grad_spikes(_floor), n_nonfinite),
    s162 {state_probe {p: acc}, op_probe {p: {j: acc}}, probe_n_fit, probe_n_score, state_edit {...} at p = l_max / 2 on chains of length l_max (note 162-1), key_cos (per head)}.
Both: §120's recipe (24k, bs 64, AdamW 1e-4 wd 0.01, warmup 100 + cosine, clip 1.0), ckpt_last / ckpt_prev every --ckpt_every, --resume.
--auto_micro: before training, a memory check at L = l_max (bs / micro rows); micro doubles until the peak is <= --mem_budget GiB.
    python -m boxes.deltas5 --arm delta --l_max 32 --seed 1 --out boxes/runs_fsq/s162_long_s1
    python -m boxes.deltas5 --arm direct --seed 1 --out boxes/runs_fsq/s162_direct_s1"""
import os
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
from boxes import core as bc  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
from boxes.fsq120 import EOT, Family, make_batch, make_opt, lr_at, driver_version, env_info  # noqa: E402
from boxes.gen97 import NST, STTOK, decode_to_el  # noqa: E402
from boxes.latent86 import OPTOK  # noqa: E402

EQTOK = bc.tok(" =")[0]
OPT = np.array([OPTOK[e] for e in range(120)])                               # (120, 6) item tokens
STT = np.array(STTOK)                                                       # (120, 6) state text
TRANSP = [e for e in range(120) if sum(int(c == chr(65 + i)) for i, c in enumerate(s5.arr_of(e))) == 3]   # the 10 transpositions
assert len(TRANSP) == 10 and OPT.shape == (120, 6) and STT.shape == (120, NST)
SCHED = ((2000, 2), (5000, 4), (9000, 8), (14000, 16), (18000, 32), (None, 64))


def lmax_sched(step, cap):
    for lim, L in SCHED:
        if lim is None or step <= lim:
            return min(L, cap)


def amp_ctx(amp):
    return torch.autocast("cuda", dtype=amp, enabled=amp is not None)


def comp_chain(s0, X):
    """states after each item: COMP chain (N, L)."""
    s = np.array(s0); out = np.zeros(X.shape, np.int64)
    for t in range(X.shape[1]):
        s = s5.COMP[s, X[:, t]]; out[:, t] = s
    return out


# ------------------------------------------------------------------------------------------------------------------ DIRECT
class Direct(nn.Module):
    def __init__(self, g):
        super().__init__(); self.g = g

    def prefix(self, s0, X, dev):
        B = len(X); ids = np.concatenate([STT[s0], OPT[X].reshape(B, -1), np.full((B, 1), EQTOK)], 1)
        return torch.tensor(ids, device=dev)

    def loss(self, s0, X, y, dev):
        pre = self.prefix(s0, X, dev); ans = torch.tensor(STT[y], device=dev); P = pre.shape[1]
        h, _ = self.g(self.g.embed(torch.cat([pre, ans[:, :-1]], 1)), None); lg = self.g.logits(h[:, P - 1:]).float()
        return Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), ans.reshape(-1))

    @torch.no_grad()
    def greedy(self, s0, X, dev):
        h, past = self.g(self.g.embed(self.prefix(s0, X, dev)), None); out = []
        for k in range(NST):
            nt = self.g.logits(h[:, -1]).float().argmax(-1); out.append(nt)
            if k < NST - 1:
                h, past = self.g(self.g.embed(nt[:, None]), past)
        return decode_to_el(torch.stack(out, 1).cpu().numpy())[0]


# ------------------------------------------------------------------------------------------------------------------ DELTA
def fwd_read(g, x, mem, S):
    """the answer decoder pass: the read on S entering block mem.layer (read_fixed), no write; final hidden after ln_f."""
    B, T, D = x.shape; hd = D // g.H; h = x + g.w("wpe.weight")[:T][None]
    for i in range(g.L):
        if i == mem.layer:
            h = DM.read_fixed(mem, h, S)
        h = DM._block(g, i, h, hd)
    return Fn.layer_norm(h, (D,), g.w("ln_f.weight"), g.w("ln_f.bias"))


class DeltaS5(nn.Module):
    def __init__(self, g, layer=6, write="top"):
        super().__init__(); self.g = g; self.mem = DM.DeltaMem(g, layer, write)

    def start(self, B, dev):
        return self.mem.zero_state(B, dev)

    def ids_start(self, s0, dev):
        return torch.tensor(np.c_[np.full(len(s0), EOT), STT[s0]], device=dev)

    def ids_item(self, x, dev):
        return torch.tensor(np.c_[np.full(len(x), EOT), OPT[x]], device=dev)

    def write(self, S, ids, cap=None):
        return DM.gpt_forward_mem(self.g, self.g.embed(ids), self.mem, S)[1] if cap is None else DM.gpt_forward_mem_r(self.g, self.g.embed(ids), self.mem, S, cap=cap)[1]

    def run(self, s0, X, dev, S=None, t0=0):
        """the start segment (unless S is given) and items X[:, t0:]; returns S."""
        if S is None:
            S = self.write(self.start(len(X), dev), self.ids_start(s0, dev))
        for t in range(t0, X.shape[1]):
            S = self.write(S, self.ids_item(X[:, t], dev))
        return S

    def dec_logits(self, S, ans):
        B = S.shape[0]; ids = torch.cat([torch.full((B, 1), EOT, device=S.device), ans[:, :-1]], 1)
        return self.g.logits(fwd_read(self.g, self.g.embed(ids), self.mem, S)).float()

    def loss(self, S, y, dev):
        ans = torch.tensor(STT[y], device=dev); lg = self.dec_logits(S, ans)
        return Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), ans.reshape(-1))

    @torch.no_grad()
    def greedy(self, S):
        B = S.shape[0]; ids = torch.full((B, 1), EOT, device=S.device)
        for _ in range(NST):
            nt = self.g.logits(fwd_read(self.g, self.g.embed(ids), self.mem, S)[:, -1]).float().argmax(-1); ids = torch.cat([ids, nt[:, None]], 1)
        return decode_to_el(ids[:, 1:].cpu().numpy())[0]


@torch.no_grad()
def rollout(m, s0, X, dev, amp, bs, every=True, keep=(), override=None):
    """hard chains; decoded state at every position (every) or the last only; keep: positions p (states after item p, flattened fp32);
    override (p, fn): S after item p replaced by fn(S, rows)."""
    N, L = X.shape; pred = np.full((N, L) if every else (N,), -1); kept = {p: None for p in keep}
    for i in range(0, N, bs):
        sl = slice(i, i + bs); B = len(X[sl])
        with amp_ctx(amp):
            S = m.write(m.start(B, dev), m.ids_start(s0[sl], dev))
            for t in range(L):
                S = m.write(S, m.ids_item(X[sl, t], dev))
                if override is not None and override[0] == t + 1:
                    S = override[1](S, np.arange(N)[sl])
                if t + 1 in kept:
                    if kept[t + 1] is None:
                        kept[t + 1] = np.zeros((N, S[0].numel()), np.float32)
                    kept[t + 1][sl] = S.float().flatten(1).cpu().numpy()
                if every:
                    pred[sl, t] = m.greedy(S)
            if not every:
                pred[sl] = m.greedy(S)
    return pred, kept


def read_ratio(m, dev, amp, n=256, L=8, seed=16201):
    """note 147-2: median ||read_del|| / ||read_add|| over the item hand-offs (the read at the start of item segments)."""
    rng = np.random.default_rng(seed); s0, X, _ = make_batch(Family("s5"), rng, n, L); was = m.training; m.eval(); m.mem.norms = []
    with torch.no_grad(), amp_ctx(amp):                                     # writer passes only (the decoder's reads would add entries)
        m.run(s0, X, dev)
    nr = m.mem.norms[1:]; m.mem.norms = None; m.train(was)
    r = torch.stack([d_ / a_.clamp_min(1e-12) for d_, a_ in nr]); return float(r.median())


def write_grad_norm(m):
    D = getattr(m.mem, "D", 768); g_ = []                                   # §178: the memory's width
    for p_, sl in ((m.mem.ln_w.weight, None), (m.mem.ln_w.bias, None), (m.mem.qkv.weight, slice(D, 3 * D)), (m.mem.qkv.bias, slice(D, 3 * D)),
                   (m.mem.gate.weight, None), (m.mem.gate.bias, None)):
        if p_.grad is not None:
            g_.append((p_.grad[sl] if sl is not None else p_.grad).float().norm())
    return float(torch.stack(g_).norm()) if g_ else 0.0



@torch.no_grad()
def pass_states(m, rng, k, dev, amp, Lmax=32):
    """§166 pass: k random training chains (random start, one length Lp ~ U(1..Lmax)) run through the CURRENT model; returns (their final
    memories (k, 12, DH, DH) fp32, final states g (k,), s0, X) - the passed chains then skip segment 0 and are scored against g o items."""
    Lp = int(rng.integers(1, Lmax + 1)); s0, X, Y = make_batch(Family("s5"), rng, k, Lp)
    with amp_ctx(amp):
        S = m.run(s0, X, dev)
    return S.float(), Y[:, -1], s0, X


# ------------------------------------------------------------------------------------------------------------------ descriptive blocks
def _multi_r2(Xf, Yf, Xs, Ys, lam=1e-2):
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6; Zf = np.c_[(Xf - mu) / sd, np.ones(len(Xf))]; Zs = np.c_[(Xs - mu) / sd, np.ones(len(Xs))]
    W = np.linalg.solve(Zf.T @ Zf + lam * len(Zf) * np.eye(Zf.shape[1]), Zf.T @ Yf); r = Ys - Zs @ W
    return float(1 - (r ** 2).sum() / max(((Ys - Ys.mean(0)) ** 2).sum(), 1e-12))


def s155_recurrence(m, dev, amp, n=128, L=8, rows=4000, seed=15501):
    """§155's block on S5 chains: write_var_from_read / input / input_read R^2 (held-out ridge on the write input u), rho_jvp (median ||J v||,
    J = dS_{t+1} / dS_t, v a random unit direction) and rho_loop (||J v - J v|fade 0||: the read -> write part); fp32, SDPA math backend."""
    was = m.training; m.eval(); rng = np.random.default_rng(seed); s0, X, _ = make_batch(Family("s5"), rng, n, L); out = {}; caps = []
    with torch.no_grad():
        S = m.write(m.start(n, dev), m.ids_start(s0, dev)); Ss = [S.clone()]
        for t in range(L):
            cap = {}
            with amp_ctx(amp):
                S = m.write(S, m.ids_item(X[:, t], dev), cap=cap)
            a6 = m.mem.ln(cap["h6"].float()); u = (1 - m.mem.gamma) * a6 + m.mem.gamma * m.mem.ln_w(cap["hw"].float())
            caps.append((a6.flatten(0, 1).cpu().numpy(), cap["read"].float().flatten(0, 1).cpu().numpy(), u.flatten(0, 1).cpu().numpy())); Ss.append(S.clone())
    a6 = np.concatenate([c[0] for c in caps]); rd = np.concatenate([c[1] for c in caps]); u = np.concatenate([c[2] for c in caps])
    ix = rng.permutation(len(u))[:rows]; h = len(ix) // 2; f_, s_ = ix[:h], ix[h:]
    out["write_var_from_read_r2"] = _multi_r2(rd[f_], u[f_], rd[s_], u[s_]); out["write_var_from_input_r2"] = _multi_r2(a6[f_], u[f_], a6[s_], u[s_])
    ar = np.concatenate([a6, rd], 1); out["write_var_from_input_read_r2"] = _multi_r2(ar[f_], u[f_], ar[s_], u[s_])
    out["write_var_incremental_read_r2"] = out["write_var_from_input_read_r2"] - out["write_var_from_input_r2"]
    try:
        from torch.func import jvp
        from torch.nn.attention import SDPBackend, sdpa_kernel
        nj = min(32, n); rh, rl = [], []
        for t in range(0, L, 2):
            S0 = Ss[t][:nj].float(); v = torch.randn_like(S0); v = v / v.flatten(1).norm(dim=1)[:, None, None, None]; x = m.g.embed(m.ids_item(X[:nj, t], dev))
            f_ = lambda S_: DM.gpt_forward_mem_r(m.g, x, m.mem, S_)[1]  # noqa: E731
            with sdpa_kernel(SDPBackend.MATH):
                _, Jv = jvp(f_, (S0,), (v,))
            fd_ = m.mem.fade.clone(); m.mem.fade.fill_(0.0)
            try:
                with sdpa_kernel(SDPBackend.MATH):
                    _, Jv0 = jvp(f_, (S0,), (v,))
            finally:
                m.mem.fade.copy_(fd_)
            rh.append(Jv.detach().flatten(1).norm(dim=1).cpu().numpy()); rl.append((Jv - Jv0).detach().flatten(1).norm(dim=1).cpu().numpy())
        out.update({"rho_jvp": float(np.median(np.concatenate(rh))), "rho_loop": float(np.median(np.concatenate(rl))), "rho_method": "torch.func.jvp (SDPA math backend), fp32"})
    except Exception as e:                                                  # noqa: BLE001
        out["rho_error"] = repr(e)[:300]
    m.train(was); return out


def _fit_cls(Xf, yf, dev, k=120, lams=(1e-9, 1e-8, 1e-7, 1e-6)):
    """note 155-4's near-OLS (standardised, dual K = Z Z^T / d, lambda x n by 3-fold CV argmax accuracy, edge-widened), one-hot k-way.
    Returns (W raw-space (d, k) fp32 on dev, b (k,), lambda): scores = x W + b."""
    Zf = torch.tensor(Xf, device=dev, dtype=torch.float32); mu, sd = Zf.mean(0), Zf.std(0) + 1e-6; Zf = (Zf - mu) / sd; d = Zf.shape[1]; n = len(Zf)
    K = (Zf @ Zf.T).double() / d; Y = torch.tensor(np.eye(k)[yf], device=dev, dtype=torch.float64); fold = torch.arange(n, device=dev) % 3; yt = torch.tensor(yf, device=dev)
    def fit(idx, lam):
        ym = Y[idx].mean(0); return torch.linalg.solve(K[idx][:, idx] + lam * len(idx) * torch.eye(len(idx), device=dev, dtype=torch.float64), Y[idx] - ym), ym
    def cv(lam):
        sc = []
        for f in range(3):
            tr = torch.nonzero(fold != f).squeeze(1); va = torch.nonzero(fold == f).squeeze(1); al, ym = fit(tr, lam)
            sc.append(float(((K[va][:, tr] @ al + ym).argmax(1) == yt[va]).double().mean()))
        return float(np.mean(sc))
    best = max(lams, key=cv)
    if best in (min(lams), max(lams)):
        best = max([best] + [best * 10 ** e for e in ((-1, -2) if best == min(lams) else (1, 2))], key=cv)
    al, ym = fit(torch.arange(n, device=dev), best); Wz = (Zf.T.double() @ al) / d; W = (Wz / sd.double()[:, None]); b = ym - mu.double() @ W
    del K, Zf
    return W.float(), b.float(), float(best)


def _score(W, b, Xs, dev, bs=2048):
    out = []
    for i in range(0, len(Xs), bs):
        out.append((torch.tensor(Xs[i:i + bs], device=dev) @ W + b).cpu())
    return torch.cat(out).numpy()


def probes(run_fn, P, dev, n_fit=8000, n_score=2000, seed=16202):
    """state_probe {p: acc}, op_probe {p: {j: acc}} from the flattened memory after item p (training-distribution chains: random start,
    uniform items; fit n_fit, score a fresh n_score). run_fn(s0, X, keep) -> {p: (N, d) features}."""
    fam = Family("s5"); rng = np.random.default_rng(seed); Lp = max(P)
    s0f, Xf, Yf = make_batch(fam, rng, n_fit, Lp); s0s, Xs, Ys = make_batch(fam, rng, n_score, Lp); Kf = run_fn(s0f, Xf, P); Ks = run_fn(s0s, Xs, P)
    sp, op, lam = {}, {}, {}
    for p in P:
        W, b, l_ = _fit_cls(Kf[p], Yf[:, p - 1], dev); sp[str(p)] = float((_score(W, b, Ks[p], dev).argmax(1) == Ys[:, p - 1]).mean()); lam[str(p)] = l_; op[str(p)] = {}
        for j in range(p):
            W, b, _ = _fit_cls(Kf[p], Xf[:, p - 1 - j], dev); op[str(p)][str(j)] = float((_score(W, b, Ks[p], dev).argmax(1) == Xs[:, p - 1 - j]).mean())
    return {"state_probe": sp, "op_probe": op, "state_probe_lambda": lam, "probe_n_fit": int(n_fit), "probe_n_score": int(n_score),
            "probe_def": "near-OLS 120-way (155-4): standardised dual ridge, 3-fold CV lambda 1e-9..1e-6 (edge-widened), one-hot argmax; random-start fit / score chains"}


def state_edit(run_fn, cont_fn, ell, dev, n=1000, n_fit=8000, seed=16203):
    """after item p = ell / 2 on length-ell chains: the minimum-norm change of the flattened memory along the state probe's (w_g' - w_g),
    g' = g o tau (tau a random transposition), that makes g' the argmax with g's original margin; then continue. propagation_share = final
    answer == g' o x_{p+1..ell}; random_edit_share = the same with an equal-norm random direction; null_value = the unedited answer equals the
    target. Rows: those whose probe argmax is g at p (margin > 0) and whose edit reaches the margin. run_fn(s0, X, keep) -> {p: features};
    cont_fn(S_flat (N, d) fp32, X_rest) -> final states."""
    fam = Family("s5"); rng = np.random.default_rng(seed); p = ell // 2
    s0f, Xf, Yf = make_batch(fam, rng, n_fit, p); W, b, _ = _fit_cls(run_fn(s0f, Xf, (p,))[p], Yf[:, p - 1], dev)
    s0, X, Y = make_batch(fam, rng, n, ell); Xp = run_fn(s0, X[:, :p], (p,))[p]; sc = _score(W, b, Xp, dev); g = Y[:, p - 1]
    tau = rng.choice(TRANSP, n); g2 = s5.COMP[g, tau]; tgt = comp_chain(g2, X[:, p:])[:, -1]
    ok = sc.argmax(1) == g; m0 = sc[np.arange(n), g] - np.where(np.eye(120, dtype=bool)[g], -np.inf, sc).max(1)
    Wt = W.cpu().numpy(); D = Wt[:, g2].T - Wt[:, g].T; Dn = D / np.linalg.norm(D, axis=1, keepdims=True); WD = Dn @ Wt   # (n, 120): d s_k / d alpha
    alpha = np.full(n, np.nan)
    for r in np.nonzero(ok)[0]:
        def marg(a_):
            s_ = sc[r] + a_ * WD[r]; return s_[g2[r]] - np.max(np.delete(s_, g2[r]))
        hi = 1.0
        while marg(hi) < m0[r] and hi < 1e8:
            hi *= 2
        if marg(hi) < m0[r]:
            continue
        lo = 0.0
        for _ in range(60):
            mid = (lo + hi) / 2; lo, hi = (mid, hi) if marg(mid) < m0[r] else (lo, mid)
        alpha[r] = hi
    use = ~np.isnan(alpha); rows = np.nonzero(use)[0]
    if not len(rows):                                                       # no row with a correct probe and a reachable margin (e.g. an untrained model)
        return {"propagation_share": None, "random_edit_share": None, "null_value": None, "unedited_correct": None, "n": 0, "n_drawn": int(n),
                "n_probe_correct": int(ok.sum()), "p": int(p), "chain_len": int(ell)}
    Xe = Xp[rows] + alpha[rows, None] * Dn[rows]; Rv = rng.standard_normal(Xe.shape).astype(np.float32); Rv *= (alpha[rows] / np.linalg.norm(Rv, axis=1))[:, None]
    fe = cont_fn(Xe.astype(np.float32), X[rows, p:]); fr = cont_fn((Xp[rows] + Rv).astype(np.float32), X[rows, p:]); fu = cont_fn(Xp[rows], X[rows, p:])
    return {"propagation_share": float((fe == tgt[rows]).mean()) if len(rows) else None, "random_edit_share": float((fr == tgt[rows]).mean()) if len(rows) else None,
            "null_value": float((fu == tgt[rows]).mean()) if len(rows) else None, "unedited_correct": float((fu == Y[rows, -1]).mean()) if len(rows) else None,
            "n": int(len(rows)), "n_drawn": int(n), "n_probe_correct": int(ok.sum()), "p": int(p), "chain_len": int(ell),
            "edit_norm_median": float(np.median(alpha[rows])) if len(rows) else None, "state_norm_median": float(np.median(np.linalg.norm(Xp[rows], axis=1))) if len(rows) else None}


def donor_swap(m, dev, amp, ell, bs, n=1000, seed=16204):
    """whole-memory swap after item p (up to three evenly spaced items) on length-ell chains; donor_consistent = final == donor's g_p o the
    recipient's later items; recipient_consistent = final == the recipient's own final; chance_overlap = both targets equal."""
    fam = Family("s5"); rng = np.random.default_rng(seed); s0, X, Y = make_batch(fam, rng, n, ell); dn = rng.permutation(n)
    pos = sorted({max(1, min(ell - 1, round(ell * k / 4))) for k in (1, 2, 3)}); out = {}
    for p in pos:
        _, kd = rollout(m, s0[dn], X[dn, :p], dev, amp, bs, every=False, keep=(p,)); Sd = kd[p]; sh = (12, DM.DH, DM.DH)
        fin, _ = rollout(m, s0, X, dev, amp, bs, every=False, override=(p, lambda S, r_: torch.tensor(Sd[r_].reshape((len(r_),) + sh), device=dev, dtype=S.dtype)))
        td = comp_chain(Y[dn, p - 1], X[:, p:])[:, -1] if p < ell else Y[dn, p - 1]; tr = Y[:, -1]
        out["pos%d" % p] = {"donor_consistent": float((fin == td).mean()), "recipient_consistent": float((fin == tr).mean()), "chance_overlap": float((td == tr).mean()), "n": int(n)}
    return out


@torch.no_grad()
def key_cos(m, dev, amp, n=256, L=8, seed=16205):
    """per head: mean |cos| between the write keys of an item segment's tokens (distinct pairs), keys = qkv(u)[k rows], u the write input."""
    rng = np.random.default_rng(seed); s0, X, _ = make_batch(Family("s5"), rng, n, L); acc = None; cnt = 0
    S = m.write(m.start(n, dev), m.ids_start(s0, dev))
    for t in range(L):
        cap = {}
        with amp_ctx(amp):
            S = m.write(S, m.ids_item(X[:, t], dev), cap=cap)
        u = (1 - m.mem.gamma) * m.mem.ln(cap["h6"].float()) + m.mem.gamma * m.mem.ln_w(cap["hw"].float())
        k = m.mem.qkv(u)[..., 768:1536].view(n, u.shape[1], DM.H_ADD + DM.H_DEL, DM.DH); k = Fn.normalize(k, dim=-1)
        cs = torch.einsum("bthd,bshd->bhts", k, k).abs(); T = cs.shape[-1]; off = ~torch.eye(T, dtype=torch.bool, device=dev)
        v = cs[..., off].mean(-1).sum(0); acc = v if acc is None else acc + v; cnt += n
    return [round(float(x_), 5) for x_ in acc / cnt]


# ------------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["direct", "delta"]); ap.add_argument("--family", default="s5", choices=["s5"])
    ap.add_argument("--l_max", type=int, default=32); ap.add_argument("--steps", type=int, default=24000); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--micro", type=int, default=1)
    ap.add_argument("--mem_write", default="top", choices=["top"]); ap.add_argument("--mem_layer", type=int, default=6)
    ap.add_argument("--write_mix_start", type=int, default=2000); ap.add_argument("--write_mix_end", type=int, default=6000)
    ap.add_argument("--ckpt_every", type=int, default=2000); ap.add_argument("--resume", default=None); ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_bs", type=int, default=250); ap.add_argument("--test_n", type=int, default=2000); ap.add_argument("--ext_n", type=int, default=500)
    ap.add_argument("--long_len", type=int, default=512); ap.add_argument("--probe_n_fit", type=int, default=8000); ap.add_argument("--probe_n_score", type=int, default=2000)
    ap.add_argument("--edit_p", type=int, default=0, help="state_edit position p (chains of length 2p); default l_max / 2 (chains of length l_max): note 162-1")
    ap.add_argument("--edit_n", type=int, default=1000); ap.add_argument("--donor_n", type=int, default=1000); ap.add_argument("--rec_every", type=int, default=4000)
    ap.add_argument("--auto_micro", action="store_true"); ap.add_argument("--mem_budget", type=float, default=21.0); ap.add_argument("--mem_check", action="store_true")
    ap.add_argument("--sched_scale", type=float, default=1.0, help="smokes only"); ap.add_argument("--data_dir", default=os.path.join(ROOT, "bench", "data", "s5"))
    ap.add_argument("--post_train_from", default=None, help="§166: post-train <src run> (dir under boxes/runs_fsq or a path): its args + model.pt")
    ap.add_argument("--pt_steps", type=int, default=1000); ap.add_argument("--pt_lr", type=float, default=2e-5)
    ap.add_argument("--init_mode", default="zero", choices=["zero", "pass"]); ap.add_argument("--init_p", type=float, default=0.5)
    ap.add_argument("--state_source", default="current", choices=["current"])
    ap.add_argument("--eval_only", action="store_true", help="re-evaluate <out>/model.pt with the run's own args.json (no training; args -> args_eval.json; log appended)")
    ap.add_argument("--fp32", action="store_true"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.eval_only:                                                         # the run's own args (Audit 2026-10-09: §166 S5 evals failed on a cloud data_dir)
        keep_ = {k: getattr(a, k) for k in ("out", "device", "eval_only", "eval_bs", "test_n", "ext_n", "long_len", "probe_n_fit", "probe_n_score", "edit_n", "donor_n", "fp32", "data_dir")}
        for k, v in json.load(open(os.path.join(a.out, "args.json"))).items():
            if hasattr(a, k):
                setattr(a, k, v)
        for k, v in keep_.items():
            setattr(a, k, v)
        a.resume = None; a.auto_micro = False; a.mem_check = False
    if a.post_train_from and not a.eval_only:                               # §166: the source run's args, this run's own out / device / pt_* / eval sizes
        src_ = a.post_train_from if os.path.isdir(a.post_train_from) else os.path.join(ROOT, "boxes", "runs_fsq", a.post_train_from)
        keep_ = {k: getattr(a, k) for k in ("out", "device", "post_train_from", "pt_steps", "pt_lr", "init_mode", "init_p", "state_source", "eval_bs", "test_n", "ext_n",
                                            "long_len", "probe_n_fit", "probe_n_score", "edit_n", "donor_n", "fp32", "mem_check", "sched_scale", "micro", "data_dir")}   # data_dir: local (Audit)
        for k, v in json.load(open(os.path.join(src_, "args.json"))).items():
            if hasattr(a, k):
                setattr(a, k, v)
        for k, v in keep_.items():
            setattr(a, k, v)
        assert a.arm == "delta", "§166: deltas5 post-training is for the delta arm"
        a.pt_src_dir = src_; a.post_train_from = os.path.basename(os.path.normpath(a.post_train_from)); a.steps = a.pt_steps; a.resume = None; a.ckpt_every = 0; a.auto_micro = False
    a.carrier = "deltamem" if a.arm == "delta" else "direct"; os.makedirs(a.out, exist_ok=True); dev = a.device; sc = a.sched_scale
    if a.arm == "direct":
        a.l_max = 64; a.mem_write = a.write_mix_start = a.write_mix_end = None
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); fam = Family("s5")
    amp = None if (a.fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    g = s5.load_gpt2(dev); m = (Direct(g) if a.arm == "direct" else DeltaS5(g, a.mem_layer, a.mem_write)).to(dev)
    D_ = a.arm == "delta"

    PRNG = np.random.default_rng(16800 + a.seed)                            # §166: the pass mask / source chains' own rng (paired batches with PLAIN)

    def train_loss(s0, X, y, pm=None):
        if not D_:
            return m.loss(s0, X, y, dev)
        if pm is None or not pm.any():
            return m.loss(m.run(s0, X, dev), y, dev)
        Sg, g, _, _ = pass_states(m, PRNG, int(pm.sum()), dev, amp); yp = comp_chain(g, X[pm])[:, -1]   # §166 pass: no segment 0; target g o items (own rng: paired batches)
        lp = m.loss(m.run(None, X[pm], dev, S=Sg.to(dev)), yp, dev); np_ = int(pm.sum())
        if np_ == len(X):
            return lp
        return (lp * np_ + m.loss(m.run(s0[~pm], X[~pm], dev), y[~pm], dev) * (len(X) - np_)) / len(X)

    def mem_check(micro):
        torch.cuda.reset_peak_memory_stats(dev); r_ = np.random.default_rng(0); s0, X, Y = make_batch(fam, r_, a.bs // micro, a.l_max); m.train()
        with amp_ctx(amp):
            l_ = train_loss(s0, X, Y[:, -1])
        l_.backward(); m.zero_grad(set_to_none=True); return torch.cuda.max_memory_allocated(dev) / 2 ** 30
    if a.mem_check or (a.auto_micro and dev.startswith("cuda") and not a.resume):
        mc = []
        while True:
            pk = mem_check(a.micro); mc.append({"micro": a.micro, "peak_gib": pk}); print("MEM", a.micro, pk, flush=True)
            if pk <= a.mem_budget or a.micro >= a.bs or a.mem_check:
                break
            a.micro *= 2
        json.dump({"checks": mc, "micro": a.micro, "peak_gib": mc[-1]["peak_gib"], "L": a.l_max, "bs": a.bs, "arm": a.arm}, open(os.path.join(a.out, "mem_check.json"), "w", newline="\n"), indent=1)
        if a.mem_check:
            return
        assert mc[-1]["peak_gib"] <= a.mem_budget, "memory: %.1f GiB at micro %d" % (mc[-1]["peak_gib"], a.micro)
    info = {**vars(a), "train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_driver": driver_version(), **env_info(),
            "segments": "direct: [start text, items' 6-token texts, ' ='] -> state text" if not D_ else "[EOT, start text], [EOT, item text] per item, decoder [EOT] read-only -> state text"}
    json.dump(info, open(os.path.join(a.out, ("args_resume_%d.json" % int(time.time())) if a.resume else ("args_eval.json" if a.eval_only else "args.json")), "w", newline="\n"), indent=1)
    opt = make_opt(m, a.lr); step0 = 1; n_nonfinite = 0; wg_hist = []; wwin = []; wlast = []; rec_hist = []; hist = []
    log = open(os.path.join(a.out, "log.jsonl"), "a" if (a.resume or a.eval_only) else "w", newline="\n"); t0 = time.time()
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False); m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"].cpu()); step0 = ck["step"] + 1; n_nonfinite = ck["n_nonfinite"]; wg_hist = ck["wg_hist"]; rec_hist = ck["rec_hist"]; t0 = time.time() - ck["elapsed"]
        if dev.startswith("cuda") and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device=dev)
        log.write(json.dumps({"step": ck["step"], "resumed": a.resume}) + "\n"); log.flush()
    elif a.eval_only:                                                       # the trained model; no training steps
        m.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev)); step0 = a.steps + 1
        log.write(json.dumps({"eval_only": True, "at": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n"); log.flush()
    elif D_ and a.post_train_from:                                          # §166: the source's trained weights (no g_add re-calibration)
        m.load_state_dict(torch.load(os.path.join(a.pt_src_dir, "model.pt"), map_location=dev))
    elif D_:                                                                # note 147-2: g_add = the median read-norm ratio at step 0 (fade 1 for the measurement)
        m.mem.fade.fill_(1.0); r0 = read_ratio(m, dev, amp); m.mem.s_add.data.fill_(math.log(max(r0, 1e-12))); m.mem.fade.fill_(0.0)
        json.dump({"g_add_init": r0}, open(os.path.join(a.out, "s162_init.json"), "w", newline="\n"))
    for step in range(step0, a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = (a.pt_lr * min(1.0, step / 50)) if a.post_train_from else lr_at(a.lr, gr, step, a.steps)   # §166: constant after 50 warm-up steps
        m.train()
        if D_ and a.post_train_from:                                        # §166: the trained model's final schedule values
            m.mem.fade.fill_(1.0); m.mem.gamma.fill_(1.0)
        elif D_:
            m.mem.fade.fill_(min(1.0, step / max(1, int(500 * sc))))
            m.mem.gamma.fill_(0.0 if step <= a.write_mix_start * sc else min(1.0, (step - a.write_mix_start * sc) / max(1.0, (a.write_mix_end - a.write_mix_start) * sc)))
        L = int(rng.integers(1, lmax_sched(1e12 if a.post_train_from else step / sc, a.l_max) + 1)); s0, X, Y = make_batch(fam, rng, a.bs, L); opt.zero_grad(); tot = 0.0; ch = a.bs // a.micro
        PM = (PRNG.random(a.bs) < a.init_p) if (a.post_train_from and a.init_mode == "pass" and a.init_p > 0) else None   # §166 pass rows (own rng, Audit)
        for k_ in range(a.micro):
            sl = slice(k_ * ch, (k_ + 1) * ch)
            with amp_ctx(amp):
                lk = train_loss(s0[sl], X[sl], Y[sl, -1], None if PM is None else PM[sl])
            (lk / a.micro).backward(); tot += float(lk.item()) / a.micro
        if D_:
            wwin.append(write_grad_norm(m))
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if not (math.isfinite(tot) and bool(torch.isfinite(gn))):
            log.write(json.dumps({"step": step, "nonfinite": 1}) + "\n"); log.flush(); n_nonfinite += 1; opt.zero_grad()
        else:
            opt.step(); hist.append(tot)
        if step % a.log_every == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(hist)) if hist else None, "L_max": lmax_sched(1e12 if a.post_train_from else step / sc, a.l_max), "sec": round(time.time() - t0)}; hist = []
            if D_ and wwin:                                                 # note 155-1 / 160-1: spike vs the preceding 1k steps' median (+ the 1.0 floor)
                md_ = float(np.median(wlast)) if wlast else 0.0; mx_ = float(np.max(wwin)); late = step > a.write_mix_start * sc
                w_rec = {"step": step, "mean": float(np.mean(wwin)), "max": mx_, "median_prev_1k": md_, "spike_100x": bool(late and mx_ > 100 * md_ > 0),
                         "spike_100x_raw": bool(mx_ > 100 * md_ > 0), "spike_100x_floor": bool(late and mx_ > 100 * md_ > 0 and mx_ > 1.0)}
                wlast.extend(wwin); del wlast[:-int(1000 * sc) or None]; wwin = []; wg_hist.append(w_rec)
                rec.update({"g_add": float(torch.exp(m.mem.s_add)), "g_del": float(torch.exp(m.mem.s_del)), "fade": float(m.mem.fade), "gamma": float(m.mem.gamma),
                            "write_grad_norm": w_rec["mean"], "write_grad_spike_100x_floor": w_rec["spike_100x_floor"]})
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if D_ and a.rec_every and step % max(1, int(a.rec_every * sc)) == 0 and step < a.steps:   # §155 recurrence per window (descriptive)
            rr_ = {"step": step, "gamma": float(m.mem.gamma), **s155_recurrence(m, dev, amp)}; rec_hist.append(rr_); log.write(json.dumps({"step": step, "s155_recurrence": rr_}) + "\n"); log.flush()
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps:
            cp = os.path.join(a.out, "ckpt_last.pt")
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(dev) if dev.startswith("cuda") else None, "step": step, "n_nonfinite": n_nonfinite,
                        "wg_hist": wg_hist, "rec_hist": rec_hist, "elapsed": time.time() - t0}, cp)
    train_sec = round(time.time() - t0)
    if not a.eval_only:
        torch.save(m.state_dict(), os.path.join(a.out, "model.pt"))
    m.eval(); t1 = time.time()
    res = {"arm": a.arm, "family": "s5", "seed": a.seed, "l_max": a.l_max, "train_sec": train_sec, "n_nonfinite": n_nonfinite, "eval_dtype": str(amp), "micro": a.micro,
           "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_gpu": info["train_gpu"], "train_driver": info["train_driver"]}
    Xte, Yte = s5.load_csv(os.path.join(a.data_dir, "S5=512.csv"), limit=a.test_n); s5.check_labels(Xte, Yte); z = np.zeros(len(Xte), np.int64)
    if not D_:
        acc = {}
        for t in (1, 2, 4, 8, 16, 32, 64, 128):
            pr = np.concatenate([m.greedy(z[i:i + a.eval_bs], Xte[i:i + a.eval_bs, :t], dev) for i in range(0, len(Xte), a.eval_bs)]); acc[str(t)] = float((pr == Yte[:, t - 1]).mean())
        res.update({"acc_by_length": acc, "acc_by_length_scored": [str(t) for t in (1, 2, 4, 8, 16, 32, 64)], "test_n": int(len(Xte))})
    else:
        Lt = min(a.long_len, Xte.shape[1]); pred, _ = rollout(m, z, Xte[:, :Lt], dev, amp, a.eval_bs); C = pred == Yte[:, :Lt]
        res["final_acc_by_length"] = {str(t): float(C[:, t - 1].mean()) for t in range(1, a.l_max + 1)}
        res["acc_by_length_cumulative"] = {str(t): float(C[:, :t].all(1).mean()) for t in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if t <= Lt}
        res["long_acc"] = float(C.all(1).mean()); res["long_len"] = int(Lt); res["test_n"] = int(len(Xte))
        res["acc_by_length_ext"] = {str(t): float(C[:, t - 1].mean()) for t in (48, 64) if t <= Lt}   # §166: final-state accuracy beyond the trained 32
        if a.ext_n:
            Xx, Yx = s5.load_csv(os.path.join(a.data_dir, "S5=2048.csv"), limit=a.ext_n); s5.check_labels(Xx, Yx)
            px, _ = rollout(m, np.zeros(len(Xx), np.int64), Xx, dev, amp, a.eval_bs); res["acc_2048"] = float((px == Yx).all(1).mean()); res["acc_2048_n"] = int(len(Xx))
        ell = min(8, a.l_max); res["donor"] = donor_swap(m, dev, amp, ell, a.eval_bs, n=a.donor_n)
        sh = (12, DM.DH, DM.DH)
        def run_fn(s0_, X_, keep):
            return rollout(m, s0_, X_, dev, amp, a.eval_bs, every=False, keep=tuple(keep))[1]
        def cont_fn(Sf, Xr):
            out_ = []
            for i in range(0, len(Sf), a.eval_bs):
                S_ = torch.tensor(Sf[i:i + a.eval_bs].reshape((-1,) + sh), device=dev)
                with torch.no_grad(), amp_ctx(amp):
                    S_ = m.run(None, Xr[i:i + a.eval_bs], dev, S=S_); out_.append(m.greedy(S_))
            return np.concatenate(out_)
        P = [p for p in (2, 4, 8, 16, 32) if p <= a.l_max]
        pe = a.edit_p or a.l_max // 2                                       # note 162-1 (3b0cf289): after item p = l_max / 2 on chains of length l_max
        res["s162"] = {**probes(run_fn, P, dev, a.probe_n_fit, a.probe_n_score), "state_edit": state_edit(run_fn, cont_fn, 2 * pe, dev, n=a.edit_n, n_fit=a.probe_n_fit),
                       "key_cos": key_cos(m, dev, amp)}
        res["s155"] = {"recurrence": rec_hist + [{"step": "end", **s155_recurrence(m, dev, amp)}], "write_grad_norm": wg_hist, "n_nonfinite": n_nonfinite,
                       "write_grad_spikes": [x_ for x_ in wg_hist if x_.get("spike_100x")], "write_grad_spikes_floor": [x_ for x_ in wg_hist if x_.get("spike_100x_floor")],
                       "mem_write": a.mem_write, "write_mix": [a.write_mix_start, a.write_mix_end], "gamma_final": float(m.mem.gamma),
                       "g_add": float(torch.exp(m.mem.s_add)), "g_del": float(torch.exp(m.mem.s_del))}
    if a.post_train_from:                                                   # §166
        res["s166"] = {"post_train_from": a.post_train_from, "pt_steps": a.pt_steps, "pt_lr": a.pt_lr, "init_mode": a.init_mode, "init_p": a.init_p, "state_source": a.state_source,
                       "pass_def": "init_p of chains start from the current model's final memory of a random-start chain of one length U(1..32) per batch; segment 0 skipped; target g o items"}
    res["eval_sec"] = round(time.time() - t1)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps({k: res[k] for k in ("acc_by_length", "final_acc_by_length", "long_acc") if k in res})[:600], flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
