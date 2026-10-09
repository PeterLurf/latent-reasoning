"""THEORY §167 DAS-EDIT (c08fbbd0 + note 167-1; scorer theory/score_s167.py): distributed alignment search on the DeltaMem-R memory of kept models.
An orthonormal R in R^{D x k} (torch orthogonal parametrisation; D = 12 x 64 x 64 = the flattened memory, all heads) patches the recipient's memory at the
variable's point, S_rec <- S_rec + R R^T (S_don - S_rec), then the frozen model continues the recipient. Loss: CE on the answer digits vs the counterfactual.
Fit on 4,000 pairs from item set A (note 167-3: PCA init, Adam lr {1e-3, 3e-3}, <= 5,000 steps, batch 64, early stop on 500 validation pairs), IIA on 1,000 pairs from a DISJOINT set B.
    MULT (multpair models, e.g. s161_mix_4x4_s1): the accumulator a_t after pair t in {4, 8, 12}; counterfactual = the recipient's later digits continued
         from the donor's a_t (multpair.later_gt, as donor_delta); pairs with a_t differing. A = generator 4x4 items minus the bench test; B = the bench test.
    S5 (deltas5 delta models, e.g. s162_short_s1): g after item p = l_max / 2 on chains of length l_max; counterfactual = g_don o the recipient's later
         items; pairs with g differing. A / B = random-start chains from two seeds (disjoint by construction).
eval_das.json: {family, points: {t: {k: {k: {iia, iia_train, n, steps}}, unchanged_target_share, random_subspace, shuffled_targets, subspace_var_share,
necessity_drop, random_model, read_alignment, n_pairs}}, carry_matched, per_head_type, scaling, gates}. IIA = row-level: every counterfactual digit
(MULT: the later digits; S5: the final state) decoded exactly.
    python -m boxes.das167 --run boxes/runs_fsq/s161_mix_4x4_s1 [--device cuda:0] [--ks 1,2,4,8,16,64,2048]"""
import os
os.environ["BOXES_NB"] = "5"
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
import boxes.deltamem as DM  # noqa: E402

D = (DM.H_ADD + DM.H_DEL) * DM.DH * DM.DH
SH = (DM.H_ADD + DM.H_DEL, DM.DH, DM.DH)


def amp_ctx(amp):
    return torch.autocast("cuda", dtype=amp, enabled=amp is not None)


class Rot(nn.Module):
    """an orthonormal D x k basis R = QR(W).Q of a free D x k parameter W (a differentiable orthogonal parametrisation; torch's Householder one is
    too slow at D = 49,152, k = 2,048); coords: optional index subset (per-head-type DAS)."""
    def __init__(self, k, dev, coords=None, init=None):
        super().__init__(); self.coords = coords; d = D if coords is None else len(coords)
        self.W = nn.Parameter(init.clone().to(dev) if init is not None else torch.randn(d, k, device=dev) / d ** 0.5)

    def R(self):
        return torch.linalg.qr(self.W).Q                                    # (d, k), orthonormal columns

    def patch(self, Sr, Sd):
        """Sr / Sd (B, D) fp32 -> Sr + R R^T (Sd - Sr) (on coords only when given)."""
        R = self.R()
        if self.coords is None:
            return Sr + ((Sd - Sr) @ R) @ R.T
        out = Sr.clone(); c = self.coords; out[:, c] = Sr[:, c] + ((Sd[:, c] - Sr[:, c]) @ R) @ R.T; return out


class FixedRot:
    """an unfitted orthonormal basis (the random-subspace control) / the whole-state swap (R = I: patch -> Sd)."""
    def __init__(self, Q=None):
        self.Q = Q

    def R(self):
        return self.Q

    def patch(self, Sr, Sd):
        return Sd if self.Q is None else Sr + ((Sd - Sr) @ self.Q) @ self.Q.T


def random_rot(k, dev, gen):
    Q, _ = torch.linalg.qr(torch.randn(D, k, generator=gen)); return FixedRot(Q.to(dev))


# ---------------------------------------------------------------------------------------------------------- MULT family (multpair)
class MultFam:
    name = "mult"

    def __init__(self, run, dev, amp, random_model=False):
        import boxes.multpair as MP
        from boxes.multcol import DTOK, gen_operands, load_bench
        self.MP, self.DTOK, self.dev, self.amp = MP, torch.tensor(DTOK, device=dev), dev, amp
        a = json.load(open(os.path.join(run, "args.json"))); self.args = a
        m = MP.MultDelta(s5.load_gpt2(dev), MP.PairFam(), a.get("mem_layer", 6), a.get("mem_write")).to(dev); m.add_reset = a.get("add_reset"); m.cidx_off = None; m.cidx_r = None
        if random_model:                                                    # note 167-1: an untrained DeltaMem-R (random read-out W_o, fade / gamma 1), GPT-2 trunk
            g_ = torch.Generator().manual_seed(16799)
            with torch.no_grad():
                m.mem.wo.weight.copy_(torch.randn(m.mem.wo.weight.shape, generator=g_) * 0.02); m.mem.fade.fill_(1.0)
                if hasattr(m.mem, "gamma"):
                    m.mem.gamma.fill_(1.0)
        else:
            m.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location=dev))
        m.eval()
        for p_ in m.parameters():
            p_.requires_grad_(False)
        self.m = m; teA, teB = load_bench(os.path.join(ROOT, "bench", "data", "mult_4x4_test.txt"))
        ex = {(x.tobytes(), y.tobytes()) for x, y in zip(teA, teB)} | {(y.tobytes(), x.tobytes()) for x, y in zip(teA, teB)}
        r = np.random.default_rng(16710); A, B = gen_operands(r, 6000, 4), gen_operands(r, 6000, 4)
        keep = np.array([(x.tobytes(), y.tobytes()) not in ex for x, y in zip(A, B)]); A, B = A[keep], B[keep]
        self.sets = {"A": (A[:4500], B[:4500]), "B": (teA, teB)}
        self.pairs, self.col, self.emit = MP.schedule(4, 4); self.T = len(self.pairs)

    def points(self):
        return [4, 8, 12]

    def states(self, A, B, t, bs=250):
        _, kept, _ = self.MP.rollout_delta(self.m, A, B, self.dev, self.amp, bs, keep_pos=[t]); return kept[t].reshape(len(A), -1)

    def make_pairs(self, A, B, t, rng, n):
        """recipient / donor index pairs (donor != recipient, a_t differing), the counterfactual digits (N, 8) and the later columns sl."""
        accA = self.MP.accumulators(A, B)[0]; ir = rng.permutation(len(A)); idn = rng.permutation(len(A)); ok = (ir != idn) & (accA[ir, t - 1] != accA[idn, t - 1])
        ir, idn = ir[ok][:n], idn[ok][:n]; Pr, _, _ = self.MP.columns(A[ir], B[ir])
        gt, sl = self.MP.later_gt(A[ir], B[ir], accA[idn, t - 1], t, self.pairs, self.col, self.emit, Pr)
        return ir, idn, gt, sl, Pr

    def cont_logits(self, A, B, S, t):
        """continue recipients (A, B) from memory S (B, D) after pair t; logits (B, n_emit_after, V) at the later emission steps, their columns."""
        m = self.m; self.MP.set_ops(m, A, B, self.dev); S = S.view((-1,) + SH); outs = []; cols = []
        for tt in range(t, self.T):
            with amp_ctx(self.amp):
                lg, S = m.step(S, tt)
            if self.emit[tt]:
                outs.append(lg.float()); cols.append(int(self.col[tt]))
        return torch.stack(outs, 1), cols

    def loss(self, lg, cols, gt):
        tgt = self.DTOK[torch.tensor(gt[:, cols], device=self.dev)]; return Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1))

    def decode(self, lg, cols):
        lut = {int(v): i for i, v in enumerate(self.DTOK.tolist())}; tok = lg.argmax(-1).cpu().numpy(); return np.vectorize(lambda x: lut.get(int(x), -1))(tok), cols

    def iia(self, dec, cols, gt):
        return (dec == gt[:, cols]).all(1)


# ---------------------------------------------------------------------------------------------------------- S5 family (deltas5)
class S5Fam:
    name = "s5"

    def __init__(self, run, dev, amp, random_model=False):
        import boxes.deltas5 as D5
        from boxes.fsq120 import Family, make_batch
        self.D5, self.dev, self.amp = D5, dev, amp; a = json.load(open(os.path.join(run, "args.json"))); self.args = a; self.l_max = int(a["l_max"])
        m = D5.DeltaS5(s5.load_gpt2(dev), a.get("mem_layer", 6), a.get("mem_write", "top")).to(dev)
        if random_model:
            g_ = torch.Generator().manual_seed(16799)
            with torch.no_grad():
                m.mem.wo.weight.copy_(torch.randn(m.mem.wo.weight.shape, generator=g_) * 0.02); m.mem.fade.fill_(1.0); m.mem.gamma.fill_(1.0)
        else:
            m.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location=dev))
        m.eval()
        for p_ in m.parameters():
            p_.requires_grad_(False)
        self.m = m; fam = Family("s5")
        s0a, Xa, _ = make_batch(fam, np.random.default_rng(16720), 4500, self.l_max); s0b, Xb, _ = make_batch(fam, np.random.default_rng(16721), 1000, self.l_max)
        self.sets = {"A": (s0a, Xa), "B": (s0b, Xb)}

    def points(self):
        return [self.l_max // 2]

    def states(self, s0, X, p, bs=250):
        return self.D5.rollout(self.m, s0, X[:, :p], self.dev, self.amp, bs, every=False, keep=(p,))[1][p]

    def make_pairs(self, s0, X, p, rng, n):
        G = self.D5.comp_chain(s0, X); ir = rng.permutation(len(X)); idn = rng.permutation(len(X)); ok = (ir != idn) & (G[ir, p - 1] != G[idn, p - 1])
        ir, idn = ir[ok][:n], idn[ok][:n]; gt = self.D5.comp_chain(G[idn, p - 1], X[ir, p:])[:, -1]; own = G[ir, -1]
        return ir, idn, gt[:, None], np.array([0]), own[:, None]

    def cont_logits(self, s0, X, S, p):
        m = self.m; S = S.view((-1,) + SH)
        with amp_ctx(self.amp):
            S = m.run(None, X[:, p:], self.dev, S=S)
        return S, [0]

    def loss(self, S, cols, gt):
        ans = torch.tensor(self.D5.STT[gt[:, 0]], device=self.dev)
        with amp_ctx(self.amp):
            lg = self.m.dec_logits(S, ans)
        return Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), ans.reshape(-1))

    def decode(self, S, cols):
        return self.m.greedy(S)[:, None], cols

    def iia(self, dec, cols, gt):
        return (dec == gt).all(1)


# ---------------------------------------------------------------------------------------------------------- the DAS engine
def evaluate(F, Xs, Sr_all, Sd_all, ir, idn, gt, t, rot, bs=125, ablate=None):
    """row-level IIA on the given pairs (rot None + Sd = Sr: unpatched; ablate (R, mean): the subspace mean-ablated)."""
    X1, X2 = Xs; ok = []
    for i in range(0, len(ir), bs):
        r_, d_ = ir[i:i + bs], idn[i:i + bs]; Sr = torch.tensor(Sr_all[r_], device=F.dev); Sd = torch.tensor(Sd_all[d_], device=F.dev)
        with torch.no_grad():
            if ablate is not None:
                R, mu = ablate; Sp = Sr - ((Sr - mu) @ R) @ R.T
            else:
                Sp = Sr if rot is None else rot.patch(Sr, Sd)
            out, cols = F.cont_logits(X1[r_], X2[r_], Sp, t); dec, cols = F.decode(out, cols)
        ok.append(F.iia(dec, cols, gt[i:i + bs]))
    return float(np.concatenate(ok).mean())


def pca_basis(Sr_all, Sd_all, tr, dev, kmax, n=4000):
    """note 167-3: the top-kmax principal directions (D, kmax) of S_don - S_rec over (up to n of) the fit pairs (randomised SVD, uncentred)."""
    idx = np.arange(min(n, len(tr[0]))); M = torch.tensor(Sd_all[tr[1][idx]] - Sr_all[tr[0][idx]], device=dev)
    _, _, V = torch.svd_lowrank(M, q=min(kmax + 16, len(idx)), niter=4); del M
    return V[:, :kmax].contiguous()


def fit_sweep(F, Xs, Sr_all, Sd_all, tr, va, t, k, V=None, lrs=(1e-3, 3e-3), steps=5000, seed=16730):
    """note 167-3: PCA-initialised (V given) fits at each lr, the best by validation IIA; returns (rot, steps, val_best, lr_best)."""
    best = None
    for lr in lrs:
        r_, st, vb = fit(F, Xs, Sr_all, Sd_all, tr, va, t, k, steps=steps, lr=lr, seed=seed, init=None if V is None else V[:, :k])
        if best is None or vb > best[2]:
            best = (r_, st, vb, lr)
    return best


def fit(F, Xs, Sr_all, Sd_all, tr, va, t, k, coords=None, steps=2000, bs=64, lr=1e-3, seed=16730, every=100, patience=3, init=None):
    """tr / va = (ir, idn, gt). Adam on R; early stop on validation IIA (checked every `every` steps)."""
    torch.manual_seed(seed + k); rot = Rot(k, F.dev, coords, init=init); opt = torch.optim.Adam(rot.parameters(), lr=lr); X1, X2 = Xs; rng = np.random.default_rng(seed + k)
    best, best_state, bad, st = -1.0, None, 0, 0
    for st in range(1, steps + 1):
        b = rng.integers(0, len(tr[0]), bs); r_, d_ = tr[0][b], tr[1][b]
        Sr = torch.tensor(Sr_all[r_], device=F.dev); Sd = torch.tensor(Sd_all[d_], device=F.dev)
        out, cols = F.cont_logits(X1[r_], X2[r_], rot.patch(Sr, Sd), t); loss = F.loss(out, cols, tr[2][b])
        opt.zero_grad(); loss.backward(); opt.step()
        if st % every == 0 or st == steps:
            v = evaluate(F, Xs, Sr_all, Sd_all, va[0], va[1], va[2], t, rot)
            if v > best + 1e-4:
                best, bad = v, 0; best_state = {k_: v_.detach().clone() for k_, v_ in rot.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        rot.load_state_dict(best_state)
    return rot, st, best


def read_alignment(F, Xs, Sr_all, ir, t, rot, n=256):
    """the share of R's columns inside the downstream read's row space: per head, the span of the queries (additive: phi(q); delta: normalised q) of the
    next pass's read over n recipients' tokens (99% energy), acting on S's key index; returns ||P R||^2_F / k."""
    m = F.m; mem = m.mem; X1, X2 = Xs; r_ = ir[:n]
    with torch.no_grad():
        S = torch.tensor(Sr_all[r_], device=F.dev).view((-1,) + SH)
        if F.name == "mult":
            F.MP.set_ops(m, X1[r_], X2[r_], F.dev); x = m.g.embed(m.tokens(t, len(r_), F.dev))
        else:
            x = m.g.embed(m.ids_item(X2[r_, t], F.dev))
        cap = {}
        with amp_ctx(F.amp):
            DM.gpt_forward_mem_r(m.g, x, mem, S, cap=cap)
        h6 = cap["h6"].float(); q = mem.qkv(mem.ln(h6))[..., :768].view(h6.shape[0], h6.shape[1], DM.H_ADD + DM.H_DEL, DM.DH)
        qa = Fn.elu(q[:, :, :DM.H_ADD]) + 1; qd = Fn.normalize(q[:, :, DM.H_ADD:], dim=-1); Qh = torch.cat([qa, qd], 2).flatten(0, 1)   # (n T, 12, DH)
        R = rot.R().detach().view(DM.H_ADD + DM.H_DEL, DM.DH, DM.DH, -1); tot = 0.0
        for h in range(DM.H_ADD + DM.H_DEL):
            U, s, _ = torch.linalg.svd(Qh[:, h].T, full_matrices=False); e = (s ** 2).cumsum(0) / (s ** 2).sum(); r = int((e < 0.99).sum()) + 1; U = U[:, :r]   # (DH, r)
            Rh = R[h]                                                       # additive: S[k, v]; delta: S[v, k] (the read contracts q with k)
            P = torch.einsum("ar,ab...->rb...", U, Rh) if h < DM.H_ADD else torch.einsum("ar,ba...->br...", U, Rh)
            tot += float((P ** 2).sum())
    return tot / R.shape[-1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ks", default="1,2,4,8,16,64,2048"); ap.add_argument("--ctrl_ks", default="16,64"); ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--n_train", type=int, default=4000); ap.add_argument("--n_val", type=int, default=500); ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--extras", action="store_true", help="carry_matched / per_head_type / scaling (descriptive)"); ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = a.device; t0 = time.time(); ks = [int(x) for x in a.ks.split(",")]; cks = [int(x) for x in a.ctrl_ks.split(",")]
    amp = None if (a.fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    fam_name = "s5" if json.load(open(os.path.join(a.run, "args.json"))).get("family") == "s5" else "mult"; FC = S5Fam if fam_name == "s5" else MultFam
    out_ = a.out or os.path.join(a.run, "eval_das.json")
    if os.path.exists(out_) and not os.path.exists(out_.replace("eval_das.json", "eval_das_v1.json")):   # note 167-3: the first fit kept as v1
        os.replace(out_, out_.replace("eval_das.json", "eval_das_v1.json"))
    F = FC(a.run, dev, amp); res = {"family": fam_name, "run": os.path.basename(a.run.rstrip("/")), "points": {}, "ks": ks, "ctrl_ks": cks, "D": D, "gates": {},
                                    "iia_def": "row level: every counterfactual digit (MULT: the later digits; S5: the final state) decoded exactly"}
    gen = torch.Generator().manual_seed(16740)
    for t in F.points():
        A1, A2 = F.sets["A"]; B1, B2 = F.sets["B"]; rng = np.random.default_rng(16750 + t)
        SA = F.states(A1, A2, t); SB = F.states(B1, B2, t)
        tr_all = F.make_pairs(A1[:4000], A2[:4000], t, rng, a.n_train); va_ir, va_idn, va_gt, _, _ = F.make_pairs(A1[4000:], A2[4000:], t, rng, a.n_val)
        va = (va_ir + 4000, va_idn + 4000, va_gt); tr = tr_all[:3]
        te_ir, te_idn, te_gt, te_sl, te_own = F.make_pairs(B1, B2, t, rng, a.n_test); te = (te_ir, te_idn, te_gt); XA, XB = (A1, A2), (B1, B2)
        pt = {"k": {}, "n_pairs": {"train": int(len(tr[0])), "val": int(len(va[0])), "test": int(len(te[0]))}}
        pt["unchanged_target_share"] = float(((te_gt[:, te_sl] if fam_name == "mult" else te_gt) == (te_own[:, te_sl] if fam_name == "mult" else te_own)).all(1).mean())
        pt["k"][str(D)] = {"iia": evaluate(F, XB, SB, SB, te[0], te[1], te[2], t, FixedRot()), "iia_train": None, "n": int(len(te[0])), "steps": 0,
                           "def": "k = D: the whole-state swap"}
        Xc = torch.tensor(SA[:2000] - SA[:2000].mean(0), device=dev); tot_var = float((Xc ** 2).sum()); mu = torch.tensor(SA.mean(0), device=dev)
        own_ok = evaluate(F, XB, SB, SB, te[0], te[0], te_own, t, None)
        pt["own_accuracy_unpatched"] = own_ok; pt["random_subspace"] = {}; pt["subspace_var_share"] = {}; pt["necessity_drop"] = {}; pt["read_alignment"] = {}
        pt["necessity_random"] = {}; pt["fit"] = {"init": "pca (top-k of S_don - S_rec over the fit pairs)", "lrs": [1e-3, 3e-3], "steps_max": a.steps, "lr_best": {}}
        V = pca_basis(SA, SA, tr, dev, max(ks))
        for k in ks:
            rot, st, vbest, lrb = fit_sweep(F, XA, SA, SA, tr, va, t, k, V=V, steps=a.steps); pt["fit"]["lr_best"][str(k)] = lrb
            iia = evaluate(F, XB, SB, SB, te[0], te[1], te[2], t, rot); sub = np.arange(min(1000, len(tr[0])))
            iia_tr = evaluate(F, XA, SA, SA, tr[0][sub], tr[1][sub], tr[2][sub], t, rot)
            pt["k"][str(k)] = {"iia": iia, "iia_train": iia_tr, "n": int(len(te[0])), "steps": int(st), "val_best": vbest}
            R = rot.R().detach(); pt["subspace_var_share"][str(k)] = float(((Xc @ R) ** 2).sum()) / tot_var
            pt["necessity_drop"][str(k)] = own_ok - evaluate(F, XB, SB, SB, te[0], te[0], te_own, t, None, ablate=(R, mu))
            rr_ = random_rot(k, dev, gen); pt["random_subspace"][str(k)] = evaluate(F, XB, SB, SB, te[0], te[1], te[2], t, rr_)
            pt["necessity_random"][str(k)] = own_ok - evaluate(F, XB, SB, SB, te[0], te[0], te_own, t, None, ablate=(rr_.R(), mu))   # note 167-3
            pt["read_alignment"][str(k)] = read_alignment(F, XB, SB, te[0], t, rot)
            print(t, k, pt["k"][str(k)], flush=True)
        pt["shuffled_targets"] = {}
        for k in cks:                                                       # R fitted to a mismatched donor's counterfactuals
            sh = np.random.default_rng(16760 + k).permutation(len(tr[0])); trs = (tr[0], tr[1][sh], tr[2]); tes = np.random.default_rng(16761 + k).permutation(len(te[0]))
            rot, _, _, _ = fit_sweep(F, XA, SA, SA, trs, va, t, k, V=pca_basis(SA, SA, trs, dev, k), steps=a.steps, seed=16770); pt["shuffled_targets"][str(k)] = evaluate(F, XB, SB, SB, te[0], te[1][tes], te[2], t, rot)
        res["points"][str(t)] = pt; json.dump(res, open(a.out or os.path.join(a.run, "eval_das.json"), "w", newline="\n"), indent=1)
    # random_model (note 167-1, required): the same DAS on an untrained DeltaMem-R at the control k's
    del F; torch.cuda.empty_cache() if dev.startswith("cuda") else None; Fr = FC(a.run, dev, amp, random_model=True)
    for t in Fr.points():
        A1, A2 = Fr.sets["A"]; B1, B2 = Fr.sets["B"]; rng = np.random.default_rng(16750 + t); SA = Fr.states(A1, A2, t); SB = Fr.states(B1, B2, t)
        tr = Fr.make_pairs(A1[:4000], A2[:4000], t, rng, a.n_train)[:3]; va_ir, va_idn, va_gt, _, _ = Fr.make_pairs(A1[4000:], A2[4000:], t, rng, a.n_val)
        va = (va_ir + 4000, va_idn + 4000, va_gt); te_ir, te_idn, te_gt, _, _ = Fr.make_pairs(B1, B2, t, rng, a.n_test); res["points"][str(t)]["random_model"] = {}
        for k in cks:
            rot, _, _, _ = fit_sweep(Fr, (A1, A2), SA, SA, tr, va, t, k, V=pca_basis(SA, SA, tr, dev, k), steps=a.steps); res["points"][str(t)]["random_model"][str(k)] = evaluate(Fr, (B1, B2), SB, SB, te_ir, te_idn, te_gt, t, rot)
    res["extras"] = None; res["extras_note"] = "carry_matched / per_head_type / scaling (descriptive): not yet run"
    res["random_model_def"] = ("untrained DeltaMem-R: W_o ~ N(0, 0.02^2) (the default W_o = 0 would make the memory inert), fade 1, gamma 1; qkv / gate / LN at their "
                               "default init, i.e. qkv from block 6's c_attn and LN_w from block 11's ln_1 (pretrained-derived): an untrained MEMORY, not untrained "
                               "features (Audit); the pretrained GPT-2 trunk")
    res["sec"] = round(time.time() - t0); json.dump(res, open(a.out or os.path.join(a.run, "eval_das.json"), "w", newline="\n"), indent=1); print("DAS DONE", res["sec"], flush=True)


if __name__ == "__main__":
    main()
