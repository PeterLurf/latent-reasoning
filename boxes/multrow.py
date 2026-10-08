"""THEORY §139 DISTILL-2 (683318e2, note 139-1): multiplier-major multiplication on §120's FSQ carrier with an optional teacher-state
distillation. fsq120's model unchanged (FSQ120 + its §133 SOFT / §134 copy machinery); the segment gains the operand a and one multiplier
digit, as §132's tables / §137's columns:
    segment j = [EOT, carried code, a's digit tokens (" d", LSB first), b_j, position token j],  j = 1 .. L
FINAL answer only: the code after the last segment is decoded (the answer decoder) into S_L = a x (b_1 .. b_L), 8 digit tokens LSB first.
Curriculum: L ~ U(1, L_max), L_max 1 / 2 / 3 / 4 at 0 / 1k / 2k / 3k. Base (arm a): SOFT-16 (--d_q 16 --soft_beta 0.1) + §134 copy rows
(--copy_frac 0.1: another problem's S_L text -> copy writer -> code -> the same decoder), 24k, bs 64, lr 1e-4, the 127-3 guard, no restarts.
Arms b / c (--distill): for steps with a non-zero weight the TC-M teacher (boxes/runs_mult/m98_tc_4x4_s1) greedy-generates its CoT for the
batch; its hidden at --distill_layer at the last token of each S_j text gives the target; a linear readout (768 -> 8 x 10) on the student's
e_j = W_up(code_j) predicts S_j's 8 digits: KL to the teacher-probe distribution (probe_L<layer>.pt) if the probe's teacher-digit agreement
>= 0.99, else CE on the teacher's decoded S_j digits (s139.target_mode). Weight 1 to --distill_full, linear to 0 at --distill_end (arm b:
3000 / 4000; arm c: --distill_end 0 = never fades). Weight 0 -> the aux term (and the teacher) is skipped: exactly 0 gradient.
eval.json: test_acc (S_4 = the product exact on the 1,000 items), acc_by_length (L = 1..4), train_acc, copy_acc, donor_step2 (the code
after segment 2 from a donor: expected = donor S_2 + a_rec x (b_rec,3 b_rec,4) x 10^2, exact; donor / recipient / chance), s139
{target_mode, first_exact_step (first 1k-step quick test >= 0.99 on 256 test items), readout_sj_acc_end, probe_r2 {sj, bprefix} per j,
copy_acc}, rho (bf16; the scored one = the on-host --contraction_only --fp32 file: T0 1, K 3).
    python -m boxes.multrow --family mult_row --task 4x4 --seed 1 --copy_frac 0.1 --out boxes/runs_fsq/s139_a_4x4_s1
    python -m boxes.multrow ... --distill --distill_teacher boxes/runs_mult/m98_tc_4x4_s1 --distill_layer L --distill_full 3000 --distill_end 4000
THEORY §142 MULT-READ (a27cf1cf, note 142-1): NOISE-64 carrier (--d_q 64 --no_round --dither 0.25 --soft_beta 0 --copy_frac 0: continuous z_b,
U(+-0.25) at training hand-offs after 2k, clean at test), --lmax_steps 2000,5000,9000 (L_max 1 / 2 / 3 / 4 at 0 / 2k / 5k / 9k), --reader: at
every hand-off j < L the shared answer decoder decodes S_j from code_j (answer_ce, the 8-digit answer format) against the TC-M teacher's own
greedy S_j (--teacher_cache, built by scratch/teacher_cache142.py, sha-keyed to the deduped split; label_source "teacher_generated"); a row
whose teacher text is invalid gets no term at that hand-off; loss = CE_final (gold S_L) + mean over hand-offs of CE_j. Gate: once L_max = 4,
every 1k steps greedy final exact on a fixed 512-item train subset at L 4; the first >= 0.95 saves gate.pt (s142.gate_step).
B forks: --fork_from <a run>/gate.pt --fork_steps 20000 --removal early|late --removal_every 4000 --rewarm 200 --fixed_L 4: reader targets
removed S_1, S_2, S_3 (early) or S_3, S_2, S_1 (late) at +0 / +4k / +8k, each removal with a fresh AdamW and a 200-step LR re-warm-up
(1e-4, cosine over the fork); 10% of rows also drop the next target; a removed target has no loss term (exactly 0 gradient).
s142.fork_start / s142.stages: test_acc, acc_by_length, donor at the fork start, each removal boundary and the end."""
import os
os.environ["BOXES_NB"] = "5"
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
try:                                                                        # local hard memory cap
    from boxes import memcap  # noqa: E402
except ImportError:
    memcap = None
from boxes import s5  # noqa: E402
from boxes import core as bc  # noqa: E402
import boxes.fsq120 as F  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
from boxes.fsq120 import EOT, FSQ120, driver_version, env_info, lr_at, make_opt  # noqa: E402
from boxes.multcol import DTOK, COLTOK, dedup, load_bench, to_int, ridge_r2  # noqa: E402

NDIG = 8
LMAX = (1000, 2000, 3000)                                                   # §139: L_max 1 / 2 / 3 / 4 at 0 / 1k / 2k / 3k


def lmax(step, th=None):
    """L_max at `step`: L k while step <= th[k - 1], else 4 (§139 default LMAX; §142 --lmax_steps 2000,5000,9000)."""
    for L, lim in enumerate(th or LMAX, 1):
        if step <= lim:
            return L
    return 4


def split_sha(A, B):
    """§142: the teacher cache key (as scratch/teacher_cache142.py)."""
    import hashlib
    h = hashlib.sha256(); h.update(np.ascontiguousarray(A, np.int8).tobytes()); h.update(np.ascontiguousarray(B, np.int8).tobytes()); return h.hexdigest()


def digits8(v):
    v = np.asarray(v, np.int64); return np.stack([(v // 10 ** i) % 10 for i in range(NDIG)], -1)


def sj_values(A, B):
    """S_j = a x (b_1 .. b_j) for j = 1..n -> (N, n) int64."""
    a = to_int(A); return np.stack([a * to_int(B[:, :j]) for j in range(1, B.shape[1] + 1)], 1)


def distill_weight(step, on, full, end, sc=1.0):
    """§139 aux weight: 1 to `full`, linear to 0 at `end`, 0 after; end <= 0 = never fades (arm c); 0 when distillation is off."""
    if not on:
        return 0.0
    if end <= 0 or step <= full * sc:
        return 1.0
    return max(0.0, 1.0 - (step - full * sc) / ((end - full) * sc))


class RowFam:
    name = "mult_row"; n_items = 4; n_states = 10 ** NDIG
    item_tok = np.zeros((4, 1), np.int64)                                   # unused (MultRow.segment builds the tokens)

    def answer_tokens(self, v):
        return np.array(DTOK)[digits8(v)]

    def decode(self, toks):
        lut = {t: i for i, t in enumerate(DTOK)}; d = np.vectorize(lambda t: lut.get(int(t), -1))(toks)
        return np.where((d >= 0).all(1), (d * (10 ** np.arange(toks.shape[1]))[None]).sum(1), -1)


class MultRow(FSQ120):
    """segment j = [EOT, carried code, a digits, b_j, position j]; self.actx (B, n) a-digit tokens, self.btok (B, n) b-digit tokens."""
    def __init__(self, g, fam):
        super().__init__(g, fam); self.actx = None; self.btok = None; self.readout = None

    def segment(self, prev, items):
        g = self.g; B = items.shape[0]; dev = items.device; eot = g.embed(torch.full((B, 1), EOT, device=dev)); j = int(items[0])
        x = torch.cat([eot, prev.to(eot.dtype), g.embed(self.actx), g.embed(self.btok[:, j:j + 1]), g.embed(torch.full((B, 1), COLTOK[j + 1], device=dev))], 1)
        h, _ = g(x, None)
        return h[:, -1]

    def chain_read(self, start_emb, items, labels, rtok, rmask):
        """§142 reader chain: CE_final on the gold S_L after the last segment, and at every hand-off t < L - 1 (j = t + 1 < L) the shared answer
        decoder on code_j against rtok[:, t] (8 teacher answer tokens) for the rows of rmask[:, t] (valid and not removed). A hand-off with no
        rows adds no term. Returns (CE_final, mean over hand-offs of CE_j or None, number of reader hand-offs)."""
        B, L = items.shape; prev = start_emb; rd = []
        for t in range(L):
            h = self.segment(prev, items[:, t]); qs, q, zb = self.quant(h); e = self.embed_code(qs)
            if t < L - 1 and rmask is not None and bool(rmask[:, t].any()):
                rows = torch.nonzero(rmask[:, t]).squeeze(1); rd.append(self.answer_ce(e[rows], rtok[rows, t], qs[rows]))
            prev = e[:, None]
        fin = self.answer_ce(e, torch.tensor(self.fam.answer_tokens(labels[:, L - 1]), device=items.device), qs)
        return fin, (torch.stack(rd).mean() if rd else None), len(rd)


def amp_ctx(amp):
    return torch.autocast("cuda", dtype=amp, enabled=amp is not None)


def set_ops(m, A, B, dev):
    m.actx = torch.tensor(np.array(DTOK)[A], device=dev); m.btok = torch.tensor(np.array(DTOK)[B], device=dev)


@torch.no_grad()
def rollout(m, A, B, dev, amp, bs, L=None, override=None, record=False):
    """Hard chains over L multiplier digits; the answer decoded after every segment (acc_by_length). override: (p, codes) replaces the code
    after segment p (1-based). Returns dec (N, L) [, codes (N, L, d), e (N, L, 768)]."""
    if isinstance(m, MultRowDelta):                                         # §152: the reader decodes S_j after each segment
        dec = rollout_drow(m, A, B, dev, amp, bs, L=L, override=override)[0]
        return (dec, None, None) if record else dec
    N = len(A); L = L or B.shape[1]; dec = np.full((N, L), -1, np.int64); codes = np.zeros((N, L, F.D_Q), np.float32) if record else None
    E = np.zeros((N, L, 768), np.float32) if record else None
    for i in range(0, N, bs):
        sl = slice(i, i + bs); Bn = len(A[sl]); set_ops(m, A[sl], B[sl], dev); prev = m.start(Bn, dev)
        for t in range(L):
            with amp_ctx(amp):
                h = m.segment(prev, torch.full((Bn,), t, device=dev)); qs, q, zb = m.quant(h)
                if override is not None and override[0] == t + 1:
                    q = torch.tensor(override[1][sl], device=dev, dtype=qs.dtype); qs = q
                e = m.embed_code(qs); dec[sl, t] = m.fam.decode(m.answer_greedy(e, NDIG, qs).cpu().numpy())
            if record:
                codes[sl, t] = q.float().cpu().numpy(); E[sl, t] = e.float().cpu().numpy()
            prev = e[:, None]
    return (dec, codes, E) if record else dec


@torch.no_grad()
def copy_acc(m, dev, amp, rng, n=256):
    v = rng.integers(1, 10 ** NDIG, n); ans = torch.tensor(m.fam.answer_tokens(v), device=dev)
    with amp_ctx(amp):
        e, qs, q, zb = m.copy_write(ans); out = m.answer_greedy(e, NDIG, qs)
    return float((m.fam.decode(out.cpu().numpy()) == v).mean())


def donor_step2(m, A, B, dev, amp, bs, rng, n=1000):
    N = min(n, len(A)); ir, idn = rng.permutation(len(A))[:N], rng.permutation(len(A))[:N]; Ar, Br, Ad, Bd = A[ir], B[ir], A[idn], B[idn]
    _, dcodes, _ = rollout(m, Ad, Bd, dev, amp, bs, L=2, record=True); got = rollout(m, Ar, Br, dev, amp, bs, override=(2, dcodes[:, 1]))[:, -1]
    exp = sj_values(Ad, Bd)[:, 1] + to_int(Ar) * to_int(Br[:, 2:4]) * 100; own = sj_values(Ar, Br)[:, -1]
    bmix = to_int(np.concatenate([Bd[:, :2], Br[:, 2:4]], 1))                # §142 (b_d1 b_d2 b_r3 b_r4): the donor's b-prefix, the recipient's rest
    return {"pos2": {"donor_consistent": float((got == exp).mean()), "recipient_consistent": float((got == own).mean()), "chance_overlap": float((exp == own).mean()),
                     "bprefix_consistent": float((got == to_int(Ar) * bmix).mean()), "operands_consistent": float((got == to_int(Ad) * bmix).mean()), "n": N}}


@torch.no_grad()
def ext_4x8_low8(m, dev, amp, bs, n=1000, seed=14204):
    """§142 (note 142-1, descriptive): 4-digit a x 8-digit b, chains of 8 segments (positions 5-8 unseen: COLTOK[5..8] by segment()'s
    position-j = COLTOK[j] convention); exact match of the decoded value with S_j mod 10^8 at every j (j = 5..8 is the extrapolation)."""
    from boxes.multcol import gen_operands
    rng = np.random.default_rng(seed); A = gen_operands(rng, n, 4); B = gen_operands(rng, n, 8)
    dec = rollout(m, A, B, dev, amp, bs, L=8); S = sj_values(A, B) % 10 ** NDIG
    acc = {str(j + 1): float((dec[:, j] == S[:, j]).mean()) for j in range(8)}
    return {"acc_by_j": acc, "acc_5_8_mean": float(np.mean([acc[str(j)] for j in range(5, 9)])), "n": n, "metric": "S_j mod 1e8 exact (note 142-1 (a))"}


@torch.no_grad()
def contraction(m, A, B, dev, amp, T0=1, n=500, seed=12301):
    """fsq120's contraction formulas on L = 4 multiplier chains: perturb z_b after segment T0, follow K = 4 - T0 hand-offs (window K)."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(A))[:n]; A, B = A[idx], B[idx]; N = len(A); L = B.shape[1]; K = L - T0; W = K
    set_ops(m, A, B, dev)
    def step(prev, t, add=None):
        with amp_ctx(amp):
            h = m.segment(prev, torch.full((N,), t, device=dev)); qs, q, zb = m.quant(h, noise=add); e = m.embed_code(qs); e.qs_ = qs
        return zb.float(), e
    def dec(e):
        with amp_ctx(amp):
            return m.fam.decode(m.answer_greedy(e, NDIG, e.qs_).cpu().numpy())
    prev = m.start(N, dev); zs = []
    for t in range(T0 - 1):
        z, e = step(prev, t); zs.append(z); prev = e[:, None]
    pre = prev; zT0, eT0 = step(prev, T0 - 1); zs.append(zT0); clean = [zT0]; cdec = []; c = eT0[:, None]
    for k in range(K):
        z, e = step(c, T0 + k); zs.append(z); clean.append(z); cdec.append(dec(e)); c = e[:, None]
    std_d = torch.cat(zs).std(0); U = torch.tensor(rng.standard_normal((N, zT0.shape[-1])), dtype=torch.float32, device=dev); R = {}
    for eps in F.CONTRACT_EPS:
        zp, ep = step(pre, T0 - 1, add=eps * std_d * U); dist = [(zp - zT0).norm(dim=-1)]; agree = []; c = ep[:, None]
        for k in range(K):
            z, e = step(c, T0 + k); dist.append((z - clean[k + 1]).norm(dim=-1)); agree.append(float((dec(e) == cdec[k]).mean())); c = e[:, None]
        D = torch.stack(dist).cpu().numpy(); ratio = D[1:] / np.maximum(D[:-1], 1e-6); rho_k = np.median(ratio, axis=1)
        rk_nz = [float(np.median(ratio[k][D[k] > 0])) for k in range(W) if (D[k] > 0).any()]
        R[str(eps)] = {"rho_k": rho_k.round(4).tolist(), "rho_geo": float(np.exp(np.log(np.maximum(rho_k[:W], 1e-6)).mean())), "agree_k": [round(v, 4) for v in agree],
                       "dist_med_k": np.median(D, axis=1).round(4).tolist(), "exact_recovery_share": float((D[1:W + 1] == 0).all(0).mean()),
                       "rho_nonzero": float(np.exp(np.log(np.maximum(rk_nz, 1e-6)).mean())) if rk_nz else None}
    basin = max([e_ for e_ in F.CONTRACT_EPS if R[str(e_)]["agree_k"][W - 1] >= 0.99], default=0.0)
    return {"rho": R["0.1"]["rho_geo"], "rho_nonzero": R["0.1"]["rho_nonzero"], "basin": basin, "exact_recovery_share": R["0.1"]["exact_recovery_share"],
            "detail": {"T0": T0, "K": K, "window": W, "n": N, "std_d": std_d.cpu().numpy().round(4).tolist(), "eps": R}}


# ================================================================ THEORY §152 DELTA-READ (378fe8e6 + note 152-1): --carrier deltamem
EQTOK = bc.tok(" =")[0]


# ================================================================ THEORY §157 READ-PANEL (c13b78da): read-side changes on §152's DeltaMem
PER_P = (10.0, 100.0, 1000.0); NSC = 2                                      # PERIODIC: frozen periods; scalar value channels per additive head


class PeriodicMem(DM.DeltaMem):
    """§157 PERIODIC: DeltaMem + per additive head 2 scalar value channels (W_s: LN_6(x) -> 2 values, rounded with a straight-through
    estimator, written as phi(k) w_r^T into 2 extra state columns: state (B, 12, 64, 66)); the scalar read s = phi(q)^T S_s ->
    [s, cos(2 pi s / P), sin(2 pi s / P)], P in {10, 100, 1000} (84 features) -> a zero-init Linear(84, 768) added beside W_o under the same fade.
    fp32, autocast off (as the branch). round_err: list -> |w - round(w)| per write."""
    periodic = True

    def __init__(self, g, layer=6):
        super().__init__(g, layer); dev = g.w("wte.weight").device
        self.ws = nn.Linear(768, DM.H_ADD * NSC).to(dev)
        with torch.no_grad():
            self.ws.weight.normal_(0, 0.02); self.ws.bias.zero_()
        self.pm = nn.Linear(DM.H_ADD * NSC * (1 + 2 * len(PER_P)), 768).to(dev)
        with torch.no_grad():
            self.pm.weight.zero_(); self.pm.bias.zero_()
        self.round_err = None

    def zero_state(self, B, dev):
        return torch.zeros(B, DM.H_ADD + DM.H_DEL, DM.DH, DM.DH + NSC, device=dev)

    @staticmethod
    def feats(sv):
        f = [sv] + [torch.cos(2 * math.pi * sv / P_) for P_ in PER_P] + [torch.sin(2 * math.pi * sv / P_) for P_ in PER_P]
        return torch.stack(f, -1).flatten(-3)

    def _forward(self, x, S, rec=None):
        B, T, D = x.shape; HA, HD, DH = DM.H_ADD, DM.H_DEL, DM.DH; a = self.ln(x.float()); q, k, v = self.qkv(a).split(D, -1)
        q, k, v = (t.view(B, T, HA + HD, DH) for t in (q, k, v)); g_ = self.gate(a); al = torch.sigmoid(g_[..., :HD]); be = 2 * torch.sigmoid(g_[..., HD:])
        w = self.ws(a).view(B, T, HA, NSC); wr = w + (torch.round(w) - w).detach()
        if self.round_err is not None:
            self.round_err.append((w - torch.round(w)).abs().detach().flatten().cpu())
        if self.alpha_rec is not None:
            self.alpha_rec.append(al.mean(dim=(0, 1)).detach())
        Sa, Ss, Sd = S[:, :HA, :, :DH].float(), S[:, :HA, :, DH:].float(), S[:, HA:, :, :DH].float(); outs = []; pfs = []
        for t in range(T):
            pk, pq = Fn.elu(k[:, t, :HA]) + 1, Fn.elu(q[:, t, :HA]) + 1
            Sa = Sa + pk[..., :, None] * v[:, t, :HA][..., None, :]; Ss = Ss + pk[..., :, None] * wr[:, t][..., None, :]
            oa = torch.einsum("bhk,bhkv->bhv", pq, Sa); sv = torch.einsum("bhk,bhkc->bhc", pq, Ss)
            kd = Fn.normalize(k[:, t, HA:], dim=-1); qd = Fn.normalize(q[:, t, HA:], dim=-1); vd = v[:, t, HA:]; b_ = be[:, t][..., None, None]
            Skk = torch.einsum("bhvk,bhk->bhv", Sd, kd); Sd = al[:, t][..., None, None] * (Sd - b_ * Skk[..., :, None] * kd[..., None, :]) + b_ * vd[..., :, None] * kd[..., None, :]
            od = torch.einsum("bhvk,bhk->bhv", Sd, qd); oa = torch.exp(self.s_add) * oa; od = torch.exp(self.s_del) * od
            if self.norms is not None and t == T - 1:
                self.norms.append((od.flatten(1).norm(dim=1).detach(), oa.flatten(1).norm(dim=1).detach()))
            if self.ablate == "add":
                oa = torch.zeros_like(oa); sv = torch.zeros_like(sv)
            if self.ablate == "delta":
                od = torch.zeros_like(od)
            outs.append(torch.cat([oa, od], 1).reshape(B, D)); pfs.append(self.feats(sv).reshape(B, -1))
            if rec is not None and t == T - 1:
                rec.append(oa.reshape(B, -1).detach())
        o = torch.stack(outs, 1); pf = torch.stack(pfs, 1)
        Sn = torch.cat([torch.cat([Sa, Ss], -1), torch.cat([Sd, torch.zeros_like(Sd[..., :NSC])], -1)], 1)
        return x + (self.fade * (self.wo(o) + self.pm(pf))).to(x.dtype), Sn

    def read_fixed_p(self, x, S):
        """the reader pass's read on a fixed state (mem_read's role), with the periodic scalar read."""
        with torch.autocast("cuda", enabled=False):
            B, T, D = x.shape; HA, DH = DM.H_ADD, DM.DH; a = self.ln(x.float()); q = self.qkv(a)[..., :D].view(B, T, HA + DM.H_DEL, DH); pq = Fn.elu(q[:, :, :HA]) + 1
            oa = torch.exp(self.s_add) * torch.einsum("bthk,bhkv->bthv", pq, S[:, :HA, :, :DH].float()); sv = torch.einsum("bthk,bhkc->bthc", pq, S[:, :HA, :, DH:].float())
            od = torch.exp(self.s_del) * torch.einsum("bhvk,bthk->bthv", S[:, HA:, :, :DH].float(), Fn.normalize(q[:, :, HA:], dim=-1))
            if self.ablate == "add":
                oa = torch.zeros_like(oa); sv = torch.zeros_like(sv)
            if self.ablate == "delta":
                od = torch.zeros_like(od)
            return x + (self.fade * (self.wo(torch.cat([oa, od], 2).reshape(B, T, D)) + self.pm(self.feats(sv)))).to(x.dtype)


class MultiRead(nn.Module):
    """§157 MULTIREAD: R = 4 STATIC learned queries per head (12 heads; Perceiver-style, not input-dependent) with learned slot IDs read the pass-start state at the start of every pass
    (segments and the reader); their 4 read vectors ([g_add read_add || g_del read_del], W_o-projected, then a zero-init Z) + slot ID are
    prepended as 4 positions (no wpe) that every layer attends to; the original positions see them with attention bias log(fade) (fade
    0 -> 1 over 500 steps; at fade 0 the read tokens are omitted: bitwise GPT-2). Read tokens attend among themselves only."""
    R = 4

    def __init__(self, dev):
        super().__init__()
        self.Q = nn.Parameter(torch.randn(DM.H_ADD + DM.H_DEL, self.R, DM.DH, device=dev) * 0.02)
        self.slot = nn.Parameter(torch.randn(self.R, 768, device=dev) * 0.02); self.Z = nn.Linear(768, 768).to(dev)
        with torch.no_grad():
            self.Z.weight.zero_(); self.Z.bias.zero_()
        self.register_buffer("fade", torch.zeros((), device=dev))

    def reads(self, mem, S):
        with torch.autocast("cuda", enabled=False):
            HA, DH = DM.H_ADD, DM.DH
            ra = torch.exp(mem.s_add) * torch.einsum("hrk,bhkv->bhrv", Fn.elu(self.Q[:HA]) + 1, S[:, :HA, :, :DH].float())
            rd = torch.exp(mem.s_del) * torch.einsum("bhvk,hrk->bhrv", S[:, HA:, :, :DH].float(), Fn.normalize(self.Q[HA:], dim=-1))
            return torch.cat([ra, rd], 1).permute(0, 2, 1, 3).reshape(S.shape[0], self.R, -1)   # (B, R, 768): heads concatenated as W_o's input

    def tokens(self, mem, S):
        with torch.autocast("cuda", enabled=False):
            return self.slot[None] + self.Z(mem.wo(self.reads(mem, S)))



class CCQMem(DM.DeltaMem):
    """§157 CCQ (note 157-1; 2606.01294v2 Eq. 6 / 9 / 10): READ o_t = S_t q_clean, q_clean = q_t - lambda_t Sigma_t q_t, lambda_t =
    sigmoid(W_l q_t + b_l) from the PRE-SPLIT query (768 -> one scalar per head; W_l = 0, b_l = -4.6 at init); Sigma_t = C_t - mu_t mu_t^T per head
    (fp32, undecayed, count-normalised, centred) over the chain's writes so far (the same writes as S). q and k l2-normalised per head; on the
    additive heads phi(k) and phi(q) are l2-normalised in BOTH the write and the read (disclosed adaptation). Every read position (segments and
    the reader pass), all 12 heads. Packed state (B, 12, 64, 130): [..., :64] S, [..., 64:128] C (k x k), [..., 128] mu, [:, 0, 0, 129] the
    write count. ccq_off: lambda forced to 0 at read (the read-time ablation). lam_rec: list -> mean lambda per head per pass."""
    ccq = True

    def __init__(self, g, layer=6):
        super().__init__(g, layer); dev = g.w("wte.weight").device
        self.wl = nn.Linear(768, DM.H_ADD + DM.H_DEL).to(dev)
        with torch.no_grad():
            self.wl.weight.zero_(); self.wl.bias.fill_(-4.6)
        self.ccq_off = False; self.lam_rec = None

    def zero_state(self, B, dev):
        return torch.zeros(B, DM.H_ADD + DM.H_DEL, DM.DH, 2 * DM.DH + 2, device=dev)

    def _feat(self, qk):
        """l2-normalised features: phi = elu + 1 on the additive heads, raw on the delta heads, then l2 per head."""
        HA = DM.H_ADD
        return torch.cat([Fn.normalize(Fn.elu(qk[..., :HA, :]) + 1, dim=-1), Fn.normalize(qk[..., HA:, :], dim=-1)], -2)

    def _lam(self, qfull):
        lam = torch.sigmoid(self.wl(qfull))
        if self.lam_rec is not None:
            self.lam_rec.append(lam.flatten(0, -2).mean(0).detach())
        return torch.zeros_like(lam) if self.ccq_off else lam

    def _read(self, Qn, lam, Sa, Sd, C, mu):
        """Qn (B, 12, 64) normalised queries; lam (B, 12); -> oa (B, 6, 64), od (B, 6, 64) with q_clean."""
        Sig = C - mu[..., :, None] * mu[..., None, :]; qc = Qn - lam[..., None] * torch.einsum("bhij,bhj->bhi", Sig, Qn); HA = DM.H_ADD
        return torch.einsum("bhk,bhkv->bhv", qc[:, :HA], Sa), torch.einsum("bhvk,bhk->bhv", Sd, qc[:, HA:])

    def _forward(self, x, S, rec=None):
        B, T, D = x.shape; HA, HD, DH = DM.H_ADD, DM.H_DEL, DM.DH; a = self.ln(x.float()); qf, k, v = self.qkv(a).split(D, -1); lam = self._lam(qf)
        q, k, v = (t.view(B, T, HA + HD, DH) for t in (qf, k, v)); Qn, Kn = self._feat(q), self._feat(k)
        g_ = self.gate(a); al = torch.sigmoid(g_[..., :HD]); be = 2 * torch.sigmoid(g_[..., HD:])
        if self.alpha_rec is not None:
            self.alpha_rec.append(al.mean(dim=(0, 1)).detach())
        Sa, Sd = S[:, :HA, :, :DH].float(), S[:, HA:, :, :DH].float(); C = S[:, :, :, DH:2 * DH].float(); mu = S[:, :, :, 2 * DH].float(); cnt = S[:, 0, 0, 2 * DH + 1].float(); outs = []
        for t in range(T):
            pk = Kn[:, t, :HA]; Sa = Sa + pk[..., :, None] * v[:, t, :HA][..., None, :]
            kd = Kn[:, t, HA:]; vd = v[:, t, HA:]; b_ = be[:, t][..., None, None]; Skk = torch.einsum("bhvk,bhk->bhv", Sd, kd)
            Sd = al[:, t][..., None, None] * (Sd - b_ * Skk[..., :, None] * kd[..., None, :]) + b_ * vd[..., :, None] * kd[..., None, :]
            cnt = cnt + 1; c_ = cnt[:, None, None]; Kt = Kn[:, t]; mu = mu + (Kt - mu) / c_                 # count-normalised, undecayed, centred
            C = C + (Kt[..., :, None] * Kt[..., None, :] - C) / c_[..., None]
            oa, od = self._read(Qn[:, t], lam[:, t], Sa, Sd, C, mu); oa = torch.exp(self.s_add) * oa; od = torch.exp(self.s_del) * od
            if self.norms is not None and t == T - 1:
                self.norms.append((od.flatten(1).norm(dim=1).detach(), oa.flatten(1).norm(dim=1).detach()))
            if self.ablate == "add":
                oa = torch.zeros_like(oa)
            if self.ablate == "delta":
                od = torch.zeros_like(od)
            outs.append(torch.cat([oa, od], 1).reshape(B, D))
            if rec is not None and t == T - 1:
                rec.append(oa.reshape(B, -1).detach())
        o = torch.stack(outs, 1); cn = torch.zeros_like(mu)[..., None]; cn[:, 0, 0, 0] = cnt
        Sn = torch.cat([torch.cat([Sa, Sd], 1), C, mu[..., None], cn], -1)
        return x + (self.fade * self.wo(o)).to(x.dtype), Sn

    def read_fixed_c(self, x, S):
        """the reader pass's read on a fixed (pass-start) state and its key statistics."""
        with torch.autocast("cuda", enabled=False):
            B, T, D = x.shape; HA, DH = DM.H_ADD, DM.DH; a = self.ln(x.float()); qf = self.qkv(a)[..., :D]; lam = self._lam(qf)
            Qn = self._feat(qf.view(B, T, HA + DM.H_DEL, DH)); Sa, Sd = S[:, :HA, :, :DH].float(), S[:, HA:, :, :DH].float(); C = S[:, :, :, DH:2 * DH].float(); mu = S[:, :, :, 2 * DH].float()
            Sig = C - mu[..., :, None] * mu[..., None, :]; qc = Qn - lam[..., None] * torch.einsum("bhij,bthj->bthi", Sig, Qn)
            oa = torch.exp(self.s_add) * torch.einsum("bthk,bhkv->bthv", qc[:, :, :HA], Sa); od = torch.exp(self.s_del) * torch.einsum("bhvk,bthk->bthv", Sd, qc[:, :, HA:])
            if self.ablate == "add":
                oa = torch.zeros_like(oa)
            if self.ablate == "delta":
                od = torch.zeros_like(od)
            return x + (self.fade * self.wo(torch.cat([oa, od], 2).reshape(B, T, D))).to(x.dtype)


@torch.no_grad()
def periodic_extras(m, teA, teB, dev, amp, n=256):
    """§157 PERIODIC read-side diagnostics (Audit): at the reader position predicting digit k (teacher-forced, after segment 4):
    read_round_err = |sv - round(sv)| of the scalar read sv = phi(q)^T S_s (median and share < 0.25, pooled over heads and channels) per column;
    offcol_weight = the share of sum_j phi(q) . phi(k_j) (additive heads; phi = elu + 1 > 0, so every write enters every read) on write tokens
    outside the digit's column (a_i in segment s belongs to column i + s; EOT / b_j / step tokens count as outside)."""
    n = min(n, len(teA)); A, B = teA[:n], teB[:n]; HA, DH = DM.H_ADD, DM.DH; xs = []
    hk = m.mem.register_forward_pre_hook(lambda mod_, inp_: xs.append(inp_[0].detach().float()))
    try:
        _, kept, _, _ = rollout_drow(m, A, B, dev, amp, n, keep=(4,), decode=False)
    finally:
        hk.remove()
    S4 = torch.tensor(kept[4], device=dev); S4v = sj_values(A, B)[:, -1]; ans = torch.tensor(np.array(DTOK)[digits8(S4v)], device=dev); cp = []
    with amp_ctx(amp):
        fwd_mem(m.g, m.g.embed(m.reader_ids(n, dev, ans)), m.mem, S4, read_only=True, cap=cp)
    a = m.mem.ln(cp[0].float()); q = m.mem.qkv(a)[..., :768].view(n, -1, HA + DM.H_DEL, DH)[:, :, :HA]; pq = Fn.elu(q) + 1
    sv = torch.einsum("bthk,bhkc->bthc", pq, S4[:, :HA, :, DH:].float())                                 # (n, T, 6, 2)
    K = torch.cat([Fn.elu(m.mem.qkv(m.mem.ln(x_))[..., 768:1536].view(n, -1, HA + DM.H_DEL, DH)[:, :, :HA]) + 1 for x_ in xs[:4]], 1)   # (n, 28, 6, 64)
    col = np.full(K.shape[1], -1)
    for s_ in range(4):
        for i in range(4):
            col[s_ * MultRowDelta.SEG_T + 3 + i] = i + s_
    W = torch.einsum("bthd,bjhd->bthj", pq, K).cpu().numpy(); pos = (list(range(2, 2 + 2 * NDIG, 2)) if getattr(m, "sep", None) is not None else list(range(1, 1 + NDIG)))
    rre, off = {}, {}
    for k in range(NDIG):
        e_ = (sv[:, pos[k]] - torch.round(sv[:, pos[k]])).abs().flatten().cpu().numpy(); rre[str(k + 1)] = {"median": float(np.median(e_)), "share_lt_0.25": float((e_ < 0.25).mean())}
        w_ = W[:, pos[k]]; oc = col != k; off[str(k + 1)] = float((w_[..., oc].sum(-1) / np.maximum(w_.sum(-1), 1e-12)).mean())
    off["mean"] = float(np.mean([v for v in off.values()]))
    return {"read_round_err": rre, "offcol_weight": off,
            "offcol_def": "additive heads, phi(q) . phi(k_j) share on write tokens outside the digit's column (reader position, after segment 4, teacher-forced)"}


@torch.no_grad()
def ccq_extras(m, teA, teB, dev, amp, bs, n=256):
    """§157 CCQ descriptive: ccq_lambda_by_head (mean lambda over every read of a probe batch: segments + the final reader pass), offcol_weight
    (additive heads, exact linear-attention weights: at the reader position predicting digit k (teacher-forced, after segment 4), the share of
    |q_clean . phi_n(k_j)| on write tokens j outside column k-1 (a_i in segment s belongs to column i + s; EOT / b_j / step tokens count as
    outside)), and the read-time ablation (lambda forced to 0 at read: test_acc, acc_by_length)."""
    n = min(n, len(teA)); A, B = teA[:n], teB[:n]; HA, DH = DM.H_ADD, DM.DH; xs = []
    hk = m.mem.register_forward_pre_hook(lambda mod_, inp_: xs.append(inp_[0].detach().float())); m.mem.lam_rec = []
    try:
        _, kept, _, _ = rollout_drow(m, A, B, dev, amp, n, keep=(4,), decode=False)
    finally:
        hk.remove()
    S4 = torch.tensor(kept[4], device=dev); S4v = sj_values(A, B)[:, -1]; ans = torch.tensor(np.array(DTOK)[digits8(S4v)], device=dev)
    cp = []
    with amp_ctx(amp):
        fwd_mem(m.g, m.g.embed(m.reader_ids(n, dev, ans)), m.mem, S4, read_only=True, cap=cp)
    lam_all = torch.stack(m.mem.lam_rec).mean(0); m.mem.lam_rec = None
    a = m.mem.ln(cp[0].float()); qf = m.mem.qkv(a)[..., :768]; lam = torch.sigmoid(m.mem.wl(qf)); Qn = m.mem._feat(qf.view(n, -1, HA + DM.H_DEL, DH))
    C = S4[:, :, :, DH:2 * DH].float(); mu = S4[:, :, :, 2 * DH].float(); Sig = C - mu[..., :, None] * mu[..., None, :]
    qc = (Qn - lam[..., None] * torch.einsum("bhij,bthj->bthi", Sig, Qn))[:, :, :HA]                    # (n, T, 6, 64)
    K = torch.cat([m.mem._feat(m.mem.qkv(m.mem.ln(x_))[..., 768:1536].view(n, -1, HA + DM.H_DEL, DH))[:, :, :HA] for x_ in xs[:4]], 1)   # (n, 28, 6, 64)
    col = np.full(K.shape[1], -1)
    for s_ in range(4):
        for i in range(4):
            col[s_ * MultRowDelta.SEG_T + 3 + i] = i + s_
    W = torch.einsum("bthd,bjhd->bthj", qc, K).abs().cpu().numpy(); pos = (list(range(2, 2 + 2 * NDIG, 2)) if getattr(m, "sep", None) is not None else list(range(1, 1 + NDIG)))
    off = {}
    for k in range(NDIG):
        w_ = W[:, pos[k]]; oc = col != k; off[str(k + 1)] = float((w_[..., oc].sum(-1) / np.maximum(w_.sum(-1), 1e-12)).mean())
    off["mean"] = float(np.mean([v for v in off.values()]))
    m.mem.ccq_off = True; dk = rollout_drow(m, teA, teB, dev, amp, bs)[0]; m.mem.ccq_off = False; S_ = sj_values(teA, teB)
    return {"ccq_lambda_by_head": [round(float(v), 6) for v in lam_all], "offcol_weight": off,
            "read_time_lambda0": {"test_acc": float((dk[:, -1] == S_[:, -1]).mean()), "acc_by_length": {str(j + 1): float((dk[:, j] == S_[:, j]).mean()) for j in range(4)}},
            "offcol_def": "additive heads, |q_clean . phi_n(k_j)| share on write tokens outside the digit's column (reader position, after segment 4, teacher-forced)"}


def _mblock(g, i, h, hd, mask):
    """a GPT-2 block with an explicit additive attention mask (§157 MULTIREAD)."""
    B, T, D = h.shape; pre = f"h.{i}."
    a = Fn.layer_norm(h, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
    q, k, v = (a @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")).split(D, -1)
    q, k, v = (t.view(B, T, g.H, hd).transpose(1, 2) for t in (q, k, v))
    o = Fn.scaled_dot_product_attention(q, k, v, attn_mask=mask.to(q.dtype))
    h = h + o.transpose(1, 2).reshape(B, T, D) @ g.w(pre + "attn.c_proj.weight") + g.w(pre + "attn.c_proj.bias")
    a = Fn.layer_norm(h, (D,), g.w(pre + "ln_2.weight"), g.w(pre + "ln_2.bias"))
    mm = Fn.gelu(a @ g.w(pre + "mlp.c_fc.weight") + g.w(pre + "mlp.c_fc.bias"), approximate="tanh")
    return h + mm @ g.w(pre + "mlp.c_proj.weight") + g.w(pre + "mlp.c_proj.bias")


def _fwd_mem_mr(g, x, mem, S, read_only=False, cap=None, mrcap=None):
    """fwd_mem with MULTIREAD's 4 prepended read tokens (computed from the pass-start S). mrcap: dict <- 'tok0' (B, R, 768) the read tokens'
    layer-0 vectors, 'tokL7' their residual after block 6."""
    B, T, D = x.shape; hd = D // g.H; R = mem.mr.R; rt = mem.mr.tokens(mem, S).to(x.dtype)
    if mrcap is not None:
        mrcap["tok0"] = rt.detach()
    h = torch.cat([rt, x + g.w("wpe.weight")[:T][None]], 1); L = R + T
    mask = torch.full((L, L), float("-inf"), device=x.device); mask[:R, :R] = 0.0; mask[R:, :R] = torch.log(mem.mr.fade.float())
    mask[R:, R:] = torch.full((T, T), float("-inf"), device=x.device).triu(1)
    for i in range(g.L):
        if i == mem.layer:
            ho = h[:, R:]
            if cap is not None:
                cap.append(ho.detach())
            if read_only:
                ho = mem_read(mem, ho, S)
            else:
                ho, S = mem(ho, S)
            h = torch.cat([h[:, :R], ho], 1)
        h = _mblock(g, i, h, hd, mask)
        if mrcap is not None and i == 6:
            mrcap["tokL7"] = h[:, :R].detach()
    return Fn.layer_norm(h[:, R:], (D,), g.w("ln_f.weight"), g.w("ln_f.bias")), S


def mem_read(mem, x, S):
    """§152 reader pass: DeltaMem's read on a FIXED state - no write (k / v / alpha / beta unused), S untouched; the same read gains exp(s),
    ablation, fade and W_o as DeltaMem.forward. fp32 with autocast off (as the branch)."""
    if getattr(mem, "periodic", False):                                     # §157 PERIODIC: the scalar channels too
        return mem.read_fixed_p(x, S)
    if getattr(mem, "ccq", False):                                          # §157 CCQ: the contracted query
        return mem.read_fixed_c(x, S)
    with torch.autocast("cuda", enabled=False):
        B, T, D = x.shape; a = mem.ln(x.float()); q = mem.qkv(a)[..., :D].view(B, T, DM.H_ADD + DM.H_DEL, DM.DH)
        Sa, Sd = S[:, :DM.H_ADD].float(), S[:, DM.H_ADD:].float()
        oa = torch.exp(mem.s_add) * torch.einsum("bthk,bhkv->bthv", Fn.elu(q[:, :, :DM.H_ADD]) + 1, Sa)
        od = torch.exp(mem.s_del) * torch.einsum("bhvk,bthk->bthv", Sd, Fn.normalize(q[:, :, DM.H_ADD:], dim=-1))
        if mem.ablate == "add":
            oa = torch.zeros_like(oa)
        if mem.ablate == "delta":
            od = torch.zeros_like(od)
        return x + (mem.fade * mem.wo(torch.cat([oa, od], 2).reshape(B, T, D))).to(x.dtype)


def fwd_mem(g, x, mem, S, read_only=False, cap=None):
    """boxes.deltamem.gpt_forward_mem, op for op (checked bitwise in scratch/gate152.py), plus: read_only (the reader pass: mem_read, S returned
    as given) and cap (a list: the residual entering the branch, i.e. after mem.layer blocks, (B, T, 768))."""
    if getattr(mem, "mr", None) is not None and float(mem.mr.fade) > 0:     # §157 MULTIREAD (fade 0: the read tokens are omitted)
        return _fwd_mem_mr(g, x, mem, S, read_only, cap)
    if getattr(mem, "write_mode", None) is not None and not read_only:       # §160 R-ROWS: DeltaMem-R (reads the pass-start S; writes from u)
        cd_ = {} if cap is not None else None; out_, S_ = DM.gpt_forward_mem_r(g, x, mem, S, cap=cd_)
        if cap is not None:
            cap.append(cd_["h6"])
        return out_, S_
    B, T, D = x.shape; hd = D // g.H; h = x + g.w("wpe.weight")[:T][None]
    for i in range(g.L):
        if i == mem.layer:
            if cap is not None:
                cap.append(h.detach())
            if read_only:
                h = mem_read(mem, h, S)
            else:
                h, S = mem(h, S)
        pre = f"h.{i}."
        a = Fn.layer_norm(h, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
        q, k, v = (a @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")).split(D, -1)
        q, k, v = (t.view(B, T, g.H, hd).transpose(1, 2) for t in (q, k, v))
        o = Fn.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + o.transpose(1, 2).reshape(B, T, D) @ g.w(pre + "attn.c_proj.weight") + g.w(pre + "attn.c_proj.bias")
        a = Fn.layer_norm(h, (D,), g.w(pre + "ln_2.weight"), g.w(pre + "ln_2.bias"))
        mm = Fn.gelu(a @ g.w(pre + "mlp.c_fc.weight") + g.w(pre + "mlp.c_fc.bias"), approximate="tanh")
        h = h + mm @ g.w(pre + "mlp.c_proj.weight") + g.w(pre + "mlp.c_proj.bias")
    return Fn.layer_norm(h, (D,), g.w("ln_f.weight"), g.w("ln_f.bias")), S


class MultRowDelta(MultRow):
    """§152 DELTA-READ: no carried code token. Segment j (note 152-1 (2), seg_order b_first) = [EOT, b_j, COLTOK[j], a's 4 digits LSB first], a
    fresh GPT-2 pass (attention inside the segment), the DeltaMem state (B, 12, 64, 64) the only cross-segment channel (writes at block 6).
    Reader / decoder pass = [EOT, " ="] + 8 answer digits LSB first (teacher-forced in training, greedy at test) through GPT-2's own LM head
    (full-vocabulary CE); the memory is READ-ONLY there (mem_read: no writes; the state handed on is untouched)."""
    SEG_T = 7

    def __init__(self, g, fam, layer=6):
        super().__init__(g, fam); self.mem = DM.DeltaMem(g, layer); self.pass_input = None; self.think_tok = None; self.Lp = None   # §153

    def start(self, B, dev, states=None):
        return self.mem.zero_state(B, dev)

    def seg_ids(self, t, B, dev):
        if self.pass_input is None:                                         # §152: [EOT, b_j, j, a]
            return torch.cat([torch.full((B, 1), EOT, device=dev), self.btok[:, t:t + 1], torch.full((B, 1), COLTOK[t + 1], device=dev), self.actx], 1)
        full = lambda v: torch.full((B, 1), v, device=dev)  # noqa: E731
        prob = [full(EOT), self.actx, full(XTOK), self.btok[:, :(self.Lp or self.btok.shape[1])]]   # §153: [EOT, a, " x", b's first L digits]
        if self.pass_input == "visible":                                    # the whole problem every pass + step token j
            return torch.cat(prob + [full(COLTOK[t + 1])], 1)
        if t < 0:                                                           # pure / repeat pass 0: the problem (writes only, no read-out)
            return torch.cat(prob, 1)
        return torch.cat([full(EOT), full(self.think_tok if self.pass_input == "repeat" else COLTOK[t + 1]), torch.full((B, 4), REGTOK, device=dev)], 1)

    def write(self, S, t, cap=None):
        _, S = fwd_mem(self.g, self.g.embed(self.seg_ids(t, S.shape[0], S.device)), self.mem, S, cap=cap); return S

    def reader_ids(self, B, dev, ans=None):
        """The reader pass's input: [EOT, " ="] (+ the teacher-forced answer); §156 interleave: [EOT, " =", |, d1, |, .., d7, |] (separators forced)."""
        ids = torch.tensor([[EOT, EQTOK]], device=dev).expand(B, 2)
        if getattr(self, "sep", None) is None:
            return ids if ans is None else torch.cat([ids, ans[:, :NDIG - 1]], 1)
        sep = torch.full((B, 1), self.sep, device=dev, dtype=ids.dtype); parts = [ids, sep]
        if ans is not None:
            for k in range(NDIG - 1):
                parts += [ans[:, k:k + 1], sep]
        return torch.cat(parts, 1)

    def read_logits(self, S, ans=None):
        """[EOT, " ="] (+ the first 7 answer tokens, teacher-forced) -> full-vocabulary logits at positions 1.. (B, T - 1, V); §156 interleave:
        the logits at the 8 separators (each digit predicted at the '|' before it; no loss on separators)."""
        B = S.shape[0]; ids = self.reader_ids(B, S.device, ans)
        out, _ = fwd_mem(self.g, self.g.embed(ids), self.mem, S, read_only=True)
        if getattr(self, "sep", None) is not None:
            return self.g.logits(out[:, 2::2]).float()
        return self.g.logits(out[:, 1:]).float()

    def read_ce(self, S, ans):
        lg = self.read_logits(S, ans); return Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), ans.reshape(-1))

    @torch.no_grad()
    def read_greedy(self, S):
        B = S.shape[0]; ids = self.reader_ids(B, S.device); outs = []; sp = getattr(self, "sep", None)
        for k in range(NDIG):
            out, _ = fwd_mem(self.g, self.g.embed(ids), self.mem, S, read_only=True); nt = self.g.logits(out[:, -1]).float().argmax(-1)
            outs.append(nt); ids = torch.cat([ids, nt[:, None]] + ([torch.full((B, 1), sp, device=S.device, dtype=ids.dtype)] if sp is not None and k < NDIG - 1 else []), 1)
        return torch.stack(outs, 1)

    def chain_read(self, start_emb, items, labels, rtok, rmask):
        """CE_final (gold S_L, after the last segment) and, with rmask, the reader CE_j at every hand-off j < L (teacher S_j, rows of rmask[:, j-1]);
        the reader passes leave the state untouched. Returns (CE_final, mean CE_j or None, number of reader hand-offs)."""
        B, L = items.shape; S = start_emb; rd = []; self.Lp = L
        if self.pass_input in ("pure", "repeat"):                           # §153 pass 0
            S = self.write(S, -1)
        for t in range(L):
            S = self.write(S, int(items[0, t]))
            if t < L - 1 and rmask is not None and bool(rmask[:, t].any()):
                rows = torch.nonzero(rmask[:, t]).squeeze(1); rd.append(self.read_ce(S[rows], rtok[rows, t]))
        fin = self.read_ce(S, torch.tensor(self.fam.answer_tokens(labels[:, L - 1]), device=items.device))
        return fin, (torch.stack(rd).mean() if rd else None), len(rd)


@torch.no_grad()
def rollout_drow(m, A, B, dev, amp, bs, L=None, override=None, keep=(), norms=False, caps=False, decode=True, skip=(), Lp=None):
    """§152 chains over L segments; after each the reader greedily decodes S_j (read-only). override (p, states): the whole state after segment p
    replaced (1-based). keep: segments whose states are returned (N, 12, 64, 64) fp32; norms: (N, L, 2) Frobenius norms (additive, delta);
    caps: the residual entering the branch per segment (N, L, 7, 768). §153: skip (passes j not run: the memory passes through), Lp (b digits in the
    problem; default min(L, b's length)); pure / repeat run pass 0 first. Returns dec (N, L), kept, norms, caps."""
    N = len(A); L = L or B.shape[1]; dec = np.full((N, L), -1, np.int64); kept = {p_: None for p_ in keep}   # §157: allocated from the state's own shape
    nrm = np.zeros((N, L, 2), np.float32) if norms else None; C = np.zeros((N, L, MultRowDelta.SEG_T, 768), np.float32) if caps else None
    for i in range(0, N, bs):
        sl = slice(i, i + bs); Bn = len(A[sl]); set_ops(m, A[sl], B[sl], dev); S = m.start(Bn, dev); m.Lp = Lp or min(L, B.shape[1])
        if m.pass_input in ("pure", "repeat"):
            with amp_ctx(amp):
                S = m.write(S, -1)
        for t in range(L):
            cp = [] if caps else None
            with amp_ctx(amp):
                if t + 1 not in skip:
                    S = m.write(S, t, cap=cp)
                if override is not None and override[0] == t + 1:
                    S = torch.tensor(override[1][sl], device=dev, dtype=S.dtype)
                if decode:
                    dec[sl, t] = m.fam.decode(m.read_greedy(S).cpu().numpy())
            if t + 1 in kept:
                if kept[t + 1] is None:
                    kept[t + 1] = np.zeros((N,) + tuple(S.shape[1:]), np.float32)
                kept[t + 1][sl] = S.float().cpu().numpy()
            if norms:
                nrm[sl, t, 0] = S[:, :DM.H_ADD].float().flatten(1).norm(dim=1).cpu().numpy(); nrm[sl, t, 1] = S[:, DM.H_ADD:].float().flatten(1).norm(dim=1).cpu().numpy()
            if caps:
                C[sl, t] = cp[0].float().cpu().numpy()
    return dec, kept, nrm, C


def donor_drow(m, A, B, dev, amp, bs, rng, n=1000):
    """§152 SERIAL (row level): the donor's whole memory state swapped in after segment 2; donors drawn so that a_donor != a_rec and the b-prefix
    (b_1, b_2) differs. donor_consistent = donor S_2 + a_rec x (b_rec,3 b_rec,4) x 100; bprefix_consistent = a_rec x (b_d1 b_d2 b_r3 b_r4);
    operands_consistent = a_donor x (b_d1 b_d2 b_r3 b_r4); recipient_consistent = the recipient's own product; chance_overlap = donor-expected
    = own product."""
    N = min(n, len(A)); ir = rng.permutation(len(A))[:N]; idn = rng.permutation(len(A))[:N]; ai = to_int(A); bp = to_int(B[:, :2])
    for _ in range(100):
        bad = np.nonzero((ai[idn] == ai[ir]) | (bp[idn] == bp[ir]))[0]
        if not len(bad):
            break
        idn[bad] = rng.integers(0, len(A), len(bad))
    Ar, Br, Ad, Bd = A[ir], B[ir], A[idn], B[idn]
    _, kept, _, _ = rollout_drow(m, Ad, Bd, dev, amp, bs, L=2, keep=(2,), decode=False, Lp=Bd.shape[1]); got = rollout_drow(m, Ar, Br, dev, amp, bs, override=(2, kept[2]))[0][:, -1]
    exp = sj_values(Ad, Bd)[:, 1] + to_int(Ar) * to_int(Br[:, 2:4]) * 100; own = sj_values(Ar, Br)[:, -1]; bmix = to_int(np.concatenate([Bd[:, :2], Br[:, 2:4]], 1))
    return {"pos2": {"donor_consistent": float((got == exp).mean()), "recipient_consistent": float((got == own).mean()), "chance_overlap": float((exp == own).mean()),
                     "bprefix_consistent": float((got == to_int(Ar) * bmix).mean()), "operands_consistent": float((got == to_int(Ad) * bmix).mean()), "n": N,
                     "donor_rule": "a_donor != a_rec and (b_1, b_2) differ; whole state after segment 2"}}


def opt_drow(m, lr):
    """§147's grouping: GPT-2 (+ the unused FSQ120 heads) in one AdamW group, DeltaMem in its own (300-step warm-up from step 0)."""
    mp = {id(p_) for p_ in m.mem.parameters()}
    return torch.optim.AdamW([{"params": [p_ for p_ in m.parameters() if id(p_) not in mp], "mult": 1.0},
                              {"params": list(m.mem.parameters()), "mult": 1.0, "warm_from": 0, "name": "deltamem"}], lr=lr, weight_decay=0.01)


@torch.no_grad()
def read_ratio_drow(m, A, B, dev, amp, n=256):
    """note 147-2 on §152 segments: per hand-off the median of ||g_del read_del|| / ||g_add read_add|| at each segment's last token (probe batch)."""
    was = m.training; m.eval(); m.mem.norms = []; rollout_drow(m, A[:n], B[:n], dev, amp, n, decode=False); nr = m.mem.norms; m.mem.norms = None; m.train(was)
    r = [(d_ / a_.clamp_min(1e-12)).float().cpu().numpy() for d_, a_ in nr]
    return [float(np.median(x)) for x in r], float(np.median(np.concatenate(r)))


@torch.no_grad()
def alpha_drow(m, A, B, dev, amp, n=256):
    """note 147-3 record: per delta head the mean alpha over a fixed probe batch (every segment token), and sigmoid(gate bias)."""
    was = m.training; m.eval(); m.mem.alpha_rec = []; rollout_drow(m, A[:n], B[:n], dev, amp, n, decode=False); r = torch.stack(m.mem.alpha_rec).mean(0); m.mem.alpha_rec = None; m.train(was)
    return {"mean_alpha": [round(float(v), 6) for v in r], "sigmoid_bias": [round(float(v), 6) for v in torch.sigmoid(m.mem.gate.bias[:DM.H_DEL].detach())]}


def kridge_r2(Xf, yf, Xs, ys, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    """Ridge on standardised features in dual (kernel) form, for d >> n (flattened memory states): K = Z Z^T / d; lambda (x n) by 3-fold CV on
    the fit half; held-out R^2 on the score half. Equals primal ridge with penalty lambda n d."""
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6; Zf = ((Xf - mu) / sd).astype(np.float32); Zs = ((Xs - mu) / sd).astype(np.float32); d = Zf.shape[1]
    def fit_pred(Za, ya, Zb, lam):
        K = (Za @ Za.T).astype(np.float64) / d; ym = ya.mean(); al = np.linalg.solve(K + lam * len(Za) * np.eye(len(Za)), ya - ym)
        return (Zb @ Za.T).astype(np.float64) / d @ al + ym
    def r2(y_, p_):
        return float(1 - ((y_ - p_) ** 2).sum() / max(((y_ - y_.mean()) ** 2).sum(), 1e-12))
    fold = np.arange(len(Zf)) % 3
    best = max(lams, key=lambda l_: np.mean([r2(yf[fold == k], fit_pred(Zf[fold != k], yf[fold != k], Zf[fold == k], l_)) for k in range(3)]))
    return r2(ys, fit_pred(Zf, yf, Zs, best)), best


def compute_drow(m, L=4):
    """§98 convention (boxes.core.Meter: KV cache, each position processed once; full-vocabulary LM head rows), per item: L segments of 7
    tokens (full 12-block passes, although blocks 6-11 of a segment are never read), then the reader: [EOT, " ="] (1 logit row) + 7 more
    single-token passes (8 digits). DeltaMem branch: 2 x branch params per processed token, plus per head and token 4 DH^2 (additive write +
    read) / 10 DH^2 (delta: S k, the rank-1 update, decay, read) in segments and 2 DH^2 (read only) in reader tokens."""
    mt = bc.Meter(m.g); rows = 0
    pi = getattr(m, "pass_input", None)                                     # §153: pass lengths by input
    segs = [MultRowDelta.SEG_T] * L if pi is None else ([7 + L] * L if pi == "visible" else [6 + L] + [6] * L)
    RT = MultiRead.R if getattr(m.mem, "mr", None) is not None else 0      # §157 MULTIREAD: the read tokens are processed in every pass
    for T_ in segs:
        mt.step(T_ + RT, 0, 0)
    sp_ = getattr(m, "sep", None) is not None; st_ = 2 if sp_ else 1; c0 = 3 if sp_ else 2   # §156 interleave: forced separators counted
    mt.step(c0 + RT, 0, 1); rows += 1
    for k in range(NDIG - 1):
        mt.step(st_, c0 + st_ * k, 1); rows += 1
    nb = sum(p_.numel() for n_, p_ in m.mem.named_parameters() if not n_.startswith("s_")); seg_tok = sum(segs); rd_tok = (2 + 2 * NDIG - 1) if getattr(m, "sep", None) is not None else (2 + NDIG - 1); D2 = DM.DH * DM.DH
    fb = 2 * nb * (seg_tok + rd_tok) + (DM.H_ADD * 4 + DM.H_DEL * 10) * D2 * seg_tok + (DM.H_ADD + DM.H_DEL) * 2 * D2 * rd_tok
    d = mt.as_dict()
    return {"flops": d["flops"] + fb, "flops_trunk": d["flops"], "flops_branch": float(fb), "sequential_passes": d["seq_passes"], "lm_head_rows": rows,
            "tokens_processed": d["tokens_processed"], "convention": "§98 (boxes.core.Meter: KV cache, full vocabulary); segments counted as full 12-block passes"}


def kridge_fit_pred(Xf, yf, Xs, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    """kridge_r2's estimator returning (predictions on Xs, lambda) (note 152-5; as theory/probe_s152_colsum_exact.py)."""
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6; Zf = ((Xf - mu) / sd).astype(np.float32); Zs = ((Xs - mu) / sd).astype(np.float32); d = Zf.shape[1]
    def fit_pred(Za, ya, Zb, lam):
        K = (Za @ Za.T).astype(np.float64) / d; ym = ya.mean(); al = np.linalg.solve(K + lam * len(Za) * np.eye(len(Za)), ya - ym)
        return (Zb @ Za.T).astype(np.float64) / d @ al + ym
    def r2(y_, p_):
        return float(1 - ((y_ - p_) ** 2).sum() / max(((y_ - y_.mean()) ** 2).sum(), 1e-12))
    fold = np.arange(len(Zf)) % 3
    best = max(lams, key=lambda l_: np.mean([r2(yf[fold == k], fit_pred(Zf[fold != k], yf[fold != k], Zf[fold == k], l_)) for k in range(3)]))
    return fit_pred(Zf, yf, Zs, best), best


def colsum_exact(m, teA, teB, dev, amp, bs):
    """note 152-5 (descriptive): is the carry-save column sum stored EXACTLY? colsum_readout's states (additive heads after segments 1-4), targets
    (carry-save column sums k = 1..7, no carries) and halves; the same kernel ridge refit to return predictions. Per column: final = round(pred) == y
    after segment 4; live = pooled over the (segment, column) cells holding >= 1 product; R^2; max |residual| (final)."""
    N = len(teA); h = N // 2; _, kept, _, _ = rollout_drow(m, teA, teB, dev, amp, bs, keep=(1, 2, 3, 4), decode=False)
    cs = np.zeros((N, 4, 7), np.float64)
    for j in range(4):
        for i in range(4):
            cs[:, j:, i + j] += (teA[:, i] * teB[:, j])[:, None]
    XA = np.stack([kept[j + 1][:, :DM.H_ADD].reshape(N, -1) for j in range(4)], 1); del kept
    out = {"final_exact_by_column": {}, "live_exact_by_column": {}, "r2_by_column": {}, "max_abs_resid_final": {}, "lambda": {}}
    for k in range(7):
        y = cs[:, :, k]
        if y.std() == 0:
            continue
        pred, lam = kridge_fit_pred(XA[:h].reshape(h * 4, -1), y[:h].reshape(-1), XA[h:].reshape((N - h) * 4, -1)); P = pred.reshape(N - h, 4); Y = y[h:]
        live = np.array([[j >= k - 3 and k <= j + 3 for j in range(4)]] * (N - h))
        out["final_exact_by_column"][str(k + 1)] = float((np.rint(P[:, 3]) == Y[:, 3]).mean()); out["live_exact_by_column"][str(k + 1)] = float((np.rint(P[live]) == Y[live]).mean())
        out["r2_by_column"][str(k + 1)] = float(1 - ((Y - P) ** 2).sum() / max(((Y - Y.mean()) ** 2).sum(), 1e-12))
        out["max_abs_resid_final"][str(k + 1)] = float(np.abs(P[:, 3] - Y[:, 3]).max()); out["lambda"][str(k + 1)] = lam
    return out


# ================================================================ THEORY §156 SEP-SLOT arm (b): separator slots in §152's reader decode
_RSLOT = {"pos": None, "cap": None, "rep": None, "i": 0, "layer": None}    # slot_patch: capture h[:, pos] at every block input; replace at one (reader pass)


def row_carries(A, B):
    """carry into each digit k of S_4 = a x b (LSB first) and the products-only column sums."""
    n = len(A); cs = np.zeros((n, NDIG), np.int64)
    for i in range(A.shape[1]):
        for j in range(B.shape[1]):
            cs[:, i + j] += A[:, i] * B[:, j]
    ci = np.zeros((n, NDIG), np.int64)
    for k in range(1, NDIG):
        ci[:, k] = (cs[:, k - 1] + ci[:, k - 1]) // 10
    return ci, cs


def _patch_shares(got, cs_r, ci_d, own, dd):
    """§156 slot_patch shares for one layer: donor_carry (recipient column sum + donor carry), own (recipient digit), donor_digit (donor's own)."""
    tgt = (cs_r + ci_d) % 10
    return {"donor_carry_share": float((got == tgt).mean()), "own_digit_share": float((got == own).mean()), "donor_digit_share": float((got == dd).mean()),
            "chance_overlap": {"carry_vs_own": float((tgt == own).mean()), "donor_digit_vs_own": float((dd == own).mean()), "carry_vs_donor_digit": float((tgt == dd).mean())}}


@torch.no_grad()
def s156_row_eval(m, teA, teB, dev, amp, bs, slot_col=4, seed=15601):
    """§156 (b) descriptive on the final decode after segment 4 (teacher-forced reader pass): seq_len; ring_probe (per slot k: held-out ridge R^2
    of cos / sin(2 pi carry_k / C) from the reader's final residual at the slot, pre-ln_f); slot_patch (column slot_col: the slot residual at every
    block input of the reader pass from a donor with a different carry-in: (recipient column sum + donor carry) mod 10 vs the recipient's digit)."""
    from boxes.multcol import ridge_r2
    n = len(teA); h = n // 2; S4 = sj_values(teA, teB)[:, -1]; ans = np.array(DTOK)[digits8(S4)]; ci, cs = row_carries(teA, teB); C = int(ci.max()) + 1
    _, kept, _, _ = rollout_drow(m, teA, teB, dev, amp, bs, keep=(4,), decode=False); St = kept[4]; H = np.zeros((n, NDIG, 768), np.float32)
    pos = [2 + 2 * k for k in range(NDIG)]
    def reader_resid(S, a_ids):                                             # the reader pass's final residual (pre-ln_f) at every position
        g = m.g; ids = m.reader_ids(S.shape[0], S.device, a_ids); x = g.embed(ids); B_, T_, D_ = x.shape; hd = D_ // g.H; hh = x + g.w("wpe.weight")[:T_][None]
        for i in range(g.L):
            if i == m.mem.layer:
                hh = mem_read(m.mem, hh, S)
            hh = DM._block(g, i, hh, hd) if not _RSLOT_hook(i) else _patched_block(g, i, hh, hd)
        return hh
    for s_ in range(0, n, bs):
        sl = slice(s_, s_ + bs)
        with amp_ctx(amp):
            hh = reader_resid(torch.tensor(St[sl], device=dev), torch.tensor(ans[sl], device=dev))
        H[sl] = hh[:, pos].float().cpu().numpy()
    ring = {}
    for k in range(1, NDIG):
        th = 2 * np.pi * ci[:, k] / C
        ring[str(k + 1)] = {"cos_r2": ridge_r2(H[:h, k], np.cos(th[:h]), H[h:, k], np.cos(th[h:])) if np.cos(th).std() > 0 else None,
                            "sin_r2": ridge_r2(H[:h, k], np.sin(th[:h]), H[h:, k], np.sin(th[h:])) if np.sin(th).std() > 0 else None}
    ring["C"] = C; ring["def"] = "per slot (digit k of S_4): held-out ridge R^2 of cos / sin(2 pi carry_in_k / C), the reader's final residual at the slot (after segment 4)"
    k = slot_col - 1; p_ = pos[k]; rng = np.random.default_rng(seed); ir = rng.permutation(n)[:min(500, n)]; idn = rng.permutation(n)[:len(ir)]
    for _ in range(100):
        bad = np.nonzero(ci[idn, k] == ci[ir, k])[0]
        if not len(bad):
            break
        idn[bad] = rng.integers(0, n, len(bad))
    keep = ci[idn, k] != ci[ir, k]; ir, idn = ir[keep], idn[keep]; nl = m.g.L; got = np.zeros((nl, len(ir)), np.int64); lut = {t: i_ for i_, t in enumerate(DTOK)}
    for s_ in range(0, len(ir), bs):
        a_, d_ = ir[s_:s_ + bs], idn[s_:s_ + bs]
        _RSLOT.update({"pos": p_, "cap": [], "rep": None, "i": 0, "layer": None})
        with amp_ctx(amp):
            reader_resid(torch.tensor(St[d_], device=dev), torch.tensor(ans[d_], device=dev))
        cap = _RSLOT["cap"]
        for lay in range(nl):                                               # Audit: one block input per pass, swept 0 .. 11
            _RSLOT.update({"cap": None, "rep": cap, "i": 0, "layer": lay})
            with amp_ctx(amp):
                hh = reader_resid(torch.tensor(St[a_], device=dev), torch.tensor(ans[a_], device=dev))
            D_ = hh.shape[-1]; lg = m.g.logits(Fn.layer_norm(hh[:, p_], (D_,), m.g.w("ln_f.weight"), m.g.w("ln_f.bias"))).float()
            got[lay, s_:s_ + bs] = [lut.get(int(t), -1) for t in lg.argmax(-1).tolist()]
        _RSLOT.update({"pos": None, "rep": None, "i": 0, "layer": None})
    D8 = digits8(S4); by_layer = {str(lay): _patch_shares(got[lay], cs[ir, k], ci[idn, k], D8[ir, k], D8[idn, k]) for lay in range(nl)}
    patch = {"column": slot_col, "n": int(len(ir)), "by_layer": by_layer, "layers": "reader-pass block inputs 0 .. 11 (after segment 4)",
             "def": "the reader pass's slot residual replaced at ONE block input (per layer) by a donor's with a different carry-in (teacher-forced); "
                    "shares: donor_carry = (recipient column sum + donor carry) mod 10, own = recipient digit, donor_digit = the donor's own digit"}
    seg_t = MultRowDelta.SEG_T * 4; rd_t = 2 + 2 * NDIG
    return {"seq_len": int(seg_t + rd_t), "seq_len_def": "4 segments x 7 + the reader pass [EOT, ' ='] + 8 x ('|', digit)", "ring_probe": ring, "slot_patch": patch,
            "answer_sep": m.sep_mode, "sep_token_id": m.sep, "slot_positions_reader": pos}


def _RSLOT_hook(i):
    return _RSLOT["pos"] is not None


def _patched_block(g, i, hh, hd):
    if _RSLOT["cap"] is not None:
        _RSLOT["cap"].append(hh[:, _RSLOT["pos"]].detach().clone())
    elif _RSLOT["rep"] is not None and _RSLOT["i"] == _RSLOT["layer"]:    # Audit: a single-block patch
        hh = hh.clone(); hh[:, _RSLOT["pos"]] = _RSLOT["rep"][_RSLOT["i"]].to(hh.dtype)
    _RSLOT["i"] += 1
    return DM._block(g, i, hh, hd)


@torch.no_grad()
def s157_eval(m, teA, teB, trA, trB, dev, amp, bs, a, n_fit=4000):
    """§157 descriptive: digit_acc_teacher_forced (per column of S_4), read_exposure (near-OLS ridge, lambda <= 1e-8, of the accumulator c^_k =
    column sum + carry-in from the reader pass's block-6 read at the position predicting digit k; MULTIREAD also from the read tokens' layer-0
    vectors and their residual after block 6; fit on n_fit train items, scored on the test items: exact = round(pred) == c^_k, per column),
    key_cos (mean |cos| between an item's write keys, per head), write_round_err (PERIODIC), offcol_weight (CCQ: None until built)."""
    from boxes.multcol import load_bench_head  # noqa: F401
    def acc_t(A, B):
        cs = np.zeros((len(A), NDIG), np.int64)
        for j in range(4):
            for i in range(4):
                cs[:, i + j] += A[:, i] * B[:, j]
        ac = np.zeros_like(cs); c_ = np.zeros(len(A), np.int64)
        for k in range(NDIG):
            ac[:, k] = cs[:, k] + c_; c_ = ac[:, k] // 10
        return ac
    def caps(A, B):
        N = len(A); S4 = sj_values(A, B)[:, -1]; ans = np.array(DTOK)[digits8(S4)]; rd = np.zeros((N, NDIG, 768), np.float32); pr = np.zeros((N, NDIG), np.int64)
        mr = getattr(m.mem, "mr", None) is not None; t0 = np.zeros((N, MultiRead.R * 768), np.float32) if mr else None; t7 = np.zeros_like(t0) if mr else None
        _, kept, _, _ = rollout_drow(m, A, B, dev, amp, bs, keep=(4,), decode=False); lut = {t: i_ for i_, t in enumerate(DTOK)}
        for s_ in range(0, N, bs):
            sl = slice(s_, s_ + bs); S = torch.tensor(kept[4][sl], device=dev); an = torch.tensor(ans[sl], device=dev); ids = m.reader_ids(len(an), dev, an)
            x = m.g.embed(ids); cp = []; mc = {} if mr else None
            hook = []; hk = None
            def pre_hook(mod_, inp_):
                hook.append(inp_[0].detach())
            with amp_ctx(amp):
                if mr and float(m.mem.mr.fade) > 0:
                    out, _ = _fwd_mem_mr(m.g, x, m.mem, S, read_only=True, cap=cp, mrcap=mc)
                else:
                    out, _ = fwd_mem(m.g, x, m.mem, S, read_only=True, cap=cp)
                h6 = cp[0]; rdv = mem_read(m.mem, h6, S) - h6                                  # the read's contribution at every position
            pos = slice(2, None, 2) if getattr(m, "sep", None) is not None else slice(1, 1 + NDIG)
            rd[sl] = rdv[:, pos].float().cpu().numpy(); pr[sl] = np.vectorize(lambda t_: lut.get(int(t_), -1))(m.g.logits(out[:, pos]).float().argmax(-1).cpu().numpy())
            if mr and mc:
                t0[sl] = mc["tok0"].flatten(1).float().cpu().numpy(); t7[sl] = mc["tokL7"].flatten(1).float().cpu().numpy()
        return rd, pr, t0, t7
    def ols(Xf, yf, Xs, lams=(1e-10, 1e-8)):
        mu, sd = Xf.mean(0), Xf.std(0) + 1e-6; Zf = (Xf - mu) / sd; Zs = (Xs - mu) / sd; d = Zf.shape[1]; fold = np.arange(len(Zf)) % 3
        def fit(Za, ya, lam):
            ym = ya.mean(); return np.linalg.solve(Za.T @ Za + lam * len(Za) * np.eye(d), Za.T @ (ya - ym)), ym
        def r2(y_, p_):
            return 1 - ((y_ - p_) ** 2).sum() / max(((y_ - y_.mean()) ** 2).sum(), 1e-12)
        best = max(lams, key=lambda l_: np.mean([r2(yf[fold == k], Zf[fold == k] @ fit(Zf[fold != k], yf[fold != k], l_)[0] + fit(Zf[fold != k], yf[fold != k], l_)[1]) for k in range(3)]))
        W, ym = fit(Zf, yf, best); return Zs @ W + ym
    ac_te = acc_t(teA, teB); trA, trB = trA[:n_fit], trB[:n_fit]; ac_tr = acc_t(trA, trB)
    Rtr, _, T0tr, T7tr = caps(trA, trB); Rte, Pte, T0te, T7te = caps(teA, teB); out = {"read_mode": a.read_mode}
    out["digit_acc_teacher_forced"] = {str(k + 1): float((Pte[:, k] == ac_te[:, k] % 10).mean()) for k in range(NDIG)}
    rx = {"block6_read": {}}
    for k in range(NDIG):
        if ac_tr[:, k].std() > 0:
            rx["block6_read"][str(k + 1)] = float((np.rint(ols(Rtr[:, k].astype(np.float64), ac_tr[:, k].astype(np.float64), Rte[:, k].astype(np.float64))) == ac_te[:, k]).mean())
    if T0tr is not None:
        for nm, Xtr, Xte in (("read_tokens_layer0", T0tr, T0te), ("read_tokens_after_block6", T7tr, T7te)):
            rx[nm] = {str(k + 1): float((np.rint(ols(Xtr.astype(np.float64), ac_tr[:, k].astype(np.float64), Xte.astype(np.float64))) == ac_te[:, k]).mean())
                      for k in range(NDIG) if ac_tr[:, k].std() > 0}
    rx["def"] = "exact = round(pred) == c^_k (column sum + carry-in), near-OLS ridge (lambda <= 1e-8, standardised), fit %d train items, scored on the test items" % len(trA)
    out["read_exposure"] = rx
    kc = []                                                                 # key_cos: the write keys entering each segment's branch
    hkc = m.mem.register_forward_pre_hook(lambda mod_, inp_: kc.append(inp_[0].detach().float()))
    try:
        rollout_drow(m, teA[:128], teB[:128], dev, amp, 128, decode=False)
    finally:
        hkc.remove()
    K = torch.cat([m.mem.qkv(m.mem.ln(x_))[..., 768:1536] for x_ in kc], 1).view(len(kc[0]), -1, DM.H_ADD + DM.H_DEL, DM.DH)   # (B, 4 x 7, 12, 64)
    Kn = Fn.normalize(K, dim=-1); cs_ = torch.einsum("bthd,bshd->bhts", Kn, Kn).abs(); n_ = cs_.shape[-1]; off = ~torch.eye(n_, dtype=torch.bool, device=cs_.device)
    out["key_cos"] = [round(float(cs_[:, h_][:, off].mean()), 5) for h_ in range(cs_.shape[1])]
    if getattr(m.mem, "periodic", False):
        m.mem.round_err = []; rollout_drow(m, teA[:256], teB[:256], dev, amp, 256, decode=False); e_ = torch.cat(m.mem.round_err).numpy(); m.mem.round_err = None
        out["write_round_err"] = {"mean": float(e_.mean()), "median": float(np.median(e_)), "p90": float(np.quantile(e_, 0.9)), "max": float(e_.max()), "n": int(e_.size)}
    else:
        out["write_round_err"] = None
    out["offcol_weight"] = None
    if getattr(m.mem, "ccq", False):                                        # note 157-1
        out.update(ccq_extras(m, teA, teB, dev, amp, bs))
    if getattr(m.mem, "periodic", False):                                   # Audit (978a2399 review): read-side diagnostics
        out.update(periodic_extras(m, teA, teB, dev, amp))
    if getattr(m.mem, "mr", None) is not None:
        out["multiread_queries"] = "static learned queries (Perceiver-style), not input-dependent; 4 per head, each with a learned slot ID"
    return out


# ================================================================ THEORY §160 R-ROWS (785c3838): DeltaMem-R's write source on the row harness
def _row_acc(A, B, j):
    """after segment j (1-based): per column k the accumulator c^_k = carry-save column sum of a x (b_1 .. b_j) + carry-in; its units = digit k of S_j."""
    n = len(A); cs = np.zeros((n, NDIG), np.int64)
    for jj in range(j):
        for i in range(A.shape[1]):
            cs[:, i + jj] += A[:, i] * B[:, jj]
    ac = np.zeros_like(cs); c_ = np.zeros(n, np.int64)
    for k in range(NDIG):
        ac[:, k] = cs[:, k] + c_; c_ = ac[:, k] // 10
    return ac


def _dual_w(Xf, yf, dev, lams=(1e-9, 1e-8, 1e-7, 1e-6)):
    """dual ridge (standardised, lambda by 3-fold CV R^2, edge-widened) returning the RAW-space weight w, mu, intercept: pred = (x - mu) . w + b."""
    Zf = torch.tensor(Xf, device=dev, dtype=torch.float32); mu, sd = Zf.mean(0), Zf.std(0) + 1e-6; Zf = (Zf - mu) / sd; d = Zf.shape[1]; n = len(Zf)
    K = (Zf @ Zf.T).double() / d; Y = torch.tensor(yf, device=dev, dtype=torch.float64); fold = torch.arange(n, device=dev) % 3
    def fit(idx, lam):
        ym = Y[idx].mean(); return torch.linalg.solve(K[idx][:, idx] + lam * len(idx) * torch.eye(len(idx), device=dev, dtype=torch.float64), Y[idx] - ym), ym
    def cv(lam):
        sc = []
        for k in range(3):
            tr = torch.nonzero(fold != k).squeeze(1); va = torch.nonzero(fold == k).squeeze(1); al, ym = fit(tr, lam); p_ = K[va][:, tr] @ al + ym
            sc.append(float(1 - ((Y[va] - p_) ** 2).sum() / ((Y[va] - Y[va].mean()) ** 2).sum().clamp_min(1e-12)))
        return float(np.mean(sc))
    best = max(lams, key=cv)
    if best in (min(lams), max(lams)):
        best = max([best] + [best * 10 ** k for k in ((-1, -2) if best == min(lams) else (1, 2))], key=cv)
    al, ym = fit(torch.arange(n, device=dev), best); wz = (Zf.T.double() @ al) / d
    return (wz / sd.double()).float(), mu, float(ym), best


def s160_probes(m, teA, teB, trA, trB, dev, amp, bs, n_fit=8000, ks=(2, 3, 4), deltas=(-1, 1), seed=16001):
    """§160 descriptive. what_state_rows: from the flattened state after segments 1..4 (n_fit train -> the test items; dual near-OLS, CV lambda):
    10-way digit classifiers of a_i and the seen b_j, and colsum_cls (the units digit of c^_k and its carry class, k with >= 1 product so far).
    c1_edit: a minimum-norm edit of the state after segment 2 along the raw-space ridge probe w of S_2's column-k units digit (delta w / ||w||^2:
    the probe moves by exactly delta), k = 2..4, delta +-1; the final answer after rows 3-4 scored against the edited S_2 (S_4 + delta 10^(k-1)):
    follow_share (all rows and rows where the edited digit stays in 0..9), unedited_share (== S_4), and an equal-norm random edit."""
    from boxes.multpair import _dual_fit_pred
    trA, trB = trA[:n_fit], trB[:n_fit]; _, Ktr, _, _ = rollout_drow(m, trA, trB, dev, amp, bs, keep=(1, 2, 3, 4), decode=False)
    _, Kte, _, _ = rollout_drow(m, teA, teB, dev, amp, bs, keep=(1, 2, 3, 4), decode=False); ws = {}
    for j in range(1, 5):
        Xf = Ktr[j].reshape(len(trA), -1); Xs = Kte[j].reshape(len(teA), -1); r_ = {"a": {}, "b": {}, "colsum_cls": {}}
        for nm, Dtr, Dte, idx in (("a", trA, teA, range(trA.shape[1])), ("b", trB, teB, range(j))):
            for i in idx:
                pr_, lam_ = _dual_fit_pred(Xf, np.eye(10)[Dtr[:, i]], Xs, dev, cls=True); r_[nm][str(i + 1)] = {"acc": float((pr_.argmax(1) == Dte[:, i]).mean()), "lambda": lam_}
        atr, ate = _row_acc(trA, trB, j), _row_acc(teA, teB, j)
        for k in range(min(NDIG, j + trA.shape[1] - 1)):
            rc = {}
            for nm_, ff_, ft_ in (("units", atr[:, k] % 10, ate[:, k] % 10), ("carry", atr[:, k] // 10, ate[:, k] // 10)):
                kk = int(max(ff_.max(), ft_.max())) + 1; pr_, lam_ = _dual_fit_pred(Xf, np.eye(kk)[ff_], Xs, dev, cls=True)
                rc[nm_] = {"acc": float((pr_.argmax(1) == ft_).mean()), "lambda": lam_, "classes": kk}
            r_["colsum_cls"][str(k + 1)] = rc
        r_["n_fit"] = int(len(trA)); r_["n_score"] = int(len(teA)); ws["after_segment%d" % j] = r_
    S2 = sj_values(teA, teB)[:, 1]; S4 = sj_values(teA, teB)[:, -1]; X2tr = Ktr[2].reshape(len(trA), -1); X2te = Kte[2]; c1 = {"by_k": {}}; tot = {"f": 0, "fn": 0, "nn": 0, "u": 0, "r": 0, "n": 0}
    rng = np.random.default_rng(seed); S2tr = sj_values(trA, trB)[:, 1]
    for k in ks:
        w, mu, b0, lam_ = _dual_w(X2tr, ((S2tr // 10 ** (k - 1)) % 10).astype(float), dev); wn = w.cpu().numpy(); u = wn / float(wn @ wn)
        dig = (S2 // 10 ** (k - 1)) % 10; byd = {}
        for dl in deltas:
            Ed = (X2te.reshape(len(teA), -1) + dl * u[None]).reshape(X2te.shape).astype(np.float32); got = rollout_drow(m, teA, teB, dev, amp, bs, override=(2, Ed))[0][:, -1]
            rv = rng.standard_normal(u.shape).astype(np.float32); rv *= np.linalg.norm(dl * u) / np.linalg.norm(rv)
            Er = (X2te.reshape(len(teA), -1) + rv[None]).reshape(X2te.shape).astype(np.float32); gr = rollout_drow(m, teA, teB, dev, amp, bs, override=(2, Er))[0][:, -1]
            tgt = S4 + dl * 10 ** (k - 1); nw = (dig + dl >= 0) & (dig + dl <= 9)
            byd["%+d" % dl] = {"follow_share": float((got == tgt).mean()), "follow_share_nowrap": float((got[nw] == tgt[nw]).mean()) if nw.any() else None,
                               "unedited_share": float((got == S4).mean()), "random_edit_follow_share": float((gr == tgt).mean()), "n": int(len(teA)), "n_nowrap": int(nw.sum())}
            tot["f"] += int((got == tgt).sum()); tot["fn"] += int((got[nw] == tgt[nw]).sum()); tot["nn"] += int(nw.sum()); tot["u"] += int((got == S4).sum()); tot["r"] += int((gr == tgt).sum()); tot["n"] += len(teA)
        c1["by_k"][str(k)] = {"probe_lambda": lam_, **byd}
    c1["pooled"] = {"follow_share": tot["f"] / max(tot["n"], 1), "follow_share_nowrap": tot["fn"] / max(tot["nn"], 1), "unedited_share": tot["u"] / max(tot["n"], 1),
                    "random_edit_follow_share": tot["r"] / max(tot["n"], 1), "n": tot["n"]}
    c1["def"] = "minimum-norm edit delta w / ||w||^2 of the state after segment 2 along the raw-space ridge probe of S_2's column-k units digit (8k train fit)"
    return {"what_state_rows": ws, "c1_edit": c1}


def s152_eval(m, teA, teB, dev, amp, bs, a, dec):
    """§152 descriptive block (eval mode, no draws): pp_probe, add_state_norm_by_segment, bprefix_probe (+ probe_r2), colsum_readout,
    head_ablation, alpha_by_head; sj_decode_acc (the reader after each segment = acc_by_length, A only)."""
    from boxes.fsq120 import _ridge_cls_acc
    from boxes.multcol import gen_operands
    S = sj_values(teA, teB); N = len(teA); h = N // 2; out = {}
    out["sj_decode_acc"] = {str(j + 1): float((dec[:, j] == S[:, j]).mean()) for j in range(4)} if a.reader else None
    _, kept, nrm, C = rollout_drow(m, teA, teB, dev, amp, bs, keep=(1, 2, 3, 4), norms=True, caps=True, decode=False)
    pp = {"P_j_last_pos": {}, "aibj_at_ai": {}, "def": "per segment j: 10-way ridge classifiers per digit on the residual entering the branch (after 6 blocks), fit / score halves"}
    ai = to_int(teA)
    for j in range(4):
        P = ai * teB[:, j]; X = C[:, j, -1]
        pp["P_j_last_pos"][str(j + 1)] = [float(_ridge_cls_acc(X[:h], (P[:h] // 10 ** k) % 10, X[h:], (P[h:] // 10 ** k) % 10)) for k in range(5)]
        pp["aibj_at_ai"][str(j + 1)] = {}
        for i in range(4):
            v = teA[:, i] * teB[:, j]; X = C[:, j, 3 + i]
            pp["aibj_at_ai"][str(j + 1)][str(i + 1)] = [float(_ridge_cls_acc(X[:h], (v[:h] // 10 ** k) % 10, X[h:], (v[h:] // 10 ** k) % 10)) for k in range(2)]
    out["pp_probe"] = pp
    xA = gen_operands(np.random.default_rng(14004), min(a.ext_n, 1000), 4); xB = gen_operands(np.random.default_rng(14005), min(a.ext_n, 1000), 8)
    _, _, nx, _ = rollout_drow(m, xA, xB, dev, amp, bs, L=8, norms=True, decode=False)
    out["add_state_norm_by_segment"] = {"additive": {str(j + 1): float(np.median((nrm if j < 4 else nx)[:, j, 0])) for j in range(8)},
                                        "delta": {str(j + 1): float(np.median((nrm if j < 4 else nx)[:, j, 1])) for j in range(8)},
                                        "def": "median Frobenius norm of each head type's state after segment j (1-4: test chains, 5-8: 4x8 ext chains)"}
    bpre = np.stack([to_int(teB[:, :j]) for j in range(1, 5)], 1); bp = {"bprefix": {}, "sj": {}, "lambda": {}}
    for j in range(4):
        X = kept[j + 1].reshape(N, -1)
        r_b, l_b = kridge_r2(X[:h], np.log10(1 + bpre[:h, j]), X[h:], np.log10(1 + bpre[h:, j])); r_s, l_s = kridge_r2(X[:h], np.log10(1 + S[:h, j]), X[h:], np.log10(1 + S[h:, j]))
        bp["bprefix"][str(j + 1)] = r_b; bp["sj"][str(j + 1)] = r_s; bp["lambda"][str(j + 1)] = [l_b, l_s]
    bp["def"] = "held-out kernel-ridge R^2 (kridge_r2) of log10(1 + b-prefix) / log10(1 + S_j) from the flattened memory state after segment j"
    out["bprefix_probe"] = bp; out["probe_r2"] = {"sj": bp["sj"], "bprefix": bp["bprefix"], "source": "memory state (bprefix_probe)"}
    cs = np.zeros((N, 4, 7), np.float64)                                    # carry-save column sums after segment j: sum_{i + j' = k, j' < j} a_i b_j'
    for j in range(4):
        for i in range(4):
            cs[:, j:, i + j] += (teA[:, i] * teB[:, j])[:, None]
    XA = np.stack([kept[j + 1][:, :DM.H_ADD].reshape(N, -1) for j in range(4)], 1); col = {}
    for k in range(7):
        y = cs[:, :, k]
        if y.std() > 0:
            col[str(k + 1)] = kridge_r2(XA[:h].reshape(h * 4, -1), y[:h].reshape(-1), XA[h:].reshape((N - h) * 4, -1), y[h:].reshape(-1))[0]
    out["colsum_readout"] = {"r2_by_column": col, "def": "held-out kernel-ridge R^2 of carry-save column sum k (no carries) from the additive state, pooled over segments 1-4"}
    del kept, C, XA
    abl = {}
    for ht in ("add", "delta"):
        m.mem.ablate = ht; dk = rollout_drow(m, teA, teB, dev, amp, bs)[0]; m.mem.ablate = None
        abl[ht + "_zeroed"] = {"test_acc": float((dk[:, -1] == S[:, -1]).mean()), "acc_by_length": {str(j + 1): float((dk[:, j] == S[:, j]).mean()) for j in range(4)}}
    out["head_ablation"] = abl
    return out


# ================================================================ THEORY §153 THINK-PASS (ea154f30 + note 153-1): --pass_input on §152's carrier
XTOK, REGTOK = bc.tok(" \u00d7")[0], bc.tok(" _")[0]                     # " x" (times sign, one GPT-2 token); the register token (fixed, told apart by position)


def pass_list(m, L):
    """The write passes of an L-chain: pure / repeat add pass 0 (the problem, writes only, index -1) before the L step passes."""
    return ([-1] if m.pass_input in ("pure", "repeat") else []) + list(range(L))


@torch.no_grad()
def pass_reads(m, A, B, dev, amp, bs, L=None, keep_mem0=False):
    """The branch read [g_add read_add || g_del read_del] (W_o's input, every position) of each write pass, flattened: {pass: (N, T x 768)}
    (pass "0" = pure pass 0, "j" = step pass j); keep_mem0: the flattened memory after pure pass 0 (N, 49152)."""
    L = L or B.shape[1]; buf = []; hk = m.mem.wo.register_forward_hook(lambda mod_, inp_, out_: buf.append(inp_[0].detach().float()))
    out = {}; M0 = []
    try:
        for i in range(0, len(A), bs):
            sl = slice(i, i + bs); Bn = len(A[sl]); set_ops(m, A[sl], B[sl], dev); m.Lp = B.shape[1]; S = m.start(Bn, dev)
            for t in pass_list(m, L):
                buf.clear()
                with amp_ctx(amp):
                    S = m.write(S, t)
                out.setdefault(str(t + 1), []).append(buf[0].flatten(1).cpu().numpy())
                if t < 0 and keep_mem0:
                    M0.append(S.float().flatten(1).cpu().numpy())
    finally:
        hk.remove()
    return {k: np.concatenate(v) for k, v in out.items()}, (np.concatenate(M0) if M0 else None)


def kridge_cls_acc(Xf, yf, Xs, ys, k=10, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    """10-way linear classifier per digit for d >> n: multi-output kernel ridge (K = Z Z^T / d) on one-hot targets, argmax; lambda (x n) by 3-fold
    CV accuracy on the fit half; held-out accuracy on the score half (as fsq120._ridge_cls_acc, dual form)."""
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6; Zf = ((Xf - mu) / sd).astype(np.float32); Zs = ((Xs - mu) / sd).astype(np.float32); d = Zf.shape[1]
    def fit_pred(Za, ya, Zb, lam):
        K = (Za @ Za.T).astype(np.float64) / d; Y = np.eye(k)[ya.astype(int)]; Ym = Y.mean(0)
        return ((Zb @ Za.T).astype(np.float64) / d @ np.linalg.solve(K + lam * len(Za) * np.eye(len(Za)), Y - Ym) + Ym).argmax(1)
    fold = np.arange(len(Zf)) % 3
    best = max(lams, key=lambda l_: np.mean([(fit_pred(Zf[fold != j], yf[fold != j], Zf[fold == j], l_) == yf[fold == j]).mean() for j in range(3)]))
    return float((fit_pred(Zf, yf, Zs, best) == ys).mean())


def digit_recall(X, A, B, h):
    """held-out recall of a's and b's digits (4 + 4, LSB first) from features X (fit / score halves)."""
    return {"a": [kridge_cls_acc(X[:h], A[:h, i], X[h:], A[h:, i]) for i in range(A.shape[1])],
            "b": [kridge_cls_acc(X[:h], B[:h, i], X[h:], B[h:, i]) for i in range(B.shape[1])]}


@torch.no_grad()
def attn_bj_share(m, A, B, dev, amp, n=256):
    """§153 VISIBLE: per step pass j, the attention mass (mean over the 12 layers, 12 heads and the query positions after b_j) on b_j's position,
    beside the mean mass on the other b positions; explicit softmax(q k^T / sqrt(hd) + causal) on the same residual stream as fwd_mem."""
    g = m.g; A, B = A[:n], B[:n]; Lb = B.shape[1]; set_ops(m, A, B, dev); m.Lp = Lb; S = m.start(len(A), dev); res = {}
    for t in range(Lb):
        x = g.embed(m.seg_ids(t, len(A), dev)); Bn, T, D = x.shape; hd = D // g.H; h = x + g.w("wpe.weight")[:T][None]; shares = []
        mask = torch.full((T, T), float("-inf"), device=dev).triu(1)
        with amp_ctx(amp):
            for i in range(g.L):
                if i == m.mem.layer:
                    h, S = m.mem(h, S)
                pre = f"h.{i}."; a_ = Fn.layer_norm(h, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
                q, k, v = (a_ @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")).split(D, -1)
                q, k, v = (t_.view(Bn, T, g.H, hd).transpose(1, 2) for t_ in (q, k, v))
                pr = ((q.float() @ k.float().transpose(-1, -2)) / math.sqrt(hd) + mask).softmax(-1); shares.append(pr)
                o = Fn.scaled_dot_product_attention(q, k, v, is_causal=True)
                h = h + o.transpose(1, 2).reshape(Bn, T, D) @ g.w(pre + "attn.c_proj.weight") + g.w(pre + "attn.c_proj.bias")
                a_ = Fn.layer_norm(h, (D,), g.w(pre + "ln_2.weight"), g.w(pre + "ln_2.bias"))
                h = h + Fn.gelu(a_ @ g.w(pre + "mlp.c_fc.weight") + g.w(pre + "mlp.c_fc.bias"), approximate="tanh") @ g.w(pre + "mlp.c_proj.weight") + g.w(pre + "mlp.c_proj.bias")
        P = torch.stack(shares)                                             # (layers, B, heads, T, T)
        bpos = [6 + j for j in range(Lb)]; pj = bpos[t]; qs = list(range(pj + 1, T))
        on = float(P[:, :, :, qs, pj].mean()); oth = float(np.mean([float(P[:, :, :, [q_ for q_ in qs], pk].mean()) for pk in bpos if pk != pj]))
        res[str(t + 1)] = {"on_bj": on, "on_other_b_mean": oth, "by_layer_on_bj": [round(float(P[l_, :, :, qs, pj].mean()), 5) for l_ in range(P.shape[0])]}
    res["def"] = "mean attention probability from pass j's positions after b_j onto b_j's position (all layers / heads), vs the other b positions"
    return res


def s153_eval(m, teA, teB, dev, amp, bs, a, dec):
    """§153 descriptive block (eval mode, no draws): noop {K6, K8}; attn_bj_share (visible); operand_recall_pass1 + bj_recall_by_pass (pure /
    repeat); head_roles; mem_norm_by_pass; sj_decode_acc (A); reader_text; head_ablation."""
    from boxes.multcol import to_int as ti_
    S = sj_values(teA, teB); N = len(teA); h = N // 2; out = {}
    out["sj_decode_acc"] = {str(j + 1): float((dec[:, j] == S[:, j]).mean()) for j in range(4)} if a.reader else None
    nz = {}
    for K in (6, 8):                                                        # extra passes after the 4 (step tokens 5.., the problem unchanged)
        dk = rollout_drow(m, teA, teB, dev, amp, bs, L=K, Lp=4)[0]; _, kp, _, _ = rollout_drow(m, teA[:256], teB[:256], dev, amp, bs, L=K, Lp=4, keep=tuple(range(4, K + 1)), decode=False)
        ch = {str(k): float(np.median(np.linalg.norm((kp[k] - kp[k - 1]).reshape(len(kp[k]), -1), axis=1) / np.maximum(np.linalg.norm(kp[k - 1].reshape(len(kp[k]), -1), axis=1), 1e-12)))
              for k in range(5, K + 1)}
        nz["K%d" % K] = {"test_acc": float((dk[:, -1] == S[:, -1]).mean()), "mem_change_by_extra_pass": ch}
    nz["def"] = "4x4 test items with K passes (the decode after the last); mem_change = median ||S_k - S_k-1|| / ||S_k-1|| per extra pass k (256 items)"
    out["noop"] = nz
    out["attn_bj_share"] = attn_bj_share(m, teA, teB, dev, amp) if a.pass_input == "visible" else None
    rd, M0 = pass_reads(m, teA, teB, dev, amp, bs, keep_mem0=a.pass_input in ("pure", "repeat"))
    if a.pass_input in ("pure", "repeat"):
        out["operand_recall_pass1"] = {"memory_after_pass0": digit_recall(M0, teA, teB, h), "reads_pass1": digit_recall(rd["1"], teA, teB, h),
                                       "def": "held-out 10-way kernel-ridge probes of a's / b's digits (LSB first): (a) flattened memory after pass 0, (b) branch reads at pass 1's positions"}
        out["bj_recall_by_pass"] = {j: digit_recall(rd[j], teA, teB, h)["b"] for j in ("1", "2", "3", "4")}
    else:
        out["operand_recall_pass1"] = None; out["bj_recall_by_pass"] = None
    del M0
    roles = {}; abl = {}
    for ht in ("add", "delta"):
        m.mem.ablate = ht; dk = rollout_drow(m, teA, teB, dev, amp, bs)[0]; rk, _ = pass_reads(m, teA, teB, dev, amp, bs); m.mem.ablate = None
        abl[ht + "_zeroed"] = {"test_acc": float((dk[:, -1] == S[:, -1]).mean()), "acc_by_length": {str(j + 1): float((dk[:, j] == S[:, j]).mean()) for j in range(4)}}
        roles[ht + "_zeroed"] = {**abl[ht + "_zeroed"], "recall_reads_pass1": digit_recall(rk["1"], teA, teB, h)}
    out["head_ablation"] = abl; out["head_roles"] = roles
    _, kp, nrm, _ = rollout_drow(m, teA, teB, dev, amp, bs, norms=True, decode=False)
    mn = {"additive": {str(j + 1): float(np.median(nrm[:, j, 0])) for j in range(4)}, "delta": {str(j + 1): float(np.median(nrm[:, j, 1])) for j in range(4)}}
    if a.pass_input in ("pure", "repeat"):
        n0 = min(256, len(teA)); set_ops(m, teA[:n0], teB[:n0], dev); m.Lp = 4
        with torch.no_grad(), amp_ctx(amp):
            S0 = m.write(m.start(n0, dev), -1)
        mn["additive"]["0"] = float(S0[:, :DM.H_ADD].float().flatten(1).norm(dim=1).median()); mn["delta"]["0"] = float(S0[:, DM.H_ADD:].float().flatten(1).norm(dim=1).median())
    mn["def"] = "median Frobenius norm of each head type's state after pass j (0 = pure pass 0, 256 items)"
    out["mem_norm_by_pass"] = mn
    rt = {}; set_ops(m, teA[:20], teB[:20], dev); m.Lp = 4
    with torch.no_grad(), amp_ctx(amp):
        Sx = m.start(20, dev)
        for t in pass_list(m, 4):
            Sx = m.write(Sx, t)
            if t >= 0:
                rt[str(t + 1)] = [bc.ENC.decode([int(v) for v in row]) for row in m.read_greedy(Sx).cpu().numpy()]
    out["reader_text"] = {"by_pass": rt, "gold_S_j": {str(j + 1): [int(v) for v in S[:20, j]] for j in range(4)}}
    return out


def skip_eval(m, A, B, dev, amp, bs):
    """note 153-1: the step-j pass (j = 2, 3) not run (the memory passes through unchanged; later passes keep their own step tokens), over L = 4
    items with b_j != 0: row_skip_consistent = (pred == S_4 - a b_j 10^(j-1)), full_product = (pred == S_4)."""
    from boxes.multcol import to_int as ti_
    out = {}
    for j in (2, 3):
        keep = B[:, j - 1] != 0; Ak, Bk = A[keep], B[keep]; got = rollout_drow(m, Ak, Bk, dev, amp, bs, skip=(j,))[0][:, -1]
        S4 = sj_values(Ak, Bk)[:, -1]; rs = S4 - ti_(Ak) * Bk[:, j - 1] * 10 ** (j - 1)
        out["pass%d" % j] = {"row_skip_consistent": float((got == rs).mean()), "full_product": float((got == S4).mean()), "n": int(keep.sum()),
                             "n_bj0_excluded": int((~keep).sum()), "excludes_bj0": True}
    return out


@torch.no_grad()
def ext_4x5(m, dev, amp, bs, n=1000, seed=15304):
    """§153: 4-digit a x 5-digit b, K = 5 passes (step token 5 unseen in training); the 8-digit reader: exact S_5 mod 1e8 (test_acc), and the
    exact rate on the items with S_5 < 1e8."""
    from boxes.multcol import gen_operands
    rng = np.random.default_rng(seed); A = gen_operands(rng, n, 4); B = gen_operands(rng, n, 5); dec = rollout_drow(m, A, B, dev, amp, bs, L=5)[0]; S5 = sj_values(A, B)[:, 4]
    lo = S5 < 10 ** NDIG
    return {"test_acc": float((dec[:, 4] == S5 % 10 ** NDIG).mean()), "test_acc_items_lt_1e8": float((dec[lo, 4] == S5[lo]).mean()) if lo.any() else None,
            "share_lt_1e8": float(lo.mean()), "acc_by_j_low8": {str(j + 1): float((dec[:, j] == sj_values(A, B)[:, j] % 10 ** NDIG).mean()) for j in range(5)},
            "n": n, "metric": "S_5 mod 1e8 exact (the reader writes 8 digits)"}


class Teacher:
    """TC-M teacher on the fly (§139): greedy CoT for a batch, hidden at `layer` at the last token of each S_j text, decoded S_j digits."""
    SJ_END = {1: 5, 2: 19, 3: 36, 4: 56}                                    # end (exclusive) of S_j's digits in the 56 generated tokens

    def __init__(self, d, layer, probe_path, dev):
        from boxes import mult98 as M98
        self.M98 = M98; self.g = s5.load_gpt2(dev); self.g.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=dev)); self.g.eval()
        for p_ in self.g.parameters():
            p_.requires_grad_(False)
        self.layer = layer; self.dev = dev; self.kl = probe_path is not None                # CE mode (note 139-2): no probe, no hidden pass
        if self.kl:
            pr = torch.load(probe_path, map_location=dev); assert pr["layer"] == layer, "probe layer != --distill_layer"
            self.W, self.b, self.mu, self.sd = (pr[k].to(dev).float() for k in ("W", "b", "mu", "sd"))
        self.dig = {int(t): i for i, t in enumerate(M98.DIGIT_IDS)}

    @torch.no_grad()
    def targets(self, A, B, L, amp):
        """-> probe probs (bs, L, 8, 10), decoded digits (bs, L, 8) (-1 = invalid), validity (bs, L)."""
        from boxes.teacher135 import hidden_layers
        q = torch.tensor(self.M98.question({"A": A, "B": B}, np.arange(len(A))), device=self.dev)
        with amp_ctx(amp):
            last, past = self.M98.seq_forward(self.g, q, 0, None); out = self.M98.seq_greedy(self.g, last, past, 56)
            P = None
            if self.kl:
                hs = hidden_layers(self.g, torch.cat([q, out], 1), upto=self.layer)[self.layer]
                pos = torch.tensor([q.shape[1] + self.SJ_END[j] - 1 for j in range(1, L + 1)], device=self.dev)
                H = (hs[:, pos].float() - self.mu) / self.sd; P = torch.softmax((H @ self.W + self.b).view(len(A), L, NDIG, 10), -1)
        gen = out.cpu().numpy(); dec = np.zeros((len(A), L, NDIG), np.int64)
        st = {1: 0, 2: 13, 3: 29, 4: 48}
        for j in range(1, L + 1):
            seg = gen[:, st[j]:self.SJ_END[j]]; dd = np.vectorize(lambda t: self.dig.get(int(t), -1))(seg); dec[:, j - 1, :seg.shape[1]] = dd
        return P, torch.tensor(dec, device=self.dev), torch.tensor((dec >= 0).all(2), device=self.dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="mult_row", choices=["mult_row"]); ap.add_argument("--task", default="4x4", choices=["4x4"])
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--steps", type=int, default=24000); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--eval_bs", type=int, default=250); ap.add_argument("--test_n", type=int, default=1000)
    ap.add_argument("--d_q", type=int, default=16); ap.add_argument("--levels", type=int, default=8); ap.add_argument("--soft_beta", type=float, default=0.1)
    ap.add_argument("--cotenant_runs", type=int, default=1, help="record only (Audit): runs sharing this GPU (queue P), so wall-clock is not misread")
    ap.add_argument("--fixed_L", type=int, default=0, help="§139 arm A' (note 139-3): every training chain has this L (no prefix problems / curriculum); 0 = the curriculum")
    ap.add_argument("--copy_frac", type=float, default=0.1); ap.add_argument("--copy_gate_4k", action="store_true", help="§139: stop if copy_acc < 0.9 at 4k (arm a s1)")
    ap.add_argument("--sched_scale", type=float, default=1.0); ap.add_argument("--p_hard_start", type=int, default=2000); ap.add_argument("--p_hard_end", type=int, default=4000)
    ap.add_argument("--log_every", type=int, default=100); ap.add_argument("--quick_every", type=int, default=1000); ap.add_argument("--ckpt_every", type=int, default=0)
    ap.add_argument("--resume", default=None); ap.add_argument("--stop_after", type=int, default=0); ap.add_argument("--train_n", type=int, default=0, help="smoke only")
    ap.add_argument("--distill", action="store_true"); ap.add_argument("--distill_teacher", default="boxes/runs_mult/m98_tc_4x4_s1"); ap.add_argument("--distill_layer", type=int, default=None)
    ap.add_argument("--distill_probe_dir", default="boxes/runs_fsq/s139_teacher_probe"); ap.add_argument("--distill_full", type=int, default=3000); ap.add_argument("--distill_end", type=int, default=4000)
    ap.add_argument("--donor_n", type=int, default=1000); ap.add_argument("--contract_n", type=int, default=500); ap.add_argument("--mem_check", action="store_true")
    ap.add_argument("--from_args", default=None); ap.add_argument("--contraction_only", action="store_true"); ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--no_round", action="store_true", help="§142 NOISE-64: carry z_b continuously (as fsq120 §123)")
    ap.add_argument("--dither", type=float, default=0.0, help="§142: U(-s, s) on z_b at training hand-offs after 2k (with --no_round; clean at test)")
    ap.add_argument("--lmax_steps", default="", help="§142: L_max k+1 from these steps, e.g. 2000,5000,9000 (default §139's 1000,2000,3000)")
    ap.add_argument("--reader", action="store_true", help="§142: decode S_j from code_j at every hand-off j < L (teacher targets)")
    ap.add_argument("--teacher_cache", default="boxes/runs_fsq/s142_teacher_cache/teacher_sj_4x4_train.npz")
    ap.add_argument("--gate_every", type=int, default=1000); ap.add_argument("--gate_n", type=int, default=512); ap.add_argument("--gate_thr", type=float, default=0.95)
    ap.add_argument("--fork_from", default=None, help="§142 B: start from this gate.pt"); ap.add_argument("--fork_steps", type=int, default=0)
    ap.add_argument("--removal", default=None, choices=[None, "early", "late"]); ap.add_argument("--removal_every", type=int, default=4000)
    ap.add_argument("--rewarm", type=int, default=200); ap.add_argument("--preview_frac", type=float, default=0.1); ap.add_argument("--ext_n", type=int, default=1000)
    ap.add_argument("--carrier", default="code", choices=["code", "deltamem"], help="§152: deltamem = §147's DeltaMem as the only cross-segment channel (no code token)")
    ap.add_argument("--mem_layer", type=int, default=6); ap.add_argument("--mem_add_heads", type=int, default=6); ap.add_argument("--mem_delta_heads", type=int, default=6)
    ap.add_argument("--keep_ckpt_steps", default="", help="§152: model-only copies at these steps -> s152.serial_by_ckpt (donor pos2), then deleted")
    ap.add_argument("--mem_write", default=None, choices=[None, "top", "block9"], help="§160 R-ROWS: DeltaMem-R's write source (with --carrier deltamem)")
    ap.add_argument("--write_mix_start", type=int, default=2000); ap.add_argument("--write_mix_end", type=int, default=6000)
    ap.add_argument("--probe_n_fit", type=int, default=8000, help="§160 probes: train fit items (smokes only change it)")
    ap.add_argument("--s160_probe_only", action="store_true", help="§160 post-hoc (descriptive): s160 probes on <out>/model.pt -> eval_s160_posthoc.json")
    ap.add_argument("--read_mode", default=None, choices=[None, "periodic", "multiread", "ccq"], help="§157 READ-PANEL (with --carrier deltamem)")
    ap.add_argument("--answer_sep", default=None, choices=[None, "interleave"], help="§156 (b): forced '|' before each digit of the reader decode")
    ap.add_argument("--sep_token", default="|")
    ap.add_argument("--keep_ckpt_retain", action="store_true", help="note 152-5: keep the --keep_ckpt_steps copies after the eval (default: deleted)")
    ap.add_argument("--no_opt_reset", action="store_true", help="§152 B: keep AdamW's moments at each removal (LR re-warm only)")
    ap.add_argument("--pass_input", default=None, choices=[None, "visible", "pure", "repeat"], help="§153 THINK-PASS pass inputs (with --carrier deltamem)")
    ap.add_argument("--think_token", default="", help="§153 repeat: the one fixed think token's text (a single GPT-2 token)")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.from_args:
        keep = {k: getattr(a, k) for k in ("contraction_only", "fp32", "device", "out", "from_args")}
        for k, v in json.load(open(a.from_args)).items():
            if hasattr(a, k):
                setattr(a, k, v)
        for k, v in keep.items():
            setattr(a, k, v)
    assert not (a.no_round and a.soft_beta), "§142 NOISE: --no_round needs --soft_beta 0 (SOFT rounds)"
    assert not (a.dither and not a.no_round), "--dither needs --no_round"
    assert not a.fork_from or (a.reader and a.removal and a.fixed_L == 4 and a.fork_steps > 0), "§142 B: --fork_from needs --reader --removal --fixed_L 4 --fork_steps"
    assert not (a.reader and (a.distill or a.copy_frac)), "§142: the reader runs without §139's distill / copy rows"
    if a.carrier == "deltamem":                                             # §152 (378fe8e6 / note 152-1)
        assert not (a.no_round or a.dither or a.copy_frac or a.distill), "§152: DeltaMem is the only channel (--copy_frac 0; no --no_round / --dither / --distill)"
        assert (a.mem_add_heads, a.mem_delta_heads) == (DM.H_ADD, DM.H_DEL), "§152: 6 + 6 heads (boxes/deltamem.py)"
        a.mem_group_gain = True; a.seg_order = "b_first" if not a.pass_input else None
    assert not a.pass_input or a.carrier == "deltamem", "§153: --pass_input needs --carrier deltamem"
    assert a.pass_input != "repeat" or len(bc.tok(a.think_token)) == 1, "§153 repeat: --think_token must be one GPT-2 token"
    if a.pass_input:                                                        # §153 records
        a.times_token, a.register_token = " \u00d7", " _"
    assert not a.no_opt_reset or a.fork_from, "--no_opt_reset: §152 B forks only"
    assert not a.read_mode or (a.carrier == "deltamem" and not a.pass_input), "§157: --read_mode on §152's DeltaMem"
    assert not a.mem_write or (a.carrier == "deltamem" and not a.pass_input and not a.read_mode and not a.answer_sep), "§160: --mem_write on §152's DeltaMem only"
    if not a.mem_write:
        a.write_mix_start = a.write_mix_end = None
    if a.read_mode == "ccq":                                                # note 157-1: the paper's equations (2606.01294v2 Eq. 6 / 9 / 10)
        a.ccq_lambda_init_bias = -4.6; a.ccq_feature_norm = "l2 per head (additive: phi = elu + 1, then l2, in write and read)"
    if a.answer_sep:                                                        # §156 (b): one token, no merge with ' d' on re-encoding
        assert a.carrier == "deltamem" and not a.pass_input, "§156 (b): --answer_sep on §152's reader"
        st_ = bc.tok(a.sep_token); assert len(st_) == 1, "--sep_token must be one GPT-2 token"; smp = []
        for d_ in range(10):
            smp += [st_[0], DTOK[d_]]
        assert bc.ENC.encode(bc.ENC.decode(bc.tok(" =") + smp)) == bc.tok(" =") + smp, "§156: separator / digit tokens merge on re-encoding"
        a.sep_token_id = st_[0]
    else:
        a.sep_token = None
    if a.fork_from:
        a.steps = a.fork_steps                                              # the fork's own schedule (scorer reads fork_steps)
    th = tuple(int(x) for x in a.lmax_steps.split(",")) if a.lmax_steps else None; assert th is None or len(th) == 3, "--lmax_steps: 3 steps"
    a.label_source = "teacher_generated" if a.reader else None
    dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); sc = a.sched_scale
    if a.d_q != F.D_Q or a.levels != F.LEV:
        F.D_Q, F.LEV = a.d_q, a.levels; F.HALF = (F.LEV - 1) / 2; F.SHIFT = math.atanh(F.OFF / F.HALF)
    memcap and memcap.apply(dev)
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    os.makedirs(a.out, exist_ok=True); fam = RowFam(); dname = lambda: torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"  # noqa: E731
    teA, teB = load_bench(os.path.join(ROOT, "bench", "data", "mult_4x4_test.txt")); teA, teB = teA[:a.test_n], teB[:a.test_n]
    def mk():
        m_ = (MultRowDelta(s5.load_gpt2(dev), fam, a.mem_layer) if a.carrier == "deltamem" else MultRow(s5.load_gpt2(dev), fam)).to(dev); m_.soft_beta = a.soft_beta; m_.no_round = a.no_round
        if a.answer_sep:                                                    # §156 (b)
            m_.sep = bc.tok(a.sep_token)[0]; m_.sep_mode = a.answer_sep
        if a.mem_write:                                                     # §160 R-ROWS
            m_.mem = DM.DeltaMem(m_.g, a.mem_layer, a.mem_write).to(dev)
        if a.read_mode == "periodic":                                       # §157
            m_.mem = PeriodicMem(m_.g, a.mem_layer).to(dev)
        if a.read_mode == "multiread":
            m_.mem.mr = MultiRead(dev)
        if a.read_mode == "ccq":
            m_.mem = CCQMem(m_.g, a.mem_layer).to(dev)
        if a.pass_input:                                                    # §153
            m_.pass_input = a.pass_input; m_.think_tok = bc.tok(a.think_token)[0] if a.pass_input == "repeat" else None
        if a.distill:
            m_.readout = nn.Linear(768, NDIG * 10).to(dev)
        return m_
    if a.contraction_only:
        m = mk(); m.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev)); m.eval(); m.p_hard = 1.0; ampc = None if a.fp32 else amp
        ct = contraction(m, teA, teB, dev, ampc, n=a.contract_n); ct.update({"eval_dtype": str(ampc), "device": dname(), "driver": driver_version()})
        json.dump(ct, open(os.path.join(a.out, "eval_contraction_fp32.json" if a.fp32 else "eval_contraction.json"), "w", newline="\n"), indent=1)
        print("CONTRACTION", json.dumps({k: ct[k] for k in ("rho", "rho_nonzero", "basin")}), flush=True); return
    if a.train_n:
        from boxes.multcol import load_bench_head
        trA, trB = load_bench_head(os.path.join(ROOT, "bench", "data", "mult_train", "4x4_train.txt"), a.train_n)
    else:
        trA, trB = load_bench(os.path.join(ROOT, "bench", "data", "mult_train", "4x4_train.txt"))
    n_raw = len(trA); trA, trB, n_rem = dedup(trA, trB, teA, teB)
    if a.s160_probe_only:                                                   # §160 post-hoc (descriptive): the s160 probes on a trained model (e.g. §152 A s1's null)
        m = mk(); m.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev)); m.eval(); m.p_hard = 1.0
        o_ = {"s160": s160_probes(m, teA, teB, trA, trB, dev, amp, a.eval_bs, n_fit=a.probe_n_fit), "model": os.path.basename(a.out.rstrip("/")), "mem_write": a.mem_write,
              "eval_dtype": str(amp), "device": dname(), "driver": driver_version()}
        json.dump(o_, open(os.path.join(a.out, "eval_s160_posthoc.json"), "w", newline="\n"), indent=1); print("S160POST", json.dumps(o_["s160"]["c1_edit"]["pooled"]), flush=True); return
    rdec = rok = None; s142 = {}
    if a.reader:                                                            # §142 teacher targets: the sha-keyed cache (note 142-1)
        cp_ = a.teacher_cache if os.path.isabs(a.teacher_cache) else os.path.join(ROOT, a.teacher_cache); z = np.load(cp_)
        ssha = split_sha(trA, trB); assert str(z["split_sha"]) == ssha, "teacher cache split sha %s != this split %s" % (str(z["split_sha"])[:12], ssha[:12])
        tpath = os.path.join(ROOT, a.distill_teacher, "model.pt")
        if os.path.exists(tpath):                                           # the teacher is local (not shipped to Vast hosts): check its sha too
            import hashlib
            hh = hashlib.sha256()
            with open(tpath, "rb") as f_:
                for blk in iter(lambda: f_.read(1 << 22), b""):
                    hh.update(blk)
            assert hh.hexdigest() == str(z["teacher_sha"]), "teacher cache teacher sha mismatch"
        rdec = z["dec"].astype(np.int64); rok = z["ok"].astype(bool); gold = digits8(sj_values(trA, trB)); mism = rok & (rdec != gold).any(-1)
        s142.update({"teacher_invalid_share": float(1 - rok.mean()), "teacher_vs_gold_mismatch_share": float(mism.sum() / max(rok.sum(), 1)),
                     "cache_split_sha": ssha, "cache_teacher_sha": str(z["teacher_sha"])})
        assert s142["teacher_invalid_share"] <= 0.01, "139-2 gate: teacher format validity < 0.99 over the cache - stop and tell Theory"
        a.teacher_cache_sha = str(z["teacher_sha"])[:16] + "/" + ssha[:16]; del gold, mism
    teacher = None; target_mode = None; probe_info = None
    if a.distill:
        pj = json.load(open(os.path.join(ROOT, a.distill_probe_dir, "teacher_probe.json"))); a.distill_layer = a.distill_layer if a.distill_layer is not None else pj["chosen_layer"]
        target_mode = "kl" if pj["probe_vs_teacher_digit_agree"] >= 0.99 else "ce"; probe_info = {k: pj.get(k) for k in ("chosen_layer", "gate", "probe_vs_teacher_digit_agree", "probe_vs_gold_exact")}
        teacher = Teacher(os.path.join(ROOT, a.distill_teacher), a.distill_layer,
                          os.path.join(ROOT, a.distill_probe_dir, "probe_L%d.pt" % a.distill_layer) if target_mode == "kl" else None, dev)
    info = {**vars(a), "train_items": int(len(trA)), "train_items_raw": int(n_raw), "train_removed_test_overlap": n_rem, "target_mode": target_mode, "teacher_probe": probe_info,
            "train_gpu": dname(), "train_driver": driver_version(), **env_info()}
    json.dump({**info, "resumed_at": time.strftime("%Y-%m-%d %H:%M:%S")} if a.resume else info,
              open(os.path.join(a.out, ("args_resume_%d.json" % int(time.time())) if a.resume else "args.json"), "w", newline="\n"), indent=1)
    m = mk(); opt = opt_drow(m, a.lr) if a.carrier == "deltamem" else make_opt(m, a.lr); m.train()
    gate_step = gate_exact = None; fork_start = None; stages = []; stage_start = 1; order = None; alpha_hist = []; fin_hist = {}; wwin = []; wlast = []; wg_hist = []
    keep_steps = {int(x) for x in a.keep_ckpt_steps.split(",") if x.strip()}
    if a.fork_from:                                                         # §142 B: the A run's gate model, same seed
        gk = torch.load(a.fork_from if os.path.isabs(a.fork_from) else os.path.join(ROOT, a.fork_from), map_location=dev, weights_only=False)
        m.load_state_dict(gk["model"]); s142["fork_gate_step"] = gk.get("step"); s142["fork_gate_train_exact"] = gk.get("gate_train_exact")
        order = (1, 2, 3) if a.removal == "early" else (3, 2, 1)
    if a.carrier == "deltamem" and not (a.resume or a.mem_check or a.fork_from):   # note 147-2: g_add = the median read-norm ratio at step 0
        rr0, med0 = read_ratio_drow(m, teA, teB, dev, amp); m.mem.s_add.data.fill_(math.log(med0)); rr1, med1 = read_ratio_drow(m, teA, teB, dev, amp)
        init152 = {"g_add_init": med0, "g_del_init": 1.0, "ratio_by_handoff_before_gain": rr0, "read_norm_ratio_by_handoff_step0": rr1, "pooled_median_after_gain": med1}
        json.dump(init152, open(os.path.join(a.out, "s152_init.json"), "w", newline="\n"), indent=1); print("S152_INIT", json.dumps(init152), flush=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a" if a.resume else "w", newline="\n"); t0 = time.time(); hist = []; chist = []; dhist = []; L127 = {}; n_nonfinite = 0; step0 = 1
    Ls = []; crng = np.random.default_rng(a.seed + 13400); first_exact = None; n_copy = int(round(a.copy_frac * a.bs)) if a.copy_frac else 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False); m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["np_rng"]; crng.bit_generator.state = ck["crng"]; torch.set_rng_state(ck["torch_rng"].cpu()); L127 = ck["L127"]; n_nonfinite = ck["n_nonfinite"]
        first_exact = ck.get("first_exact"); gate_step = ck.get("gate_step"); gate_exact = ck.get("gate_exact")   # §142 state
        fork_start = ck.get("fork_start"); stages = ck.get("stages") or []; stage_start = ck.get("stage_start", 1)
        alpha_hist = ck.get("alpha_hist") or []; fin_hist = ck.get("fin_hist") or {}                     # §152
        if dev.startswith("cuda") and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device=dev)
        step0 = ck["step"] + 1; t0 = time.time() - ck["elapsed"]; log.write(json.dumps({"step": ck["step"], "resumed": a.resume}) + "\n"); log.flush()
    def w_at(step):
        return distill_weight(step, a.distill, a.distill_full, a.distill_end, sc)
    def reader_mask(ii, L, removed=(), preview=None):
        """§142: (bs, L - 1) rows / hand-offs with a reader term: teacher-valid, not removed; preview (j, rows) also drops j on those rows."""
        mk_ = rok[ii][:, :L - 1].copy()
        for j in removed:
            if j <= L - 1:
                mk_[:, j - 1] = False
        if preview is not None and preview[0] <= L - 1:
            mk_[preview[1], preview[0] - 1] = False
        return mk_
    def train_step(step, L, A, B, w, do_step=True, ii=None, rmask=None):
        set_ops(m, A, B, dev); X = torch.tensor(np.tile(np.arange(L), (len(A), 1)), device=dev); lab = sj_values(A, B)[:, :L]
        if a.reader:                                                        # §142: CE_final (gold S_L) + mean over hand-offs of CE_j (teacher S_j)
            rtok = torch.tensor(np.array(DTOK)[rdec[ii][:, :max(L - 1, 1)].clip(0)], device=dev)
            with amp_ctx(amp):
                fin, rd, nrd = m.chain_read(m.start(len(A), dev), X, lab, rtok, torch.tensor(rmask, device=dev) if L > 1 else None)
            return fin, rd
        if a.carrier == "deltamem":                                         # §152 CTRL: FINAL only, through the reader pass
            with amp_ctx(amp):
                fin, _, _ = m.chain_read(m.start(len(A), dev), X, lab, None, None)
            return fin, None
        m.tsd_on = w > 0; m._tsd_e = []
        with amp_ctx(amp):
            loss, _, _, _ = m.chain(m.start(len(A), dev), X, lab)            # FINAL only: CE on S_L after the last segment
        aux = None
        if w > 0:                                                           # §139 readout on e_j = W_up(code_j) vs the teacher target
            P, dec, ok = teacher.targets(A, B, L, amp); E = torch.stack(m._tsd_e, 1).float(); lg = m.readout(E).view(len(A), L, NDIG, 10).float()
            if target_mode == "kl":
                aux = (Fn.kl_div(Fn.log_softmax(lg, -1), P, reduction="none").sum(-1).mean(-1) * ok).sum() / ok.sum().clamp_min(1)
            else:
                ce = Fn.cross_entropy(lg.reshape(-1, 10), dec.clamp_min(0).reshape(-1), reduction="none").view(len(A), L, NDIG).mean(-1)
                aux = (ce * ok).sum() / ok.sum().clamp_min(1)
        m.tsd_on = False; m._tsd_e = []
        return loss, aux
    if a.mem_check:
        m.p_hard = 1.0; m.noise = a.dither if a.no_round else 0.0; ii = rng.integers(0, len(trA), a.bs)
        loss, aux = train_step(1, 4, trA[ii], trB[ii], 1.0 if a.distill else 0.0, ii=ii, rmask=np.ones((a.bs, 3), bool) if a.reader else None)
        (loss + (aux if aux is not None else 0)).backward(); m.eval(); rollout(m, teA[:a.eval_bs], teB[:a.eval_bs], dev, amp, a.eval_bs, record=True)
        pk = torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.startswith("cuda") else None
        json.dump({"peak_gib": pk, "bs": a.bs, "eval_bs": a.eval_bs, "distill": a.distill, "reader": a.reader, "L": 4, "carrier": a.carrier}, open(os.path.join(a.out, "mem_check.json"), "w", newline="\n"), indent=1)
        print("MEM", pk); return
    def stage_eval():                                                       # §142 B: fork_start / stage records
        was = m.training; m.eval(); nz = m.noise; m.noise = 0.0; d_ = rollout(m, teA, teB, dev, amp, a.eval_bs); S_ = sj_values(teA, teB)
        r_ = {"test_acc": float((d_[:, -1] == S_[:, -1]).mean()), "acc_by_length": {str(L_ + 1): float((d_[:, L_] == S_[:, L_]).mean()) for L_ in range(4)},
              "donor": (donor_drow if a.carrier == "deltamem" else donor_step2)(m, teA, teB, dev, amp, a.eval_bs, np.random.default_rng(13902), n=a.donor_n)}
        if a.carrier == "deltamem":                                         # §152 B: removed S_j stay decoded (retention), no loss term
            r_["sj_decode_acc"] = {str(j + 1): r_["acc_by_length"][str(j + 1)] for j in range(3)}
        m.noise = nz; m.train(was); return r_
    gi = np.random.default_rng(14201).permutation(len(trA))[:a.gate_n]       # §142 gate: a fixed 512-item TRAIN subset at L 4
    re_ = int(a.removal_every * sc)
    if a.fork_from and fork_start is None:
        fork_start = stage_eval(); log.write(json.dumps({"step": 0, "fork_start": fork_start}) + "\n"); log.flush()
    for step in range(step0, a.steps + 1):
        removed, preview = (), None
        if a.fork_from:                                                     # §142 B: stage k removes order[:k + 1]; fresh AdamW + re-warm-up at each removal
            stg = min(2, (step - 1) // re_); removed = order[:stg + 1]
            if step > 1 and (step - 1) % re_ == 0 and (step - 1) // re_ <= 2 and step != stage_start:
                if not a.no_opt_reset:                                      # §152 B (--no_opt_reset): AdamW's moments kept, LR re-warm only
                    opt = opt_drow(m, a.lr) if a.carrier == "deltamem" else make_opt(m, a.lr)
                stage_start = step
            for gr in opt.param_groups:
                gr["lr"] = a.lr * 0.5 * (1 + math.cos(math.pi * step / a.steps)) * min(1.0, (step - stage_start + 1) / (a.rewarm * sc))
        else:
            for gr in opt.param_groups:
                gr["lr"] = lr_at(a.lr, gr, step, a.steps)
        m.p_hard = 0.0 if step <= a.p_hard_start * sc else min(1.0, (step - a.p_hard_start * sc) / ((a.p_hard_end - a.p_hard_start) * sc)); m.train()
        if a.carrier == "deltamem" and not a.fork_from:
            m.mem.fade.fill_(min(1.0, step / max(1, int(500 * sc))))         # §147: W_o fade-in over 500 steps (a fork starts from gate.pt's fade 1)
            if a.mem_write:                                                 # §160: gamma 0 until write_mix_start, linear to 1 at write_mix_end
                m.mem.gamma.fill_(0.0 if step <= a.write_mix_start * sc else min(1.0, (step - a.write_mix_start * sc) / max(1.0, (a.write_mix_end - a.write_mix_start) * sc)))
            if getattr(m.mem, "mr", None) is not None:                      # §157 MULTIREAD: the read tokens' attention faded in over 500 steps
                m.mem.mr.fade.fill_(min(1.0, step / max(1, int(500 * sc))))
        m.noise = (a.dither if (a.fork_from or step > 2000 * sc) else 0.0) if a.no_round else 0.0   # §142 NOISE: dither after 2k (the fork is past 2k)
        L = a.fixed_L if a.fixed_L else int(rng.integers(1, lmax(step / sc, th) + 1)); Ls.append(L); ii = rng.integers(0, len(trA), a.bs - n_copy); w = w_at(step)
        rmask = None
        if a.reader:
            if a.fork_from and len(removed) < 3:                            # ICoT-SI smoothing: 10% of rows also drop the next target
                pr_ = np.nonzero(rng.random(len(ii)) < a.preview_frac)[0]; preview = (order[len(removed)], pr_)
            rmask = reader_mask(ii, L, removed, preview)
        opt.zero_grad(); loss, aux = train_step(step, L, trA[ii], trB[ii], w, ii=ii, rmask=rmask)
        tot_loss = loss + ((aux if a.reader else w * aux) if aux is not None else 0)
        ctot = 0.0
        if n_copy:                                                          # §134 copy rows: another problem's S_L text -> copy writer -> decoder
            cv = sj_values(trA[crng.integers(0, len(trA), n_copy)], trB[crng.integers(0, len(trA), n_copy)])[:, -1]; cans = torch.tensor(fam.answer_tokens(cv), device=dev)
            with amp_ctx(amp):
                ce_, cqs_, _, _ = m.copy_write(cans); lc = m.answer_ce(ce_, cans, cqs_)
            tot_loss = tot_loss + lc; ctot = float(lc.item())
        tot_loss.backward(); tot = float(loss.item())
        if a.mem_write:                                                     # §160 (155-1's fallback rule): the write rows' grad norm before clipping
            from boxes.multpair import write_grad_norm
            wwin.append(write_grad_norm(m))
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if not (math.isfinite(tot) and math.isfinite(ctot) and bool(torch.isfinite(gn)) and (aux is None or math.isfinite(float(aux.item())))):   # 127-3 guard
            bad = "loss" if not math.isfinite(tot) else next((n_ for n_, p_ in m.named_parameters() if p_.grad is not None and not bool(torch.isfinite(p_.grad).all())), "grad_norm")
            log.write(json.dumps({"step": step, "nonfinite": 1, "first": bad}) + "\n"); log.flush(); n_nonfinite += 1; opt.zero_grad()
        else:
            opt.step(); hist.append(tot); chist.append(ctot)
            if a.fork_from and a.carrier == "deltamem":
                fin_hist[step] = tot                                        # §152 B: final CE either side of each removal
            if aux is not None:
                dhist.append(float(aux.item()))
        if step % a.log_every == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(hist)) if hist else None, "L_max": a.fixed_L or lmax(step / sc, th), "L_drawn": sorted(set(Ls)), "p_hard": m.p_hard,
                   "sec": round(time.time() - t0)}; Ls = []
            if n_copy:
                rec["copy_loss"] = float(np.mean(chist)) if chist else None
            if a.distill:
                rec.update({"distill_w": w, "distill_loss": float(np.mean(dhist)) if dhist else None})
            if a.reader:                                                    # §142: the reader term (mean over hand-offs); "loss" stays CE_final
                rec.update({"reader_loss": float(np.mean(dhist)) if dhist else None, "noise": m.noise})
            if a.carrier == "deltamem":                                     # note 147-2: the read gains per window
                rec.update({"g_add": float(torch.exp(m.mem.s_add)), "g_del": float(torch.exp(m.mem.s_del)), "fade": float(m.mem.fade)})
            if a.mem_write and wwin:                                        # §160: per window; spike vs the preceding 1k's median, scored after write_mix_start
                md_ = float(np.median(wlast)) if wlast else 0.0
                w_rec = {"step": step, "mean": float(np.mean(wwin)), "max": float(np.max(wwin)), "median_prev_1k": md_,
                         "spike_100x": bool(step > a.write_mix_start * sc and np.max(wwin) > 100 * md_ > 0), "spike_100x_raw": bool(np.max(wwin) > 100 * md_ > 0),
                         "spike_100x_floor": bool(step > a.write_mix_start * sc and np.max(wwin) > 100 * md_ > 0 and np.max(wwin) > 1.0)}   # note 160-1 (33ec47e5): the scored trigger
                rec.update({"gamma": float(m.mem.gamma), "write_grad_norm": w_rec["mean"], "write_grad_spike_100x": w_rec["spike_100x"], "write_grad_spike_100x_floor": w_rec["spike_100x_floor"]}); wg_hist.append(w_rec)
                wlast.extend(wwin); del wlast[:-int(1000 * sc) or None]; wwin = []
                if a.fork_from:
                    rec.update({"removed": list(removed), "lr": opt.param_groups[0]["lr"]})
            if n_copy and (step % 500 == 0 or step == a.steps):
                m.eval(); rec["copy_acc"] = copy_acc(m, dev, amp, np.random.default_rng(13901)); m.train()
            hist, chist, dhist = [], [], []; log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True); L127[step] = rec["loss"]
            if a.copy_gate_4k and step == int(4000 * sc) and n_copy and (rec.get("copy_acc") or 0) < 0.9:
                log.write(json.dumps({"step": step, "stopped": "copy_acc < 0.9 at 4k (§139 base check)", "copy_acc": rec.get("copy_acc")}) + "\n"); log.close()
                print("STOPPED copy gate", flush=True); return
        if not a.fork_from and step % int(a.gate_every * sc) == 0 and lmax(step / sc, th) == 4 and not a.fixed_L:   # §142 removal gate (A)
            m.eval(); nz_ = m.noise; m.noise = 0.0; gx = float((rollout(m, trA[gi], trB[gi], dev, amp, a.eval_bs)[:, -1] == sj_values(trA[gi], trB[gi])[:, -1]).mean())
            m.noise = nz_; m.train(); log.write(json.dumps({"step": step, "gate_train_exact": gx}) + "\n"); log.flush()
            if gate_step is None and gx >= a.gate_thr:
                gate_step, gate_exact = step, gx
                torch.save({"model": m.state_dict(), "step": step, "gate_train_exact": gx, "seed": a.seed}, os.path.join(a.out, "gate.pt"))
                log.write(json.dumps({"step": step, "gate_open": True, "gate_train_exact": gx}) + "\n"); log.flush()
        if a.fork_from and step % re_ == 0 and step // re_ in (1, 2):       # §142 B: stage record at each removal boundary
            st_ = {"step": step, "removed": list(removed), **stage_eval()}; stages.append(st_); log.write(json.dumps({"stage": st_}) + "\n"); log.flush()
        if a.quick_every and step % int(a.quick_every * sc) == 0:            # first_exact_step: 256 test items at L = 4
            m.eval(); qa = float((rollout(m, teA[:256], teB[:256], dev, amp, 256)[:, -1] == sj_values(teA[:256], teB[:256])[:, -1]).mean()); m.train()
            log.write(json.dumps({"step": step, "quick_test_acc": qa}) + "\n"); log.flush()
            if first_exact is None and qa >= 0.99:
                first_exact = step
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps:
            if a.carrier == "deltamem":                                     # note 147-3: alpha per head at each checkpoint
                ah_ = {"step": step, **alpha_drow(m, teA, teB, dev, amp)}; alpha_hist.append(ah_); log.write(json.dumps({"step": step, "alpha_by_head": ah_}) + "\n"); log.flush()
            cp = os.path.join(a.out, "ckpt_last.pt")
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(dev) if dev.startswith("cuda") else None,
                        "np_rng": rng.bit_generator.state, "crng": crng.bit_generator.state, "step": step, "L127": L127, "n_nonfinite": n_nonfinite,
                        "first_exact": first_exact, "elapsed": time.time() - t0, "gate_step": gate_step, "gate_exact": gate_exact,
                        "fork_start": fork_start, "stages": stages, "stage_start": stage_start, "alpha_hist": alpha_hist, "fin_hist": fin_hist}, cp + ".tmp")
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            os.replace(cp + ".tmp", cp); log.write(json.dumps({"step": step, "ckpt": "ckpt_last.pt"}) + "\n"); log.flush()
        if step in keep_steps:                                              # §152: model-only copy for the end-of-run serial_by_ckpt
            torch.save(m.state_dict(), os.path.join(a.out, "ckpt_s%d_model.pt" % step)); log.write(json.dumps({"step": step, "ckpt_keep": "ckpt_s%d_model.pt" % step}) + "\n"); log.flush()
        if a.stop_after and step >= a.stop_after:
            log.close(); print("STOP_AFTER", step, flush=True); return
    train_sec = round(time.time() - t0); torch.save(m.state_dict(), os.path.join(a.out, "model.pt")); m.eval(); m.p_hard = 1.0; t1 = time.time()
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    if a.carrier == "deltamem":                                             # §152 eval (no contraction: an integrator by design)
        S = sj_values(teA, teB); dec = rollout(m, teA, teB, dev, amp, a.eval_bs); ti = np.random.default_rng(13900).integers(0, len(trA), 1000)
        res = {"family": a.family, "task": a.task, "seed": a.seed, "carrier": a.carrier, "test_acc": float((dec[:, -1] == S[:, -1]).mean()),
               "acc_by_length": {str(L + 1): float((dec[:, L] == S[:, L]).mean()) for L in range(4)},
               "train_acc": float((rollout(m, trA[ti], trB[ti], dev, amp, a.eval_bs)[:, -1] == sj_values(trA[ti], trB[ti])[:, -1]).mean()), "train_sec": train_sec,
               "final_loss": L127[max(L127)] if L127 else None, "n_nonfinite": n_nonfinite, "train_removed_test_overlap": n_rem, "eval_dtype": str(amp), "gpu": dname(),
               "train_gpu": info["train_gpu"], "train_driver": info["train_driver"]}
        res["long_acc"] = res["test_acc"]; res["donor"] = donor_drow(m, teA, teB, dev, amp, a.eval_bs, np.random.default_rng(13902), n=a.donor_n)
        res["ext_4x8_low8"] = ext_4x8_low8(m, dev, amp, a.eval_bs, n=a.ext_n); res["compute"] = compute_drow(m)
        Cd = digits8(dec[:, -1].clip(0)) == digits8(S[:, -1]); Cd[dec[:, -1] < 0] = False                      # §156: per digit column of S_4
        res["acc_by_column"] = {"per_column": [round(float(v), 4) for v in Cd.mean(0)], "cumulative": [round(float(Cd[:, :k + 1].all(1).mean()), 4) for k in range(NDIG)]}
        if a.answer_sep:
            res["s156"] = s156_row_eval(m, teA, teB, dev, amp, a.eval_bs)
        if a.read_mode:                                                     # §157
            res["s157"] = s157_eval(m, teA, teB, trA, trB, dev, amp, a.eval_bs, a)
        if a.mem_write:                                                     # §160
            res["s160"] = {**s160_probes(m, teA, teB, trA, trB, dev, amp, a.eval_bs, n_fit=a.probe_n_fit), "mem_write": a.mem_write,
                           "write_mix": [a.write_mix_start, a.write_mix_end], "gamma_final": float(m.mem.gamma), "write_grad_norm": wg_hist,
                           "write_grad_spikes": [x_ for x_ in wg_hist if x_.get("spike_100x")],
                           "write_grad_spikes_floor": [x_ for x_ in wg_hist if x_.get("spike_100x_floor")]}
        if a.pass_input:                                                    # §153: the pass-skip SERIAL test (note 153-1) and 4x5 at K = 5
            res["skip"] = skip_eval(m, teA, teB, dev, amp, a.eval_bs); res["ext_4x5"] = ext_4x5(m, dev, amp, a.eval_bs, n=a.ext_n)
        ip_ = os.path.join(a.out, "s152_init.json"); i152 = json.load(open(ip_)) if os.path.exists(ip_) else {}
        s152 = {k: s142.get(k) for k in ("teacher_invalid_share", "teacher_vs_gold_mismatch_share", "cache_split_sha", "cache_teacher_sha")}
        s152.update({k: v for k, v in s142.items() if k.startswith("fork_gate")})
        s152.update({"gate_step": gate_step, "gate_train_exact": gate_exact, "label_source": a.label_source, "reader": a.reader, "seg_order": a.seg_order,
                     "removal": a.removal, "no_opt_reset": a.no_opt_reset, "lmax_steps": a.lmax_steps, "g_add_init": i152.get("g_add_init"),
                     "g_add": float(torch.exp(m.mem.s_add)), "g_del": float(torch.exp(m.mem.s_del)), "gain_param": "exp(s)",
                     "read_norm_ratio_by_handoff": {"step0": i152.get("read_norm_ratio_by_handoff_step0"), "end": read_ratio_drow(m, teA, teB, dev, amp)[0]},
                     "alpha_by_head": {"end": alpha_drow(m, teA, teB, dev, amp), "checkpoints": alpha_hist}})
        s152.update(s153_eval(m, teA, teB, dev, amp, a.eval_bs, a, dec) if a.pass_input else s152_eval(m, teA, teB, dev, amp, a.eval_bs, a, dec))
        if not a.pass_input:                                                # note 152-5 (descriptive): exact column-sum storage, final model
            s152["colsum_exact_by_column"] = {str(a.steps): colsum_exact(m, teA, teB, dev, amp, a.eval_bs)}
        if a.pass_input:
            s152.update({"pass_input": a.pass_input, "think_token": a.think_token or None})
        if a.fork_from:
            stages.append({"step": a.steps, "removed": list(order), **stage_eval()})
            for k_, st_ in enumerate(stages):                               # CE either side of the removal that st_'s last-removed S_j came from
                r0 = k_ * re_ + 1; bf = [fin_hist[t_] for t_ in range(r0 - 50, r0) if t_ in fin_hist]; af = [fin_hist[t_] for t_ in range(r0, r0 + 50) if t_ in fin_hist]
                st_.update({"removal_step": r0, "final_ce_before": float(np.mean(bf)) if bf else None, "final_ce_after": float(np.mean(af)) if af else None})
            s152.update({"fork_start": fork_start, "stages": stages})
        kept_ = []
        if keep_steps:                                                      # §152: donor pos2 on the kept copies (24k / 36k) and the final model
            sb_ = {}
            for st_ in sorted(keep_steps):
                cp_ = os.path.join(a.out, "ckpt_s%d_model.pt" % st_)
                if os.path.exists(cp_):
                    m.load_state_dict(torch.load(cp_, map_location=dev)); m.eval(); kept_.append(os.path.basename(cp_))
                    sb_[str(st_)] = donor_drow(m, teA, teB, dev, amp, a.eval_bs, np.random.default_rng(13902), n=a.donor_n)["pos2"]
                    if not a.pass_input:                                    # note 152-5: the same readout on each kept copy
                        s152["colsum_exact_by_column"][str(st_)] = colsum_exact(m, teA, teB, dev, amp, a.eval_bs)
                else:
                    sb_[str(st_)] = None
            sb_[str(a.steps)] = res["donor"]["pos2"]; s152["serial_by_ckpt"] = sb_
        res["s153" if a.pass_input else "s152"] = s152; res["eval_sec"] = round(time.time() - t1)
        json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
        for ck_ in ["ckpt_last.pt", "ckpt_prev.pt"] + ([] if a.keep_ckpt_retain else kept_):   # retention: kept copies deleted unless --keep_ckpt_retain
            if os.path.exists(os.path.join(a.out, ck_)):
                os.remove(os.path.join(a.out, ck_))
        print("EVAL", json.dumps({k: res[k] for k in ("test_acc", "acc_by_length")}), flush=True); print("DONE", a.out, flush=True); return
    S = sj_values(teA, teB); dec, codes, E = rollout(m, teA, teB, dev, amp, a.eval_bs, record=True)
    ti = np.random.default_rng(13900).integers(0, len(trA), 1000)
    res = {"family": a.family, "task": a.task, "seed": a.seed, "test_acc": float((dec[:, -1] == S[:, -1]).mean()), "acc_by_length": {str(L + 1): float((dec[:, L] == S[:, L]).mean()) for L in range(4)},
           "train_acc": float((rollout(m, trA[ti], trB[ti], dev, amp, a.eval_bs)[:, -1] == sj_values(trA[ti], trB[ti])[:, -1]).mean()), "train_sec": train_sec,
           "final_loss": L127[max(L127)] if L127 else None, "n_nonfinite": n_nonfinite, "train_removed_test_overlap": n_rem, "eval_dtype": str(amp), "gpu": dname(),
           "train_gpu": info["train_gpu"], "train_driver": info["train_driver"]}
    res["long_acc"] = res["test_acc"]; res["donor"] = donor_step2(m, teA, teB, dev, amp, a.eval_bs, np.random.default_rng(13902), n=a.donor_n)
    ct = contraction(m, teA, teB, dev, amp, n=a.contract_n); res.update({"rho": ct["rho"], "rho_nonzero": ct["rho_nonzero"], "basin": ct["basin"], "contraction": ct["detail"]})
    h = len(teA) // 2; bpre = np.stack([to_int(teB[:, :j]) for j in range(1, 5)], 1)
    pr = {"sj": {}, "bprefix": {}}
    for j in range(4):                                                      # ridge probes on code_j: S_j vs the b-prefix identity (held-out half)
        pr["sj"][str(j + 1)] = ridge_r2(codes[:h, j], np.log10(1 + S[:h, j]), codes[h:, j], np.log10(1 + S[h:, j]))
        pr["bprefix"][str(j + 1)] = ridge_r2(codes[:h, j], np.log10(1 + bpre[:h, j]), codes[h:, j], np.log10(1 + bpre[h:, j]))
    ro = None
    if m.readout is not None:                                               # the frozen readout on the end codes (e_j), exact S_j digits
        with torch.no_grad():
            lg = m.readout(torch.tensor(E, device=dev)).view(len(E), 4, NDIG, 10).argmax(-1).cpu().numpy()
        ro = float((lg == digits8(S)).all(-1).mean())
    res["s139"] = {"target_mode": target_mode, "first_exact_step": first_exact, "readout_sj_acc_end": ro, "probe_r2": pr, "copy_acc": copy_acc(m, dev, amp, np.random.default_rng(13903)) if n_copy else None,
                   "distill": a.distill, "distill_layer": a.distill_layer, "distill_full": a.distill_full, "distill_end": a.distill_end, "teacher_probe": probe_info}
    if a.reader or a.no_round or a.fork_from or a.lmax_steps:               # §142 block (scorer: s142.*, ext_4x8)
        if a.fork_from:
            stages.append({"step": a.steps, "removed": list(order), **stage_eval()})
        s142.update({"gate_step": gate_step, "gate_train_exact": gate_exact, "fork_start": fork_start, "stages": stages or None, "probe_r2": pr,
                     "label_source": a.label_source, "reader": a.reader, "removal": a.removal, "lmax_steps": a.lmax_steps})
        res["s142"] = s142; res["ext_4x8"] = ext_4x8_low8(m, dev, amp, a.eval_bs, n=a.ext_n)
    res["eval_sec"] = round(time.time() - t1)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):
        if os.path.exists(os.path.join(a.out, ck_)):
            os.remove(os.path.join(a.out, ck_))
    print("EVAL", json.dumps({k: res[k] for k in ("test_acc", "acc_by_length", "rho")}), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
