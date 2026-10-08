"""THEORY §140 DIGPAIR (ce1b3442): multiplication one digit pair per segment on §120's FSQ carrier. fsq120's model unchanged (FSQ120: FSQ d 8 /
L 8, W_up / W_down / LN, the answer decoder) and §128's recipe (p_hard 2000 -> 4000, 24k, bs 64, lr 1e-4, AdamW wd 0.01, clip 1.0, the
127-3 guard; --stop_plateau = 123-3's 2k flag for restart slots via scratch/restart140.py; no COPY). Steps: one segment per digit pair (i, j)
in output-column order (column k = i + j - 1 ascending, i ascending within a column), then one flush segment: n^2 + 1 steps.
    pair segment  = [EOT, carried code, a_i, b_j, column flag]   (note 140-1: no index tokens; only those two digits are visible)
                    column flag " ;" = the first pair of a new column (take acc // 10 first), " ," = the same column
    flush segment = [EOT, carried code, " ="]                     (the op is identical at every position: 4x8 has no unseen tokens)
The carried quantity is the column accumulator: acc after a pair = (carry into its column) + the column's products so far. At the last pair
of column k the answer decoder reads the code and emits p_k = acc mod 10 (CE); the next column starts from acc // 10; the flush emits
p_2n = the final carry. Nothing is emitted at other steps. Curriculum: chains over the pairs of the first c columns (+ the flush when c = 2n),
targets = the low c digits, c ~ U(1, c_max), c_max 1 / 2 / 4 / 8 at 0 / 2k / 5k / 9k (4x4); 1 / 2 / 4 / 8 / 10 at 0 / 2k / 5k / 9k / 12k (5x5, note 140-2; 10 = 9 columns + the flush).
eval.json: test_acc (all 2n digits exact on the 1,000 bench items), acc_by_column {per_column, cumulative}, train_acc, donor (pos4 / pos8 /
pos12: the code after pair t from a donor; the recipient's later digits with the donor's accumulator, exact; digit-level donor /
recipient / chance per position + row level), s140_probe {r2_acc_code, r2_acc_h (h before W_down), r2_colsum_context} at full length,
ext_4x8 (4x4 models; 32 pairs + flush; no unseen tokens (note 140-1)), n_codes, rho (bf16; scored = on-host --contraction_only --fp32).
    python -m boxes.multpair --family mult_pair --task 4x4 --seed 1 --stop_plateau --ckpt_every 2000 --out boxes/runs_fsq/s140_pair_4x4_s1
THEORY §145 MULTI-SLOT (b1d7741c): --slots K --carry_slots C --d_q 64 --no_round --dither 0.25 (§140 otherwise). Segment = [EOT, carried slot_1 ..
slot_C, a_i, b_j, column flag, w_1 .. w_K] (flush: [EOT, slots, " =", w_1 .. w_K]); w_k = learned write embeddings initialised from the distinct
unused tokens " W" " X" " Y" " Z"; write position k's last hidden -> its own LN_k / W_down_k -> NOISE-64 (continuous; dither U(+-0.25) at
training hand-offs after 2k) -> its own W_up_k = slot_k. Slot 1 uses the base LN / W_down / W_up. Only slots 1..C are carried (CARRY1: C = 1,
slots 2..4 written and dropped). The start is C zero codes; the answer decoder reads [EOT, carried slots]. Donor / contraction on the carried
codes (all C slots; contraction on their concatenation); s145_probe, s145.{acc_by_magnitude, slot_cosine, slot_swap, slot_ablation}."""
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
from boxes.fsq120 import EOT, FSQ120, driver_version, env_info, lr_at, make_opt  # noqa: E402
from boxes.multcol import DTOK, columns, dedup, gen_operands, load_bench, ridge_r2  # noqa: E402
import boxes.deltamem as DM  # noqa: E402

EQTOK, NEWTOK, SAMETOK = bc.tok(" =")[0], bc.tok(" ;")[0], bc.tok(" ,")[0]   # flush / new-column / same-column flags (note 140-1)
WTOK = [bc.tok(w)[0] for w in (" W", " X", " Y", " Z")]                    # §145 write-embedding initialisers (single, unused tokens)
BLANK = bc.tok(" _")[0]                                                     # §150 DIGIT arm's blank slot token (" _", single, unused here)
CIDX = [bc.tok(" " + chr(65 + i))[0] for i in range(21)]                    # §161 column-index tokens " A".." U" = values 1..21 (single tokens)
assert all(len(bc.tok(" " + chr(65 + i))) == 1 for i in range(21)) and len(set(CIDX)) == 21 and not set(CIDX) & (set(DTOK) | {EQTOK, NEWTOK, SAMETOK, BLANK} | set(WTOK))
assert all(len(bc.tok(w)) == 1 for w in (" W", " X", " Y", " Z")) and not set(WTOK) & (set(DTOK) | {EQTOK, NEWTOK, SAMETOK})
assert all(len(bc.tok(w)) == 1 for w in (" =", " ;", " ,")) and len({EQTOK, NEWTOK, SAMETOK}) == 3 and not {EQTOK, NEWTOK, SAMETOK} & set(DTOK)
CMAX = {"4x4": ((2000, 1), (5000, 2), (9000, 4), (None, 8)), "5x5": ((2000, 1), (5000, 2), (9000, 4), (12000, 8), (None, 10))}   # 5x5: note 140-2


def cmax(step, task):
    if task not in CMAX:
        raise SystemExit("§140: the %s curriculum is not registered" % task)
    for lim, c in CMAX[task]:
        if lim is None or step <= lim:
            return c


def schedule(na, nb):
    """Pairs (i, j) 0-based in column order (k = i + j ascending, i ascending), then the flush (-1, -1). Returns pairs (P, 2), the column of
    each step (flush = na + nb - 1), and the emission flag per step (last pair of its column, or the flush)."""
    pairs = [(i, k - i) for k in range(na + nb - 1) for i in range(na) if 0 <= k - i < nb] + [(-1, -1)]
    col = np.array([i + j if i >= 0 else na + nb - 1 for i, j in pairs]); emit = np.array([t == len(pairs) - 1 or col[t + 1] != col[t] for t in range(len(pairs))])
    return np.array(pairs), col, emit


def accumulators(A, Bd):
    """Exact per-step state: acc after each step (pairs: carry-in + the column's products so far; flush: the final carry) and the emitted
    digit at emission steps. Also the context-only colsum mod 10 per step (the step's column, products only)."""
    N, na = A.shape; nb = Bd.shape[1]; pairs, col, emit = schedule(na, nb); T = len(pairs)
    acc = np.zeros((N, T), np.int64); dig = np.full((N, T), -1, np.int64); a = np.zeros(N, np.int64)
    _, _, S = columns(A, Bd)
    for t, (i, j) in enumerate(pairs):
        if t > 0 and col[t] != col[t - 1]:                                  # a new column starts from the carry
            a = a // 10
        if i >= 0:
            a = a + A[:, i].astype(np.int64) * Bd[:, j]
        acc[:, t] = a
        if emit[t]:
            dig[:, t] = a % 10
    return acc, dig, S[:, col], pairs, col, emit


class PairFam:
    def __init__(self):
        self.name = "mult_pair"; self.n_items = 64; self.n_states = 10; self.item_tok = np.zeros((64, 1), np.int64)

    def answer_tokens(self, d):
        return np.array(DTOK)[np.asarray(d)][:, None]

    def decode(self, toks):
        lut = {t: i for i, t in enumerate(DTOK)}
        return np.array([lut.get(int(v), -1) for v in toks[:, 0]])


class MultPair(FSQ120):
    """items[:, t] = step index t into self.pairs; self.adig / self.bdig (B, n) digit tokens."""
    def __init__(self, g, fam):
        super().__init__(g, fam); self.adig = self.bdig = None; self.pairs = None; self.keep_h = False; self.last_h = None

    def segment(self, prev, items):
        g = self.g; B = items.shape[0]; dev = items.device; eot = g.embed(torch.full((B, 1), EOT, device=dev)); t = int(items[0]); i, j = (int(v) for v in self.pairs[t])
        if i < 0:                                                           # flush
            tail = torch.full((B, 1), EQTOK, device=dev)
        else:
            flag = NEWTOK if self.newcol[t] else SAMETOK
            tail = torch.cat([self.adig[:, i:i + 1], self.bdig[:, j:j + 1], torch.full((B, 1), flag, device=dev)], 1)
        h, _ = g(torch.cat([eot, prev.to(eot.dtype), g.embed(tail)], 1), None)
        if self.keep_h:
            self.last_h = h[:, -1].float()
        return h[:, -1]


def set_ops(m, A, Bd, dev):
    m.pa = m.pb = None                                                      # §158: per-step digits off unless a free batch sets them
    m.adig = torch.tensor(np.array(DTOK)[A], device=dev); m.bdig = torch.tensor(np.array(DTOK)[Bd], device=dev); m.pairs, col, _ = schedule(A.shape[1], Bd.shape[1])
    m.newcol = np.r_[True, col[1:] != col[:-1]]


def amp_ctx(amp):
    return torch.autocast("cuda", dtype=amp, enabled=amp is not None)


@torch.no_grad()
def rollout(m, A, Bd, dev, amp, bs, override=None, record=False):
    """Hard chains over all n^2 + 1 steps; decode at emission steps. override: (t, codes) replaces the code after step t (1-based).
    Returns digits (N, 2n) [, codes (N, T, d), h (N, T, 768)]."""
    N = len(A); pairs, col, emit = schedule(A.shape[1], Bd.shape[1]); T = len(pairs); out = np.full((N, A.shape[1] + Bd.shape[1]), -1)
    codes = np.zeros((N, T, F.D_Q), np.float32) if record else None; H = np.zeros((N, T, 768), np.float32) if record else None; m.keep_h = record
    for s in range(0, N, bs):
        sl = slice(s, s + bs); B = len(A[sl]); set_ops(m, A[sl], Bd[sl], dev); prev = m.start(B, dev)
        for t in range(T):
            with amp_ctx(amp):
                h = m.segment(prev, torch.full((B,), t, device=dev)); qs, q, zb = m.quant(h)
                if override is not None and override[0] == t + 1:
                    q = torch.tensor(override[1][sl], device=dev, dtype=qs.dtype); qs = q
                e = m.embed_code(qs)
                if emit[t]:
                    out[sl, col[t]] = m.fam.decode(m.answer_greedy(e, 1).cpu().numpy())
            if record:
                codes[sl, t] = q.float().cpu().numpy(); H[sl, t] = m.last_h.cpu().numpy()
            prev = e[:, None]
    m.keep_h = False
    return (out, codes, H) if record else out


def acc_report(pred, P):
    """+ note 140-1: per-step accuracy (digit k given all earlier digits correct) and the first-error position distribution."""
    C = pred == P; L = C.shape[1]; pre = np.cumprod(np.c_[np.ones(len(C), bool), C[:, :-1]], 1).astype(bool)
    fe = np.where(C.all(1), -1, np.argmax(~C, 1))
    return {"test_acc": float(C.all(1).mean()), "acc_by_column": {"per_column": [round(float(v), 4) for v in C.mean(0)],
            "cumulative": [round(float(C[:, :k + 1].all(1).mean()), 4) for k in range(L)]}, "digit_acc": float(C.mean()), "n": int(len(P)),
            "per_step_given_correct_prefix": [round(float(C[pre[:, k], k].mean()), 4) if pre[:, k].any() else None for k in range(L)],
            "first_error_position": {"none": int((fe < 0).sum()), **{str(k + 1): int((fe == k).sum()) for k in range(L)}}}


def donor(m, A, Bd, dev, amp, bs, rng, positions=(4, 8, 12), n=2000):
    """The code after pair t (1-based) from a donor; expected = the recipient's later digits from its pairs t+1.. with the donor's
    accumulator (exact). Scored on the digits emitted after step t."""
    N = min(n, len(A)); ir, idn = rng.permutation(len(A))[:N], rng.permutation(len(A))[:N]; Ar, Br, Ad, Bdn = A[ir], Bd[ir], A[idn], Bd[idn]
    pairs, col, emit = schedule(A.shape[1], Bd.shape[1]); Pr, _, _ = columns(Ar, Br); accd, _, _, _, _, _ = accumulators(Ad, Bdn)
    _, dcodes, _ = rollout(m, Ad, Bdn, dev, amp, bs, record=True); res = {}
    for p in positions:
        if p >= len(pairs):
            continue
        got = rollout(m, Ar, Br, dev, amp, bs, override=(p, dcodes[:, p - 1])); gt = Pr.copy(); a = accd[:, p - 1].copy()
        for t in range(p, len(pairs)):                                     # continue the recipient's pairs from the donor's accumulator
            i, j = pairs[t]
            if col[t] != col[t - 1]:
                a = a // 10
            if i >= 0:
                a = a + Ar[:, i] * Br[:, j]
            if emit[t]:
                gt[:, col[t]] = a % 10
        later = sorted({int(col[t]) for t in range(p, len(pairs)) if emit[t]}); sl = np.array(later)
        res["pos%d" % p] = {"donor_consistent": float((got[:, sl] == gt[:, sl]).mean()), "recipient_consistent": float((got[:, sl] == Pr[:, sl]).mean()),
                            "chance_overlap": float((gt[:, sl] == Pr[:, sl]).mean()), "later_digits": len(later), "n": N,
                            "row_level": {"donor_consistent": float((got[:, sl] == gt[:, sl]).all(1).mean()), "recipient_consistent": float((got[:, sl] == Pr[:, sl]).all(1).mean()),
                                          "chance_overlap": float((gt[:, sl] == Pr[:, sl]).all(1).mean())}}
    return res


@torch.no_grad()
def contraction(m, A, Bd, dev, amp, T0=2, n=500, seed=12301):
    """fsq120's contraction formulas on pair chains: perturb z_b after step T0, follow the remaining hand-offs; rho window min(8, K)."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(A))[:n]; A, Bd = A[idx], Bd[idx]; N = len(A); pairs, _, _ = schedule(A.shape[1], Bd.shape[1])
    L = len(pairs); K = L - T0; W = min(8, K); set_ops(m, A, Bd, dev)
    def step(prev, t, add=None):
        with amp_ctx(amp):
            h = m.segment(prev, torch.full((N,), t, device=dev)); qs, q, zb = m.quant(h, noise=add); e = m.embed_code(qs)
        return zb.float(), e
    def dec(e):
        with amp_ctx(amp):
            return m.fam.decode(m.answer_greedy(e, 1).cpu().numpy())
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
            "detail": {"T0": T0, "K": K, "window": W, "n": N, "L": L, "std_d": std_d.cpu().numpy().round(4).tolist(), "eps": R}}


def s140_probe(m, A, Bd, dev, amp, bs, n=1024, seed=14001):
    """R^2 of the true accumulator from code_t and from h_t (before W_down), and of the context-only colsum mod 10 of the step's column from
    code_t; full-length chains, pooled over steps (pairs + flush); ridge, fit on half the problems, scored on the rest."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(A))[:n]; A, Bd = A[idx], Bd[idx]; acc, _, S, _, _, _ = accumulators(A, Bd)
    _, Q, H = rollout(m, A, Bd, dev, amp, bs, record=True); h = len(A) // 2; T = acc.shape[1]
    def pool(X, y):
        return ridge_r2(X[:h].reshape(h * T, -1), y[:h].reshape(-1).astype(float), X[h:].reshape((len(A) - h) * T, -1), y[h:].reshape(-1).astype(float))
    return {"r2_acc_code": pool(Q, acc), "r2_acc_h": pool(H, acc), "r2_colsum_context": pool(Q, S), "n": int(len(A)), "steps": int(T)}


class MultSlot(MultPair):
    """§145: K write positions per segment, K slots (own LN / W_down / W_up each; slot 1 = the base maps), NOISE-64, C slots carried."""
    def __init__(self, g, fam, K, C, carry_from="first"):
        super().__init__(g, fam); self.K, self.C = K, C; self.cidx = list(range(K - C, K)) if carry_from == "last" else list(range(C))   # note 145-1: CARRY1 carries w_K's slot
        dev = g.w("wte.weight").device   # GPT-2 is already on the device (the new maps move with .to)
        self.w = nn.Parameter(g.embed(torch.tensor(WTOK[:K], device=dev)).detach().float().clone())
        self.lns = nn.ModuleList([nn.LayerNorm(768) for _ in range(K - 1)]); self.downs = nn.ModuleList([nn.Linear(768, F.D_Q) for _ in range(K - 1)])
        self.ups = nn.ModuleList([nn.Linear(F.D_Q, 768) for _ in range(K - 1)]); self.ablate = None

    def maps(self, k):
        return (self.ln, self.down, self.up) if k == 0 else (self.lns[k - 1], self.downs[k - 1], self.ups[k - 1])

    def start(self, B, dev, states=None):
        z0 = torch.zeros(B, F.D_Q, device=dev)
        return torch.stack([self.maps(k)[2](z0) for k in self.cidx], 1)

    def write(self, prev, t, noise=None):
        """One segment -> (Z (B, K, d) codes, E (B, K, 768) slot embeddings). noise: (B, K, d) added to z_b (contraction)."""
        g = self.g; B = prev.shape[0]; dev = prev.device; eot = g.embed(torch.full((B, 1), EOT, device=dev)); i, j = (int(v) for v in self.pairs[t])
        if i < 0:
            tail = torch.full((B, 1), EQTOK, device=dev)
        else:
            tail = torch.cat([self.adig[:, i:i + 1], self.bdig[:, j:j + 1], torch.full((B, 1), NEWTOK if self.newcol[t] else SAMETOK, device=dev)], 1)
        x = torch.cat([eot, prev.to(eot.dtype), g.embed(tail), self.w[None].expand(B, -1, -1).to(eot.dtype)], 1); h, _ = g(x, None); Hw = h[:, -self.K:]
        if self.keep_h:
            self.last_h = Hw.float()
        Z, E = [], []
        for k in range(self.K):
            ln, dn, up = self.maps(k); zb = torch.tanh(dn(ln(Hw[:, k].float())) + F.SHIFT) * F.HALF - F.OFF
            if self.training and self.noise > 0:
                zb = zb + (torch.rand_like(zb) * 2 - 1) * self.noise
            if noise is not None:
                zb = zb + noise[:, k]
            if self.ablate is not None and self.ablate == k:
                zb = torch.zeros_like(zb)
            Z.append(zb); E.append(up(zb))
        return torch.stack(Z, 1), torch.stack(E, 1)

    def embed_slots(self, Z, idx=None):
        idx = idx if idx is not None else list(range(Z.shape[1])); return torch.stack([self.maps(k)[2](Z[:, i]) for i, k in enumerate(idx)], 1)

    def logits(self, Ec):
        g = self.g; eot = g.embed(torch.full((Ec.shape[0], 1), EOT, device=Ec.device)); h, _ = g(torch.cat([eot, Ec.to(eot.dtype)], 1), None)
        return g.logits(h[:, -1]).float()

    def chain(self, start_emb, items, labels=None, trace_mask=None, record=False, override=None):
        B, L = items.shape; prev = start_emb; losses = []
        for t in range(L):
            Z, E = self.write(prev, int(items[0, t])); Ec = E[:, self.cidx]
            if labels is not None and trace_mask is not None and bool(trace_mask[:, t].any()):
                rows = torch.nonzero(trace_mask[:, t]).squeeze(1); ans = torch.tensor(self.fam.answer_tokens(labels[rows.cpu().numpy(), t]), device=items.device)
                losses.append(Fn.cross_entropy(self.logits(Ec[rows]), ans[:, 0]))
            prev = Ec
        return (torch.stack(losses).mean() if losses else None), prev, None, None


@torch.no_grad()
def rollout_slots(m, A, Bd, dev, amp, bs, override=None, record=False, swap=None):
    """§145 hard chains (all n^2 + 1 steps); decode at emission steps. override: (t, Zc (N, C, d)) replaces every carried code after step t;
    swap: (t, k, Zk (N, d)) replaces only slot k after step t. Returns digits [, codes (N, T, K, d)]."""
    N = len(A); pairs, col, emit = schedule(A.shape[1], Bd.shape[1]); T = len(pairs); out = np.full((N, A.shape[1] + Bd.shape[1]), -1)
    codes = np.zeros((N, T, m.K, F.D_Q), np.float32) if record else None
    for s_ in range(0, N, bs):
        sl = slice(s_, s_ + bs); B = len(A[sl]); set_ops(m, A[sl], Bd[sl], dev); prev = m.start(B, dev)
        for t in range(T):
            with amp_ctx(amp):
                Z, E = m.write(prev, t)
                if override is not None and override[0] == t + 1:
                    Zc = torch.tensor(override[1][sl], device=dev, dtype=Z.dtype); Z = Z.clone(); E = E.clone(); Z[:, m.cidx] = Zc; E[:, m.cidx] = m.embed_slots(Zc, m.cidx).to(E.dtype)
                if swap is not None and swap[0] == t + 1:
                    Z = Z.clone(); Z[:, swap[1]] = torch.tensor(swap[2][sl], device=dev, dtype=Z.dtype); E = E.clone(); E[:, swap[1]] = m.maps(swap[1])[2](Z[:, swap[1]]).to(E.dtype)
                Ec = E[:, m.cidx]
                if emit[t]:
                    out[sl, col[t]] = m.fam.decode(m.logits(Ec).argmax(-1).cpu().numpy()[:, None])
            if record:
                codes[sl, t] = Z.float().cpu().numpy()
            prev = Ec
    return (out, codes) if record else out


def later_gt(Ar, Br, acc0, p, pairs, col, emit, Pr):
    """The recipient's digits after step p, continued from accumulator acc0 (exact; as donor())."""
    gt = Pr.copy(); a = acc0.copy()
    for t in range(p, len(pairs)):
        i, j = pairs[t]
        if col[t] != col[t - 1]:
            a = a // 10
        if i >= 0:
            a = a + Ar[:, i] * Br[:, j]
        if emit[t]:
            gt[:, col[t]] = a % 10
    return gt, np.array(sorted({int(col[t]) for t in range(p, len(pairs)) if emit[t]}))


def consist(got, gt, Pr, sl):
    return {"donor_consistent": float((got[:, sl] == gt[:, sl]).mean()), "recipient_consistent": float((got[:, sl] == Pr[:, sl]).mean()),
            "chance_overlap": float((gt[:, sl] == Pr[:, sl]).mean()), "later_digits": int(len(sl)),
            "row_level": {"donor_consistent": float((got[:, sl] == gt[:, sl]).all(1).mean()), "recipient_consistent": float((got[:, sl] == Pr[:, sl]).all(1).mean()),
                          "chance_overlap": float((gt[:, sl] == Pr[:, sl]).all(1).mean())}}


def donor_slots(m, A, Bd, dev, amp, bs, rng, positions=(4, 8, 12), n=2000, swap_pos=8):
    """§145: every carried slot of a donor injected after pair p (row-level read); and per-slot swaps at swap_pos (only slot k from the donor)."""
    N = min(n, len(A)); ir, idn = rng.permutation(len(A))[:N], rng.permutation(len(A))[:N]; Ar, Br, Ad, Bdn = A[ir], Bd[ir], A[idn], Bd[idn]
    pairs, col, emit = schedule(A.shape[1], Bd.shape[1]); Pr, _, _ = columns(Ar, Br); accd, _, _, _, _, _ = accumulators(Ad, Bdn)
    _, dcodes = rollout_slots(m, Ad, Bdn, dev, amp, bs, record=True); res = {}; sw = {}
    for p in positions:
        if p >= len(pairs):
            continue
        got = rollout_slots(m, Ar, Br, dev, amp, bs, override=(p, dcodes[:, p - 1][:, m.cidx])); gt, sl = later_gt(Ar, Br, accd[:, p - 1], p, pairs, col, emit, Pr)
        res["pos%d" % p] = {**consist(got, gt, Pr, sl), "n": N}
    if swap_pos < len(pairs):
        gt, sl = later_gt(Ar, Br, accd[:, swap_pos - 1], swap_pos, pairs, col, emit, Pr)
        for k in m.cidx:
            got = rollout_slots(m, Ar, Br, dev, amp, bs, swap=(swap_pos, k, dcodes[:, swap_pos - 1, k])); sw["slot%d" % (k + 1)] = {**consist(got, gt, Pr, sl), "pos": swap_pos}
    return res, sw


@torch.no_grad()
def contraction_slots(m, A, Bd, dev, amp, T0=2, n=500, seed=12301):
    """§145: multpair's contraction on the concatenated carried codes (C x d): perturb every carried z_b after step T0, follow the rest."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(A))[:n]; A, Bd = A[idx], Bd[idx]; N = len(A); pairs, _, _ = schedule(A.shape[1], Bd.shape[1])
    L = len(pairs); K = L - T0; W = min(8, K); set_ops(m, A, Bd, dev); C = m.C
    def step(prev, t, add=None):
        with amp_ctx(amp):
            nz = None
            if add is not None:
                nz = torch.zeros(N, m.K, getattr(m, "zdim", F.D_Q), device=dev); nz[:, m.cidx] = add.view(N, C, -1)
            Z, E = m.write(prev, t, noise=nz)
        return Z[:, m.cidx].reshape(N, -1).float(), E[:, m.cidx]
    def dec(Ec):
        with amp_ctx(amp):
            return m.fam.decode(m.logits(Ec).argmax(-1).cpu().numpy()[:, None])
    prev = m.start(N, dev); zs = []
    for t in range(T0 - 1):
        z, e = step(prev, t); zs.append(z); prev = e
    pre = prev; zT0, eT0 = step(prev, T0 - 1); zs.append(zT0); clean = [zT0]; cdec = []; c = eT0
    for k in range(K):
        z, e = step(c, T0 + k); zs.append(z); clean.append(z); cdec.append(dec(e)); c = e
    std_d = torch.cat(zs).std(0); U = torch.tensor(rng.standard_normal((N, zT0.shape[-1])), dtype=torch.float32, device=dev); R = {}
    for eps in F.CONTRACT_EPS:
        zp, ep = step(pre, T0 - 1, add=eps * std_d * U); dist = [(zp - zT0).norm(dim=-1)]; agree = []; c = ep
        for k in range(K):
            z, e = step(c, T0 + k); dist.append((z - clean[k + 1]).norm(dim=-1)); agree.append(float((dec(e) == cdec[k]).mean())); c = e
        D = torch.stack(dist).cpu().numpy(); ratio = D[1:] / np.maximum(D[:-1], 1e-6); rho_k = np.median(ratio, axis=1)
        R[str(eps)] = {"rho_k": rho_k.round(4).tolist(), "rho_geo": float(np.exp(np.log(np.maximum(rho_k[:W], 1e-6)).mean())), "agree_k": [round(v, 4) for v in agree],
                       "dist_med_k": np.median(D, axis=1).round(4).tolist()}
    basin = max([e_ for e_ in F.CONTRACT_EPS if R[str(e_)]["agree_k"][W - 1] >= 0.99], default=0.0)
    return {"rho": R["0.1"]["rho_geo"], "basin": basin, "detail": {"T0": T0, "K": K, "window": W, "n": N, "L": L, "carried_dims": int(zT0.shape[-1]), "eps": R}}


def s145_extras(m, teA, teB, dev, amp, bs, pred, codes, n=1024, seed=14501):
    """§145 descriptive: per-slot x per-digit R^2 (the accumulator's units / tens / hundreds from each slot's code, ridge, fit on half the problems)
    and the accumulator's R^2 from all K slots; accuracy of each emitted digit by the accumulator's magnitude (< 100 / 100-199 / >= 200); the
    pairwise cosine of the K slots' codes; slot ablation (one slot's code zeroed at every step)."""
    acc, dig, _, pairs, col, emit = accumulators(teA, teB); P, _, _ = columns(teA, teB); out = {}
    ok = np.stack([pred[:, col[t]] == dig[:, t] for t in range(len(pairs)) if emit[t]], 1); av = np.stack([acc[:, t] for t in range(len(pairs)) if emit[t]], 1)
    out["acc_by_magnitude"] = {nm_: (float(ok[msk].mean()) if msk.any() else None) for nm_, msk in (("lt100", av < 100), ("100_199", (av >= 100) & (av < 200)), ("ge200", av >= 200))}
    out["magnitude_counts"] = {"lt100": int((av < 100).sum()), "100_199": int(((av >= 100) & (av < 200)).sum()), "ge200": int((av >= 200).sum())}
    Zc = codes.reshape(-1, m.K, codes.shape[-1]); cs = []
    for k in range(m.K):
        for l in range(k + 1, m.K):
            a_, b_ = Zc[:, k], Zc[:, l]; cs.append(float(((a_ * b_).sum(1) / (np.linalg.norm(a_, axis=1) * np.linalg.norm(b_, axis=1) + 1e-9)).mean()))
    out["slot_cosine"] = {"pairwise_mean": float(np.mean(cs)) if cs else None, "pairs": cs}
    abl = {}
    for k in m.cidx:
        m.ablate = k; pk = rollout_slots(m, teA, teB, dev, amp, bs); m.ablate = None; r_ = acc_report(pk, P)
        abl["slot%d" % (k + 1)] = {"test_acc": r_["test_acc"], "per_column": r_["acc_by_column"]["per_column"]}
    out["slot_ablation"] = abl
    rng = np.random.default_rng(seed); idx = rng.permutation(len(teA))[:n]; A, Bd = teA[idx], teB[idx]; ac, _, _, _, _, _ = accumulators(A, Bd)
    _, Q = rollout_slots(m, A, Bd, dev, amp, bs, record=True); h = len(A) // 2; T = ac.shape[1]
    def pool(X, y):
        return ridge_r2(X[:h].reshape(h * T, -1), y[:h].reshape(-1).astype(float), X[h:].reshape((len(A) - h) * T, -1), y[h:].reshape(-1).astype(float))
    pr = {"slot%d" % (k + 1): {nm_: pool(Q[:, :, k], (ac // 10 ** e_) % 10) for e_, nm_ in ((0, "units"), (1, "tens"), (2, "hundreds"))} for k in range(m.K)}
    return out, {"per_slot_digit_r2": pr, "r2_acc_all_slots": pool(Q.reshape(len(A), T, -1), ac), "r2_acc_carried_slots": pool(Q[:, :, m.cidx].reshape(len(A), T, -1), ac), "carried": [k + 1 for k in m.cidx],
                 "n": int(len(A)), "steps": int(T)}


class MultDelta(MultPair):
    """§147 DELTA-STATE: no carried code token; segment = [EOT, a_i, b_j, column flag] / [EOT, " ="], a fresh GPT-2 pass each, the DeltaMem state
    (B, 12, 64, 64) the only cross-segment channel; emission = GPT-2's LM head at the segment's last position (full-vocabulary CE)."""
    def __init__(self, g, fam, layer=6, write=None):
        super().__init__(g, fam); self.mem = DM.DeltaMem(g, layer, write)

    def start(self, B, dev, states=None):
        return self.mem.zero_state(B, dev)

    def tokens(self, t, B, dev):
        i, j = (int(v) for v in self.pairs[t])
        if i < 0:
            tail = torch.full((B, 1), EQTOK, device=dev)
        elif getattr(self, "pa", None) is not None:                         # §158: per-step digits (free chains)
            tail = torch.cat([self.pa[:, t:t + 1], self.pb[:, t:t + 1], torch.full((B, 1), NEWTOK if self.newcol[t] else SAMETOK, device=dev)], 1)
        else:
            tail = torch.cat([self.adig[:, i:i + 1], self.bdig[:, j:j + 1], torch.full((B, 1), NEWTOK if self.newcol[t] else SAMETOK, device=dev)], 1)
        if getattr(self, "cidx_off", None) is not None:                     # §161 MIX+IDX: token(c + r); c = i + j - 1 (1-based), the flush = last column + 1
            c_ = (self.adig.shape[1] + self.bdig.shape[1]) if i < 0 else i + j + 1
            r_ = self.cidx_r if (self.training and getattr(self, "cidx_r", None) is not None and len(self.cidx_r) == B) else torch.zeros(B, dtype=torch.long, device=dev)
            v_ = c_ + r_; assert int(v_.max()) <= len(CIDX), "§161: column index %d > %d tokens" % (int(v_.max()), len(CIDX))
            tail = torch.cat([tail, torch.tensor(CIDX, device=dev)[v_ - 1][:, None]], 1)
        return torch.cat([torch.full((B, 1), EOT, device=dev), tail], 1)

    def step(self, S, t, rec=None):
        if getattr(self, "add_reset", None) == "column" and t > 0 and self.newcol[t]:   # §154 SHIFT: additive state zeroed before a new column / the flush
            S = torch.cat([torch.zeros_like(S[:, :DM.H_ADD]), S[:, DM.H_ADD:]], 1)
        B = S.shape[0]; out, S = DM.gpt_forward_mem(self.g, self.g.embed(self.tokens(t, B, S.device)), self.mem, S, rec)
        return self.g.logits(out[:, -1]).float(), S

    def chain(self, start_emb, items, labels=None, trace_mask=None, record=False, override=None):
        B, L = items.shape; S = start_emb; losses = []
        for t in range(L):
            lg, S = self.step(S, int(items[0, t]))
            if labels is not None and trace_mask is not None and bool(trace_mask[:, t].any()):
                rows = torch.nonzero(trace_mask[:, t]).squeeze(1); ans = torch.tensor(self.fam.answer_tokens(labels[rows.cpu().numpy(), t]), device=items.device)
                losses.append(Fn.cross_entropy(lg[rows], ans[:, 0]))
        return (torch.stack(losses).mean() if losses else None), S, None, None


@torch.no_grad()
def rollout_delta(m, A, Bd, dev, amp, bs, override=None, record=False, keep_pos=(), reads=False, perturb=None, skip=()):
    """§147 chains over all n^2 + 1 steps, decode at emission steps. override: (t, S_donor (N, 12, 64, 64)) replaces the whole state after step t
    (1-based); keep_pos: steps whose state is returned; reads: the additive heads' read at each segment's last token (N, T, 384)."""
    N = len(A); pairs, col, emit = schedule(A.shape[1], Bd.shape[1]); T = len(pairs); out = np.full((N, A.shape[1] + Bd.shape[1]), -1)
    kept = {p_: np.zeros((N, 12, DM.DH, DM.DH), np.float32) for p_ in keep_pos}; R = np.zeros((N, T, DM.H_ADD * DM.DH), np.float32) if reads else None
    for s_ in range(0, N, bs):
        sl = slice(s_, s_ + bs); B = len(A[sl]); set_ops(m, A[sl], Bd[sl], dev); S = m.start(B, dev)
        for t in range(T):
            if t + 1 in skip:                                               # note 155-4: pair t's pass not run (memory unchanged) ...
                if getattr(m, "add_reset", None) == "column" and t > 0 and m.newcol[t]:   # ... but a column boundary's reset still applies (Theory)
                    S = torch.cat([torch.zeros_like(S[:, :DM.H_ADD]), S[:, DM.H_ADD:]], 1)
                continue
            rec = [] if reads else None
            with amp_ctx(amp):
                lg, S = m.step(S, t, rec)
            if override is not None and override[0] == t + 1:
                S = torch.tensor(override[1][sl], device=dev, dtype=S.dtype)
            if emit[t]:
                out[sl, col[t]] = m.fam.decode(lg.argmax(-1).cpu().numpy()[:, None])
            if t + 1 in kept:
                kept[t + 1][sl] = S.float().cpu().numpy()
            if reads:
                R[sl, t] = rec[0].float().cpu().numpy()
    return out, kept, R


def _dual_fit_pred(Xf, Yf, Xs, dev, lams=(1e-9, 1e-8, 1e-7, 1e-6), cls=False):
    """near-OLS for d >> n (note 155-4, Theory): standardised features (fit-set statistics), dual ridge K = Z Z^T / d, lambda (x n) by 3-fold CV on
    the fit set over `lams` (score: accuracy of the argmax for one-hot targets when cls, else R^2), then refit on all; fp32 features on `dev`,
    K / solves in fp64. Returns (predictions on Xs, lambda)."""
    Zf = torch.tensor(Xf, device=dev, dtype=torch.float32); mu, sd = Zf.mean(0), Zf.std(0) + 1e-6; Zf = (Zf - mu) / sd; d = Zf.shape[1]; n = len(Zf)
    K = (Zf @ Zf.T).double() / d; Y = torch.tensor(Yf, device=dev, dtype=torch.float64); Y = Y[:, None] if Y.dim() == 1 else Y
    fold = torch.arange(n, device=dev) % 3
    def fit(idx, lam):
        ym = Y[idx].mean(0); return torch.linalg.solve(K[idx][:, idx] + lam * len(idx) * torch.eye(len(idx), device=dev, dtype=torch.float64), Y[idx] - ym), ym
    def score(yt, yp):
        if cls:
            return float((yp.argmax(1) == yt.argmax(1)).double().mean())
        return float(1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean(0)) ** 2).sum().clamp_min(1e-12))
    def cv(lam):
        sc = []
        for k in range(3):
            tr = torch.nonzero(fold != k).squeeze(1); va = torch.nonzero(fold == k).squeeze(1); al, ym = fit(tr, lam); sc.append(score(Y[va], K[va][:, tr] @ al + ym))
        return float(np.mean(sc))
    best = max(lams, key=cv)
    if best in (min(lams), max(lams)):                                      # Audit (152-5's lesson): at the grid edge -> widen two decades that side
        ext = [best * 10 ** k for k in ((-1, -2) if best == min(lams) else (1, 2))]; best = max([best] + ext, key=cv)
    al, ym = fit(torch.arange(n, device=dev), best); out = []
    for i in range(0, len(Xs), 500):
        Zs = (torch.tensor(Xs[i:i + 500], device=dev, dtype=torch.float32) - mu) / sd; out.append(((Zs @ Zf.T).double() / d @ al + ym).cpu().numpy())
    p_ = np.concatenate(out); return (p_ if cls or Y.shape[1] > 1 else p_[:, 0]), best


@torch.no_grad()
def what_state(m, teA, teB, trA, trB, dev, amp, bs, n_fit=8000, skip_pairs=(4, 7), points=(4, 8, 12), seed=14002):
    """note 155-4 (descriptive; Audit pre-registered): what the memory state holds.
    (1) pass_skip: pair #t's pass not run (the memory passes through unchanged; later passes keep their inputs and flags; under --add_reset the
        column-boundary reset STILL applies - Theory: a harness operation, not the pair's computation), t = 4 / 7 (columns 3 / 4's first pairs); rows with a_i b_j != 0; row level:
        row_skip_consistent = (answer == product - a_i b_j 10^(i+j)), full_product = (answer == product).
    (2) donor operands_consistent: the donor's whole state swapped in after pair P (P = 4 / 8 / 12; donor_delta's sampling); hybrid operands = the
        donor's digit at each a_i / b_j position appearing in pairs 1..P, else the recipient's; row-level share answer == hybrid product.
    (3) operand_probe: from the flattened S after pairs 4 / 8 / 12, near-OLS (dual, lambda by 3-fold CV over 1e-9..1e-6, standardised) one-hot regressions -> argmax of each seen a_i / b_j
        digit and rounded-exact regressions of each partial column sum (products in pairs 1..P only); fit on n_fit train items, scored on the test items."""
    from boxes.multcol import to_int                                         # (multpair does not import it at module level)
    pairs, col, emit = schedule(teA.shape[1], teB.shape[1]); T = len(pairs); pw = 10 ** np.arange(teA.shape[1] + teB.shape[1], dtype=np.int64)
    def ans_int(got):
        return np.where((got >= 0).all(1), (got * pw[None, :got.shape[1]]).sum(1), -1)
    out = {"pass_skip": {}}; prod = to_int(teA) * to_int(teB)
    for t in skip_pairs:
        i, j = (int(v) for v in pairs[t - 1]); keep = (teA[:, i] * teB[:, j]) != 0; Ak, Bk = teA[keep], teB[keep]
        got = ans_int(rollout_delta(m, Ak, Bk, dev, amp, bs, skip=(t,))[0]); pk = prod[keep]; rs = pk - Ak[:, i].astype(np.int64) * Bk[:, j] * 10 ** (i + j)
        out["pass_skip"]["pair%d" % t] = {"a_i": i + 1, "b_j": j + 1, "column": i + j + 1, "row_skip_consistent": float((got == rs).mean()), "full_product": float((got == pk).mean()),
                                          "n": int(keep.sum()), "n_excluded_zero_product": int((~keep).sum()), "skip_keeps_reset": getattr(m, "add_reset", None) == "column"}
    rng = np.random.default_rng(seed); N = min(1000, len(teA)); ir, idn = rng.permutation(len(teA))[:N], rng.permutation(len(teA))[:N]
    Ar, Br, Ad, Bdn = teA[ir], teB[ir], teA[idn], teB[idn]; _, kept, _ = rollout_delta(m, Ad, Bdn, dev, amp, bs, keep_pos=[p_ for p_ in points if p_ < T]); dn = {}
    for P in kept:
        got = ans_int(rollout_delta(m, Ar, Br, dev, amp, bs, override=(P, kept[P]))[0]); seenA = {int(pairs[t_][0]) for t_ in range(P)}; seenB = {int(pairs[t_][1]) for t_ in range(P)}
        hA = np.stack([Ad[:, i] if i in seenA else Ar[:, i] for i in range(teA.shape[1])], 1); hB = np.stack([Bdn[:, j] if j in seenB else Br[:, j] for j in range(teB.shape[1])], 1)
        dn["pos%d" % P] = {"operands_consistent": float((got == to_int(hA) * to_int(hB)).mean()), "recipient_consistent": float((got == to_int(Ar) * to_int(Br)).mean()),
                           "donor_full_consistent": float((got == to_int(Ad) * to_int(Bdn)).mean()), "seen_a": sorted(i + 1 for i in seenA), "seen_b": sorted(j + 1 for j in seenB), "n": int(N)}
    out["donor_operands"] = dn
    trA, trB = trA[:n_fit], trB[:n_fit]; acc_tr = accumulators(trA, trB)[0]; acc_te = accumulators(teA, teB)[0]
    _, Ktr, _ = rollout_delta(m, trA, trB, dev, amp, bs, keep_pos=list(points)); _, Kte, _ = rollout_delta(m, teA, teB, dev, amp, bs, keep_pos=list(points)); op = {}
    for P in points:
        Xf = Ktr[P].reshape(len(trA), -1); Xs = Kte[P].reshape(len(teA), -1); seenA = sorted({int(pairs[t_][0]) for t_ in range(P)}); seenB = sorted({int(pairs[t_][1]) for t_ in range(P)}); r_ = {"a": {}, "b": {}, "colsum": {}}
        for nm, D_tr, D_te, idx in (("a", trA, teA, seenA), ("b", trB, teB, seenB)):
            for i in idx:
                pr_, lam_ = _dual_fit_pred(Xf, np.eye(10)[D_tr[:, i]], Xs, dev, cls=True); r_[nm][str(i + 1)] = {"acc": float((pr_.argmax(1) == D_te[:, i]).mean()), "lambda": lam_}
        csf = np.zeros((len(trA), 7)); cst = np.zeros((len(teA), 7))
        for t_ in range(P):
            i, j = (int(v) for v in pairs[t_]); csf[:, i + j] += trA[:, i] * trB[:, j]; cst[:, i + j] += teA[:, i] * teB[:, j]
        for c in sorted({int(pairs[t_][0]) + int(pairs[t_][1]) for t_ in range(P)}):
            pr_, lam_ = _dual_fit_pred(Xf, csf[:, c], Xs, dev); r_["colsum"][str(c + 1)] = {"exact": float((np.rint(pr_) == cst[:, c]).mean()), "lambda": lam_}
        r_["colsum_cls"] = {}                                               # note 155-4 correction (287aa8e5): like-for-like 10-way classifiers
        for c in sorted({int(pairs[t_][0]) + int(pairs[t_][1]) for t_ in range(P)}):
            tl = max(t_ for t_ in range(P) if int(pairs[t_][0]) + int(pairs[t_][1]) == c)   # c^_k after column c's last pair so far (carry-in + its products)
            yf, ys = acc_tr[:, tl], acc_te[:, tl]; rc = {}
            for nm_, ff_, ft_ in (("units", yf % 10, ys % 10), ("carry", yf // 10, ys // 10)):
                kk = int(max(ff_.max(), ft_.max())) + 1; pr_, lam_ = _dual_fit_pred(Xf, np.eye(kk)[ff_], Xs, dev, cls=True)
                rc[nm_] = {"acc": float((pr_.argmax(1) == ft_).mean()), "lambda": lam_, "classes": kk}
            r_["colsum_cls"][str(c + 1)] = rc
        r_["n_fit"] = int(len(trA)); r_["n_score"] = int(len(teA)); op["after_pair%d" % P] = r_
    out["operand_probe"] = op; out["n_fit_train"] = int(len(trA)); out["def"] = what_state.__doc__
    return out


def compute_pair(m, na, nb):
    """boxes.core.Meter per item (§98 / §152 convention; full-vocabulary LM-head rows): every pair segment [EOT, a_i, b_j, flag] (4 tokens) and the
    flush [EOT, " ="] (2) is a fresh 12-block pass (context 0); a logit row at each emission step. DeltaMem branch: 2 x branch params per token, plus
    per head and token the read (2 DH^2) and the writes (additive 2 DH^2, delta 8 DH^2: S k, the rank-1 update, decay); §155 R (pass-start reads)
    and §147 (read after write) cost the same under this count."""
    pairs, col, emit = schedule(na, nb); mt = bc.Meter(m.g); tok = 0; rows = 0
    for t in range(len(pairs)):
        T_ = (2 if pairs[t][0] < 0 else 4) + (1 if getattr(m, "cidx_off", None) is not None else 0); r_ = 1 if emit[t] else 0; mt.step(T_, 0, r_); tok += T_; rows += r_
    nb_ = sum(p_.numel() for n_, p_ in m.mem.named_parameters() if not n_.startswith("s_")); D2 = DM.DH ** 2
    fb = 2 * nb_ * tok + (DM.H_ADD + DM.H_DEL) * 2 * D2 * tok + (DM.H_ADD * 2 + DM.H_DEL * 8) * D2 * tok; d = mt.as_dict()
    return {"flops": d["flops"] + fb, "flops_trunk": d["flops"], "flops_branch": float(fb), "sequential_passes": d["seq_passes"], "lm_head_rows": rows,
            "tokens_processed": tok, "convention": "§98 (boxes.core.Meter, full vocabulary); every pair segment a fresh pass"}


def donor_delta(m, A, Bd, dev, amp, bs, rng, positions=(4, 8, 12), n=1000):
    """§147 SERIAL: the donor's whole memory state swapped in after pair p; the recipient's later digits with the donor's accumulator (row level)."""
    N = min(n, len(A)); ir, idn = rng.permutation(len(A))[:N], rng.permutation(len(A))[:N]; Ar, Br, Ad, Bdn = A[ir], Bd[ir], A[idn], Bd[idn]
    pairs, col, emit = schedule(A.shape[1], Bd.shape[1]); Pr, _, _ = columns(Ar, Br); accd, _, _, _, _, _ = accumulators(Ad, Bdn)
    _, kept, _ = rollout_delta(m, Ad, Bdn, dev, amp, bs, keep_pos=[p_ for p_ in positions if p_ < len(pairs)]); res = {}
    for p in kept:
        got, _, _ = rollout_delta(m, Ar, Br, dev, amp, bs, override=(p, kept[p])); gt, sl = later_gt(Ar, Br, accd[:, p - 1], p, pairs, col, emit, Pr)
        res["pos%d" % p] = {**consist(got, gt, Pr, sl), "n": N}
    return res


def acc_by_magnitude(pred, A, Bd):
    acc, dig, _, pairs, col, emit = accumulators(A, Bd)
    ok = np.stack([pred[:, col[t]] == dig[:, t] for t in range(len(pairs)) if emit[t]], 1); av = np.stack([acc[:, t] for t in range(len(pairs)) if emit[t]], 1)
    return {nm_: (float(ok[msk].mean()) if msk.any() else None) for nm_, msk in (("lt100", av < 100), ("100_199", (av >= 100) & (av < 200)), ("ge200", av >= 200))}


@torch.no_grad()
def rho_delta(m, A, Bd, dev, amp, T0=2, n=300, seed=12301):
    """§147 (descriptive): perturb the whole memory state after step T0 by eps x per-element std x u; follow the later steps; median distance
    ratios per head type (additive / delta), and 133-1's first-3 form of each."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(A))[:n]; A, Bd = A[idx], Bd[idx]; N = len(A); pairs, _, _ = schedule(A.shape[1], Bd.shape[1])
    L = len(pairs); set_ops(m, A, Bd, dev); S = m.start(N, dev); clean = []
    for t in range(L):
        with amp_ctx(amp):
            _, S = m.step(S, t)
        clean.append(S.float())
    sd = torch.stack(clean[T0 - 1:]).std(0).clamp_min(1e-6); U = torch.tensor(rng.standard_normal(tuple(clean[0].shape)), dtype=torch.float32, device=dev); out = {}
    for eps in (0.01, 0.1):
        S = clean[T0 - 1] + eps * sd * U; da, dd = [], []
        for t in range(T0, L):
            da.append((S[:, :DM.H_ADD] - clean[t - 1][:, :DM.H_ADD]).flatten(1).norm(dim=1)); dd.append((S[:, DM.H_ADD:] - clean[t - 1][:, DM.H_ADD:]).flatten(1).norm(dim=1))
            with amp_ctx(amp):
                _, S = m.step(S, t)
            S = S.float()
        r = {}
        for nm_, D_ in (("additive", da), ("delta", dd)):
            D_ = torch.stack(D_).cpu().numpy(); rk = np.median(D_[1:] / np.maximum(D_[:-1], 1e-9), axis=1); dk = np.median(D_, axis=1)
            use = [rk[k] for k in range(min(3, len(rk))) if dk[k] > 1e-6 and rk[k] > 0]
            r[nm_] = {"rho_k": rk.round(4).tolist(), "dist_med_k": dk.round(6).tolist(), "rho_k3": float(np.exp(np.mean(np.log(use)))) if use else None}
        out[str(eps)] = r
    return {"T0": T0, "n": N, "eps": out}


@torch.no_grad()
def read_ratio_by_handoff(m, A, Bd, dev, amp, n=256):
    """§147 (note 147-2): per hand-off, the median over a probe batch of ||g_del read_del|| / ||g_add read_add|| at the segment's last token;
    also the pooled median (all hand-offs). One batch of n test items, eval mode, no draws."""
    was = m.training; m.eval(); m.mem.norms = []; rollout_delta(m, A[:n], Bd[:n], dev, amp, n); nr = m.mem.norms; m.mem.norms = None; m.train(was)
    r = [(d_ / a_.clamp_min(1e-12)).float().cpu().numpy() for d_, a_ in nr]
    return [float(np.median(x)) for x in r], float(np.median(np.concatenate(r)))


@torch.no_grad()
def alpha_by_head(m, A, Bd, dev, amp, n=256):
    """§147 note 147-3: per delta head, the mean decay alpha over a fixed probe batch (n test items, every token of every segment), and the
    input-independent sigmoid(gate bias)."""
    was = m.training; m.eval(); m.mem.alpha_rec = []; rollout_delta(m, A[:n], Bd[:n], dev, amp, n); r = torch.stack(m.mem.alpha_rec).mean(0); m.mem.alpha_rec = None; m.train(was)
    return {"mean_alpha": [round(float(v), 6) for v in r], "sigmoid_bias": [round(float(v), 6) for v in torch.sigmoid(m.mem.gate.bias[:DM.H_DEL].detach())]}


def s147_extras(m, teA, teB, dev, amp, bs, pred):
    """§147 descriptive: acc_by_magnitude; head_ablation (additive / delta reads zeroed at test); colsum_readout_r2 (ridge of the true column
    accumulator from the additive heads' read phi(q)^T S at each segment's last token, fit on half the problems, pooled over steps) and per
    column index; rho_by_head_type (rho_delta)."""
    P, _, _ = columns(teA, teB); out = {"acc_by_magnitude": acc_by_magnitude(pred, teA, teB)}; abl = {}
    for ht in ("add", "delta"):
        m.mem.ablate = ht; pk, _, _ = rollout_delta(m, teA, teB, dev, amp, bs); m.mem.ablate = None; r_ = acc_report(pk, P)
        abl[ht + "_zeroed"] = {"test_acc": r_["test_acc"], "per_column": r_["acc_by_column"]["per_column"]}
    out["head_ablation"] = abl
    acc, _, _, pairs, col, _ = accumulators(teA, teB); _, _, R = rollout_delta(m, teA, teB, dev, amp, bs, reads=True); h = len(teA) // 2; T = acc.shape[1]
    out["colsum_readout_r2"] = ridge_r2(R[:h].reshape(h * T, -1), acc[:h].reshape(-1).astype(float), R[h:].reshape((len(teA) - h) * T, -1), acc[h:].reshape(-1).astype(float))
    byc = {}
    for c_ in sorted(set(col.tolist())):
        ts = np.where(col == c_)[0]
        if (acc[:, ts].std() > 0):
            byc[str(int(c_) + 1)] = ridge_r2(R[:h][:, ts].reshape(-1, R.shape[-1]), acc[:h][:, ts].reshape(-1).astype(float), R[h:][:, ts].reshape(-1, R.shape[-1]), acc[h:][:, ts].reshape(-1).astype(float))
    out["colsum_readout_r2_by_column"] = byc
    out["colsum_readout_def"] = "ridge R^2 of the true column accumulator from the additive heads' read phi(q)^T S (384 d) at each segment's last token"
    out["rho_by_head_type"] = rho_delta(m, teA, teB, dev, amp)
    return out


# ================================================================ THEORY §154 SHIFT-MEM (ed7857cc): column-addressed additive memory
@torch.no_grad()
def reads_by_step(m, A, Bd, dev, amp, bs, pd=None):
    """The branch read [g_add read_add || g_del read_del] (W_o's input) at each segment's last token: (N, T, 768) fp32."""
    N = len(A); pairs, _, _ = schedule(A.shape[1], Bd.shape[1]); T = len(pairs); R = np.zeros((N, T, 2 * DM.H_ADD * DM.DH), np.float32); buf = []
    hk = m.mem.wo.register_forward_hook(lambda mod_, inp_, out_: buf.append(inp_[0][:, -1].detach().float()))
    try:
        for s_ in range(0, N, bs):
            sl = slice(s_, s_ + bs); B = len(A[sl]); set_ops(m, A[sl], Bd[sl], dev); S = m.start(B, dev)
            if pd is not None:                                              # §158: per-step digits (free chains)
                m.pa = torch.tensor(np.array(DTOK)[pd[0][sl]], device=dev); m.pb = torch.tensor(np.array(DTOK)[pd[1][sl]], device=dev)
            for t in range(T):
                buf.clear()
                with amp_ctx(amp):
                    _, S = m.step(S, t)
                R[sl, t] = buf[0].cpu().numpy()
    finally:
        hk.remove()
    return R


def carry_in(A, Bd):
    """(N, n_cols) carry into each column k (0 for k = 0): floor(c_{k-1} / 10), c_k = the column's products + its carry-in."""
    acc, _, _, pairs, col, emit = accumulators(A, Bd); nc = A.shape[1] + Bd.shape[1] - 1; ci = np.zeros((len(A), nc), np.int64)
    last = {int(col[t]): t for t in range(len(pairs)) if pairs[t][0] >= 0}
    for k in range(1, nc):
        ci[:, k] = acc[:, last[k - 1]] // 10
    return ci


def carry_probe(m, A, Bd, dev, amp, bs):
    """held-out ridge R^2 of each column's carry-in from the delta heads' read (384-d) at the column's first pair (k >= 1)."""
    R = reads_by_step(m, A, Bd, dev, amp, bs)[:, :, DM.H_ADD * DM.DH:]; pairs, col, _ = schedule(A.shape[1], Bd.shape[1]); ci = carry_in(A, Bd); h = len(A) // 2
    first = {}
    for t in range(len(pairs)):
        if pairs[t][0] >= 0 and int(col[t]) not in first:
            first[int(col[t])] = t
    out = []
    for k in range(1, A.shape[1] + Bd.shape[1] - 1):
        y = ci[:, k].astype(float); X = R[:, first[k]]
        out.append(ridge_r2(X[:h], y[:h], X[h:], y[h:]) if y[:h].std() > 0 and y[h:].std() > 0 else None)
    return out


@torch.no_grad()
def c7_edit(m, A, Bd, dev, amp, bs, deltas=(-2, -1, 1, 2), cols=(2, 3, 4, 5, 6, 7), lam=1e-3):
    """§154 C7 (SHIFT; Theory's layout): a raw-space ridge probe w of the ACCUMULATOR c^_k (column sum + carry-in, = acc at the column's last pair)
    from the additive read at the emission steps (fit half). On the score half, ONE column k per rollout: delta * u (u = w / ||w||^2, the probe's
    decoded accumulator moves by exactly delta) added to the additive read before W_o at column k's emission step only. Per (k, delta):
    shifted_share = column k's digit == (c^_k + delta) mod 10; unedited_share = the same rows with no edit, digit == c^_k mod 10;
    carry_follow_share = column k+1's digit == (c^_{k+1} - floor(c^_k / 10) + floor((c^_k + delta) / 10)) mod 10 (k = 7: the flush digit);
    unedited_next_share = column k+1's digit == c^_{k+1} mod 10 with the edit (reference). Columns 1-based. Note: the edit changes the read
    output only; the memory state is written from the residual entering block 6, so no state downstream of the edit can change."""
    acc, dig, _, pairs, col, emit = accumulators(A, Bd); R = reads_by_step(m, A, Bd, dev, amp, bs)[:, :, :DM.H_ADD * DM.DH]; h = len(A) // 2; et = np.nonzero(emit)[0]
    X = R[:h][:, et].reshape(-1, R.shape[-1]).astype(np.float64); y = acc[:h][:, et].reshape(-1).astype(np.float64); mu, ym = X.mean(0), y.mean()
    w = np.linalg.solve((X - mu).T @ (X - mu) + lam * len(X) * np.eye(X.shape[1]), (X - mu).T @ (y - ym)); Xs = R[h:][:, et].reshape(-1, R.shape[-1])
    r2 = float(1 - (((Xs - mu) @ w + ym - acc[h:][:, et].reshape(-1)) ** 2).sum() / max(((acc[h:][:, et] - acc[h:][:, et].mean()) ** 2).sum(), 1e-12))
    u = torch.tensor(w / (w @ w), dtype=torch.float32, device=dev); Ar, Br = A[h:], Bd[h:]; N = len(Ar); T = len(pairs); C = acc[h:][:, et]   # (N, n_emit) c^ per emission
    ed = {"t": -1, "delta": 0.0}
    def pre(mod_, inp_):
        if ed["t"] < 0:
            return None
        x = inp_[0].clone(); x[:, -1, :DM.H_ADD * DM.DH] += ed["delta"] * u.to(x.dtype); return (x,)
    def run(t_edit, dl):
        got = np.full((N, T), -1, np.int64)
        for s_ in range(0, N, bs):
            sl = slice(s_, s_ + bs); B = len(Ar[sl]); set_ops(m, Ar[sl], Br[sl], dev); S = m.start(B, dev)
            for t in range(T):
                ed["t"] = t if t == t_edit else -1; ed["delta"] = float(dl)
                with amp_ctx(amp):
                    lg, S = m.step(S, t)
                ed["t"] = -1
                if emit[t]:
                    got[sl, t] = m.fam.decode(lg.argmax(-1).cpu().numpy()[:, None])
        return got[:, et]
    hk = m.mem.wo.register_forward_pre_hook(pre); out = {"probe_r2_score_half": r2, "probe_target": "accumulator c^_k = column sum + carry-in (acc at the emission step)"}
    try:
        base = run(-1, 0.0)
        for dl in deltas:
            byc = {}; tot = {"shifted": 0, "unedited": 0, "follow": 0, "next_unedited": 0, "n": 0}
            for k in cols:
                g_ = run(int(et[k - 1]), dl); ck, cn = C[:, k - 1], C[:, k]
                sh = g_[:, k - 1] == (ck + dl) % 10; ue = base[:, k - 1] == ck % 10
                fo = g_[:, k] == (cn - ck // 10 + (ck + dl) // 10) % 10; nu = g_[:, k] == cn % 10
                byc[str(k)] = {"shifted_share": float(sh.mean()), "unedited_share": float(ue.mean()), "carry_follow_share": float(fo.mean()),
                               "unedited_next_share": float(nu.mean()), "carry_changes_share": float(((ck + dl) // 10 != ck // 10).mean()), "n": int(N)}
                tot["shifted"] += int(sh.sum()); tot["unedited"] += int(ue.sum()); tot["follow"] += int(fo.sum()); tot["next_unedited"] += int(nu.sum()); tot["n"] += int(N)
            out["%+d" % dl] = {"shifted_share": tot["shifted"] / tot["n"], "unedited_share": tot["unedited"] / tot["n"], "carry_follow_share": tot["follow"] / tot["n"],
                               "unedited_next_share": tot["next_unedited"] / tot["n"], "n": tot["n"], "by_column": byc}
    finally:
        hk.remove()
    out["def"] = ("one column k (1-based, 2-7) edited per rollout at its emission step (score half): delta * w / ||w||^2 on the additive read before W_o; "
                  "w = raw-space ridge of c^_k on the additive read at emission steps (fit half)")
    return out


def _raw_ridge(X, y, lam=1e-3):
    """raw-space (centered, unstandardised) ridge: w, mu, ym with pred = (x - mu) w + ym."""
    X = X.astype(np.float64); y = y.astype(np.float64); mu, ym = X.mean(0), y.mean()
    return np.linalg.solve((X - mu).T @ (X - mu) + lam * len(X) * np.eye(X.shape[1]), (X - mu).T @ (y - ym)), mu, ym


@torch.no_grad()
def _branch_in(m, S, t):
    """The residual entering the branch for step t's segment (independent of S: blocks 0-5 see the segment's tokens only)."""
    if getattr(m.mem, "write_mode", None) is not None:                     # §155: the read is a function, not the module's forward
        cap = {}; DM.gpt_forward_mem_r(m.g, m.g.embed(m.tokens(t, S.shape[0], S.device)), m.mem, S, cap=cap); return cap["h6"].float()
    cap = []; hk = m.mem.register_forward_pre_hook(lambda mod_, inp_: cap.append(inp_[0].detach().float()))
    try:
        m.step(S, t)
    finally:
        hk.remove()
    return cap[0]


@torch.no_grad()
def c7_state_edit(m, A, Bd, dev, amp, bs, deltas=(-2, -1, 1, 2), cols=(2, 3, 4, 5, 6, 7), prop_cols=(2, 3, 4, 5), pd=None, only_prop=False):
    """§154 C7 STATE edits (descriptive; Audit / Theory). S is a function of the input tokens only (no read -> write path), so an edit reaches
    later columns only through the perturbed state itself.
    (a) additive: entering column k's emission step (after that step's SHIFT reset when it starts a column), Delta S_add per head h =
        phi(q_T,h) (delta u_h / g_add)^T / ||phi(q_T,h)||^2 (minimum-norm rank 1), u = w_add / ||w_add||^2 for the raw ridge of the accumulator
        c^_k on the additive read at emission steps: the read at the emission token moves by exactly delta u. Reports column k's
        shifted_share / unedited_share and column k+1's digit (next_unchanged_share; under SHIFT the additive store is zeroed before k+1).
    (b) delta: after column k's last pair, Delta S_del per head = (r_h / g_del) z_h^T / ||z_h||^2 with z = A_0 ... A_{T-1} q_d (the
        segment's delta transforms A_t = alpha_t (I - beta_t k_t k_t^T) applied to the first-pair token's normalised q), r = delta w_c / ||w_c||^2
        for the raw ridge of column k+1's carry-in on the delta read at its first pair: that read moves by exactly delta r. Reports column
        k+1's carry_shift_share (digit == (c^_{k+1} + delta) mod 10), unedited_share and the probe's held-out R^2. k = 7: k+1 is the flush."""
    if pd is None:
        acc, dig, _, pairs, col, emit = accumulators(A, Bd); ci = carry_in(A, Bd)
    else:                                                                   # §158 free chains: the drawn pairs' own column sums and true carries
        acc, dig, pairs, col, emit = acc_steps(pd[0], pd[1], A.shape[1], Bd.shape[1]); nc = A.shape[1] + Bd.shape[1] - 1; ci = np.zeros((len(A), nc), np.int64)
        last = {int(col[t]): t for t in range(len(pairs)) if pairs[t][0] >= 0}
        for k_ in range(1, nc):
            ci[:, k_] = acc[:, last[k_ - 1]] // 10
    T = len(pairs); h = len(A) // 2; et = np.nonzero(emit)[0]
    R = reads_by_step(m, A, Bd, dev, amp, bs, pd=pd); RA, RD = R[:, :, :DM.H_ADD * DM.DH], R[:, :, DM.H_ADD * DM.DH:]
    first = {}
    for t in range(T):
        if int(col[t]) not in first:
            first[int(col[t])] = t                                          # the flush is its own column (index na + nb - 1)
    wa, mua, yma = _raw_ridge(RA[:h][:, et].reshape(-1, RA.shape[-1]), acc[:h][:, et].reshape(-1)); ua = torch.tensor(wa / (wa @ wa), dtype=torch.float32, device=dev)
    wc, r2c = {}, {}
    for k in cols:                                                          # carry-in probe of column k+1 (0-based column index k)
        tf = first[k]; X = RD[:, tf]; y = ci_or_flush(acc, ci, et, k)
        w_, mu_, ym_ = _raw_ridge(X[:h], y[:h]); wc[k] = torch.tensor(w_ / max(w_ @ w_, 1e-12), dtype=torch.float32, device=dev)
        r2c[str(k + 1)] = float(1 - (((X[h:] - mu_) @ w_ + ym_ - y[h:]) ** 2).sum() / max(((y[h:] - y[h:].mean()) ** 2).sum(), 1e-12))
    Ar, Br = A[h:], Bd[h:]; N = len(Ar); C = acc[h:][:, et]; ga, gd = float(torch.exp(m.mem.s_add)), float(torch.exp(m.mem.s_del)); HA = DM.H_ADD
    def run(edit):
        got = np.full((N, T), -1, np.int64)
        for s_ in range(0, N, bs):
            sl = slice(s_, s_ + bs); Bn = len(Ar[sl]); set_ops(m, Ar[sl], Br[sl], dev); S = m.start(Bn, dev)
            if pd is not None:                                              # §158: the score half's per-step digits
                m.pa = torch.tensor(np.array(DTOK)[pd[0][h:][sl]], device=dev); m.pb = torch.tensor(np.array(DTOK)[pd[1][h:][sl]], device=dev)
            for t in range(T):
                rst = getattr(m, "add_reset", None)
                if edit is not None and edit[0] == t:
                    if rst == "column" and t > 0 and m.newcol[t]:
                        S = torch.cat([torch.zeros_like(S[:, :HA]), S[:, HA:]], 1)
                    S = edit[1](S, t); m.add_reset = None
                with amp_ctx(amp):
                    lg, S = m.step(S, t)
                m.add_reset = rst
                if emit[t]:
                    got[sl, t] = m.fam.decode(lg.argmax(-1).cpu().numpy()[:, None])
        return got[:, et]
    def add_edit(dl):
        def f(S, t):
            x = _branch_in(m, S, t)[:, -1]; a_ = m.mem.ln(x); q = m.mem.qkv(a_)[:, :768].view(-1, HA + DM.H_DEL, DM.DH)[:, :HA]; ph = Fn.elu(q) + 1
            r = (dl * ua / ga).view(HA, DM.DH)                              # target read change per head (before the gain)
            dS = ph[..., :, None] * r[None, :, None, :] / (ph ** 2).sum(-1)[..., None, None]
            return torch.cat([S[:, :HA] + dS.to(S.dtype), S[:, HA:]], 1)
        return f
    def del_edit(dl, k, rand_dir=None):
        def f(S, t):
            x = _branch_in(m, S, t); a_ = m.mem.ln(x); q, kk, _ = m.mem.qkv(a_).split(768, -1); B_, T_ = x.shape[:2]
            q = q.view(B_, T_, HA + DM.H_DEL, DM.DH)[:, :, HA:]; kk = Fn.normalize(kk.view(B_, T_, HA + DM.H_DEL, DM.DH)[:, :, HA:], dim=-1)
            g_ = m.mem.gate(a_); al = torch.sigmoid(g_[..., :DM.H_DEL]); be = 2 * torch.sigmoid(g_[..., DM.H_DEL:])
            z = Fn.normalize(q[:, -1], dim=-1)                              # (B, H, DH): z = A_0 ... A_{T-1} q_d
            if getattr(m.mem, "write_mode", None) is None:                  # §155 (R): the read sees the pass-start state, z = q_d
                for tt in range(T_ - 1, -1, -1):
                    kd = kk[:, tt]; z = al[:, tt][..., None] * (z - be[:, tt][..., None] * kd * (kd * z).sum(-1, keepdim=True))
            if rand_dir is not None:                                        # propagation's random control: same Frobenius norm, random rank-1
                zr = torch.randn_like(z); rr = torch.randn(z.shape[0], DM.H_DEL, DM.DH, device=z.device)
                dS = (dl * wc[k] / gd).view(DM.H_DEL, DM.DH)[None, :, :, None] * z[..., None, :] / (z ** 2).sum(-1).clamp_min(1e-12)[..., None, None]
                nrm = dS.flatten(1).norm(dim=1); dR = rr[..., :, None] * zr[..., None, :]; dR = dR * (nrm / dR.flatten(1).norm(dim=1).clamp_min(1e-12))[:, None, None, None]
                return torch.cat([S[:, :HA], S[:, HA:] + dR.to(S.dtype)], 1)
            r = (dl * wc[k] / gd).view(DM.H_DEL, DM.DH)
            dS = r[None, :, :, None] * z[..., None, :] / (z ** 2).sum(-1).clamp_min(1e-12)[..., None, None]   # S_del is (v x k): dS z = r
            return torch.cat([S[:, :HA], S[:, HA:] + dS.to(S.dtype)], 1)
        return f
    base = run(None); out = {"additive": {}, "delta": {}, "carry_probe_r2_score_half": r2c, "descriptive": True}
    for dl in (() if only_prop else deltas):
        bya, byd = {}, {}
        for k in cols:
            ck, cn = C[:, k - 1], C[:, k]
            g_ = run((int(et[k - 1]), add_edit(dl)))
            bya[str(k)] = {"shifted_share": float((g_[:, k - 1] == (ck + dl) % 10).mean()), "unedited_share": float((base[:, k - 1] == ck % 10).mean()),
                           "carry_follow_share": float((g_[:, k] == (cn - ck // 10 + (ck + dl) // 10) % 10).mean()),
                           "next_unchanged_share": float((g_[:, k] == base[:, k]).mean()), "n": int(N)}
            g_ = run((int(first[k]), del_edit(dl, k)))
            byd[str(k + 1)] = {"carry_shift_share": float((g_[:, k] == (cn + dl) % 10).mean()), "unedited_share": float((base[:, k] == cn % 10).mean()),
                               "prev_unchanged_share": float((g_[:, k - 1] == base[:, k - 1]).mean()), "n": int(N)}
        out["additive"]["%+d" % dl] = {"shifted_share": float(np.mean([v["shifted_share"] for v in bya.values()])),
                                       "unedited_share": float(np.mean([v["unedited_share"] for v in bya.values()])),
                                       "carry_follow_share": float(np.mean([v["carry_follow_share"] for v in bya.values()])), "n": int(N * len(bya)), "by_column": bya}
        out["delta"]["%+d" % dl] = {"carry_shift_share": float(np.mean([v["carry_shift_share"] for v in byd.values()])), "by_next_column": byd}
    pr = {"by_k": {}}; tot = {"prop": 0, "rand": 0, "null": 0, "n": 0, "coup": 0, "coup_r": 0}
    for k in prop_cols:                                                     # §155 / §154 propagation: carry edit at column k+1's first pair -> column k+2
        if k + 1 >= C.shape[1]:
            continue
        c1, c2 = C[:, k], C[:, k + 1]; byd = {}
        for dl in (-1, 1):
            chg = (c1 + dl) // 10 != c1 // 10; tgt = (c2 - c1 // 10 + (c1 + dl) // 10) % 10
            if not chg.any():
                byd["%+d" % dl] = {"n": 0}; continue
            g_ = run((int(first[k]), del_edit(dl, k))); torch.manual_seed(15502 + k * 10 + dl); gr_ = run((int(first[k]), del_edit(dl, k, rand_dir=True)))
            ps, rs, nv = (g_[chg, k + 1] == tgt[chg]), (gr_[chg, k + 1] == tgt[chg]), (base[chg, k + 1] == tgt[chg])
            oc = [c_ for c_ in range(C.shape[1]) if c_ not in (k, k + 1)]  # note 155-2: any non-target column (not k+1 / k+2, 1-based) changes its digit
            cp_ = (g_[chg][:, oc] != base[chg][:, oc]).any(1); cr_ = (gr_[chg][:, oc] != base[chg][:, oc]).any(1)
            byd["%+d" % dl] = {"propagation_share": float(ps.mean()), "random_edit_share": float(rs.mean()), "null_value": float(nv.mean()), "n": int(chg.sum()),
                               "coupling_share": float(cp_.mean()), "random_coupling_share": float(cr_.mean())}
            tot["prop"] += int(ps.sum()); tot["rand"] += int(rs.sum()); tot["null"] += int(nv.sum()); tot["n"] += int(chg.sum())
            tot["coup"] += int(cp_.sum()); tot["coup_r"] += int(cr_.sum())
        pr["by_k"][str(k)] = byd
    pr["pooled"] = {"propagation_share": tot["prop"] / max(tot["n"], 1), "random_edit_share": tot["rand"] / max(tot["n"], 1), "null_value": tot["null"] / max(tot["n"], 1), "n": tot["n"],
                    "coupling_share": tot["coup"] / max(tot["n"], 1), "random_coupling_share": tot["coup_r"] / max(tot["n"], 1)}
    pr["def"] = ("delta-head carry edit at column k+1's first pair (k = 2..5, delta +-1, minimum-norm rank 1 moving column k+1's carry-in probe by delta); "
                 "column k+2's digit vs (c^_{k+2} - r_{k+1} + floor((c^_{k+1} + delta)/10)) mod 10 on rows where the edit changes r_{k+1}; "
                 "random_edit: an equal-norm random rank-1 edit; null_value: the unedited digit already equals the edited-chain value")
    out["propagation"] = pr
    out["def"] = c7_state_edit.__doc__
    return out


def acc_by_column_carry(m, A, Bd, dev, amp, bs):
    """§154 (Theory 8596a342): per column (1-based; the last = the flush digit), the emitted digit's accuracy split by carry-in = 0 / != 0."""
    acc, _, _, pairs, col, emit = accumulators(A, Bd); ci = carry_in(A, Bd); et = np.nonzero(emit)[0]; got = rollout_delta(m, A, Bd, dev, amp, bs)[0]
    P, _, _ = columns(A, Bd); out = {}
    for k in range(len(et)):
        c_ = ci[:, k] if k < ci.shape[1] else acc[:, et[k - 1]] // 10; ok = got[:, k] == P[:, k]
        out[str(k + 1)] = {"carry0": float(ok[c_ == 0].mean()) if (c_ == 0).any() else None, "carry_nonzero": float(ok[c_ != 0].mean()) if (c_ != 0).any() else None,
                           "n_carry0": int((c_ == 0).sum()), "n_carry_nonzero": int((c_ != 0).sum())}
    return out


@torch.no_grad()
def alpha_by_flag(m, A, Bd, dev, amp, n=256):
    """§154 (Theory 8596a342): per delta head, the mean alpha over new-column segments vs same-column segments vs the flush (fixed probe batch)."""
    was = m.training; m.eval(); m.mem.alpha_rec = []; A, Bd = A[:n], Bd[:n]; set_ops(m, A, Bd, dev); pairs, _, _ = schedule(A.shape[1], Bd.shape[1]); S = m.start(len(A), dev)
    for t in range(len(pairs)):
        with amp_ctx(amp):
            _, S = m.step(S, t)
    rec = torch.stack(m.mem.alpha_rec).cpu().numpy(); m.mem.alpha_rec = None; m.train(was); nc = np.asarray(m.newcol); fl = np.array([p_[0] < 0 for p_ in pairs])
    grp = {"new_column": nc & ~fl, "same_column": ~nc & ~fl, "flush": fl}
    return {k: [round(float(v), 6) for v in rec[g_].mean(0)] for k, g_ in grp.items() if g_.any()}


def ci_or_flush(acc, ci, et, k):
    """carry into 0-based column k; for the flush column (k = n_cols) the final carry = c^ at the flush."""
    return (acc[:, et[k]] if k >= ci.shape[1] else ci[:, k]).astype(float)


def write_grad_norm(m):
    """§155 (note 155-1 / Audit): the write rows' gradient norm before clipping: LN_w, qkv's k / v rows ([768:]) and the gate (the read's q rows excluded)."""
    D = 768; g_ = []
    for p_, sl in ((m.mem.ln_w.weight, None), (m.mem.ln_w.bias, None), (m.mem.qkv.weight, slice(D, 3 * D)), (m.mem.qkv.bias, slice(D, 3 * D)),
                   (m.mem.gate.weight, None), (m.mem.gate.bias, None)):
        if p_.grad is not None:
            g_.append((p_.grad[sl] if sl is not None else p_.grad).float().norm())
    return float(torch.stack(g_).norm()) if g_ else 0.0


@torch.no_grad()
def r_caps(m, A, Bd, dev, amp, n=128):
    """§155: per step (h6, read contribution, write input u) on a fixed batch, R mode (SHIFT reset applied as in step)."""
    A, Bd = A[:n], Bd[:n]; set_ops(m, A, Bd, dev); pairs, _, _ = schedule(A.shape[1], Bd.shape[1]); S = m.start(len(A), dev); out = []
    for t in range(len(pairs)):
        if getattr(m, "add_reset", None) == "column" and t > 0 and m.newcol[t]:
            S = torch.cat([torch.zeros_like(S[:, :DM.H_ADD]), S[:, DM.H_ADD:]], 1)
        cap = {}
        with amp_ctx(amp):
            _, S = DM.gpt_forward_mem_r(m.g, m.g.embed(m.tokens(t, len(A), dev)), m.mem, S, cap=cap)
        a6 = m.mem.ln(cap["h6"].float()); u = (1 - m.mem.gamma) * a6 + m.mem.gamma * m.mem.ln_w(cap["hw"].float())
        out.append((a6.flatten(0, 1).cpu().numpy(), cap["read"].float().flatten(0, 1).cpu().numpy(), u.flatten(0, 1).cpu().numpy(), S.clone()))
    return out


def _multi_r2(Xf, Yf, Xs, Ys, lam=1e-2):
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6; Zf = np.c_[(Xf - mu) / sd, np.ones(len(Xf))]; Zs = np.c_[(Xs - mu) / sd, np.ones(len(Xs))]
    W = np.linalg.solve(Zf.T @ Zf + lam * len(Zf) * np.eye(Zf.shape[1]), Zf.T @ Yf); r = Ys - Zs @ W
    return float(1 - (r ** 2).sum() / max(((Ys - Ys.mean(0)) ** 2).sum(), 1e-12))


def s155_recurrence(m, A, Bd, dev, amp, n=128, rows=4000, seed=15501):
    """§155 per window (fixed batch): write_var_from_read_r2 / write_var_from_input_r2 (held-out ridge, pooled-variance R^2 of the write input u
    from the read contribution / from LN_6(h_6), over positions of all steps, `rows` sampled) and rho_jvp (median ||J v|| over rows and steps,
    J = dS_{t+1} / dS_t, v a random unit direction per row; torch.func.jvp, fp32, autocast off)."""
    was = m.training; m.eval(); caps = r_caps(m, A, Bd, dev, amp, n); a6 = np.concatenate([c[0] for c in caps]); rd = np.concatenate([c[1] for c in caps]); u = np.concatenate([c[2] for c in caps])
    rng = np.random.default_rng(seed); ix = rng.permutation(len(u))[:rows]; h = len(ix) // 2; f_, s_ = ix[:h], ix[h:]
    out = {"write_var_from_read_r2": _multi_r2(rd[f_], u[f_], rd[s_], u[s_]), "write_var_from_input_r2": _multi_r2(a6[f_], u[f_], a6[s_], u[s_])}
    ar = np.concatenate([a6, rd], 1); out["write_var_from_input_read_r2"] = _multi_r2(ar[f_], u[f_], ar[s_], u[s_])   # Audit (3): incremental
    out["write_var_incremental_read_r2"] = out["write_var_from_input_read_r2"] - out["write_var_from_input_r2"]
    try:
        from torch.func import jvp
        nj = min(32, caps[0][3].shape[0]); rh = []; rl = []; set_ops(m, A[:nj], Bd[:nj], dev)
        for t in range(1, len(caps), 2):
            S0 = caps[t - 1][3][:nj].float(); v = torch.randn_like(S0); v = v / v.flatten(1).norm(dim=1)[:, None, None, None]
            if getattr(m, "add_reset", None) == "column" and m.newcol[t]:
                S0 = torch.cat([torch.zeros_like(S0[:, :DM.H_ADD]), S0[:, DM.H_ADD:]], 1); v = torch.cat([torch.zeros_like(v[:, :DM.H_ADD]), v[:, DM.H_ADD:]], 1)
                v = v / v.flatten(1).norm(dim=1).clamp_min(1e-12)[:, None, None, None]
            x = m.g.embed(m.tokens(t, nj, dev))
            try:                                                            # SDPA's fused kernels lack forward AD: the math backend (same function)
                from torch.nn.attention import SDPBackend, sdpa_kernel
                with sdpa_kernel(SDPBackend.MATH):
                    _, Jv = jvp(lambda S_: DM.gpt_forward_mem_r(m.g, x, m.mem, S_)[1], (S0,), (v,))
                meth = "torch.func.jvp (SDPA math backend), fp32"
            except Exception:                                               # noqa: BLE001
                eps_ = 1e-3 * float(S0.flatten(1).norm(dim=1).median().clamp_min(1.0)); f_ = lambda S_: DM.gpt_forward_mem_r(m.g, x, m.mem, S_)[1]  # noqa: E731
                Jv = (f_(S0 + eps_ * v) - f_(S0 - eps_ * v)) / (2 * eps_); meth = "central difference (eps 1e-3 x median ||S||), fp32"
            rh.append(Jv.detach().flatten(1).norm(dim=1).cpu().numpy())
            fd_ = m.mem.fade.clone(); m.mem.fade.fill_(0.0)                 # note 155-1 (2): rho_loop = ||Jv - Jv|fade 0|| (the read -> write part)
            try:                                                            # Audit: fade is always restored (an end-of-run failure would zero later reads)
                try:
                    from torch.nn.attention import SDPBackend, sdpa_kernel
                    with sdpa_kernel(SDPBackend.MATH):
                        _, Jv0 = jvp(lambda S_: DM.gpt_forward_mem_r(m.g, x, m.mem, S_)[1], (S0,), (v,))
                    meth0 = "torch.func.jvp (SDPA math backend), fp32"
                except Exception:                                           # noqa: BLE001
                    eps_ = 1e-3 * float(S0.flatten(1).norm(dim=1).median().clamp_min(1.0)); f_ = lambda S_: DM.gpt_forward_mem_r(m.g, x, m.mem, S_)[1]  # noqa: E731
                    Jv0 = (f_(S0 + eps_ * v) - f_(S0 - eps_ * v)) / (2 * eps_); meth0 = "central difference (eps 1e-3 x median ||S||), fp32"
            finally:
                m.mem.fade.copy_(fd_)
            rl.append((Jv - Jv0).detach().flatten(1).norm(dim=1).cpu().numpy())
        out["rho_jvp"] = float(np.median(np.concatenate(rh))); out["rho_loop"] = float(np.median(np.concatenate(rl))); out["rho_method"] = meth; out["rho_loop_method_fade0"] = meth0
    except Exception as e:                                                  # noqa: BLE001
        out["rho_jvp"] = None; out["rho_loop"] = None; out["rho_method"] = "jvp failed: %s" % str(e)[:120]
    m.train(was); return out


# ================================================================ THEORY §158 FREE-PAIRS (c94ee2bf + note 158-1)
def free_steps(rng, n, na, nb):
    """n free chains: every pair's digits drawn independently (the same (i, j) schedule and flags). -> PA, PB (n, T) digits per step (the flush 0)."""
    pairs, _, _ = schedule(na, nb); T = len(pairs); PA = rng.integers(0, 10, (n, T)); PB = rng.integers(0, 10, (n, T)); fl = np.array([p_[0] < 0 for p_ in pairs])
    PA[:, fl] = 0; PB[:, fl] = 0
    return PA, PB


def steps_of(A, Bd):
    """consistent chains as per-step digits (a_i, b_j of each pair)."""
    pairs, _, _ = schedule(A.shape[1], Bd.shape[1]); ia = np.array([max(int(p_[0]), 0) for p_ in pairs]); jb = np.array([max(int(p_[1]), 0) for p_ in pairs])
    fl = np.array([p_[0] < 0 for p_ in pairs]); PA = A[:, ia].copy(); PB = Bd[:, jb].copy(); PA[:, fl] = 0; PB[:, fl] = 0
    return PA, PB


def acc_steps(PA, PB, na, nb):
    """§158 labels from per-step digits, as accumulators(): acc after each step (new column: carry = acc // 10, then + this pair's product), the
    emitted digit at emission steps, and the per-step product. = the digits of sum_t a^(t) b^(t) 10^(i_t + j_t), LSB first."""
    pairs, col, emit = schedule(na, nb); N, T = PA.shape; acc = np.zeros((N, T), np.int64); dig = np.full((N, T), -1, np.int64); a = np.zeros(N, np.int64)
    for t in range(T):
        if t > 0 and col[t] != col[t - 1]:
            a = a // 10
        if pairs[t][0] >= 0:
            a = a + PA[:, t].astype(np.int64) * PB[:, t]
        acc[:, t] = a
        if emit[t]:
            dig[:, t] = a % 10
    return acc, dig, pairs, col, emit


def label_check(PA, PB, na, nb):
    """§158 gate: the emitted digits equal the digits of sum_t a^(t) b^(t) 10^(i_t + j_t) (exact)."""
    _, dig, pairs, col, emit = acc_steps(PA, PB, na, nb); v = np.zeros(len(PA), np.int64)
    for t, (i, j) in enumerate(pairs):
        if i >= 0:
            v = v + PA[:, t].astype(np.int64) * PB[:, t] * 10 ** (int(i) + int(j))
    want = np.stack([(v // 10 ** k) % 10 for k in range(na + nb)], 1); got = np.stack([dig[:, t] for t in range(len(pairs)) if emit[t]], 1)
    return bool((got == want).all())


@torch.no_grad()
def rollout_free(m, PA, PB, na, nb, dev, amp, bs, override=None, keep_pos=()):
    """rollout_delta over per-step digits (free or consistent chains): (N, na + nb) emitted digits, kept states."""
    N = len(PA); pairs, col, emit = schedule(na, nb); T = len(pairs); out = np.full((N, na + nb), -1); kept = {p_: np.zeros((N, 12, DM.DH, DM.DH), np.float32) for p_ in keep_pos}
    Az, Bz = np.zeros((1, na), np.int64), np.zeros((1, nb), np.int64)
    for s_ in range(0, N, bs):
        sl = slice(s_, s_ + bs); Bn = len(PA[sl]); set_ops(m, np.repeat(Az, Bn, 0), np.repeat(Bz, Bn, 0), dev)
        m.pa = torch.tensor(np.array(DTOK)[PA[sl]], device=dev); m.pb = torch.tensor(np.array(DTOK)[PB[sl]], device=dev); S = m.start(Bn, dev)
        for t in range(T):
            with amp_ctx(amp):
                lg, S = m.step(S, t)
            if override is not None and override[0] == t + 1:
                S = torch.tensor(override[1][sl], device=dev, dtype=S.dtype)
            if emit[t]:
                out[sl, col[t]] = m.fam.decode(lg.argmax(-1).cpu().numpy()[:, None])
            if t + 1 in kept:
                kept[t + 1][sl] = S.float().cpu().numpy()
        m.pa = m.pb = None
    return out, kept


def s158_extras(m, teA, teB, dev, amp, bs, n_free=1000, n_fit=8000, seed=15801, points=(4, 8, 12)):
    """§158 descriptive: free_pairs_test_acc (1,000 held-out free chains, row level); by_item_type {consistent, free}: acc, row-level donor
    (the donor's whole state after pair p = 4 / 8 / 12; expected = the recipient's later digits carried from the donor's accumulator), and the
    propagation edit (c7_state_edit's, on the free chains via per-step digits); acc_vs_raw from S after pairs 4 / 8 / 12 (near-OLS, dual, CV lambda,
    n_fit free chains -> the held-out free chains): acc_exact (rounded partial column sums) and raw_digit_acc (10-way, the seen a^(t) / b^(t)),
    per point and pooled."""
    na, nb = teA.shape[1], teB.shape[1]; rng = np.random.default_rng(seed); FA, FB = free_steps(rng, n_free, na, nb); facc, fdig, pairs, col, emit = acc_steps(FA, FB, na, nb)
    et = np.nonzero(emit)[0]; Ff = np.stack([fdig[:, t] for t in et], 1); got = rollout_free(m, FA, FB, na, nb, dev, amp, bs)[0]; out = {"free_pairs_test_acc": float((got == Ff).all(1).mean())}
    CA, CB = steps_of(teA, teB); cacc, cdig, _, _, _ = acc_steps(CA, CB, na, nb); Cf = np.stack([cdig[:, t] for t in et], 1); cg = rollout_free(m, CA, CB, na, nb, dev, amp, bs)[0]
    bt = {}
    for nm, PA_, PB_, acc_, Dg in (("consistent", CA, CB, cacc, Cf), ("free", FA, FB, facc, Ff)):
        g_ = cg if nm == "consistent" else got; r_ = {"acc": float((g_ == Dg).all(1).mean()), "n": int(len(PA_))}; drng = np.random.default_rng(seed + 1); N = min(1000, len(PA_))
        ir, idn = drng.permutation(len(PA_))[:N], drng.permutation(len(PA_))[:N]; _, kept = rollout_free(m, PA_[idn], PB_[idn], na, nb, dev, amp, bs, keep_pos=[p_ for p_ in points if p_ < len(pairs)])
        dn = {}
        for P in kept:
            gp = rollout_free(m, PA_[ir], PB_[ir], na, nb, dev, amp, bs, override=(P, kept[P]))[0]
            a0 = acc_[idn, P - 1].copy(); exp_ = Dg[ir].copy(); a = a0; started = False
            for t in range(P, len(pairs)):                                    # the recipient's later pairs, carried from the donor's accumulator
                if col[t] != col[t - 1]:
                    a = a // 10
                if pairs[t][0] >= 0:
                    a = a + PA_[ir, t].astype(np.int64) * PB_[ir, t]
                if emit[t]:
                    exp_[:, col[t]] = a % 10
            later = sorted({int(col[t]) for t in range(P, len(pairs)) if emit[t]})
            dn["pos%d" % P] = {"row_level": {"donor_consistent": float((gp[:, later] == exp_[:, later]).all(1).mean()), "recipient_consistent": float((gp[:, later] == Dg[ir][:, later]).all(1).mean()),
                                             "chance_overlap": float((exp_[:, later] == Dg[ir][:, later]).all(1).mean())}, "n": int(N)}
        r_["donor"] = dn; bt[nm] = r_
        if nm == "free":                                                    # note 158-1: the propagation edit on the free chains (descriptive)
            Az, Bz = np.zeros((len(PA_), na), np.int64), np.zeros((len(PA_), nb), np.int64)
            r_["propagation"] = c7_state_edit(m, Az, Bz, dev, amp, bs, pd=(PA_, PB_), only_prop=True)["propagation"]
    out["by_item_type"] = bt
    frng = np.random.default_rng(seed + 2); TA, TB = free_steps(frng, n_fit, na, nb); _, Ktr = rollout_free(m, TA, TB, na, nb, dev, amp, bs, keep_pos=list(points))
    _, Kte = rollout_free(m, FA, FB, na, nb, dev, amp, bs, keep_pos=list(points)); avr = {}; ex_all, rw_all = [], []
    for P in points:
        Xf = Ktr[P].reshape(n_fit, -1); Xs = Kte[P].reshape(n_free, -1); csf = np.zeros((n_fit, na + nb)); cst = np.zeros((n_free, na + nb)); r_ = {"acc_exact": {}, "raw_digit_acc": {}}
        for t in range(P):
            i, j = (int(v) for v in pairs[t]); csf[:, i + j] += TA[:, t] * TB[:, t]; cst[:, i + j] += FA[:, t] * FB[:, t]
        for c in sorted({int(pairs[t][0]) + int(pairs[t][1]) for t in range(P)}):
            pr_, lam_ = _dual_fit_pred(Xf, csf[:, c], Xs, dev); e_ = float((np.rint(pr_) == cst[:, c]).mean()); r_["acc_exact"][str(c + 1)] = {"exact": e_, "lambda": lam_}; ex_all.append(e_)
        for t in range(P):
            for nm, Dtr, Dte in (("a", TA, FA), ("b", TB, FB)):
                pr_, lam_ = _dual_fit_pred(Xf, np.eye(10)[Dtr[:, t]], Xs, dev, cls=True); e_ = float((pr_.argmax(1) == Dte[:, t]).mean()); r_["raw_digit_acc"]["%s_pair%d" % (nm, t + 1)] = {"acc": e_, "lambda": lam_}; rw_all.append(e_)
        r_["acc_exact_mean"] = float(np.mean([v["exact"] for v in r_["acc_exact"].values()])); r_["raw_digit_acc_mean"] = float(np.mean([v["acc"] for v in r_["raw_digit_acc"].values()]))
        avr["after_pair%d" % P] = r_
    avr["pooled"] = {"acc_exact": float(np.mean(ex_all)), "raw_digit_acc": float(np.mean(rw_all)), "n_fit": int(n_fit), "n_score": int(n_free)}
    out["acc_vs_raw"] = avr; out["label_check_free"] = label_check(FA, FB, na, nb); out["def"] = s158_extras.__doc__
    return out


def s154_extras(m, teA, teB, dev, amp, bs, ext_n, shift):
    """§154 descriptive block: colsum_r2_by_segment (4x4), carry_probe_by_column {4x4, 4x8}, c7_edit (SHIFT)."""
    acc, _, _, pairs, col, emit = accumulators(teA, teB); R = reads_by_step(m, teA, teB, dev, amp, bs)[:, :, :DM.H_ADD * DM.DH]; h = len(teA) // 2; out = {}
    out["colsum_r2_by_segment"] = [ridge_r2(R[:h, t], acc[:h, t].astype(float), R[h:, t], acc[h:, t].astype(float)) if acc[:h, t].std() > 0 and acc[h:, t].std() > 0 else None
                                   for t in range(len(pairs))]
    out["colsum_def"] = "held-out ridge R^2 of the current column's running sum (incl. carry-in) from the additive read (384-d) at each segment's last token"
    xA, xB = gen_operands(np.random.default_rng(14004), ext_n, 4), gen_operands(np.random.default_rng(14005), ext_n, 8)
    out["carry_probe_by_column"] = {"4x4": carry_probe(m, teA, teB, dev, amp, bs), "4x8": carry_probe(m, xA, xB, dev, amp, bs),
                                    "def": "held-out ridge R^2 of column k's carry-in floor(c_{k-1}/10) from the delta read (384-d) at column k's first pair, k = 1.."}
    out["c7_edit"] = c7_edit(m, teA, teB, dev, amp, bs) if shift else None
    return out


def sched_tau(step, spec, sc=1.0):
    """§150 --slot_tau t0:t1:a:b: tau = t0 until a, linear to t1 at b, t1 after."""
    t0, t1, a_, b_ = (float(v) for v in spec.split(":")); a_, b_ = a_ * sc, b_ * sc
    return t0 if step <= a_ else (t1 if step >= b_ else t0 + (t1 - t0) * (step - a_) / (b_ - a_))


def sched_hard(step, spec, sc=1.0):
    """§150 --slot_hard a:b: the straight-through fraction p_hard, 0 until a, linear to 1 at b."""
    a_, b_ = (float(v) for v in spec.split(":")); a_, b_ = a_ * sc, b_ * sc
    return 0.0 if step <= a_ else (1.0 if step >= b_ else (step - a_) / (b_ - a_))


class MultVocab(MultSlot):
    """§150 VOCAB-SLOT: §145's K write positions; slot_k is a token of GPT-2's own vocabulary (FULL: all 50,257; DIGIT: the 10 digit tokens +
    the blank " _"). logits_k = ln_f(h(w_k)) . wte.detach()[V]^T (g's output is already after ln_f); training: Gumbel-softmax at tau, soft
    slot = p . wte.detach()[V]; straight-through (one-hot forward, soft backward) on a per-row fraction p_hard; test: hard argmax, no noise
    (soft test: no Gumbel, tau_final). All K slots are carried; the start is K blank (DIGIT) / " _" (FULL) tokens."""
    def __init__(self, g, fam, K, space):
        super().__init__(g, fam, K, K); self.space = space; dev = g.w("wte.weight").device
        V = torch.tensor(DTOK + [BLANK] if space == "digits" else list(range(g.w("wte.weight").shape[0])), device=dev)
        self.register_buffer("V", V); self.tau = 1.0; self.p_hard_slot = 0.0; self.soft_test = False; self.tau_final = 0.3
        self.last_ids = None; self.last_p = None; self.zdim = 768

    def Wv(self):
        w_ = self.g.w("wte.weight").detach()                                # note 150-1: the slot path never moves the vocabulary
        return w_ if self.space == "vocab" else w_[self.V]                  # FULL: the view itself (no 154 MB copy per hand-off)

    def start(self, B, dev, states=None):
        i0 = int((self.V == BLANK).nonzero()[0, 0]) if self.space == "digits" else BLANK
        e = self.g.w("wte.weight").detach()[BLANK]; self.last_ids = None
        return e[None, None].expand(B, self.K, -1).float().clone()

    def write(self, prev, t, noise=None):
        g = self.g; B = prev.shape[0]; dev = prev.device; eot = g.embed(torch.full((B, 1), EOT, device=dev)); i, j = (int(v) for v in self.pairs[t])
        if i < 0:
            tail = torch.full((B, 1), EQTOK, device=dev)
        else:
            tail = torch.cat([self.adig[:, i:i + 1], self.bdig[:, j:j + 1], torch.full((B, 1), NEWTOK if self.newcol[t] else SAMETOK, device=dev)], 1)
        x = torch.cat([eot, prev.to(eot.dtype), g.embed(tail), self.w[None].expand(B, -1, -1).to(eot.dtype)], 1); h, _ = g(x, None)
        Hw = h[:, -self.K:].float()
        if noise is not None:                                               # contraction: perturb h(w_k) before the head (note 150-1 (2))
            Hw = Hw + noise
        with torch.autocast("cuda", enabled=False):                         # fp32 slot head, no bf16 copy of W per hand-off
            return self._head(Hw, B, dev)

    def _head(self, Hw, B, dev):
        W = self.Wv(); W = W if W.dtype == torch.float32 else W.float(); lg = Hw @ W.T   # (B, K, |V|)
        if self.training and not self.soft_test:
            gum = -torch.log((-torch.log(torch.rand_like(lg).clamp_min(1e-10))).clamp_min(1e-10))   # Gumbel(0, 1)
            p = torch.softmax((lg + gum) / self.tau, -1); ids = p.argmax(-1)
            if self.p_hard_slot > 0:                                         # straight-through on a per-row fraction
                hard = (torch.rand(B, 1, 1, device=dev) < self.p_hard_slot).float(); oh = Fn.one_hot(ids, W.shape[0]).float()
                p = hard * (oh - p).detach() + hard * p + (1 - hard) * p
            E = p @ W
        elif self.soft_test:
            p = torch.softmax(lg / self.tau_final, -1); ids = p.argmax(-1); E = p @ W
        else:
            p = torch.softmax(lg, -1); ids = lg.argmax(-1); E = W[ids]
        self.last_ids = ids; self.last_p = p; self.last_lg = lg
        return Hw, E

    def embed_ids(self, ids):
        return self.Wv().float()[ids]


@torch.no_grad()
def rollout_vocab(m, A, Bd, dev, amp, bs, override=None, record=False, extra=False):
    """§150 hard chains; override (t, ids (N, K)) replaces every slot token after step t (1-based). record -> slot token ids (N, T, K) (indices
    into V) and slot embeddings (N, T, K, 768); extra -> per hand-off slot entropy / top-2 margin."""
    N = len(A); pairs, col, emit = schedule(A.shape[1], Bd.shape[1]); T = len(pairs); out = np.full((N, A.shape[1] + Bd.shape[1]), -1)
    ids_all = np.zeros((N, T, m.K), np.int64) if (record or extra) else None; E_all = np.zeros((N, T, m.K, 768), np.float32) if record else None
    ent = np.zeros((N, T, m.K), np.float32) if extra else None; mar = np.zeros((N, T, m.K), np.float32) if extra else None
    for s_ in range(0, N, bs):
        sl = slice(s_, s_ + bs); B = len(A[sl]); set_ops(m, A[sl], Bd[sl], dev); prev = m.start(B, dev)
        for t in range(T):
            with amp_ctx(amp):
                _, E = m.write(prev, t)
            ids = m.last_ids
            if override is not None and override[0] == t + 1:
                ids = torch.tensor(override[1][sl], device=dev); E = m.embed_ids(ids)
            with amp_ctx(amp):
                if emit[t]:
                    out[sl, col[t]] = m.fam.decode(m.logits(E).argmax(-1).cpu().numpy()[:, None])
            if ids_all is not None:
                ids_all[sl, t] = ids.cpu().numpy()
            if record:
                E_all[sl, t] = E.float().cpu().numpy()
            if extra:
                pp = torch.softmax(m.last_lg.float(), -1); ent[sl, t] = (-(pp * pp.clamp_min(1e-12).log()).sum(-1)).cpu().numpy()
                top2 = m.last_lg.float().topk(2, -1).values; mar[sl, t] = (top2[..., 0] - top2[..., 1]).cpu().numpy()
            prev = E.float()
    return out, ids_all, E_all, ent, mar


def donor_vocab(m, A, Bd, dev, amp, bs, rng, positions=(4, 8, 12), n=2000):
    """§150 SERIAL: all K slot tokens of a donor swapped in after pair p; the recipient's later digits with the donor's accumulator (row level)."""
    N = min(n, len(A)); ir, idn = rng.permutation(len(A))[:N], rng.permutation(len(A))[:N]; Ar, Br, Ad, Bdn = A[ir], Bd[ir], A[idn], Bd[idn]
    pairs, col, emit = schedule(A.shape[1], Bd.shape[1]); Pr, _, _ = columns(Ar, Br); accd, _, _, _, _, _ = accumulators(Ad, Bdn)
    _, dids, _, _, _ = rollout_vocab(m, Ad, Bdn, dev, amp, bs, extra=True); res = {}
    for p in positions:
        if p >= len(pairs):
            continue
        got = rollout_vocab(m, Ar, Br, dev, amp, bs, override=(p, dids[:, p - 1]))[0]; gt, sl = later_gt(Ar, Br, accd[:, p - 1], p, pairs, col, emit, Pr)
        res["pos%d" % p] = {**consist(got, gt, Pr, sl), "n": N}
    return res


def s150_extras(m, teA, teB, dev, amp, bs, pred, wte0, n=1024, seed=15001):
    """§150 descriptive: soft_test_acc, acc_by_magnitude, slot_token_hist (top 20 per slot, token text), distinct_tokens_per_slot,
    spells_accumulator_share (DIGIT: the 4 slots read LSB first, blank = 0, equal the accumulator), slot_entropy, nn_wte_dist (soft test slot
    embedding's distance to the nearest slot-vocabulary row, over that row's norm), slot_margin {median, p05} (top-1 minus top-2 logit, hard
    test), wte_drift (||wte - wte0|| / ||wte0||), s150_probe (per slot x digit R^2 of the accumulator's units / tens / hundreds from the slot
    embedding, ridge, fit on half the problems)."""
    P, _, _ = columns(teA, teB); out = {"acc_by_magnitude": acc_by_magnitude(pred, teA, teB)}
    m.soft_test = True; ps = rollout_vocab(m, teA, teB, dev, amp, bs)[0]; out["soft_test_acc"] = acc_report(ps, P)["test_acc"]
    set_ops(m, teA[:bs], teB[:bs], dev); prev = m.start(min(bs, len(teA)), dev); d_ = []
    with torch.no_grad():
        for t in range(len(schedule(teA.shape[1], teB.shape[1])[0])):
            with amp_ctx(amp):
                _, E = m.write(prev, t)
            W = m.Wv().float(); dd = torch.cdist(E.float().reshape(-1, 768), W).min(1); d_.append((dd.values / W[dd.indices].norm(dim=-1).clamp_min(1e-9)).cpu().numpy()); prev = E.float()
    m.soft_test = False; out["nn_wte_dist"] = {"median": float(np.median(np.concatenate(d_))), "p95": float(np.percentile(np.concatenate(d_), 95)), "mode": "soft test (tau_final, no Gumbel)"}
    _, ids, _, ent, mar = rollout_vocab(m, teA, teB, dev, amp, bs, extra=True); acc, _, _, pairs, col, _ = accumulators(teA, teB); Vl = m.V.cpu().numpy()
    hist = {}; dist = {}
    for k in range(m.K):
        u, c = np.unique(Vl[ids[:, :, k]].ravel(), return_counts=True); o = np.argsort(-c)[:20]
        hist["slot%d" % (k + 1)] = [[int(u[i]), bc.dec_arr([int(u[i])]), int(c[i])] for i in o]   # [token id, text, count]; dist["slot%d" % (k + 1)] = int(len(u))
    out["slot_token_hist"] = hist; out["distinct_tokens_per_slot"] = dist
    out["slot_entropy"] = {"slot%d" % (k + 1): float(ent[:, :, k].mean()) for k in range(m.K)}
    out["slot_margin"] = {"median": float(np.median(mar)), "p05": float(np.percentile(mar, 5))}
    if m.space == "digits":                                                 # index 0..9 = the digit, 10 = blank (read as 0)
        dg = np.where(ids == 10, 0, ids); val = (dg * (10 ** np.arange(m.K))[None, None]).sum(-1); out["spells_accumulator_share"] = float((val == acc).mean())
    else:
        out["spells_accumulator_share"] = None
    out["wte_drift"] = float((m.g.w("wte.weight").detach().float() - wte0).norm() / wte0.norm())
    rng = np.random.default_rng(seed); idx = rng.permutation(len(teA))[:n]; A, Bd = teA[idx], teB[idx]; ac, _, _, _, _, _ = accumulators(A, Bd)
    _, _, Q, _, _ = rollout_vocab(m, A, Bd, dev, amp, bs, record=True); h = len(A) // 2; T = ac.shape[1]
    def pool(X, y):
        return ridge_r2(X[:h].reshape(h * T, -1), y[:h].reshape(-1).astype(float), X[h:].reshape((len(A) - h) * T, -1), y[h:].reshape(-1).astype(float))
    pr = {"slot%d" % (k + 1): {nm_: pool(Q[:, :, k], (ac // 10 ** e_) % 10) for e_, nm_ in ((0, "units"), (1, "tens"), (2, "hundreds"))} for k in range(m.K)}
    return out, {"per_slot_digit_r2": pr, "r2_acc_all_slots": pool(Q.reshape(len(A), T, -1), ac), "n": int(len(A)), "steps": int(T)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="mult_pair", choices=["mult_pair"]); ap.add_argument("--task", default="4x4", choices=["4x4", "5x5"])
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--steps", type=int, default=24000); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--eval_bs", type=int, default=250); ap.add_argument("--test_n", type=int, default=1000)
    ap.add_argument("--ext_n", type=int, default=1000); ap.add_argument("--donor_n", type=int, default=2000); ap.add_argument("--contract_n", type=int, default=500)
    ap.add_argument("--contract_T0", type=int, default=2); ap.add_argument("--d_q", type=int, default=8); ap.add_argument("--levels", type=int, default=8)
    ap.add_argument("--sched_scale", type=float, default=1.0); ap.add_argument("--p_hard_start", type=int, default=2000); ap.add_argument("--p_hard_end", type=int, default=4000)
    ap.add_argument("--log_every", type=int, default=100); ap.add_argument("--stop_plateau", action="store_true"); ap.add_argument("--ckpt_every", type=int, default=0)
    ap.add_argument("--resume", default=None); ap.add_argument("--stop_after", type=int, default=0); ap.add_argument("--train_n", type=int, default=0, help="smoke only")
    ap.add_argument("--mem_check", action="store_true"); ap.add_argument("--from_args", default=None); ap.add_argument("--contraction_only", action="store_true")
    ap.add_argument("--s158_free_prop_only", action="store_true", help="§158 post-hoc: free-item propagation on <out>/model.pt -> eval_s158_posthoc.json")
    ap.add_argument("--what_state_only", action="store_true", help="note 155-4 post-hoc (descriptive): what_state on <out>/model.pt -> eval_what_state.json")
    ap.add_argument("--c7_state_only", action="store_true", help="§154 post-hoc (descriptive): C7 state edits on <out>/model.pt -> eval_c7_state.json")
    ap.add_argument("--fp32", action="store_true"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    ap.add_argument("--slots", type=int, default=0, help="§145: K write positions / slots (0 = §140's single code)"); ap.add_argument("--carry_slots", type=int, default=0)
    ap.add_argument("--carry_from", default="first", choices=["first", "last"], help="§145 note 145-1: carry the slots read at the LAST write positions (CARRY1: w_K)")
    ap.add_argument("--no_round", action="store_true", help="§145 NOISE-64 carrier"); ap.add_argument("--dither", type=float, default=0.0)
    ap.add_argument("--keep_ckpt_steps", default="", help="§151: comma list of steps; model-only copies there, the row-level donor on each at the end (s151.serial_by_ckpt), then deleted")
    ap.add_argument("--shape_mix", default="", help="§161: training chains 4 x n, n drawn per batch from this list (e.g. 2,3,4,5); off by default")
    ap.add_argument("--col_index_offset", type=int, default=None, help="§161 MIX+IDX: column-index token c + r per segment, r ~ U{0..R} per training chain")
    ap.add_argument("--probe_n_fit", type=int, default=8000, help="note 155-4 / §158 probes: train fit items (smokes only change it)")
    ap.add_argument("--free_pairs", type=float, default=0.0, help="§158 FREE-PAIRS: the share of training chains with independent per-pair digits")
    ap.add_argument("--mem_write", default=None, choices=[None, "top", "block9"], help="§155 DELTA-R: write k / v / alpha / beta from the top residual (mixed in by gamma)")
    ap.add_argument("--write_mix_start", type=int, default=2000); ap.add_argument("--write_mix_end", type=int, default=6000)
    ap.add_argument("--add_reset", default=None, choices=[None, "column"], help="§154 SHIFT: zero the additive heads' state before each new column / the flush")
    ap.add_argument("--add_rotary", default=None, choices=[None, "column"], help="§154 ROTARY (not built yet)"); ap.add_argument("--rotary_read_offsets", default=None)
    ap.add_argument("--rotary_base", type=float, default=100.0)
    ap.add_argument("--carrier", default="code", choices=["code", "deltamem"], help="§147: deltamem = the linear-recurrent memory, no carried code")
    ap.add_argument("--slot_space", default=None, choices=[None, "vocab", "digits"], help="§150 VOCAB-SLOT: slots are GPT-2 vocabulary tokens")
    ap.add_argument("--micro", type=int, default=1, help="gradient accumulation chunks (memory only; bs must divide)")
    ap.add_argument("--slot_gumbel", action="store_true"); ap.add_argument("--slot_tau", default="1:0.3:2000:6000"); ap.add_argument("--slot_hard", default="6000:8000")
    ap.add_argument("--mem_layer", type=int, default=6); ap.add_argument("--mem_add_heads", type=int, default=6); ap.add_argument("--mem_delta_heads", type=int, default=6)
    a = ap.parse_args()
    if a.from_args:
        keep = {k: getattr(a, k) for k in ("contraction_only", "fp32", "device", "out", "from_args", "c7_state_only", "what_state_only", "s158_free_prop_only")}
        for k, v in json.load(open(a.from_args)).items():
            if hasattr(a, k):
                setattr(a, k, v)
        for k, v in keep.items():
            setattr(a, k, v)
    assert not a.slots or a.slot_space or (a.no_round and 1 <= a.carry_slots <= a.slots <= 4), "§145: --slots K (<= 4) needs --no_round and 1 <= --carry_slots <= K"
    assert a.bs % a.micro == 0, "--micro must divide --bs"
    a.mem_group_gain = a.carrier == "deltamem"
    assert not a.free_pairs or (a.carrier == "deltamem" and 0 < a.free_pairs < 1), "§158: --free_pairs in (0, 1) with --carrier deltamem"
    assert not (a.add_reset or a.add_rotary or a.mem_write) or a.carrier == "deltamem", "§154 / §155: --add_reset / --add_rotary / --mem_write need --carrier deltamem"
    if not a.mem_write:
        a.write_mix_start = a.write_mix_end = None
    assert not a.add_rotary, "§154 ROTARY is not built yet"
    if not a.add_rotary:
        a.rotary_base = None                               # note 147-2 (the scorer checks it)
    a.slot_wte_detached = bool(a.slot_space)                                # note 150-1: the whole slot path (logits + mixture) uses wte.detach()
    assert not a.slot_space or (a.slots == 4 and a.slot_gumbel and not a.no_round), "§150: --slots 4 --slot_space vocab|digits --slot_gumbel (no NOISE carrier)"
    dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); sc = a.sched_scale; n = int(a.task[0])
    SHM = [int(x) for x in a.shape_mix.split(",") if x.strip()]             # §161
    assert not (SHM or a.col_index_offset is not None) or (a.task == "4x4" and a.carrier == "deltamem" and not a.free_pairs), "§161: 4x4 DeltaMem chains only"
    a.col_index_tokens = [" " + chr(65 + i) for i in range(21)] if a.col_index_offset is not None else None
    if a.d_q != F.D_Q or a.levels != F.LEV:
        F.D_Q, F.LEV = a.d_q, a.levels; F.HALF = (F.LEV - 1) / 2; F.SHIFT = math.atanh(F.OFF / F.HALF)
    memcap and memcap.apply(dev)
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    os.makedirs(a.out, exist_ok=True); fam = PairFam(); dname = lambda: torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"  # noqa: E731
    teA, teB = load_bench(os.path.join(ROOT, "bench", "data", "mult_%s_test.txt" % a.task)); teA, teB = teA[:a.test_n], teB[:a.test_n]
    assert a.carrier == "code" or (not a.slots and a.mem_add_heads == DM.H_ADD and a.mem_delta_heads == DM.H_DEL), "§147: --carrier deltamem (6 + 6 heads), no slots"
    def mk():
        if a.carrier == "deltamem":
            md_ = MultDelta(s5.load_gpt2(dev), fam, a.mem_layer, a.mem_write).to(dev); md_.add_reset = a.add_reset; md_.cidx_off = a.col_index_offset; md_.cidx_r = None; return md_
        if a.slot_space:
            m_ = MultVocab(s5.load_gpt2(dev), fam, a.slots, a.slot_space).to(dev); m_.tau_final = float(a.slot_tau.split(":")[1]); return m_
        m_ = (MultSlot(s5.load_gpt2(dev), fam, a.slots, a.carry_slots, a.carry_from) if a.slots else MultPair(s5.load_gpt2(dev), fam)).to(dev); m_.no_round = a.no_round
        return m_
    RO = (lambda *x, **k: rollout_delta(*x, **{kk: vv for kk, vv in k.items() if kk != "record"})[0]) if a.carrier == "deltamem" else (
        (lambda *x, **k: rollout_vocab(*x, **{kk: vv for kk, vv in k.items() if kk != "record"})[0]) if a.slot_space else (rollout_slots if a.slots else rollout))
    if a.s158_free_prop_only:                                               # §158 post-hoc (note 158-1): the free-item propagation
        m = mk(); m.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev)); m.eval()
        FA, FB = free_steps(np.random.default_rng(15801), 1000, n, n); Az, Bz = np.zeros((1000, n), np.int64), np.zeros((1000, n), np.int64)
        pr_ = c7_state_edit(m, Az, Bz, dev, amp, a.eval_bs, pd=(FA, FB), only_prop=True)["propagation"]
        o_ = {"s158": {"by_item_type": {"free": {"propagation": pr_, "n_chains": 1000, "seed": 15801}}}, "eval_dtype": str(amp), "device": dname(), "driver": driver_version()}
        json.dump(o_, open(os.path.join(a.out, "eval_s158_posthoc.json"), "w", newline="\n"), indent=1); print("S158POST", json.dumps(pr_["pooled"]), flush=True); return
    if a.c7_state_only:                                                     # §154 post-hoc (on-host via drain_posthoc)
        m = mk(); m.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev)); m.eval(); m.mem.fade.fill_(1.0)
        c7 = {"c7_state_edit": c7_state_edit(m, teA, teB, dev, amp, a.eval_bs), "acc_by_column_carry": acc_by_column_carry(m, teA, teB, dev, amp, a.eval_bs),
              "alpha_by_flag": alpha_by_flag(m, teA, teB, dev, amp), "eval_dtype": str(amp), "device": dname(), "driver": driver_version(), "add_reset": a.add_reset,
              "note": "§154 post-hoc (Theory 8596a342 / Audit): the shipped eval predates these fields; descriptive"}
        if a.mem_write:                                                     # §155 (note 155-2): the post-hoc propagation incl. coupling_share
            c7["s155"] = {"propagation": c7["c7_state_edit"]["propagation"]}
        json.dump(c7, open(os.path.join(a.out, "eval_s155_posthoc.json" if a.mem_write else "eval_s154_posthoc.json"), "w", newline="\n"), indent=1)
        print("S154POST", json.dumps({k: c7["c7_state_edit"][k] for k in ("additive", "delta")})[:400], flush=True); return
    if a.contraction_only:
        m = mk(); m.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev)); m.eval(); m.p_hard = 1.0
        ampc = None if a.fp32 else amp; ct = (contraction_slots if a.slots else contraction)(m, teA, teB, dev, ampc, T0=a.contract_T0, n=a.contract_n); ct.update({"eval_dtype": str(ampc), "device": dname(), "driver": driver_version()})
        json.dump(ct, open(os.path.join(a.out, "eval_contraction_fp32.json" if a.fp32 else "eval_contraction.json"), "w", newline="\n"), indent=1)
        print("CONTRACTION", json.dumps({k: ct.get(k) for k in ("rho", "rho_nonzero", "basin")}), flush=True); return
    cmax(1, a.task)                                                         # refuses an unregistered task schedule before any work
    if a.train_n:
        from boxes.multcol import load_bench_head
        trA, trB = load_bench_head(os.path.join(ROOT, "bench", "data", "mult_train", "%s_train.txt" % a.task), a.train_n)
    else:
        trA, trB = load_bench(os.path.join(ROOT, "bench", "data", "mult_train", "%s_train.txt" % a.task))
    n_raw = len(trA); trA, trB, n_rem = dedup(trA, trB, teA, teB)
    if a.what_state_only:                                                   # note 155-4 post-hoc (on-host via drain_posthoc, or a local GPU)
        m = mk(); m.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev)); m.eval()
        ws = {"what_state": what_state(m, teA, teB, trA, trB, dev, amp, a.eval_bs, n_fit=a.probe_n_fit), "model": os.path.basename(a.out.rstrip("/")), "add_reset": a.add_reset,
              "mem_write": a.mem_write, "eval_dtype": str(amp), "device": dname(), "driver": driver_version()}
        json.dump(ws, open(os.path.join(a.out, "eval_what_state.json"), "w", newline="\n"), indent=1)
        print("WHATSTATE", json.dumps({k: ws["what_state"][k] for k in ("pass_skip", "donor_operands")})[:500], flush=True); return
    info = {**vars(a), "train_items": int(len(trA)), "train_items_raw": int(n_raw), "train_removed_test_overlap": n_rem, "flush_token": " =", "pair_flags": "column_start", "flag_tokens": {"new_column": " ;", "same_column": " ,", "flush": " ="}, "train_gpu": dname(),
            "train_driver": driver_version(), **env_info()}
    json.dump({**info, "resumed_at": time.strftime("%Y-%m-%d %H:%M:%S")} if a.resume else info,
              open(os.path.join(a.out, ("args_resume_%d.json" % int(time.time())) if a.resume else "args.json"), "w", newline="\n"), indent=1)
    m = mk(); m.train(); pairs, col, emit = schedule(n, n); wte0 = m.g.w("wte.weight").detach().float().clone()
    if a.carrier == "deltamem" and not a.resume and not a.mem_check:        # note 147-2: g_add = median ||read_del|| / ||read_add|| at step 0
        rr0, med0 = read_ratio_by_handoff(m, teA, teB, dev, amp); m.mem.s_add.data.fill_(math.log(med0)); rr1, med1 = read_ratio_by_handoff(m, teA, teB, dev, amp)
        init147 = {"g_add_init": med0, "g_del_init": 1.0, "ratio_by_handoff_before_gain": rr0, "read_norm_ratio_by_handoff_step0": rr1, "pooled_median_after_gain": med1}
        json.dump(init147, open(os.path.join(a.out, "s147_init.json"), "w", newline="\n"), indent=1); print("S147_INIT", json.dumps(init147), flush=True)
    if a.carrier == "deltamem":                                             # §147: the branch's own AdamW group, 300-step warm-up (warm_from 0)
        mp = {id(p_) for p_ in m.mem.parameters()}
        opt = torch.optim.AdamW([{"params": [p_ for p_ in m.parameters() if id(p_) not in mp], "mult": 1.0},
                                 {"params": list(m.mem.parameters()), "mult": 1.0, "warm_from": 0, "name": "deltamem"}], lr=a.lr, weight_decay=0.01)
    else:
        opt = make_opt(m, a.lr)
    log = open(os.path.join(a.out, "log.jsonl"), "a" if a.resume else "w", newline="\n"); t0 = time.time(); hist = []; L127 = {}; n_nonfinite = 0; step0 = 1; alpha_hist = []; rec_hist = []; wg_hist = []; wwin = []; wlast = []   # §155
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False); m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["np_rng"]; torch.set_rng_state(ck["torch_rng"].cpu()); L127 = ck["L127"]; n_nonfinite = ck["n_nonfinite"]; rec_hist = ck.get("rec_hist") or []; wg_hist = ck.get("wg_hist") or []
        if dev.startswith("cuda") and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device=dev)
        step0 = ck["step"] + 1; t0 = time.time() - ck["elapsed"]; log.write(json.dumps({"step": ck["step"], "resumed": a.resume}) + "\n"); log.flush()
        alpha_hist = ck.get("alpha_hist", [])
    def batch(c, bs_, nb_force=None):
        """A chain over the pairs of the first c columns (+ the flush when c = 2n); labels / trace mask at the emission steps.
        §158 --free_pairs P > 0: each row is, with probability P, a free chain (every pair's digits drawn independently; label = the digits of
        sum_t a^(t) b^(t) 10^(i_t + j_t)); P = 0 draws nothing extra (bit-identical batches)."""
        ii = rng.integers(0, len(trA), bs_); A, Bd = trA[ii], trB[ii]; batch.R = None
        if SHM:                                                             # §161: one n per batch (a batch shares its step schedule); b ~ fresh n-digit operands
            nb_ = int(nb_force or SHM[int(rng.integers(0, len(SHM)))])
            if nb_ != n:
                Bd = gen_operands(rng, bs_, nb_)
            _, dig, _, _, _, _ = accumulators(A, Bd); pairs_, col_, emit_ = schedule(n, nb_); ce_ = n + nb_ if c >= 2 * n else min(c, n + nb_)   # c = 2n: the full 4 x n chain
            T = int(np.searchsorted(col_, ce_ - 1, side="right")) if ce_ < n + nb_ else len(pairs_)
        else:
            _, dig, _, _, _, _ = accumulators(A, Bd); emit_ = emit
            T = int(np.searchsorted(col, c - 1, side="right")) if c < 2 * n else len(pairs)     # steps whose column < c
        if a.col_index_offset is not None:                                  # §161 MIX+IDX: r per chain
            batch.R = rng.integers(0, a.col_index_offset + 1, bs_); m.cidx_r = torch.tensor(batch.R, device=dev)
        set_ops(m, A, Bd, dev); X = torch.tensor(np.tile(np.arange(T), (bs_, 1)), device=dev); batch.PD = None
        if a.free_pairs > 0:                                                # §158
            fr = rng.random(bs_) < a.free_pairs; PA, PB = steps_of(A, Bd); FA, FB = free_steps(rng, bs_, n, n); PA[fr] = FA[fr]; PB[fr] = FB[fr]
            _, dig, _, _, _ = acc_steps(PA, PB, n, n); m.pa = torch.tensor(np.array(DTOK)[PA], device=dev); m.pb = torch.tensor(np.array(DTOK)[PB], device=dev)
            batch.PD = (PA, PB); batch.n_free = int(fr.sum())
        tm = torch.tensor(np.tile(emit_[:T], (bs_, 1)), device=dev); lab = np.where(dig[:, :T] >= 0, dig[:, :T], 0); batch.AB = (A, Bd)
        return X, lab, tm, T
    if a.mem_check:                                                         # --micro: one chunk of bs / micro rows (what training holds at once)
        m.p_hard = 1.0; X, lab, tm, T = batch(2 * n, a.bs // a.micro, nb_force=max(SHM) if SHM else None)
        with amp_ctx(amp):
            loss, _, _, _ = m.chain(m.start(a.bs // a.micro, dev), X, lab, tm)
        loss.backward(); m.eval(); RO(m, gen_operands(rng, a.eval_bs, n), gen_operands(rng, a.eval_bs, 2 * n), dev, amp, a.eval_bs, record=not a.carrier == "deltamem")
        pk = torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.startswith("cuda") else None
        json.dump({"peak_gib": pk, "bs": a.bs, "micro": a.micro, "chunk": a.bs // a.micro, "eval_bs": a.eval_bs, "task": a.task, "slots": a.slots, "slot_space": a.slot_space, "longest_eval_chain": "4x8 (33 steps)"}, open(os.path.join(a.out, "mem_check.json"), "w", newline="\n"), indent=1); print("MEM", pk); return
    keep_steps = {int(x) for x in a.keep_ckpt_steps.split(",") if x.strip()}
    for step in range(step0, a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(a.lr, gr, step, a.steps)
        m.p_hard = 0.0 if step <= a.p_hard_start * sc else min(1.0, (step - a.p_hard_start * sc) / ((a.p_hard_end - a.p_hard_start) * sc)); m.train()
        m.noise = (a.dither if step > 2000 * sc else 0.0) if a.no_round else 0.0   # §145 NOISE-64: dither after 2k (no draws otherwise)
        if a.carrier == "deltamem":
            m.mem.fade.fill_(min(1.0, step / max(1, int(500 * sc))))         # §147: W_o fade-in over 500 steps
            if a.mem_write:                                                 # §155: gamma 0 until write_mix_start, linear to 1 at write_mix_end
                m.mem.gamma.fill_(0.0 if step <= a.write_mix_start * sc else min(1.0, (step - a.write_mix_start * sc) / max(1.0, (a.write_mix_end - a.write_mix_start) * sc)))
        if a.slot_space:                                                    # §150: Gumbel tau and the straight-through fraction
            m.tau = sched_tau(step, a.slot_tau, sc); m.p_hard_slot = sched_hard(step, a.slot_hard, sc)
        c = int(rng.integers(1, cmax(step / sc, a.task) + 1)); X, lab, tm, T = batch(c, a.bs); opt.zero_grad()
        if a.micro == 1:
            with amp_ctx(amp):
                loss, _, _, _ = m.chain(m.start(a.bs, dev), X, lab, tm)      # CE at the emission steps (mean over them)
            loss.backward(); tot = float(loss.item())
        else:                                                               # Audit (§150 FULL): gradient accumulation, memory only
            A_, B_ = batch.AB; ch = a.bs // a.micro; tot = 0.0
            for k_ in range(a.micro):
                sl_ = slice(k_ * ch, (k_ + 1) * ch); set_ops(m, A_[sl_], B_[sl_], dev)
                if batch.R is not None:                                     # §161: the chunk's r
                    m.cidx_r = torch.tensor(batch.R[sl_], device=dev)
                if batch.PD is not None:                                    # §158: the chunk's per-step digits
                    m.pa = torch.tensor(np.array(DTOK)[batch.PD[0][sl_]], device=dev); m.pb = torch.tensor(np.array(DTOK)[batch.PD[1][sl_]], device=dev)
                with amp_ctx(amp):
                    lk, _, _, _ = m.chain(m.start(ch, dev), X[sl_], lab[sl_], tm[sl_])
                (lk / a.micro).backward(); tot += float(lk.item()) / a.micro
            loss = torch.tensor(tot)
        if a.mem_write:                                                     # §155: the write path's gradient norm (before clipping)
            wwin.append(write_grad_norm(m))
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if not (math.isfinite(tot) and bool(torch.isfinite(gn))):
            bad = "loss" if not math.isfinite(tot) else next((n_ for n_, p_ in m.named_parameters() if p_.grad is not None and not bool(torch.isfinite(p_.grad).all())), "grad_norm")
            log.write(json.dumps({"step": step, "nonfinite": 1, "first": bad}) + "\n"); log.flush(); n_nonfinite += 1; opt.zero_grad()
        else:
            opt.step(); hist.append(tot)
        if step % a.log_every == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(hist)) if hist else None, "c_max": cmax(step / sc, a.task), "p_hard": m.p_hard, "sec": round(time.time() - t0)}; hist = []
            if a.carrier == "deltamem":                                     # note 147-2 (Audit): the read gains per window
                rec.update({"g_add": float(torch.exp(m.mem.s_add)), "g_del": float(torch.exp(m.mem.s_del)), "fade": float(m.mem.fade)})
            if a.mem_write and wwin:                                        # §155: write-gradient norm per window; spike vs the last 1k steps' median
                md_ = float(np.median(wlast)) if wlast else 0.0             # note 155-1: the PRECEDING 1k steps' median (current window excluded)
                w_rec = {"step": step, "mean": float(np.mean(wwin)), "max": float(np.max(wwin)), "median_prev_1k": md_,
                         "spike_100x": bool(step > a.write_mix_start * sc and np.max(wwin) > 100 * md_ > 0),   # scored only once the loop can form
                         "spike_100x_raw": bool(np.max(wwin) > 100 * md_ > 0),
                         "spike_100x_floor": bool(step > a.write_mix_start * sc and np.max(wwin) > 100 * md_ > 0 and np.max(wwin) > 1.0)}   # note 160-1 / 159-1 (33ec47e5): the scored trigger
                wlast.extend(wwin); del wlast[:-int(1000 * sc) or None]
                rec.update({"gamma": float(m.mem.gamma), "write_grad_norm": w_rec["mean"], "write_grad_spike_100x": w_rec["spike_100x"], "write_grad_spike_100x_floor": w_rec["spike_100x_floor"]}); wg_hist.append(w_rec); wwin = []
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True); L127[step] = rec["loss"]
        if a.mem_write and step % max(1, int(1000 * sc)) == 0:             # §155: recurrence + write_norm_ratio per 1k on a fixed batch
            m.mem.wstat = []; rr_ = {"step": step, "gamma": float(m.mem.gamma), **s155_recurrence(m, teA, teB, dev, amp)}
            rr_["write_norm_ratio"] = float(np.median([w_ / max(b_, 1e-12) for w_, b_ in m.mem.wstat])) if m.mem.wstat else None; m.mem.wstat = None
            rec_hist.append(rr_); log.write(json.dumps({"step": step, "s155_recurrence": rr_}) + "\n"); log.flush(); m.train()
        if step == int(2000 * sc) and L127:                                 # 123-3's flag logged for every attempt; stops only with --stop_plateau
            gl = lambda t_: L127[min(L127, key=lambda s_: abs(s_ - t_ * sc))]  # noqa: E731
            fl = bool(gl(2000) >= 0.9 * gl(1000) and gl(2000) >= 0.25 * gl(100))
            log.write(json.dumps({"step": step, "flag_123_3": fl, "loss_100": gl(100), "loss_1k": gl(1000), "loss_2k": gl(2000), "stop_plateau": a.stop_plateau}) + "\n"); log.flush()
            if fl and a.stop_plateau and (a.carrier == "deltamem" or a.slot_space) and gl(2000) < 1.0:   # note 147-1 (1) (§147, §150): a misfire
                log.write(json.dumps({"step": step, "flag_misfire_147_1": True, "loss_2k": gl(2000)}) + "\n"); log.flush(); fl = False
            if fl and a.stop_plateau:
                log.write(json.dumps({"step": step, "stopped": "plateau (123-3 flag at 2k)", "loss_100": gl(100), "loss_1k": gl(1000), "loss_2k": gl(2000)}) + "\n")
                log.close(); print("STOPPED plateau at 2k", flush=True); return
        if a.slot_space and (step % a.log_every == 0 or step == a.steps):      # §150: tau / p_hard / wte drift per window
            log.write(json.dumps({"step": step, "s150": {"tau": m.tau, "p_hard_slot": m.p_hard_slot, "wte_drift": float((m.g.w("wte.weight").detach().float() - wte0).norm() / wte0.norm())}}) + chr(10)); log.flush()
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps:
            if a.carrier == "deltamem":                                     # note 147-3: alpha per head at each checkpoint
                ah_ = {"step": step, **alpha_by_head(m, teA, teB, dev, amp)}; alpha_hist.append(ah_); log.write(json.dumps({"step": step, "alpha_by_head": ah_}) + chr(10)); log.flush()
            cp = os.path.join(a.out, "ckpt_last.pt")
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(dev) if dev.startswith("cuda") else None,
                        "np_rng": rng.bit_generator.state, "step": step, "L127": L127, "n_nonfinite": n_nonfinite, "elapsed": time.time() - t0, "alpha_hist": alpha_hist, "rec_hist": rec_hist, "wg_hist": wg_hist}, cp + ".tmp")
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            os.replace(cp + ".tmp", cp); log.write(json.dumps({"step": step, "ckpt": "ckpt_last.pt"}) + "\n"); log.flush()
        if step in keep_steps:                                              # §151: model-only copy for the end-of-run serial_by_ckpt
            torch.save(m.state_dict(), os.path.join(a.out, "ckpt_s%d_model.pt" % step)); log.write(json.dumps({"step": step, "ckpt_keep": "ckpt_s%d_model.pt" % step}) + "\n"); log.flush()
        if a.stop_after and step >= a.stop_after:
            log.close(); print("STOP_AFTER", step, flush=True); return
    train_sec = round(time.time() - t0); torch.save(m.state_dict(), os.path.join(a.out, "model.pt")); m.eval(); m.p_hard = 1.0; t1 = time.time()
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    m.noise = 0.0; Pte, _, _ = columns(teA, teB)
    if a.carrier == "deltamem" or a.slot_space:
        pred = RO(m, teA, teB, dev, amp, a.eval_bs); codes = None
    else:
        pred, codes = RO(m, teA, teB, dev, amp, a.eval_bs, record=True)[:2]
    res = acc_report(pred, Pte)
    ti = np.random.default_rng(14000).integers(0, len(trA), 1000); Ptr, _, _ = columns(trA[ti], trB[ti])
    res.update({"family": a.family, "task": a.task, "seed": a.seed, "long_acc": res["test_acc"], "train_acc": float((RO(m, trA[ti], trB[ti], dev, amp, a.eval_bs) == Ptr).all(1).mean()),
                "train_sec": train_sec, "final_loss": L127[max(L127)] if L127 else None, "n_nonfinite": n_nonfinite, "train_removed_test_overlap": n_rem, "eval_dtype": str(amp),
                "gpu": dname(), "train_gpu": info["train_gpu"], "train_driver": info["train_driver"]})
    if a.task == "4x4" and a.ext_n:
        xA, xB = gen_operands(np.random.default_rng(14004), a.ext_n, 4), gen_operands(np.random.default_rng(14005), a.ext_n, 8); xA, xB, rem = dedup(xA, xB, teA, teB)
        Px, _, _ = columns(xA, xB); res["ext_4x8"] = {**acc_report(RO(m, xA, xB, dev, amp, a.eval_bs), Px), "dedup_removed": rem, "seeds": [14004, 14005]}
    if SHM and a.ext_n:                                                     # §161: ext 4x6 / 4x7 (r = 0)
        for nb_, s1_, s2_ in ((6, 14016, 14017), (7, 14018, 14019)):
            xA, xB = gen_operands(np.random.default_rng(s1_), a.ext_n, n), gen_operands(np.random.default_rng(s2_), a.ext_n, nb_); Px, _, _ = columns(xA, xB)
            res["ext_4x%d" % nb_] = {**acc_report(RO(m, xA, xB, dev, amp, a.eval_bs), Px), "seeds": [s1_, s2_]}
    if a.task == "5x5" and a.ext_n:                                         # §159 (C2, descriptive): 5x8 on 5x5 models (40 pairs + flush)
        xA, xB = gen_operands(np.random.default_rng(14014), a.ext_n, 5), gen_operands(np.random.default_rng(14015), a.ext_n, 8); xA, xB, rem = dedup(xA, xB, teA, teB)
        Px, _, _ = columns(xA, xB); res["ext_5x8"] = {**acc_report(RO(m, xA, xB, dev, amp, a.eval_bs), Px), "dedup_removed": rem, "seeds": [14014, 14015]}
    if a.carrier == "deltamem":                                             # §147 (no contraction term: an integrator by design)
        res["donor"] = donor_delta(m, teA, teB, dev, amp, a.eval_bs, np.random.default_rng(14002), n=min(a.donor_n, 1000))
        res["compute"] = compute_pair(m, teA.shape[1], teB.shape[1])        # Theory (C3): Meter flops / sequential passes
        res["s147"] = {**s147_extras(m, teA, teB, dev, amp, a.eval_bs, pred), "mem_layer": a.mem_layer, "mem_add_heads": a.mem_add_heads, "mem_delta_heads": a.mem_delta_heads}
        i147 = json.load(open(os.path.join(a.out, "s147_init.json"))) if os.path.exists(os.path.join(a.out, "s147_init.json")) else {}
        res["s147"]["read_norm_ratio_by_handoff"] = {"step0": i147.get("read_norm_ratio_by_handoff_step0"), "end": read_ratio_by_handoff(m, teA, teB, dev, amp)[0]}
        res["s147"]["alpha_by_head"] = {"end": alpha_by_head(m, teA, teB, dev, amp), "checkpoints": alpha_hist}
        res["s147"].update({"g_add_init": i147.get("g_add_init"), "g_add": float(torch.exp(m.mem.s_add)), "g_del": float(torch.exp(m.mem.s_del)), "mem_group_gain": True, "gain_param": "exp(s)"})
        if a.add_reset or a.add_rotary or a.mem_write:                      # §154 / §155: ext_4x16 (eval only) + the s154 block
            yA, yB = gen_operands(np.random.default_rng(14006), a.ext_n, 4), gen_operands(np.random.default_rng(14007), a.ext_n, 16); Py, _, _ = columns(yA, yB)
            res["ext_4x16"] = {**acc_report(RO(m, yA, yB, dev, amp, a.eval_bs), Py), "seeds": [14006, 14007], "n": int(a.ext_n)}
            res["s154"] = {**s154_extras(m, teA, teB, dev, amp, a.eval_bs, a.ext_n, a.add_reset == "column"), "add_reset": a.add_reset, "add_rotary": a.add_rotary}
        if a.mem_write or a.add_reset:                                      # §154 / §155: the C7 state edits + propagation, inline (post-hoc for b161)
            c7_ = c7_state_edit(m, teA, teB, dev, amp, a.eval_bs)
            res["s154"].update({"c7_state_edit": c7_, "acc_by_column_carry": acc_by_column_carry(m, teA, teB, dev, amp, a.eval_bs),
                                "alpha_by_flag": alpha_by_flag(m, teA, teB, dev, amp), "propagation": c7_["propagation"]})
        if a.free_pairs:                                                    # §158 block (beside §155's)
            res["s158"] = {**s158_extras(m, teA, teB, dev, amp, a.eval_bs, n_fit=a.probe_n_fit, n_free=min(1000, max(len(teA), 20))), "free_pairs": a.free_pairs, "gpu": dname()}
        if a.mem_write:                                                     # §155 block
            with torch.no_grad():
                n_ = min(256, len(teA)); set_ops(m, teA[:n_], teB[:n_], dev); S_ = m.start(n_, dev); mn_ = {"additive": [], "delta": []}
                for t_ in range(len(schedule(teA.shape[1], teB.shape[1])[0])):
                    with amp_ctx(amp):
                        _, S_ = m.step(S_, t_)
                    mn_["additive"].append(float(S_[:, :DM.H_ADD].float().flatten(1).norm(dim=1).median())); mn_["delta"].append(float(S_[:, DM.H_ADD:].float().flatten(1).norm(dim=1).median()))
            res["s155"] = {"propagation": res["s154"]["propagation"], "recurrence": rec_hist + [{"step": "end", **s155_recurrence(m, teA, teB, dev, amp)}],
                           "mem_norm_by_pass": mn_, "write_norm_ratio": [x_ for x_ in rec_hist if "write_norm_ratio" in x_ and x_.get("gamma", 1) < 1],
                           "write_grad_norm": wg_hist, "n_nonfinite": n_nonfinite, "mem_write": a.mem_write, "write_mix": [a.write_mix_start, a.write_mix_end],
                           "gamma_final": float(m.mem.gamma), "write_grad_spikes": [x_ for x_ in wg_hist if x_.get("spike_100x")],
                           "write_grad_spikes_floor": [x_ for x_ in wg_hist if x_.get("spike_100x_floor")]}
        res["n_codes"] = None; res["eval_sec"] = round(time.time() - t1)
    elif a.slot_space:                                                      # §150 (contraction = the on-host fp32 file, perturbing h(w_k))
        res["donor"] = donor_vocab(m, teA, teB, dev, amp, a.eval_bs, np.random.default_rng(14002), n=a.donor_n)
        ex_, pr_ = s150_extras(m, teA, teB, dev, amp, a.eval_bs, pred, wte0); res["s150"] = {**ex_, "slot_space": a.slot_space, "slot_tau": a.slot_tau, "slot_hard": a.slot_hard}
        res["s150_probe"] = pr_; ct = contraction_slots(m, teA, teB, dev, amp, T0=a.contract_T0, n=a.contract_n); res.update({"rho": ct["rho"], "basin": ct["basin"], "contraction": ct["detail"]})
        res["n_codes"] = None; res["eval_sec"] = round(time.time() - t1)
    elif a.slots:                                                           # §145
        res["donor"], sw_ = donor_slots(m, teA, teB, dev, amp, a.eval_bs, np.random.default_rng(14002), n=a.donor_n)
        ct = contraction_slots(m, teA, teB, dev, amp, T0=a.contract_T0, n=a.contract_n); res.update({"rho": ct["rho"], "basin": ct["basin"], "contraction": ct["detail"]})
        ex_, pr_ = s145_extras(m, teA, teB, dev, amp, a.eval_bs, pred, codes); res["s145"] = {**ex_, "slot_swap": sw_, "slots": a.slots, "carry_slots": a.carry_slots}
        res["s145_probe"] = pr_; res["n_codes"] = None; res["eval_sec"] = round(time.time() - t1)
    else:
        res["donor"] = donor(m, teA, teB, dev, amp, a.eval_bs, np.random.default_rng(14002), n=a.donor_n)
        ct = contraction(m, teA, teB, dev, amp, T0=a.contract_T0, n=a.contract_n); res.update({"rho": ct["rho"], "rho_nonzero": ct["rho_nonzero"], "basin": ct["basin"], "contraction": ct["detail"]})
        res["s140_probe"] = s140_probe(m, teA, teB, dev, amp, a.eval_bs); res["n_codes"] = int(len(np.unique(codes.reshape(-1, codes.shape[-1]), axis=0))); res["eval_sec"] = round(time.time() - t1)
    kept_ = []
    if keep_steps and a.carrier == "deltamem":                              # §151 (note 151-1): the row-level donor (pairs 4 / 8 / 12) at 24k / 36k + final
        sb_ = {}
        for st_ in sorted(keep_steps):
            cp_ = os.path.join(a.out, "ckpt_s%d_model.pt" % st_)
            if os.path.exists(cp_):
                m.load_state_dict(torch.load(cp_, map_location=dev)); m.eval(); kept_.append(cp_)
                sb_[str(st_)] = donor_delta(m, teA, teB, dev, amp, a.eval_bs, np.random.default_rng(14002), n=min(a.donor_n, 1000))
            else:
                sb_[str(st_)] = None
        sb_["final"] = res["donor"]
        res["s151"] = {"serial_by_ckpt": sb_, "keep_ckpt_steps": sorted(keep_steps), "donor_def": "donor_delta, rng 14002, n min(donor_n, 1000), as the final donor"}
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    for cp_ in kept_:                                                       # retention: the copies served serial_by_ckpt only
        os.remove(cp_)
    for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):
        if os.path.exists(os.path.join(a.out, ck_)):
            os.remove(os.path.join(a.out, ck_))
    print("EVAL", json.dumps({k: res.get(k) for k in ("test_acc", "train_acc", "rho", "n_codes")}), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
