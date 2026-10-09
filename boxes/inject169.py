"""THEORY §169 INJECT (5fffefd1; scorer theory/score_s169.py): a task-agnostic latent harness. §155 R's DeltaMem-R (6 + 6 heads after block 6, top
writes, gamma 2k -> 6k, pass-start reads, W_o fade-in, g_add calibrated at step 0) is the ONLY cross-pass state.
    think pass t (t = 1..K) = [EOT, the FULL problem text " a_1 .. a_4 * b_1 .. b_n", " ."] - a fresh cache, the same text every pass; it reads the
                              memory then writes (no chunking, no schedule, no flags)
    decoder                 = [EOT, problem text, " ="] -> the 4 + n answer digits LSB first (teacher-forced in training, greedy at test); it reads the
                              memory and writes NOTHING. K = 0 is the same-harness single pass.
Training: FINAL only, §161's 4 x n mix (n ~ U{2..5} per batch; multpair's gen_operands minus the 4x4 bench test, both orders), 165-1's curriculum (CE on the
lowest min(c, 4 + n) digits, c >= 8 -> all), 48k, AdamW 1e-4 wd 0.01, warm-up 100 + cosine, clip 1.0; ckpts / resume; --auto_micro (4x5 training + the
4x16 eval at K' = 4K). Optional (§168 step 2): --micro (gradient accumulation), --ckpt_passes (torch.utils.checkpoint around each think pass).
eval.json: test_acc / acc_by_column (4x4 bench, K' = K), ext_4x6 / 4x7 / 4x8 / 4x16 (§161's seeds), compute / compute_4x8 (Meter), s169 {acc_by_kprime
{0, 1, K/2, K, 2K, 4K}, ext_4x8_by_kprime {K, 2K, 4K}, acc_by_size_kprime {4x4 / 4x6 / 4x8 / 4x16: {K': acc}}, skip_pass {t: acc}, inprog_by_pass
{pass: {col: {units, carry}}} (155-4's colsum_cls on the memory after think pass t; 8k generator 4x4 fit, scored on the bench), spikes, n_nonfinite}.
    python -m boxes.inject169 --think_k 16 --seed 1 --out boxes/runs_fsq/s169_k16_4x4_s1"""
import os
os.environ["BOXES_NB"] = "5"
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
from boxes import core as bc  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
from boxes.fsq120 import EOT, make_opt, lr_at, driver_version, env_info  # noqa: E402
from boxes.multcol import DTOK, gen_operands, load_bench  # noqa: E402
from boxes.deltas5 import fwd_read, write_grad_norm  # noqa: E402
from boxes.fixedchunk import FixedChunk, DT, STAR, EQ, PAD, product_digits, acc_report, draw, cmax, amp_ctx  # noqa: E402


def think_text(A, B):
    """the think-pass input after EOT: a digits, " *", b digits (LSB first), " ." -> (N, na + nb + 2)."""
    return np.concatenate([DT[A], np.full((len(A), 1), STAR), DT[B], np.full((len(A), 1), PAD)], 1)


NEWT, SAMET = bc.tok(" ;")[0], bc.tok(" ,")[0]                              # multpair's column flags (NEWTOK / SAMETOK)


def pair_seg(A, B, t):
    """§177 (d): pass t's (1-based) §161 pair segment after EOT: [a_i, b_j, flag] for the t-th pair in column order (k = i + j ascending, i ascending),
    flag ' ;' when the pair opens its column, else ' ,' (= multpair.schedule / tokens; K = na * nb pair passes, no flush) -> (N, 3)."""
    na, nb = A.shape[1], B.shape[1]; pairs = [(i, k - i) for k in range(na + nb - 1) for i in range(na) if 0 <= k - i < nb]
    i, j = pairs[t - 1]; new = t == 1 or (pairs[t - 2][0] + pairs[t - 2][1]) != i + j
    return np.stack([DT[A[:, i]], DT[B[:, j]], np.full(len(A), NEWT if new else SAMET)], 1)


def dec_prefix(A, B):
    """the decoder's prefix after EOT: the problem text and " =" -> (N, na + nb + 2)."""
    return np.concatenate([DT[A], np.full((len(A), 1), STAR), DT[B], np.full((len(A), 1), EQ)], 1)


class Inject(FixedChunk):
    # §177 (a): self.pidx = nn.Embedding(rows, 768), zero-initialised, set by main (a registered submodule; absent = §169). No class-level default:
    # a class attribute would shadow the submodule (nn.Module.__getattr__ is only consulted after normal lookup fails; gate177 caught it).
    p_off = 0                                                               # §177 (a): this batch's offset o (0 at test)

    def write(self, S, ids, t=None):
        """one think pass; §177 (a): pass t (1-based, global) adds pidx[o + t] to the think token's input (the last position, " .")."""
        if getattr(self, "pidx", None) is None:
            return super().write(S, ids)
        r_ = self.p_off + t; assert t is not None and 0 <= r_ < self.pidx.num_embeddings, "§177 (a): o + t must stay inside the index table"
        x = self.g.embed(ids); x = torch.cat([x[:, :-1], (x[:, -1] + self.pidx.weight[r_].to(x.dtype))[:, None]], 1)
        return DM.gpt_forward_mem(self.g, x, self.mem, S)[1]

    def write_mixed(self, S, ids, last, mask, t):
        """§177 (d): right-padded per-item inputs (ids incl. EOT), the index added at each row's last real position `last` (B,), padded writes masked."""
        x = self.g.embed(ids)
        if getattr(self, "pidx", None) is not None:
            r_ = self.p_off + t; assert 0 <= r_ < self.pidx.num_embeddings
            oh = Fn.one_hot(last, x.shape[1]).to(x.dtype); x = x + oh[:, :, None] * self.pidx.weight[r_].to(x.dtype)[None, None, :]
        return DM.gpt_forward_mem(self.g, x, self.mem, S, mask=mask)[1]

    def think_fade(self, S, A, B, p, rng_f, t0=0):
        """§177 (d): K = na * nb write-only passes; pass t sees, per item, its pair segment (prob 1 - p) or the full problem text (prob p)."""
        N = len(A); full = think_text(A, B); Lf = full.shape[1]
        for t in range(1, A.shape[1] * B.shape[1] + 1):
            use = rng_f.random(N) < p; seg = pair_seg(A, B, t)
            if not use.any():                                               # all pair segments: no padding (p = 0 is exactly the pair schedule)
                ids = self.ids(torch.tensor(seg, device=S.device)); last = torch.full((N,), 3, device=S.device); mask = None
            elif use.all():
                ids = self.ids(torch.tensor(full, device=S.device)); last = torch.full((N,), Lf, device=S.device); mask = None
            else:
                body = np.full((N, Lf), EOT); body[use] = full[use]; body[~use, :3] = seg[~use]; ids = self.ids(torch.tensor(body, device=S.device))
                last = torch.tensor(np.where(use, Lf, 3), device=S.device); mask = torch.ones(ids.shape, dtype=torch.bool, device=S.device)
                mask[torch.tensor(~use, device=S.device), 4:] = False
            S = self.write_mixed(S, ids, last, mask, t0 + t)
        return S

    def think(self, S, A, B, K, skip=(), override=None, ckpt=False, t0=0):
        ids = self.ids(torch.tensor(think_text(A, B), device=S.device))
        for t in range(1, K + 1):
            if t not in skip:
                S = checkpoint(self.write, S, ids, t0 + t, use_reentrant=False) if (ckpt and S.requires_grad is not None and torch.is_grad_enabled()) else self.write(S, ids, t0 + t)
            if override is not None and override[0] == t:
                S = override[1]
        return S

    def dec_logits(self, S, A, B, ans):
        """teacher-forced: [EOT, prefix, ans[:-1]] -> logits predicting the answer digits (positions from " =" on)."""
        pre = torch.tensor(dec_prefix(A, B), device=S.device); x = self.ids(torch.cat([pre, ans[:, :-1]], 1)); P = pre.shape[1] + 1
        h = fwd_read(self.g, self.g.embed(x), self.mem, S); return self.g.logits(h[:, P - 1:]).float()

    @torch.no_grad()
    def greedy_ans(self, S, A, B, L):
        x = self.ids(torch.tensor(dec_prefix(A, B), device=S.device)); lut = {int(t): i for i, t in enumerate(DTOK)}; out = []
        for _ in range(L):
            nt = self.g.logits(fwd_read(self.g, self.g.embed(x), self.mem, S)[:, -1]).float().argmax(-1); out.append(nt); x = torch.cat([x, nt[:, None]], 1)
        return np.vectorize(lambda t: lut.get(int(t), -1))(torch.stack(out, 1).cpu().numpy())


@torch.no_grad()
def predict(m, A, B, K, dev, amp, bs, skip=(), keep=None):
    N = len(A); L = A.shape[1] + B.shape[1]; out = np.full((N, L), -1); kept = {}
    for i in range(0, N, bs):
        sl = slice(i, i + bs); S = m.mem.zero_state(len(A[sl]), dev)
        with amp_ctx(amp):
            ids = m.ids(torch.tensor(think_text(A[sl], B[sl]), device=dev))
            for t in range(1, K + 1):
                if t not in skip:
                    S = m.write(S, ids, t)
                if keep and t in keep:
                    kept.setdefault(t, []).append(S.float().flatten(1).cpu().numpy())
            out[sl] = m.greedy_ans(S, A[sl], B[sl], L)
    return out, {k_: np.concatenate(v) for k_, v in kept.items()}


@torch.no_grad()
def predict_fade(m, A, B, p, dev, amp, bs, seed=17890):
    """§177 (d): greedy answers after the K = na * nb mixed passes at fade fraction p (per-item draws from rng `seed`)."""
    N = len(A); L = A.shape[1] + B.shape[1]; out = np.full((N, L), -1); rng = np.random.default_rng(seed)
    for i in range(0, N, bs):
        sl = slice(i, i + bs)
        with amp_ctx(amp):
            S = m.think_fade(m.mem.zero_state(len(A[sl]), dev), A[sl], B[sl], p, rng); out[sl] = m.greedy_ans(S, A[sl], B[sl], L)
    return out


def compute_inj(m, na, nb, K):
    """Meter per item (§98 convention): K think passes of 1 + na + nb + 2 tokens (fresh, no rows); the decoder's prefix pass (1 + na + nb + 2 tokens,
    1 row) then na + nb - 1 KV-cached single-token passes (1 row each); plus the DeltaMem-R branch (as fixedchunk.compute_fc)."""
    mt = bc.Meter(m.g); Tt = 1 + na + nb + 2; tok = 0
    for _ in range(K):
        mt.step(Tt, 0, 0); tok += Tt
    mt.step(Tt, 0, 1); tok_r = Tt
    for j in range(na + nb - 1):
        mt.step(1, Tt + j, 1); tok_r += 1
    nb_ = sum(p_.numel() for n_, p_ in m.mem.named_parameters() if not n_.startswith("s_")); D2 = DM.DH ** 2
    fb = 2 * nb_ * (tok + tok_r) + (DM.H_ADD + DM.H_DEL) * 2 * D2 * (tok + tok_r) + (DM.H_ADD * 2 + DM.H_DEL * 8) * D2 * tok; d = mt.as_dict()
    return {"flops": d["flops"] + fb, "flops_trunk": d["flops"], "flops_branch": float(fb), "sequential_passes": d["seq_passes"], "lm_head_rows": na + nb,
            "tokens_processed": tok + tok_r, "think_passes": K, "convention": "§98 (boxes.core.Meter); think passes fresh; decoder prefix + (na + nb - 1) KV-cached passes"}


def colsum_cls(Kf, Af, Bf, Ks, As, Bs, dev):
    """155-4's colsum_cls: per column k the full accumulator c^_k (products + carry-in), 10-way units and the carry class, from the memory."""
    from boxes.multpair import _dual_fit_pred
    def acc_cols(A, B):
        n = len(A); L = A.shape[1] + B.shape[1] - 1; cs = np.zeros((n, L), np.int64)
        for i in range(A.shape[1]):
            for j in range(B.shape[1]):
                cs[:, i + j] += A[:, i] * B[:, j]
        ac = np.zeros_like(cs); c_ = np.zeros(n, np.int64)
        for k in range(L):
            ac[:, k] = cs[:, k] + c_; c_ = ac[:, k] // 10
        return ac
    af, as_ = acc_cols(Af, Bf), acc_cols(As, Bs); out = {}
    for k in range(af.shape[1]):
        r = {}
        for nm, yf, ys in (("units", af[:, k] % 10, as_[:, k] % 10), ("carry", af[:, k] // 10, as_[:, k] // 10)):
            kk = int(max(yf.max(), ys.max())) + 1; pr, _ = _dual_fit_pred(Kf, np.eye(kk)[yf], Ks, dev, cls=True); r[nm] = float((pr.argmax(1) == ys).mean())
        out[str(k + 1)] = r
    return out


def sample_k(rng_k, RK, PG, bptt_last, K):
    """§176: this batch's think passes -> ((passes without grad, passes with grad), {histogram key: drawn value}); fixed K -> ((0, K), {}).
    RK = [rbar, sigma, cap]: K = 1 + Poisson(e^tau), tau ~ N(log rbar - sigma^2 / 2, sigma), K <= cap (bptt_last 'all' or n: the last n with grad).
    PG = [k1max, k2max]: k1 ~ U{0..k1max} without grad, k2 ~ U{1..k2max} with grad. rng_k is its own stream (independent of the operand draws)."""
    if RK:
        tau = rng_k.normal(math.log(RK[0]) - RK[1] ** 2 / 2, RK[1]); k = min(1 + int(rng_k.poisson(math.exp(tau))), int(RK[2]))
        return ((0, k) if bptt_last == "all" else (max(0, k - int(bptt_last)), min(k, int(bptt_last)))), {"K": k}
    if PG:
        k1, k2 = int(rng_k.integers(0, PG[0] + 1)), int(rng_k.integers(1, PG[1] + 1)); return (k1, k2), {"k1": k1, "k2": k2}
    return (0, K), {}


def think_kk(m, S, A, B, kk, ckpt=False):
    """kk[0] think passes without gradient (§176: PROG's k1; RK beyond bptt_last), then kk[1] with gradient. (0, K) is §169's m.think exactly."""
    if kk[0]:
        with torch.no_grad():
            S = m.think(S, A, B, kk[0])
    return m.think(S, A, B, kk[1], ckpt=ckpt, t0=kk[0])                      # global pass numbers (§177 (a)'s index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--think_k", required=True, help="an int, or '4n' (§177 (d): K = 4n pair-schedule passes, the rule K)"); ap.add_argument("--shape_mix", default="2,3,4,5"); ap.add_argument("--steps", type=int, default=48000)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--micro", type=int, default=1); ap.add_argument("--ckpt_passes", action="store_true", help="activation checkpointing around each think pass (§168 step 2)")
    ap.add_argument("--mem_write", default="top", choices=["top", "block9"]); ap.add_argument("--write_mix_start", type=int, default=2000); ap.add_argument("--write_mix_end", type=int, default=6000)
    ap.add_argument("--ckpt_every", type=int, default=2000); ap.add_argument("--resume", default=None); ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_bs", type=int, default=250); ap.add_argument("--test_n", type=int, default=1000); ap.add_argument("--ext_n", type=int, default=1000)
    ap.add_argument("--probe_n_fit", type=int, default=8000); ap.add_argument("--auto_micro", action="store_true"); ap.add_argument("--eval_at", type=int, default=0, help="also evaluate test_acc / ext_4x8 at this step -> eval_<k>k.json"); ap.add_argument("--fast_add", action="store_true", help="deltamem.FAST_ADD: the additive heads' pass writes as one sum (Audit 2026-10-09; scratch/gate_fastadd.py)"); ap.add_argument("--mem_budget", type=float, default=21.0)
    ap.add_argument("--mem_check", action="store_true"); ap.add_argument("--sched_scale", type=float, default=1.0, help="smokes only"); ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--rand_k", default="", help="§176 RK: 'rbar,sigma,cap' -> per batch K = 1 + Poisson(e^tau), tau ~ N(log rbar - sigma^2 / 2, sigma), K <= cap")
    ap.add_argument("--prog", default="", help="§176 PROG: 'k1max,k2max' -> per batch k1 ~ U{0..k1max} passes without grad, then k2 ~ U{1..k2max} with grad")
    ap.add_argument("--fade", default="", help="§177 (d): 'start,end' steps of the pair -> full-text ramp p 0 -> 1 (requires --think_k 4n)")
    ap.add_argument("--pass_index", type=int, default=0, help="§177 (a): rows of a learned pass-index table (0 = off); training offset o ~ U{0..64 - K}")
    ap.add_argument("--bptt_last", default="all", help="§176 RK: 'all' or n (backprop through the last n passes); auto_micro falls back to 8 if K = cap does not fit")
    ap.add_argument("--bench", default=os.path.join("bench", "data", "mult_4x4_test.txt")); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); a.arm = "inject"; os.makedirs(a.out, exist_ok=True); dev = a.device; sc = a.sched_scale; FADE = a.think_k == "4n"; K = 16 if FADE else int(a.think_k); DM.FAST_ADD = bool(a.fast_add)
    assert FADE == bool(a.fade), "§177 (d): --think_k 4n goes with --fade"
    SHM = [int(x) for x in a.shape_mix.split(",") if x.strip()]; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    amp = None if (a.fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    teA, teB = load_bench(a.bench if os.path.isabs(a.bench) else os.path.join(ROOT, a.bench))
    EXCL = {(x.tobytes(), y.tobytes()) for x, y in zip(teA, teB)} | {(y.tobytes(), x.tobytes()) for x, y in zip(teA, teB)}; teA, teB = teA[:a.test_n], teB[:a.test_n]
    m = Inject(s5.load_gpt2(dev), 6, a.mem_write).to(dev)
    if a.pass_index:                                                        # §177 (a): zero-initialised, so step 0 is §169 exactly
        m.pidx = torch.nn.Embedding(a.pass_index, 768).to(dev); torch.nn.init.zeros_(m.pidx.weight)
    rng_o = np.random.default_rng(17700 + a.seed)                           # §177 (a): the offset stream (unused without --pass_index)
    RK = [float(x_) for x_ in a.rand_k.split(",")] if a.rand_k else None; PG = [int(x_) for x_ in a.prog.split(",")] if a.prog else None
    assert not (RK and PG); rng_k = np.random.default_rng(17600 + a.seed); k_hist = {}; kwin = []   # §176: K draws on their own stream (independent of n)
    KEVAL = max(64, 4 * K) if (RK or PG) else 4 * K
    if FADE:                                                                # §177 (d): p(step); the fade-fraction eval steps (note 177-2 / 177-3)
        f0_, f1_ = [float(x_) for x_ in a.fade.split(",")]; rng_f = np.random.default_rng(17800 + a.seed); p_now = [0.0]
        p_at = lambda st: min(1.0, max(0.0, (st - f0_ * sc) / max(1.0, (f1_ - f0_) * sc)))   # noqa: E731
        EVF = {int(round(f0_ * sc)): "scaffold"}; EVF.update({int(round((f0_ + q * (f1_ - f0_)) * sc)): "%g" % q for q in (0.25, 0.5, 0.75, 1.0)}); fade_ev = {}

    def draw_k():
        kk, rec_ = sample_k(rng_k, RK, PG, a.bptt_last, K)
        for nm_, v_ in rec_.items():
            h_ = k_hist.setdefault(nm_, {}); h_[str(v_)] = h_.get(str(v_), 0) + 1
        if rec_:
            kwin.append(kk[0] + kk[1])
        return kk

    def train_loss(A, B, c, kk=None):
        if FADE:
            S = m.think_fade(m.mem.zero_state(len(A), dev), A, B, p_now[0], rng_f)
        else:
            S = think_kk(m, m.mem.zero_state(len(A), dev), A, B, kk or (0, K), a.ckpt_passes)
        P = product_digits(A, B); ans = torch.tensor(DT[P], device=dev)
        lg = m.dec_logits(S, A, B, ans); ce = P.shape[1] if c >= 8 else min(c, P.shape[1])
        return Fn.cross_entropy(lg[:, :ce].reshape(-1, lg.shape[-1]), ans[:, :ce].reshape(-1))

    def mem_check(micro):
        torch.cuda.reset_peak_memory_stats(dev); r_ = np.random.default_rng(0); A, B = gen_operands(r_, a.bs // micro, 4), gen_operands(r_, a.bs // micro, max(SHM)); m.train()
        kw = (0, K) if not (RK or PG) else ((0, int(RK[2])) if (RK and a.bptt_last == "all") else ((int(RK[2]) - int(a.bptt_last), int(a.bptt_last)) if RK else (PG[0], PG[1])))
        p_keep = p_now[0] if FADE else None
        if FADE:
            p_now[0] = 1.0                                                  # §177 (d): the longest inputs (full text) at K = 4 x max n
        with amp_ctx(amp):
            l_ = train_loss(A, B, 8, kw)                                    # §176: the worst-case pass count
        if FADE:
            p_now[0] = p_keep
        l_.backward(); m.zero_grad(set_to_none=True); m.eval()
        predict(m, gen_operands(r_, a.eval_bs, 4), gen_operands(r_, a.eval_bs, 16), KEVAL, dev, amp, a.eval_bs); m.train()
        return torch.cuda.max_memory_allocated(dev) / 2 ** 30
    if a.mem_check or (a.auto_micro and dev.startswith("cuda") and not a.resume):
        mc = []
        while True:
            try:                                                            # 2026-10-09: an OOM inside the check (a hard memory cap) = over budget -> larger micro
                pk = mem_check(a.micro)
            except torch.OutOfMemoryError:
                pk = float("inf")
            if pk == float("inf"):                                       # after the except clause, so the failed graph is released
                m.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            mc.append({"micro": a.micro, "peak_gib": pk}); print("MEM", a.micro, pk, flush=True)
            if pk <= a.mem_budget or a.mem_check:
                break
            if a.micro >= a.bs:
                if RK and a.bptt_last == "all":                             # §176: full backprop to K = cap does not fit -> the last 8 passes
                    a.bptt_last = "8"; a.micro = 1; mc.append({"bptt_last": "8"}); continue
                break
            a.micro *= 2
        json.dump({"checks": mc, "micro": a.micro, "peak_gib": mc[-1]["peak_gib"], "train_shape": "4x%d" % max(SHM), "eval_shape": "4x16 at K' = %d" % KEVAL,
                   "train_passes": "fixed K" if not (RK or PG) else ("RK cap %d, bptt_last %s" % (int(RK[2]), a.bptt_last) if RK else "PROG %d no-grad + %d grad" % (PG[0], PG[1]))},
                  open(os.path.join(a.out, "mem_check.json"), "w", newline="\n"), indent=1)
        if a.mem_check:
            return
        assert mc[-1]["peak_gib"] <= a.mem_budget
    info = {**vars(a), "train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_driver": driver_version(), **env_info(), "hostprobe_sps": os.environ.get("HOSTPROBE_SPS"),
            "think_input": "[EOT, a * b, ' .'] (the full problem text, every pass)", "decoder_input": "[EOT, a * b, ' ='] + answer digits (read-only)"}
    json.dump(info, open(os.path.join(a.out, ("args_resume_%d.json" % int(time.time())) if a.resume else "args.json"), "w", newline="\n"), indent=1)
    opt = make_opt(m, a.lr); step0 = 1; n_nonfinite = 0; wg_hist = []; wwin = []; wlast = []; hist = []
    log = open(os.path.join(a.out, "log.jsonl"), "a" if a.resume else "w", newline="\n"); t0 = time.time()
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False); m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"].cpu()); step0 = ck["step"] + 1; n_nonfinite = ck["n_nonfinite"]; wg_hist = ck["wg_hist"]; t0 = time.time() - ck["elapsed"]
        if dev.startswith("cuda") and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device=dev)
        log.write(json.dumps({"step": ck["step"], "resumed": a.resume}) + "\n"); log.flush()
    elif K > 0:                                                             # note 147-2: g_add = median ||read_del|| / ||read_add|| at the think hand-offs
        r_ = np.random.default_rng(16901); A0, B0 = gen_operands(r_, 128, 4), gen_operands(r_, 128, 4); m.mem.norms = []
        with torch.no_grad(), amp_ctx(amp):
            m.think(m.mem.zero_state(128, dev), A0, B0, min(K, 4))
        nr = m.mem.norms[1:]; m.mem.norms = None; r0 = float(torch.stack([d_ / a_.clamp_min(1e-12) for d_, a_ in nr]).median()) if nr else 1.0
        m.mem.s_add.data.fill_(math.log(max(r0, 1e-12))); json.dump({"g_add_init": r0}, open(os.path.join(a.out, "s169_init.json"), "w", newline="\n"))
    for step in range(step0, a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(a.lr, gr, step, a.steps)
        m.train(); m.mem.fade.fill_(min(1.0, step / max(1, int(500 * sc))))
        m.mem.gamma.fill_(0.0 if step <= a.write_mix_start * sc else min(1.0, (step - a.write_mix_start * sc) / max(1.0, (a.write_mix_end - a.write_mix_start) * sc)))
        c = int(rng.integers(1, cmax(step / sc) + 1)); A, B = draw(rng, a.bs, SHM, EXCL); opt.zero_grad(); tot = 0.0; ch = a.bs // a.micro; kk = draw_k()
        if FADE:
            p_now[0] = p_at(step); kk = (0, A.shape[1] * B.shape[1])         # §177 (d): K = 4n this batch
        if a.pass_index:                                                    # o uses this batch's actual pass count; gate: o + K <= 64 < rows (Audit)
            m.p_off = int(rng_o.integers(0, 64 - (kk[0] + kk[1]) + 1)); assert m.p_off + kk[0] + kk[1] <= 64 < a.pass_index
        for k_ in range(a.micro):
            sl = slice(k_ * ch, (k_ + 1) * ch)
            with amp_ctx(amp):
                lk = train_loss(A[sl], B[sl], c, kk)
            (lk / a.micro).backward(); tot += float(lk.item()) / a.micro
        wwin.append(write_grad_norm(m)); gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if not (math.isfinite(tot) and bool(torch.isfinite(gn))):
            log.write(json.dumps({"step": step, "nonfinite": 1}) + "\n"); log.flush(); n_nonfinite += 1; opt.zero_grad()
        else:
            opt.step(); hist.append(tot)
        if FADE and step in EVF:                                            # §177 (d): 4x4 at the fade fraction reached (scaffold = p 0 at the fade start)
            m.eval(); m.p_off = 0; pq = p_at(step); acc_ = float((predict_fade(m, teA, teB, pq, dev, amp, a.eval_bs) == product_digits(teA, teB)).all(1).mean())
            fade_ev[EVF[step]] = {"step": step, "p": pq, "acc": acc_}; json.dump(fade_ev[EVF[step]], open(os.path.join(a.out, "eval_%dk.json" % (step // 1000)), "w", newline="\n"), indent=1)
            print("FADE EVAL", EVF[step], fade_ev[EVF[step]], flush=True); m.train()
        if a.eval_at and step == a.eval_at:                                 # Theory 7caa3a82 (§174's matched budget): accuracy at this step, no checkpoint kept
            m.eval(); m.p_off = 0; P_ = product_digits(teA, teB); xa, xb = gen_operands(np.random.default_rng(14004), a.ext_n, 4), gen_operands(np.random.default_rng(14005), a.ext_n, 8)
            e24 = {"step": step, "test_acc": float((predict(m, teA, teB, K, dev, amp, a.eval_bs)[0] == P_).all(1).mean()),
                   "ext_4x8": {"test_acc": float((predict(m, xa, xb, K, dev, amp, a.eval_bs)[0] == product_digits(xa, xb)).all(1).mean()), "seeds": [14004, 14005]}}
            json.dump(e24, open(os.path.join(a.out, "eval_%dk.json" % (step // 1000)), "w", newline="\n"), indent=1); m.train()
        if step % a.log_every == 0 or step == a.steps:
            md_ = float(np.median(wlast)) if wlast else 0.0; mx_ = float(np.max(wwin)); late = step > a.write_mix_start * sc
            w_rec = {"step": step, "mean": float(np.mean(wwin)), "max": mx_, "median_prev_1k": md_, "spike_100x": bool(late and mx_ > 100 * md_ > 0),
                     "spike_100x_floor": bool(late and mx_ > 100 * md_ > 0 and mx_ > 1.0)}
            wlast.extend(wwin); del wlast[:-int(1000 * sc) or None]; wwin = []; wg_hist.append(w_rec)
            rec = {"step": step, "loss": float(np.mean(hist)) if hist else None, "c_max": cmax(step / sc), "sec": round(time.time() - t0), "g_add": float(torch.exp(m.mem.s_add)),
                   "g_del": float(torch.exp(m.mem.s_del)), "fade": float(m.mem.fade), "gamma": float(m.mem.gamma), "write_grad_norm": w_rec["mean"], "write_grad_spike_100x_floor": w_rec["spike_100x_floor"]}
            if RK or PG:
                rec["k_train_mean"] = float(np.mean(kwin)) if kwin else None; kwin.clear()
            hist = []; log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps:
            cp = os.path.join(a.out, "ckpt_last.pt")
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(dev) if dev.startswith("cuda") else None, "step": step, "n_nonfinite": n_nonfinite, "wg_hist": wg_hist,
                        "elapsed": time.time() - t0}, cp)
    train_sec = round(time.time() - t0); torch.save(m.state_dict(), os.path.join(a.out, "model.pt")); m.eval(); m.p_off = 0; t1 = time.time(); bs = a.eval_bs
    Pte = product_digits(teA, teB); pred, _ = predict(m, teA, teB, K, dev, amp, bs); res = acc_report(pred, Pte)
    res.update({"arm": "inject", "think_k": K, "seed": a.seed, "train_sec": train_sec, "n_nonfinite": n_nonfinite, "eval_dtype": str(amp), "micro": a.micro,
                "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_gpu": info["train_gpu"], "train_driver": info["train_driver"]})
    xs = {4: (teA, teB)}
    for nb_, s1_, s2_ in ((6, 14016, 14017), (7, 14018, 14019), (8, 14004, 14005), (16, 14006, 14007)):
        xA, xB = gen_operands(np.random.default_rng(s1_), a.ext_n, 4), gen_operands(np.random.default_rng(s2_), a.ext_n, nb_); xs[nb_] = (xA, xB)
        res["ext_4x%d" % nb_] = {**acc_report(predict(m, xA, xB, (4 * nb_ if FADE else K), dev, amp, bs)[0], product_digits(xA, xB)), "seeds": [s1_, s2_]}
    res["compute"] = compute_inj(m, 4, 4, K); res["compute_4x8"] = compute_inj(m, 4, 8, 32 if FADE else K)
    kps = sorted({0, 1, K // 2, K, 2 * K, 4 * K}) if not (RK or PG) else [0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64]; s = {"acc_by_kprime": {}, "ext_4x8_by_kprime": {}, "acc_by_size_kprime": {}, "skip_pass": {}, "write_grad_norm": wg_hist,
                                                         "write_grad_spikes_floor": [x_ for x_ in wg_hist if x_["spike_100x_floor"]], "n_nonfinite": n_nonfinite,
                                                         "gamma_final": float(m.mem.gamma), "g_add": float(torch.exp(m.mem.s_add)), "g_del": float(torch.exp(m.mem.s_del))}
    for nb_ in (4, 6, 8, 16):
        Pn = product_digits(*xs[nb_]); s["acc_by_size_kprime"]["4x%d" % nb_] = {str(kp): float((predict(m, *xs[nb_], kp, dev, amp, bs)[0] == Pn).all(1).mean()) for kp in kps}
    s["acc_by_kprime"] = dict(s["acc_by_size_kprime"]["4x4"])
    s["ext_4x8_by_kprime"] = {str(kp): s["acc_by_size_kprime"]["4x8"][str(kp)] for kp in (sorted({K, 2 * K, 4 * K}) if not (RK or PG) else kps)}
    pts = sorted({t for t in (1, K // 4, K // 2, 3 * K // 4, K) if t >= 1}) if K else []
    for t in pts:
        s["skip_pass"][str(t)] = float((predict(m, teA, teB, K, dev, amp, bs, skip=(t,))[0] == Pte).all(1).mean())
    ipts = sorted({t for t in (K // 4, K // 2, 3 * K // 4, K) if t >= 1}) if K else []
    if ipts:
        fA, fB = draw(np.random.default_rng(16903), a.probe_n_fit, [4], EXCL); _, Kf = predict(m, fA, fB, K, dev, amp, bs, keep=set(ipts)); _, Ks = predict(m, teA, teB, K, dev, amp, bs, keep=set(ipts))
        s["inprog_by_pass"] = {str(t): colsum_cls(Kf[t], fA, fB, Ks[t], teA, teB, dev) for t in ipts}; s["probe_n_fit"] = int(a.probe_n_fit)
    if K:                                                                   # Audit (descriptive): the memory's norms per pass up to 4K (the additive heads grow ~K x)
        mn = {}
        with torch.no_grad():
            nA, nB = teA[:256], teB[:256]; S = m.mem.zero_state(len(nA), dev); ids = m.ids(torch.tensor(think_text(nA, nB), device=dev))
            for t in range(1, 4 * K + 1):
                with amp_ctx(amp):
                    S = m.write(S, ids, t)
                mn[str(t)] = {"additive": float(S[:, :DM.H_ADD].float().flatten(1).norm(dim=1).median()), "delta": float(S[:, DM.H_ADD:].float().flatten(1).norm(dim=1).median())}
        s["mem_norm_by_pass"] = mn
    res["s169"] = s
    if a.pass_index:                                                        # §177 (a) / (d) fields (score_s177: "K" = the rule K; (d): 4n, 16 at 4x4, 32 at 4x8)
        K8_ = 32 if FADE else K; sz8 = s["acc_by_size_kprime"]["4x8"]
        res["s177"] = {"acc_by_kprime": {"0": s["acc_by_kprime"]["0"], "1": s["acc_by_kprime"]["1"], "K": s["acc_by_kprime"][str(K)], "2K": s["acc_by_kprime"][str(2 * K)]},
                       "ext_4x8_by_kprime": {"K": sz8[str(K8_)], "2K": sz8[str(2 * K8_)]}, "skip_pass": s["skip_pass"], "rule_k": "4n" if FADE else K,
                       "pass_index_norm_by_row": [round(float(v), 5) for v in m.pidx.weight.detach().float().norm(dim=1).cpu().tolist()]}
        if FADE:                                                            # note 177-3: acc_scaffold_12k (p 0, pair inputs, K = 4n); acc_by_p at the steps p reaches
            pe = p_at(a.steps)
            res["s177"].update({"acc_scaffold_12k": (fade_ev.get("scaffold") or {}).get("acc"), "acc_by_p": {k_: v_["acc"] for k_, v_ in fade_ev.items() if k_ != "scaffold"},
                                "fade_evals": fade_ev, "rule_k_by_size": {"4x4": 16, "4x6": 24, "4x7": 28, "4x8": 32, "4x16": 64},
                                "acc_at_end_p": {"p": pe, "acc": float((predict_fade(m, teA, teB, pe, dev, amp, bs) == Pte).all(1).mean())}})
    if RK or PG:                                                            # §176: the norms to 64, the sampled-K histogram, §173's halt
        mn = {}
        with torch.no_grad():
            nA, nB = teA[:256], teB[:256]; S = m.mem.zero_state(len(nA), dev); ids = m.ids(torch.tensor(think_text(nA, nB), device=dev))
            for t in range(1, KEVAL + 1):
                with amp_ctx(amp):
                    S = m.write(S, ids, t)
                mn[str(t)] = {"additive": float(S[:, :DM.H_ADD].float().flatten(1).norm(dim=1).median()), "delta": float(S[:, DM.H_ADD:].float().flatten(1).norm(dim=1).median())}
        res["s176"] = {"mem_norm_by_pass": mn, "k_hist": k_hist, "bptt_last": a.bptt_last, "kprimes": kps}
        from boxes.halt173 import halt_fields
        res["halt"], res["halt_fixed_k"], res["halt_acc_by_t"], res["halt_n"] = halt_fields("inject", m, {"think_k": K}, dev, amp, a.ext_n, bs, a.bench)
    res["eval_sec"] = round(time.time() - t1)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):
        if os.path.exists(os.path.join(a.out, ck_)):
            os.remove(os.path.join(a.out, ck_))
    print("EVAL", json.dumps({"test_acc": res["test_acc"], "acc_by_kprime": s["acc_by_kprime"]}), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
