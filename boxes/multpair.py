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
targets = the low c digits, c ~ U(1, c_max), c_max 1 / 2 / 4 / 8 at 0 / 2k / 5k / 9k (4x4; 5x5's schedule is not registered yet).
eval.json: test_acc (all 2n digits exact on the 1,000 bench items), acc_by_column {per_column, cumulative}, train_acc, donor (pos4 / pos8 /
pos12: the code after pair t from a donor; the recipient's later digits with the donor's accumulator, exact; digit-level donor /
recipient / chance per position + row level), s140_probe {r2_acc_code, r2_acc_h (h before W_down), r2_colsum_context} at full length,
ext_4x8 (4x4 models; 32 pairs + flush; no unseen tokens (note 140-1)), n_codes, rho (bf16; scored = on-host --contraction_only --fp32).
    python -m boxes.multpair --family mult_pair --task 4x4 --seed 1 --stop_plateau --ckpt_every 2000 --out boxes/runs_fsq/s140_pair_4x4_s1"""
import os
os.environ["BOXES_NB"] = "5"
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

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

EQTOK, NEWTOK, SAMETOK = bc.tok(" =")[0], bc.tok(" ;")[0], bc.tok(" ,")[0]   # flush / new-column / same-column flags (note 140-1)
assert all(len(bc.tok(w)) == 1 for w in (" =", " ;", " ,")) and len({EQTOK, NEWTOK, SAMETOK}) == 3 and not {EQTOK, NEWTOK, SAMETOK} & set(DTOK)
CMAX = {"4x4": ((2000, 1), (5000, 2), (9000, 4), (None, 8))}


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
    ap.add_argument("--fp32", action="store_true"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.from_args:
        keep = {k: getattr(a, k) for k in ("contraction_only", "fp32", "device", "out", "from_args")}
        for k, v in json.load(open(a.from_args)).items():
            if hasattr(a, k):
                setattr(a, k, v)
        for k, v in keep.items():
            setattr(a, k, v)
    dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); sc = a.sched_scale; n = int(a.task[0])
    if a.d_q != F.D_Q or a.levels != F.LEV:
        F.D_Q, F.LEV = a.d_q, a.levels; F.HALF = (F.LEV - 1) / 2; F.SHIFT = math.atanh(F.OFF / F.HALF)
    memcap and memcap.apply(dev)
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    os.makedirs(a.out, exist_ok=True); fam = PairFam(); dname = lambda: torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"  # noqa: E731
    teA, teB = load_bench(os.path.join(ROOT, "bench", "data", "mult_%s_test.txt" % a.task)); teA, teB = teA[:a.test_n], teB[:a.test_n]
    if a.contraction_only:
        m = MultPair(s5.load_gpt2(dev), fam).to(dev); m.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev)); m.eval(); m.p_hard = 1.0
        ampc = None if a.fp32 else amp; ct = contraction(m, teA, teB, dev, ampc, T0=a.contract_T0, n=a.contract_n); ct.update({"eval_dtype": str(ampc), "device": dname(), "driver": driver_version()})
        json.dump(ct, open(os.path.join(a.out, "eval_contraction_fp32.json" if a.fp32 else "eval_contraction.json"), "w", newline="\n"), indent=1)
        print("CONTRACTION", json.dumps({k: ct[k] for k in ("rho", "rho_nonzero", "basin")}), flush=True); return
    cmax(1, a.task)                                                         # refuses an unregistered task schedule before any work
    if a.train_n:
        from boxes.multcol import load_bench_head
        trA, trB = load_bench_head(os.path.join(ROOT, "bench", "data", "mult_train", "%s_train.txt" % a.task), a.train_n)
    else:
        trA, trB = load_bench(os.path.join(ROOT, "bench", "data", "mult_train", "%s_train.txt" % a.task))
    n_raw = len(trA); trA, trB, n_rem = dedup(trA, trB, teA, teB)
    info = {**vars(a), "train_items": int(len(trA)), "train_items_raw": int(n_raw), "train_removed_test_overlap": n_rem, "flush_token": " =", "pair_flags": "column_start", "flag_tokens": {"new_column": " ;", "same_column": " ,", "flush": " ="}, "train_gpu": dname(),
            "train_driver": driver_version(), **env_info()}
    json.dump({**info, "resumed_at": time.strftime("%Y-%m-%d %H:%M:%S")} if a.resume else info,
              open(os.path.join(a.out, ("args_resume_%d.json" % int(time.time())) if a.resume else "args.json"), "w", newline="\n"), indent=1)
    m = MultPair(s5.load_gpt2(dev), fam).to(dev); opt = make_opt(m, a.lr); m.train(); pairs, col, emit = schedule(n, n)
    log = open(os.path.join(a.out, "log.jsonl"), "a" if a.resume else "w", newline="\n"); t0 = time.time(); hist = []; L127 = {}; n_nonfinite = 0; step0 = 1
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False); m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["np_rng"]; torch.set_rng_state(ck["torch_rng"].cpu()); L127 = ck["L127"]; n_nonfinite = ck["n_nonfinite"]
        if dev.startswith("cuda") and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device=dev)
        step0 = ck["step"] + 1; t0 = time.time() - ck["elapsed"]; log.write(json.dumps({"step": ck["step"], "resumed": a.resume}) + "\n"); log.flush()
    def batch(c, bs_):
        """A chain over the pairs of the first c columns (+ the flush when c = 2n); labels / trace mask at the emission steps."""
        ii = rng.integers(0, len(trA), bs_); A, Bd = trA[ii], trB[ii]; _, dig, _, _, _, _ = accumulators(A, Bd)
        T = int(np.searchsorted(col, c - 1, side="right")) if c < 2 * n else len(pairs)     # steps whose column < c
        set_ops(m, A, Bd, dev); X = torch.tensor(np.tile(np.arange(T), (bs_, 1)), device=dev)
        tm = torch.tensor(np.tile(emit[:T], (bs_, 1)), device=dev); lab = np.where(dig[:, :T] >= 0, dig[:, :T], 0)
        return X, lab, tm, T
    if a.mem_check:
        m.p_hard = 1.0; X, lab, tm, T = batch(2 * n, a.bs)
        with amp_ctx(amp):
            loss, _, _, _ = m.chain(m.start(a.bs, dev), X, lab, tm)
        loss.backward(); m.eval(); rollout(m, gen_operands(rng, a.eval_bs, n), gen_operands(rng, a.eval_bs, 2 * n), dev, amp, a.eval_bs, record=True)
        pk = torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.startswith("cuda") else None
        json.dump({"peak_gib": pk, "bs": a.bs, "eval_bs": a.eval_bs, "task": a.task}, open(os.path.join(a.out, "mem_check.json"), "w", newline="\n"), indent=1); print("MEM", pk); return
    for step in range(step0, a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(a.lr, gr, step, a.steps)
        m.p_hard = 0.0 if step <= a.p_hard_start * sc else min(1.0, (step - a.p_hard_start * sc) / ((a.p_hard_end - a.p_hard_start) * sc)); m.train()
        c = int(rng.integers(1, cmax(step / sc, a.task) + 1)); X, lab, tm, T = batch(c, a.bs); opt.zero_grad()
        with amp_ctx(amp):
            loss, _, _, _ = m.chain(m.start(a.bs, dev), X, lab, tm)          # CE at the emission steps (mean over them)
        loss.backward(); tot = float(loss.item()); gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if not (math.isfinite(tot) and bool(torch.isfinite(gn))):
            bad = "loss" if not math.isfinite(tot) else next((n_ for n_, p_ in m.named_parameters() if p_.grad is not None and not bool(torch.isfinite(p_.grad).all())), "grad_norm")
            log.write(json.dumps({"step": step, "nonfinite": 1, "first": bad}) + "\n"); log.flush(); n_nonfinite += 1; opt.zero_grad()
        else:
            opt.step(); hist.append(tot)
        if step % a.log_every == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(hist)) if hist else None, "c_max": cmax(step / sc, a.task), "p_hard": m.p_hard, "sec": round(time.time() - t0)}; hist = []
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True); L127[step] = rec["loss"]
        if step == int(2000 * sc) and L127:                                 # 123-3's flag logged for every attempt; stops only with --stop_plateau
            gl = lambda t_: L127[min(L127, key=lambda s_: abs(s_ - t_ * sc))]  # noqa: E731
            fl = bool(gl(2000) >= 0.9 * gl(1000) and gl(2000) >= 0.25 * gl(100))
            log.write(json.dumps({"step": step, "flag_123_3": fl, "loss_100": gl(100), "loss_1k": gl(1000), "loss_2k": gl(2000), "stop_plateau": a.stop_plateau}) + "\n"); log.flush()
            if fl and a.stop_plateau:
                log.write(json.dumps({"step": step, "stopped": "plateau (123-3 flag at 2k)", "loss_100": gl(100), "loss_1k": gl(1000), "loss_2k": gl(2000)}) + "\n")
                log.close(); print("STOPPED plateau at 2k", flush=True); return
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps:
            cp = os.path.join(a.out, "ckpt_last.pt")
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(dev) if dev.startswith("cuda") else None,
                        "np_rng": rng.bit_generator.state, "step": step, "L127": L127, "n_nonfinite": n_nonfinite, "elapsed": time.time() - t0}, cp + ".tmp")
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            os.replace(cp + ".tmp", cp); log.write(json.dumps({"step": step, "ckpt": "ckpt_last.pt"}) + "\n"); log.flush()
        if a.stop_after and step >= a.stop_after:
            log.close(); print("STOP_AFTER", step, flush=True); return
    train_sec = round(time.time() - t0); torch.save(m.state_dict(), os.path.join(a.out, "model.pt")); m.eval(); m.p_hard = 1.0; t1 = time.time()
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    Pte, _, _ = columns(teA, teB); pred, codes, _ = rollout(m, teA, teB, dev, amp, a.eval_bs, record=True); res = acc_report(pred, Pte)
    ti = np.random.default_rng(14000).integers(0, len(trA), 1000); Ptr, _, _ = columns(trA[ti], trB[ti])
    res.update({"family": a.family, "task": a.task, "seed": a.seed, "long_acc": res["test_acc"], "train_acc": float((rollout(m, trA[ti], trB[ti], dev, amp, a.eval_bs) == Ptr).all(1).mean()),
                "train_sec": train_sec, "final_loss": L127[max(L127)] if L127 else None, "n_nonfinite": n_nonfinite, "train_removed_test_overlap": n_rem, "eval_dtype": str(amp),
                "gpu": dname(), "train_gpu": info["train_gpu"], "train_driver": info["train_driver"]})
    if a.task == "4x4" and a.ext_n:
        xA, xB = gen_operands(np.random.default_rng(14004), a.ext_n, 4), gen_operands(np.random.default_rng(14005), a.ext_n, 8); xA, xB, rem = dedup(xA, xB, teA, teB)
        Px, _, _ = columns(xA, xB); res["ext_4x8"] = {**acc_report(rollout(m, xA, xB, dev, amp, a.eval_bs), Px), "dedup_removed": rem, "seeds": [14004, 14005]}
    res["donor"] = donor(m, teA, teB, dev, amp, a.eval_bs, np.random.default_rng(14002), n=a.donor_n)
    ct = contraction(m, teA, teB, dev, amp, T0=a.contract_T0, n=a.contract_n); res.update({"rho": ct["rho"], "rho_nonzero": ct["rho_nonzero"], "basin": ct["basin"], "contraction": ct["detail"]})
    res["s140_probe"] = s140_probe(m, teA, teB, dev, amp, a.eval_bs); res["n_codes"] = int(len(np.unique(codes.reshape(-1, codes.shape[-1]), axis=0))); res["eval_sec"] = round(time.time() - t1)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):
        if os.path.exists(os.path.join(a.out, ck_)):
            os.remove(os.path.join(a.out, ck_))
    print("EVAL", json.dumps({k: res[k] for k in ("test_acc", "train_acc", "rho", "n_codes")}), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
