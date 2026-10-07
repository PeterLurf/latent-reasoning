"""THEORY §137 (9c9b1899; note 137-1): multiplication column by column on §120's FSQ carrier. fsq120's model unchanged (FSQ120: FSQ
d 8 / L 8, W_up / W_down / LN, the answer decoder) and §128's recipe (p_hard 2000 -> 4000, warmup + cosine, 24k, bs 64, lr 1e-4, AdamW
wd 0.01, clip 1.0, the 127-3 guard; --stop_plateau = 123-3's 2k flag for the restart slots, scratch/restart137.py). Only the segment
gains the operands, as §132's tables:
    segment k = [EOT, carried code, a's n digit tokens (" d", LSB first), b's n digit tokens, column-index token k],  k = 1 .. 2n
The code after segment k is decoded by the answer decoder into product digit p_k (one token; CE at every hand-off in training, greedy at
test). Start = the fixed zero code (carry 0). Curriculum: chains of L columns, L ~ U(1, L_max), targets = the product's low L digits;
L_max 2 / 4 / 8 at 0 / 2k / 5k (4x4), 2 / 4 / 8 / 10 at 0 / 2k / 5k / 9k (5x5). Train: the bench train split (a, b and the '####'
product only - the CoT text is never read), minus any item whose (a, b) or (b, a) is a test item (count logged, note 137-1).
Test: the 1,000 published items. eval.json: test_acc (all 2n digits exact), acc_by_column {per_column, cumulative}, train_acc, donor
(pos2 / pos4 / pos6: the code after column p is a donor's; digit-level shares over columns p+1 .. 2n of the recipient: donor_consistent
= digits matching the recipient's columns recomputed with the donor's carry, recipient_consistent = matching its own, chance_overlap =
columns where both carries give the same digit; + row-level variants), s137_probe {r2_carry_code, r2_carry_h, r2_colsum_context}
(ridge linear probes, fit on half the problems, R^2 on the rest, pooled over columns at L = 2n, + per column), ext_4x8 (4x4 models:
1,000 generated 4x8 items, deduped against the test items incl. commuted pairs; columns > 8 unseen in training), n_codes /
codes_per_carry, rho / basin (bf16; the scored rho = the on-host fp32 --contraction_only file). log.jsonl: the 123-3 flag decision
with loss(100 / 1k / 2k) at 2k for every attempt (note 137-1).
    python -m boxes.multcol --family mult_col --task 4x4 --seed 1 --stop_plateau --ckpt_every 2000 --out boxes/runs_fsq/s137_col_4x4_s1
    python -m boxes.multcol --from_args <run>/args.json --contraction_only --fp32 --out <run>      (or --prefix_only)"""
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
try:                                                                        # local hard memory cap; absent on hosts that lack it -> no cap
    from boxes import memcap  # noqa: E402
except ImportError:
    memcap = None
from boxes import s5  # noqa: E402
from boxes import core as bc  # noqa: E402
import boxes.fsq120 as F  # noqa: E402
from boxes.fsq120 import EOT, FSQ120, driver_version, env_info, lr_at, make_opt  # noqa: E402

DTOK = [bc.tok(" %d" % d)[0] for d in range(10)]                         # operand / answer digits
COLTOK = [bc.tok(str(k))[0] for k in range(25)]                           # column index k (bare number: never a " d" token)
assert all(len(bc.tok(str(k))) == 1 for k in range(25)) and not set(DTOK) & set(COLTOK)
DATA = os.path.join(ROOT, "bench", "data")
LMAX = {"4x4": ((2000, 2), (5000, 4), (None, 8)), "5x5": ((2000, 2), (5000, 4), (9000, 8), (None, 10))}


class MultFam:
    """COLUMN family: items = column indices 1 .. 2n, answers = product digits 0..9."""
    def __init__(self):
        self.name = "mult_col"; self.n_items = len(COLTOK); self.n_states = 10; self.item_tok = np.array([[t] for t in COLTOK])   # (items, 1), as §132

    def answer_tokens(self, d):
        return np.array(DTOK)[np.asarray(d)][:, None]

    def decode(self, toks):
        lut = {t: i for i, t in enumerate(DTOK)}
        return np.array([lut.get(int(v), -1) for v in toks[:, 0]])


class MultCol(FSQ120):
    """FSQ120 whose segment is [EOT, carried code, a digits, b digits, column token]; self.ctx: (B, 2n) operand token ids."""
    def __init__(self, g, fam):
        super().__init__(g, fam); self.ctx = None; self.keep_h = False; self.last_h = None

    def segment(self, prev, items):
        g = self.g; B = items.shape[0]; eot = g.embed(torch.full((B, 1), EOT, device=items.device))
        x = torch.cat([eot, prev.to(eot.dtype), g.embed(self.ctx), g.embed(self.itok[items])], 1); h, _ = g(x, None)
        if self.keep_h:                                                     # the probe's h_k (before W_down)
            self.last_h = h[:, -1].float()
        return h[:, -1]


def lmax(step, task):
    for lim, L in LMAX[task]:
        if lim is None or step <= lim:
            return L


def columns(A, Bd):
    """Operands LSB first -> product digits P (N, ma + mb), carry OUT of each column C (what code_k must carry), context-only column
    sums mod 10 S (sum_{i+j=k} a_i b_j mod 10: what a model could compute without the carried code). Exact integer arithmetic."""
    N, ma = A.shape; mb = Bd.shape[1]; L = ma + mb; P = np.zeros((N, L), np.int64); C = np.zeros((N, L), np.int64); S = np.zeros((N, L), np.int64)
    c = np.zeros(N, np.int64)
    for k in range(L):
        s = np.zeros(N, np.int64)
        for i in range(max(0, k - mb + 1), min(ma, k + 1)):
            s += A[:, i].astype(np.int64) * Bd[:, k - i]
        S[:, k] = s % 10; s = s + c; P[:, k] = s % 10; c = s // 10; C[:, k] = c
    assert (c == 0).all()
    return P, C, S


def ctx_tokens(A, Bd):
    return np.concatenate([np.array(DTOK)[A], np.array(DTOK)[Bd]], 1)


def to_int(D):
    return (D.astype(np.int64) * (10 ** np.arange(D.shape[1], dtype=np.int64))[None]).sum(1)


def load_bench(path):
    """bench text -> (A, B) digit arrays (LSB first). Reads only a, b and the '####' product; asserts the product = exact a x b."""
    A, Bd, P = [], [], []
    with open(path) as f:
        for ln in f:
            q = ln.split("||")[0]; a_, b_ = q.split("*"); A.append(a_.split()); Bd.append(b_.split()); P.append(ln.split("####")[1].split())
    A, Bd, P = np.array(A, np.int64), np.array(Bd, np.int64), np.array(P, np.int64); P2, _, _ = columns(A, Bd)
    assert (P2 == P).all(), "%s: the '####' product != exact a x b" % path
    return A, Bd


def dedup(A, Bd, teA, teB):
    """Drop items whose (a, b) or (b, a) is a test item (note 137-1). Returns kept arrays and the count removed."""
    te = set(zip(to_int(teA).tolist(), to_int(teB).tolist())); te |= {(y, x) for x, y in te}
    keep = np.array([(x, y) not in te for x, y in zip(to_int(A).tolist(), to_int(Bd).tolist())], bool)
    return A[keep], Bd[keep], int((~keep).sum())


def gen_operands(rng, n, m):
    D = rng.integers(0, 10, (n, m)); D[:, -1] = rng.integers(1, 10, n); return D


def amp_ctx(amp):
    return torch.autocast("cuda", dtype=amp, enabled=amp is not None)


@torch.no_grad()
def rollout(m, A, Bd, dev, amp, bs, override=None, record=False):
    """Hard chains over all columns 1 .. L; decode every step. override: (p, codes (N, d)) replaces the code after column p (1-based).
    Returns pred (N, L) [, codes (N, L, d), h (N, L, 768)]."""
    N = len(A); L = A.shape[1] + Bd.shape[1]; pred = np.full((N, L), -1); ctx = ctx_tokens(A, Bd)
    codes = np.zeros((N, L, F.D_Q), np.float32) if record else None; H = np.zeros((N, L, 768), np.float32) if record else None; m.keep_h = record
    for i in range(0, N, bs):
        sl = slice(i, i + bs); B = len(A[sl]); prev = m.start(B, dev); m.ctx = torch.tensor(ctx[sl], device=dev)
        for t in range(L):
            with amp_ctx(amp):
                h = m.segment(prev, torch.full((B,), t + 1, device=dev)); qs, q, zb = m.quant(h)
                if override is not None and override[0] == t + 1:
                    q = torch.tensor(override[1][sl], device=dev, dtype=qs.dtype); qs = q
                e = m.embed_code(qs); pred[sl, t] = m.fam.decode(m.answer_greedy(e, 1).cpu().numpy())
            if record:
                codes[sl, t] = q.float().cpu().numpy(); H[sl, t] = m.last_h.cpu().numpy()
            prev = e[:, None]
    m.keep_h = False
    return (pred, codes, H) if record else pred


def acc_report(pred, P):
    C = pred == P
    return {"test_acc": float(C.all(1).mean()), "acc_by_column": {"per_column": [round(float(v), 4) for v in C.mean(0)],
            "cumulative": [round(float(C[:, :k + 1].all(1).mean()), 4) for k in range(C.shape[1])]}, "digit_acc": float(C.mean()), "n": int(len(P))}


def donor(m, A, Bd, dev, amp, bs, rng, positions=(2, 4, 6), n=2000):
    """SERIAL (§137): the code after column p of a donor problem replaces the recipient's; scored on the recipient's columns p+1 .. L.
    Digit-level shares (note 137-1) + row-level variants (all later digits)."""
    N = min(n, len(A)); ir = rng.permutation(len(A))[:N]; idn = rng.permutation(len(A))[:N]; Ar, Br, Ad, Bdn = A[ir], Bd[ir], A[idn], Bd[idn]
    L = A.shape[1] + Bd.shape[1]; Pr, _, _ = columns(Ar, Br); _, Cd, _ = columns(Ad, Bdn); _, dcodes, _ = rollout(m, Ad, Bdn, dev, amp, bs, record=True); res = {}
    for p in positions:
        if p >= L:
            continue
        got = rollout(m, Ar, Br, dev, amp, bs, override=(p, dcodes[:, p - 1])); gt = Pr.copy(); carry = Cd[:, p - 1].copy(); mb = Br.shape[1]
        for k in range(p, L):                                               # the recipient's columns with the donor's carry
            s = carry.copy()
            for i in range(max(0, k - mb + 1), min(Ar.shape[1], k + 1)):
                s += Ar[:, i] * Br[:, k - i]
            gt[:, k] = s % 10; carry = s // 10
        t = slice(p, L)
        res["pos%d" % p] = {"donor_consistent": float((got[:, t] == gt[:, t]).mean()), "recipient_consistent": float((got[:, t] == Pr[:, t]).mean()),
                            "chance_overlap": float((gt[:, t] == Pr[:, t]).mean()),
                            "row_level": {"donor_consistent": float((got[:, t] == gt[:, t]).all(1).mean()), "recipient_consistent": float((got[:, t] == Pr[:, t]).all(1).mean()),
                                          "chance_overlap": float((gt[:, t] == Pr[:, t]).all(1).mean())}, "n": N, "later_columns": L - p}
    return res


@torch.no_grad()
def contraction(m, A, Bd, dev, amp, T0=2, n=500, seed=12301):
    """fsq120.contraction's formulas on column chains: perturb z_b after column T0 by eps * std_d * u and follow the K = L - T0
    remaining hand-offs; rho = the geometric mean of the per-hand-off median distance ratios over k = 1 .. min(8, K)."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(A))[:n]; A, Bd = A[idx], Bd[idx]; N = len(A); L = A.shape[1] + Bd.shape[1]; K = L - T0; W = min(8, K)
    m.ctx = torch.tensor(ctx_tokens(A, Bd), device=dev)
    def step(prev, t, add=None):                                            # t = 0-based column; its token is t + 1
        with amp_ctx(amp):
            h = m.segment(prev, torch.full((N,), t + 1, device=dev)); qs, q, zb = m.quant(h, noise=add); e = m.embed_code(qs)
        return zb.float(), e
    def dec(e):
        with amp_ctx(amp):
            return m.fam.decode(m.answer_greedy(e, 1).cpu().numpy())
    prev = m.start(N, dev); zs = []
    for t in range(T0 - 1):
        z, e = step(prev, t); zs.append(z); prev = e[:, None]
    pre_T0 = prev; zT0, eT0 = step(prev, T0 - 1); zs.append(zT0); clean = [zT0]; cdec = []; c = eT0[:, None]
    for k in range(K):
        z, e = step(c, T0 + k); zs.append(z); clean.append(z); cdec.append(dec(e)); c = e[:, None]
    std_d = torch.cat(zs).std(0); U = torch.tensor(rng.standard_normal((N, zT0.shape[-1])), dtype=torch.float32, device=dev); R = {}
    for eps in F.CONTRACT_EPS:
        zp, ep = step(pre_T0, T0 - 1, add=eps * std_d * U); dist = [(zp - zT0).norm(dim=-1)]; agree = []; c = ep[:, None]
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


def ridge_r2(Xf, yf, Xs, ys, lam=1e-3):
    """Linear probe: ridge (intercept, standardised features) fit on (Xf, yf), R^2 on (Xs, ys)."""
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6; Xf, Xs = (Xf - mu) / sd, (Xs - mu) / sd; Xf = np.c_[Xf, np.ones(len(Xf))]; Xs = np.c_[Xs, np.ones(len(Xs))]
    w = np.linalg.solve(Xf.T @ Xf + lam * len(Xf) * np.eye(Xf.shape[1]), Xf.T @ yf); r = ys - Xs @ w
    return float(1 - (r ** 2).sum() / max(((ys - ys.mean()) ** 2).sum(), 1e-12))


def s137_probe(m, A, Bd, dev, amp, bs, n=1024, seed=13701):
    """note 137-1: R^2 of the true carry from code_k and from h_k (before W_down), and of the context-only column sum mod 10 from
    code_k, at L = 2n. Pooled over columns (+ per column); fit on half the problems, scored on the other half."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(A))[:n]; A, Bd = A[idx], Bd[idx]; _, C, S = columns(A, Bd)
    _, Q, H = rollout(m, A, Bd, dev, amp, bs, record=True); h = len(A) // 2; L = C.shape[1]
    def pool(X, y):
        return ridge_r2(X[:h].reshape(h * L, -1), y[:h].reshape(-1).astype(float), X[h:].reshape((len(A) - h) * L, -1), y[h:].reshape(-1).astype(float))
    out = {"r2_carry_code": pool(Q, C), "r2_carry_h": pool(H, C), "r2_colsum_context": pool(Q, S), "n": int(len(A)), "L": int(L), "per_column": {}}
    for k in range(L):
        if len(np.unique(C[:, k])) > 1:
            out["per_column"][str(k + 1)] = {"r2_carry_code": ridge_r2(Q[:h, k], C[:h, k].astype(float), Q[h:, k], C[h:, k].astype(float)),
                                             "r2_colsum_context": ridge_r2(Q[:h, k], S[:h, k].astype(float), Q[h:, k], S[h:, k].astype(float))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="mult_col", choices=["mult_col"]); ap.add_argument("--task", default="4x4", choices=sorted(LMAX))
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--steps", type=int, default=24000); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--eval_bs", type=int, default=250); ap.add_argument("--test_n", type=int, default=1000)
    ap.add_argument("--ext_n", type=int, default=1000); ap.add_argument("--donor_n", type=int, default=2000); ap.add_argument("--contract_n", type=int, default=500)
    ap.add_argument("--contract_T0", type=int, default=2); ap.add_argument("--d_q", type=int, default=8); ap.add_argument("--levels", type=int, default=8)
    ap.add_argument("--sched_scale", type=float, default=1.0); ap.add_argument("--p_hard_start", type=int, default=2000); ap.add_argument("--p_hard_end", type=int, default=4000)
    ap.add_argument("--log_every", type=int, default=100); ap.add_argument("--stop_plateau", action="store_true", help="123-3's flag at 2k (restart slots)")
    ap.add_argument("--ckpt_every", type=int, default=0); ap.add_argument("--resume", default=None); ap.add_argument("--stop_after", type=int, default=0)
    ap.add_argument("--train_n", type=int, default=0, help="smoke only: read the first N train lines"); ap.add_argument("--mem_check", action="store_true")
    ap.add_argument("--from_args", default=None); ap.add_argument("--contraction_only", action="store_true"); ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--prefix_only", action="store_true"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.from_args:                                                         # post-hoc: the run's own configuration; mode / device / out from the call
        keep = {k: getattr(a, k) for k in ("contraction_only", "fp32", "prefix_only", "device", "out", "from_args")}
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
    os.makedirs(a.out, exist_ok=True); fam = MultFam(); dname = lambda: torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"  # noqa: E731
    teA, teB = load_bench(os.path.join(DATA, "mult_%s_test.txt" % a.task)); teA, teB = teA[:a.test_n], teB[:a.test_n]
    if a.contraction_only or a.prefix_only:
        m = MultCol(s5.load_gpt2(dev), fam).to(dev); m.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev)); m.eval(); m.p_hard = 1.0
        if a.contraction_only:
            ampc = None if a.fp32 else amp; ct = contraction(m, teA, teB, dev, ampc, T0=a.contract_T0, n=a.contract_n)
            ct.update({"eval_dtype": str(ampc), "device": dname(), "driver": driver_version()})
            json.dump(ct, open(os.path.join(a.out, "eval_contraction_fp32.json" if a.fp32 else "eval_contraction.json"), "w", newline="\n"), indent=1)
            print("CONTRACTION", json.dumps({k: ct[k] for k in ("rho", "rho_nonzero", "basin", "exact_recovery_share")}), flush=True)
        else:
            P, _, _ = columns(teA, teB); C = rollout(m, teA, teB, dev, amp, a.eval_bs) == P
            pf = {"acc_by_prefix": {str(t): float(C[:, :t].all(1).mean()) for t in (1, 2, 4, 8) if t <= C.shape[1]}, "acc_at_pos": {str(t + 1): float(C[:, t].mean()) for t in range(C.shape[1])},
                  "n": int(len(C)), "eval_dtype": str(amp), "device": dname(), "driver": driver_version()}
            json.dump(pf, open(os.path.join(a.out, "eval_prefix.json"), "w", newline="\n"), indent=1); print("PREFIX", json.dumps(pf), flush=True)
        return
    trA, trB = load_bench(os.path.join(DATA, "mult_train", "%s_train.txt" % a.task)) if not a.train_n else load_bench_head(os.path.join(DATA, "mult_train", "%s_train.txt" % a.task), a.train_n)
    n_train_raw = len(trA); trA, trB, n_train_removed = dedup(trA, trB, teA, teB)
    info = {**vars(a), "train_items": int(len(trA)), "train_items_raw": int(n_train_raw), "train_removed_test_overlap": n_train_removed, "train_gpu": dname(),
            "train_driver": driver_version(), **env_info()}
    json.dump({**info, "resumed_at": time.strftime("%Y-%m-%d %H:%M:%S")} if a.resume else info,
              open(os.path.join(a.out, ("args_resume_%d.json" % int(time.time())) if a.resume else "args.json"), "w", newline="\n"), indent=1)
    m = MultCol(s5.load_gpt2(dev), fam).to(dev); opt = make_opt(m, a.lr); m.train()
    log = open(os.path.join(a.out, "log.jsonl"), "a" if a.resume else "w", newline="\n"); t0 = time.time(); hist = []; L127 = {}; n_nonfinite = 0; step0 = 1
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False); m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["np_rng"]; torch.set_rng_state(ck["torch_rng"].cpu()); hist = ck["hist"]; L127 = ck["L127"]; n_nonfinite = ck["n_nonfinite"]
        if dev.startswith("cuda") and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device=dev)
        step0 = ck["step"] + 1; t0 = time.time() - ck["elapsed"]; log.write(json.dumps({"step": ck["step"], "resumed": a.resume}) + "\n"); log.flush()
    def batch(L_, bs_):
        ii = rng.integers(0, len(trA), bs_); A, Bd = trA[ii], trB[ii]; P, _, _ = columns(A, Bd)
        return torch.tensor(ctx_tokens(A, Bd), device=dev), torch.tensor(np.tile(np.arange(1, L_ + 1), (bs_, 1)), device=dev), P[:, :L_]
    if a.mem_check:                                                         # the largest training shape (L = 2n, bs) and the largest eval shape, then exit
        m.p_hard = 1.0; m.ctx, X, P = batch(2 * n, a.bs); tm = torch.ones(a.bs, 2 * n, dtype=torch.bool, device=dev)
        with amp_ctx(amp):
            loss, _, _, _ = m.chain(m.start(a.bs, dev), X, P, tm)
        loss.backward(); m.eval(); rollout(m, gen_operands(rng, a.eval_bs, n), gen_operands(rng, a.eval_bs, 2 * n), dev, amp, a.eval_bs, record=True)
        pk = torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.startswith("cuda") else None
        json.dump({"peak_gib": pk, "bs": a.bs, "eval_bs": a.eval_bs, "task": a.task}, open(os.path.join(a.out, "mem_check.json"), "w", newline="\n"), indent=1); print("MEM", pk); return
    for step in range(step0, a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(a.lr, gr, step, a.steps)
        m.p_hard = 0.0 if step <= a.p_hard_start * sc else min(1.0, (step - a.p_hard_start * sc) / ((a.p_hard_end - a.p_hard_start) * sc)); m.train()
        L = int(rng.integers(1, lmax(step / sc, a.task) + 1)); m.ctx, X, P = batch(L, a.bs); tm = torch.ones(a.bs, L, dtype=torch.bool, device=dev)
        opt.zero_grad()
        with amp_ctx(amp):
            loss, _, _, _ = m.chain(m.start(a.bs, dev), X, P, tm)            # CE at every column (mean over columns)
        loss.backward(); tot = float(loss.item()); gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if not (math.isfinite(tot) and bool(torch.isfinite(gn))):          # 127-3 guard: skip the update, log it
            bad = "loss" if not math.isfinite(tot) else next((n_ for n_, p_ in m.named_parameters() if p_.grad is not None and not bool(torch.isfinite(p_.grad).all())), "grad_norm")
            log.write(json.dumps({"step": step, "nonfinite": 1, "first": bad}) + "\n"); log.flush(); n_nonfinite += 1; opt.zero_grad()
        else:
            opt.step(); hist.append(tot)
        if step % a.log_every == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(hist)) if hist else None, "L_max": lmax(step / sc, a.task), "p_hard": m.p_hard, "sec": round(time.time() - t0)}; hist = []
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True); L127[step] = rec["loss"]
        if step == int(2000 * sc) and L127:                                 # 123-3's flag: logged for EVERY attempt (note 137-1); stops only with --stop_plateau
            gl = lambda t_: L127[min(L127, key=lambda s_: abs(s_ - t_ * sc))]  # noqa: E731
            fl = bool(gl(2000) >= 0.9 * gl(1000) and gl(2000) >= 0.25 * gl(100))
            log.write(json.dumps({"step": step, "flag_123_3": fl, "loss_100": gl(100), "loss_1k": gl(1000), "loss_2k": gl(2000), "stop_plateau": a.stop_plateau}) + "\n"); log.flush()
            if fl and a.stop_plateau:
                log.write(json.dumps({"step": step, "stopped": "plateau (123-3 flag at 2k)", "loss_100": gl(100), "loss_1k": gl(1000), "loss_2k": gl(2000)}) + "\n")
                log.close(); print("STOPPED plateau at 2k", flush=True); return
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps:   # atomic, previous kept (note 135-6)
            cp = os.path.join(a.out, "ckpt_last.pt")
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(dev) if dev.startswith("cuda") else None,
                        "np_rng": rng.bit_generator.state, "step": step, "hist": hist, "L127": L127, "n_nonfinite": n_nonfinite, "elapsed": time.time() - t0}, cp + ".tmp")
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            os.replace(cp + ".tmp", cp); log.write(json.dumps({"step": step, "ckpt": "ckpt_last.pt"}) + "\n"); log.flush()
        if a.stop_after and step >= a.stop_after:
            log.close(); print("STOP_AFTER", step, flush=True); return
    train_sec = round(time.time() - t0); torch.save(m.state_dict(), os.path.join(a.out, "model.pt")); m.eval(); m.p_hard = 1.0; t1 = time.time()
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    Pte, Cte, _ = columns(teA, teB); pred, codes, _ = rollout(m, teA, teB, dev, amp, a.eval_bs, record=True); res = acc_report(pred, Pte)
    erng = np.random.default_rng(13700); ti = erng.integers(0, len(trA), 1000); Ptr, _, _ = columns(trA[ti], trB[ti])
    res.update({"family": a.family, "task": a.task, "seed": a.seed, "long_acc": res["test_acc"], "train_acc": float((rollout(m, trA[ti], trB[ti], dev, amp, a.eval_bs) == Ptr).all(1).mean()),
                "train_sec": train_sec, "final_loss": L127[max(L127)] if L127 else None, "n_nonfinite": n_nonfinite, "train_removed_test_overlap": n_train_removed,
                "eval_dtype": str(amp), "gpu": dname(), "train_gpu": info["train_gpu"], "train_driver": info["train_driver"]})
    if a.task == "4x4" and a.ext_n:                                         # C2 descriptive: 4x8 (b 8 digits; columns > 8 unseen)
        xA, xB = gen_operands(np.random.default_rng(13704), a.ext_n, 4), gen_operands(np.random.default_rng(13705), a.ext_n, 8); xA, xB, rem = dedup(xA, xB, teA, teB)
        Px, _, _ = columns(xA, xB); res["ext_4x8"] = {**acc_report(rollout(m, xA, xB, dev, amp, a.eval_bs), Px), "dedup_removed": rem, "seeds": [13704, 13705]}
    res["donor"] = donor(m, teA, teB, dev, amp, a.eval_bs, np.random.default_rng(13702), n=a.donor_n)
    ct = contraction(m, teA, teB, dev, amp, T0=a.contract_T0, n=a.contract_n)
    res.update({"rho": ct["rho"], "rho_nonzero": ct["rho_nonzero"], "basin": ct["basin"], "exact_recovery_share": ct["exact_recovery_share"], "contraction": ct["detail"]})
    res["s137_probe"] = s137_probe(m, teA, teB, dev, amp, a.eval_bs)
    flat = codes.reshape(-1, codes.shape[-1]); res["n_codes"] = int(len(np.unique(flat, axis=0))); per = {}
    for cd, cv in zip(map(bytes, flat.astype(np.int8)), Cte.reshape(-1)):
        per.setdefault(int(cv), set()).add(cd)
    res["codes_per_carry"] = float(np.mean([len(v) for v in per.values()])); res["eval_sec"] = round(time.time() - t1)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):
        if os.path.exists(os.path.join(a.out, ck_)):
            os.remove(os.path.join(a.out, ck_))
    print("EVAL", json.dumps({k: res[k] for k in ("test_acc", "train_acc", "rho", "n_codes")}), flush=True); print("DONE", a.out, flush=True)


def load_bench_head(path, n):
    """Smoke only: the first n lines of a bench file (same parsing / checks)."""
    import itertools
    tmp = path + ".head%d.tmp" % os.getpid()
    with open(path) as f, open(tmp, "w") as g:
        g.writelines(itertools.islice(f, n))
    try:
        return load_bench(tmp)
    finally:
        os.remove(tmp)


if __name__ == "__main__":
    main()
