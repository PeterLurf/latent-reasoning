"""THEORY §132: in-context transition tables on §120's FSQ carrier (size ladder). Reuses fsq120's model unchanged (FSQ120: FSQ
d = 8, L = 8, W_up / W_down / LN, the answer decoder, the FINAL chain loss) and its schedule (L_max 2 / 4 / 8 / 16 / 32, soft -> hard
rounding over 2k-4k, warmup + cosine, 24k steps, bs 64, lr 1e-4, AdamW wd 0.01, clip 1.0) with 127-3's non-finite guard; only the
segment gains a per-problem table prefix:  segment = [EOT, table text, carried code, item]  (the state lives only in the code).
Table: m ops (" a" " b" " c"), each a uniformly random permutation of k states; text = explicit pairs in canonical order
" a0>3 a1>5 ... b0>2 ...", 4 tokens per pair (' a', '0', '>', '3'). Start state 0 (the code-zero start, as §128's families).
Answer = the state digit " d" through the decoder [EOT, code] (no table).
Splits: rung 1 (k 4, m 2): all 576 tables; OOD = the 240 with op a an involution (a o a = id); the other 336 shuffled with
default_rng(12900): held-out = the first 68, train = the remaining 268 (written to <out>/split.json). Rung 2 (k 6, m 3): held-out =
2,000 distinct random non-involution tables from default_rng(12901); OOD = 500 involution-a tables from default_rng(12902); training
draws fresh tables, rejecting involution-a tables and held-out hashes (rejections and the train / held-out overlap - must be 0 -
logged in eval.json). Micro-batching: a training batch is split into the smallest number of micro-batches (a divisor of bs) with
rows x L x segment length <= --max_tokens (gradient accumulation; memory only).
eval.json (score_s132 keys): long_acc (cumulative at 512, test_n problems on held-out tables), train_acc (the same on training-
distribution tables; note 132-1), long_acc_transitive (held-out tables reaching all k states from 0; note 132-2), heldout_n_tables, heldout_overlap (training draws that hit a held-out table; must be 0), acc_by_length, ext_2048
({"cumulative"}, ext_n x 2,048), donor (same table, positions 2 / 4 / 6 on length-8 chains, with chance_overlap), donor_difftable
(the donor code from a problem with a different held-out table, read against the recipient's table: donor target = the donor's
absolute state then the recipient's items under the recipient's table), table_swap (a new held-out table from position 256:
accuracy before / after, cumulative and per position), ood_acc (cumulative at 512 on OOD tables), cert_sample (fsq120's CERT per
table on 20 held-out tables: holds, n_moore, status, reachable_states; moore_eq_reachable counts n_moore == reachable states, 132-2), probe_ophist at 2k and at the end (R^2 of z_b on the last 1 / 2 / 3 ops and on
the state; positions 2..7 of 1,024 length-8 chains on training tables; per-group means fit on half the chains, scored on the other
half), plus final_loss, train_gpu / train_driver, n_nonfinite, split / overlap info. --stop_plateau as fsq120 (123-3 at 2k).
    python -m boxes.table132 --k 4 --m 2 --seed 1 --out boxes/runs_fsq/s132_k4_s1
"""
import os
os.environ["BOXES_NB"] = "5"
import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
from boxes import core as bc  # noqa: E402
from boxes.fsq120 import DIGTOK, EOT, FSQ120, _r2, certificate, driver_version, lmax_at, lr_at, make_opt  # noqa: E402

OPTOK = [bc.tok(" " + c)[0] for c in "abc"]
NUMTOK = [bc.tok(str(d))[0] for d in range(10)]
GTTOK = bc.tok(">")[0]
assert bc.tok(" a0>3") == [OPTOK[0], NUMTOK[0], GTTOK, NUMTOK[3]], bc.tok(" a0>3")


class TableFam:
    """§132 family: k states, m ops; delta needs a table. Holds one fixed table for CERT (delta(s, x) = T[x, s])."""
    def __init__(self, k, m, T=None):
        self.name = "table"; self.k, self.m = k, m; self.n_states, self.n_items = k, m; self.T = T
        self.item_tok = np.array([[OPTOK[o]] for o in range(m)])

    def delta(self, s, x):
        return self.T[np.asarray(x), np.asarray(s)]

    def answer_tokens(self, s):
        return np.array(DIGTOK)[np.asarray(s)][:, None]

    def decode(self, toks):
        lut = {t: i for i, t in enumerate(DIGTOK[:self.k])}
        return np.array([lut.get(int(v), -1) for v in toks[:, 0]])


def table_tokens(T):
    """T (m, k) -> the table text token ids (m * k * 4)."""
    m, k = T.shape
    return np.array([t for o in range(m) for s in range(k) for t in (OPTOK[o], NUMTOK[s], GTTOK, NUMTOK[int(T[o, s])])])


def involution_a(T):
    return bool((T[0][T[0]] == np.arange(T.shape[1])).all())


def reachable(T):
    """States reachable from start 0 under the table's ops (note 132-2)."""
    seen = {0}; fr = [0]
    while fr:
        fr = [int(T[o, s_]) for s_ in fr for o in range(T.shape[0]) if int(T[o, s_]) not in seen]; seen |= set(fr)
    return len(seen)


def thash(T):
    return hashlib.sha1(np.asarray(T, np.int8).tobytes()).hexdigest()


class Table132(FSQ120):
    """FSQ120 with a per-problem table prefix: segment = [EOT, table text, carried code, item]. self.ctx: (B or 1, Tc) token ids."""
    def __init__(self, g, fam):
        super().__init__(g, fam); self.ctx = None

    def segment(self, prev, items):
        g = self.g; B = items.shape[0]; eot = g.embed(torch.full((B, 1), EOT, device=items.device))
        c = self.ctx if self.ctx.shape[0] == B else self.ctx.expand(B, -1)
        x = torch.cat([eot, g.embed(c), prev.to(eot.dtype), g.embed(self.itok[items])], 1); h, _ = g(x, None)
        return h[:, -1]


class Splits:
    def __init__(self, k, m):
        self.k, self.m = k, m
        if (k, m) == (4, 2):
            perms = [np.array(p) for p in itertools.permutations(range(k))]
            allT = [np.stack(c) for c in itertools.product(perms, repeat=m)]
            ood = [T for T in allT if involution_a(T)]; rest = [T for T in allT if not involution_a(T)]
            order = np.random.default_rng(12900).permutation(len(rest))
            self.held = [rest[i] for i in order[:68]]; self.train = [rest[i] for i in order[68:]]; self.ood = ood
            assert len(allT) == 576 and len(ood) == 240 and len(self.held) == 68 and len(self.train) == 268
        else:
            r1 = np.random.default_rng(12901); held, seen = [], set()
            while len(held) < 2000:
                T = np.stack([r1.permutation(k) for _ in range(m)])
                if not involution_a(T) and thash(T) not in seen:
                    seen.add(thash(T)); held.append(T)
            r2 = np.random.default_rng(12902); ood = []
            while len(ood) < 500:
                T = np.stack([r2.permutation(k) for _ in range(m)])
                if involution_a(T):
                    ood.append(T)
            self.held, self.ood, self.train = held, ood, None
        self.held_hash = {thash(T) for T in self.held}; self.rejected = 0; self.overlap = 0

    def draw_train(self, rng, n):
        if self.train is not None:
            out = [self.train[i] for i in rng.integers(0, len(self.train), n)]
        else:
            out = []
            while len(out) < n:
                T = np.stack([rng.permutation(self.k) for _ in range(self.m)])
                if involution_a(T) or thash(T) in self.held_hash:
                    self.rejected += 1; continue
                out.append(T)
        self.overlap += sum(thash(T) in self.held_hash for T in out)          # must stay 0
        return out


def train_tables(sp, rng, n):
    """n tables from the training distribution with the caller's RNG (no effect on the training sampler's counters)."""
    if sp.train is not None:
        return [sp.train[i] for i in rng.integers(0, len(sp.train), n)]
    out = []
    while len(out) < n:
        T = np.stack([rng.permutation(sp.k) for _ in range(sp.m)])
        if not involution_a(T) and thash(T) not in sp.held_hash:
            out.append(T)
    return out


def problems(rng, tables, L, m):
    """tables: list of (m, k) -> ctx tokens (B, Tc), items (B, L), labels (B, L) (start state 0)."""
    T = np.stack(tables); B = len(tables); x = rng.integers(0, m, (B, L)); lab = np.zeros((B, L), np.int64); s = np.zeros(B, np.int64)
    for t in range(L):
        s = T[np.arange(B), x[:, t], s]; lab[:, t] = s
    return np.stack([table_tokens(Ti) for Ti in tables]), x, lab


def amp_ctx(amp):
    return torch.autocast("cuda", dtype=amp, enabled=amp is not None)


@torch.no_grad()
def rollout(m, ctx, X, dev, amp, bs, ctx2=None, switch=None):
    """Hard chains; decode at every position. ctx2 / switch: a new table prefix from position `switch` (0-based) on."""
    N, L = X.shape; pred = np.full((N, L), -1)
    for i in range(0, N, bs):
        sl = slice(i, i + bs); B = len(X[sl]); prev = m.start(B, dev)
        for t in range(L):
            m.ctx = torch.tensor((ctx2 if (ctx2 is not None and t >= switch) else ctx)[sl], device=dev)
            with amp_ctx(amp):
                h = m.segment(prev, torch.tensor(X[sl, t], device=dev)); qs, q, zb = m.quant(h); e = m.embed_code(qs)
                pred[sl, t] = m.fam.decode(m.answer_greedy(e, 1).cpu().numpy())
            prev = e[:, None]
    return pred


@torch.no_grad()
def donor(m, sp, dev, amp, rng, n=2000, bs=500, diff=False):
    """Same-table (diff=False) or different-table donor at positions 2 / 4 / 6 on length-8 chains (held-out tables)."""
    ia = rng.integers(0, len(sp.held), n); ib = (ia + rng.integers(1, len(sp.held), n)) % len(sp.held) if diff else ia
    TA = [sp.held[i] for i in ia]; TB = [sp.held[i] for i in ib]
    cA, xA, lA = problems(rng, TA, 8, sp.m); cB, xB, lB = problems(rng, TB, 8, sp.m); TAa = np.stack(TA); res = {}
    for pos in (2, 4, 6):
        got = []
        for i in range(0, n, bs):
            sl = slice(i, i + bs); B = len(xA[sl])
            with amp_ctx(amp):
                m.ctx = torch.tensor(cB[sl], device=dev); _, _, codesB, _ = m.chain(m.start(B, dev), torch.tensor(xB[sl], device=dev), record=True)
                m.ctx = torch.tensor(cA[sl], device=dev)
                _, eA, _, _ = m.chain(m.start(B, dev), torch.tensor(xA[sl], device=dev), override=(pos - 1, codesB[pos - 1].float()))
                got.append(m.fam.decode(m.answer_greedy(eA, 1).cpu().numpy()))
        got = np.concatenate(got); d = lB[:, pos - 1].copy()
        for t in range(pos, 8):
            d = TAa[np.arange(n), xA[:, t], d]
        res["pos%d" % pos] = {"donor_consistent": float((got == d).mean()), "recipient_consistent": float((got == lA[:, -1]).mean()),
                              "chance_overlap": float((d == lA[:, -1]).mean())}
    return res


@torch.no_grad()
def probe_ophist(m, sp, dev, amp, n=1024, seed=13206):
    """R^2 of z_b on the last 1 / 2 / 3 ops and on the state (positions 2..7 of length-8 chains on training-distribution tables)."""
    was = m.training; m.eval(); rng = np.random.default_rng(seed)
    tables = train_tables(sp, rng, n)
    c, x, lab = problems(rng, tables, 8, sp.m); m.ctx = torch.tensor(c, device=dev); prev = m.start(n, dev); Z = []
    for t in range(8):
        with amp_ctx(amp):
            h = m.segment(prev, torch.tensor(x[:, t], device=dev)); qs, q, zb = m.quant(h); e = m.embed_code(qs)
        Z.append(zb.float().cpu().numpy()); prev = e[:, None]
    Z = np.stack(Z, 1); P = list(range(2, 8)); half = n // 2; M = sp.m
    keys = {"last1": x[:, P], "last2": x[:, [p - 1 for p in P]] * M + x[:, P], "last3": (x[:, [p - 2 for p in P]] * M + x[:, [p - 1 for p in P]]) * M + x[:, P],
            "state": lab[:, P]}
    Zf, Zs = Z[:half][:, P].reshape(-1, Z.shape[-1]), Z[half:][:, P].reshape(-1, Z.shape[-1]); out = {"n": n, "positions": "2-7"}
    for nm, g_ in keys.items():
        out["r2_" + nm] = _r2(g_[:half].reshape(-1), g_[half:].reshape(-1), Zf, Zs)
    m.train(was); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, required=True); ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--steps", type=int, default=24000); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--eval_bs", type=int, default=250); ap.add_argument("--test_n", type=int, default=2000)
    ap.add_argument("--test_len", type=int, default=512); ap.add_argument("--ext_n", type=int, default=500); ap.add_argument("--donor_n", type=int, default=2000)
    ap.add_argument("--cert_n", type=int, default=20); ap.add_argument("--cert_time", type=int, default=600); ap.add_argument("--cert_cap", type=int, default=1_000_000)
    ap.add_argument("--max_tokens", type=int, default=40000, help="micro-batch so that rows x L x segment length <= this (memory only)")
    ap.add_argument("--sched_scale", type=float, default=1.0, help="smoke only"); ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--stop_plateau", action="store_true"); ap.add_argument("--check_marker", default=None, help="as fsq120 (note 128-1)")
    ap.add_argument("--mem_check", type=int, default=0, help="run N steps at L_max 32, print peak memory, exit")
    ap.add_argument("--stop_after", type=int, default=0, help="local prototype only (docs/LOCAL_BACKLOG.md): stop after this step (log kept, no model / eval)")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); sc = a.sched_scale
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    os.makedirs(a.out, exist_ok=True); sp = Splits(a.k, a.m); fam = TableFam(a.k, a.m)
    seglen = 1 + a.m * a.k * 4 + 1 + 1
    json.dump({**vars(a), "family": "table", "arm": "final", "segment_tokens": seglen, "train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
               "train_driver": driver_version()}, open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    json.dump({"k": a.k, "m": a.m, "held_out": [T.tolist() for T in sp.held], "ood": [T.tolist() for T in sp.ood],
               "train": None if sp.train is None else [T.tolist() for T in sp.train]}, open(os.path.join(a.out, "split.json"), "w", newline="\n"))
    m = Table132(s5.load_gpt2(dev), fam).to(dev); opt = make_opt(m, a.lr)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time(); hist = []; L127 = {}; n_nonfinite = 0; probes = {}
    probes["init"] = probe_ophist(m, sp, dev, amp); m.train()
    for step in range(1, a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(a.lr, gr, step, a.steps)
        m.p_hard = 0.0 if step <= 2000 * sc else min(1.0, (step - 2000 * sc) / (2000 * sc)); m.train()
        L = 32 if a.mem_check else int(rng.integers(1, lmax_at(step / sc) + 1))
        c, x, lab = problems(rng, sp.draw_train(rng, a.bs), L, a.m)
        need = math.ceil(a.bs * L * seglen / a.max_tokens); nmb = next(d for d in range(1, a.bs + 1) if a.bs % d == 0 and d >= need)
        opt.zero_grad(); tot = 0.0
        for mi in range(nmb):
            sl = slice(mi * a.bs // nmb, (mi + 1) * a.bs // nmb); m.ctx = torch.tensor(c[sl], device=dev)
            with amp_ctx(amp):
                loss, _, _, _ = m.chain(m.start(len(x[sl]), dev), torch.tensor(x[sl], device=dev), lab[sl])
            (loss / nmb).backward(); tot += loss.item() / nmb
        gnorm_ = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if not (math.isfinite(tot) and bool(torch.isfinite(gnorm_))):        # 127-3 guard: skip the update, log it
            log.write(json.dumps({"step": step, "nonfinite": 1}) + "\n"); log.flush(); n_nonfinite += 1; opt.zero_grad()
        else:
            opt.step(); hist.append(tot)
        if a.mem_check and step >= a.mem_check:
            mx = torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.startswith("cuda") else 0
            print("MEMCHECK k %d m %d L 32 seglen %d micro %d peak %.2f GiB" % (a.k, a.m, seglen, nmb, mx), flush=True); return
        if step % a.log_every == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(hist)) if hist else None, "L_max": lmax_at(step / sc), "p_hard": m.p_hard, "micro": nmb,
                   "sec": round(time.time() - t0)}; hist = []
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
            if rec["loss"] is not None:
                L127[step] = rec["loss"]
        if step == int(2000 * sc):
            probes["2k"] = probe_ophist(m, sp, dev, amp); log.write(json.dumps({"step": step, "probe_ophist": probes["2k"]}) + "\n"); log.flush(); m.train()
        if a.stop_plateau and step == int(2000 * sc) and L127:              # 123-3's flag (as fsq120)
            gl = lambda t_: L127[min(L127, key=lambda s_: abs(s_ - t_ * sc))]  # noqa: E731
            if gl(2000) >= 0.9 * gl(1000) and gl(2000) >= 0.25 * gl(100):
                claimed = False
                if a.check_marker:
                    try:
                        os.makedirs(os.path.dirname(a.check_marker) or ".", exist_ok=True); os.mkdir(a.check_marker); claimed = True
                    except FileExistsError:
                        claimed = False
                if claimed:
                    open(os.path.join(a.out, "CHECK_RUN"), "w").write(a.out + "\n")
                    log.write(json.dumps({"step": step, "flagged": "plateau (123-3 flag at 2k)", "check_run": True, "loss_100": gl(100), "loss_1k": gl(1000), "loss_2k": gl(2000)}) + "\n"); log.flush()
                else:
                    log.write(json.dumps({"step": step, "stopped": "plateau (123-3 flag at 2k)", "loss_100": gl(100), "loss_1k": gl(1000), "loss_2k": gl(2000)}) + "\n")
                    log.close(); print("STOPPED plateau at 2k", flush=True); return
        if a.stop_after and step >= a.stop_after:                           # local prototype only: no model / eval
            log.write(json.dumps({"step": step, "stopped": "stop_after"}) + "\n"); log.close(); print("STOP_AFTER", step, flush=True); return
    train_sec = round(time.time() - t0); torch.save(m.state_dict(), os.path.join(a.out, "model.pt")); m.eval()
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    t1 = time.time(); er = np.random.default_rng(13200)
    res = {"family": "table", "k": a.k, "m": a.m, "seed": a.seed, "train_sec": train_sec, "final_loss": L127[max(L127)] if L127 else None,
           "n_nonfinite": n_nonfinite, "eval_dtype": str(amp), "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
           "train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_driver": driver_version(),
           "split": {"held_out": len(sp.held), "ood": len(sp.ood), "train": None if sp.train is None else len(sp.train), "train_rejected_draws": sp.rejected,
                     "train_heldout_overlap": sp.overlap}}
    held = lambda r_, n_: [sp.held[i] for i in r_.integers(0, len(sp.held), n_)]  # noqa: E731
    TT = held(er, a.test_n); c, X, Y = problems(er, TT, a.test_len, a.m); m.ctx = None; P = rollout(m, c, X, dev, amp, a.eval_bs); C = P == Y
    full = np.array([reachable(T) == a.k for T in TT])                     # note 132-2: tables reaching all k states from 0
    res["long_acc_transitive"] = float(C[full].all(1).mean()) if full.any() else None; res["transitive_share"] = float(full.mean())
    res["long_acc"] = float(C.all(1).mean()); res["heldout_n_tables"] = len(sp.held); res["heldout_overlap"] = sp.overlap   # note 132-1
    ct, Xt, Yt = problems(er, train_tables(sp, np.random.default_rng(13207), a.test_n), a.test_len, a.m)
    res["train_acc"] = float((rollout(m, ct, Xt, dev, amp, a.eval_bs) == Yt).all(1).mean())                           # note 132-1: training tables
    res["acc_by_length"] = {str(t): float(C[:, :t].all(1).mean()) for t in (16, 32, 64, 128, 256, 512) if t <= C.shape[1]}
    if a.ext_n:
        c2, X2, Y2 = problems(er, held(er, a.ext_n), 2048 if a.test_len >= 512 else a.test_len, a.m); P2 = rollout(m, c2, X2, dev, amp, a.eval_bs)
        res["ext_2048"] = {"n": a.ext_n, "len": int(X2.shape[1]), "cumulative": float((P2 == Y2).all(1).mean())}
    res["donor"] = donor(m, sp, dev, amp, np.random.default_rng(13201), n=a.donor_n)
    res["donor_difftable"] = donor(m, sp, dev, amp, np.random.default_rng(13202), n=a.donor_n, diff=True)
    sr = np.random.default_rng(13203); nsw = min(1000, a.test_n); sw = min(256, a.test_len // 2)
    T1 = held(sr, nsw); T2 = held(sr, nsw)
    xs = sr.integers(0, a.m, (nsw, a.test_len)); ys = np.zeros_like(xs); s_ = np.zeros(nsw, np.int64); A1, A2 = np.stack(T1), np.stack(T2)
    for t in range(a.test_len):
        s_ = (A1 if t < sw else A2)[np.arange(nsw), xs[:, t], s_]; ys[:, t] = s_
    ps = rollout(m, np.stack([table_tokens(T) for T in T1]), xs, dev, amp, a.eval_bs, ctx2=np.stack([table_tokens(T) for T in T2]), switch=sw); Cs = ps == ys
    res["table_swap"] = {"n": nsw, "switch": sw, "acc_before_cum": float(Cs[:, :sw].all(1).mean()), "acc_after_cum": float(Cs[:, sw:].all(1).mean()),
                         "acc_after_mean": float(Cs[:, sw:].mean()), "acc_after_first8": float(Cs[:, sw:sw + 8].mean()),
                         "same_table_share": float(np.mean([(t1 == t2).all() for t1, t2 in zip(T1, T2)]))}
    oc, ox, oy = problems(er, [sp.ood[i] for i in er.integers(0, len(sp.ood), min(1000, a.test_n))], a.test_len, a.m)
    res["ood_acc"] = float((rollout(m, oc, ox, dev, amp, a.eval_bs) == oy).all(1).mean())
    cr = np.random.default_rng(13204); cs = []
    for i in cr.choice(len(sp.held), min(a.cert_n, len(sp.held)), replace=False):
        T = sp.held[i]; ft = TableFam(a.k, a.m, T); m.fam = ft; m.ctx = torch.tensor(table_tokens(T)[None], device=dev)
        cc = certificate(m, ft, dev, amp, a.cert_cap, time_limit=a.cert_time)
        cs.append({"table": T.tolist(), "holds": cc.get("holds"), "n_moore": cc.get("n_moore"), "status": cc.get("status"), "n_reachable": cc.get("n_reachable"),
                   "reachable_states": reachable(T)})
    m.fam = fam
    res["cert_sample"] = {"n": len(cs), "holds": sum(1 for x_ in cs if x_["holds"]), "moore_eq_reachable": sum(1 for x_ in cs if x_["n_moore"] == x_["reachable_states"]),
                         "moore_eq_k": sum(1 for x_ in cs if x_["n_moore"] == a.k), "tables": cs}   # note 132-2: moore_eq_reachable is the test
    probes["end"] = probe_ophist(m, sp, dev, amp); res["probe_ophist"] = probes; res["eval_sec"] = round(time.time() - t1)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps({k_: res[k_] for k_ in ("long_acc", "ood_acc")}), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
