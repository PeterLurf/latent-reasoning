"""THEORY §173 CONV-HALT (registration 2026-10-09 01:57 EDT; scorer theory/score_s173.py), post hoc on a think-pass run's model.pt -> <run>/eval_s173_halt.json.
a(t) = the decoder's greedy answer read from M_t, t = 0 .. 2K (a(0) = the K' = 0 answer: the zero state, or for §165 the state after the input chunks).
    "answer2"  (the rule): h = the first t >= 2 with a(t) = a(t-1) = a(t-2); none by 2K -> h = 2K (capped). The answer is a(h).
    descriptive, eps = 0.01, the same cap, h = the first t >= 1 whose change is < eps:
    "read_0.01"  ||r_t - r_{t-1}|| / ||r_t||: r_t = the decoder's memory read of M_t (pre-W_o [g_add read_add || g_del read_del]) at the position that
                 predicts the first answer token (the last position of the decoder's first forward call: " =" for §169 / §170 / §171, EOT for §165;
                 for §170's right-padded batch, each item's own " =")
    "delta_0.01" ||Sd_t - Sd_{t-1}|| / ||Sd_t|| on the 6 gated-delta heads' state only
    "full_0.01"  the same on the full state
Fields (score_s173): halt {rule: {size: {acc, mean_h, frac_capped}}}, halt_fixed_k {size: acc at t = K}; plus acc_by_t {size: [acc at t = 0 .. 2K]}.
Sizes: 4x4 (the bench test), 4x6 / 4x8 / 4x16 (§161's ext seeds) for the multiplication arms; n in {2, 4, 6, 8, 10, 12, 16} (§170's test_set) for §170.
Arms by args.json: boxes.fixedchunk (§165: "chunk_k" in args), inject169 (arm "inject", "shape_mix" in args), nested170 (task "nested" / arm "inject" with
n_max_train), readslots171 (arm "readslots").
    python -m boxes.halt173 --run boxes/runs_fsq/s169_k8_4x4_s1 [--n 1000] [--device cuda:0]"""
import os
os.environ.setdefault("BOXES_NB", "5")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
from boxes.multcol import gen_operands, load_bench  # noqa: E402
from boxes.fixedchunk import PAD, product_digits, amp_ctx  # noqa: E402

EXT = ((6, 14016, 14017), (8, 14004, 14005), (16, 14006, 14007))
NEST = (2, 4, 6, 8, 10, 12, 16)
EPS = 0.01


class ReadTap:
    """records the residual entering the read (block mem.layer) on the FIRST read_fixed call after arm(); the decoder's first forward call."""
    def __init__(self):
        self.x = None; self.on = False; self._orig = DM.read_fixed

    def __enter__(self):
        tap = self

        def rf(mem, x, S, rec=None):
            if tap.on and tap.x is None:
                tap.x = x.detach()
            return tap._orig(mem, x, S, rec)
        DM.read_fixed = rf
        return self

    def __exit__(self, *e):
        DM.read_fixed = self._orig

    def arm(self):
        self.x = None; self.on = True


def read_vec(mem, x, S):
    """read_fixed's reads at one position per row: x (b, D) residual, S (b, 12, DH, DH) -> (b, 12 * DH) [g_add read_add || g_del read_del]."""
    with torch.autocast("cuda", enabled=False):
        b, D = x.shape; q = mem.qkv(mem.ln(x.float()))[..., :D].view(b, DM.H_ADD + DM.H_DEL, DM.DH)
        oa = torch.exp(mem.s_add) * torch.einsum("bhk,bhkv->bhv", Fn.elu(q[:, :DM.H_ADD]) + 1, S[:, :DM.H_ADD].float())
        od = torch.exp(mem.s_del) * torch.einsum("bhvk,bhk->bhv", S[:, DM.H_ADD:].float(), Fn.normalize(q[:, DM.H_ADD:], dim=-1))
        return torch.cat([oa, od], 1).flatten(1)


# ---- adapters: init(batch) -> (S0, ctx); step(S, ctx) -> S; decode(S, ctx) -> (answers (b, L) or (b,), rpos (b,) or None); truth(batch)
class MultAdapter:
    def __init__(self, kind, m, a, dev):
        self.kind, self.m, self.a, self.dev = kind, m, a, dev

    def init(self, AB):
        A, B = AB; m = self.m
        if self.kind == "fixedchunk":
            S = m.encode(A, B, self.a["chunk_k"], self.dev); return S, {"A": A, "B": B, "ids": m.ids(torch.full((len(A), 1), PAD, device=self.dev))}
        if self.kind == "inject":
            from boxes.inject169 import think_text
            return m.mem.zero_state(len(A), self.dev), {"A": A, "B": B, "ids": m.ids(torch.tensor(think_text(A, B), device=self.dev))}
        return m.mem.zero_state(len(A), self.dev), {"A": A, "B": B, "kv": m.encode(A, B, self.dev)}

    def step(self, S, c):
        return self.m.slot_pass(S, c["kv"]) if self.kind == "readslots" else self.m.write(S, c["ids"])

    def decode(self, S, c):
        L = c["A"].shape[1] + c["B"].shape[1]
        if self.kind == "fixedchunk":
            return self.m.greedy(S, L), None
        if self.kind == "inject":
            return self.m.greedy_ans(S, c["A"], c["B"], L), None
        return self.m.greedy_ans(S, c["kv"], L), None

    @staticmethod
    def truth(AB):
        return product_digits(*AB)


class NestAdapter:
    def __init__(self, m, dev):
        self.m, self.dev = m, dev

    def init(self, items):
        x, msk = self.m.think_inputs(items, self.dev); return self.m.mem.zero_state(len(items), self.dev), {"items": items, "x": x, "msk": msk}

    def step(self, S, c):
        return self.m.write(S, c["x"], c["msk"])

    def decode(self, S, c):
        from boxes.nested170 import NUM_IDS, ids
        lg = self.m.dec_logits(S, c["items"], self.dev); return lg[:, NUM_IDS].argmax(-1).cpu().numpy(), np.array([len(ids(it["syms"])) + 1 for it in c["items"]])

    @staticmethod
    def truth(items):
        return np.array([it["t"] for it in items])


@torch.no_grad()
def trace(ad, batch, T, dev, amp):
    """per t = 0 .. T: correct (b,), answers, read change, delta / full state change (t >= 1)."""
    truth = ad.truth(batch); cor, ans, chg = [], [], {"read": [None], "delta": [None], "full": [None]}; rprev = Sprev = None
    with ReadTap() as tap, amp_ctx(amp):
        S, c = ad.init(batch)
        for t in range(T + 1):
            if t:
                S = ad.step(S, c)
            tap.arm(); a_, rpos = ad.decode(S, c); tap.on = False
            x = tap.x[:, -1] if rpos is None else tap.x[torch.arange(tap.x.shape[0], device=dev), torch.tensor(rpos, device=dev)]
            r = read_vec(ad.m.mem, x, S); Sf = S.float()
            if t:
                chg["read"].append(((r - rprev).norm(dim=1) / r.norm(dim=1).clamp_min(1e-12)).cpu().numpy())
                dd = Sf[:, DM.H_ADD:].flatten(1); dp = Sprev[:, DM.H_ADD:].flatten(1)
                chg["delta"].append(((dd - dp).norm(dim=1) / dd.norm(dim=1).clamp_min(1e-12)).cpu().numpy())
                chg["full"].append(((Sf - Sprev).flatten(1).norm(dim=1) / Sf.flatten(1).norm(dim=1).clamp_min(1e-12)).cpu().numpy())
            rprev, Sprev = r, Sf; ans.append(a_)
            cor.append((a_ == truth).all(1) if a_.ndim == 2 else (a_ == truth))
    return np.stack(cor), ans, chg


def halts(cor, ans, chg, T):
    """-> {rule: (h (b,), capped (b,), correct at h (b,))}."""
    b = cor.shape[1]; out = {}
    same = [None] + [(ans[t] == ans[t - 1]).all(1) if ans[t].ndim == 2 else ans[t] == ans[t - 1] for t in range(1, T + 1)]
    h = np.full(b, T); cap = np.ones(b, bool)
    for t in range(2, T + 1):
        hit = cap & same[t] & same[t - 1]; h[hit] = t; cap[hit] = False
    out["answer2"] = (h, cap)
    for k in ("read", "delta", "full"):
        h = np.full(b, T); cap = np.ones(b, bool)
        for t in range(1, T + 1):
            hit = cap & (chg[k][t] < EPS); h[hit] = t; cap[hit] = False
        out["%s_%g" % (k, EPS)] = (h, cap)
    return {r: (h, c_, cor[h, np.arange(b)]) for r, (h, c_) in out.items()}


def load_model(run, dev, init=False):
    """init (note 173-1): the arm's model at initialisation - the pretrained trunk and fresh memory parameters built under the run's seed (as the run's
    own torch.manual_seed before construction), no checkpoint loaded, gamma forced to 1 so the writes use the top-layer source."""
    a = json.load(open(os.path.join(run, "args.json"))); g = s5.load_gpt2(dev)
    if init:
        torch.manual_seed(int(a.get("seed", 1)))
    if "chunk_k" in a:
        from boxes.fixedchunk import FixedChunk
        kind, m = "fixedchunk", FixedChunk(g, 6, a.get("mem_write", "top"))
    elif a.get("arm") == "readslots":
        from boxes.readslots171 import ReadSlots
        kind, m = "readslots", ReadSlots(g, a["n_slots"], 6, "top", bool(a.get("blind_decoder")))
    elif "n_max_train" in a:
        from boxes.nested170 import InjectN
        assert a.get("arm") == "inject"; kind, m = "nested", InjectN(g, 6, "top")
    else:
        from boxes.inject169 import Inject
        assert a.get("arm") == "inject"; kind, m = "inject", Inject(g, 6, a.get("mem_write", "top"))
    m = m.to(dev)
    if init:
        with torch.no_grad():
            m.mem.gamma.fill_(1.0); m.mem.fade.fill_(1.0)
    else:
        m.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location=dev, weights_only=False))
    m.eval()
    return kind, m, a


def halt_fields(kind, m, a, dev, amp, n_items=1000, bs=250, bench=os.path.join("bench", "data", "mult_4x4_test.txt")):
    """§173 on a loaded model (m.eval()): -> (halt {rule: {size: {acc, mean_h, frac_capped}}}, halt_fixed_k {size: acc}, acc_by_t {size: [...]}, n {size}).
    a: the run's args (dict); K = a["think_k"], cap 2K. Also called from a run's own eval (inject169 §176)."""
    K = int(a["think_k"]); T = 2 * K; t0 = time.time()
    if kind == "nested":
        from boxes.nested170 import test_set
        ad = NestAdapter(m, dev); TE = test_set(a.get("per_n", 1000)); sets = {str(n): [it for it in TE if it["n"] == n][:n_items] for n in NEST}
        chunks = {s: [v[i:i + bs] for i in range(0, len(v), bs)] for s, v in sets.items()}
    else:
        ad = MultAdapter(kind, m, a, dev); teA, teB = load_bench(bench if os.path.isabs(bench) else os.path.join(ROOT, bench)); sets = {"4x4": (teA[:n_items], teB[:n_items])}
        for nb_, s1_, s2_ in EXT:
            sets["4x%d" % nb_] = (gen_operands(np.random.default_rng(s1_), n_items, 4), gen_operands(np.random.default_rng(s2_), n_items, nb_))
        chunks = {s: [(A[i:i + bs], B[i:i + bs]) for i in range(0, len(A), bs)] for s, (A, B) in sets.items()}
    halt, fixed, by_t, nn = {}, {}, {}, {}
    for s, chs in chunks.items():
        cs, hs = [], []
        for ch in chs:
            cor, ans, chg = trace(ad, ch, T, dev, amp); cs.append(cor); hs.append(halts(cor, ans, chg, T))
        cor = np.concatenate(cs, 1); fixed[s] = float(cor[K].mean()); by_t[s] = [round(float(v), 4) for v in cor.mean(1)]; nn[s] = int(cor.shape[1])
        for r in hs[0]:
            h = np.concatenate([x[r][0] for x in hs]); cap = np.concatenate([x[r][1] for x in hs]); ok = np.concatenate([x[r][2] for x in hs])
            halt.setdefault(r, {})[s] = {"acc": float(ok.mean()), "mean_h": float(h.mean()), "frac_capped": float(cap.mean())}
        print("halt", s, "done", round(time.time() - t0), flush=True)
    return halt, fixed, by_t, nn


@torch.no_grad()
def dir_fields(kind, m, a, dev, amp, n_items=500, bs=250, bench=os.path.join("bench", "data", "mult_4x4_test.txt")):
    """§177 direction diagnostic (Theory, descriptive): on n_items 4x4 bench items, per pass t = 1..2K the mean over items of cos(x_t, x_{t-1}) for
    x = r (the decoder's memory read at its first query, as read_0.01), Sd (the flattened delta-head state) and Sa (the flattened additive state).
    M_0 is the zero state (§169 / §171) or the post-input state (§165); a cosine against a zero vector is skipped (counted in n_by_t)."""
    K = int(a["think_k"]); T = 2 * K; acc = {k_: {t: [] for t in range(1, T + 1)} for k_ in ("read", "delta", "add")}
    if kind == "nested":                                                    # note 173-1: nested at its training size, n pooled 3..8
        from boxes.nested170 import test_set
        ad = NestAdapter(m, dev); pool = [it for it in test_set(a.get("per_n", 1000)) if 3 <= it["n"] <= 8]
        its = [pool[i] for i in np.random.default_rng(17730).permutation(len(pool))[:n_items]]; chunks = [its[i:i + bs] for i in range(0, len(its), bs)]
    else:
        ad = MultAdapter(kind, m, a, dev); teA, teB = load_bench(bench if os.path.isabs(bench) else os.path.join(ROOT, bench))
        A, B = teA[:n_items], teB[:n_items]; chunks = [(A[i:i + bs], B[i:i + bs]) for i in range(0, len(A), bs)]

    def cos(x, y):
        nx, ny = x.norm(dim=1), y.norm(dim=1); ok = (nx > 0) & (ny > 0)
        return ((x * y).sum(1)[ok] / (nx[ok] * ny[ok])).cpu().numpy()
    for ch in chunks:
        with ReadTap() as tap, amp_ctx(amp):
            S, c = ad.init(ch); prev = None
            for t in range(T + 1):
                if t:
                    S = ad.step(S, c)
                tap.arm(); _, rpos = ad.decode(S, c); tap.on = False
                x = tap.x[:, -1] if rpos is None else tap.x[torch.arange(tap.x.shape[0], device=dev), torch.tensor(rpos, device=dev)]
                cur = {"read": read_vec(m.mem, x, S).float(), "delta": S[:, DM.H_ADD:].float().flatten(1), "add": S[:, :DM.H_ADD].float().flatten(1)}
                if prev is not None:
                    for k_ in cur:
                        acc[k_][t].append(cos(cur[k_], prev[k_]))
                prev = cur
    out = {}
    for k_, nm in (("read", "read_cos"), ("delta", "delta_cos"), ("add", "add_cos")):
        out[nm] = {str(t): (float(np.concatenate(v).mean()) if sum(len(x_) for x_ in v) else None) for t, v in acc[k_].items()}
        out[nm.replace("_cos", "_n_by_t")] = {str(t): int(sum(len(x_) for x_ in v)) for t, v in acc[k_].items()}
    out["n_items"] = int(sum(len(c[0]) if kind != "nested" else len(c) for c in chunks)); out["cap"] = T
    out["items"] = "nested: test_set n pooled 3..8, seed 17730" if kind == "nested" else "the 4x4 bench test"
    out["m0"] = ("the POST-INPUT state (the input chunks written): t = 1 compares think pass 1 with it - not comparable naively with a zero M_0 (Theory)"
                 if kind == "fixedchunk" else "the zero state (t = 1 undefined)")
    return out


@torch.no_grad()
def step_probe_delta(kind, m, a, dev, amp, bs=250, n_fit=4000, n_test=1000, bench=os.path.join("bench", "data", "mult_4x4_test.txt")):
    """note 173-1 (descriptive): multinomial logistic probes (nested170.logreg_acc, as 170-2) from the pass-t DELTA state (flattened 6 x 64 x 64), t = 1..K.
    nested: target v_t (the t-th leftmost reduction's value), fit on n_fit train_items (n <= 8, test-excluded, seed 17731), test n_test test items at n <= 8
    (seed 17022) -> {t: acc}. mult: target the units digit of column c's full sum (= product digit c, LSB first), c = 1..8 at 4x4, fit on n_fit generator
    4x4 items (bench-excluded, seed 17731), test the bench's first n_test -> {t: {c: acc}}. The passes are recomputed up to each t (one t's states in memory)."""
    from boxes.nested170 import logreg_acc
    K = int(a["think_k"])
    if kind == "nested":
        from boxes.nested170 import test_set, train_items, reductions
        ad = NestAdapter(m, dev); TE = test_set(a.get("per_n", 1000)); excl = {tuple(it["syms"]) for it in TE}
        fit = train_items(np.random.default_rng(17731), n_fit, 8, excl); pool = [it for it in TE if it["n"] <= 8]
        tst = [pool[i] for i in np.random.default_rng(17022).permutation(len(pool))[:n_test]]
        vf, vs = [reductions(it["root"]) for it in fit], [reductions(it["root"]) for it in tst]
        bf = [fit[i:i + bs] for i in range(0, len(fit), bs)]; bt = [tst[i:i + bs] for i in range(0, len(tst), bs)]
    else:
        ad = MultAdapter(kind, m, a, dev); teA, teB = load_bench(bench if os.path.isabs(bench) else os.path.join(ROOT, bench))
        ex = {(x.tobytes(), y.tobytes()) for x, y in zip(teA, teB)}; r_ = np.random.default_rng(17731); fA, fB = gen_operands(r_, 3 * n_fit, 4), gen_operands(r_, 3 * n_fit, 4)
        keep = np.array([(x.tobytes(), y.tobytes()) not in ex for x, y in zip(fA, fB)]); fA, fB = fA[keep][:n_fit], fB[keep][:n_fit]; tA, tB = teA[:n_test], teB[:n_test]
        Yf, Ys = product_digits(fA, fB), product_digits(tA, tB)
        bf = [(fA[i:i + bs], fB[i:i + bs]) for i in range(0, len(fA), bs)]; bt = [(tA[i:i + bs], tB[i:i + bs]) for i in range(0, len(tA), bs)]

    def states(batches, t):
        out = []
        for ch in batches:
            with amp_ctx(amp):
                S, c = ad.init(ch)
                for _ in range(t):
                    S = ad.step(S, c)
            out.append(S[:, DM.H_ADD:].float().flatten(1).cpu().numpy())
        return np.concatenate(out)
    res = {}
    for t in range(1, K + 1):
        Xf, Xs = states(bf, t), states(bt, t)
        if kind == "nested":
            fi = [i for i, v in enumerate(vf) if len(v) >= t]; si = [i for i, v in enumerate(vs) if len(v) >= t]
            if len(fi) >= 100 and len(si) >= 20:
                res[str(t)] = logreg_acc(Xf[fi], np.array([vf[i][t - 1] for i in fi]), Xs[si], np.array([vs[i][t - 1] for i in si]), dev)
        else:
            res[str(t)] = {str(c + 1): logreg_acc(Xf, Yf[:, c], Xs, Ys[:, c], dev) for c in range(Yf.shape[1])}
        print("step_probe_delta t", t, flush=True)
    return res


@torch.no_grad()
def dir_pair_fields(run, dev, amp, n_items=500, bs=125, bench=os.path.join("bench", "data", "mult_4x4_test.txt")):
    """§177 direction baseline on the WORKING pair-segment models (Theory: boxes.multpair --carrier deltamem, e.g. §161 MIX / §155 R): each of the
    n^2 + 1 = 17 segments (16 pairs in column order, then the flush) is one pass; per segment t = 1..17 the mean over n_items 4x4 bench items of
    cos(x_t, x_{t-1}) for x = r (the memory read of M_t at the flush segment's " =" query: the residual entering the read there does not depend on
    the memory, so one fixed query per item, as read_0.01's first-query read), Sd and Sa (flattened delta / additive state). Built as das167.MultFam."""
    import boxes.multpair as MP
    from boxes.multcol import load_bench as lb_
    a = json.load(open(os.path.join(run, "args.json"))); assert a.get("family") == "mult_pair" and a.get("carrier") == "deltamem", "a pair-segment DeltaMem run"
    m = MP.MultDelta(s5.load_gpt2(dev), MP.PairFam(), a.get("mem_layer", 6), a.get("mem_write")).to(dev); m.add_reset = a.get("add_reset"); m.cidx_off = None; m.cidx_r = None
    m.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location=dev)); m.eval()
    teA, teB = lb_(bench if os.path.isabs(bench) else os.path.join(ROOT, bench)); A, B = teA[:n_items], teB[:n_items]; pairs, col, emit = MP.schedule(4, 4); T = len(pairs)
    acc = {k_: {t: [] for t in range(1, T + 1)} for k_ in ("read", "delta", "add")}

    def cos(x, y):
        nx, ny = x.norm(dim=1), y.norm(dim=1); ok = (nx > 0) & (ny > 0)
        return ((x * y).sum(1)[ok] / (nx[ok] * ny[ok])).cpu().numpy()
    for i in range(0, len(A), bs):
        sl = slice(i, i + bs); MP.set_ops(m, A[sl], B[sl], dev); S = m.start(len(A[sl]), dev)
        with ReadTap() as tap, amp_ctx(amp):
            tap.arm(); m.step(S, T - 1); tap.on = False; xq = tap.x[:, -1]   # the flush segment's " =" query (memory-independent)
            prev = None
            for t in range(T):
                _, S = m.step(S, t)
                cur = {"read": read_vec(m.mem, xq, S).float(), "delta": S[:, DM.H_ADD:].float().flatten(1), "add": S[:, :DM.H_ADD].float().flatten(1)}
                if prev is not None:
                    for k_ in cur:
                        acc[k_][t + 1].append(cos(cur[k_], prev[k_]))
                prev = cur
    out = {}
    for k_, nm in (("read", "read_cos"), ("delta", "delta_cos"), ("add", "add_cos")):
        out[nm] = {str(t): (float(np.concatenate(v).mean()) if sum(len(x_) for x_ in v) else None) for t, v in acc[k_].items()}
        out[nm.replace("_cos", "_n_by_t")] = {str(t): int(sum(len(x_) for x_ in v)) for t, v in acc[k_].items()}
    out.update({"n_items": int(len(A)), "segments": T, "segment_def": "t = 1..16 the pairs in column order, t = 17 the flush; x_0 = the start state (zero)"})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--bs", type=int, default=250)
    ap.add_argument("--fp32", action="store_true"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", default=None)
    ap.add_argument("--bench", default=os.path.join("bench", "data", "mult_4x4_test.txt")); ap.add_argument("--dir177", action="store_true", help="§177 direction diagnostic only")
    o = ap.parse_args(); dev = o.device; t0 = time.time()
    amp = None if (o.fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    a_ = json.load(open(os.path.join(o.run, "args.json")))
    if o.dir177 and a_.get("family") == "mult_pair":                        # §177 baseline on the working pair-segment models (Theory)
        r_ = dir_pair_fields(o.run, dev, amp, min(o.n, 500), min(o.bs, 125), o.bench)
        r_.update({"run": o.run.replace(os.sep, "/"), "kind": "multpair", "eval_dtype": str(amp), "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
                   "sec": round(time.time() - t0)})
        json.dump(r_, open(o.out or os.path.join(o.run, "eval_s177_dir.json"), "w", newline="\n"), indent=1); print("DONE dir177 pair", o.run, flush=True); return
    kind, m, a = load_model(o.run, dev); K = int(a["think_k"]); T = 2 * K; assert K >= 1
    if o.dir177:                                                            # §177 (Theory): the direction diagnostic -> eval_s177_dir.json
        r_ = dir_fields(kind, m, a, dev, amp, min(o.n, 500), o.bs, o.bench)
        r_["step_probe_delta"] = step_probe_delta(kind, m, a, dev, amp, o.bs, bench=o.bench)   # note 173-1 (descriptive)
        del m; torch.cuda.empty_cache() if dev.startswith("cuda") else None
        _, m0, _ = load_model(o.run, dev, init=True); c0 = dir_fields(kind, m0, a, dev, amp, min(o.n, 500), o.bs, o.bench)
        r_["cos_at_init"] = {"read": c0["read_cos"], "delta": c0["delta_cos"], "add": c0["add_cos"],
                             "def": "the arm's model at initialisation: pretrained trunk, fresh memory parameters under the run's seed, gamma 1 (note 173-1)"}
        r_.update({"run": o.run.replace(os.sep, "/"), "kind": kind, "think_k": K, "eval_dtype": str(amp), "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
                   "sec": round(time.time() - t0), "def": "per pass t: mean over items of cos(x_t, x_{t-1}); r = the decoder's memory read at its first query (as read_0.01), "
                   "delta / add = the flattened delta-head / additive-head state (boxes/halt173.dir_fields)"})
        json.dump(r_, open(o.out or os.path.join(o.run, "eval_s177_dir.json"), "w", newline="\n"), indent=1); print("DONE dir177", o.run, flush=True); return
    halt, fixed, by_t, nn = halt_fields(kind, m, a, dev, amp, o.n, o.bs, o.bench)
    res = {"run": o.run.replace(os.sep, "/"), "kind": kind, "think_k": K, "cap": T, "eps": EPS, "halt": halt, "halt_fixed_k": fixed, "acc_by_t": by_t, "n": nn,
           "eval_dtype": str(amp), "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "sec": round(time.time() - t0),
           "rules": "answer2: first t >= 2 with a(t) = a(t-1) = a(t-2), cap 2K; *_0.01: first t >= 1 with relative change < 0.01, cap 2K (see boxes/halt173.py)"}
    json.dump(res, open(o.out or os.path.join(o.run, "eval_s173_halt.json"), "w", newline="\n"), indent=1); print("DONE", o.run, flush=True)


if __name__ == "__main__":
    main()
