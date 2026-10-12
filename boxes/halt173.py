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


def rule_k(a):
    """the run's think K: an int, or §177 (d)'s '4n' -> 16 (its rule K at the 4x4 training size)."""
    return 16 if str(a["think_k"]) == "4n" else int(a["think_k"])


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
        ha, hd = getattr(mem, "h_add", DM.H_ADD), getattr(mem, "h_del", DM.H_DEL)   # §178: the memory's own head counts
        b, D = x.shape; q = mem.qkv(mem.ln(x.float()))[..., :D].view(b, ha + hd, DM.DH)
        oa = torch.exp(mem.s_add) * DM.add_read(mem, Fn.elu(q[:, :ha]) + 1, S[:, :ha].float(), DM.state_z(mem, S))   # §185: the run's own additive read
        od = torch.exp(mem.s_del) * torch.einsum("bhvk,bhk->bhv", S[:, ha:ha + hd].float(), Fn.normalize(q[:, ha:], dim=-1))
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
        c["t"] = c.get("t", 0) + 1                                          # the global pass number (§177 (a)'s index)
        if self.kind == "readslots":
            return self.m.slot_pass(S, c["kv"])
        return self.m.write(S, c["ids"], c["t"]) if self.kind == "inject" else self.m.write(S, c["ids"])

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
                ha_ = getattr(ad.m.mem, "h_add", DM.H_ADD); hd_ = getattr(ad.m.mem, "h_del", DM.H_DEL); dd = Sf[:, ha_:ha_ + hd_].flatten(1); dp = Sprev[:, ha_:ha_ + hd_].flatten(1)
                chg["delta"].append(((dd - dp).norm(dim=1) / dd.norm(dim=1).clamp_min(1e-12)).cpu().numpy())
                F1, F0 = Sf[:, :ha_ + hd_].flatten(1), Sprev[:, :ha_ + hd_].flatten(1)   # §185: the 12 memory heads only (den's z block excluded)
                chg["full"].append(((F1 - F0).norm(dim=1) / F1.norm(dim=1).clamp_min(1e-12)).cpu().numpy())
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
    moved = [None] + [(ans[t] != ans[0]).any(1) if ans[t].ndim == 2 else ans[t] != ans[0] for t in range(1, T + 1)]   # note 173-3 MOVED2: the answer has left a(0)
    h = np.full(b, T); cap = np.ones(b, bool)
    for t in range(2, T + 1):
        hit = cap & same[t] & same[t - 1] & moved[t]; h[hit] = t; cap[hit] = False
    out["moved2"] = (h, cap)
    for k in ("read", "delta", "full"):
        h = np.full(b, T); cap = np.ones(b, bool)
        for t in range(1, T + 1):
            hit = cap & (chg[k][t] < EPS); h[hit] = t; cap[hit] = False
        out["%s_%g" % (k, EPS)] = (h, cap)
    return {r: (h, c_, cor[h, np.arange(b)]) for r, (h, c_) in out.items()}


def load_model(run, dev, init=False):
    """init (note 173-1): the arm's model at initialisation - the pretrained trunk and fresh memory parameters built under the run's seed (as the run's
    own torch.manual_seed before construction), no checkpoint loaded, gamma forced to 1 so the writes use the top-layer source."""
    a = json.load(open(os.path.join(run, "args.json")))
    if "n_max_train" in a:                                                  # nested (§170 / §175 / §177 f / §178): the run's own trunk + memory geometry
        from boxes.nested170 import make_trunk
        g = make_trunk(a, dev)
    else:
        g = s5.load_gpt2(dev)
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
        from boxes.nested170 import mem_geometry
        assert a.get("arm") == "inject"; ml_, ha_, hd_ = mem_geometry(a); kind, m = "nested", InjectN(g, ml_, "top", ha_, hd_, a.get("add_norm") or None); m.nopass = bool(a.get("aux_nopass"))   # §187
    else:
        from boxes.inject169 import Inject
        assert a.get("arm") == "inject"; kind, m = "inject", Inject(g, 6, a.get("mem_write", "top")); m.blind = bool(a.get("blind_decoder"))   # §179
        if a.get("pass_index"):                                             # §177 (a): the index table (rows from args; trained values from model.pt)
            m.enable_pass_index()
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
    K = rule_k(a); T = 2 * K; t0 = time.time()
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
    K = rule_k(a); T = 2 * K; acc = {k_: {t: [] for t in range(1, T + 1)} for k_ in ("read", "delta", "add")}
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
                cur = {"read": read_vec(m.mem, x, S).float(), "delta": S[:, getattr(m.mem, "h_add", DM.H_ADD):getattr(m.mem, "h_add", DM.H_ADD) + getattr(m.mem, "h_del", DM.H_DEL)].float().flatten(1), "add": S[:, :getattr(m.mem, "h_add", DM.H_ADD)].float().flatten(1)}
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
    K = rule_k(a)
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
            out.append(S[:, getattr(ad.m.mem, "h_add", DM.H_ADD):getattr(ad.m.mem, "h_add", DM.H_ADD) + getattr(ad.m.mem, "h_del", DM.H_DEL)].float().flatten(1).cpu().numpy())
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
                cur = {"read": read_vec(m.mem, xq, S).float(), "delta": S[:, getattr(m.mem, "h_add", DM.H_ADD):getattr(m.mem, "h_add", DM.H_ADD) + getattr(m.mem, "h_del", DM.H_DEL)].float().flatten(1), "add": S[:, :getattr(m.mem, "h_add", DM.H_ADD)].float().flatten(1)}
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


def cap_add_state(m, S, C):
    """note 185-1: S_add rescaled to norm C when its per-item norm exceeds C (direction kept; delta heads and any z block untouched)."""
    ha = m.mem.h_add; cur = S[:, :ha].float().flatten(1).norm(dim=1)
    f = torch.where(cur > C, C / cur.clamp_min(1e-12), torch.ones_like(cur)).to(S.dtype)
    return torch.cat([S[:, :ha] * f[:, None, None, None], S[:, ha:]], 1)


@torch.no_grad()
def deepaux_fields(m, a, dev, amp, bs=100, tmax=12, n_items=1000, cap_c=None, items_out=False, items_n=(9, 10), items_tmax=None, items_only=False):
    """note 187-1 (Theory / Audit, descriptive): whole-string aux exact match on deep items is confounded by copy length out of distribution, so
    (a) the aux HANDLE-token accuracy (teacher-forced argmax at the newly reduced value, as s175.aux_acc_by_role.handle) per pass t = 1..tmax on the
        n 9..12 test items (s175's ext set: rng 17023, 1,000 items), each item at t <= its reductions;
    (b) STEP-RECOVER on those items (s175's probe: multinomial logistic from the decoder read r_t to v_t, nested170.step_recover's form):
        transfer = fit on 4,000 training-distribution items (n <= 8, rng 17021, test-excluded; t <= 8) -> test on the n 9..12 items;
        within_ext = fit on a random half of the n 9..12 items (rng 18701), test on the other half, t = 1..tmax;
    (c) check: (a)'s handle accuracy on s175's n <= 8 set (rng 17022, t <= K) - to compare with the run's eval.json aux_acc_by_role.handle.
    The aux prompt is the model's own (no pass token under §187 --aux_nopass). cap_c (Theory 2026-10-11): the 185-1 cap after every write in (a) / (c)
    only (STEP-RECOVER stays uncapped: nested170.dec_reads)."""
    from contextlib import nullcontext
    from boxes.nested170 import test_set, train_items, aux_of, dec_reads, reductions, logreg_acc, P
    ctx = (lambda: torch.autocast("cuda", dtype=amp)) if amp is not None else nullcontext
    K = rule_k(a); TE = test_set(a.get("per_n", 1000)); EX = {tuple(it["syms"]) for it in TE if it["n"] >= int(a.get("excl_min_n", 1))}
    pe = [it for it in TE if 9 <= it["n"] <= 12]; te12 = [pe[i] for i in np.random.default_rng(17023).permutation(len(pe))[:n_items]]
    pool = [it for it in TE if it["n"] <= 8]; te8 = [pool[i] for i in np.random.default_rng(17022).permutation(len(pool))[:n_items]]

    def handle(items, T, rows=None):
        role = {}
        for i in range(0, len(items), bs):
            ch = items[i:i + bs]; x, msk = m.think_inputs(ch, dev)
            with ctx():
                S = m.mem.zero_state(len(ch), dev)
                for t in range(1, T + 1):
                    S = m.write(S, x, msk)
                    if cap_c is not None:
                        S = cap_add_state(m, S, cap_c)
                    idx = [j for j, it in enumerate(ch) if len(aux_of(it)) >= t]
                    if not idx:
                        continue
                    sub = [ch[j] for j in idx]; lg_a, y_a = m.aux_logits(S[idx], sub, t, dev); pr = lg_a.argmax(-1).cpu().numpy(); yy = y_a.cpu().numpy()
                    for r_, it in enumerate(sub):
                        sy, hi = aux_of(it)[t - 1]; L_ = len(sy); ok_ = pr[r_, :L_] == yy[r_, :L_]
                        rr = role.setdefault(t, {"handle": [], "copy": []}); rr["handle"].append(bool(ok_[hi])); rr["copy"].extend(bool(ok_[k_]) for k_ in range(L_) if k_ != hi)
                        if rows is not None:                                # note 187-2: per-item handle record
                            rows[i + idx[r_]]["handle_by_t"][str(t)] = int(bool(ok_[hi]))
        return {str(t): {"handle": float(np.mean(d["handle"])), "copy": (float(np.mean(d["copy"])) if d["copy"] else None), "n": len(d["handle"])} for t, d in sorted(role.items())}

    if items_only:                                                          # note 187-3: the per-item rows alone (no summary / probes)
        it_ = [it for it in TE if it["n"] in tuple(items_n)]
        rows = [{"n": it["n"], "nest_depth": it["depth"], "n_syms": len(it["syms"]), "think_tokens": int(m.think_inputs([it], "cpu")[1].sum()), "handle_by_t": {}} for it in it_]
        handle(it_, items_tmax or tmax, rows); return {"_items": rows}
    out = {"handle_ext": handle(te12, tmax)}
    if items_out:                                                           # note 187-2 (amended): ALL n = 9 and n = 10 test items (1,000 each), a separate pass
        it910 = [it for it in TE if it["n"] in tuple(items_n)]
        rows = [{"n": it["n"], "nest_depth": it["depth"], "n_syms": len(it["syms"]), "think_tokens": int(m.think_inputs([it], "cpu")[1].sum()), "handle_by_t": {}} for it in it910]
        handle(it910, items_tmax or tmax, rows); out["_items"] = rows
    print("deepaux handle_ext", {t: round(v["handle"], 3) for t, v in out["handle_ext"].items()}, flush=True)
    out["handle_le8_check"] = handle(te8, K)
    tr = train_items(np.random.default_rng(17021), int(os.environ.get("NESTED170_PROBE_N_FIT", 4000)), 8, EX)
    Rf = dec_reads(m, tr, min(8, tmax), dev, ctx, bs); Re = dec_reads(m, te12, tmax, dev, ctx, bs)
    vf = [reductions(it["root"]) for it in tr]; ve = [reductions(it["root"]) for it in te12]; tr_acc, wi_acc, info = {}, {}, {"transfer": {}, "within_ext": {}}
    for t in range(1, min(8, tmax) + 1):
        fi = [i for i, v in enumerate(vf) if len(v) >= t]; si = [i for i, v in enumerate(ve) if len(v) >= t]
        if len(fi) >= 100 and len(si) >= 20:
            tr_acc[str(t)] = logreg_acc(Rf[t - 1, fi], np.array([vf[i][t - 1] for i in fi]), Re[t - 1, si], np.array([ve[i][t - 1] for i in si]), dev)
            info["transfer"][str(t)] = {"fit": len(fi), "test": len(si)}
    half = np.random.default_rng(18701).permutation(len(te12)); A_, B_ = set(half[:len(te12) // 2].tolist()), set(half[len(te12) // 2:].tolist())
    for t in range(1, tmax + 1):
        fi = [i for i, v in enumerate(ve) if len(v) >= t and i in A_]; si = [i for i, v in enumerate(ve) if len(v) >= t and i in B_]
        if len(fi) >= 50 and len(si) >= 20:
            yf = np.array([ve[i][t - 1] for i in fi]); ys = np.array([ve[i][t - 1] for i in si])
            wi_acc[str(t)] = logreg_acc(Re[t - 1, fi], yf, Re[t - 1, si], ys, dev)
            info["within_ext"][str(t)] = {"fit": len(fi), "test": len(si), "majority": float((ys == np.bincount(yf, minlength=P).argmax()).mean())}
    out.update({"step_recover_ext_transfer": tr_acc, "step_recover_ext_within": wi_acc, "step_recover_ext_info": info, "tmax": tmax,
                "items": {"n_9_12": {"n": len(te12), "seed": 17023}, "n_le_8": {"n": len(te8), "seed": 17022}, "probe_fit_seed": 17021, "within_split_seed": 18701},
                "probe": "multinomial logistic on standardised r_t (the decoder read, 768-d; nested170.step_recover's representation), L2 1e-3, LBFGS 200",
                "nopass": bool(getattr(m, "nopass", False)), "cap_c": cap_c, "cap_scope": "handle_ext / handle_le8_check only" if cap_c is not None else None})
    return out


@torch.no_grad()
def selfdepth_fields(m, a, dev, amp, bs=100, ns=(9, 10), T=16):
    """note 187-4 (Theory, descriptive; pre-stated): self-verified depth on ALL test items at each n. Per item, T greedy passes from M_0 = 0:
    after pass t the aux greedy chain (as s175's aux eval: [the model's aux prompt] then greedy, exact match of the true step e_{t+1} + EOT for t <= n);
    halt t = the first t whose aux output is a bare number (one number token then EOT, AUX-DONE's rule; T if none); final value = that number;
    the answer decoder's prediction at K' = n + 2. KEEP iff final value == answer prediction AND halt t == n.
    -> per n: keep_rate, chain_precision = P(all n aux steps exact | kept), answer_precision = P(answer right | kept), and the same over all items."""
    from contextlib import nullcontext
    from boxes.nested170 import test_set, aux_of, ids, NUM_IDS, EOT as NEOT
    ctx = (lambda: torch.autocast("cuda", dtype=amp)) if amp is not None else nullcontext
    lut = {int(tk): v for v, tk in enumerate(NUM_IDS)}; TE = test_set(a.get("per_n", 1000)); out = {}
    for n in ns:
        its = [it for it in TE if it["n"] == n]; kp = n + 2; kp4 = n + 4; assert kp4 <= T
        H, FV, AP, AP4, CH = [], [], [], [], []
        for i in range(0, len(its), bs):
            ch = its[i:i + bs]; b = len(ch); x, msk = m.think_inputs(ch, dev)
            halt = np.full(b, T); fval = np.full(b, -1); chain_ok = np.ones(b, bool); ans = ans4 = None
            with ctx():
                S = m.mem.zero_state(b, dev)
                for t in range(1, T + 1):
                    S = m.write(S, x, msk)
                    if t == kp:
                        ans = m.dec_logits(S, ch, dev)[:, NUM_IDS].argmax(-1).cpu().numpy()
                    if t == kp4:                                    # Theory 2026-10-11: the strict filter's second answer read
                        ans4 = m.dec_logits(S, ch, dev)[:, NUM_IDS].argmax(-1).cpu().numpy()
                    tg = [ids(aux_of(it)[t - 1][0]) + [NEOT] if t <= len(aux_of(it)) else None for it in ch]
                    L = max([len(q) for q in tg if q is not None] + [2]); g = m.aux_greedy(S, t, L)
                    for r in range(b):
                        if tg[r] is not None and list(g[r, :len(tg[r])]) != tg[r]:
                            chain_ok[r] = False
                        if halt[r] == T and fval[r] == -1 and int(g[r, 0]) in lut and int(g[r, 1]) == NEOT:
                            halt[r] = t; fval[r] = lut[int(g[r, 0])]
            H.append(halt); FV.append(fval); AP.append(ans); AP4.append(ans4); CH.append(chain_ok)
        h, fv, ap, ap4, chn = np.concatenate(H), np.concatenate(FV), np.concatenate(AP), np.concatenate(AP4), np.concatenate(CH); tgt = np.array([it["t"] for it in its])
        keep = (fv == ap) & (h == n) & (fv >= 0); strict = keep & (ap == ap4)
        c_i, c_ii, c_iii = (fv == ap) & (fv >= 0), h == n, ap == ap4                # Theory 2026-10-11: each filter component separately

        def stats(msk):
            return {"keep_rate": float(msk.mean()), "n_kept": int(msk.sum()), "chain_precision": float(chn[msk].mean()) if msk.any() else None,
                    "answer_precision": float((ap == tgt)[msk].mean()) if msk.any() else None}
        comps = {"i_answer_agree": stats(c_i), "ii_halt_at_n": stats(c_ii), "iii_kprime_agree": stats(c_iii), "i_and_iii": stats(c_i & c_iii),
                 "base_i_and_ii": stats(c_i & c_ii), "strict_i_ii_iii": stats(c_i & c_ii & c_iii)}
        out[str(n)] = {"n_items": int(len(its)), "kprime_answer": kp, "keep_rate": float(keep.mean()), "n_kept": int(keep.sum()),
                       "chain_precision": float(chn[keep].mean()) if keep.any() else None, "answer_precision": float((ap == tgt)[keep].mean()) if keep.any() else None,
                       "all_items": {"chain_exact": float(chn.mean()), "answer_acc": float((ap == tgt).mean()), "halt_at_n": float((h == n).mean()),
                                     "final_eq_answer": float(((fv == ap) & (fv >= 0)).mean()), "no_halt": float((fv < 0).mean()),
                                     "answer_acc_kprime_n4": float((ap4 == tgt).mean()), "answer_n2_eq_n4": float((ap == ap4).mean())},
                       "strict": {"def": "keep AND answer at K' n+2 == answer at K' n+4", "keep_rate": float(strict.mean()), "n_kept": int(strict.sum()),
                                  "chain_precision": float(chn[strict].mean()) if strict.any() else None, "answer_precision": float((ap == tgt)[strict].mean()) if strict.any() else None},
                       "halt_hist": {str(k_): int(c_) for k_, c_ in zip(*np.unique(h, return_counts=True))}, "components": comps,
                       "components_def": "i: aux bare number == answer at K' n+2; ii: halt t == n; iii: answer n+2 == answer n+4"}
        print("selfdepth n", n, {k: out[str(n)][k] for k in ("keep_rate", "chain_precision", "answer_precision")}, flush=True)
    return out


@torch.no_grad()
def posoffset_fields(m, a, dev, amp, bs=125, offs=(0, 3, 6, 9), ns=(3, 4, 5, 6, 7, 8), keep0=False):
    """note 187-5 (Theory / Audit, descriptive): every forward path (think passes, main decoder, aux decoder) adds wpe[off : off + T] instead of wpe[:T]
    (deltamem.POS_OFFSET; gate187_posoffset.json). Per off and n (ALL test items at n): the answer accuracy at K' = n + 2, and the aux handle accuracy
    by pass t <= n (teacher-forced, as s175.aux_acc_by_role.handle). keep0 (187-5a KEEP-SINK): position 0 keeps wpe[0] on the think / main-decoder
    paths, and the aux decoder (no question in its prompt) is read at offset 0."""
    from contextlib import nullcontext
    from boxes.nested170 import test_set, aux_of, NUM_IDS
    ctx = (lambda: torch.autocast("cuda", dtype=amp)) if amp is not None else nullcontext
    TE = test_set(a.get("per_n", 1000)); out = {}
    try:
        for off in offs:
            DM.POS_OFFSET = int(off); DM.POS_KEEP0 = bool(keep0); res = {}
            for n in ns:
                its = [it for it in TE if it["n"] == n]; kp = n + 2; corr = []; hd = {t: [] for t in range(1, n + 1)}
                for i in range(0, len(its), bs):
                    ch = its[i:i + bs]; x, msk = m.think_inputs(ch, dev); tg = np.array([it["t"] for it in ch])
                    with ctx():
                        S = m.mem.zero_state(len(ch), dev)
                        for t in range(1, kp + 1):
                            S = m.write(S, x, msk)
                            if t <= n:
                                if keep0:                                       # 187-5a: the aux prompt has no question -> wpe[:T]
                                    DM.POS_OFFSET = 0
                                lg_a, y_a = m.aux_logits(S, ch, t, dev); pr = lg_a.argmax(-1).cpu().numpy(); yy = y_a.cpu().numpy()
                                DM.POS_OFFSET = int(off)
                                for r_, it in enumerate(ch):
                                    sy, hi = aux_of(it)[t - 1]; hd[t].append(bool(pr[r_, hi] == yy[r_, hi]))
                        corr.append(m.dec_logits(S, ch, dev)[:, NUM_IDS].argmax(-1).cpu().numpy() == tg)
                res[str(n)] = {"answer_acc_kprime_n2": float(np.concatenate(corr).mean()), "handle_by_t": {str(t): float(np.mean(v)) for t, v in hd.items()}, "n_items": len(its)}
            res["pooled_answer_n3_8"] = float(np.mean([res[str(n)]["answer_acc_kprime_n2"] for n in ns])); out[str(off)] = res
            print("posoffset", off, round(res["pooled_answer_n3_8"], 4), flush=True)
    finally:
        DM.POS_OFFSET = 0; DM.POS_KEEP0 = False
    return out


@torch.no_grad()
def sizeprobe_fields(m, a, dev, amp, bs=125, ts=(2, 4, 6), cap=300):
    """note 187-7 SIZE-PROBE (Theory; Literature / Audit): does the decoder's memory read encode the problem SIZE n? Items: test n = 3..12 with nesting
    depth <= 2, m = min over n of the count (cap 300), the first m per n; half fit / half test per n. After t passes from M_0 = 0 (t in ts), the read at the
    main decoder's read position (its " =" token; ReadTap + read_vec as trace()): additive part r_add = g_add read_add (h_add heads), delta part.
    Probes (nested170.logreg_acc, n_cls 10, labels n - 3): full = r_add; norm = the per-head norms ||r_add,h||; dir = per-head unit vectors;
    full_adddel = r_add || r_del."""
    from boxes.nested170 import test_set, ids, pad, TID, EOT as NEOT, logreg_acc
    TE = test_set(a.get("per_n", 1000)); ns = list(range(3, 13)); pool = {n: [it for it in TE if it["n"] == n and it["depth"] <= 2] for n in ns}
    mm = min(cap, min(len(v) for v in pool.values())); fit, tst = [], []
    for n in ns:
        fit += pool[n][:mm // 2]; tst += pool[n][mm // 2:mm]
    ha, dh = m.mem.h_add, DM.DH; out = {}

    def feats(items, t):
        F, Pn = [], []
        with ReadTap() as tap, amp_ctx(amp):
            for i in range(0, len(items), bs):
                ch = items[i:i + bs]; x, msk = m.think_inputs(ch, dev); S = m.mem.zero_state(len(ch), dev)
                for _ in range(t):
                    S = m.write(S, x, msk)
                seqs = [[NEOT] + ids(it["syms"]) + [TID[" ="]] for it in ch]; rpos = torch.tensor([len(q) - 1 for q in seqs], device=dev)
                tap.arm(); m.dec_logits(S, ch, dev); assert tap.x is not None, "SIZE-PROBE: the read tap did not fire"
                xr = tap.x[torch.arange(len(ch), device=dev), rpos]; F.append(read_vec(m.mem, xr, S).float().cpu().numpy())
                with torch.autocast("cuda", enabled=False):                 # 187-7a norm_pre: phi(q)^T S_add before any RMS scaling / DEN division
                    D_ = xr.shape[-1]; q_ = m.mem.qkv(m.mem.ln(xr.float()))[..., :D_].view(len(ch), ha + m.mem.h_del, dh)
                    pre = torch.einsum("bhk,bhkv->bhv", Fn.elu(q_[:, :ha]) + 1, S[:, :ha].float()); Pn.append(pre.norm(dim=-1).cpu().numpy())
        return np.concatenate(F), np.concatenate(Pn)
    yf = np.array([it["n"] - 3 for it in fit]); ys = np.array([it["n"] - 3 for it in tst])
    for t in ts:
        (Rf, Pf), (Rs, Ps) = feats(fit, t), feats(tst, t); r = {}
        for nm, fn in (("full", lambda R: R[:, :ha * dh]),
                       ("norm", lambda R: np.linalg.norm(R[:, :ha * dh].reshape(len(R), ha, dh), axis=2)),
                       ("dir", lambda R: (R[:, :ha * dh].reshape(len(R), ha, dh) / (np.linalg.norm(R[:, :ha * dh].reshape(len(R), ha, dh), axis=2, keepdims=True) + 1e-12)).reshape(len(R), -1)),
                       ("full_adddel", lambda R: R)):
            r[nm] = logreg_acc(fn(Rf), yf, fn(Rs), ys, dev, n_cls=10)
        r["norm_pre"] = logreg_acc(Pf, yf, Ps, ys, dev, n_cls=10)
        nr = np.linalg.norm(Rs[:, :ha * dh].reshape(len(Rs), ha, dh), axis=2)
        if a.get("add_norm") == "rms":                                      # 187-7a: the post-RMS read's per-head norm is constant across items (wrong-tensor tripwire)
            sp_ = (nr.max(0) - nr.min(0)) / (nr.mean(0) + 1e-12); assert float(sp_.max()) < 1e-3, "187-7a: RMS read norm not constant (%s)" % sp_
            r["rms_norm_rel_spread_max"] = float(sp_.max())
        r.update({"n_fit": int(len(fit)), "n_test": int(len(tst)), "norm_cv_mean": float((nr.std(0) / (nr.mean(0) + 1e-12)).mean())})
        out[str(t)] = r; print("sizeprobe t", t, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    return out, mm


@torch.no_grad()
def readclamp_fields(m, a, dev, amp, bs=125, n_ref=500, kmax=12, ns_in=(3, 4, 5, 6, 7, 8), ns_deep=(9, 10, 11, 12)):
    """note 187-8 READ-CLAMP (Theory / Audit): (refs) 500 n <= 8 training items (rng 18708): per additive head, the per-token norm of the post-gain read on
    S_k, k = 0..12, per read path think (the pass k+1 read) / dec (dec_logits) / aux (aux_logits at t = k, items with k <= n; kmax 8 for aux) -> p99.
    Two arms in one process: unclamped (deltamem.READ_CLAMP None) and clamped (any token's head read above its (path, head, k) p99 scaled to it).
    (1) per-item rows on ALL n 9..12 test items: n, nest_depth, handle_by_t (t <= n; as deepaux), clamped arm + clamped_t3_6 (any read of the item clamped in
    passes 3..6 or the aux at t 3..6); (2) indist: handle_mean (t <= n, n 3..8 test items) and k8_pooled (pooled n 3..8 answer at K' = 8);
    (3) frac_clamped by n / t / path (share of items with any clamped read) and depth_n2 (n 9..12 answer at K' = n + 2)."""
    from contextlib import nullcontext
    from boxes.nested170 import test_set, train_items, aux_of, NUM_IDS
    ctx = (lambda: torch.autocast("cuda", dtype=amp)) if amp is not None else nullcontext
    TE = test_set(a.get("per_n", 1000)); EX = {tuple(it["syms"]) for it in TE if it["n"] >= int(a.get("excl_min_n", 1))}

    def rc(path, k):
        if DM.READ_CLAMP is not None:
            DM.READ_CLAMP["path"], DM.READ_CLAMP["k"] = path, int(k)
    try:
        DM.READ_CLAMP = {"mode": "collect", "store": {}, "path": None, "k": 0}; tr = train_items(np.random.default_rng(18708), n_ref, 8, EX)
        for i in range(0, len(tr), bs):
            ch = tr[i:i + bs]; x, msk = m.think_inputs(ch, dev)
            with ctx():
                S = m.mem.zero_state(len(ch), dev)
                for k in range(0, kmax + 1):
                    rc("dec", k); m.dec_logits(S, ch, dev)
                    idx = [j for j, it in enumerate(ch) if 1 <= k <= len(aux_of(it))]
                    if idx:
                        rc("aux", k); m.aux_logits(S[idx], [ch[j] for j in idx], k, dev)
                    rc("think", k); S = m.write(S, x, msk)
        st = DM.READ_CLAMP["store"]; ref = {key: torch.quantile(torch.cat(v).float(), 0.99, dim=0) for key, v in st.items()}
        ref_mean = {key: torch.cat(v).float().mean(0) for key, v in st.items()}
        kmx = {p_: max(k_ for (pp, k_) in ref if pp == p_) for p_ in ("think", "dec", "aux")}

        def arm(mode):
            clamp = mode == "clamp"
            DM.READ_CLAMP = {"mode": mode, "ref": ref, "ref_mean": ref_mean, "kmax": kmx, "hits": [], "path": None, "k": 0} if mode else None
            res = {"items": [], "fc": {}, "fr": {}}
            fr_acc = {}

            def call(path, k, fn, rows):                                    # one read-path call; returns the per-row clamp hits of that call
                if DM.READ_CLAMP is None or not clamp:
                    rc(path, k); return fn(), None
                DM.READ_CLAMP["hits"] = []; rc(path, k); r_ = fn(); h_ = torch.zeros(rows, dtype=torch.bool); na, nt = 0, 0
                for (_, _, hit, nab, per) in DM.READ_CLAMP["hits"]:
                    h_ |= hit[:rows]; na += int(nab[:rows].sum()); nt += per * rows
                fr_acc.setdefault((path, k), [0, 0]); fr_acc[(path, k)][0] += na; fr_acc[(path, k)][1] += nt
                return r_, h_.numpy()

            def run(items, n, K, want_items):
                hand, ans, cl36, fcn = {t: [] for t in range(1, n + 1)}, [], [], {}; fr_acc.clear()
                for i in range(0, len(items), bs):
                    ch = items[i:i + bs]; b = len(ch); x, msk = m.think_inputs(ch, dev); c36 = np.zeros(b, bool)
                    with ctx():
                        S = m.mem.zero_state(b, dev)
                        for t in range(1, K + 1):
                            S, h_ = call("think", t - 1, lambda: m.write(S, x, msk), b)
                            if h_ is not None:
                                fcn.setdefault("t%d_think" % t, []).append(h_); c36 |= h_ if 3 <= t <= 6 else False
                            if t <= n:
                                (lg_a, y_a), h_ = call("aux", t, lambda: m.aux_logits(S, ch, t, dev), b)
                                pr = lg_a.argmax(-1).cpu().numpy(); yy = y_a.cpu().numpy()
                                for r_, it in enumerate(ch):
                                    sy, hi = aux_of(it)[t - 1]; hand[t].append(bool(pr[r_, hi] == yy[r_, hi]))
                                if h_ is not None:
                                    fcn.setdefault("t%d_aux" % t, []).append(h_); c36 |= h_ if 3 <= t <= 6 else False
                        lg, h_ = call("dec", K, lambda: m.dec_logits(S, ch, dev), b)
                        if h_ is not None:
                            fcn.setdefault("K%d_dec" % K, []).append(h_)
                        ans.append(lg[:, NUM_IDS].argmax(-1).cpu().numpy() == np.array([it["t"] for it in ch])); cl36.append(c36)
                acc = float(np.concatenate(ans).mean()); fc = {k_: float(np.concatenate(v).mean()) for k_, v in fcn.items()}
                if clamp:
                    res["fr"][str(n)] = {"%s_k%d" % k_: (v[0] / v[1] if v[1] else None) for k_, v in fr_acc.items()}
                if want_items:
                    H = {t: np.array(v) for t, v in hand.items()}; C = np.concatenate(cl36)
                    for j, it in enumerate(items):
                        row = {"n": it["n"], "nest_depth": it["depth"], "handle_by_t": {str(t): int(H[t][j]) for t in range(1, n + 1)}}
                        if clamp:
                            row["clamped_t3_6"] = bool(C[j])
                        res["items"].append(row)
                return acc, {t: float(np.mean(v)) for t, v in hand.items()}, fc
            hm, k8 = [], []
            for n in ns_in:
                its = [it for it in TE if it["n"] == n]; acc8, hd, fc = run(its, n, 8, False); hm += [hd[t] for t in hd]; k8.append(acc8)
                if clamp:
                    res["fc"][str(n)] = fc
            res["indist"] = {"handle_mean": float(np.mean(hm)), "k8_pooled": float(np.mean(k8)), "def": "handle_mean = mean over n 3..8 and t <= n of the per-t handle accuracy; k8_pooled = mean over n 3..8 of the answer accuracy at K' = 8"}
            dn = {}
            for n in ns_deep:
                its = [it for it in TE if it["n"] == n]; accn, _, fc = run(its, n, n + 2, True); dn[str(n)] = accn
                if clamp:
                    res["fc"][str(n)] = fc
            res["depth_n2"] = {"by_n": dn, "mean": float(np.mean(list(dn.values())))}
            print("readclamp", mode or "unclamped", res["indist"]["handle_mean"], res["indist"]["k8_pooled"], res["depth_n2"]["mean"], flush=True)
            return res
        un = arm(None); cl = arm("clamp"); rs = arm("rescale")
        for d_ in (un, rs):
            d_.pop("fc", None); d_.pop("fr", None)
        cl["frac_clamped"] = cl.pop("fc"); cl["frac_reads_above_p99"] = cl.pop("fr")
    finally:
        DM.READ_CLAMP = None
    refs = {"%s_k%d" % k_: [float(v) for v in r_] for k_, r_ in ref.items()}
    return {"unclamped": un, "clamped": cl, "rescaled": rs, "ref": refs, "ref_mean": {"%s_k%d" % k_: [float(v) for v in r_] for k_, r_ in ref_mean.items()},
            "frac_def": "frac_clamped[n][t<t>_<path> | K<K>_dec] = share of items with any clamped read in that call; frac_reads_above_p99[n][<path>_k<k>] = share of (token, head) reads above the p99 ref", "ref_def": "p99 over tokens (padding included) of each additive head's post-gain read norm on S_k, per path; 500 n<=8 train items rng 18708; aux refs k 1..8 (k > 8 uses k 8)"}


@torch.no_grad()
def auxdone_fields(m, a, dev, amp, n_items=1000, bs=100, cap=16, ns=(2, 4, 6, 8, 10, 12), cap_c=None):
    """note 182-2 AUX-DONE (Theory, descriptive; nested runs with §175's aux decoder): per n, n_items test items, think passes t = 1..cap from M_0 = 0.
    After each pass: the main decoder's answer, and §175's aux decoder greedy from [EOT, " Step", " t", " :"] reading M_t. The item halts at the first t
    whose aux output is ONE number token (a mod-11 value) followed by EOT (2 greedy tokens decide it); cap 16 (no halt -> 16, capped).
    cap_c (Theory / Audit 2026-10-10): the 185-1 NORM-CAP applied after every write (S_add rescaled to norm cap_c), else uncapped."""
    from contextlib import nullcontext
    from boxes.nested170 import test_set, NUM_IDS, EOT as NEOT
    ctx = (lambda: torch.autocast("cuda", dtype=amp)) if amp is not None else nullcontext
    lut = {int(tk): v for v, tk in enumerate(NUM_IDS)}; TE = test_set(a.get("per_n", 1000)); out = {}
    for n in ns:
        its = [it for it in TE if it["n"] == n][:n_items]; tgt = np.array([it["t"] for it in its]); H, MA, AX, MAIN = [], [], [], []
        for i in range(0, len(its), bs):
            ch = its[i:i + bs]; x, msk = m.think_inputs(ch, dev); main_t, num_t, done_t = [], [], []
            with ctx():
                S = m.mem.zero_state(len(ch), dev)
                for t in range(1, cap + 1):
                    S = m.write(S, x, msk)
                    if cap_c is not None:
                        S = cap_add_state(m, S, cap_c)
                    lg = m.dec_logits(S, ch, dev); main_t.append(lg[:, NUM_IDS].argmax(-1).cpu().numpy())
                    g2 = m.aux_greedy(S, t, 2); num_t.append(np.array([lut.get(int(r[0]), -1) for r in g2])); done_t.append(np.array([(int(r[0]) in lut) and int(r[1]) == NEOT for r in g2]))
            main_t, num_t, done_t = np.stack(main_t), np.stack(num_t), np.stack(done_t)            # (cap, b)
            hit = done_t.any(0); h = np.where(hit, done_t.argmax(0) + 1, cap); col = np.arange(len(ch))
            H.append(h); MA.append(main_t[h - 1, col]); AX.append(np.where(hit, num_t[h - 1, col], -1)); MAIN.append(hit)
        h, ma, ax, hit = np.concatenate(H), np.concatenate(MA), np.concatenate(AX), np.concatenate(MAIN)
        out[str(n)] = {"n_items": int(len(h)), "acc_at_halt": float((ma == tgt).mean()), "mean_halt": float(h.mean()), "median_halt": float(np.median(h)),
                       "mean_halt_minus_n": float((h - n).mean()), "frac_capped": float((~hit).mean()),
                       "aux_num_acc": float((ax[hit] == tgt[hit]).mean()) if hit.any() else None, "agree": float((ax[hit] == ma[hit]).mean()) if hit.any() else None,
                       "aux_done_by_t": [round(float((hit & (h <= t)).mean()), 4) for t in range(1, cap + 1)]}
        d_ = h - n; bk = {"early": d_ < 0, "at_n": d_ == 0, "n+1": d_ == 1, "n+2": d_ == 2, "later": d_ > 2}   # Theory / Audit: halt - n buckets (capped items included)
        out[str(n)]["halt_minus_n"] = {"share": {k_: float(v_.mean()) for k_, v_ in bk.items()},
                                       "counts": {str(k_): int(c_) for k_, c_ in zip(*np.unique(d_, return_counts=True))}}
        out[str(n)]["acc_at_halt_by_bucket"] = {k_: (float((ma[v_] == tgt[v_]).mean()) if v_.any() else None) for k_, v_ in
                                                {"early": d_ < 0, "at_n": d_ == 0, "n+1": d_ == 1, "later": d_ > 1}.items()}
        print("auxdone n", n, out[str(n)]["acc_at_halt"], out[str(n)]["mean_halt"], flush=True)
    return out


@torch.no_grad()
def normcap_fields(m, a, dev, amp, bs=100, run=None, kprimes=None, c_override=None):
    """note 185-1 (Theory, descriptive): inference-only NORM-CAP for a nested run. C = the median ||S_add|| after K passes over 1,000 TRAINING items
    (nested170.train_items, rng 18501, test-excluded as the run: test_set + the SERIAL pool); then acc_by_n at each K' of the run's --kprimes set
    (default {0, 1, K/2, K, 2K}) with S_add rescaled to norm C after every write when it exceeds C (direction kept; delta untouched). The uncapped
    acc_by_n_kprime is copied from the run's eval.json. c_override (--normcap_c): use this C instead (inf = the UNCAPPED arm at any K' set)."""
    from contextlib import nullcontext
    from boxes.nested170 import test_set, train_items, generate, render, NUM_IDS
    ctx = (lambda: torch.autocast("cuda", dtype=amp)) if amp is not None else nullcontext
    K = rule_k(a); ha = m.mem.h_add; TE = test_set(a.get("per_n", 1000)); EX = {tuple(it["syms"]) for it in TE if it["n"] >= int(a.get("excl_min_n", 1))}
    r_sp = np.random.default_rng(17300)
    for _ in range(int(a.get("serial_pool", 2200))):
        r__, _t = generate(r_sp, 8); EX.add(tuple(render(r__)))
    tr = train_items(np.random.default_rng(18501), 1000, int(a.get("n_max_train", 8)), EX) if c_override is None else []; nrm = []
    for i in range(0, len(tr), bs):
        ch = tr[i:i + bs]; x, msk = m.think_inputs(ch, dev)
        with ctx():
            S = m.mem.zero_state(len(ch), dev)
            for _ in range(K):
                S = m.write(S, x, msk)
        nrm.append(S[:, :ha].float().flatten(1).norm(dim=1).cpu().numpy())
    C = float(np.median(np.concatenate(nrm))) if c_override is None else float(c_override)
    kps = sorted({int(v) for v in str(kprimes or a.get("kprimes") or "").split(",") if v.strip()} | {K}) if (kprimes or a.get("kprimes")) else sorted({0, 1, K // 2, K, 2 * K})   # --kprimes overrides the run's set
    T_ = np.array([it["t"] for it in TE]); N_ = np.array([it["n"] for it in TE]); capped = {}
    for kp in kps:
        out = []
        for i in range(0, len(TE), bs):
            ch = TE[i:i + bs]; x, msk = m.think_inputs(ch, dev)
            with ctx():
                S = m.mem.zero_state(len(ch), dev)
                for _ in range(kp):
                    S = cap_add_state(m, m.write(S, x, msk), C)
                out.append(m.dec_logits(S, ch, dev)[:, NUM_IDS].argmax(-1).cpu().numpy())
        A_ = np.concatenate(out); capped[str(kp)] = {str(n): float((A_ == T_)[N_ == n].mean()) for n in range(1, 17)}
        print("normcap K'", kp, round(float((A_ == T_)[(N_ >= 3) & (N_ <= 8)].mean()), 4), flush=True)
    unc = (json.load(open(os.path.join(run, "eval.json"))).get("s170") or {}).get("acc_by_n_kprime") if run and os.path.exists(os.path.join(run, "eval.json")) else None
    return {"C": C, "C_source": ("median ||S_add|| after K = %d passes over 1,000 train items (nested170.train_items rng 18501, test + SERIAL pool excluded)" % K) if c_override is None else "--normcap_c %s (given)" % c_override,
            "acc_by_n_kprime": capped, "acc_by_n_kprime_uncapped": unc, "kprimes": kps,
            "def": "inference-only: after every write, S_add rescaled to norm C when ||S_add|| > C (per item, direction kept; delta untouched) - note 185-1"}


@torch.no_grad()
def tshift_fields(m, a, dev, amp, n_items=1000, bs=100, ns=(4, 6, 8)):
    """note 182-2 follow-up T-SHIFT (Theory, descriptive): is §175's aux "done" a pass COUNTER (the prompt's pass token) or does the memory carry
    COMPLETION? Per n, M_t after exactly t real passes for t in {n-1, n, n+1}; at each, the aux greedy (2 tokens) with the prompt's pass token set to
    t-1, t, t+1 (M_t fixed) -> the share of items whose output is "done" (one number token then EOT), and the share whose number is the answer."""
    from contextlib import nullcontext
    from boxes.nested170 import test_set, NUM_IDS, EOT as NEOT
    ctx = (lambda: torch.autocast("cuda", dtype=amp)) if amp is not None else nullcontext
    lut = {int(tk): v for v, tk in enumerate(NUM_IDS)}; TE = test_set(a.get("per_n", 1000)); out = {}
    for n in ns:
        its = [it for it in TE if it["n"] == n][:n_items]; cells = {}
        for i in range(0, len(its), bs):
            ch = its[i:i + bs]; tg = np.array([it["t"] for it in ch]); x, msk = m.think_inputs(ch, dev)
            with ctx():
                S = m.mem.zero_state(len(ch), dev)
                for t in range(1, n + 2):
                    S = m.write(S, x, msk)
                    if t < n - 1:
                        continue
                    for p in (t - 1, t, t + 1):
                        g2 = m.aux_greedy(S, p, 2); num = np.array([lut.get(int(r[0]), -1) for r in g2]); done = np.array([(int(r[0]) in lut) and int(r[1]) == NEOT for r in g2])
                        c_ = cells.setdefault("real%d_prompt%d" % (t, p), {"done": [], "num_ok": []}); c_["done"].append(done); c_["num_ok"].append(done & (num == tg))
        out[str(n)] = {k_: {"real_t": int(k_.split("_")[0][4:]), "prompt_t": int(k_.split("_")[1][6:]), "done": float(np.concatenate(v_["done"]).mean()),
                            "done_and_correct": float(np.concatenate(v_["num_ok"]).mean())} for k_, v_ in sorted(cells.items())}
        out[str(n)]["n_items"] = len(its); print("tshift n", n, {k_: round(v_["done"], 3) for k_, v_ in out[str(n)].items() if k_ != "n_items"}, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--bs", type=int, default=250)
    ap.add_argument("--fp32", action="store_true"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", default=None)
    ap.add_argument("--bench", default=os.path.join("bench", "data", "mult_4x4_test.txt")); ap.add_argument("--dir177", action="store_true", help="§177 direction diagnostic only")
    ap.add_argument("--auxdone", action="store_true", help="note 182-2 AUX-DONE halting (nested + §175 aux) only -> eval_s182_auxdone.json")
    ap.add_argument("--tshift", action="store_true", help="note 182-2 follow-up T-SHIFT (aux done vs real / prompt pass) only -> eval_s182_tshift.json")
    ap.add_argument("--deepaux", action="store_true", help="note 187-1: aux handle accuracy + STEP-RECOVER on n 9..12 (nested + §175 aux) only -> eval_s187_deepaux.json")
    ap.add_argument("--deepaux_items", action="store_true", help="--deepaux: also per-item rows (n, nesting depth, token length, handle 0/1 per t) -> eval_s187_deepaux_items.json (note 187-2)")
    ap.add_argument("--deepaux_items_only", action="store_true", help="note 187-3: only the per-item rows (no summary) -> --items_file")
    ap.add_argument("--items_n", default="9,10", help="--deepaux_items(_only): the n set (all test items at each n)")
    ap.add_argument("--items_tmax", type=int, default=0, help="--deepaux_items(_only): passes recorded (0 = 12)")
    ap.add_argument("--items_file", default="eval_s187_deepaux_items.json", help="--deepaux_items(_only): the rows file name in the run dir")
    ap.add_argument("--selfdepth", action="store_true", help="note 187-4: self-verified depth filter (aux chain + AUX-DONE halt + answer at K' = n + 2) on ALL n 9 / 10 test items -> eval_s187_selfdepth_pre.json")
    ap.add_argument("--posoffset_keep0", action="store_true", help="note 187-5a KEEP-SINK: --posoffset with position 0 kept at wpe[0] -> eval_s187_posoffset_keep0.json")
    ap.add_argument("--posoffset", action="store_true", help="note 187-5: answer (K' n+2) + aux handle by t on n 3..8 with wpe offset 0 / 3 / 6 / 9 on every forward -> eval_s187_posoffset.json")
    ap.add_argument("--sizeprobe", action="store_true", help="note 187-7 SIZE-PROBE: probes for n from the decoder read after t = 2 / 4 / 6 passes -> eval_s187_sizeprobe.json")
    ap.add_argument("--sizeprobe_cap", type=int, default=300, help="--sizeprobe: items per n cap (smokes)")
    ap.add_argument("--readclamp", action="store_true", help="note 187-8 READ-CLAMP (unclamped vs p99-clamped additive reads; per-item rows n 9..12) -> eval_s187_readclamp.json")
    ap.add_argument("--normcap", action="store_true", help="note 185-1 inference-only NORM-CAP (nested) only -> eval_s185_normcap.json")
    ap.add_argument("--normcap_c", default="", help="--normcap: C given instead of computed: a number ('inf' = the uncapped arm) or an eval_s185_normcap.json path (its normcap.C)")
    ap.add_argument("--normcap_factor", type=float, default=1.0, help="--normcap with --normcap_c: C x this factor (the cap-level sweep, Theory 2026-10-11)")
    ap.add_argument("--kprimes", default="", help="--normcap: the K' set (default: the run's --kprimes, else {0, 1, K/2, K, 2K})")
    ap.add_argument("--cap_c", default="", help="--auxdone under the 185-1 cap: C as a number or an eval_s185_normcap.json path (its normcap.C) -> eval_s182_auxdone_capped.json")
    o = ap.parse_args(); dev = o.device; t0 = time.time(); o.out = o.out if (o.out and o.out.endswith(".json")) else None   # gen_vast_job appends --out <dir>: anything but a .json path -> the default file in --run
    amp = None if (o.fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    a_ = json.load(open(os.path.join(o.run, "args.json")))
    if o.dir177 and a_.get("family") == "mult_pair":                        # §177 baseline on the working pair-segment models (Theory)
        r_ = dir_pair_fields(o.run, dev, amp, min(o.n, 500), min(o.bs, 125), o.bench)
        r_.update({"run": o.run.replace(os.sep, "/"), "kind": "multpair", "eval_dtype": str(amp), "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
                   "sec": round(time.time() - t0)})
        json.dump(r_, open(o.out or os.path.join(o.run, "eval_s177_dir.json"), "w", newline="\n"), indent=1); print("DONE dir177 pair", o.run, flush=True); return
    kind, m, a = load_model(o.run, dev); K = rule_k(a); T = 2 * K; assert K >= 1
    if o.normcap:                                                           # note 185-1 (Theory): inference-only NORM-CAP -> eval_s185_normcap.json
        assert kind == "nested", "NORM-CAP (185-1) is for nested runs"
        r_ = {"normcap": normcap_fields(m, a, dev, amp, o.bs, o.run, o.kprimes or None, ((float(json.load(open(o.normcap_c))["normcap"]["C"]) if o.normcap_c.endswith(".json") else float(o.normcap_c)) * o.normcap_factor) if o.normcap_c else None),
              "normcap_factor": o.normcap_factor if o.normcap_c else None, "normcap_c_source": o.normcap_c.replace(os.sep, "/") or None, "run": o.run.replace(os.sep, "/"), "kind": kind, "eval_dtype": str(amp),
              "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "sec": round(time.time() - t0)}
        json.dump(r_, open(o.out or os.path.join(o.run, "eval_s185_normcap.json"), "w", newline="\n"), indent=1); print("DONE normcap", o.run, flush=True); return
    if o.readclamp:                                                         # note 187-8 (Theory): READ-CLAMP -> eval_s187_readclamp.json
        assert kind == "nested" and a.get("aux_step"), "READ-CLAMP needs a nested run with §175's aux decoder"
        r_ = readclamp_fields(m, a, dev, amp, o.bs); r_.update({"run": o.run.replace(os.sep, "/"), "kind": kind, "eval_dtype": str(amp),
                                                       "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "sec": round(time.time() - t0)})
        json.dump(r_, open(o.out or os.path.join(o.run, "eval_s187_readclamp.json"), "w", newline="\n"), indent=1); print("DONE readclamp", o.run, flush=True); return
    if o.sizeprobe:                                                         # note 187-7 (Theory): SIZE-PROBE -> eval_s187_sizeprobe.json
        assert kind == "nested", "SIZE-PROBE is for nested runs"
        bt_, mm_ = sizeprobe_fields(m, a, dev, amp, o.bs, cap=o.sizeprobe_cap)
        r_ = {"by_t": bt_, "chance": 0.1, "per_n": mm_, "run": o.run.replace(os.sep, "/"), "kind": kind, "eval_dtype": str(amp),
              "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "sec": round(time.time() - t0),
              "def": "test n 3..12, nesting <= 2, first per_n per n, half fit / half test per n; read at the main decoder's ' =' after t passes from 0; logreg n_cls 10 (labels n-3) on full = g_add read_add (h_add heads), norm = per-head norms, dir = per-head unit vectors, full_adddel = add || delta; norm_cv_mean = mean over heads of the test-set coefficient of variation of the per-head norm"}
        json.dump(r_, open(o.out or os.path.join(o.run, "eval_s187_sizeprobe.json"), "w", newline="\n"), indent=1); print("DONE sizeprobe", o.run, flush=True); return
    if o.posoffset or o.posoffset_keep0:                                    # note 187-5 / 5a (Theory): position-offset eval -> eval_s187_posoffset(_keep0).json
        assert kind == "nested" and a.get("aux_step"), "posoffset needs a nested run trained with §175's aux decoder"
        r_ = {"by_offset": posoffset_fields(m, a, dev, amp, o.bs, keep0=o.posoffset_keep0), "keep0": bool(o.posoffset_keep0), "run": o.run.replace(os.sep, "/"), "kind": kind, "eval_dtype": str(amp),
              "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "sec": round(time.time() - t0),
              "def": "deltamem.POS_OFFSET = off: think passes, main decoder and aux decoder all add wpe[off : off + T]; answer at K' = n + 2; handle = teacher-forced aux argmax at the reduced value after pass t <= n"}
        json.dump(r_, open(o.out or os.path.join(o.run, "eval_s187_posoffset_keep0.json" if o.posoffset_keep0 else "eval_s187_posoffset.json"), "w", newline="\n"), indent=1); print("DONE posoffset", o.run, flush=True); return
    if o.selfdepth:                                                         # note 187-4 (Theory): self-verified depth -> eval_s187_selfdepth_pre.json
        assert kind == "nested" and a.get("aux_step"), "selfdepth needs a nested run trained with §175's aux decoder"
        r_ = {"by_n": selfdepth_fields(m, a, dev, amp, o.bs), "run": o.run.replace(os.sep, "/"), "kind": kind, "eval_dtype": str(amp), "passes": 16,
              "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "sec": round(time.time() - t0),
              "def": "keep iff the aux chain's bare-number halt is at t = n AND its number equals the answer decoder at K' = n + 2; chain_precision = all n aux steps exact (greedy, + EOT) among kept; answer_precision = answer right among kept"}
        json.dump(r_, open(o.out or os.path.join(o.run, "eval_s187_selfdepth_pre.json"), "w", newline="\n"), indent=1); print("DONE selfdepth", o.run, flush=True); return
    if o.deepaux:                                                           # note 187-1 (Theory / Audit): deep-item aux handle + STEP-RECOVER -> eval_s187_deepaux.json
        assert kind == "nested" and a.get("aux_step"), "deepaux needs a nested run trained with §175's aux decoder"
        cc_ = ((float(json.load(open(o.normcap_c))["normcap"]["C"]) if o.normcap_c.endswith(".json") else float(o.normcap_c)) * o.normcap_factor) if o.normcap_c else None
        ins_ = tuple(int(v) for v in o.items_n.split(","))
        da_ = deepaux_fields(m, a, dev, amp, o.bs, n_items=o.n, cap_c=cc_, items_out=o.deepaux_items or o.deepaux_items_only, items_n=ins_, items_tmax=o.items_tmax or None,
                             items_only=o.deepaux_items_only); items_ = da_.pop("_items", None)
        if items_ is not None:                                              # note 187-2: the rows go to their own file; the summary is unchanged
            json.dump({"run": o.run.replace(os.sep, "/"), "items": items_, "item_set": "ALL test_set items with n in %s (notes 187-2 / 187-3)" % (ins_,), "items_tmax": o.items_tmax or 12, "cap_c": cc_,
                       "def": "per item: n, nest_depth (max bracket nesting, nested170.depth), n_syms, think_tokens (incl. EOT and ' .'), handle_by_t[t] = 1 if the teacher-forced aux argmax at the reduced value is right after pass t (t <= its reductions)"},
                      open(os.path.join(os.path.dirname(o.out) if o.out else o.run, o.items_file), "w", newline="\n"))
        if o.deepaux_items_only:
            print("DONE deepaux_items_only", o.run, flush=True); return
        r_ = {"deepaux": da_, "normcap_factor": o.normcap_factor if o.normcap_c else None,"run": o.run.replace(os.sep, "/"), "kind": kind, "eval_dtype": str(amp),
              "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "sec": round(time.time() - t0)}
        json.dump(r_, open(o.out or os.path.join(o.run, "eval_s187_deepaux.json"), "w", newline="\n"), indent=1); print("DONE deepaux", o.run, flush=True); return
    if o.tshift:                                                            # note 182-2 follow-up (Theory): T-SHIFT -> eval_s182_tshift.json
        assert kind == "nested" and a.get("aux_step") and not a.get("aux_nopass"), "T-SHIFT needs a nested run trained with §175's aux decoder (and a pass token: not §187)"
        r_ = {"by_n": tshift_fields(m, a, dev, amp, o.n, o.bs), "run": o.run.replace(os.sep, "/"), "kind": kind, "eval_dtype": str(amp),
              "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "sec": round(time.time() - t0),
              "def": "cells real<t>_prompt<p>: M_t after exactly t real passes (t in n-1..n+1), aux greedy 2 tokens with pass token p (t-1..t+1); done = one number token then EOT; done_and_correct = done and the number is the answer"}
        json.dump(r_, open(o.out or os.path.join(o.run, "eval_s182_tshift.json"), "w", newline="\n"), indent=1); print("DONE tshift", o.run, flush=True); return
    if o.auxdone:                                                           # note 182-2 (Theory): AUX-DONE halting -> eval_s182_auxdone.json
        assert kind == "nested" and a.get("aux_step"), "AUX-DONE needs a nested run trained with §175's aux decoder"
        cc_ = (float(json.load(open(o.cap_c))["normcap"]["C"]) if o.cap_c.endswith(".json") else float(o.cap_c)) if o.cap_c else None
        r_ = {"by_n": auxdone_fields(m, a, dev, amp, o.n, o.bs, cap_c=cc_)}; ref_ = os.path.join(o.run, "eval_s173_halt.json")
        if cc_ is not None:
            r_["cap_c"] = cc_; r_["cap_c_source"] = o.cap_c.replace(os.sep, "/"); r_["cap_def"] = "185-1 NORM-CAP after every write: S_add rescaled to norm cap_c when larger (delta untouched)"
        if os.path.exists(ref_):
            h_ = json.load(open(ref_)); r_["ref_eval_s173_halt"] = {"halt_fixed_k": h_.get("halt_fixed_k"), "answer2": (h_.get("halt") or {}).get("answer2"), "think_k": h_.get("think_k")}
        r_.update({"run": o.run.replace(os.sep, "/"), "kind": kind, "cap": 16, "eval_dtype": str(amp), "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
                   "sec": round(time.time() - t0), "rule": "halt at the first pass t (1..16) whose aux greedy output is one number token then EOT (2 greedy tokens); "
                   "acc_at_halt = main decoder at the halt pass; aux_num_acc / agree over halted items; aux_done_by_t = cumulative share halted (note 182-2)"})
        json.dump(r_, open(o.out or os.path.join(o.run, "eval_s182_auxdone_capped.json" if cc_ is not None else "eval_s182_auxdone.json"), "w", newline="\n"), indent=1); print("DONE auxdone", o.run, flush=True); return
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
