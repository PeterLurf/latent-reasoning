"""THEORY §170 NESTED (eed3f25a; scorer theory/score_s170.py): Feng et al. 2305.15408's Arithmetic(n, 11), Alg. 1 - data, rendering, evaluation, CoT.
Generator: t ~ U(Z11) is the expression; n times: a uniform random number position, an operator uniformly from + - * /, and (t1, t2) uniformly among the
pairs with o(t1, t2) = that number (t2 != 0 for /; / = multiplication by the modular inverse); the number becomes "t1 o t2". Brackets exactly where
precedence or left associativity needs them (a child of lower precedence, or a right child of equal precedence). Spaced tokens: " ( 3 + 10 ) * 4 =".
TEXT-COT (Feng's format): each step reduces the leftmost operation evaluable under precedence and brackets (a node whose operands are both numbers;
re-rendering drops a bracket around the single number in the same step): "e1 = e2 = ... = t"."""
import os
import re

import numpy as np
import tiktoken

P = 11
OPS = ["+", "-", "*", "/"]
PREC = {"+": 1, "-": 1, "*": 2, "/": 2}
ENC = tiktoken.get_encoding("gpt2")
VOCAB = [" %d" % v for v in range(P)] + [" " + o for o in OPS] + [" (", " )", " =", " ."]
TOK = {w: ENC.encode(w) for w in VOCAB}
assert all(len(t) == 1 for t in TOK.values()) and all(ENC.decode(t) == w for w, t in TOK.items()), "§170: every symbol must be one GPT-2 token"
TID = {w: t[0] for w, t in TOK.items()}; NUM_IDS = np.array([TID[" %d" % v] for v in range(P)])
INV = {v: pow(v, P - 2, P) for v in range(1, P)}


def apply(o, a, b):
    return (a + b) % P if o == "+" else (a - b) % P if o == "-" else (a * b) % P if o == "*" else (a * INV[b]) % P


PAIRS = {(o, v): [(a, b) for a in range(P) for b in range(P) if not (o == "/" and b == 0) and apply(o, a, b) == v] for o in OPS for v in range(P)}


class Node:
    __slots__ = ("op", "l", "r", "v")

    def __init__(self, v=None, op=None, l=None, r=None):
        self.v, self.op, self.l, self.r = v, op, l, r


def generate(rng, n):
    """Alg. 1: returns (root, t)."""
    t = int(rng.integers(0, P)); root = Node(v=t); leaves = [root]
    for _ in range(n):
        i = int(rng.integers(0, len(leaves))); leaf = leaves.pop(i); o = OPS[int(rng.integers(0, 4))]; prs = PAIRS[(o, leaf.v)]; a, b = prs[int(rng.integers(0, len(prs)))]
        leaf.op, leaf.l, leaf.r, leaf.v = o, Node(v=a), Node(v=b), None; leaves.insert(i, leaf.r); leaves.insert(i, leaf.l)
    return root, t


def render(nd, parent=None, right=False):
    """minimal-bracket rendering -> list of symbols (" 3", " +", " (" ...)."""
    if nd.op is None:
        return [" %d" % nd.v]
    s = render(nd.l, nd.op, False) + [" " + nd.op] + render(nd.r, nd.op, True)
    if parent is not None and (PREC[nd.op] < PREC[parent] or (right and PREC[nd.op] == PREC[parent])):
        s = [" ("] + s + [" )"]
    return s


def value(nd):
    return nd.v if nd.op is None else apply(nd.op, value(nd.l), value(nd.r))


def depth(syms):
    d = m = 0
    for s in syms:
        d += s == " ("; d -= s == " )"; m = max(m, d)
    return m


def eval_tokens(syms):
    """an INDEPENDENT Z11 evaluator: shunting-yard over the rendered symbols (precedence, left associativity, brackets)."""
    out, st = [], []
    for s in syms:
        w = s.strip()
        if w.isdigit():
            out.append(int(w))
        elif w in PREC:
            while st and st[-1] in PREC and PREC[st[-1]] >= PREC[w]:
                b, a = out.pop(), out.pop(); out.append(apply(st.pop(), a, b))
            st.append(w)
        elif w == "(":
            st.append(w)
        elif w == ")":
            while st[-1] != "(":
                b, a = out.pop(), out.pop(); out.append(apply(st.pop(), a, b))
            st.pop()
    while st:
        b, a = out.pop(), out.pop(); out.append(apply(st.pop(), a, b))
    assert len(out) == 1
    return out[0]


def _leftmost_ready(nd, path=()):
    """the leftmost node (in rendering order) whose operands are both numbers."""
    if nd.op is None:
        return None
    r = _leftmost_ready(nd.l, path + ("l",))
    if r is not None:
        return r
    if nd.l.op is None and nd.r.op is None:
        return path
    return _leftmost_ready(nd.r, path + ("r",))


def _copy(nd):
    return Node(v=nd.v) if nd.op is None else Node(op=nd.op, l=_copy(nd.l), r=_copy(nd.r))


def cot_steps(root):
    """Feng's CoT: the list of expressions e1 (= the input) .. t, each the previous with its leftmost evaluable operation reduced."""
    nd = _copy(root); steps = [render(nd)]
    while nd.op is not None:
        path = _leftmost_ready(nd); tgt = nd
        for p in path:
            tgt = getattr(tgt, p)
        tgt.v = apply(tgt.op, tgt.l.v, tgt.r.v); tgt.op = tgt.l = tgt.r = None; steps.append(render(nd))
    return steps


def ids(syms):
    return [TID[s] for s in syms]


def self_check(n_items=100_000, seed=17000):
    """the generator gate: the independent evaluator re-derives t on n_items samples (n ~ U{1..16}); returns (ok share, n)."""
    rng = np.random.default_rng(seed); ok = 0
    for _ in range(n_items):
        root, t = generate(rng, int(rng.integers(1, 17))); syms = render(root); ok += (eval_tokens(syms) == t) and (value(root) == t)
        if _ < 2000:                                                        # the CoT ends at t and every step evaluates to t
            st = cot_steps(root); assert st[-1] == [" %d" % t] and all(eval_tokens(s) == t for s in st), "cot"
    return ok / n_items, n_items


# ====================================================================================================== §170 runs (DIRECT / INJECT / TEXT-COT)
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("BOXES_NB", "5")
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
EOT = 50256


def test_set(per_n=1000, n_max=16, seed=17100):
    """1,000 items per n in 1..16 (fixed seed): list of dicts {n, syms, t, depth, root}."""
    rng = np.random.default_rng(seed); out = []
    for n in range(1, n_max + 1):
        for _ in range(per_n):
            r, t = generate(rng, n); s = render(r); out.append({"n": n, "syms": s, "t": t, "depth": depth(s), "root": r})
    return out


def train_items(rng, bs, n_max, excl):
    out = []
    while len(out) < bs:
        r, t = generate(rng, int(rng.integers(1, n_max + 1))); s = render(r)
        if tuple(s) not in excl:
            out.append({"syms": s, "t": t, "root": r})
    return out


def pad(seqs, fill=EOT):
    T = max(len(q) for q in seqs); ids = np.full((len(seqs), T), fill); m = np.zeros((len(seqs), T), bool)
    for r, q in enumerate(seqs):
        ids[r, :len(q)] = q; m[r, :len(q)] = True
    return ids, m


def cot_ids(root):
    st = cot_steps(root); out = []
    for k, s in enumerate(st[1:]):
        out += ids(s) + ([TID[" ="]] if k < len(st) - 2 else [])
    return out + [EOT]


# ---- §175 STEP-AUX (Theory 8965745e): a read-only step decoder after each think pass ----
STEP_ID, COLON_ID = ENC.encode(" Step"), ENC.encode(" :")
assert len(STEP_ID) == 1 and len(COLON_ID) == 1
STEP_ID, COLON_ID = STEP_ID[0], COLON_ID[0]


def pass_tok(t):
    tk = ENC.encode(" %d" % t); assert len(tk) == 1; return tk[0]


def _render_pos(nd, tgt, parent=None, right=False):
    """render() plus the symbol index of node tgt in it (None if tgt is not in this subtree)."""
    if nd.op is None:
        return [" %d" % nd.v], (0 if nd is tgt else None)
    ls, lp = _render_pos(nd.l, tgt, nd.op, False); rs, rp = _render_pos(nd.r, tgt, nd.op, True)
    sy = ls + [" " + nd.op] + rs; po = lp if lp is not None else (len(ls) + 1 + rp if rp is not None else None)
    if parent is not None and (PREC[nd.op] < PREC[parent] or (right and PREC[nd.op] == PREC[parent])):
        sy = [" ("] + sy + [" )"]; po = None if po is None else po + 1
    return sy, po


def aux_steps(root):
    """§175: per reduction t = 1..n, (e_{t+1} as symbols = cot_steps(root)[t], the index of the newly reduced value in it)."""
    nd = _copy(root); out = []
    while nd.op is not None:
        path = _leftmost_ready(nd); tgt = nd
        for p_ in path:
            tgt = getattr(tgt, p_)
        tgt.v = apply(tgt.op, tgt.l.v, tgt.r.v); tgt.op = tgt.l = tgt.r = None; out.append(_render_pos(nd, tgt))
    return out


def aux_of(it):
    if "aux" not in it:
        it["aux"] = aux_steps(it["root"])
    return it["aux"]


class InjectN(torch.nn.Module):
    """§169's inject harness on a variable-length text: think = [EOT, expr, " ."] (right-padded, padded writes masked); decoder = [EOT, expr, " ="] -> the
    answer token (read-only)."""
    def __init__(self, g, layer=6, write="top"):
        super().__init__(); import boxes.deltamem as DM; self.DM = DM; self.g = g; self.mem = DM.DeltaMem(g, layer, write)

    def think_inputs(self, items, dev):
        ids_, m_ = pad([[EOT] + ids(it["syms"]) + [TID[" ."]] for it in items]); return torch.tensor(ids_, device=dev), torch.tensor(m_, device=dev)

    def write(self, S, x, mask):
        return self.DM.gpt_forward_mem(self.g, self.g.embed(x), self.mem, S, mask=mask)[1]

    def think(self, S, x, mask, K, skip=(), override=None, ckpt=False):
        from torch.utils.checkpoint import checkpoint
        for t in range(1, K + 1):
            if t not in skip:
                S = checkpoint(self.write, S, x, mask, use_reentrant=False) if (ckpt and torch.is_grad_enabled()) else self.write(S, x, mask)
            if override is not None and override[0] == t:
                S = override[1]
        return S

    def dec_logits(self, S, items, dev):
        from boxes.deltas5 import fwd_read
        seqs = [[EOT] + ids(it["syms"]) + [TID[" ="]] for it in items]; ids_, _ = pad(seqs); x = torch.tensor(ids_, device=dev)
        h = fwd_read(self.g, self.g.embed(x), self.mem, S); pos = torch.tensor([len(q) - 1 for q in seqs], device=dev)
        return self.g.logits(h[torch.arange(len(seqs), device=dev), pos]).float()   # (B, V) at " ="

    def aux_logits(self, S, items, t, dev):
        """§175: the SAME trunk, read-only (fwd_read: the read on S, no write), over [EOT, " Step", " t", " :"] + e_{t+1} (teacher-forced). It sees no
        problem text and is never fed back. -> (logits (b, L, V) predicting e_{t+1} then EOT, targets (b, L), -100 = padding)."""
        from boxes.deltas5 import fwd_read
        pre = [EOT, STEP_ID, pass_tok(t), COLON_ID]; seqs, tg = [], []
        for it in items:
            tgt = ids(aux_of(it)[t - 1][0]) + [EOT]; seqs.append(pre + tgt[:-1]); tg.append(tgt)
        x_, _ = pad(seqs); y_, _ = pad(tg, -100); x = torch.tensor(x_, device=dev); y = torch.tensor(y_, device=dev)
        assert x.shape[1] == len(pre) + y.shape[1] - 1
        h = fwd_read(self.g, self.g.embed(x), self.mem, S)
        return self.g.logits(h[:, len(pre) - 1:]).float(), y

    @torch.no_grad()
    def aux_greedy(self, S, t, L):
        """§175 eval: greedy L tokens after [EOT, " Step", " t", " :"] reading S -> (b, L) token ids."""
        from boxes.deltas5 import fwd_read
        x = torch.tensor([[EOT, STEP_ID, pass_tok(t), COLON_ID]] * S.shape[0], device=S.device)
        for _ in range(L):
            nt = self.g.logits(fwd_read(self.g, self.g.embed(x), self.mem, S)[:, -1]).float().argmax(-1); x = torch.cat([x, nt[:, None]], 1)
        return x[:, 4:].cpu().numpy()


def ans_token_acc(lg, items):
    return (lg[:, NUM_IDS].argmax(-1).cpu().numpy() == np.array([it["t"] for it in items])), lg.argmax(-1).cpu().numpy()



def reductions(root):
    """the values of Feng's CoT reductions in cot_steps' (leftmost-handle) order: v_1 .. v_n."""
    nd = _copy(root); vals = []
    while nd.op is not None:
        path = _leftmost_ready(nd); tgt = nd
        for p_ in path:
            tgt = getattr(tgt, p_)
        tgt.v = apply(tgt.op, tgt.l.v, tgt.r.v); tgt.op = tgt.l = tgt.r = None; vals.append(tgt.v)
    return vals


def _mem_read(mem, x, S):
    """read_fixed's reads at one position per row (as boxes/halt173.read_vec): x (b, D) residual entering the read, S -> (b, 768) [g_add read_add || g_del read_del]."""
    import torch.nn.functional as F_
    from boxes.deltamem import H_ADD, H_DEL, DH
    with torch.autocast("cuda", enabled=False):
        b, D = x.shape; q = mem.qkv(mem.ln(x.float()))[..., :D].view(b, H_ADD + H_DEL, DH)
        oa = torch.exp(mem.s_add) * torch.einsum("bhk,bhkv->bhv", F_.elu(q[:, :H_ADD]) + 1, S[:, :H_ADD].float())
        od = torch.exp(mem.s_del) * torch.einsum("bhvk,bhk->bhv", S[:, H_ADD:].float(), F_.normalize(q[:, H_ADD:], dim=-1))
        return torch.cat([oa, od], 1).flatten(1)


@torch.no_grad()
def dec_reads(m, items, T, dev, ac, bs):
    """note 170-2: after think pass t = 1..T, the decoder's memory read r_t at its " =" position (the read entering block mem.layer; the residual there does
    not depend on the memory) -> (T, N, 768)."""
    g = m.g; out = []
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]; seqs = [[EOT] + ids(it["syms"]) + [TID[" ="]] for it in ch]; ids_, _ = pad(seqs); x = torch.tensor(ids_, device=dev)
        pos = torch.tensor([len(q) - 1 for q in seqs], device=dev); xt, msk = m.think_inputs(ch, dev); rs = []
        with ac():
            h = g.embed(x) + g.w("wpe.weight")[:x.shape[1]][None]
            for j in range(m.mem.layer):
                h = m.DM._block(g, j, h, h.shape[-1] // g.H)
            h6 = h[torch.arange(len(ch), device=dev), pos]; S = m.mem.zero_state(len(ch), dev)
            for _ in range(T):
                S = m.write(S, xt, msk); rs.append(_mem_read(m.mem, h6, S))
        out.append(torch.stack(rs).float().cpu().numpy())
    return np.concatenate(out, 1)


def logreg_acc(Xf, yf, Xs, ys, dev, wd=1e-3, iters=200):
    """multinomial logistic probe (standardised features, L2 wd, LBFGS) fit on (Xf, yf) -> accuracy on (Xs, ys)."""
    import torch.nn.functional as F_
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    Xf_ = torch.tensor((Xf - mu) / sd, device=dev, dtype=torch.float32); Xs_ = torch.tensor((Xs - mu) / sd, device=dev, dtype=torch.float32)
    yf_ = torch.tensor(yf, device=dev, dtype=torch.long); ys_ = torch.tensor(ys, device=dev, dtype=torch.long)
    with torch.enable_grad(), torch.autocast("cuda", enabled=False):
        W = torch.zeros(Xf.shape[1], P, device=dev, requires_grad=True); b = torch.zeros(P, device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

        def clos():
            opt.zero_grad(); l_ = F_.cross_entropy(Xf_ @ W + b, yf_) + wd * (W ** 2).sum(); l_.backward(); return l_
        opt.step(clos)
    return float(((Xs_ @ W + b).argmax(1) == ys_).float().mean())


def step_recover(m, excl, TE, T, dev, ac, bs, n_fit=4000, n_test=1000):
    """note 170-2 (descriptive): per pass t, a multinomial logistic probe from r_t to v_t (the t-th leftmost reduction's value, Z11); fit on n_fit
    training-distribution items (seed 17021), tested on n_test test items at n <= 8 (seed 17022); items with fewer than t reductions are dropped at t."""
    tr = train_items(np.random.default_rng(17021), n_fit, 8, excl); pool = [it for it in TE if it["n"] <= 8]
    te = [pool[i] for i in np.random.default_rng(17022).permutation(len(pool))[:n_test]]
    Rf, Rs = dec_reads(m, tr, T, dev, ac, bs), dec_reads(m, te, T, dev, ac, bs); vf = [reductions(it["root"]) for it in tr]; vs = [reductions(it["root"]) for it in te]
    acc, nn, maj = {}, {}, {}
    for t in range(1, T + 1):
        fi = [i for i, v in enumerate(vf) if len(v) >= t]; si = [i for i, v in enumerate(vs) if len(v) >= t]
        if len(fi) < 100 or len(si) < 20:
            continue
        yf = np.array([vf[i][t - 1] for i in fi]); ys = np.array([vs[i][t - 1] for i in si])
        acc[str(t)] = logreg_acc(Rf[t - 1, fi], yf, Rs[t - 1, si], ys, dev); nn[str(t)] = {"fit": len(fi), "test": len(si)}
        maj[str(t)] = float((ys == np.bincount(yf, minlength=P).argmax()).mean())
    return acc, {"n": nn, "majority_baseline": maj, "probe": "multinomial logistic on standardised r_t (768-d), L2 1e-3, LBFGS 200", "seeds": [17021, 17022]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True, choices=["direct", "inject", "textcot"]); ap.add_argument("--think_k", type=int, default=0)
    ap.add_argument("--steps", type=int, default=48000); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n_max_train", type=int, default=8); ap.add_argument("--micro", type=int, default=1); ap.add_argument("--ckpt_passes", action="store_true")
    ap.add_argument("--mem_write", default="top", choices=["top"]); ap.add_argument("--write_mix_start", type=int, default=2000); ap.add_argument("--write_mix_end", type=int, default=6000)
    ap.add_argument("--ckpt_every", type=int, default=2000); ap.add_argument("--resume", default=None); ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--cot_eval_bs", type=int, default=64, help="TEXT-COT generation batch (KV cache; b223 OOMed at 250 on n = 16 chains)"); ap.add_argument("--eval_only", action="store_true", help="evaluate <out>/model.pt (no training; writes args_eval.json; model.pt / args.json / log untouched)"); ap.add_argument("--eval_bs", type=int, default=250); ap.add_argument("--per_n", type=int, default=1000); ap.add_argument("--serial_pool", type=int, default=2200); ap.add_argument("--auto_micro", action="store_true")
    ap.add_argument("--mem_budget", type=float, default=21.0); ap.add_argument("--mem_check", action="store_true"); ap.add_argument("--time_steps", type=int, default=0, help="cost check: train this many steps, report s/step, exit")
    ap.add_argument("--eval_at", type=int, default=0, help="also evaluate acc_by_n at this step -> eval_<k>k.json"); ap.add_argument("--aux_step", type=float, default=0.0, help="§175: lambda of the read-only step decoder CE (0 = off: the §170 path)"); ap.add_argument("--aux_fade", default="36000,48000", help="§175: lambda constant to the first step, then linear to 0 at the second"); ap.add_argument("--fast_add", action="store_true", help="deltamem.FAST_ADD: the additive heads' pass writes as one sum (Audit 2026-10-09; scratch/gate_fastadd.py)"); ap.add_argument("--sched_scale", type=float, default=1.0, help="smokes only"); ap.add_argument("--fp32", action="store_true"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); a.task = "nested"; import boxes.deltamem as DM_; DM_.FAST_ADD = bool(a.fast_add); os.makedirs(a.out, exist_ok=True); dev = a.device; sc = a.sched_scale; K = a.think_k if a.arm == "inject" else 0
    if a.arm != "inject":
        a.mem_write = a.write_mix_start = a.write_mix_end = None; a.think_k = 0
    from boxes import s5
    from boxes import core as bc
    from boxes.fsq120 import make_opt, lr_at, driver_version, env_info
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    amp = None if (a.fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    ac = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
    TE = test_set(a.per_n); EXCL = {tuple(it["syms"]) for it in TE}; g = s5.load_gpt2(dev)
    r_sp = np.random.default_rng(17300); SPOOL = []                         # note 170-1: the dedicated n = 8 SERIAL pool (excluded from training)
    for _ in range(a.serial_pool):
        r__, t__ = generate(r_sp, 8); s__ = render(r__); SPOOL.append({"n": 8, "syms": s__, "t": t__, "depth": depth(s__), "root": r__})
    EXCL |= {tuple(it["syms"]) for it in SPOOL}
    m = InjectN(g, 6, a.mem_write).to(dev) if a.arm == "inject" else None

    def loss_items(items):
        if a.arm == "direct":
            seqs = [[EOT] + ids(it["syms"]) + [TID[" ="]] for it in items]; ids_, _ = pad(seqs); x = torch.tensor(ids_, device=dev)
            with ac():
                h, _ = g(g.embed(x), None)
            pos = torch.tensor([len(q) - 1 for q in seqs], device=dev); lg = g.logits(h[torch.arange(len(seqs), device=dev), pos]).float()
            return Fn.cross_entropy(lg, torch.tensor(NUM_IDS[[it["t"] for it in items]], device=dev))
        if a.arm == "textcot":
            seqs, lab = [], []
            for it in items:
                pre = [EOT] + ids(it["syms"]) + [TID[" ="]]; tgt = cot_ids(it["root"]); seqs.append(pre + tgt); lab.append([-100] * len(pre) + tgt)
            ids_, _ = pad(seqs); lb, _ = pad(lab, -100); x = torch.tensor(ids_, device=dev); y = torch.tensor(lb, device=dev)
            with ac():
                h, _ = g(g.embed(x), None)
            lg = g.logits(h[:, :-1]).float(); return Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), y[:, 1:].reshape(-1), ignore_index=-100)
        x, msk = m.think_inputs(items, dev)
        if not lam_now[0]:                                                  # §170 (and §175 at lambda 0): unchanged
            with ac():
                S = m.think(m.mem.zero_state(len(items), dev), x, msk, K, ckpt=a.ckpt_passes); lg = m.dec_logits(S, items, dev)
            return Fn.cross_entropy(lg, torch.tensor(NUM_IDS[[it["t"] for it in items]], device=dev))
        from torch.utils.checkpoint import checkpoint
        per = [[] for _ in items]                                           # §175: the aux CE (token mean) of each item's passes 1..#steps
        with ac():
            S = m.mem.zero_state(len(items), dev)
            for t in range(1, K + 1):
                S = checkpoint(m.write, S, x, msk, use_reentrant=False) if (a.ckpt_passes and torch.is_grad_enabled()) else m.write(S, x, msk)
                idx = [i for i, it in enumerate(items) if len(aux_of(it)) >= t]
                if idx:
                    lg_a, y_a = m.aux_logits(S[idx], [items[i] for i in idx], t, dev)
                    ce = Fn.cross_entropy(lg_a.transpose(1, 2), y_a, ignore_index=-100, reduction="none").sum(1) / (y_a != -100).sum(1)
                    for r_, i in enumerate(idx):
                        per[i].append(ce[r_])
            lg = m.dec_logits(S, items, dev)
        la = Fn.cross_entropy(lg, torch.tensor(NUM_IDS[[it["t"] for it in items]], device=dev))
        aux = torch.stack([torch.stack(p_).mean() for p_ in per if p_]).mean(); aux_hist.append((float(la), float(aux)))
        return la + lam_now[0] * aux

    def mem_check(micro):
        torch.cuda.reset_peak_memory_stats(dev); r_ = np.random.default_rng(0); its = [it for it in train_items(r_, 4 * a.bs, a.n_max_train, set()) if True]
        its = sorted(its, key=lambda it: -len(it["syms"]))[:a.bs // micro]; l_ = loss_items(its); l_.backward(); (m or g).zero_grad(set_to_none=True)
        evaluate_n(sorted([it for it in TE if it["n"] == 16], key=lambda it: -len(it["syms"]))[:a.eval_bs], 2 * K)   # the n = 16 eval at K' = 2K
        return torch.cuda.max_memory_allocated(dev) / 2 ** 30

    @torch.no_grad()
    def inject_answers(items, Kp, skip=(), override=None):
        out = []
        for i in range(0, len(items), a.eval_bs):
            ch = items[i:i + a.eval_bs]; x, msk = m.think_inputs(ch, dev)
            with ac():
                S = m.mem.zero_state(len(ch), dev)
                for t in range(1, Kp + 1):
                    if t not in skip:
                        S = m.write(S, x, msk)
                    if override is not None and override[0] == t:
                        S = torch.tensor(override[1][i:i + a.eval_bs], device=dev, dtype=S.dtype)
                lg = m.dec_logits(S, ch, dev)
            out.append(lg[:, NUM_IDS].argmax(-1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def inject_states(items, t):
        out = []
        for i in range(0, len(items), a.eval_bs):
            ch = items[i:i + a.eval_bs]; x, msk = m.think_inputs(ch, dev)
            with ac():
                S = m.mem.zero_state(len(ch), dev)
                for _ in range(t):
                    S = m.write(S, x, msk)
            out.append(S.float().cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def direct_answers(items):
        out = []
        for i in range(0, len(items), a.eval_bs):
            ch = items[i:i + a.eval_bs]; seqs = [[EOT] + ids(it["syms"]) + [TID[" ="]] for it in ch]; ids_, _ = pad(seqs); x = torch.tensor(ids_, device=dev)
            with ac():
                h, _ = g(g.embed(x), None)
            pos = torch.tensor([len(q) - 1 for q in seqs], device=dev); out.append(g.logits(h[torch.arange(len(ch), device=dev), pos]).float()[:, NUM_IDS].argmax(-1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def cot_answers(items):
        """greedy CoT, budget = the true CoT length + 50, batched by equal prefix length (KV cache); the answer = the text after the last " =" before EOT
        when it is a single number; returns (answers (-1 = wrong form), generated token counts)."""
        ans = np.full(len(items), -1); ntok = np.zeros(len(items), np.int64); by_len = {}
        for i, it in enumerate(items):
            by_len.setdefault(len(it["syms"]), []).append(i)
        for L, idx in by_len.items():
            for s in range(0, len(idx), a.cot_eval_bs):
                ii = idx[s:s + a.cot_eval_bs]; x = torch.tensor([[EOT] + ids(items[j]["syms"]) + [TID[" ="]] for j in ii], device=dev)
                budget = max(len(cot_ids(items[j]["root"])) for j in ii) + 50
                with ac():
                    h, past = g(g.embed(x), None)
                gen = []; done = torch.zeros(len(ii), dtype=torch.bool, device=dev)
                for _ in range(budget):
                    nt = g.logits(h[:, -1]).float().argmax(-1); gen.append(nt); done |= nt == EOT
                    if bool(done.all()):
                        break
                    with ac():
                        h, past = g(g.embed(nt[:, None]), past)
                G = torch.stack(gen, 1).cpu().numpy(); del h, past, gen
                if dev.startswith("cuda"):
                    torch.cuda.empty_cache()
                for r, j in enumerate(ii):
                    row = list(G[r]); e = row.index(EOT) if EOT in row else len(row); body = row[:e]; ntok[j] = e + (1 if EOT in row else 0)
                    if EOT not in row:
                        continue
                    last = max([k for k, tk in enumerate(body) if tk == TID[" ="]], default=-1); tail = body[last + 1:]
                    if len(tail) == 1 and tail[0] in NUM_IDS:
                        ans[j] = int(np.where(NUM_IDS == tail[0])[0][0])
        return ans, ntok

    f0_, f1_ = [float(x_) for x_ in a.aux_fade.split(",")]; lam_now = [0.0]; aux_hist = []

    def lam_at(step):
        """§175: lambda = aux_step to f0, then linear to 0 at f1 (sched_scale-scaled)."""
        if not a.aux_step:
            return 0.0
        return a.aux_step if step <= f0_ * sc else a.aux_step * max(0.0, (f1_ * sc - step) / max(1.0, (f1_ - f0_) * sc))

    @torch.no_grad()
    def aux_eval(items, Kp, tmax):
        """§175: after each think pass t <= min(n, tmax): greedy exact match of e_{t+1} + EOT; teacher-forced argmax at the newly reduced value vs the copied tokens."""
        ex, role = {}, {}
        for i in range(0, len(items), a.eval_bs):
            ch = items[i:i + a.eval_bs]; x, msk = m.think_inputs(ch, dev)
            with ac():
                S = m.mem.zero_state(len(ch), dev)
                for t in range(1, Kp + 1):
                    S = m.write(S, x, msk)
                    if t > tmax:
                        continue
                    idx = [j for j, it in enumerate(ch) if len(aux_of(it)) >= t]
                    if not idx:
                        continue
                    sub = [ch[j] for j in idx]; lg_a, y_a = m.aux_logits(S[idx], sub, t, dev); pr = lg_a.argmax(-1).cpu().numpy(); yy = y_a.cpu().numpy()
                    gen = m.aux_greedy(S[idx], t, int((yy != -100).sum(1).max()))
                    for r_, it in enumerate(sub):
                        sy, hi = aux_of(it)[t - 1]; L_ = len(sy); tgt = ids(sy) + [EOT]; ok_ = pr[r_, :L_] == yy[r_, :L_]
                        rr = role.setdefault(t, {"handle": [], "copy": []}); rr["handle"].append(bool(ok_[hi])); rr["copy"].extend(bool(ok_[k_]) for k_ in range(L_) if k_ != hi)
                        ex.setdefault(t, []).append(bool(list(gen[r_, :len(tgt)]) == tgt))
        return ({str(t): float(np.mean(v)) for t, v in sorted(ex.items())},
                {str(t): {k_: (float(np.mean(v)) if v else None) for k_, v in d.items()} for t, d in sorted(role.items())}, {str(t): len(v) for t, v in sorted(ex.items())})

    def s175_fields():
        pool = [it for it in TE if it["n"] <= 8]; te8 = [pool[i] for i in np.random.default_rng(17022).permutation(len(pool))[:1000]]
        pe = [it for it in TE if 9 <= it["n"] <= 12]; te12 = [pe[i] for i in np.random.default_rng(17023).permutation(len(pe))[:1000]]
        acc8, role8, n8 = aux_eval(te8, K, K); acc12, _, n12 = aux_eval(te12, 16, 16); sr, sri = step_recover(m, EXCL, TE, K, dev, ac, a.eval_bs)
        return {"aux_acc_by_pass": acc8, "aux_acc_by_pass_ext": acc12, "aux_acc_by_role": role8, "step_recover": sr, "step_recover_info": sri,
                "n_by_pass": n8, "n_by_pass_ext": n12, "ext_kprime": 16, "items": {"n_le_8": {"n": len(te8), "seed": 17022}, "n_9_12": {"n": len(te12), "seed": 17023}}}

    def evaluate_n(items, Kp=None):
        if a.arm == "direct":
            return direct_answers(items)
        if a.arm == "textcot":
            return cot_answers(items)[0]
        return inject_answers(items, K if Kp is None else Kp)

    lam_now[0] = a.aux_step                                                 # §175: memory-check with the aux on (the largest graph)
    if a.mem_check or (a.auto_micro and dev.startswith("cuda") and not a.resume and not a.eval_only):
        mc = []
        while True:
            pk = mem_check(a.micro); mc.append({"micro": a.micro, "peak_gib": pk}); print("MEM", a.micro, pk, flush=True)
            if pk <= a.mem_budget or a.micro >= a.bs or a.mem_check:
                break
            a.micro *= 2
        if a.arm == "textcot" and dev.startswith("cuda"):                   # Audit 2026-10-09 (b223 OOM): the worst CoT eval batch = the longest n = 16 chain x cot_eval_bs
            worst = max([it for it in TE if it["n"] == 16], key=lambda it: len(cot_ids(it["root"])))
            while True:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev); cot_answers([worst] * a.cot_eval_bs); pk_ = torch.cuda.max_memory_allocated(dev) / 2 ** 30
                mc.append({"cot_eval_bs": a.cot_eval_bs, "peak_gib": pk_, "chain_tokens": len(cot_ids(worst["root"]))}); print("MEM cot", a.cot_eval_bs, pk_, flush=True)
                if pk_ <= a.mem_budget or a.cot_eval_bs <= 4:
                    break
                a.cot_eval_bs //= 2
        json.dump({"checks": mc, "micro": a.micro, "peak_gib": mc[-1]["peak_gib"], "train": "the longest of 4 bs draws at n <= %d" % a.n_max_train, "eval": "n = 16 at K' = 2K"},
                  open(os.path.join(a.out, "mem_check.json"), "w", newline="\n"), indent=1)
        if a.mem_check:
            return
        assert mc[-1]["peak_gib"] <= a.mem_budget
    info = {**vars(a), "train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_driver": driver_version(), **env_info(), "hostprobe_sps": os.environ.get("HOSTPROBE_SPS"),
            "format": {"direct": "[EOT, expr, ' ='] -> the answer token", "inject": "think [EOT, expr, ' .'] (padded writes masked); decoder [EOT, expr, ' ='] -> answer (read-only)",
                       "textcot": "[EOT, expr, ' ='] then ' e2 = e3 = ... = t' EOT (Feng; loss on all generated tokens)"}[a.arm]}
    if a.eval_only:                                                         # Theory 2026-10-09: train_gpu / train_driver are the training run's (args.json); the eval GPU is "gpu"
        a0_ = json.load(open(os.path.join(a.out, "args.json"))); info["eval_gpu"], info["eval_driver"] = info["train_gpu"], info["train_driver"]
        info["train_gpu"], info["train_driver"] = a0_.get("train_gpu"), a0_.get("train_driver")
    if a.aux_step:                                                          # §175 (Audit): state the aux decoder's weight sharing in the run's args
        info["aux_decoder"] = ("the SAME GPT-2 trunk and LM head as the main decoder (shared weights, no new parameters), read-only on M_t via fwd_read: "
                               "[EOT, ' Step', ' t', ' :'] + e_{t+1} + EOT (teacher-forced); no problem text, no write, never fed back; dropped at test")
    json.dump(info, open(os.path.join(a.out, "args_eval.json" if a.eval_only else (("args_resume_%d.json" % int(time.time())) if a.resume else "args.json")), "w", newline="\n"), indent=1)
    params = m if m is not None else g; opt = make_opt(params, a.lr); step0 = 1; n_nonfinite = 0; hist = []; wg_hist = []; wwin = []; wlast = []
    log = open(os.path.join(a.out, "log.jsonl"), "a" if (a.resume or a.eval_only) else "w", newline="\n"); t0 = time.time()
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False); params.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); rng.bit_generator.state = ck["np_rng"]
        step0 = ck["step"] + 1; n_nonfinite = ck["n_nonfinite"]; wg_hist = ck.get("wg_hist", []); t0 = time.time() - ck["elapsed"]; log.write(json.dumps({"step": ck["step"], "resumed": a.resume}) + "\n"); log.flush()
    elif m is not None and K > 0:                                           # note 147-2: g_add calibration at the think hand-offs
        its = train_items(np.random.default_rng(17001), 64, a.n_max_train, set()); x, msk = m.think_inputs(its, dev); m.mem.norms = []
        with torch.no_grad(), ac():
            m.think(m.mem.zero_state(64, dev), x, msk, min(K, 4))
        nr = m.mem.norms[1:]; m.mem.norms = None; r0 = float(torch.stack([d_ / a_.clamp_min(1e-12) for d_, a_ in nr]).median()) if nr else 1.0
        m.mem.s_add.data.fill_(math.log(max(r0, 1e-12))); json.dump({"g_add_init": r0}, open(os.path.join(a.out, "s170_init.json"), "w", newline="\n"))
    last = a.steps if not a.time_steps else min(a.steps, step0 + a.time_steps - 1); tt0 = time.time()
    if a.eval_only:                                                         # no training: the saved model, the training log's time / non-finite count
        params.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev, weights_only=True)); last = step0 - 1
        lg_ = [json.loads(l_) for l_ in open(os.path.join(a.out, "log.jsonl"))]; n_nonfinite = sum(1 for r_ in lg_ if r_.get("nonfinite"))
        sec_ = [r_["sec"] for r_ in lg_ if "sec" in r_]; t0 = time.time() - (sec_[-1] if sec_ else 0); wg_hist = [r_ for r_ in lg_ if "write_grad_norm" in r_]
    for step in range(step0, last + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(a.lr, gr, step, a.steps)
        params.train()
        if m is not None:
            m.mem.fade.fill_(min(1.0, step / max(1, int(500 * sc))))
            m.mem.gamma.fill_(0.0 if step <= a.write_mix_start * sc else min(1.0, (step - a.write_mix_start * sc) / max(1.0, (a.write_mix_end - a.write_mix_start) * sc)))
        lam_now[0] = lam_at(step)
        its = train_items(rng, a.bs, a.n_max_train, EXCL); opt.zero_grad(); tot = 0.0; ch = a.bs // a.micro
        for k_ in range(a.micro):
            lk = loss_items(its[k_ * ch:(k_ + 1) * ch]); (lk / a.micro).backward(); tot += float(lk.item()) / a.micro
        if m is not None:
            from boxes.deltas5 import write_grad_norm
            wwin.append(write_grad_norm(m))
        gn = torch.nn.utils.clip_grad_norm_(params.parameters(), 1.0)
        if not (math.isfinite(tot) and bool(torch.isfinite(gn))):
            n_nonfinite += 1; opt.zero_grad(); log.write(json.dumps({"step": step, "nonfinite": 1}) + "\n"); log.flush()
        else:
            opt.step(); hist.append(tot)
        if a.eval_at and step == a.eval_at:                                 # Theory 7caa3a82 (§174's matched budget): acc_by_n at this step, no checkpoint kept
            params.eval(); Ne = np.array([it["n"] for it in TE]); Ce = evaluate_n(TE) == np.array([it["t"] for it in TE])
            e_ = {"step": step, "acc_by_n": {str(n): float(Ce[Ne == n].mean()) for n in range(1, 17)}}
            if a.aux_step and K:
                e_["s175"] = s175_fields(); e_["aux_lambda"] = lam_now[0]
            json.dump(e_, open(os.path.join(a.out, "eval_%dk.json" % (step // 1000)), "w", newline="\n"), indent=1)
        if step % a.log_every == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(hist)) if hist else None, "sec": round(time.time() - t0)}; hist = []
            if a.aux_step:
                rec.update({"aux_lambda": lam_now[0], "answer_ce": float(np.mean([x_[0] for x_ in aux_hist])) if aux_hist else None,
                            "aux_ce": float(np.mean([x_[1] for x_ in aux_hist])) if aux_hist else None}); aux_hist.clear()
            if m is not None and wwin:
                md_ = float(np.median(wlast)) if wlast else 0.0; mx_ = float(np.max(wwin)); late = step > a.write_mix_start * sc
                w_rec = {"step": step, "mean": float(np.mean(wwin)), "max": mx_, "median_prev_1k": md_, "spike_100x_floor": bool(late and mx_ > 100 * md_ > 0 and mx_ > 1.0)}
                wlast.extend(wwin); del wlast[:-int(1000 * sc) or None]; wwin = []; wg_hist.append(w_rec)
                rec.update({"gamma": float(m.mem.gamma), "fade": float(m.mem.fade), "write_grad_norm": w_rec["mean"], "write_grad_spike_100x_floor": w_rec["spike_100x_floor"]})
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps:
            cp = os.path.join(a.out, "ckpt_last.pt")
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            torch.save({"model": params.state_dict(), "opt": opt.state_dict(), "np_rng": rng.bit_generator.state, "step": step, "n_nonfinite": n_nonfinite, "wg_hist": wg_hist,
                        "elapsed": time.time() - t0}, cp)
    if a.time_steps:                                                        # Audit's cost check
        sps = (time.time() - tt0) / max(1, last - step0 + 1); json.dump({"arm": a.arm, "think_k": K, "steps_timed": last - step0 + 1, "sec_per_step": sps, "est_hours_48k": sps * a.steps / 3600,
                                                                       "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "micro": a.micro},
                                                                      open(os.path.join(a.out, "timing.json"), "w", newline="\n"), indent=1); print("TIMING", sps, flush=True)
        json.dump(json.load(open(os.path.join(a.out, "timing.json"))), open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1); return   # so the drain fetches it
    train_sec = round(time.time() - t0)
    if not a.eval_only:
        torch.save(params.state_dict(), os.path.join(a.out, "model.pt"))
    opt.zero_grad(set_to_none=True); opt.state.clear()                      # Audit 2026-10-09: free the optimizer state and grads before the eval (b223 OOM)
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    params.eval(); t1 = time.time()
    N = np.array([it["n"] for it in TE]); Dp = np.array([it["depth"] for it in TE]); T_ = np.array([it["t"] for it in TE])
    if a.arm == "textcot":
        A_, NT = cot_answers(TE)
    else:
        A_ = evaluate_n(TE); NT = None
    C = A_ == T_
    res = {"task": "nested", "eval_only": bool(a.eval_only), "arm": a.arm, "think_k": K, "seed": a.seed, "acc_by_n": {str(n): float(C[N == n].mean()) for n in range(1, 17)},
           "acc_by_depth": {str(d): float(C[Dp == d].mean()) for d in sorted(set(Dp.tolist()))}, "n_by_depth": {str(d): int((Dp == d).sum()) for d in sorted(set(Dp.tolist()))},
           "train_sec": train_sec, "n_nonfinite": n_nonfinite, "micro": a.micro, "eval_dtype": str(amp), "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
           "train_gpu": info["train_gpu"], "train_driver": info["train_driver"], "per_n": a.per_n}
    if NT is not None:
        res["cot_tokens_by_n"] = {str(n): float(NT[N == n].mean()) for n in range(1, 17)}
    mt = lambda: bc.Meter(g)  # noqa: E731
    cb = {}
    for n in range(1, 17):
        L_ = float(np.mean([len(it["syms"]) for it in TE if it["n"] == n])) + 2; M_ = mt()
        if a.arm == "direct":
            M_.step(int(round(L_)), 0, 1)
        elif a.arm == "inject":
            for _ in range(K):
                M_.step(int(round(L_)), 0, 0)
            M_.step(int(round(L_)), 0, 1)
        else:
            M_.step(int(round(L_)), 0, 1)
            for j in range(int(round(res["cot_tokens_by_n"][str(n)])) - 1):
                M_.step(1, int(round(L_)) + j, 1)
        cb[str(n)] = {"flops_trunk": M_.flops, "sequential_passes": M_.passes}
    res["compute_by_n"] = cb; res["compute_note"] = "boxes.core.Meter per item at the mean length of each n (trunk; the DeltaMem branch adds < 5% at these lengths)"
    if a.arm == "inject":
        s = {"acc_by_n_kprime": {}, "skip_pass": {}, "serial": {}, "write_grad_norm": wg_hist, "write_grad_spikes_floor": [x_ for x_ in wg_hist if x_["spike_100x_floor"]], "n_nonfinite": n_nonfinite}
        for kp in sorted({0, 1, K // 2, K, 2 * K}):
            Ak = A_ if kp == K else inject_answers(TE, kp); s["acc_by_n_kprime"][str(kp)] = {str(n): float((Ak == T_)[N == n].mean()) for n in range(1, 17)}
        for t in sorted({t for t in (1, K // 4, K // 2, 3 * K // 4, K) if t >= 1}):
            Ak = inject_answers(TE, K, skip=(t,)); s["skip_pass"][str(t)] = {str(n): float((Ak == T_)[N == n].mean()) for n in range(1, 17)}
        rec_, don_, hyb_ = serial_items(SPOOL, np.random.default_rng(17200)); s["n_serial_items"] = len(rec_); s["serial_pool"] = {"n": len(SPOOL), "seed": 17300}
        if rec_:
            tr = np.array([value(it["root"]) for it in rec_]); td = np.array([value(it["root"]) for it in don_]); th = np.array(hyb_)
            for t in sorted({t for t in (1, 2, K // 4, K // 2, 3 * K // 4) if 1 <= t < K}):
                Sd = inject_states(don_, t); got = inject_answers(rec_, K, override=(t, Sd))
                s["serial"][str(t)] = {"hybrid": float((got == th).mean()), "donor": float((got == td).mean()), "recipient": float((got == tr).mean())}
            perm = np.random.default_rng(17201).permutation(len(rec_)); tnull = sorted(s["serial"])[0] if s["serial"] else None
            if tnull is not None:
                Sn = inject_states([rec_[i] for i in perm], int(tnull)); gotn = inject_answers(rec_, K, override=(int(tnull), Sn)); s["serial_null"] = {"t": int(tnull), "hybrid": float((gotn == th).mean())}
        if K:                                                               # note 170-2 (descriptive)
            s["step_recover"], s["step_recover_info"] = step_recover(m, EXCL, TE, K, dev, ac, a.eval_bs)
        res["s170"] = s
        if a.aux_step and K:                                                # §175 (the aux is dropped at test; every §170 field comes from the main decoder)
            res["s175"] = s175_fields()
    res["eval_sec"] = round(time.time() - t1); json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):
        if os.path.exists(os.path.join(a.out, ck_)):
            os.remove(os.path.join(a.out, ck_))
    print("EVAL", json.dumps({"acc_by_n": res["acc_by_n"]}), flush=True); print("DONE", a.out, flush=True)


def serial_items(items, rng):
    """§170 SERIAL: recipients whose root has a bare-number operand a and whose innermost handle h (the first CoT reduction) is not the root; donor = h's
    operands changed (v -> v' != v) and a -> a' != a; hybrid = the recipient with h -> v' (a kept). Kept: hybrid != recipient and != donor answers."""
    rec, don, hyb = [], [], []
    for it in items:
        root = it["root"]
        if root.op is None:
            continue
        side = "l" if root.l.op is None else ("r" if root.r.op is None else None)
        path = _leftmost_ready(root)
        if side is None or path is None or len(path) == 0:
            continue
        got = None
        for _ in range(10):                                                 # a new value / operand can divide by 0 somewhere above: redraw
            d = _copy(root); hn = d
            for p_ in path:
                hn = getattr(hn, p_)
            v = apply(hn.op, hn.l.v, hn.r.v); v2 = int(rng.choice([x for x in range(P) if x != v])); prs = PAIRS[(hn.op, v2)]; a_, b_ = prs[int(rng.integers(0, len(prs)))]
            hb = _copy(root); hh = hb
            for p_ in path:
                hh = getattr(hh, p_)
            hh.l.v, hh.r.v = a_, b_; hn.l.v, hn.r.v = a_, b_; bare = getattr(d, side); bare.v = int(rng.choice([x for x in range(P) if x != bare.v]))
            try:
                got = (value(hb), value(d)); break
            except KeyError:
                continue
        if got is None:
            continue
        hyb_v, don_v = got
        if hyb_v != value(root) and hyb_v != don_v:
            rec.append(it); don.append({"syms": render(d), "t": don_v, "root": d, "n": it["n"]}); hyb.append(hyb_v)
    return rec, don, hyb


if __name__ == "__main__":
    main()
