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
        n_ = int(rng.integers(1, n_max + 1)); r, t = generate(rng, n_); s = render(r)   # (the same draws as before; n_ kept for §177 (f)'s buckets)
        if tuple(s) not in excl:
            out.append({"syms": s, "t": t, "root": r, "n": n_})
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


PASS_TOKS = frozenset(pass_tok(t) for t in range(0, 65))      # §187: every pass-index token the aux prompt could carry


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
    def __init__(self, g, layer=6, write="top", h_add=6, h_del=6, add_norm=None):
        super().__init__(); import boxes.deltamem as DM; self.DM = DM; self.g = g; self.mem = DM.DeltaMem(g, layer, write, h_add, h_del, add_norm)
        self.nopass = False                                                 # §187 AUX-NOPASS: the aux prompt without the pass token (set from args)

    def aux_prompt(self, t):
        """§175's aux prompt [EOT, " Step", " t", " :"]; §187 (--aux_nopass): [EOT, " Step", " :"] - no pass token."""
        if self.nopass:
            pre = [EOT, STEP_ID, COLON_ID]; assert not any(tk in PASS_TOKS for tk in pre), "§187: a pass token in the aux prompt"
            return pre
        return [EOT, STEP_ID, pass_tok(t), COLON_ID]

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

    def aux_logits(self, S, items, t, dev, hold=False):
        """§175: the SAME trunk, read-only (fwd_read: the read on S, no write), over [EOT, " Step", " t", " :"] + e_{t+1} (teacher-forced). It sees no
        problem text and is never fed back. -> (logits (b, L, V) predicting e_{t+1} then EOT, targets (b, L), -100 = padding).
        §182 HOLD (hold=True): for a pass t > n the target is e_{n+1} (step n's target string, the final value)."""
        from boxes.deltas5 import fwd_read
        pre = self.aux_prompt(t); seqs, tg = [], []
        for it in items:
            n_ = len(aux_of(it)); assert t <= n_ or hold, "§175: no aux target past the last reduction"
            tgt = ids(aux_of(it)[min(t, n_) - 1][0]) + [EOT]; seqs.append(pre + tgt[:-1]); tg.append(tgt)
        x_, _ = pad(seqs); y_, _ = pad(tg, -100); x = torch.tensor(x_, device=dev); y = torch.tensor(y_, device=dev)
        assert x.shape[1] == len(pre) + y.shape[1] - 1
        h = fwd_read(self.g, self.g.embed(x), self.mem, S)
        return self.g.logits(h[:, len(pre) - 1:]).float(), y

    @torch.no_grad()
    def aux_greedy(self, S, t, L):
        """§175 eval: greedy L tokens after [EOT, " Step", " t", " :"] reading S -> (b, L) token ids."""
        from boxes.deltas5 import fwd_read
        pre = self.aux_prompt(t); x = torch.tensor([pre] * S.shape[0], device=S.device)
        for _ in range(L):
            nt = self.g.logits(fwd_read(self.g, self.g.embed(x), self.mem, S)[:, -1]).float().argmax(-1); x = torch.cat([x, nt[:, None]], 1)
        return x[:, len(pre):].cpu().numpy()


def kmatch_loss(m, items, kitem, dev, ac, masked=True):
    """§177 (f) (notes 177-4 / 177-5): the answer CE (mean over items) with item i thinking kitem[i] passes -> (loss, final S in item order).
    masked (the training form): ONE batch for K_max passes; after pass t each item's state is the new one only while t <= its K (a state-level select,
    Audit: the gated-delta decay / gamma / fade could move S on a pass with no token writes), so S is frozen at its own K and the gradient flows through
    the active branch only. bucketed (the reference): items grouped by K, each group run separately (§177 as first built)."""
    tgt = torch.tensor(NUM_IDS[[it["t"] for it in items]], device=dev)
    if masked:
        x, msk = m.think_inputs(items, dev); kv = torch.tensor(np.asarray(kitem), device=dev)
        with ac():
            S = m.mem.zero_state(len(items), dev)
            for t in range(1, int(max(kitem)) + 1):
                S = torch.where((kv >= t)[:, None, None, None], m.write(S, x, msk), S)
            lg = m.dec_logits(S, items, dev)
        return Fn.cross_entropy(lg, tgt), S
    tot, Sall = 0.0, [None] * len(items)
    for k_ in sorted(set(int(k) for k in kitem)):
        ix = [i for i, k in enumerate(kitem) if int(k) == k_]; sub = [items[i] for i in ix]; x, msk = m.think_inputs(sub, dev)
        with ac():
            S = m.think(m.mem.zero_state(len(sub), dev), x, msk, k_); lg = m.dec_logits(S, sub, dev)
        tot = tot + Fn.cross_entropy(lg, tgt[ix]) * (len(sub) / len(items))
        for r_, i in enumerate(ix):
            Sall[i] = S[r_]
    return tot, torch.stack(Sall)


AUX_W = (0.368, 0.582, 0.821, 1.131, 1.576, 2.301, 3.733, 8.0)            # §184 (c342e51c): w_t = E_bar / E_t, E_t = (1/8) sum_{n=t}^{8} 1/n
_E = [sum(1.0 / n for n in range(t, 9)) / 8 for t in range(1, 9)]
assert all(abs(w - sum(_E) / 8 / e) < 1e-3 for w, e in zip(AUX_W, _E)) and abs(sum(e * w for e, w in zip(_E, AUX_W)) - sum(_E)) < 1e-3, "§184: the weight table"


def kmatch_aux_loss(m, items, kitem, dev, ac, hold=False):
    """§182 WINDOW / HOLD: note 177-5's masked single batch (item i thinks kitem[i] passes; S frozen after its K) + §175's aux CE (token mean per pass,
    mean over each item's REAL steps t <= n, then over items: §175's weighting, identical in every cell). HOLD (Audit 8e20d709 review): the hold passes
    n < t <= K_i (target e_{n+1}) form a SEPARATE term hold_aux = mean over items of the mean over their hold passes (0 without hold), so the real-step
    weight is not diluted. -> (answer CE, aux, hold_aux, S); the caller adds lambda * (aux + hold_aux)."""
    tgt = torch.tensor(NUM_IDS[[it["t"] for it in items]], device=dev); x, msk = m.think_inputs(items, dev); kv = torch.tensor(np.asarray(kitem), device=dev)
    per = [[] for _ in items]; perh = [[] for _ in items]
    with ac():
        S = m.mem.zero_state(len(items), dev)
        for t in range(1, int(max(kitem)) + 1):
            S = torch.where((kv >= t)[:, None, None, None], m.write(S, x, msk), S)
            idx = [i for i, it in enumerate(items) if t <= len(aux_of(it)) or (hold and t <= int(kitem[i]))]
            if idx:
                lg_a, y_a = m.aux_logits(S[idx], [items[i] for i in idx], t, dev, hold=hold)
                ce = Fn.cross_entropy(lg_a.transpose(1, 2), y_a, ignore_index=-100, reduction="none").sum(1) / (y_a != -100).sum(1)
                for r_, i in enumerate(idx):
                    (per if t <= len(aux_of(items[i])) else perh)[i].append(ce[r_])
        lg = m.dec_logits(S, items, dev)
    aux = torch.stack([torch.stack(p_).mean() for p_ in per if p_]).mean()
    hold_aux = torch.stack([torch.stack(p_).mean() for p_ in perh if p_]).mean() if any(perh) else aux.new_zeros(())
    return Fn.cross_entropy(lg, tgt), aux, hold_aux, S


def ans_token_acc(lg, items):
    return (lg[:, NUM_IDS].argmax(-1).cpu().numpy() == np.array([it["t"] for it in items])), lg.argmax(-1).cpu().numpy()



def make_trunk(a, dev):
    """the GPT-2 trunk: pretrained GPT-2 small (§170), or for --scratch 'L,D,H' (§178) a RANDOM-INIT GPT-2 architecture (core.random_gpt2_sd under the
    run's seed; GPT-2 tokenizer / vocab / 1024 positions). The local memory cap is applied either way. a: the run's args (dict)."""
    from boxes import s5
    if not a.get("scratch"):
        return s5.load_gpt2(dev)
    from boxes import core as bc
    try:
        from boxes import memcap
        memcap.apply(dev)
    except ImportError:
        pass
    L, D, H = (int(x) for x in a["scratch"].split(",")); gen = torch.Generator().manual_seed(17800 + int(a.get("seed", 1)))
    return bc.GPT2KV(bc.random_gpt2_sd(L, D, H, gen), H).to(dev)


def mem_geometry(a):
    """(memory layer, additive heads, delta heads): §170 = (6, 6, 6); §178 scratch 'L,D,H' = (L // 2, D / 128, D / 128) (DH 64)."""
    if not a.get("scratch"):
        return 6, 6, 6
    L, D, _ = (int(x) for x in a["scratch"].split(",")); assert D % 128 == 0
    return L // 2, D // 128, D // 128


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
    from boxes.deltamem import DH
    H_ADD, H_DEL = mem.h_add, mem.h_del                                     # §178: the memory's own head counts
    with torch.autocast("cuda", enabled=False):
        b, D = x.shape; q = mem.qkv(mem.ln(x.float()))[..., :D].view(b, H_ADD + H_DEL, DH)
        from boxes.deltamem import add_read, state_z                      # §185: the additive read as the run's own (den / rms); delta slice explicit
        oa = torch.exp(mem.s_add) * add_read(mem, F_.elu(q[:, :H_ADD]) + 1, S[:, :H_ADD].float(), state_z(mem, S))
        od = torch.exp(mem.s_del) * torch.einsum("bhvk,bhk->bhv", S[:, H_ADD:H_ADD + H_DEL].float(), F_.normalize(q[:, H_ADD:], dim=-1))
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


def step_recover(m, excl, TE, T, dev, ac, bs, n_fit=int(os.environ.get("NESTED170_PROBE_N_FIT", 4000)), n_test=1000):   # env: gate-size probe fits only
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
    ap.add_argument("--eval_at", type=int, default=0, help="also evaluate acc_by_n at this step -> eval_<k>k.json"); ap.add_argument("--aux_step", type=float, default=0.0, help="§175: lambda of the read-only step decoder CE (0 = off: the §170 path)"); ap.add_argument("--aux_fade", default="36000,48000", help="§175: lambda constant to the first step, then linear to 0 at the second"); ap.add_argument("--kmatch_impl", default="masked", choices=["masked", "bucketed"], help="§177 (f): note 177-5's masked single batch, or the per-K groups"); ap.add_argument("--add_norm", default="", choices=["", "den", "rms"], help="§185 ADD-NORM: the additive reads normalised by phi(q)^T z (den) or per-head RMS (rms); '' = off"); ap.add_argument("--aux_nopass", action="store_true", help="§187 AUX-NOPASS: the aux prompt [EOT, ' Step', ' :'] (no pass token) in training and every aux eval"); ap.add_argument("--aux_reweight", action="store_true", help="§184: the aux CE of pass t <= n weighted by AUX_W[t-1] (equal expected weight per pass position)"); ap.add_argument("--aux_hold", action="store_true", help="§182 HOLD: aux target e_{n+1} on passes n < t <= K (with --k_window and --aux_step)"); ap.add_argument("--kprimes", default="", help="§182: the K' set for s170.acc_by_n_kprime ('' = {0, 1, K/2, K, 2K})"); ap.add_argument("--path_indep", action="store_true", help="§182: s182.path_indep (K' = 16 from M_0 = 0 vs another item's 8-pass memory)"); ap.add_argument("--excl_min_n", type=int, default=1, help="note 178-3 (s178b): exclude test items from training only for n >= this (1 = §170's protocol)"); ap.add_argument("--scratch", default="", help="§178: 'L,D,H' -> a RANDOM-INIT GPT-2 trunk (no pretrained weights), memory after block L // 2 with D / 128 + D / 128 heads"); ap.add_argument("--warmup", type=int, default=100, help="LR warm-up steps (§178: 1000)"); ap.add_argument("--k_window", default="", help="§177 (f): 'lo,hi' -> each n-bucketed micro-batch runs K = n + U{lo..hi} think passes (inject)"); ap.add_argument("--fast_add", action="store_true", help="deltamem.FAST_ADD: the additive heads' pass writes as one sum (Audit 2026-10-09; scratch/gate_fastadd.py)"); ap.add_argument("--sched_scale", type=float, default=1.0, help="smokes only"); ap.add_argument("--fp32", action="store_true"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); a.task = "nested"; import boxes.deltamem as DM_; DM_.FAST_ADD = bool(a.fast_add); os.makedirs(a.out, exist_ok=True); dev = a.device; sc = a.sched_scale; K = a.think_k if a.arm == "inject" else 0
    if a.arm != "inject":
        a.mem_write = a.write_mix_start = a.write_mix_end = None; a.think_k = 0
    from boxes import s5
    from boxes import core as bc
    from boxes.fsq120 import make_opt, lr_at, driver_version, env_info
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    amp = None if (a.fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    ac = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
    TE = test_set(a.per_n); EXCL = {tuple(it["syms"]) for it in TE if it["n"] >= a.excl_min_n}; g = make_trunk(vars(a), dev); ML, HA, HD = mem_geometry(vars(a))   # note 178-3: excl_min_n
    r_sp = np.random.default_rng(17300); SPOOL = []                         # note 170-1: the dedicated n = 8 SERIAL pool (excluded from training)
    for _ in range(a.serial_pool):
        r__, t__ = generate(r_sp, 8); s__ = render(r__); SPOOL.append({"n": 8, "syms": s__, "t": t__, "depth": depth(s__), "root": r__})
    EXCL |= {tuple(it["syms"]) for it in SPOOL}
    r_ex = np.random.default_rng(17899)                                     # note 178-3: the share of training draws surviving the exclusion, per n (own stream)
    a.excl_survive_by_n = {str(n_): round(float(np.mean([tuple(render(generate(r_ex, n_)[0])) not in EXCL for _ in range(2000)])), 4) for n_ in range(1, a.n_max_train + 1)}
    m = InjectN(g, ML, a.mem_write, HA, HD, a.add_norm or None).to(dev) if a.arm == "inject" else None
    if m is not None:
        m.nopass = bool(a.aux_nopass)                                       # §187

    def loss_items(items, Kb=None):
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
                S = m.think(m.mem.zero_state(len(items), dev), x, msk, K if Kb is None else Kb, ckpt=a.ckpt_passes); lg = m.dec_logits(S, items, dev)
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
                        per[i].append(ce[r_] * AUX_W[t - 1] if a.aux_reweight else ce[r_])   # §184: w_t inside §175's per-item mean
            lg = m.dec_logits(S, items, dev)
        la = Fn.cross_entropy(lg, torch.tensor(NUM_IDS[[it["t"] for it in items]], device=dev))
        aux = torch.stack([torch.stack(p_).mean() for p_ in per if p_]).mean(); aux_hist.append((float(la), float(aux)))
        return la + lam_now[0] * aux

    def mem_check(micro):
        torch.cuda.reset_peak_memory_stats(dev); r_ = np.random.default_rng(0); its = [it for it in train_items(r_, 4 * a.bs, a.n_max_train, set()) if True]
        its = sorted(its, key=lambda it: -len(it["syms"]))[:a.bs // micro]
        if KW:                                                              # §177 (f) (Audit): the path that trains - the masked kmatch loss, every item at K_max
            if lam_now[0]:                                                  # §182: the aux on the masked kmatch path (every item at K_max)
                la_, ax_, hx_, _ = kmatch_aux_loss(m, its, [a.n_max_train + KW[1]] * len(its), dev, ac, a.aux_hold); l_ = la_ + lam_now[0] * (ax_ + hx_)
            else:
                l_, _ = kmatch_loss(m, its, [a.n_max_train + KW[1]] * len(its), dev, ac, masked=a.kmatch_impl == "masked")
        else:
            l_ = loss_items(its)
        l_.backward(); (m or g).zero_grad(set_to_none=True)
        evaluate_n(sorted([it for it in TE if it["n"] == 16], key=lambda it: -len(it["syms"]))[:a.eval_bs], max(2 * K, 20 if KW else 0))   # n = 16 at K' = 2K ((f): >= n + 4)
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
    KW = [int(x_) for x_ in a.k_window.split(",")] if a.k_window else None; rng_kw = np.random.default_rng(17900 + a.seed); kcount = {}; loss_log = []
    if KW:
        assert a.arm == "inject" and K >= a.n_max_train + KW[1], "§177 (f): --think_k must cover the largest bucket's K (memcheck / calibration)"

    def lam_at(step):
        """§175: lambda = aux_step to f0, then linear to 0 at f1 (sched_scale-scaled)."""
        if not a.aux_step:
            return 0.0
        return a.aux_step if step <= f0_ * sc else a.aux_step * max(0.0, (f1_ * sc - step) / max(1.0, (f1_ - f0_) * sc))

    @torch.no_grad()
    def aux_hold_eval(items, Kp):
        """§182 HOLD: after each pass n < t <= Kp, greedy exact match of e_{n+1} + EOT (step n's target) -> ({t - n: acc}, overall, {t - n: count})."""
        ex = {}
        for i in range(0, len(items), a.eval_bs):
            ch = items[i:i + a.eval_bs]; x, msk = m.think_inputs(ch, dev)
            with ac():
                S = m.mem.zero_state(len(ch), dev)
                for t in range(1, Kp + 1):
                    S = m.write(S, x, msk); idx = [j for j, it in enumerate(ch) if len(aux_of(it)) < t]
                    if not idx:
                        continue
                    sub = [ch[j] for j in idx]; tg_ = [ids(aux_of(it)[-1][0]) + [EOT] for it in sub]; gen = m.aux_greedy(S[idx], t, max(len(q) for q in tg_))
                    for r_, it in enumerate(sub):
                        ex.setdefault(t - len(aux_of(it)), []).append(bool(list(gen[r_, :len(tg_[r_])]) == tg_[r_]))
        allv = [v for vs in ex.values() for v in vs]
        return {str(k_): float(np.mean(v)) for k_, v in sorted(ex.items())}, (float(np.mean(allv)) if allv else None), {str(k_): len(v) for k_, v in sorted(ex.items())}

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
        out = {"aux_acc_by_pass": acc8, "aux_acc_by_pass_ext": acc12, "aux_acc_by_role": role8, "step_recover": sr, "step_recover_info": sri,
               "n_by_pass": n8, "n_by_pass_ext": n12, "ext_kprime": 16, "items": {"n_le_8": {"n": len(te8), "seed": 17022}, "n_9_12": {"n": len(te12), "seed": 17023}}}
        if a.aux_hold:                                                      # §182 HOLD: the hold passes n < t <= K on the same n <= 8 items
            hb, ha, hn = aux_hold_eval(te8, K); out.update({"aux_acc_hold_by_offset": hb, "aux_acc_hold": ha, "n_hold_by_offset": hn})
        return out

    def kn_answers(items, kfun, skip=()):
        """§177 (f): inject answers with a per-n K' = kfun(n) (items grouped by n, order kept)."""
        out = np.full(len(items), -1)
        for n_ in sorted({it["n"] for it in items}):
            ix = [i for i, it in enumerate(items) if it["n"] == n_]; out[ix] = inject_answers([items[i] for i in ix], kfun(n_), skip=tuple(t for t in skip if t <= kfun(n_)))
        return out

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
            try:                                                            # 2026-10-09: an OOM inside the check (a hard memory cap) = over budget -> larger micro
                pk = mem_check(a.micro)
            except torch.OutOfMemoryError:
                pk = float("inf")
            if pk != float("inf"):                                       # + AdamW's two fp32 states per parameter (not allocated by the check; ~1 GiB for GPT-2)
                pk += 2 * sum(p_.numel() * 4 for p_ in (m if m is not None else g).parameters() if p_.requires_grad) / 2 ** 30
            torch.cuda.empty_cache()                                    # no fragmentation carried into the next attempt / training
            if pk == float("inf"):                                       # after the except clause, so the failed graph is released
                (m or g).zero_grad(set_to_none=True); torch.cuda.empty_cache()
            mc.append({"micro": a.micro, "peak_gib": pk}); print("MEM", a.micro, pk, flush=True)
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
    params = m if m is not None else g; opt = make_opt(params, a.lr); step0 = 1; n_nonfinite = 0; hist = []; wg_hist = []; wwin = []; wlast = []; gn_hist = []; gadd_hist = []; zmin_hist = []
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
        sec_ = [r_["sec"] for r_ in lg_ if "sec" in r_]; t0 = time.time() - (sec_[-1] if sec_ else 0)
        wg_hist = [{"step": r_["step"], "mean": r_["write_grad_norm"], "spike_100x_floor": bool(r_.get("write_grad_spike_100x_floor", False)), "from_log": True}
                   for r_ in lg_ if "write_grad_norm" in r_]                 # 2026-10-10: log records -> the training wg_hist form (the eval reads spike_100x_floor)
    for step in range(step0, last + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(a.lr, gr, step, a.steps) if a.warmup == 100 else lr_at(a.lr, gr, step, a.steps) / min(1, step / 100) * min(1, step / a.warmup)   # §178: --warmup
        params.train()
        if m is not None:
            m.mem.fade.fill_(min(1.0, step / max(1, int(500 * sc))))
            m.mem.gamma.fill_(0.0 if step <= a.write_mix_start * sc else min(1.0, (step - a.write_mix_start * sc) / max(1.0, (a.write_mix_end - a.write_mix_start) * sc)))
        lam_now[0] = lam_at(step)
        if a.add_norm == "den" and m is not None and step == step0 and step <= 500:   # Audit: arm the phi(q)^T z recorder from the first step
            m.mem.zmin_rec = []
        its = train_items(rng, a.bs, a.n_max_train, EXCL); opt.zero_grad(); tot = 0.0; ch = a.bs // a.micro
        if KW:                                                              # §177 (f): the same items / n marginal as §170, bucketed by n, one K per bucket
            assert not lam_now[0] or a.kmatch_impl == "masked", "§182: the aux rides on the masked kmatch batch"
            kb_ = {}
            for n_ in sorted({it["n"] for it in its}):                     # one K per n-bucket per step, drawn in the same order / stream as first built
                bk = [it for it in its if it["n"] == n_]; Kb = n_ + int(rng_kw.integers(KW[0], KW[1] + 1)); kb_[n_] = Kb
                assert all(it["n"] == n_ for it in bk) and KW[0] <= Kb - n_ <= KW[1]; kcount.setdefault(str(n_), {}); kcount[str(n_)][str(Kb)] = kcount[str(n_)].get(str(Kb), 0) + 1
            kit = [kb_[it["n"]] for it in its]                              # note 177-5: the masked single batch (bucketed = --kmatch_impl bucketed)
            for c_ in range(0, len(its), ch):
                sub, ks = its[c_:c_ + ch], kit[c_:c_ + ch]
                if lam_now[0]:                                              # §182 WINDOW / HOLD
                    la_, ax_, hx_, _ = kmatch_aux_loss(m, sub, ks, dev, ac, a.aux_hold); aux_hist.append((float(la_), float(ax_), float(hx_))); lk = la_ + lam_now[0] * (ax_ + hx_)
                else:
                    lk, _ = kmatch_loss(m, sub, ks, dev, ac, masked=a.kmatch_impl == "masked")
                (lk * len(sub) / a.bs).backward(); tot += float(lk.item()) * len(sub) / a.bs
        else:
          for k_ in range(a.micro):
            lk = loss_items(its[k_ * ch:(k_ + 1) * ch]); (lk / a.micro).backward(); tot += float(lk.item()) / a.micro
        if m is not None:
            from boxes.deltas5 import write_grad_norm
            wwin.append(write_grad_norm(m))
        gn = torch.nn.utils.clip_grad_norm_(params.parameters(), 1.0)
        if a.aux_reweight or a.add_norm:                                    # §184 / §185: the pre-clip grad norm per step (clip fires above 1.0)
            gn_hist.append(float(gn))
        if a.add_norm == "den" and m is not None and step <= 500:           # Audit: min phi(q)^T z over this step's reads, steps 1..500
            if m.mem.zmin_rec:
                zmin_hist.append((step, min(m.mem.zmin_rec)))
            m.mem.zmin_rec = [] if step < 500 else None
        if a.add_norm and m is not None:                                    # §185: the additive path's grad norm (the additive heads' q / k / v rows + the rms gain)
            gq = m.mem.qkv.weight.grad
            if gq is not None:
                ra = [r for blk in range(3) for r in range(blk * m.mem.D, blk * m.mem.D + m.mem.h_add * m.DM.DH)]; ga = gq[ra].float().pow(2).sum()
                if getattr(m.mem, "add_gain", None) is not None and m.mem.add_gain.grad is not None:
                    ga = ga + m.mem.add_gain.grad.float().pow(2).sum()
                gadd_hist.append((step, float(ga.sqrt())))
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
            rec = {"step": step, "loss": float(np.mean(hist)) if hist else None, "sec": round(time.time() - t0)}; hist = []; loss_log.append(rec["loss"])
            if a.aux_step:
                rec.update({"aux_lambda": lam_now[0], "answer_ce": float(np.mean([x_[0] for x_ in aux_hist])) if aux_hist else None,
                            "aux_ce": float(np.mean([x_[1] for x_ in aux_hist])) if aux_hist else None})
                if a.aux_hold:                                              # §182 HOLD: the separate hold term
                    rec["hold_aux_ce"] = float(np.mean([x_[2] for x_ in aux_hist])) if aux_hist else None
                aux_hist.clear()
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
    elif KW:                                                                # §177 (f): the main eval at K' = n + 1 per item
        A_ = kn_answers(TE, lambda n_: n_ + 1); NT = None
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
            for _ in range(n + 1 if KW else K):                             # §177 (f): K' = n + 1 per item
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
        for kp in sorted({int(x_) for x_ in a.kprimes.split(",")} | {K} if a.kprimes else {0, 1, K // 2, K, 2 * K}):   # §182: --kprimes
            Ak = A_ if kp == K else inject_answers(TE, kp); s["acc_by_n_kprime"][str(kp)] = {str(n): float((Ak == T_)[N == n].mean()) for n in range(1, 17)}
        if a.kprimes:                                                       # note 182-3 (§182 cells): n-relative K' = n and n + 2 (per-n readout, as (f)'s kn_answers)
            for kk_, f_ in (("n", lambda n_: n_), ("n+2", lambda n_: n_ + 2)):
                Ak = kn_answers(TE, f_); s["acc_by_n_kprime"][kk_] = {str(n): float((Ak == T_)[N == n].mean()) for n in range(1, 17)}
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
        if KW:                                                              # §177 (f) fields
            kfs = {"0": lambda n_: 0, "1": lambda n_: 1, "n": lambda n_: n_, "n+1": lambda n_: n_ + 1, "n+2": lambda n_: n_ + 2, "n+4": lambda n_: n_ + 4}
            s7 = {"acc_by_n_kprime": {}, "skip_pass": {}, "kmatch_counts": kcount, "k_window": a.k_window, "rule_kprime": "n+1"}
            for kk_, f_ in kfs.items():
                Ak = A_ if kk_ == "n+1" else kn_answers(TE, f_); s7["acc_by_n_kprime"][kk_] = {str(n): float((Ak == T_)[N == n].mean()) for n in range(1, 17)}
            for t in range(1, a.n_max_train + 2):                           # skip pass t at K' = n + 1, for the n with t <= n + 1
                its_ = [it for it in TE if it["n"] + 1 >= t]; Ak = kn_answers(its_, lambda n_: n_ + 1, skip=(t,)); Nk = np.array([it["n"] for it in its_])
                s7["skip_pass"][str(t)] = {str(n): float((Ak == np.array([it["t"] for it in its_]))[Nk == n].mean()) for n in sorted(set(Nk.tolist()))}
            ll = []                                                         # note 177-4: the whole log.jsonl history (survives --resume), not the in-memory list
            for l_ in open(os.path.join(a.out, "log.jsonl")):
                r_ = json.loads(l_)
                if r_.get("loss") is not None and "step" in r_:
                    ll.append(r_["loss"])
            tl = ll[-max(2, len(ll) // 10):]; h_ = len(tl) // 2
            s7["train_loss_tail"] = {"first_half": float(np.mean(tl[:h_])) if h_ else None, "second_half": float(np.mean(tl[h_:])) if tl else None, "n_logged": len(tl)}
            res["s177"] = s7; json.dump(kcount, open(os.path.join(a.out, "kmatch_counts.json"), "w", newline="\n"), indent=1)
        if a.path_indep and a.arm == "inject":                              # §182 s182.path_indep: K' = 16 from M_0 = 0 vs from another item's 8-pass memory
            pool = [it for it in TE if 3 <= it["n"] <= 8]; its_ = [pool[i] for i in np.random.default_rng(18201).permutation(len(pool))[:500]]
            der = np.roll(np.arange(len(its_)), 1); S8 = inject_states(its_, 8)[der]; A0, A1, F0, F1 = [], [], [], []   # a fixed derangement (no fixed point)
            with torch.no_grad():
                for i in range(0, len(its_), a.eval_bs):
                    ch = its_[i:i + a.eval_bs]; x, msk = m.think_inputs(ch, dev)
                    for S0, Ao, Fo in ((None, A0, F0), (S8[i:i + a.eval_bs], A1, F1)):
                        with ac():
                            Sz = m.mem.zero_state(len(ch), dev); S = Sz if S0 is None else torch.tensor(S0, device=dev, dtype=Sz.dtype)
                            for _ in range(16):
                                S = m.write(S, x, msk)
                            lg = m.dec_logits(S, ch, dev)
                        Ao.append(lg[:, NUM_IDS].argmax(-1).cpu().numpy()); Fo.append(S[:, m.mem.h_add:m.mem.h_add + m.mem.h_del].float().flatten(1).cpu())
            A0, A1 = np.concatenate(A0), np.concatenate(A1); F0, F1 = torch.cat(F0), torch.cat(F1); tt_ = np.array([it["t"] for it in its_])
            res["s182"] = {"path_indep": {"answer_agree": float((A0 == A1).mean()), "delta_cos_final": float(Fn.cosine_similarity(F0, F1, dim=1).mean()),
                                          "n_items": len(its_), "acc_from_zero": float((A0 == tt_).mean()), "acc_from_donor": float((A1 == tt_).mean()), "kprime": 16,
                                          "donor_passes": 8, "items": "test n 3..8, rng 18201 permutation[:500]", "derangement": "np.roll(sample order, 1)",
                                          "delta_cos_def": "mean over items of cos of the final delta-head states S[:, h_add:h_add + h_del]"}, "aux_hold": bool(a.aux_hold), "k_window": a.k_window}
        if a.add_norm and m is not None:                                    # §185 fields
            gh_ = np.array(gn_hist) if gn_hist else np.array([np.nan]); gb = {}
            for st_, v_ in gadd_hist:
                gb.setdefault(str((st_ - 1) // 1000 * 1000 + 1000), []).append(v_)
            rn = {}; its_ = [it for it in TE if 3 <= it["n"] <= 8][:256]; x_, mk_ = m.think_inputs(its_, dev)
            with torch.no_grad(), ac():
                S_ = m.mem.zero_state(len(its_), dev); m.mem.norms = []
                for t_ in range(1, 17):
                    S_ = m.write(S_, x_, mk_); rn[str(t_)] = float(m.mem.norms[-1][1].median()) if m.mem.norms else None
                m.mem.norms = None
            res["s185"] = {"add_norm": a.add_norm, "clip_rate": float((gh_ > 1.0).mean()), "grad_norm_preclip": {"median": float(np.median(gh_)), "p90": float(np.percentile(gh_, 90))},
                           "grad_add": {k_: {"median": float(np.median(v_)), "p99": float(np.percentile(v_, 99))} for k_, v_ in sorted(gb.items(), key=lambda kv: int(kv[0]))},
                           "den_min_first500": ({"min": min(v_ for _, v_ in zmin_hist), "by_100": {str(b_): min(v_ for s_, v_ in zmin_hist if (s_ - 1) // 100 == b_ // 100 - 1)
                                                for b_ in range(100, 501, 100) if any((s_ - 1) // 100 == b_ // 100 - 1 for s_, _ in zmin_hist)}, "n_steps": len(zmin_hist)}
                                                if zmin_hist else None),
                           "read_norm_by_pass": rn, "read_norm_def": "median ||g_add * additive read|| at each think pass's last token (mem.norms), 256 test items n 3..8",
                           "grad_add_def": "per step the norm of the additive heads' q / k / v rows of qkv.weight.grad (+ the rms gain), median / p99 per 1k steps"}
        if a.aux_reweight:                                                  # §184 fields
            gh_ = np.array(gn_hist) if gn_hist else np.array([np.nan])
            res["s184"] = {"aux_reweight": True, "weights": list(AUX_W), "clip_rate": float((gh_ > 1.0).mean()), "grad_norm_preclip": {"median": float(np.median(gh_)),
                           "p90": float(np.percentile(gh_, 90)), "n_steps": int(len(gn_hist))}, "def": "clip_grad_norm_(..., 1.0)'s pre-clip total norm per training step; clip fires when > 1.0"}
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
