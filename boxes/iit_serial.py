"""IIT-SERIAL (docs/THEORY_IITSERIAL_DRAFT_2026-10-14.md, 271db224 + addenda 119fc3c8 / e8d3835a): the interchange-intervention training term for
nested170's inject arm, called from nested170's step loop only when --iit is set and gamma(step) > 0 (Prop O': zero through step 2000, so a cell
is bitwise-paired with its control through 2000). A NEW module: the default path never imports it.
Pairing (draft §2.1, from the step's own items in order; own rng seed + 20500, no training-rng draw):
  eligible x: (e1) the root has a bare-number child a (the late leaf); (e2) the first leftmost-ready handle h is not the root; (e3) height(root) >=
  2 tau + 1; (--iit_nmax, Audit's n-restricted variant: n <= nmax; 0 = off).
  IIT donor x_d: h's operands redrawn by the generator's conditional at v' != v_1 (uniform), a -> a' != a; up to 10 redraws on a division by 0.
  y = F(x), y_hyb = F(x with h's operands := the donor's), y_d = F(x_d); kept iff y_hyb not in {y, y_d}.
  SIIT text x': x with h's operands redrawn at v'' != v_1, a kept; kept iff y(x') is defined and != y.
  B_I = min(B_I, #recipients kept for BOTH) - the same recipients serve IIT and SIIT.
Terms (draft §2.2; the swap site is the whole state after pass tau, all heads):
  IIT : S_tau^d = tau passes on x_d (with grad) -> passes tau+1..K on x -> CE(dec_logits(x), y_hyb)   (full-vocabulary CE at ' =', as the answer CE)
  SIIT: S_tau  = tau passes on x   (with grad) -> passes tau+1..K on x' -> CE(dec_logits(x'), y)
  Both run as ONE graph of 2 B_I rows (rows 0..B_I-1 IIT, B_I.. SIIT); loss = gamma * (w_I L_IIT + w_S L_SIIT); the caller backward()s it after the
  main backward (gradients accumulate; the existing clip applies to the sum). No @torch.no_grad on this path (gradient-flow rule; asserted by ast).
Lengths: a redraw changes only leaf VALUES (h's two operands, the bare leaf); every value 0..10 renders as ONE token (nested170 asserts each VOCAB
symbol is a single GPT-2 token) and the operators / brackets are untouched, so x_d, x' and the hybrid render to x's length by construction (the
shape assert in iit_loss cannot fire; G-HYB 0 / 10,029). nested170 logs write_grad_norm AFTER the IIT backward (from gamma > 0 it includes IIT
gradients); write_grad_norm_pre_iit is the main loss's alone, comparable to the controls."""
import numpy as np
import torch
import torch.nn.functional as Fn

import boxes.nested170 as N


def _height(nd):
    return 0 if nd.op is None else 1 + max(_height(nd.l), _height(nd.r))


def _redraw_h(root, path, v_excl, rng):
    """a copy of root with h's operands redrawn at a value != v_excl (Alg. 1's conditional) -> (copy, the new value)."""
    d = N._copy(root); hn = d
    for p_ in path:
        hn = getattr(hn, p_)
    v2 = int(rng.choice([x for x in range(N.P) if x != v_excl])); prs = N.PAIRS[(hn.op, v2)]; a_, b_ = prs[int(rng.integers(0, len(prs)))]
    hn.l.v, hn.r.v = a_, b_; return d, v2


def make_pairs(items, rng, tau=1, nmax=0, BI=16, hyb_out=None):
    """-> (recipients, IIT donors, y_hyb, SIIT texts, y) for up to BI recipients kept for both terms; plus counts. hyb_out: list -> the hybrid trees
    (G-HYB gate only)."""
    rec, don, yh, sx, yy = [], [], [], [], []; cnt = {"seen": 0, "eligible": 0, "kept": 0}
    for it in items:
        if len(rec) >= BI:
            break
        cnt["seen"] += 1; root = it["root"]
        if nmax and it["n"] > nmax:
            continue
        if root.op is None:
            continue
        side = "l" if root.l.op is None else ("r" if root.r.op is None else None); path = N._leftmost_ready(root)
        if side is None or path is None or len(path) == 0 or _height(root) < 2 * tau + 1:
            continue
        cnt["eligible"] += 1; hn = root
        for p_ in path:
            hn = getattr(hn, p_)
        v1 = N.apply(hn.op, hn.l.v, hn.r.v); y = N.value(root); got = None
        for _ in range(10):
            d, _ = _redraw_h(root, path, v1, rng); hb = N._copy(d); bare = getattr(d, side); bare.v = int(rng.choice([x for x in range(N.P) if x != bare.v]))
            setattr(hb, side, N.Node(v=getattr(root, side).v))        # the hybrid keeps x's late leaf (h's operands from the donor)
            try:
                got = (N.value(hb), N.value(d), d); break
            except KeyError:
                continue
        if got is None:
            continue
        y_hyb, y_d, d = got
        if y_hyb in (y, y_d):
            continue
        s_ok = None
        for _ in range(10):
            x2, _ = _redraw_h(root, path, v1, rng)
            try:
                yx2 = N.value(x2)
            except KeyError:
                continue
            s_ok = (x2, yx2); break
        if s_ok is None or s_ok[1] == y:
            continue
        rec.append(it); don.append({"syms": N.render(d), "root": d, "n": it["n"], "t": y_d}); yh.append(y_hyb)
        if hyb_out is not None:
            hb = N._copy(d); setattr(hb, side, N.Node(v=getattr(root, side).v)); hyb_out.append(hb)
        sx.append({"syms": N.render(s_ok[0]), "root": s_ok[0], "n": it["n"], "t": s_ok[1]}); yy.append(y); cnt["kept"] += 1
    return rec, don, np.array(yh, dtype=int), sx, np.array(yy, dtype=int), cnt


def iit_loss(m, items, dev, ac, K, rng, tau=1, nmax=0, BI=16, w_I=1.0, w_S=0.5, rec_S=None):
    """-> (loss tensor or None, log dict). rec_S: if a list, the S_tau tensor (retain_grad) is appended (gradflow gate)."""
    rec, don, yh, sx, yy, cnt = make_pairs(items, rng, tau, nmax, BI)
    if not rec:
        return None, {"iit_pairs": 0, **cnt}
    B = len(rec); ph1 = don + rec; ph2 = rec + sx                           # pass 1..tau inputs: [x_d ; x]; passes tau+1..K and the decoder: [x ; x']
    x1, m1 = m.think_inputs(ph1, dev); x2, m2 = m.think_inputs(ph2, dev)
    assert x1.shape == x2.shape, "IIT-SERIAL: the donor / SIIT texts must render to the recipient's length"
    with ac():
        S = m.mem.zero_state(2 * B, dev)
        for _ in range(tau):
            S = m.write(S, x1, m1)
        if rec_S is not None:
            S.retain_grad(); rec_S.append(S)
        for _ in range(tau + 1, K + 1):
            S = m.write(S, x2, m2)
        lg = m.dec_logits(S, ph2, dev)
    tg = torch.tensor(N.NUM_IDS[np.concatenate([yh, yy])], device=dev)
    l_i = Fn.cross_entropy(lg[:B], tg[:B]); l_s = Fn.cross_entropy(lg[B:], tg[B:])
    return w_I * l_i + w_S * l_s, {"iit_pairs": B, "iit_ce": float(l_i.detach()), "siit_ce": float(l_s.detach()), **cnt}
