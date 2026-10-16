"""VERSION-BANK (VB) + READY-RECODE on nested arithmetic (docs/THEORY_VERSIONBANK_DRAFT_2026-10-15.md + docs/research_passuse_2026-10-15/; Stage 0).
A NEW module: imports boxes.nested170 / boxes.deltamem read-only and changes no existing interface (§205 s1 / s2 import those).
Part 1 (this commit): the per-slot recode targets E_t and their re-render gate.
  Slots: x = [EOT] + render(root) + [" ="]; class per slot (20): VOID, EOT, " =", " (", " )", 4 ops, 11 values.
  E_t = the slot-aligned expression after t Feng reductions (nested170's leftmost-ready handle): the handle's span (its two number operands, its operator
  and its own brackets - only the reduced node's brackets can vanish, an ancestor's depend on the ancestor's operator) goes VOID and the span's LAST
  slot takes the value; E_t = E_n for t > n (HOLD).
  Gate (draft §9 Stage 0 item 5): dropping VOIDs from E_t re-renders exactly cot_steps(root)[t] inside [EOT .. " ="], for every t, on 100k items."""
import numpy as np
import torch
import torch.nn.functional as F

import boxes.nested170 as N

CLS = ["VOID", "EOT", " =", " (", " )"] + [" " + o for o in N.OPS] + [" %d" % v for v in range(N.P)]
CID = {c: i for i, c in enumerate(CLS)}; VOID = 0; NCLS = len(CLS)
assert NCLS == 20


def _render_owned(nd, parent=None, right=False):
    """nested170.render with ownership: -> [(symbol, node)] (brackets and the operator owned by the operator node, a number by its leaf)."""
    if nd.op is None:
        return [(" %d" % nd.v, nd)]
    s = _render_owned(nd.l, nd.op, False) + [(" " + nd.op, nd)] + _render_owned(nd.r, nd.op, True)
    if parent is not None and (N.PREC[nd.op] < N.PREC[parent] or (right and N.PREC[nd.op] == N.PREC[parent])):
        s = [(" (", nd)] + s + [(" )", nd)]
    return s


def recode_targets(root, t_max=None):
    """-> list E_0 .. E_{t_max} (t_max default n), each a list of class ids over the slots of [EOT] + render(root) + [" ="]; HOLD past n."""
    nd = N._copy(root); slots = [CID["EOT"]] + [CID[s] for s in N.render(nd)] + [CID[" ="]]; E = [list(slots)]
    while nd.op is not None:
        live = [j for j in range(1, len(slots) - 1) if slots[j] != VOID]; own = _render_owned(nd); assert len(own) == len(live), "VB: live slots != render"
        path = N._leftmost_ready(nd); h = nd
        for p_ in path:
            h = getattr(h, p_)
        span = [live[k] for k, (_, o) in enumerate(own) if o is h or o is h.l or o is h.r]
        assert span == list(range(span[0], span[-1] + 1)) or all(slots[j] == VOID for j in range(span[0], span[-1] + 1) if j not in span), "VB: the handle span is not contiguous over live slots"
        v = N.apply(h.op, h.l.v, h.r.v); h.v = v; h.op = h.l = h.r = None
        for j in span:
            slots[j] = VOID
        slots[span[-1]] = CID[" %d" % v]; E.append(list(slots))
    t_max = len(E) - 1 if t_max is None else t_max
    return [E[min(t, len(E) - 1)] for t in range(t_max + 1)]


def rerender(E_t):
    return [CLS[c] for c in E_t[1:-1] if c != VOID]


def gate_rerender(n_items=100_000, seed=20900, n_max=16):
    """every E_t (VOIDs dropped) == cot_steps(root)[t]; the frame slots stay; the slot count is constant; E_n's single live slot is the answer."""
    rng = np.random.default_rng(seed); bad = 0; n_steps = 0
    for _ in range(n_items):
        root, y = N.generate(rng, int(rng.integers(1, n_max + 1))); E = recode_targets(root); st = N.cot_steps(root); n_steps += len(E) - 1
        ok = len(E) == len(st) and all(rerender(e) == s for e, s in zip(E, st)) and all(len(e) == len(E[0]) and e[0] == CID["EOT"] and e[-1] == CID[" ="] for e in E)
        ok = ok and rerender(E[-1]) == [" %d" % y]
        bad += not ok
    return {"items": n_items, "steps": n_steps, "failures": bad, "pass": bad == 0}


# ------------------------------------------------------------------ Part 2: the float-mask bank block (deltamem._block is causal-only, REDTEAM F9c)
def bank_kv(g, i, V):
    """block i's own ln_1 + c_attn on stored versions V (B, T, D) -> (k, v) each (B, H, T, hd); WITH gradient (the bank cache must stay live, F9a)."""
    B, T, D = V.shape; hd = D // g.H; pre = f"h.{i}."
    a = F.layer_norm(V, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
    _, k, v = (a @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")).split(D, -1)
    return k.view(B, T, g.H, hd).transpose(1, 2), v.view(B, T, g.H, hd).transpose(1, 2)


def block_bank(g, i, h, bk, bv, bbias):
    """deltamem._block with a bank: the current version's queries attend over [bank keys (bk, bv: (B, H, Tb, hd)) | the current version, causal];
    bbias (B, H, T, Tb) float: the bank logit bias (beta + gamma * age), -inf on masked entries (padding, versions >= t). bk None = no bank.
    Bank-free when every bank entry is -inf (G1: equal to deltamem._block within fp64 rounding)."""
    B, T, D = h.shape; hd = D // g.H; pre = f"h.{i}."
    a = F.layer_norm(h, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
    q, k, v = (a @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")).split(D, -1)
    q, k, v = (t_.view(B, T, g.H, hd).transpose(1, 2) for t_ in (q, k, v))
    causal = torch.full((T, T), float("-inf"), device=h.device, dtype=q.dtype).triu(1)
    if bk is None:
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
    else:
        mask = torch.cat([bbias.to(q.dtype), causal.expand(B, g.H, T, T)], -1)
        o = F.scaled_dot_product_attention(q, torch.cat([bk.to(q.dtype), k], 2), torch.cat([bv.to(q.dtype), v], 2), attn_mask=mask)
    h = h + o.transpose(1, 2).reshape(B, T, D) @ g.w(pre + "attn.c_proj.weight") + g.w(pre + "attn.c_proj.bias")
    a = F.layer_norm(h, (D,), g.w(pre + "ln_2.weight"), g.w(pre + "ln_2.bias"))
    m = F.gelu(a @ g.w(pre + "mlp.c_fc.weight") + g.w(pre + "mlp.c_fc.bias"), approximate="tanh")
    return h + m @ g.w(pre + "mlp.c_proj.weight") + g.w(pre + "mlp.c_proj.bias")


def gate_g1(dev="cpu", B=4, T=13, Tb=26, seed=20901):
    """G1 (draft §9 item 2): fp64. (a) block_bank with every bank entry at -inf == deltamem._block (<= 1e-9 rel); (b) no bank == deltamem._block;
    (c) a finite bias moves the output (the bank is live); (d) d out / d bank K, V != 0 under a finite bias (F9a's dead-bank check at block level)."""
    import boxes.deltamem as DM
    from boxes import s5
    g = s5.load_gpt2(dev).double(); torch.manual_seed(seed); D = 768; out = {}
    h = torch.randn(B, T, D, dtype=torch.float64, device=dev); Vb = torch.randn(B, Tb, D, dtype=torch.float64, device=dev, requires_grad=True)
    ref = DM._block(g, 6, h, D // g.H); bk, bv = bank_kv(g, 6, Vb)
    ninf = torch.full((B, g.H, T, Tb), float("-inf"), dtype=torch.float64, device=dev)
    o_a = block_bank(g, 6, h, bk, bv, ninf); o_b = block_bank(g, 6, h, None, None, None)
    rel = lambda x, y: float((x - y).abs().max() / y.abs().max())  # noqa: E731
    out["masked_bank_eq_causal_rel"] = rel(o_a, ref); out["no_bank_eq_causal_rel"] = rel(o_b, ref)
    o_c = block_bank(g, 6, h, bk, bv, torch.full_like(ninf, -2.0)); out["finite_bias_moves_rel"] = rel(o_c, ref)
    o_c.sum().backward(); out["grad_bank_V_norm"] = float(Vb.grad.norm())
    out["pass"] = out["masked_bank_eq_causal_rel"] <= 1e-9 and out["no_bank_eq_causal_rel"] <= 1e-9 and out["finite_bias_moves_rel"] > 1e-6 and out["grad_bank_V_norm"] > 0
    return out


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "g1":
        print(json.dumps(gate_g1()))
    else:
        print(json.dumps(gate_rerender()))
