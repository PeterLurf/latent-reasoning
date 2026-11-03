"""THEORY §262 NAME-KEYED MEMORY (docs/research_namekeyed_2026-11-03/NK_v1.md §2): one name-keyed head beside DeltaMem, packed into the carried state S as one
extra block (S[:, NB], NB = h_add + h_del; the §185 pack_state precedent), so every consumer of S (state_n / the hold penalty / norm guard, swaps, means) sees it.
deltamem.py / deltas5.py are untouched: InjectNK hands DM.read_fixed / DM.write_seq the S[:, :NB] slice (they read only S[:, :h_add] and S[:, h_add:NB] anyway)
and runs its own read / write on S[:, NB] with the token ids it already holds.
  keys     tied: k = q = normalize(W_n e(name)), e = the trunk's wte row of a name token; one head, DHN 64; the 40 single-token names of nested_words.NAMES
           (NAME_IDS; inserted side names are drawn from the same list).
  write    every think pass: key = the pass cue's name token (QUESTION's A: the token 6 before ':'; tok_aw asserts it == spine[min(t, d)]); value
           v = W_vn ln_w(h_11 at ':') (mem.ln_w, the AW write's own LN); delta rule, beta 1: S_n <- S_n - (S_n k) k^T + v k^T (erase-then-write: S_n k = v after,
           idempotent across hold passes).
  read     entering block mem.layer (6), after DeltaMem's read, at every position whose token is a name: h_p <- h_p + W_on (S_n q_p); think passes, decoder and
           aux (fwd_read_nk). W_on = 0 at init, NO fade buffer: the model is function-preserving at step 0 and W_on still gets its gradient.
  ids      every NK read asserts ids (B, T) == the hidden's (B, T); every NK write asserts the key is a name token.
  ablate   nk.ablate = True zeroes the read r = S_n q (W_on r -> 0; eval only).
  init     NK params from their own generator (seed + SEED_NK_INIT), so the trunk / torch RNG stream of the run is the CTRL arm's.
part_nk_pilot (NK_v1 §4): cond_ws_perm[d][t] = P(pass-t WS correct | pass t-1 correct), d 3-4, t 2..d, on TEST-PERM (perm_render.test_perm_sets); ws1_perm[d] = pass-1
WS; NK runs also cond_ws_perm_ablated (nk.ablate)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import boxes.deltamem as DM
import boxes.nested_words as NW
from boxes.inject_aw import COLON, InjectAW
from boxes.inject_ow import ENC
from boxes.nested170 import pad

DHN, SEED_NK_INIT, CUE_NAME_OFF = 64, 26200, 6                              # CUE_NAME_OFF: ':' - 6 = QUESTION's A (A, ' have', '?', ' The', ' answer', ' is', ':')
NAME_IDS = tuple(ENC.encode(" " + n)[0] for n in NW.NAMES)
assert all(len(ENC.encode(" " + n)) == 1 for n in NW.NAMES) and len(set(NAME_IDS)) == len(NW.NAMES) == 40, "NK: 40 single-token names"
NAME_TOK = dict(zip(NW.NAMES, NAME_IDS))


class NK(nn.Module):
    def __init__(self, D, dev, seed):
        super().__init__()
        gen = torch.Generator().manual_seed(seed + SEED_NK_INIT)
        self.W_n = nn.Linear(D, DHN, bias=False); self.W_vn = nn.Linear(D, DHN, bias=False); self.W_on = nn.Linear(DHN, D, bias=False)
        with torch.no_grad():
            self.W_n.weight.copy_(torch.randn(DHN, D, generator=gen) * 0.02); self.W_vn.weight.copy_(torch.randn(DHN, D, generator=gen) * 0.02); self.W_on.weight.zero_()
        self.to(dev); self.register_buffer("name_ids", torch.tensor(NAME_IDS, device=dev), persistent=False); self.ablate = False   # not in the state dict

    def key(self, g, ids):
        return F.normalize(self.W_n(g.w("wte.weight")[ids].float()), dim=-1)

    def read(self, g, h, ids, Sn):
        """h (B, T, D) entering block mem.layer; ids (B, T) the tokens; Sn (B, DHN, DHN) (v x k) -> h + W_on (S_n q) at the name positions."""
        assert ids is not None and tuple(ids.shape) == tuple(h.shape[:2]), ("NK read: ids", None if ids is None else tuple(ids.shape), tuple(h.shape))
        with torch.autocast("cuda", enabled=False):
            isn = torch.isin(ids, self.name_ids); r = torch.einsum("bvk,btk->btv", Sn.float(), self.key(g, ids))
            if self.ablate:
                r = torch.zeros_like(r)
            o = self.W_on(r) * isn[..., None].to(r.dtype)
        return h + o.to(h.dtype)

    def write(self, g, Sn, key_ids, hw):
        """Sn (B, DHN, DHN); key_ids (B,) name tokens; hw (B, D) = ln_w(h_11 at ':') -> the delta-rule write (beta 1)."""
        assert bool(torch.isin(key_ids, self.name_ids).all()), "NK write: a key that is not a name token"
        with torch.autocast("cuda", enabled=False):
            k = self.key(g, key_ids); v = self.W_vn(hw.float()); Sk = torch.einsum("bvk,bk->bv", Sn.float(), k)
            return Sn.float() - Sk[:, :, None] * k[:, None, :] + v[:, :, None] * k[:, None, :]


class DeltaMemNK(DM.DeltaMem):
    """DeltaMem whose zero state carries the NK block (no new parameters; set as the instance's class by InjectNK)."""
    def zero_state(self, B, dev):
        assert getattr(self, "add_norm", None) is None, "NK: the NK block takes the slot §185 den would use"
        return torch.zeros(B, self.h_add + self.h_del + 1, DM.DH, DM.DH, device=dev)


class InjectNK(InjectAW):
    def __init__(self, g, layer, write, h_add, h_del, add_norm, seed=1):
        super().__init__(g, layer, write, h_add, h_del, add_norm); assert DHN == DM.DH and add_norm is None
        self.mem.__class__ = DeltaMemNK; self.NB = h_add + h_del; self.nk = NK(g.w("wte.weight").shape[1], g.w("wte.weight").device, seed)

    def split(self, S):
        assert S.shape[1] == self.NB + 1, ("NK: the packed state", tuple(S.shape)); return S[:, :self.NB], S[:, self.NB]

    def pass_aw(self, S, x, mask, h_low=None):
        g, mem = self.g, self.mem; B, T = x.shape; D = g.w("wte.weight").shape[1]; hd = D // g.H; assert mem.write_mode == "top", "AW: h_w = block 11's output"
        S12, Sn = self.split(S); h = (g.embed(x) + DM.wpe_slice(g, T)[None]) if h_low is None else h_low; h6 = None
        for i in range(0 if h_low is None else mem.layer, g.L):
            if i == mem.layer:
                h6 = h; h = self.nk.read(g, DM.read_fixed(mem, h, S12), x, Sn)
            h = DM._block(g, i, h, hd)
        pos = (mask.long().sum(1) - 1) if mask is not None else torch.full((B,), T - 1, device=x.device); ar = torch.arange(B, device=x.device)
        assert bool((x[ar, pos] == COLON).all()), "AW: the write position must be ':'"
        S12n = DM.write_seq(mem, h6[ar, pos][:, None], h[ar, pos][:, None], S12, None)
        with torch.autocast("cuda", enabled=False):
            Snn = self.nk.write(g, Sn, x[ar, pos - CUE_NAME_OFF], mem.ln_w(h[ar, pos].float()))
        return torch.cat([S12n, Snn.to(S12n.dtype)[:, None]], 1), h, pos

    def fwd_read_nk(self, x, S):
        """deltas5.fwd_read with the NK read (x = token ids)."""
        g, mem = self.g, self.mem; B, T = x.shape; D = g.w("wte.weight").shape[1]; hd = D // g.H; S12, Sn = self.split(S); h = g.embed(x) + DM.wpe_slice(g, T)[None]
        for i in range(g.L):
            if i == mem.layer:
                h = self.nk.read(g, DM.read_fixed(mem, h, S12), x, Sn)
            h = DM._block(g, i, h, hd)
        return F.layer_norm(h, (D,), g.w("ln_f.weight"), g.w("ln_f.bias"))

    def dec_logits(self, S, items, dev):
        seqs = [it["dec"] for it in items]; ids_, _ = pad(seqs); x = torch.tensor(ids_, device=dev)
        h = self.fwd_read_nk(x, S); pos = torch.tensor([len(q) - 1 for q in seqs], device=dev)
        return self.g.logits(h[torch.arange(len(seqs), device=dev), pos]).float()

    def aux_logits(self, S, items, t, dev, hold=False):
        assert not hold, "§228: no hold targets (NOFADE, t <= d only)"
        pre = self.aux_prompt(t); seqs, tg = [], []
        for it in items:
            assert t <= len(it["aux"]), "§228: no aux target past the last step"
            tgt = it["aux"][t - 1]; seqs.append(pre + tgt[:-1]); tg.append(tgt)
        x_, _ = pad(seqs); y_, _ = pad(tg, -100); x = torch.tensor(x_, device=dev); y = torch.tensor(y_, device=dev)
        assert x.shape[1] == len(pre) + y.shape[1] - 1
        h = self.fwd_read_nk(x, S)
        return self.g.logits(h[:, len(pre) - 1:]).float(), y

    @torch.no_grad()
    def aux_greedy(self, S, t, L):
        pre = self.aux_prompt(t); x = torch.tensor([pre] * S.shape[0], device=S.device)
        for _ in range(L):
            nt = self.g.logits(self.fwd_read_nk(x, S)[:, -1]).float().argmax(-1); x = torch.cat([x, nt[:, None]], 1)
        return x[:, len(pre):].cpu().numpy()


def nk_keys_ok(it, o):
    """tok_aw's assert (cue step, write_src answer): pass c's input carries spine[c]'s name token CUE_NAME_OFF before ':'."""
    sp = it["names"]["spine"]
    return all(q[-1] == COLON and q[-1 - CUE_NAME_OFF] == NAME_TOK[sp[c]] for c, q in enumerate(o["think_cues"], 1))


def cond_ws(pv, T, d):
    """pv {t: [pred]} (ws_readout per_item) -> {t: P(pass t correct | pass t-1 correct)} for t 2..d (None if no item had pass t-1 correct), + pass-1 acc."""
    ok = {t: np.array([int(p) == int(it["raw"]["values"][min(t, d)]) for p, it in zip(pv[str(t)], T)]) for t in range(1, d + 1)}
    return {str(t): (float(ok[t][ok[t - 1]].mean()) if ok[t - 1].any() else None) for t in range(2, d + 1)}, float(ok[1].mean()), \
        {str(t): int(ok[t - 1].sum()) for t in range(2, d + 1)}


@torch.no_grad()
def part_nk_pilot(m, g, sets, a, dev, bs):
    """NK_v1 §4: TEST-PERM d 3-4 (perm_render.test_perm_sets on REG TEST2, seeds 25150 + d), WS per pass at ':' (inject_aw.ws_readout, K = think_k) ->
    cond_ws_perm / ws1_perm (+ _n = the conditioning counts); NK runs: the same with nk.ablate -> cond_ws_perm_ablated / ws1_perm_ablated. Per item preds kept."""
    import boxes.inject_aw as AW
    import boxes.inject_aw_eval as EV
    import boxes.perm_render as PR
    K = a["think_k"]; ds = (3, 4); P_, info = PR.test_perm_sets({d: sets["test2"][d] for d in ds}, ds); T = EV.tokd({"test2": P_}, a, ds)
    out = {"info": info, "cond_ws_perm": {}, "ws1_perm": {}, "cond_n": {}, "items": {}}; arms = [("", False)] + ([("_ablated", True)] if hasattr(m, "nk") else [])
    for suf, abl in arms:
        if hasattr(m, "nk"):
            m.nk.ablate = abl
        try:
            for d in ds:
                _, pv = AW.ws_readout(m, {d: T[d]}, dev, K, bs, per_item=True); c_, w1, n_ = cond_ws(pv[str(d)], T[d], d)
                out.setdefault("cond_ws_perm" + suf, {})[str(d)] = c_; out.setdefault("ws1_perm" + suf, {})[str(d)] = w1; out.setdefault("cond_n" + suf, {})[str(d)] = n_
                out["items"].setdefault(str(d), {"sig": [it["sig"] for it in T[d]], "v": [[int(it["raw"]["values"][min(t, d)]) for t in range(1, K + 1)] for it in T[d]]})["ws_pred" + suf] = pv[str(d)]
        finally:
            if hasattr(m, "nk"):
                m.nk.ablate = False
    out["nk"] = hasattr(m, "nk"); out["def"] = ("NK_v1 §4: TEST-PERM d 3-4 (REG TEST2 re-rendered, default_rng(25150 + d)); WS = argmax over ' 0'..' 10' at ':' of pass t vs "
                                                "v_min(t, d) (inject_aw.ws_readout); cond_ws_perm[d][t] = P(pass t ok | pass t-1 ok), t 2..d; ablated = nk.ablate (W_on r -> 0)")
    return out
