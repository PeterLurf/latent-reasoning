"""DM.CHUNK_DELTA gates (Theory / Audit 2026-10-10; boxes/deltamem.delta_chunk + write_seq), CPU -> boxes/runs_fsq/chunk_delta_gates/gate_chunk_delta.json
(dumped after every gate):
  g1  fp64 unit: delta_chunk vs the token loop (write_seq's form) - final state + grads w.r.t. S0, k, v, alpha / beta logits, <= 1e-12 relative;
      cases: T multiple of C / partial last chunk / T < C; masks: none, suffix padding (incl. a row ending exactly at a chunk boundary), random
      mid-sequence; beta ~ 1.95 / ~ 0.05 / mixed; alpha ~ sigmoid(5) / ~ 0.5 / ~ 0.05 (log-space underflow)
  g2  fp64 model (ProsMem on 2 p8 train items, masked): K = 1 and 10, ckpt_passes off / on, cache_lower off / on - final state + every parameter grad
      chunk vs loop <= 1e-12 relative (Tensor.float mapped to double in this process only, as gate_cache_lower g5)
  g3  fp32 forward: the state after each of 10 passes, chunk vs loop, on 2 p8 items of ~440 question tokens, <= 1e-5 relative
  g4  flags off: gpt_forward_mem (FAST_ADD off and on) == the pre-change deltamem bitwise (output + state, mask none / some)
    python scratch/gate_chunk_delta.py <pre-change deltamem.py> [g1,g2,g3,g4]"""
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

import boxes.deltamem as DM  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "chunk_delta_gates"); OUTF = os.path.join(GD, "gate_chunk_delta.json")
OUT = json.load(open(OUTF)) if os.path.exists(OUTF) else {}


def dump():
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(OUTF, "w", newline="\n"), indent=1)


def rel(a, b):
    return float((a - b).abs().max() / (a.abs().max() + 1e-300))


def ref_loop(Sd, kd, vd, al, be, mask):
    for t in range(kd.shape[1]):                                            # write_seq's delta update, verbatim
        b_ = be[:, t][..., None, None]; Skk = torch.einsum("bhvk,bhk->bhv", Sd, kd[:, t])
        Sd_ = al[:, t][..., None, None] * (Sd - b_ * Skk[..., :, None] * kd[:, t][..., None, :]) + b_ * vd[:, t][..., :, None] * kd[:, t][..., None, :]
        Sd = Sd_ if mask is None else torch.where(mask[:, t][:, None, None, None], Sd_, Sd)
    return Sd


def g1():
    torch.manual_seed(0); dt = torch.float64; B, H, Dh, C = 3, 6, 64, 64; res = {}
    for T in (128, 133, 17):
        for mk in ("none", "suffix", "random"):
            for bcase, bmu in (("beta_hi", 3.66), ("beta_lo", -3.66), ("beta_mix", 0.0)):
                for acase, amu in (("alpha_init", 5.0), ("alpha_half", 0.0), ("alpha_small", -2.94)):
                    S0 = torch.randn(B, H, Dh, Dh, dtype=dt, requires_grad=True); k = torch.randn(B, T, H, Dh, dtype=dt, requires_grad=True)
                    v = torch.randn(B, T, H, Dh, dtype=dt, requires_grad=True)
                    ga = (amu + 0.3 * torch.randn(B, T, H, dtype=dt)).requires_grad_(); gb = (bmu + (2.0 if bcase == "beta_mix" else 0.3) * torch.randn(B, T, H, dtype=dt)).requires_grad_()
                    if mk == "none":
                        mask = None
                    elif mk == "suffix":                                    # row 1 ends exactly at a chunk boundary when T > 64
                        L = [T, min(T, 64), max(1, T - 11)]; mask = torch.arange(T)[None] < torch.tensor(L)[:, None]
                    else:
                        mask = torch.rand(B, T) > 0.3
                    W = torch.randn(B, H, Dh, Dh, dtype=dt); o = {}
                    for nm, fn in (("loop", lambda *a_: ref_loop(*a_)), ("chunk", lambda *a_: DM.delta_chunk(*a_, C))):
                        al, be = torch.sigmoid(ga), 2 * torch.sigmoid(gb); kd = F.normalize(k, dim=-1)
                        Sd = fn(S0, kd, v, al, be, mask); gr = torch.autograd.grad((Sd * W).sum(), [S0, k, v, ga, gb])
                        o[nm] = (Sd.detach(), [g_.detach() for g_ in gr])
                    rr = {"state": rel(o["loop"][0], o["chunk"][0])}
                    rr.update({"grad_" + n_: rel(a_, b_) for n_, a_, b_ in zip(("S0", "k", "v", "alpha_logit", "beta_logit"), o["loop"][1], o["chunk"][1])})
                    rr["beta_range"] = [float((2 * torch.sigmoid(gb)).min()), float((2 * torch.sigmoid(gb)).max())]; rr["alpha_mean"] = float(torch.sigmoid(ga).mean())
                    res["T%d_%s_%s_%s" % (T, mk, bcase, acase)] = rr
    OUT["g1"] = res; OUT["g1_max_rel"] = max(max(v_ for k_, v_ in r.items() if k_.startswith(("state", "grad"))) for r in res.values())
    OUT["g1_pass"] = OUT["g1_max_rel"] <= 1e-12; dump(); print("G1", OUT["g1_max_rel"], OUT["g1_pass"], flush=True)


def prosmem(fp64):
    import boxes.prosqa168s2 as P
    from boxes import s5
    g = s5.load_gpt2("cpu"); torch.manual_seed(1); m = P.ProsMem(g)
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0); m.mem.gamma.fill_(0.5)
    return P, (m.double() if fp64 else m)


def g2():
    _f = torch.Tensor.float; torch.Tensor.float = lambda self, *a, **k: self.double(); DM.FAST_ADD = True
    try:
        P, m = prosmem(True); D = P.load(os.path.join(ROOT, "prosqa", "p8data", "p8_na_train.json"))[:2]; x, msk = m.think_inputs(D, "cpu"); res = {}
        for K in (1, 10):
            for ck in (False, True):
                for cache in (False, True):
                    o = {}
                    for nm, cd in (("loop", 0), ("chunk", 64)):
                        DM.CHUNK_DELTA = cd; m.zero_grad(set_to_none=True); S = m.mem.zero_state(2, "cpu").double(); hl = m.lower(x) if cache else None
                        for _ in range(K):
                            S = checkpoint(m.write, S, x, msk, hl, use_reentrant=False) if ck else m.write(S, x, msk, hl)
                        loss = m.dec_loss(S, D, "cpu"); loss.backward()
                        o[nm] = (S.detach().clone(), {n_: p_.grad.detach().clone() for n_, p_ in m.named_parameters() if p_.grad is not None})
                    gk = set(o["loop"][1]) | set(o["chunk"][1]); gr = {k_: rel(o["loop"][1][k_], o["chunk"][1][k_]) for k_ in gk}; wk = max(gr, key=gr.get)
                    res["K%d_ckpt%d_cache%d" % (K, ck, cache)] = {"state": rel(o["loop"][0], o["chunk"][0]), "grad_max_rel": gr[wk], "worst": wk, "n_grads": len(gk),
                                                                "same_grad_set": set(o["loop"][1]) == set(o["chunk"][1]), "T": int(x.shape[1])}
                    print("G2", K, ck, cache, res["K%d_ckpt%d_cache%d" % (K, ck, cache)], flush=True)
                    OUT["g2"] = res; dump()
    finally:
        torch.Tensor.float = _f; DM.CHUNK_DELTA = 0
    OUT["g2_max_rel"] = max(max(r["state"], r["grad_max_rel"]) for r in res.values()); OUT["g2_pass"] = OUT["g2_max_rel"] <= 1e-12 and all(r["same_grad_set"] for r in res.values())
    dump(); print("G2", OUT["g2_max_rel"], OUT["g2_pass"], flush=True)


def g3():
    DM.FAST_ADD = True; P, m = prosmem(False); D = P.load(os.path.join(ROOT, "prosqa", "p8data", "p8_na_train.json"))
    D = sorted(D, key=lambda it: abs(len(it["q"]) - 440))[:2]; x, msk = m.think_inputs(D, "cpu"); st = {}
    with torch.no_grad():
        for nm, cd in (("loop", 0), ("chunk", 64)):
            DM.CHUNK_DELTA = cd; S = m.mem.zero_state(2, "cpu"); st[nm] = []
            for _ in range(10):
                S = m.write(S, x, msk); st[nm].append(S.clone())
    DM.CHUNK_DELTA = 0; r = [rel(a_, b_) for a_, b_ in zip(st["loop"], st["chunk"])]
    OUT["g3"] = {"state_rel_by_pass": r, "q_tokens": [len(it["q"]) for it in D], "T": int(x.shape[1])}; OUT["g3_max_rel"] = max(r); OUT["g3_pass"] = max(r) <= 1e-5
    dump(); print("G3", r, OUT["g3_pass"], flush=True)


def g4(pre_path):
    spec = importlib.util.spec_from_file_location("dm_pre", pre_path); PRE = importlib.util.module_from_spec(spec); spec.loader.exec_module(PRE)
    P, m = prosmem(False); D = P.load(os.path.join(ROOT, "prosqa", "p8data", "p8_na_train.json"))[:2]; x, msk = m.think_inputs(D, "cpu")
    S0 = m.mem.zero_state(2, "cpu") + 0.01 * torch.randn(2, 12, 64, 64); res = {}; DM.CHUNK_DELTA = 0
    for fa in (False, True):
        DM.FAST_ADD = fa; PRE.FAST_ADD = fa
        with torch.no_grad():
            for mn, mk in (("mask", msk), ("nomask", None)):
                a_ = DM.gpt_forward_mem(m.g, m.g.embed(x), m.mem, S0, mask=mk); b_ = PRE.gpt_forward_mem(m.g, m.g.embed(x), m.mem, S0, mask=mk)
                res["fast%d_%s" % (fa, mn)] = {"out_equal": bool(torch.equal(a_[0], b_[0])), "state_equal": bool(torch.equal(a_[1], b_[1]))}
    DM.FAST_ADD = False; OUT["g4"] = res; OUT["g4_pass"] = all(all(v.values()) for v in res.values()); dump(); print("G4", res, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(3); which = (sys.argv[2] if len(sys.argv) > 2 else "g1,g2,g3,g4").split(",")
    if "g4" in which:
        g4(sys.argv[1])
    if "g1" in which:
        g1()
    if "g3" in which:
        g3()
    if "g2" in which:
        g2()
    OUT["all_pass"] = bool(all(OUT.get(k_ + "_pass") for k_ in ("g1", "g2", "g3", "g4"))); dump(); print("ALL", OUT["all_pass"], flush=True)
