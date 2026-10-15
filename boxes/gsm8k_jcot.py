"""JOINT-COT-L1 (docs/THEORY_JOINTCOT_DRAFT_2026-10-14.md §3 / §5 / §9; user-approved via Audit 2026-10-15 01:4x; registration pending at the time of
writing) for gsm8k_stepaux's stepaux arm. A NEW module: gsm8k_stepaux imports it only when --joint_cot / --kd_beta is set.
  Teacher (plain trunk g, no memory): T1 = C's full TEXT-COT CE (pre + cot + a1 + EOS, labels from |pre|); T2 (items m >= kd_m_min) = pre +
    lines[:m-1] + a1 + EOS with an answer-only CE (labels a1[1:] + EOS; "###" input-forced). L_T = L_T1 + L_T2.
  Coupling (--kd_beta > 0): teacher states z_l at "###" - in T2 (--kd_teacher drop, the default) or T1 (full) - block outputs l in --kd_layers
    (pre ln_f, as probe_kd), DETACHED at capture; student states h_l at the decoder's "###" (p_S = 1 + |pre| in [EOT] + pre + a1 + EOS) from the
    dec_loss forward with the read live. L_KD = mean over layers of mean_{items, dims} loss(z_l - h_l) / sigma_T,l (sigma_T,l = the scalar std of
    z_l over the micro-chunk, detached; --kd_norm none: no division); loss = |.| (l1) or smooth-L1 (smoothl1). Items m < kd_m_min: no T2 CE, no KD.
  Order per micro-chunk (draft §5): teacher forward -> gamma_T * L_T / micro backward (z kept detached) -> student forward -> (L_S + beta L_KD) /
    micro backward (the caller). No @torch.no_grad on these functions (ast-checked by the gate).
Capture helpers mirror GPT2KV.forward (deltamem._block + ln_f) and ProsMem.dec_loss exactly (gate: bitwise on the plain path)."""
import numpy as np
import torch
import torch.nn.functional as Fn

import boxes.deltamem as DM
from boxes.prosqa168s2 import ANS, ENC, EOS, EOT, pad


def parse_layers(s):
    a_, _, b_ = s.partition("-"); return tuple(range(int(a_), int(b_ or a_) + 1))


def fwd_plain_layers(g, x, pos, layers):
    """GPT2KV.forward via deltamem._block -> (final hidden after ln_f, {l: block-l output (pre ln_f) at pos[r] per row r})."""
    h = g.embed(x) + DM.wpe_slice(g, x.shape[1])[None]; hd = h.shape[-1] // g.H; caps = {}; rows = torch.arange(x.shape[0], device=x.device)
    for i in range(g.L):
        h = DM._block(g, i, h, hd)
        if i + 1 in layers:
            caps[i + 1] = h[rows, pos]
    return Fn.layer_norm(h, (h.shape[-1],), g.w("ln_f.weight"), g.w("ln_f.bias")), caps


def dec_layers(m, S, its, dev, layers):
    """ProsMem.dec_loss with the block outputs captured at the decoder's "###" (p_S = 1 + |pre|) -> (answer CE, {l: (b, 768)})."""
    g = m.g; seqs = [[EOT] + it["pre"] + it["a1"] + [EOS] for it in its]; ids_, _ = pad(seqs); x = torch.tensor(ids_, device=dev)
    lab = np.full(ids_.shape, -100); ps = []
    for r, (it, q) in enumerate(zip(its, seqs)):
        n0 = 1 + len(it["pre"]); lab[r, n0:len(q)] = q[n0:]; ps.append(n0); assert q[n0] == ANS, "JOINT-COT: student '###' position"
    h = g.embed(x) + DM.wpe_slice(g, x.shape[1])[None]; hd = h.shape[-1] // g.H; caps = {}; pos = torch.tensor(ps, device=dev); rows = torch.arange(len(its), device=dev)
    for i in range(g.L):
        if i == m.mem.layer:
            h = DM.read_fixed(m.mem, h, S)
        h = DM._block(g, i, h, hd)
        if i + 1 in layers:
            caps[i + 1] = h[rows, pos]
    hf = Fn.layer_norm(h, (h.shape[-1],), g.w("ln_f.weight"), g.w("ln_f.bias")); lg = g.logits(hf[:, :-1]).float()
    return Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), torch.tensor(lab, device=dev)[:, 1:].reshape(-1), ignore_index=-100), caps


def lines_of(it):
    L = [ENC.encode("<<%s>>\n" % e) for e in it["eqs"]]; assert [t for l_ in L for t in l_] == it["cot"], "JOINT-COT: cot == its lines concatenated"; return L


def teacher(g, its, dev, ac, layers=(), mode="drop", m_min=2):
    """-> (L_T1, L_T2 or None, {l: z (n_kd, 768) detached}, kd rows (indices into its)). T1 / T2 each one padded plain forward."""
    s1, l1, p1 = [], [], []
    for it in its:
        q = it["pre"] + it["cot"] + it["a1"] + [EOS]; s1.append(q); l1.append([-100] * len(it["pre"]) + q[len(it["pre"]):]); p1.append(len(it["pre"]) + len(it["cot"]))
    r2 = [i for i, it in enumerate(its) if it["m"] >= m_min]
    ids_, _ = pad(s1); lb, _ = pad(l1, -100); x = torch.tensor(ids_, device=dev)
    with ac():
        if mode == "full" and layers and r2:
            for i in r2:
                assert s1[i][p1[i]] == ANS, "JOINT-COT: teacher T1 '###' position"
            hf, caps = fwd_plain_layers(g, x, torch.tensor(p1, device=dev), layers); z = {l: c[r2].detach().float() for l, c in caps.items()}
        else:
            hf, _ = g(g.embed(x), None); z = {}
    lg = g.logits(hf[:, :-1]).float(); L1 = Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), torch.tensor(lb, device=dev)[:, 1:].reshape(-1), ignore_index=-100)
    if not r2:
        return L1, None, z, r2
    s2, l2, pT = [], [], []
    for i in r2:
        it = its[i]; ln = lines_of(it); body = it["pre"] + [t for l_ in ln[:-1] for t in l_]; q = body + it["a1"] + [EOS]
        assert q[len(body)] == ANS, "JOINT-COT: teacher T2 '###' position"; s2.append(q); pT.append(len(body)); l2.append([-100] * (len(body) + 1) + q[len(body) + 1:])
    ids_, _ = pad(s2); lb, _ = pad(l2, -100); x = torch.tensor(ids_, device=dev)
    with ac():
        if mode == "drop" and layers:
            hf, caps = fwd_plain_layers(g, x, torch.tensor(pT, device=dev), layers); z = {l: c.detach().float() for l, c in caps.items()}
        else:
            hf, _ = g(g.embed(x), None)
    lg = g.logits(hf[:, :-1]).float(); L2 = Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), torch.tensor(lb, device=dev)[:, 1:].reshape(-1), ignore_index=-100)
    return L1, L2, z, r2


def t2_seqs(its):
    """T2 (pre + lines[:m-1] + a1 + EOS) for every item of its -> (sequences, '###' positions)."""
    s, p = [], []
    for it in its:
        body = it["pre"] + [t for l_ in lines_of(it)[:-1] for t in l_]; q = body + it["a1"] + [EOS]; assert q[len(body)] == ANS, "JOINT-COT: T2 '###'"
        s.append(q); p.append(len(body))
    return s, p


def teacher_z(g, its, dev, ac, layers, mode="drop"):
    """diagnostics: the coupling targets only (no CE, no logits) at the '###' of T2 (drop) or T1 (full), for every item of its -> {l: (n, 768)}."""
    if mode == "full":
        s = [it["pre"] + it["cot"] + it["a1"] + [EOS] for it in its]; p = [len(it["pre"]) + len(it["cot"]) for it in its]
        assert all(q[p_] == ANS for q, p_ in zip(s, p)), "JOINT-COT: T1 '###'"
    else:
        s, p = t2_seqs(its)
    ids_, _ = pad(s); x = torch.tensor(ids_, device=dev)
    with ac():
        _, caps = fwd_plain_layers(g, x, torch.tensor(p, device=dev), layers)
    return {l: c.detach().float() for l, c in caps.items()}


def t2_stats(g, its, dev, ac):
    """diagnostics (no grad by the caller): per item, the T2 '###' argmax == the first answer token (top1; C's 0.024, draft §3) and every answer
    label argmax-correct in the one teacher-forced T2 forward (tf_exact) -> (top1 list, tf_exact list)."""
    s, p = t2_seqs(its); ids_, _ = pad(s); x = torch.tensor(ids_, device=dev)
    with ac():
        hf, _ = g(g.embed(x), None)
    am = g.logits(hf[:, :-1]).argmax(-1).cpu().numpy(); t1_, tf_ = [], []
    for r, (q, p_) in enumerate(zip(s, p)):
        t1_.append(bool(am[r, p_] == q[p_ + 1])); tf_.append(bool(all(am[r, j - 1] == q[j] for j in range(p_ + 1, len(q)))))
    return t1_, tf_


def t2_prefixes(its):
    """the T2 prompts for greedy answering: pre + lines[:m-1] + '###' (T2-ACC, draft G4)."""
    return [it["pre"] + [t for l_ in lines_of(it)[:-1] for t in l_] + [ANS] for it in its]


def teacher_context(z):
    """TEACHER-CONTEXT (draft G4): per layer, the mean over dims of the between-item variance of z_l / sigma_T,l^2 (the scalar variance of z_l)."""
    return {l: float(v.var(0).mean() / v.var().clamp_min(1e-12)) for l, v in z.items() if v.shape[0] > 1}


def kd_items(z, h, sig, kind="l1"):
    """per-item coupling loss, the mean over dims and layers of loss(z_l - h_l) / sig[l] (sig None: no division) -> (n,) float tensor."""
    out = []
    for l in z:
        e = h[l].float() - z[l]; d = e.abs() if kind == "l1" else Fn.smooth_l1_loss(h[l].float(), z[l], reduction="none")
        out.append(d.mean(1) / (sig[l] if sig is not None else 1.0))
    return torch.stack(out).mean(0)


def sig_of(z, norm):
    return {l: max(float(v.std()), 1e-6) for l, v in z.items()} if norm == "teacher_std" else None


def l1_const(z, sig):
    """the item-independent floor of the L1 coupling (draft G2): min over a constant c of mean |z_l - c| / sigma = the per-dim median."""
    return float(np.mean([float((v - v.median(0).values).abs().mean()) / (sig[l] if sig is not None else 1.0) for l, v in z.items()]))


def route(m, its, dev, ac, layers, think, Ks, mode="drop", norm="teacher_std", kind="l1", bs=64):
    """KD-ROUTE (draft G1; no grad): z from the teacher on its (all m >= the run's kd_m_min - the caller filters), h at the decoder '###' reading
    S_Kp for each Kp in Ks (think(its, Kp) -> states). sigma_T over the whole batch. -> ({Kp: per-item (n,) tensor}, z, sig)."""
    with torch.no_grad():
        Z = {l: [] for l in layers}
        for i in range(0, len(its), bs):
            z_ = teacher_z(m.g, its[i:i + bs], dev, ac, layers, mode)
            for l in layers:
                Z[l].append(z_[l])
        Z = {l: torch.cat(v) for l, v in Z.items()}; sig = sig_of(Z, norm); out = {}
        for Kp in Ks:
            S = think(its, Kp); H = {l: [] for l in layers}
            for i in range(0, len(its), bs):
                with ac():
                    _, cap = dec_layers(m, S[i:i + bs], its[i:i + bs], dev, layers)
                for l in layers:
                    H[l].append(cap[l])
            out[Kp] = kd_items(Z, {l: torch.cat(v) for l, v in H.items()}, sig, kind)
    return out, Z, sig


def kd_loss(z, h, rows, norm="teacher_std", kind="l1"):
    """mean over layers of mean_{items, dims} loss(z_l - h_l[rows]) / sigma_T,l -> (L_KD, {l: term}, {l: sigma_T}, {l: sigma_S})."""
    terms, sig_t, sig_s = {}, {}, {}
    for l in z:
        hs = h[l][rows].float(); e = (hs - z[l]); d = e.abs().mean() if kind == "l1" else Fn.smooth_l1_loss(hs, z[l])
        st = z[l].std(); sig_t[l] = float(st); sig_s[l] = float(hs.detach().std())
        terms[l] = d / st.clamp_min(1e-6) if norm == "teacher_std" else d
    return torch.stack(list(terms.values())).mean(), {l: float(v.detach()) for l, v in terms.items()}, sig_t, sig_s
