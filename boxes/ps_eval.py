"""§223 PORT-STATE evals (PS2_v1.md at 54f63381 §4-§5; Note 223-2: K = d + 2, COPY-CONTROL power comparator, value floor, cf (a), budget-front
sampling, on-host 3090 fp32). NEW module. One file per cell: <run>/eval_ps.json, layout as sent to Theory 10-21 (keys below). fp32 (base cast
to float, LoRA unmerged, no autocast), the cell's own flags (BOS, M2 mask). Values are never printed; per-item outputs are stored.
  acc           test set d 1..12 (1,000 per d), K = d + 2 (DIRECT: no latents); the prediction = the argmax token after "The answer is:".
  k_sweep       K = d + {-1, 0, 1, 2}, d 1..8 (descriptive).
  decode        (M1) D(z_t) teacher-forced on R_t: value-token accuracy, all-values-correct per item; z-shuffle (another item's z_t); value floor
                (another item's R_t values at the same value index, within d, t).
  copy_control  (M1) D' (z_1 = prj(h at <bot>) = the K0-pass encoding) and D_post (the real z_t), same architecture (codi_ps.DHead: a copy of
                GPT-2 blocks 0-1) + a learned step embedding for t added to the z input, fresh init each, 5,000 steps, bs 64, AdamW lr 1e-4, the
                train stream (seed 22340, frames 0-5, excl test + pair candidates), d uniform 1..8, t uniform 1..d per item.
  answer_onset  (M1) D(z_t) teacher-forced on R_d: the argmax at R_d's number position vs the answer; floor = within-d permuted answers.
  swaps         d 3..8, pairs = nested_words.pair_candidates kept iff own / donor-final / cf distinct (<= --pairs_per_d per d), t = ceil(d/2):
                plain, prefix (the donor's z_1..z_t as the first t latent inputs in the recipient's context), floor (the same with the donor of the
                next kept pair, scored against this pair's donor-final), leak (the donor's last 6 question-content KV rows spliced in), marker (BOS +
                <bot> KV rows from the donor; z_1, the <bot> OUTPUT, stays the recipient's - swapping it is PREFIX-SWAP(1)), self (the recipient's own z_1..z_t: bitwise == plain asserted), state_full (all K inputs from the
                donor), cache_full (the donor's latent-row KV in the decoder call). cf (a): the recipient's ops after t from the donor's S_0..S_t
                (back-references j <= t read the donor's S_j); cf_b: back-references j < t read the recipient's S_j.
  budget_front  every 3rd kept pair, t 1..d-1, budgets b 0..(d+2-t): PREFIX-SWAP(t) with the latent steps after t truncated to b, then the
                decoder; b* = the smallest b with pred == donor-final (reached false -> b* = max + 1 per PS2_v1 §7, set by the scorer).
    python -m boxes.ps_eval --run boxes/runs_fsq/s223_ps_s1 --device cuda:0 [--parts acc,k_sweep,decode,copy_control,answer_onset,swaps,budget_front]"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import boxes.codi_ps as PS  # noqa: E402
import boxes.nested_words as NW  # noqa: E402

PARTS = ("acc", "k_sweep", "decode", "copy_control", "answer_onset", "swaps", "budget_front")
DESCRIPTIVE_NOT_RUN = ["QMASK", "QDROP", "QUESTION-SWAP", "INT-FOLLOW", "STATE-ONSET", "STALL", "LOGIT-LENS", "LEAD", "C7-round-trip"]
SEED_CC = 22340


# ------------------------------------------------------------------ model / cache helpers
def load_run(run, dev):
    A = json.load(open(os.path.join(run, "args.json"))); a = types.SimpleNamespace(**{k: A.get(k) for k in ("recode", "lam", "bottleneck", "direct", "codi_compat", "seed")})
    a.codi_compat = False; m, prj, D, TP = PS.build_all(a, dev); sd = torch.load(os.path.join(run, "trainable.pt"), map_location=dev, weights_only=True)
    names = {n for n, _ in TP}; assert names == set(sd), ("trainable.pt keys != the cell's trainables", sorted(names ^ set(sd))[:5])
    for n, p_ in TP:
        p_.data.copy_(sd[n])
    m.float(); prj.float()
    if D is not None:
        D.float().eval()
    m.eval(); prj.eval(); return m, prj, D, a, A


def kv_layers(pkv):
    if hasattr(pkv, "layers"):
        return [[L.keys, L.values] for L in pkv.layers]
    if hasattr(pkv, "key_cache"):
        return [[k, v] for k, v in zip(pkv.key_cache, pkv.value_cache)]
    return [[k, v] for k, v in pkv]


def kv_build(layers, like):
    if isinstance(like, (tuple, list)):
        return tuple((k, v) for k, v in layers)
    from transformers import DynamicCache
    c = DynamicCache()
    for i, (k, v) in enumerate(layers):
        c.update(k, v, i)
    return c


def num_of(tok_id, enc):
    s = enc.decode([int(tok_id)]).strip(); return int(s) if s.isdigit() else -1


@torch.no_grad()
def run(m, prj, a, b, K, z_in=None, enc_patch=None, n_lat=None, lat_kv=None, want_kv=False):
    """one batched forward. z_in: {j (1-based): (B, 1, 768)} replaces the j-th latent INPUT; enc_patch(layers) edits the encoder KV in place;
    n_lat: latent steps actually run (truncation; default K); lat_kv: per-layer [k, v] for the latent rows, replacing them before the decoder.
    -> (number-position logits (B, V), zs [z_1..z_n], latent-row KV per layer [k, v] or None)."""
    dev = b["enc"].device; B = b["enc"].shape[0]; K = 0 if a.direct else K; n_lat = K if n_lat is None else n_lat; ne = b["em"].shape[1]
    o = m(input_ids=b["enc"], attention_mask=b["em"], use_cache=True, output_hidden_states=True); pkv = o.past_key_values
    z = prj(o.hidden_states[-1][:, -1:, :])
    if enc_patch is not None:
        L = kv_layers(pkv); enc_patch(L); pkv = kv_build(L, pkv)
    am = b["em"]; zs = []
    for j in range(1, n_lat + 1):
        if z_in is not None and j in z_in:
            z = z_in[j]
        zs.append(z); am = torch.cat([am, PS._ones(B, 1, dev)], 1)
        o = m(inputs_embeds=z, attention_mask=am, use_cache=True, output_hidden_states=True, past_key_values=pkv); pkv = o.past_key_values
        z = prj(o.hidden_states[-1][:, -1:, :])
    lk = None
    if want_kv or lat_kv is not None:
        L = kv_layers(pkv)
        if want_kv:
            lk = [[k[:, :, ne:].clone(), v[:, :, ne:].clone()] for k, v in L]
        if lat_kv is not None:
            for li, (k, v) in enumerate(L):
                L[li] = [torch.cat([k[:, :, :ne], lat_kv[li][0]], 2), torch.cat([v[:, :, :ne], lat_kv[li][1]], 2)]
            pkv = kv_build(L, pkv)
    dm = am.clone()
    if a.bottleneck and not a.direct:
        dm[:, :ne] = dm[:, :ne] * (1 - b["qm"])
    dmf = torch.cat([dm, PS._ones(B, b["dec"].shape[1], dev)], 1)
    so = m(inputs_embeds=m.transformer.wte(b["dec"]), attention_mask=dmf, use_cache=True, past_key_values=pkv)
    lg = so.logits.gather(1, b["npos"][:, None, None].expand(-1, 1, so.logits.size(-1))).squeeze(1)
    return lg, zs, lk


def preds_of(lg, enc):
    return [num_of(t, enc) for t in lg.argmax(-1).tolist()]


# ------------------------------------------------------------------ D-head readouts
def d_forward(m, D, z, targets, step_emb=None, t_idx=None):
    """teacher-forced D on z (N, 1, 768) -> (logits (N, L, V), labels (N, L) with -100 pad)."""
    dev = z.device; L = max(len(r) for r in targets); N = len(targets)
    ids = torch.full((N, L), PS.PAD, device=dev); km = torch.zeros(N, L, dtype=torch.long, device=dev); lab = torch.full((N, L), -100, device=dev)
    for i, r in enumerate(targets):
        ids[i, :len(r)] = torch.tensor(r, device=dev); km[i, :len(r)] = 1; lab[i, :len(r)] = torch.tensor(r, device=dev)
    zz = z.float() + (step_emb(t_idx)[:, None, :] if step_emb is not None else 0)
    x = torch.cat([zz, m.transformer.wte(ids[:, :-1]).float()], 1); h = D(x, km); return F.linear(h, m.transformer.wte.weight.float()), lab


def value_positions(lab, vt):
    return torch.isin(lab, vt) & (lab != -100)


def value_scores(lg, lab, vt, perm_lab=None):
    """per item: (n value tokens, n correct, all correct) [+ the floor vs perm_lab's values at the same value index]."""
    pred = lg.argmax(-1); vp = value_positions(lab, vt); out = []
    for i in range(lab.shape[0]):
        idx = vp[i].nonzero().flatten(); tv = lab[i, idx]; pv = pred[i, idx]; n = len(idx); c = int((tv == pv).sum())
        fl = None
        if perm_lab is not None:
            ov = perm_lab[i][value_positions(perm_lab[i:i + 1], vt)[0]] if perm_lab is not None else None
            fl = sum(int(k < len(ov) and int(ov[k]) == int(tv[k])) for k in range(n))
        out.append((n, c, n > 0 and c == n, fl))
    return out


# ------------------------------------------------------------------ parts
def batches(its, bs):
    return [its[i:i + bs] for i in range(0, len(its), bs)]


def part_acc(m, prj, a, TE, enc, dev, bs, offsets=(2,), ds=range(1, 13)):
    res = {"K": {}, "acc_by_d": {}, "items": {}, "preds": {}} if offsets == (2,) else {"acc_by_offset": {}}
    for off in offsets:
        for d in ds:
            K = 0 if a.direct else max(0, d + off); P, ok = [], []
            for chunk in batches(TE[d], bs):
                its = [PS.tok_item(x, enc) for x in chunk]; lg, _, _ = run(m, prj, a, PS.collate(its, K, dev), K); pr = preds_of(lg, enc)
                P += pr; ok += [int(p == x["answer"]) for p, x in zip(pr, chunk)]
            if offsets == (2,):
                res["K"][str(d)] = K; res["acc_by_d"][str(d)] = float(np.mean(ok)); res["items"][str(d)] = ok; res["preds"][str(d)] = P
            else:
                res["acc_by_offset"].setdefault(str(off), {})[str(d)] = float(np.mean(ok))
    return res


def part_decode(m, prj, D, a, TE, enc, dev, bs, vt, n_per_d):
    res = {"value_acc": {}, "value_items": {}, "zshuffle_value_acc": {}, "value_floor": {}, "n_value_tokens": {}}
    for d in range(1, 9):
        K = d + 2; acc = {t: [0, 0] for t in range(1, d + 1)}; zsh = {t: [0, 0] for t in acc}; flr = {t: [0, 0] for t in acc}; items = {t: [] for t in acc}
        for chunk in batches(TE[d][:n_per_d], bs):
            its = [PS.tok_item(x, enc) for x in chunk]; _, zs, _ = run(m, prj, a, PS.collate(its, K, dev), K); B = len(its); perm = [(i + 1) % B for i in range(B)]
            for t in range(1, d + 1):
                tg = [it["R"][t - 1] for it in its]; lg, lab = d_forward(m, D, zs[t - 1], tg)
                _, plab = d_forward(m, D, zs[t - 1], [tg[p] for p in perm])
                sc = value_scores(lg, lab, vt, perm_lab=plab); acc[t][0] += sum(s[1] for s in sc); acc[t][1] += sum(s[0] for s in sc)
                flr[t][0] += sum(s[3] for s in sc); flr[t][1] += sum(s[0] for s in sc); items[t] += [int(s[2]) for s in sc]
                lg2, lab2 = d_forward(m, D, zs[t - 1][perm], tg); s2 = value_scores(lg2, lab2, vt); zsh[t][0] += sum(s[1] for s in s2); zsh[t][1] += sum(s[0] for s in s2)
        f = lambda x: {str(t): (v[0] / v[1] if v[1] else None) for t, v in x.items()}  # noqa: E731
        res["value_acc"][str(d)] = f(acc); res["zshuffle_value_acc"][str(d)] = f(zsh); res["value_floor"][str(d)] = f(flr)
        res["value_items"][str(d)] = {str(t): v for t, v in items.items()}; res["n_value_tokens"][str(d)] = {str(t): v[1] for t, v in acc.items()}
    return res


def part_onset(m, prj, D, a, TE, enc, dev, bs, vt, n_per_d):
    res = {"acc": {}, "items": {}, "floor": {}}
    for d in range(1, 9):
        K = d + 2; items = {t: [] for t in range(1, d + 1)}; fl = []
        for chunk in batches(TE[d][:n_per_d], bs):
            its = [PS.tok_item(x, enc) for x in chunk]; _, zs, _ = run(m, prj, a, PS.collate(its, K, dev), K); B = len(its)
            fl += [int(chunk[(i + 1) % B]["answer"] == chunk[i]["answer"]) for i in range(B)]
            tg = [it["R"][d - 1] for it in its]                              # R_d = "<S_d> has N <u>." + EOS: its first value token is the answer
            for t in range(1, d + 1):
                lg, lab = d_forward(m, D, zs[t - 1], tg); pred = lg.argmax(-1); vp = value_positions(lab, vt)
                for i in range(B):
                    p0 = int(vp[i].nonzero().flatten()[0]); items[t].append(int(int(pred[i, p0]) == int(lab[i, p0])))
        res["acc"][str(d)] = {str(t): float(np.mean(v)) for t, v in items.items()}; res["items"][str(d)] = {str(t): v for t, v in items.items()}; res["floor"][str(d)] = float(np.mean(fl))
    return res


def part_copy_control(m, prj, a, TE, enc, dev, bs, vt, excl, steps, n_per_d, log):
    """Note 223-2 Q2: D' (K0-pass z = z_1) and D_post (real z_t), same arch / init / budget, each + a learned step embedding."""
    heads = {}
    for nm in ("dprime", "dpost"):
        torch.manual_seed(SEED_CC); Dh = PS.DHead(m).to(dev).float(); se = nn.Embedding(13, 768).to(dev); nn.init.zeros_(se.weight)
        heads[nm] = (Dh, se, torch.optim.AdamW(list(Dh.parameters()) + list(se.parameters()), lr=1e-4, weight_decay=0.0))
    stream = NW.Stream(SEED_CC, excl); rng = np.random.default_rng(SEED_CC + 1); t0 = time.time()
    for st in range(1, steps + 1):
        d = int(rng.integers(1, 9)); K = d + 2; its = [PS.tok_item(stream.draw(d), enc) for _ in range(bs)]; ts = rng.integers(1, d + 1, size=bs)
        with torch.no_grad():
            _, zs, _ = run(m, prj, a, PS.collate(its, K, dev), K)
        Z = torch.stack([zs[int(t) - 1][i] for i, t in enumerate(ts)]); Z1 = zs[0]; tg = [it["R"][int(t) - 1] for it, t in zip(its, ts)]
        ti = torch.tensor(ts, device=dev)
        for nm, z in (("dprime", Z1), ("dpost", Z)):
            Dh, se, opt = heads[nm]; Dh.train(); lg, lab = d_forward(m, Dh, z, tg, se, ti)
            loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), lab.reshape(-1), ignore_index=-100); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if st % 500 == 0:
            log("copy_control step %d / %d (%d s)" % (st, steps, time.time() - t0))
    res = {"train": {"steps": steps, "bs": bs, "lr": 1e-4, "stream_seed": SEED_CC, "t_input": "learned step embedding (zero init) added to z, same for D' and D_post"},
           "dprime_value_acc": {}, "dpost_value_acc": {}, "value_floor": {}}
    for nm in heads:
        heads[nm][0].eval()
    for d in range(1, 9):
        K = d + 2; c = {nm: {t: [0, 0] for t in range(1, d + 1)} for nm in heads}; fl = {t: [0, 0] for t in range(1, d + 1)}
        for chunk in batches(TE[d][:n_per_d], bs):
            its = [PS.tok_item(x, enc) for x in chunk]; B = len(its); perm = [(i + 1) % B for i in range(B)]
            with torch.no_grad():
                _, zs, _ = run(m, prj, a, PS.collate(its, K, dev), K)
                for t in range(1, d + 1):
                    tg = [it["R"][t - 1] for it in its]; ti = torch.full((B,), t, device=dev)
                    for nm, z in (("dprime", zs[0]), ("dpost", zs[t - 1])):
                        Dh, se, _ = heads[nm]; lg, lab = d_forward(m, Dh, z, tg, se, ti)
                        if nm == "dprime":
                            _, plab = d_forward(m, Dh, z, [tg[p] for p in perm], se, ti); sc = value_scores(lg, lab, vt, perm_lab=plab)
                            fl[t][0] += sum(s[3] for s in sc); fl[t][1] += sum(s[0] for s in sc)
                        else:
                            sc = value_scores(lg, lab, vt)
                        c[nm][t][0] += sum(s[1] for s in sc); c[nm][t][1] += sum(s[0] for s in sc)
        for nm in heads:
            res[nm + "_value_acc"][str(d)] = {str(t): (v[0] / v[1] if v[1] else None) for t, v in c[nm].items()}
        res["value_floor"][str(d)] = {str(t): (v[0] / v[1] if v[1] else None) for t, v in fl.items()}
    return res


def cf_values(rec, don, t, mode="a"):
    """the recipient's ops after t, from the donor's state: mode a = back-references j <= t read the donor's S_j; b = j < t read the recipient's."""
    S = list(don["values"][:t + 1])
    if mode == "b":
        S = list(rec["values"][:t]) + [don["values"][t]]
    for k in range(t + 1, rec["d"] + 1):
        op, b, j = rec["ops"][k - 1], rec["operands"][k - 1], rec["src_j"][k - 1]
        S.append(NW.apply(op, S[k - 1], S[j] if op.endswith("_s") else b))
    return S[-1]


def kept_pairs(PC, d, cap):
    t = math.ceil(d / 2); out = []
    for i, (rec, don) in enumerate(PC[d]):
        cf = cf_values(rec, don, t)
        if len({rec["answer"], don["answer"], cf}) == 3:
            out.append((i, rec, don, cf, cf_values(rec, don, t, "b")))
        if len(out) >= cap:
            break
    return out, t


def part_swaps(m, prj, a, PC, enc, dev, bs, cap):
    res = {"t_rule": "ceil(d/2)", "pairs": {}, "items": {}, "self_bitwise": True}
    for d in range(3, 9):
        kp, t = kept_pairs(PC, d, cap); K = d + 2; res["pairs"][str(d)] = {"n_candidates": len(PC[d]), "n_kept": len(kp), "kept_share": None}
        scanned = kp[-1][0] + 1 if kp else 0; res["pairs"][str(d)]["kept_share"] = len(kp) / max(1, scanned)
        rows = []
        for cb in batches(list(range(len(kp))), bs):
            R = [PS.tok_item(kp[i][1], enc) for i in cb]; Dn = [PS.tok_item(kp[i][2], enc) for i in cb]; Fl = [PS.tok_item(kp[(i + 1) % len(kp)][2], enc) for i in cb]
            br, bd, bf = PS.collate(R, K, dev), PS.collate(Dn, K, dev), PS.collate(Fl, K, dev)
            assert br["enc"].shape == bd["enc"].shape and torch.equal(br["em"], bd["em"]), "pair token lengths differ"
            lp, zr, _ = run(m, prj, a, br, K); _, zd, kvd = run(m, prj, a, bd, K, want_kv=True); _, zf, _ = run(m, prj, a, bf, K)
            ne = br["em"].shape[1]
            pre = run(m, prj, a, br, K, z_in={j: zd[j - 1] for j in range(1, t + 1)})[0]
            flo = run(m, prj, a, br, K, z_in={j: zf[j - 1] for j in range(1, t + 1)})[0]
            slf = run(m, prj, a, br, K, z_in={j: zr[j - 1] for j in range(1, t + 1)})[0]
            stf = run(m, prj, a, br, K, z_in={j: zd[j - 1] for j in range(1, K + 1)})[0]
            caf = run(m, prj, a, br, K, lat_kv=kvd)[0]
            with torch.no_grad():                                                                    # the donor's encoder KV (for LEAK-SEG / MARKER-SWAP)
                Ld = kv_layers(m(input_ids=bd["enc"], attention_mask=bd["em"], use_cache=True).past_key_values)
            bot = ne - 1; bos = (bd["em"] == 1).float().argmax(1)                                   # left padded: the first real position is BOS

            def leak_patch(L):
                for li in range(len(L)):
                    L[li][0] = L[li][0].clone(); L[li][1] = L[li][1].clone(); L[li][0][:, :, bot - 6:bot] = Ld[li][0][:, :, bot - 6:bot]; L[li][1][:, :, bot - 6:bot] = Ld[li][1][:, :, bot - 6:bot]

            def marker_patch(L):
                ar = torch.arange(L[0][0].shape[0], device=dev)
                for li in range(len(L)):
                    L[li][0] = L[li][0].clone(); L[li][1] = L[li][1].clone()
                    for pos in (bos, torch.full_like(bos, bot)):
                        L[li][0][ar, :, pos] = Ld[li][0][ar, :, pos]; L[li][1][ar, :, pos] = Ld[li][1][ar, :, pos]
            lk = run(m, prj, a, br, K, enc_patch=leak_patch)[0]; mk = run(m, prj, a, br, K, enc_patch=marker_patch)[0]
            res["self_bitwise"] = bool(res["self_bitwise"] and torch.equal(slf, lp))
            P = {nm: preds_of(x, enc) for nm, x in (("plain", lp), ("prefix", pre), ("floor", flo), ("leak", lk), ("marker", mk), ("self", slf), ("state_full", stf), ("cache_full", caf))}
            for r_, i in enumerate(cb):
                pi, rec, don, cf, cfb = kp[i]
                rows.append({"pair": pi, "own": rec["answer"], "donor": don["answer"], "cf": cf, "cf_b": cfb, "floor_donor_pair": kp[(i + 1) % len(kp)][0],
                             **{nm: P[nm][r_] for nm in P}})
        res["items"][str(d)] = rows
    return res


def part_front(m, prj, a, PC, enc, dev, bs, cap):
    out = []; npairs = 0
    for d in range(3, 9):
        kp, _ = kept_pairs(PC, d, cap); sel = kp[::3]; npairs += len(sel); K = d + 2
        for cb in batches(list(range(len(sel))), bs):
            R = [PS.tok_item(sel[i][1], enc) for i in cb]; Dn = [PS.tok_item(sel[i][2], enc) for i in cb]; br, bd = PS.collate(R, K, dev), PS.collate(Dn, K, dev)
            _, zd, _ = run(m, prj, a, bd, K)
            for t in range(1, d):
                bstar = [None] * len(cb)
                for bgt in range(0, K - t + 1):
                    pr = preds_of(run(m, prj, a, br, K, z_in={j: zd[j - 1] for j in range(1, t + 1)}, n_lat=t + bgt)[0], enc)
                    for r_, i in enumerate(cb):
                        if bstar[r_] is None and pr[r_] == sel[i][2]["answer"]:
                            bstar[r_] = bgt
                for r_, i in enumerate(cb):
                    out.append([d, sel[i][0], t, bstar[r_] if bstar[r_] is not None else K - t + 1, bstar[r_] is not None])
    return {"rule": "every 3rd kept pair, t 1..d-1, budgets b 0..(d+2-t)", "items": out, "n_pairs": npairs, "n_obs": len(out)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--parts", default=",".join(PARTS))
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--per_d", type=int, default=1000, help="smokes only"); ap.add_argument("--pairs_per_d", type=int, default=1000)
    ap.add_argument("--cc_steps", type=int, default=5000, help="smokes only"); ap.add_argument("--cc_bs", type=int, default=64, help="smokes only (registered 64)"); ap.add_argument("--out", default="")
    a_ = ap.parse_args(); dev = a_.device; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; t00 = time.time()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2"); PS.VALUE_TOKS.update(PS.value_tokens(enc)); vt = torch.tensor(sorted(PS.VALUE_TOKS), device=dev)
    m, prj, D, a, A = load_run(a_.run, dev); TEF = NW.test_set(); TE = {d: v[:a_.per_d] for d, v in TEF.items()}; PC = NW.pair_candidates()
    excl = {it["sig"] for d in TEF.values() for it in d} | {x["sig"] for ps in PC.values() for pr in ps for x in pr}
    fo = a_.out or os.path.join(a_.run, "eval_ps.json"); out = json.load(open(fo)) if os.path.exists(fo) else {}
    out["meta"] = {"cell": A.get("cell"), "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
                   "driver": os.popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader").read().strip().split("\n")[0] if dev.startswith("cuda") else None,
                   "dtype": "fp32", "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                   "model_sha256": PS.sha_file(os.path.join(a_.run, "trainable.pt")), "args_sha256": PS.sha_file(os.path.join(a_.run, "args.json")),
                   "module_sha256": PS.sha_file(os.path.abspath(__file__)), "test_items_sha": NW.set_sha([it for d in sorted(TEF) for it in TEF[d]]),
                   "pair_candidates_sha": NW.set_sha([x for d in sorted(PC) for pr in PC[d] for x in pr]), "K_rule": "K = d + 2", "sec_by_part": out.get("meta", {}).get("sec_by_part", {}),
                   "per_d": a_.per_d, "pairs_per_d": a_.pairs_per_d}
    out["not_run"] = list(DESCRIPTIVE_NOT_RUN); m1 = bool(a.recode) and not a.direct; lat = not a.direct

    def save():
        json.dump(out, open(fo + ".tmp", "w", newline="\n")); os.replace(fo + ".tmp", fo)

    def log(s):
        print(s, flush=True)
    for part in a_.parts.split(","):
        t1 = time.time()
        if part == "acc":
            out["acc"] = part_acc(m, prj, a, TE, enc, dev, a_.bs)
        elif part == "k_sweep":
            out["k_sweep"] = part_acc(m, prj, a, TE, enc, dev, a_.bs, offsets=(-1, 0, 1, 2), ds=range(1, 9)) if lat else None
        elif part == "decode":
            out["decode"] = part_decode(m, prj, D, a, TE, enc, dev, a_.bs, vt, a_.per_d) if m1 else None
        elif part == "copy_control":
            out["copy_control"] = part_copy_control(m, prj, a, TE, enc, dev, a_.cc_bs, vt, excl, a_.cc_steps, a_.per_d, log) if m1 else None
        elif part == "answer_onset":
            out["answer_onset"] = part_onset(m, prj, D, a, TE, enc, dev, a_.bs, vt, a_.per_d) if m1 else None
        elif part == "swaps":
            out["swaps"] = part_swaps(m, prj, a, PC, enc, dev, a_.bs, a_.pairs_per_d) if lat else None
        elif part == "budget_front":
            out["budget_front"] = part_front(m, prj, a, PC, enc, dev, a_.bs, a_.pairs_per_d) if lat else None
        else:
            raise SystemExit("unknown part " + part)
        out["meta"]["sec_by_part"][part] = round(time.time() - t1); save(); log("PART %s done %d s" % (part, time.time() - t1))
    log("DONE %s %d s" % (fo, time.time() - t00))


if __name__ == "__main__":
    main()
