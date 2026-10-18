"""§218 LANG-VB evals (LANG_v1 §4.1 at 83f2dea4; Addendum 1; Note 218-1 8f2a2fcb): eval_lvb.json (A1 / A2 / A2-CTRL) and eval_direct.json
(DIRECT-SYM) on the SYM v1 test set (prosqa_sym.test_set: 1,000 per D, D 3..16), frozen model, fp32, tf32 off, on the stage's eval card (RTX 4060),
eval_bs per part (G-EVMEM). Each part is merged into the file as it finishes; every share has its per-item / per-pair outputs stored.
Candidate decision everywhere: sum log p("### o_R.") > sum log p("### o_P.") from the coda after K' passes (LangSLVA) or the plain forward (DIRECT).
  panel        G-L0's per-band panel, copied from scratch/gate_s218/g_l0.json (not recomputed).
  acc          one K' 24 rollout per chunk with codas at every K' 0..24: acc_k {K': {D}}, acc_d2 {D} (K' = D + 2), items {D: {K': [0/1]}},
               cand_ce_items {D} (at K' = D + 2), T_band {D} (T <= / > t_train_max, the max plain T in the run's log.jsonl).
  stall        stored V_t := V_{t-1} at t = ceil(D / 2); recovery = pooled acc(K' = D + 1, stalled) / acc(K' = D, intact), D 4..8; items per D.
  vswap        §2.5 arms on the frozen model at K' = D + 2, recipients = the test items of D 4..8, t from np rng 22610 (per item in test order: t_ii ~
               U{3..D-1}, then t_i ~ U{3..D-2} if D >= 5). Donors: prosqa_sym.premise_swap (ii_prime, i_prime, ii, reset's reference) and person_swap
               (person). ii_prime: stored V_1..V_t := the donor's (bank entries and SLVA source follow the stored entry); ii_prime_t2: the same at t = 2
               (descriptive, copied into ii_prime.t2_descriptive); i_prime: in every stored version 1..t the slots L_t = {c not an option : min(dist_R,
               dist_P) <= t + 1} take the donor's rows, every other slot the recipient's intact rows (D 5..8); ii: V_t := the donor's V_t only; person:
               V_1..V_t := the person-swap donor's, at t_ii (Note 218-1); reset: V_1..V_t := the recipient's version-0 slot rows. clean = recipient and
               (its arm's) donor correct at K' = D + 2; follow = the hybrid prefers the donor's answer (the recipient's o_P), own = it prefers o_R.
               exposed: ii_prime / ii / person = the donor's REACH frontier at t differs from the recipient's at some slot; i_prime = it differs inside
               L_t and both options lie outside L_t; reset = defined at (D, t).
               Every splice is a PRECOMPUTED stored entry (the recipient's / donor's intact-run S_s), never a splice into the live h; stored donor rows
               are normalised for SLVA by the RECIPIENT's rms0 (rollout_ls's nrm(st, rms0), the vbank convention).
  cf           CONSUMPTION-FRONT: <= 400 exposed per (D, j), D 4..8: decisive j 3..D-1 = prosqa_sym.cf_select(test items D 4..8, js = range(3, D)) with
               its default Random(22630) (G-L4b's exact call); descriptive j 1, 2, D = a second call with Random(22631); per pair the runs t = 0..D at K' = D with
               every version-0 read from pass t + 1 switched to x''s prelude (LangSLVA.v0_switch: core bank entry 0, SLVA's s = 0 key / value, the RMS
               reference of newly written versions); answers_by_t [answer = y]; t* = the least-violations change point (c in 0..D + 1, [ans = y] =
               [t >= c]; ties -> the smallest c); clean = exposed and x / x' each correct in their full runs at K' = D (endpoints_clean from the sweep reported);
               EVERY candidate (j's statement unique) stored with pos_ok / flip_ok / ok_rec / ok_don (Theory, score_s218 contract); per cell
               n_candidates / n_exposed (all test items), n_clean, curve {t: P(answer = y') over clean}, tstar, share_window (t* in {j-1, j}); decisive j 3..D-1;
               Addendum 1 item 4: clean_k2 = the same clean rule from full x / x' runs at K' = D + 2 (descriptive).
  hist24       intact / bank_ablate_slva (ver_keep: versions 1..t-2 masked in the core / coda bank and SLVA) / self_ablate (SLVA's own-slot versions
               1..t-2) / dup_newest (every reader's versions 1..t-2 read as the newest V_{t-1}) at K' = D + 2 and K' = 24, D 3..8, per-item [0/1].
  causal_deep  the §217 contract at D 9..16: arms ii_prime / i_prime (t from np rng 22612, the same per-item rule) + STALL; competent band B = {D : acc_d2
               >= 0.95} copied when acc is in the file.
  probe        PROBE(t), t in {0,1,2,3,4,5,6,7,8,10,12}, and PLANT(t) lin / xor, t in {0,1,2}, trained in LOCKSTEP on the same batches (one frozen rollout
               per batch shared by every probe): each probe = trainable copies of blocks 10-11 + the coda's beta / gamma (beta0 unused), reading V_t + bank
               versions 1..t-1 (its own ln_1 / c_attn on the stored versions); decoder path (blocks 0-9 over the decoder string) and the tied LM head
               frozen (Note 218-1). AdamW 1e-4 wd 0.01, 10,000 steps x bs 64 from a 200,000-item pool (Stream 22500, D U{3..8} from np rng 22500, after
               a 1,000-item held-out set = the stream's first draws; batch indices np rng 22501); held-out accuracy every 500 steps; a probe whose
               held-out accuracy gains > 0.02 over its last 2k steps is extended once to 20k. PROBE target: the gold string (cand_ce). PLANT: z ~ Bern(1/2)
               per item (np rng 22551 lin / 22552 xor during training; 22553 / 22554 for held-out and test), u = a unit vector in the span of V_t's
               principal components 40-60 (np rng 22550 Gaussian coefficients; xor's u2 orthogonalised to u), eps = the std of V_t along PC 50 (slot
               rows of the pool's first 2,000 items); both option-slot rows of V_t get + eps (2z - 1) u (xor: + eps (2 z1 - 1) u + eps (2 z2 - 1) u2);
               target "o1 if z = 1 else o2" (xor: z1 xor z2). Read on the test set D 3..12 (probe) / D 3..8 (plant); probe_valid {t} = lin and xor >= 0.90.
  fd           FD-MIRROR per-slot probes on the single row V_t[c] (features standardised by the PCA sample's mean / std of V_t), linear (logistic) and MLP
               (768-256-ReLU-1), Adam 1e-3, one streaming pass over 200,000 fresh items (Stream 22500, D U{6..8} from np rng 22560), positives dist(R, c) =
               t + k, negatives dist(P, c) = t + k (a slot at that distance from both is excluded); cells t in {1, 2}, k in 1..8 - t; fd_valid (s, 0), s in
               {3, 4}; fd_plant: V_1 + eps (2b - 1) u per real slot (PLANT(1)'s u / eps, recomputed identically from the probe pool; b ~ Bern(1/2), np rng 22561 train / 22562 test). AUROC on the
               test set D 6..8 per cell, pooled and by D; per-slot scores stored.
  direct       DIRECT-SYM: acc_by_D D 3..16, items, c1_band, d_star_fresh (§2.6).
    python -m boxes.langvb_eval --run boxes/runs_fsq/lvb_sym_s1 --parts acc,stall,vswap,cf,hist24,causal_deep,probe,fd --device cuda:0 --eval_bs 16
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import boxes.langvb as LV  # noqa: E402
import boxes.prosqa_sym as PS  # noqa: E402
import boxes.vbank as VB  # noqa: E402

T_SET = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12); PLANT_TS = (0, 1, 2)
PART_DS = {"acc": range(3, 17), "stall": range(4, 9), "vswap": range(4, 9), "cf": range(4, 9), "hist24": range(3, 9), "causal_deep": range(9, 17),
           "probe": range(3, 13), "fd": range(6, 9), "direct": range(3, 17)}


# ---------------------------------------------------------------- loading, items, scoring
def load(run, dev):
    from boxes import s5
    a = json.load(open(os.path.join(run, "args.json"))); direct = a.get("arch") == "direct"
    m = s5.load_gpt2(dev) if direct else LV.make_model(s5.load_gpt2(dev), a.get("table", "train")).to(dev)
    assert type(m).__name__ == a.get("arch_class", type(m).__name__), (type(m).__name__, a.get("arch_class"))
    m.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location=dev, weights_only=True), strict=True); m.eval()
    for p_ in m.parameters():
        p_.requires_grad_(False)
    return m, a


def untrained(dev, cal_n=64):
    """G-L5: LangSLVA on pretrained GPT-2 with step-0 heads, calibrated as the trainer does (rng / Stream 21110), c = 1e4 (step 0's regime)."""
    from boxes import s5
    torch.manual_seed(1); m = LV.make_model(s5.load_gpt2(dev)).to(dev); m.eval(); m.c_anneal.fill_(1e4)
    r_c = np.random.default_rng(21110); s_c = PS.Stream(21110, PS.exclusion_sigs()); its = [LV.lvb_item(s_c.draw(int(r_c.integers(3, 9)))) for _ in range(cal_n)]
    x, pad, lens, si, sp, dR, _ = LV.lvb_batch(its, dev)
    with torch.no_grad():
        cal = m.calibrate(x, pad, si, sp, dR, lens)
    for p_ in m.parameters():
        p_.requires_grad_(False)
    return m, {"arch": "lslva", "untrained": True, "cal_n": cal_n, "cal_max_m99": cal["max_m99"]}


def ev(it):
    """lvb_item + the fields the evals need: o1 / o2 names, option slot positions in slot order, dist_P, T."""
    e = LV.lvb_item(it); pos = {c: j for j, c in enumerate(e["concepts"])}
    e.update({"o1": it["names"][it["o1"]], "o2": it["names"][it["o2"]], "o1_slot": pos[it["o1"]], "o2_slot": pos[it["o2"]], "oR_slot": pos[it["o_R"]],
              "oP_slot": pos[it["o_P"]], "dist_P": it["dist_P"], "T": it["T"]})
    return e


def chunks(idx, bs):
    idx = list(idx)
    return [idx[i:i + bs] for i in range(0, len(idx), bs)]


def is_lslva(m):
    return isinstance(m, LV.LangSLVA)


@torch.no_grad()
def score_b(m, b, K, ts=None, edit=None, keepV=False):
    """b = lvb_batch(...) -> ({t: (sR, sP)} on the gold / other strings from the coda after pass t, the stored versions S if keepV)."""
    x, pad, lens, si, sp, dR, dP = b; dev = x.device; rows = torch.arange(x.shape[0], device=dev); ts = [K] if ts is None else list(ts)
    r = m.rollout_ls(x, pad, si, sp, K, coda_rows={t: rows for t in ts}, dec=dR, lens=lens, edit=edit); out = {}
    for t in ts:
        hP = m.coda_ls(t, r["V"][t], {i: r["kv"][i] for i in LV.CODA}, sp, dP, lens)
        out[t] = (m.dec_logprobs(r["coda"][t][1], sp, dR).sum(1), m.dec_logprobs(hP, sp, dP).sum(1))
    S = r["S"] if keepV else None
    del r
    return out, S


def ccand(sR, sP):
    return -torch.log_softmax(torch.stack([sR, sP], 1), 1)[:, 0]


def ablate_keep(t, nver):
    """BANK-ABLATE (vbank_eval.ablate_keep): at pass / coda t over versions 0..nver-1 keep V_0 and the newest (t - 1); versions 1..t-2 masked."""
    s = torch.arange(nver)
    return (s == 0) | (s >= t - 1)


def auroc(pos, neg):
    """Mann-Whitney AUROC with average ranks (ties 1/2)."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    x = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)]); order = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); i = 0; xs = x[order]
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(a, b):
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort").astype(float); rb = np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort").astype(float)

    def avg(v, rk):                                                         # average ranks for ties
        out = rk.copy()
        for u in np.unique(v):
            k = np.nonzero(v == u)[0]; out[k] = rk[k].mean()
        return out
    ra, rb = avg(np.asarray(a), ra), avg(np.asarray(b), rb)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------- acc / stall / hist24 / direct
def part_acc(m, TE, dev, bs, run):
    tmax = None
    lp = os.path.join(run, "log.jsonl") if run else None
    if lp and os.path.exists(lp):
        vs = [json.loads(l_).get("t_train_max") for l_ in open(lp)]; vs = [v for v in vs if v]; tmax = max(vs) if vs else None
    acc_k = {str(k): {} for k in range(25)}; items = {}; cc_items = {}; tb = {}
    for D, its in TE.items():
        A = np.zeros((len(its), 25), dtype=np.int64); C = np.zeros(len(its))
        for ch in chunks(range(len(its)), bs):
            sc, _ = score_b(m, LV.lvb_batch([its[j] for j in ch], dev), 24, ts=range(25))
            for k in range(25):
                A[ch, k] = (sc[k][0] > sc[k][1]).long().cpu().numpy()
            C[ch] = ccand(*sc[D + 2]).cpu().numpy() if D + 2 <= 24 else np.nan
        for k in range(25):
            acc_k[str(k)][str(D)] = float(A[:, k].mean())
        items[str(D)] = {str(k): [int(v) for v in A[:, k]] for k in range(25)}; cc_items[str(D)] = [round(float(v), 6) for v in C]
        if tmax is not None:
            T_ = np.array([it["T"] for it in its]); le = T_ <= tmax
            tb[str(D)] = {"le": float(A[le, min(D + 2, 24)].mean()) if le.any() else None, "gt": float(A[~le, min(D + 2, 24)].mean()) if (~le).any() else None,
                          "n_le": int(le.sum()), "n_gt": int((~le).sum())}
        print("ACC D", D, "K'=D+2", acc_k[str(min(D + 2, 24))][str(D)], flush=True)
    return {"acc_d2": {str(D): acc_k[str(D + 2)][str(D)] for D in TE}, "acc_k": acc_k, "items": items, "cand_ce_items": cc_items, "T_band": tb,
            "t_train_max": tmax, "acc_def": "candidate accuracy from the coda after K' passes, one K' 24 rollout per chunk; items in test order per D"}


def stall_by(m, TE, Ds, dev, bs, ts_override=None):
    """ts_override (self-check only): a stall pass beyond K' = D + 1 never fires, so 'stalled' must equal the intact run at K' = D + 1."""
    by = {}; items = {}; si_ = st_ = cnt = 0
    for D in Ds:
        its = TE[D]; ts = ts_override or math.ceil(D / 2); I_ = np.zeros(len(its), dtype=np.int64); S_ = np.zeros(len(its), dtype=np.int64)
        for ch in chunks(range(len(its)), bs):
            b = LV.lvb_batch([its[j] for j in ch], dev); st = {}

            def edit(t, h):
                if t == ts:
                    return st[t - 1]
                st[t] = h
                return None
            sc, _ = score_b(m, b, D); I_[ch] = (sc[D][0] > sc[D][1]).long().cpu().numpy()
            sc, _ = score_b(m, b, D + 1, edit=edit)                         # ts >= 2 (D >= 4), so st[ts - 1] is a pass output
            S_[ch] = (sc[D + 1][0] > sc[D + 1][1]).long().cpu().numpy(); st.clear()
        by[str(D)] = {"intact_kD": float(I_.mean()), "stalled_kD1": float(S_.mean()), "stall_t": ts, "n_items": len(its), "n": len(its),
                      "recovery": float(S_.sum() / I_.sum()) if I_.sum() else None}
        items[str(D)] = {"i": list(range(len(its))), "intact": [int(v) for v in I_], "stalled": [int(v) for v in S_]}
        si_ += int(I_.sum()); st_ += int(S_.sum()); cnt += len(its)
    return {"recovery": (st_ / si_) if si_ else None, "by_D": by, "items": items,
            "def": "stored V_t := V_{t-1} at t = ceil(D / 2); recovery = pooled acc(K' = D + 1, stalled) / pooled acc(K' = D, intact)"}


def part_stall(m, TE, dev, bs):
    return {"stall": stall_by(m, TE, range(4, 9), dev, bs)}


def part_contamination(run):
    """CONTAMINATED inputs (v1 §5.2) from the run's log.jsonl: test / GATE signatures among the plain draws (the trainer counts every drawn item's sig
    against the exclusion set; 0 by construction of prosqa_sym.Stream) and self_reuse; the pool's own signature hits are in pool_meta / the labeller part."""
    lp = os.path.join(run, "log.jsonl") if run else ""
    if not os.path.exists(lp):
        return {"contamination": None, "contamination_note": "no log.jsonl"}
    h = r_ = None
    for l_ in open(lp):
        d = json.loads(l_)
        if "test_gate_sig_hits" in d:
            h = max(h or 0, d["test_gate_sig_hits"])
        if "self_reuse" in d:
            r_ = max(r_ or 0, d["self_reuse"])
    return {"contamination": {"test_gate_sig_hits": h, "self_reuse": r_},
            "contamination_note": "max over log.jsonl records; test_gate_sig_hits counts plain draws whose role-symmetric signature is in the test + GATE set"}


def part_hist24(m, TE, dev, bs):
    res = {"items": {}}; arms = (("intact", None), ("bank_ablate_slva", "keep"), ("self_ablate", "self"), ("dup_newest", "dup"))
    for arm, how in arms:
        try:
            if how == "keep":
                m.ver_keep = ablate_keep
            elif how == "self":
                m.gather_hist_ablate = "self"
            elif how == "dup":
                m.dup_newest = True
            a2 = {}; a24 = {}; it_ = {}
            for D in range(3, 9):
                its = TE[D]; A2 = np.zeros(len(its), dtype=np.int64); A24 = np.zeros(len(its), dtype=np.int64)
                for ch in chunks(range(len(its)), bs):
                    sc, _ = score_b(m, LV.lvb_batch([its[j] for j in ch], dev), 24, ts=[D + 2, 24])
                    A2[ch] = (sc[D + 2][0] > sc[D + 2][1]).long().cpu().numpy(); A24[ch] = (sc[24][0] > sc[24][1]).long().cpu().numpy()
                a2[str(D)] = float(A2.mean()); a24[str(D)] = float(A24.mean()); it_[str(D)] = {"ok_d2": [int(v) for v in A2], "ok_k24": [int(v) for v in A24]}
        finally:
            m.ver_keep = None; m.gather_hist_ablate = None; m.dup_newest = False
        res[arm] = {"acc_d2": a2, "acc_k24": a24}; res["items"][arm] = it_; print("HIST24", arm, res[arm], flush=True)
    res["def"] = ("§213-21 arms on LangSLVA at K' = D + 2 and 24 from one K' 24 rollout per chunk, D 3..8: bank_ablate_slva = ver_keep ablate_keep (versions "
                  "1..t-2 masked in the core / coda bank and SLVA), self_ablate = SLVA own-slot versions 1..t-2, dup_newest = readers' versions 1..t-2 := V_{t-1}; "
                  "items {arm: {D: {ok_d2, ok_k24}}} in test order")
    return {"hist24": res}


@torch.no_grad()
def part_direct(m, TE, dev, bs):
    acc = {}; items = {}
    for D, its in TE.items():
        ok = []
        for ch in chunks(range(len(its)), bs):
            sR, sP = LV.cand_scores(m, [its[j] for j in ch], 0, dev, lambda: torch.autocast("cuda", enabled=False)); ok += (sR > sP).long().tolist()
        acc[str(D)] = float(np.mean(ok)); items[str(D)] = ok; print("DIRECT D", D, acc[str(D)], flush=True)
    c1 = [D for D in range(5, 9) if str(D) in acc and all(acc[str(d)] <= 0.65 for d in range(D, 9))]
    dst = max([D for D in range(3, 13) if acc.get(str(D), 0) > 0.65], default=None)
    return {"acc_by_D": acc, "items": items, "c1_band": c1, "d_star_fresh": dst,
            "direct_def": "§2.6: C1 = {D in 5..8 : acc(D') <= 0.65 for every D' in D..8}; d_star_fresh = the largest D in 3..12 with acc > 0.65"}


# ---------------------------------------------------------------- swaps (vswap, causal_deep)
def frontier(dist, cs, t):
    return [(dist.get(c) is not None and dist[c] <= t) for c in cs]


def Lmask(rec_raw, cs, t):
    return [(c not in (rec_raw["o_R"], rec_raw["o_P"])) and min(rec_raw["dist_R"].get(c, 10 ** 9), rec_raw["dist_P"].get(c, 10 ** 9)) <= t + 1 for c in cs]


def mk_edit(kind, tt, Vsrc, Vrec, Lm=None):
    def edit(s, h):
        sel = (tt == s) if kind == "ii" else (s <= tt)
        if not bool(sel.any()):
            return None
        if kind == "reset":
            new = Vrec[0]
        elif kind == "i_prime":
            new = torch.where(Lm[:, :, None], Vsrc[s], Vrec[s])
        else:
            new = Vsrc[s]
        return torch.where(sel[:, None, None], new.to(h.dtype), h)
    return edit


def summarise(pairs, by_t_key="t"):
    ok_ = lambda p: p["ok_rec"] and (p["ok_don"] is None or p["ok_don"])   # noqa: E731  (RESET has no donor: ok_don None, clean = ok_rec)
    ex = [p for p in pairs if p["exposed"]]; cl = [p for p in ex if ok_(p)]
    fol = lambda ps: (float(np.mean([p["y_hyb"] for p in ps])) if ps else None)  # noqa: E731
    byD = {}
    for D in sorted({p["D"] for p in pairs}):
        e_ = [p for p in ex if p["D"] == D]; c_ = [p for p in e_ if ok_(p)]
        byD[str(D)] = {"n_candidates": sum(1 for p in pairs if p["D"] == D), "n_exposed": len(e_), "n_clean": len(c_), "follow": fol(c_), "own": None if not c_ else 1 - fol(c_)}
    byt = {}
    for t in sorted({p[by_t_key] for p in pairs}):
        c_ = [p for p in cl if p[by_t_key] == t]; byt[str(t)] = {"n_clean": len(c_), "follow": fol(c_)}
    return {"n_candidates": len(pairs), "n_exposed": len(ex), "n_clean": len(cl), "follow": fol(cl), "own": None if not cl else 1 - fol(cl), "by_D": byD, "by_t": byt, "pairs": pairs}


def swap_arms(m, TE, Ds, seed, arms, dev, bs):
    rng = np.random.default_rng(seed); TT = {}
    for D in Ds:
        for i in range(len(TE[D])):
            t1 = int(rng.integers(3, D)); t2 = int(rng.integers(3, D - 1)) if D >= 5 else None; TT[(D, i)] = (t1, t2)
    P = {a: [] for a in arms}
    for D in Ds:
        raw = TE["raw"][D]; K = D + 2
        for ch in chunks(range(len(raw)), bs):
            rec = [raw[i] for i in ch]; R = [TE[D][i] for i in ch]; don = [PS.premise_swap(it) for it in rec]; Dn = [ev(d) for d in don]
            bR = LV.lvb_batch(R, dev); bD = LV.lvb_batch(Dn, dev); assert torch.equal(bR[1], bD[1]) and torch.equal(bR[3], bD[3]), "swap positions"
            scR, VR = score_b(m, bR, K, keepV=True); scD, VD = score_b(m, bD, K, keepV=True)
            okR = (scR[K][0] > scR[K][1]).tolist(); okD = (scD[K][0] > scD[K][1]).tolist(); cs = [r_["concepts"] for r_ in R]
            t1 = torch.tensor([TT[(D, i)][0] for i in ch], device=dev); S = bR[3].shape[1]
            for arm in arms:
                if arm == "i_prime" and D < 5:
                    continue
                if arm == "person":
                    per = [PS.person_swap(it) for it in rec]; Pn = [ev(d) for d in per]; bP = LV.lvb_batch(Pn, dev)
                    assert torch.equal(bR[1], bP[1]) and torch.equal(bR[3], bP[3]), "person positions"
                    scP, VP = score_b(m, bP, K, keepV=True); okS = (scP[K][0] > scP[K][1]).tolist(); Vsrc = VP; dons = per
                else:
                    okS = okD; Vsrc = VD; dons = don
                if arm == "i_prime":
                    tt = torch.tensor([TT[(D, i)][1] for i in ch], device=dev)
                    Lm = torch.zeros(len(ch), S, dtype=torch.bool, device=dev)
                    for q, i in enumerate(ch):
                        Lm[q, :len(cs[q])] = torch.tensor(Lmask(rec[q], cs[q], TT[(D, i)][1]), device=dev)
                    ed = mk_edit("i_prime", tt, Vsrc, VR, Lm)
                elif arm == "ii_prime_t2":
                    tt = torch.full_like(t1, 2); ed = mk_edit("ii_prime", tt, Vsrc, VR)
                else:
                    tt = t1; ed = mk_edit({"ii_prime": "ii_prime", "ii": "ii", "person": "ii_prime", "reset": "reset"}[arm], tt, Vsrc, VR)
                sc, _ = score_b(m, bR, K, edit=ed); yh = (sc[K][1] > sc[K][0]).tolist()
                for q, i in enumerate(ch):
                    t_ = int(tt[q]); fR = frontier(rec[q]["dist_R"], cs[q], t_); fD = frontier(dons[q]["dist_R"], cs[q], t_)
                    fl = {"pos_ok": 1}                                      # positions asserted per chunk (premise_swap / person_swap and lvb_batch)
                    if arm == "i_prime":
                        L_ = Lmask(rec[q], cs[q], t_); lt = any(a_ != b_ for a_, b_, l_ in zip(fR, fD, L_) if l_)
                        oo = all(min(rec[q]["dist_R"].get(o, 10 ** 9), rec[q]["dist_P"].get(o, 10 ** 9)) > t_ + 1 for o in (rec[q]["o_R"], rec[q]["o_P"]))   # >= 2 hops beyond t
                        exp_ = lt and oo; fl.update(lt_differs=int(lt), opts_outside=int(oo))
                    elif arm == "reset":
                        exp_ = True
                    else:
                        fd_ = any(a_ != b_ for a_, b_ in zip(fR, fD)); fl["frontier_differs"] = int(fd_)
                        exp_ = True if arm == "person" else fd_                # PERSON (descriptive): exposed = pos_ok, as the scorer recomputes
                    P[arm].append({"rec": i, "don": "person" if arm == "person" else "premise", "D": D, "t": t_, "y_hyb": int(yh[q]),
                                   "ok_rec": int(okR[q]), "ok_don": None if arm == "reset" else int(okS[q]), "exposed": int(exp_), **fl})
                if arm == "person":
                    del VP
            del VR, VD
        print("SWAP D", D, {a: summarise([p for p in P[a] if p["D"] == D])["follow"] for a in arms}, flush=True)
    return {a: summarise(P[a]) for a in arms}


def part_vswap(m, TE, dev, bs):
    out = swap_arms(m, TE, range(4, 9), 22610, ("ii_prime", "ii_prime_t2", "i_prime", "ii", "person", "reset"), dev, bs)
    out["ii_prime"]["t2_descriptive"] = {k: v for k, v in out["ii_prime_t2"].items() if k != "pairs"}   # the t = 2 pairs stay in ii_prime_t2.pairs
    out["def"] = __doc__.split("  vswap")[1].split("  cf ")[0].strip()
    return {"vswap": out}


def part_causal_deep(m, TE, dev, bs, prev):
    Ds = [D for D in range(9, 17) if D in TE]
    cd = swap_arms(m, TE, Ds, 22612, ("ii_prime", "i_prime"), dev, bs); cd["stall"] = stall_by(m, TE, Ds, dev, bs)
    if "acc_d2" in prev:
        cd["band_B"] = [int(D) for D in Ds if prev["acc_d2"].get(str(D), 0) >= 0.95]
    cd["def"] = "the §217 contract at D 9..16: ii_prime / i_prime as vswap (t from np rng 22612) + STALL at t = ceil(D / 2); band_B = {D : acc_d2 >= 0.95}"
    return {"causal_deep": cd}


# ---------------------------------------------------------------- CONSUMPTION-FRONT
def tstar(ans):
    D = len(ans) - 1; best = None
    for c in range(0, D + 2):
        v = sum(1 for t, a in enumerate(ans) if a != int(t >= c))
        if best is None or v < best[0]:
            best = (v, c)
    return best[1]


def part_cf(m, TE, dev, bs):
    """CONSUMPTION-FRONT (module docstring). Every CANDIDATE pair (recipient D 4..8, j 1..D, prosqa_sym.cf_donor not None) is stored with pos_ok (equal T
    and first-mention slots), flip_ok, ok_rec / ok_rec_d2 (x's full run at K' = D / D + 2) and, for flip_ok donors, ok_don / ok_don_d2 (x''s full run; a
    pos-failing donor runs with its own slots); the selected exposed pairs (<= 400 per (D, j)) also carry answers_by_t (t = 0..D, K' = D) and tstar."""
    import random
    Ds = [D for D in range(4, 9) if D in TE]; raw = [it for D in Ds for it in TE["raw"][D]]; base = {}; off = 0
    for D in Ds:
        base[D] = off; off += len(TE["raw"][D])
    sel = PS.cf_select(raw, js=lambda D: range(3, D))                       # decisive j 3..D-1: G-L4b's exact call (one shared Random(22630), sorted cells)
    sel.update(PS.cf_select(raw, rng=random.Random(22631), js=lambda D: [1, 2, D]))   # descriptive j 1, 2, D: a separate call (Audit G-L13 (f))
    picked = {(i, r_["j"]): r_ for lst in sel.values() for i, r_ in lst}
    okx = {}                                                                # x's full runs: K' = D and D + 2
    for D in Ds:
        its = TE[D]
        for ch in chunks(range(len(its)), bs):
            sc, _ = score_b(m, LV.lvb_batch([its[q] for q in ch], dev), D + 2, ts=[D, D + 2])
            for q, i in enumerate(ch):
                okx[base[D] + i] = (int(sc[D][0][q] > sc[D][1][q]), int(sc[D + 2][0][q] > sc[D + 2][1][q]))
    cand = []
    for i, it in enumerate(raw):
        for j in range(1, it["D"] + 1):
            r_ = picked.get((i, j)) or PS.cf_donor(it, j)
            if r_ is not None:
                cand.append((i, j, r_))
    okd = {}                                                                # x''s full runs for flip_ok candidates
    flips = [(i, j, r_) for i, j, r_ in cand if r_["flip"]]
    for ch in chunks(range(len(flips)), bs):
        grp = {}
        for q in ch:
            grp.setdefault(raw[flips[q][0]]["D"], []).append(q)
        for D, qs in grp.items():
            sc, _ = score_b(m, LV.lvb_batch([ev(flips[q][2]["donor"]) for q in qs], dev), D + 2, ts=[D, D + 2])
            for k, q in enumerate(qs):
                okd[(flips[q][0], flips[q][1])] = (int(sc[D][0][k] > sc[D][1][k]), int(sc[D + 2][0][k] > sc[D + 2][1][k]))
    sweep = {}
    for (D, j), lst in sorted(sel.items()):
        for ch in chunks(range(len(lst)), bs):
            ps = [lst[q] for q in ch]; R = [ev(raw[i]) for i, _ in ps]; Dn = [ev(r_["donor"]) for _, r_ in ps]
            bR = LV.lvb_batch(R, dev); bD = LV.lvb_batch(Dn, dev); assert torch.equal(bR[1], bD[1]) and torch.equal(bR[3], bD[3]), "CF positions"
            with torch.no_grad():
                H6sw = m.prelude(bD[0])
            A = []
            for t in range(D + 1):
                m.v0_switch = (t, H6sw, None)
                try:
                    sc, _ = score_b(m, bR, D)
                finally:
                    m.v0_switch = None
                A.append((sc[D][0] > sc[D][1]).long().tolist())
            del H6sw
            for q, (i, r_) in enumerate(ps):
                sweep[(i, j)] = [A[t][q] for t in range(D + 1)]
        print("CF", D, j, len(lst), flush=True)
    pairs = []
    for i, j, r_ in cand:
        D = raw[i]["D"]; od = okd.get((i, j), (None, None)); ox = okx[i]
        p = {"rec": i - base[D], "don": "cf_j%d" % j, "D": D, "j": j, "pos_ok": int(r_["same_len"] and r_["same_slots"]), "flip_ok": int(r_["flip"]),
             "exposed": int(r_["exposed"]), "ok_rec": ox[0], "ok_rec_d2": ox[1], "ok_don": od[0], "ok_don_d2": od[1], "selected": int((i, j) in sweep)}
        p["clean"] = int(bool(p["exposed"] and ox[0] and od[0])); p["clean_k2"] = int(bool(p["exposed"] and ox[1] and od[1]))
        if (i, j) in sweep:
            ab = sweep[(i, j)]; p.update({"answers_by_t": ab, "tstar": tstar(ab), "endpoints_clean": int(ab[D] == 1 and ab[0] == 0)})
        pairs.append(p)
    out = {}
    for D in Ds:
        for j in range(1, D + 1):
            ps = [p for p in pairs if p["D"] == D and p["j"] == j]; ex = [p for p in ps if p["exposed"]]; sl = [p for p in ex if p["selected"]]
            cl = [p for p in sl if p["clean"]]
            out.setdefault(str(D), {})[str(j)] = {
                "n_candidates": len(ps), "n_exposed": len(ex), "n_selected": len(sl), "n_clean": len(cl),
                "curve": {str(t): (float(np.mean([1 - p["answers_by_t"][t] for p in cl])) if cl else None) for t in range(D + 1)},
                "tstar": [p["tstar"] for p in cl], "share_window": (float(np.mean([p["tstar"] in (j - 1, j) for p in cl])) if cl else None),
                "share_clean_kd": (len(cl) / len(sl)) if sl else None, "share_clean_kd2": (sum(p["clean_k2"] for p in sl) / len(sl)) if sl else None}
    dec = [p for p in pairs if p.get("selected") and p["clean"] and 3 <= p["j"] <= p["D"] - 1]
    sp = {str(D): spearman([p["tstar"] for p in dec if p["D"] == D], [p["j"] for p in dec if p["D"] == D]) for D in Ds}
    lh = {str(D): {"tstar_hist": {str(c): sum(1 for p in pairs if p.get("selected") and p["clean"] and p["D"] == D and p["j"] == D and p["tstar"] == c)
                                  for c in range(D + 2)}, "curve": out[str(D)][str(D)]["curve"]} for D in Ds}
    out.update({"spearman_within_D": sp, "n_decisive_clean_by_D": {str(D): sum(1 for p in dec if p["D"] == D) for D in Ds},
                "share_window_pooled": (float(np.mean([p["tstar"] in (p["j"] - 1, p["j"]) for p in dec])) if dec else None), "n_decisive_clean": len(dec),
                "last_hop": lh, "pairs": pairs, "endpoint_mismatch": sum(1 for p in pairs if "endpoints_clean" in p and p["endpoints_clean"] != p["clean"]),
                "def": __doc__.split("  cf ")[1].split("  hist24")[0].strip()})
    return {"cf": out}


# ---------------------------------------------------------------- probes and plants (lockstep)
class Src:
    """the probe's own copies of blocks 10-11, the frozen trunk otherwise (vbank_eval.Src)."""
    def __init__(self, g, P):
        self.g, self.P, self.H = g, P, g.H

    def w(self, k):
        kk = k.replace(".", "_")
        return self.P[kk] if kk in self.P else self.g.w(k)


class Probe:
    def __init__(self, m, lr):
        g = m.g; self.P = torch.nn.ParameterDict({k: torch.nn.Parameter(v.detach().clone()) for k, v in g.p.items() if k.startswith(("h_10_", "h_11_"))})
        self.beta = torch.nn.Parameter(m.beta[4:].detach().clone()); self.gam = torch.nn.Parameter(m.gam[4:].detach().clone()); self.beta0 = m.beta0[4:].detach()
        self.src = Src(g, self.P); self.opt = torch.optim.AdamW(list(self.P.parameters()) + [self.beta, self.gam], lr=lr, weight_decay=0.01)
        self.curve = []; self.steps = 0; self.active = True

    def scores(self, m, t, S, Vt, sp, dA, dB, lens):
        kvs = {i: [(None, None)] + [VB.bank_kv(self.src, i, S[s]) for s in range(1, t)] for i in LV.CODA}; bp = (self.beta, self.beta0, self.gam)
        hA = m.coda_ls(t, Vt, kvs, sp, dA, lens, src=self.src, bp=bp); hB = m.coda_ls(t, Vt, kvs, sp, dB, lens, src=self.src, bp=bp)
        return m.dec_logprobs(hA, sp, dA).sum(1), m.dec_logprobs(hB, sp, dB).sum(1)


def draw_items(seed, D_lo, D_hi, n, rng_seed, excl):
    s = PS.Stream(seed, excl); rng = np.random.default_rng(rng_seed)
    return [ev(s.draw(int(D))) for D in rng.integers(D_lo, D_hi + 1, size=n)], s


@torch.no_grad()
def stored(m, its, K, dev, chunk):
    """the frozen model's stored versions S_0..S_K for its (chunked rollouts; the batch padded as one lvb_batch so the probe's rows line up)."""
    b = LV.lvb_batch(its, dev); x, pad, lens, si, sp = b[:5]; parts = []
    for ch in chunks(range(len(its)), chunk):
        r = m.rollout_ls(x[ch], pad[ch], si[ch], sp[ch], K); parts.append(r["S"]); del r
    S = [torch.cat([p_[k] for p_ in parts], 0) for k in range(K + 1)]
    return b, S


def plant_dirs(m, pool, t_list, dev, chunk, n=2000):
    """PCA of V_t's real slot rows over pool[:n] -> {t: (u, u2, eps, eigvals[:60], mean, std)}."""
    out = {}; rng = np.random.default_rng(22550); tm = max(t_list)
    rows = {t: [] for t in t_list}
    for ch in chunks(range(min(n, len(pool))), chunk * 4):
        b, S = stored(m, [pool[j] for j in ch], tm, dev, chunk); sp = b[4]
        for t in t_list:
            rows[t].append(S[t][sp].double().cpu())
        del S
    for t in t_list:
        X = torch.cat(rows[t], 0); mu = X.mean(0); sd = X.std(0).clamp_min(1e-6); Xc = X - mu; C = Xc.T @ Xc / (len(X) - 1)
        ev_, E = torch.linalg.eigh(C); ev_, E = ev_.flip(0), E.flip(1)
        c1 = torch.tensor(rng.standard_normal(21)); c2 = torch.tensor(rng.standard_normal(21)); B_ = E[:, 39:60]
        u = B_ @ c1; u = u / u.norm(); u2 = B_ @ c2; u2 = u2 - (u2 @ u) * u; u2 = u2 / u2.norm(); eps = float(ev_[49].clamp_min(0).sqrt())
        vn = float(X.norm(dim=1).mean())                                    # Addendum 1 item 5: eps relative to ||V_t|| (mean slot-row norm)
        out[t] = {"u": u.float().to(dev), "u2": u2.float().to(dev), "eps": eps, "eps_rel": eps / vn, "vnorm": vn, "eig": [float(v) for v in ev_[:60]],
                  "mean": mu.float().to(dev), "std": sd.float().to(dev)}
    return out


def plant_V(Vt, its, z, d, xor=False, z2=None):
    """both option-slot rows of Vt (B, S, D) get + eps (2z - 1) u (xor: + eps (2 z2 - 1) u2 as well)."""
    V = Vt.clone(); B = V.shape[0]; ar = torch.arange(B, device=V.device)
    add = d["eps"] * (2 * z - 1).float()[:, None] * d["u"][None]
    if xor:
        add = add + d["eps"] * (2 * z2 - 1).float()[:, None] * d["u2"][None]
    for key in ("o1_slot", "o2_slot"):
        j = torch.tensor([it[key] for it in its], device=V.device); V[ar, j] = V[ar, j] + add.to(V.dtype)
    return V


def probe_forward(m, pr, kind, t, its, b, S, dirs, zs, dev):
    """-> (scores of string A, string B, target index (B,)). PROBE: A = gold, B = other, target 0. PLANT: A = o1, B = o2, target 0 iff z (xor: z1 ^ z2) = 1."""
    x, pad, lens, si, sp, dR, dP = b; Vt = S[t]
    if kind == "probe":
        sA, sB = pr.scores(m, t, S, Vt, sp, dR, dP, lens); return sA, sB, torch.zeros(len(its), dtype=torch.long, device=dev)
    d1 = torch.tensor([LV.dec_ids(it["o1"]) for it in its], device=dev); d2 = torch.tensor([LV.dec_ids(it["o2"]) for it in its], device=dev)
    z1, z2 = zs; V = plant_V(Vt, its, z1, dirs[t], kind == "xor", z2); y = (z1 ^ z2) if kind == "xor" else z1
    sA, sB = pr.scores(m, t, S, V, sp, d1, d2, lens)
    return sA, sB, (1 - y).long()


def part_probe(m, TE, dev, bs, ts=T_SET, pts=PLANT_TS, steps=10000, pool_n=200000, pbs=64, chunk=16, hold_n=1000, pool_override=None, eval_every=500, group=0):
    """group > 0 (G-EVMEM: 17 lockstep probes do not fit 8 GB): probes / plants ordered by t then kind are trained in groups of `group`, each group
    replaying the SAME batch sequence (np rng 22501 restarted) - so every probe still sees the same items in the same order; only the rollout is shared
    within a group. Plant z: np rng 22551 / 22552 restarted per group (train), 22553 / 22554 restarted per probe (held-out and test)."""
    t0 = time.time()
    if pool_override is not None:                                           # G-EVMEM only: fixed (deepest) items
        held, pool = pool_override[:hold_n], pool_override
    else:
        excl = PS.exclusion_sigs(); s = PS.Stream(22500, excl); rng = np.random.default_rng(22500); Dseq = rng.integers(3, 9, size=hold_n + pool_n)
        held = [ev(s.draw(int(D))) for D in Dseq[:hold_n]]; pool = [ev(s.draw(int(D))) for D in Dseq[hold_n:]]
    print("PROBE pool", len(pool), "held", len(held), round(time.time() - t0), "s", flush=True)
    dirs = plant_dirs(m, pool, list(pts), dev, chunk) if pts else {}
    keys = sorted([("probe", t) for t in ts] + [(k, t) for t in pts for k in ("lin", "xor")], key=lambda k: (k[1], k[0])); prs = {}
    rb = rz = None; seen = set()                                            # pool indices drawn (Note 218-6)

    def zdraw(r_, n):
        return torch.tensor(r_.integers(0, 2, size=n), device=dev), torch.tensor(r_.integers(0, 2, size=n), device=dev)

    def evaluate(its, rzs, sel_keys, Kmax):
        """accuracy per probe over its (+ per-item [0/1])."""
        ok = {k: [] for k in sel_keys}; zz = {k: zdraw(rzs[k[0]], len(its)) for k in sel_keys if k[0] != "probe"}
        for ch in chunks(range(len(its)), bs):
            sub = [its[j] for j in ch]; b, S = stored(m, sub, Kmax, dev, chunk)
            with torch.no_grad():
                for k in sel_keys:
                    zs = (zz[k][0][ch], zz[k][1][ch]) if k in zz else None
                    sA, sB, y = probe_forward(m, prs[k], k[0], k[1], sub, b, S, dirs, zs, dev); ok[k] += ((sA > sB).long() == (1 - y)).long().tolist()
            del S
        return ok

    def train(sel_keys, n_steps):
        Kmax = max(k[1] for k in sel_keys)
        for _ in range(n_steps):
            ix = rb.integers(0, len(pool), pbs); seen.update(int(j) for j in ix); sub = [pool[j] for j in ix]; b, S = stored(m, sub, Kmax, dev, chunk)
            for k in sel_keys:
                pr = prs[k]; zs = zdraw(rz[k[0]], pbs) if k[0] != "probe" else None
                sA, sB, y = probe_forward(m, pr, k[0], k[1], sub, b, S, dirs, zs, dev)
                loss = F.cross_entropy(torch.stack([sA, sB], 1), y); pr.opt.zero_grad(set_to_none=True); loss.backward(); pr.opt.step(); pr.steps += 1
            del S
            st = prs[sel_keys[0]].steps
            if st % eval_every == 0:
                rh = {"lin": np.random.default_rng(22553), "xor": np.random.default_rng(22554)}; ok = evaluate(held, rh, sel_keys, Kmax)
                for k in sel_keys:
                    prs[k].curve.append((st, float(np.mean(ok[k]))))
                print("PROBE step", st, {"%s%d" % k: round(prs[k].curve[-1][1], 3) for k in sel_keys}, round(time.time() - t0), "s", flush=True)
    def read_group(gk, out_p, out_pl):
        """the test-set readout of each probe in gk (plant z from np rng 22553 / 22554, restarted per probe)."""
        for k in gk:
            Ds = [D for D in range(3, 13 if k[0] == "probe" else 9) if D in TE]; its = [it for D in Ds for it in TE[D]]; Dl = [it["D"] for it in its]
            rt = {"lin": np.random.default_rng(22553), "xor": np.random.default_rng(22554)}
            ok = evaluate(its, rt, [k], k[1])[k]; byD = {str(D): float(np.mean([o for o, d in zip(ok, Dl) if d == D])) for D in Ds}
            rec = {"acc_by_D": byD, "items": {str(D): [o for o, d in zip(ok, Dl) if d == D] for D in Ds}, "curve": prs[k].curve, "steps_run": prs[k].steps}
            if k[0] == "probe":
                out_p[str(k[1])] = rec
            else:
                d = out_pl.setdefault(str(k[1]), {"eps": dirs[k[1]]["eps"], "eps_rel": dirs[k[1]]["eps_rel"], "pcs": dirs[k[1]]["eig"]})
                rec["acc"] = float(np.mean(ok)); rec["items_flat"] = ok; d[k[0]] = rec
            print("PROBE read", k, byD, flush=True)
    out_p = {}; out_pl = {}; G = group if group and group < len(keys) else len(keys); groups = [keys[i:i + G] for i in range(0, len(keys), G)]
    for gk in groups:
        prs = {k: Probe(m, 1e-4) for k in gk}; rb = np.random.default_rng(22501); rz = {"lin": np.random.default_rng(22551), "xor": np.random.default_rng(22552)}
        print("PROBE group", gk, flush=True); train(gk, steps)
        ext = [k for k in gk if len(prs[k].curve) >= 5 and prs[k].curve[-1][1] - prs[k].curve[-5][1] > 0.02]   # curve every 500: [-5] = 2k steps earlier
        if ext:
            print("PROBE extend to %d:" % (2 * steps), ext, flush=True); train(ext, steps)
        read_group(gk, out_p, out_pl)
        prs = {}
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
    valid = {t: bool(out_pl[t].get("lin", {}).get("acc", 0) >= 0.90 and out_pl[t].get("xor", {}).get("acc", 0) >= 0.90) for t in out_pl}
    proto = __doc__.split("  probe ")[1].split("  fd ")[0].strip()
    return {"probe": out_p, "plant": out_pl, "probe_valid": valid, "probe_protocol": proto, "probe_sec": round(time.time() - t0),
            "probe_n_distinct_seen": len(seen), "probe_pool_n": len(pool), "probe_chunk": chunk, "probe_groups": [["%s%d" % k for k in g_] for g_ in groups],
            "probe_n_distinct_note": "distinct pool items drawn over all steps (with replacement; Note 218-6); every group replays the same batch sequence"}


# ---------------------------------------------------------------- FD-MIRROR
def fd_cells():
    return [(t, k) for t in (1, 2) for k in range(1, 9 - t)] + [(3, 0), (4, 0)]


def fd_rows(its, S, cells, sp):
    """per cell -> (rows (n, D) of V_t, labels 1 = dist(R) = t + k / 0 = dist(P) = t + k, (item, slot) ids)."""
    out = {}
    for (t, k) in cells:
        X, y, ids = [], [], []
        for b, it in enumerate(its):
            d = t + k
            for j, c in enumerate(it["concepts"]):
                r_, p_ = it["dist_R"].get(c), it["dist_P"].get(c)
                if (r_ == d) != (p_ == d):
                    X.append(S[t][b, j]); y.append(1 if r_ == d else 0); ids.append((b, j))
        out[(t, k)] = (torch.stack(X) if X else None, y, ids)
    return out


def part_fd(m, TE, dev, bs, n_items=200000, pbs=64, chunk=16, items_override=None):
    t0 = time.time(); cells = fd_cells(); torch.manual_seed(22560); rng = np.random.default_rng(22560); rb = np.random.default_rng(22561)
    if items_override is None:
        excl = PS.exclusion_sigs(); s = PS.Stream(22500, excl); hold, _ = draw_items(22500, 6, 8, 2000, 22565, excl)
    else:                                                                   # G-EVMEM only: fixed (deepest) items
        hold = items_override
    dirs = plant_dirs(m, hold, [1, 2, 3, 4], dev, chunk)                   # feature standardisation per t (FD's own D 6..8 sample)
    if items_override is None:                                              # the planted code = PLANT(1)'s u / eps exactly (Audit G-L13 2b): the probe part's
        rp = np.random.default_rng(22500); sp_ = PS.Stream(22500, excl); Dq = rp.integers(3, 9, size=3000)   # pool[:2000] (after the 1,000 held-out)
        ppool = [ev(sp_.draw(int(D))) for D in Dq][1000:3000]; pdir = plant_dirs(m, ppool, list(PLANT_TS), dev, chunk)[1]
    else:
        pdir = plant_dirs(m, hold, list(PLANT_TS), dev, chunk)[1]
    seen = set()
    mk = {"lin": lambda: torch.nn.Linear(768, 1), "mlp": lambda: torch.nn.Sequential(torch.nn.Linear(768, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1))}
    nets = {(mo, c): mk[mo]().to(dev) for mo in mk for c in cells + ["plant"]}; opts = {k: torch.optim.Adam(v.parameters(), lr=1e-3) for k, v in nets.items()}
    std = lambda X, t: (X - dirs[t]["mean"]) / dirs[t]["std"]  # noqa: E731
    done = 0
    while done < n_items:
        n = min(pbs, n_items - done); done += n
        its = items_override[:n] if items_override is not None else [ev(s.draw(int(D))) for D in rng.integers(6, 9, size=n)]
        b, S = stored(m, its, 4, dev, chunk); sp = b[4]; R = fd_rows(its, S, cells, sp)
        seen.update(PS.set_sha([{"text": str(it["ids"])}]) for it in its)
        Xp = S[1][sp]; bit = torch.tensor(rb.integers(0, 2, size=Xp.shape[0]), device=dev); Xp = Xp + pdir["eps"] * (2 * bit - 1).float()[:, None] * pdir["u"][None]
        R["plant"] = (Xp, bit.tolist(), None)
        for (mo, c), net in nets.items():
            X, y, _ = R[c]
            if X is None or len(y) == 0:
                continue
            t_ = 1 if c == "plant" else c[0]; lo = net(std(X.float(), t_)).squeeze(-1); loss = F.binary_cross_entropy_with_logits(lo, torch.tensor(y, device=dev, dtype=torch.float))
            opts[(mo, c)].zero_grad(set_to_none=True); loss.backward(); opts[(mo, c)].step()
        del S
        if done % (pbs * 500) == 0:
            print("FD items", done, round(time.time() - t0), "s", flush=True)
    rt = np.random.default_rng(22562); sc = {k: {"pos": [], "neg": [], "D": [], "slots": []} for k in nets}
    Ds = [D for D in (6, 7, 8) if D in TE]; its_all = [it for D in Ds for it in TE[D]]
    with torch.no_grad():
        for ch in chunks(range(len(its_all)), bs):
            its = [its_all[j] for j in ch]; b, S = stored(m, its, 4, dev, chunk); sp = b[4]; R = fd_rows(its, S, cells, sp)
            Xp = S[1][sp]; bit = torch.tensor(rt.integers(0, 2, size=Xp.shape[0]), device=dev); Xp = Xp + pdir["eps"] * (2 * bit - 1).float()[:, None] * pdir["u"][None]
            R["plant"] = (Xp, bit.tolist(), None)
            for (mo, c), net in nets.items():
                X, y, ids = R[c]
                if X is None or len(y) == 0:
                    continue
                t_ = 1 if c == "plant" else c[0]; v = net(std(X.float(), t_)).squeeze(-1).tolist(); d_ = sc[(mo, c)]
                for q, (vv, yy) in enumerate(zip(v, y)):
                    (d_["pos"] if yy else d_["neg"]).append(round(vv, 4))
                    if ids is not None:
                        bq, jq = ids[q]; d_["slots"].append([ch[bq], jq, yy, round(vv, 4)])
                    d_["D"].append(its[ids[q][0]]["D"] if ids is not None else None)
            del S
    fd = {mo: {} for mo in mk}; valid = {}; slots = {}
    for (mo, c), d_ in sc.items():
        if c == "plant":
            continue
        a = auroc(d_["pos"], d_["neg"]); t, k = c; byD = {}
        for D in Ds:
            ys = [(row[3], row[2]) for row in d_["slots"] if its_all[row[0]]["D"] == D]
            byD[str(D)] = auroc([v for v, y in ys if y], [v for v, y in ys if not y])
        rec = {"auroc": a, "n_pos": len(d_["pos"]), "n_neg": len(d_["neg"]), "by_D": byD, "pos": d_["pos"], "neg": d_["neg"]}
        if k == 0:
            valid.setdefault(str(t), {})[mo] = rec
        else:
            fd[mo].setdefault(str(t), {})[str(k)] = rec
        slots["%s_t%d_k%d" % (mo, t, k)] = d_["slots"]
    fpl = {mo: auroc(sc[(mo, "plant")]["pos"], sc[(mo, "plant")]["neg"]) for mo in mk}
    return {"fd": fd, "fd_valid": valid, "fd_plant": fpl, "fd_slot_scores": slots, "fd_eps": {str(t): dirs[t]["eps"] for t in dirs},
            "fd_protocol": __doc__.split("  fd ")[1].split("  direct")[0].strip(), "fd_items_trained": done, "fd_n_distinct_seen": len(seen), "fd_chunk": chunk, "fd_sec": round(time.time() - t0),
            "fd_plant_dir": {"source": "PLANT(1) of the probe part (pool[:2000] after the held-out, rng 22550, t_list 0..2)", "eps": pdir["eps"], "eps_rel": pdir["eps_rel"]}}


# ---------------------------------------------------------------- G-EVMEM
def part_evmem(m, TE, dev, budget=7.0, bss=(32, 16, 8, 4), chunks_=(16, 8, 4)):
    """G-EVMEM on the eval card: each part's deepest chunk -> the largest eval_bs (probe / fd: the rollout chunk) with peak <= budget GiB.
    Shapes: acc / hist24 = the longest D 16 items at K' 24 (acc: codas at every K'); stall / vswap / cf = the longest D 8 items; causal_deep = the
    longest D 16 items; probe = one lockstep training step (pbs 64 from the longest D 8 items, every probe and plant, K 12) + its held-out read;
    fd = one streaming step (64 longest D 8 items, K 4). Each candidate runs the part's own function on a 1-chunk test set (outputs discarded)."""
    def longest(D, n):
        its = sorted(range(len(TE[D])), key=lambda j: -TE[D][j]["T"])[:n]
        return [TE[D][j] for j in its], [TE["raw"][D][j] for j in its]
    def sub(D, n):
        e, r = longest(D, n); return {D: e, "raw": {D: r}}
    def try_(fn):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
        try:
            fn(); return round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2)
        except torch.OutOfMemoryError:
            return "OOM"
    res = {}
    plan = {"acc": (16, lambda T_, b: part_acc(m, {k: v for k, v in T_.items() if k != "raw"}, dev, b, "")),
            "hist24": (16, lambda T_, b: part_hist24(m, {D: T_[D] for D in T_ if D != "raw"} | {D: T_[16] for D in range(3, 9)}, dev, b)),
            "stall": (8, lambda T_, b: stall_by(m, T_, [8], dev, b)),
            "vswap": (8, lambda T_, b: swap_arms(m, T_, [8], 22610, ("ii_prime", "ii_prime_t2", "i_prime", "ii", "person", "reset"), dev, b)),
            "causal_deep": (16, lambda T_, b: swap_arms(m, T_, [16], 22612, ("ii_prime", "i_prime"), dev, b)),
            "cf": (8, lambda T_, b: part_cf(m, T_, dev, b)),                  # fed 3 bs recipients (below) so one sweep chunk fills
            "direct": (16, lambda T_, b: part_direct(G2, {16: T_[16]}, dev, b))}       # DIRECT-SYM's plain GPT-2 at the longest D 16 items
    from boxes import s5
    G2 = s5.load_gpt2(dev); G2.eval()
    for p_ in G2.parameters():
        p_.requires_grad_(False)
    for part, (D, fn) in plan.items():
        pk = {}
        for b in bss:
            T_ = sub(D, 3 * b if part == "cf" else b); pk[str(b)] = try_(lambda: fn(T_, b))
            if pk[str(b)] != "OOM" and pk[str(b)] <= budget:
                break
        ok = [int(b) for b, v in pk.items() if v != "OOM" and v <= budget]; res[part] = {"peak_gib_by_bs": pk, "eval_bs": max(ok) if ok else None, "D": D}
        print("EVMEM", part, res[part], flush=True)
    for part in ("probe", "fd"):
        pk = {}
        for c in chunks_:
            e8, _ = longest(8, 64)
            if part == "probe":
                fn = lambda: part_probe(m, {8: e8[:16]}, dev, 16, T_SET, PLANT_TS, steps=1, pool_n=64, pbs=64, chunk=c, hold_n=16, pool_override=e8, eval_every=1)  # noqa: E731  (runs the held-out read of every probe)
            else:
                fn = lambda: part_fd(m, {8: e8[:16]}, dev, 16, n_items=64, pbs=64, chunk=c, items_override=e8)  # noqa: E731
            pk[str(c)] = try_(fn)
            if pk[str(c)] != "OOM" and pk[str(c)] <= budget:
                break
        ok = [int(c) for c, v in pk.items() if v != "OOM" and v <= budget]
        res[part] = {"peak_gib_by_chunk": pk, "chunk": max(ok) if ok else None, "eval_bs": 16 if ok else None}   # the readout bs tested (16) + the rollout chunk
        print("EVMEM", part, res[part], flush=True)
    if res["probe"]["chunk"] is None:                                       # all 17 in lockstep do not fit: the largest group of the HEAVIEST probes (top t)
        e8, _ = longest(8, 64); gp = {}                                     # that fits at chunk 16 (part_probe --probe_group; plants at t <= 2 are lighter)
        for G in (8, 6, 4, 3, 2, 1):
            gp[str(G)] = try_(lambda: part_probe(m, {8: e8[:16]}, dev, 16, T_SET[-G:], (), steps=1, pool_n=64, pbs=64, chunk=16, hold_n=16, pool_override=e8,
                                                  eval_every=1))
            if gp[str(G)] != "OOM" and gp[str(G)] <= budget:
                break
        okg = [int(g) for g, v in gp.items() if v != "OOM" and v <= budget]
        res["probe"].update({"peak_gib_by_group_chunk16": gp, "group": max(okg) if okg else None, "chunk": 16 if okg else None, "eval_bs": 16 if okg else None})
        print("EVMEM probe groups", res["probe"], flush=True)
    res["plant"] = dict(res["probe"], note="PLANT is trained and read inside the probe part (same eval_bs / chunk / group)")
    res["budget_gib"] = budget; res["pass"] = all(v.get("eval_bs", v.get("chunk")) is not None for k, v in res.items() if isinstance(v, dict))
    return {"evmem": res}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default=""); ap.add_argument("--untrained", action="store_true", help="G-L5: pretrained GPT-2, step-0 heads")
    ap.add_argument("--out_dir", default="", help="where eval_lvb.json goes (default: the run dir; required with --untrained)")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--eval_bs", default="16", help="an int, or part:bs,... (G-EVMEM)")
    ap.add_argument("--parts", default="panel,acc,stall,vswap,cf,hist24,causal_deep,probe,fd"); ap.add_argument("--per_D", type=int, default=1000, help="smokes only")
    ap.add_argument("--probe_ts", default=",".join(map(str, T_SET))); ap.add_argument("--plant_ts", default=",".join(map(str, PLANT_TS)))
    ap.add_argument("--probe_steps", type=int, default=10000, help="smokes only"); ap.add_argument("--pool_n", type=int, default=200000, help="smokes only")
    ap.add_argument("--fd_items", type=int, default=200000, help="smokes only"); ap.add_argument("--chunk", type=int, default=16); ap.add_argument("--cal_n", type=int, default=64)
    ap.add_argument("--probe_group", type=int, default=0, help="probes trained per group (G-EVMEM's value; 0 = all in lockstep)")
    a = ap.parse_args(); dev = a.device; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    m, args = untrained(dev, a.cal_n) if a.untrained else load(a.run, dev); direct = not is_lslva(m)
    bsd = {}
    for tok in a.eval_bs.split(","):
        k, _, v = tok.rpartition(":"); bsd[k or "*"] = int(v)
    bs_of = lambda p: bsd.get(p, bsd.get("*", 16))  # noqa: E731
    raw = PS.test_set(a.per_D); TE = {D: [ev(it) for it in its] for D, its in raw.items()}; TE["raw"] = raw
    TEv = {D: v for D, v in TE.items() if D != "raw"}
    full_sha = PS.set_sha([it for D in sorted(raw) for it in raw[D]])       # == scratch/gate_s218/g_l0.json test_items_sha at per_D 1000
    od = a.out_dir or a.run; os.makedirs(od, exist_ok=True); fo = os.path.join(od, "eval_direct.json" if direct else "eval_lvb.json")
    out = json.load(open(fo)) if os.path.exists(fo) else {}
    out.update({"run": a.run, "untrained": bool(a.untrained), "arch": "direct" if direct else "lslva", "dtype": "fp32", "tf32": False, "per_D": a.per_D,
                "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "test_sha": {str(D): PS.set_sha(v) for D, v in raw.items()}, "test_items_sha": full_sha,
                "c": None if direct else float(m.c_anneal), **({"untrained_cal": args} if a.untrained else {})})
    out.setdefault("eval_bs", {})
    for part in a.parts.split(","):
        if direct and part not in ("direct", "panel"):
            continue
        t0 = time.time(); bs = bs_of(part)
        if dev.startswith("cuda"):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
        if part == "panel":
            r = {"panel": json.load(open(os.path.join(ROOT, "scratch", "gate_s218", "g_l0.json")))["panel"], "panel_source": "scratch/gate_s218/g_l0.json"}
        elif part == "acc":
            r = part_acc(m, TEv, dev, bs, a.run)
        elif part == "stall":
            r = part_stall(m, TEv, dev, bs)
        elif part == "vswap":
            r = part_vswap(m, TE, dev, bs)
        elif part == "cf":
            r = part_cf(m, TE, dev, bs)
        elif part == "hist24":
            r = part_hist24(m, TEv, dev, bs)
        elif part == "causal_deep":
            r = part_causal_deep(m, TE, dev, bs, out)
        elif part == "probe":
            r = part_probe(m, TEv, dev, bs, tuple(int(v) for v in a.probe_ts.split(",") if v != ""), tuple(int(v) for v in a.plant_ts.split(",") if v != ""),
                           a.probe_steps, a.pool_n, chunk=a.chunk, group=a.probe_group)
            for k_ in ("probe", "plant"):                                   # parts run in pieces merge per t
                if k_ in out:
                    out[k_].update(r.pop(k_))
            if "probe_valid" in out:
                out["probe_valid"].update(r.pop("probe_valid"))
        elif part == "fd":
            r = part_fd(m, TEv, dev, bs, a.fd_items, chunk=a.chunk)
        elif part == "direct":
            r = part_direct(m, TEv, dev, bs)
        elif part == "evmem":
            r = part_evmem(m, TE, dev)
        elif part == "contamination":
            r = part_contamination(a.run)
        elif part == "stall_selfcheck":                                     # Audit G-L13: a never-firing stall must equal the intact K' = D + 1 run
            st_ = stall_by(m, TEv, [4, 5], dev, bs, ts_override=99); ok = []
            for D in (4, 5):
                its = TEv[D]; ref = []
                for ch in chunks(range(len(its)), bs):
                    sc, _ = score_b(m, LV.lvb_batch([its[j] for j in ch], dev), D + 1); ref += (sc[D + 1][0] > sc[D + 1][1]).long().tolist()
                ok.append(ref == st_["items"][str(D)]["stalled"])
            r = {"stall_selfcheck": {"equal": ok, "pass": all(ok)}}
        else:
            raise SystemExit("unknown part " + part)
        out.update(r); out.setdefault("part_sec", {})[part] = round(time.time() - t0)
        if part not in ("evmem", "panel", "contamination", "stall_selfcheck"):
            out["eval_bs"][part] = bs
            if part == "probe":
                out["eval_bs"]["plant"] = bs                                # PLANT is read by the probe part
        if part in PART_DS:                                                 # §4.3 items_sha per part: the loaded SYM test set (D 3..16, test order; == G-L0's
            out.setdefault("items_sha", {})[part] = full_sha                # test_items_sha) + the sha of the D range the part reads
            out.setdefault("items_sha_range", {})[part] = PS.set_sha([it for D in PART_DS[part] if D in raw for it in raw[D]])
        out.setdefault("card_by_part", {})[part] = out["card"]
        if part == "evmem":                                                 # score_s218 PATHS evmem_A
            gd = os.path.join(ROOT, "boxes", "runs_fsq", "lvb_gates"); os.makedirs(gd, exist_ok=True)
            json.dump({"card": out["card"], "eval_bs": {k: v.get("eval_bs", v.get("chunk")) for k, v in r["evmem"].items() if isinstance(v, dict)},
                       **r["evmem"]}, open(os.path.join(gd, "gate_evmem.json"), "w", newline="\n"), indent=1)
        if dev.startswith("cuda"):
            out.setdefault("part_peak_gib", {})[part] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2)
        json.dump(out, open(fo + ".tmp", "w", newline="\n")); os.replace(fo + ".tmp", fo); print("PART", part, "done", round(time.time() - t0), "s", flush=True)
    print("DONE", fo, flush=True)


if __name__ == "__main__":
    main()
