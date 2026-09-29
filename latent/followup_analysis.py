"""Summaries for latent/followup.py records. All uncertainty: percentile bootstrap over graphs."""
from __future__ import annotations

import numpy as np

sig = lambda x: 1.0 / (1.0 + np.exp(-x))


def boot(x, n=2000, seed=0):
    x = np.asarray(x, np.float64)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def _gmean(vals, mask):
    """Per-graph mean of vals over mask (…, N, Q) -> (N,) averaging all leading edit axes."""
    v = np.where(mask, vals, np.nan)
    lead = tuple(range(v.ndim - 2))
    with np.errstate(all="ignore"):
        per = np.nanmean(v, axis=-1)                     # (..., N)
        return np.nanmean(per, axis=lead) if lead else per


def branch_masks(qb, att_idx, attached):
    """edited (N, W, Q): query belongs to edited slot w; other attached; decoy (non-hub queries)."""
    N, W = att_idx.shape
    edited = np.stack([qb == att_idx[:, [w]] for w in range(W)], 1)
    att_q = np.take_along_axis(attached, np.clip(qb, 0, None), 1) & (qb >= 0)
    other = att_q[:, None, :] & ~edited
    decoy = (~att_q)[:, None, :].repeat(W, 1)
    return edited, other, decoy


def collateral_stats(delta, qb, include):
    """delta (N, Q) ; include (N, NB) bool branches counted. Returns per-graph mean/sum/worst/signed."""
    N = delta.shape[0]
    res = np.full((4, N), np.nan)
    for n in range(N):
        vals, sv = [], []
        for b in np.nonzero(include[n])[0]:
            m = qb[n] == b
            vals.append(np.abs(delta[n, m]).mean()); sv.append(delta[n, m].mean())
        if vals:
            res[:, n] = [np.mean(vals), np.sum(vals), np.max(vals), np.mean(sv)]
    return res


# ------------------------------------------------------------------ dose
def summarize_dose(r, depth_min_targeted=None):
    N, W = r["att_idx"].shape
    qb, qd = r["q_branch"], r["q_dist"]
    sign = 1.0 if r["direction"] == "remove" else -1.0
    attached = np.zeros((N, 10), bool)
    for w in range(W):
        attached[np.arange(N), r["att_idx"][:, w]] = True
    edited, other, decoy = branch_masks(qb, r["att_idx"], attached)
    pb, pd = sig(r["base"]), sig(r["don"])
    clean = sign * (pb - pd)                                        # (N, W, Q)
    out = {"W": W, "k": int(r["k"]), "direction": r["direction"], "kinds": {}}
    competent = ((pb > 0.5) == (sign > 0)) & ((pd < 0.5) == (sign > 0))   # both runs correct on edited nodes
    out["clean_gap"] = boot(_gmean(clean, edited))
    out["clean_gap_logit"] = boot(_gmean(sign * (r["base"] - r["don"]), edited))
    out["competent_frac"] = boot(_gmean(competent.astype(float), edited))
    out["diff_norm"] = float(r["diff_norm"].mean())
    out["clean_by_depth"] = {int(d): boot(_gmean(clean, edited & (qd[:, None] == d)))[0] for d in range(1, 7)}
    for ki, kind in enumerate(r["kinds"]):
        kd = {"proj_norm": float(r["proj_norm"][ki].mean()), "proj_frac_norm": float((r["proj_norm"][ki] / np.maximum(r["diff_norm"], 1e-9)).mean()), "doses": {}}
        for di, lam in enumerate(r["doses"]):
            pp = sig(r["pat"][ki, di])
            tgt = sign * (pb - pp)
            tgt_l = sign * (r["base"] - r["pat"][ki, di])
            dl = pb - pp
            # collateral over unchanged reachable branches, per edit, then per graph
            col = np.stack([collateral_stats(dl[:, w], qb, attached & (np.arange(10)[None] != r["att_idx"][:, [w]]))
                            for w in range(W)], 1)                                  # (4, W, N)
            dec = np.stack([collateral_stats(dl[:, w], qb, ~attached) for w in range(W)], 1)
            with np.errstate(all="ignore"):
                colg = np.nanmean(col, 1)                                           # (4, N)
                decg = np.nanmean(dec, 1)
            tg = _gmean(tgt, edited)
            cg = _gmean(clean, edited)
            ok = cg >= 0.1
            e = {"targeted": boot(tg), "targeted_logit": boot(_gmean(tgt_l, edited)),
                 "collateral_mean": boot(colg[0]), "collateral_sum": boot(colg[1]), "collateral_worst": boot(colg[2]),
                 "collateral_signed": boot(colg[3]), "decoy_mean": boot(decg[0]),
                 "recovery_ratio_of_means": float(np.nanmean(tg[ok]) / np.nanmean(cg[ok])) if ok.any() else float("nan"),
                 "recovery_n_graphs": int(ok.sum()),
                 "targeted_competent": boot(_gmean(tgt, edited & competent)),
                 "targeted_not_competent": boot(_gmean(tgt, edited & ~competent)),
                 "by_depth": {int(d): boot(_gmean(tgt, edited & (qd[:, None] == d)))[0] for d in range(1, 7)},
                 "perturb_norm": float(lam * r["proj_norm"][ki].mean())}
            kd["doses"][float(lam)] = e
        out["kinds"][str(kind)] = kd
    return out


# ------------------------------------------------------------------ swaps
def summarize_swap(r):
    qb = r["q_branch"]
    N = qb.shape[0]
    attached = r["attached"]
    pb = sig(r["base"])
    out = {"W": int(r["W"]), "k": int(r["k"]), "conds": {}}
    for ci, cnd in enumerate(r["conds"]):
        cd = {"donor_width": {"swap": int(r["W"]), "remove_A": int(r["W"]) - 1, "insert_B": int(r["W"]) + 1,
                              "same_status": int(r["W"])}[str(cnd)], "kinds": {}}
        # clean donor effects
        P = r["A"].shape[0]
        def stats(pp):
            tA, tB, cm, cs, cw, csg, dm = [], [], [], [], [], [], []
            for pi in range(P):
                A, B = r["A"][pi], r["B"][pi]
                d = pb - pp[pi]
                mA = qb == A[:, None]; mB = qb == B[:, None]
                tA.append(np.nanmean(np.where(mA, d, np.nan), 1))
                tB.append(np.nanmean(np.where(mB, -d, np.nan), 1))
                inc = attached.copy(); inc[np.arange(N), A] = False
                c = collateral_stats(d, qb, inc)
                incd = ~attached; incd = incd.copy(); incd[np.arange(N), B] = False
                dd = collateral_stats(d, qb, incd)
                cm.append(c[0]); cs.append(c[1]); cw.append(c[2]); csg.append(c[3]); dm.append(dd[0])
            f = lambda L: boot(np.nanmean(np.stack(L), 0))
            return {"A_drop": f(tA), "B_rise": f(tB), "collateral_mean": f(cm), "collateral_sum": f(cs),
                    "collateral_worst": f(cw), "collateral_signed": f(csg), "decoy_mean": f(dm)}
        cd["clean"] = stats(sig(r["don"][ci]))
        for ki, kind in enumerate(r["kinds"]):
            cd["kinds"][str(kind)] = {float(lam): stats(sig(r["pat"][ci, ki, di])) for di, lam in enumerate(r["doses"])}
        cd["diff_norm"] = float(r["diff_norm"][ci].mean())
        out["conds"][str(cnd)] = cd
    return out


# ------------------------------------------------------------------ continuation
def summarize_continuation(r):
    R = sig(r["R"])
    out = {"W": int(r["W"]), "k": int(r["k"]), "j": int(r["j"]), "n": int(r["n"]), "sets": {}}
    sets = ["C_rec", "C_don", "B_shared", "other_attached"]
    for s in sets:
        m = r[f"mask_{s}"]
        row = {"natural_Nt": boot(_gmean(sig(r["Nt"]) - R, m)), "donor_Dn": boot(_gmean(sig(r["Dn"]) - R, m))}
        for key in r:
            if key.startswith("edit_"):
                row[key[5:]] = boot(_gmean(sig(r[key]) - R, m))
        out["sets"][s] = row
    # clean competence on the diagnostic sets
    out["competence"] = {"Nt_C_rec_reachable": boot(_gmean((sig(r["Nt"]) > 0.5).astype(float), r["mask_C_rec"])),
                         "Nt_C_don_unreachable": boot(_gmean((sig(r["Nt"]) < 0.5).astype(float), r["mask_C_don"])),
                         "Dn_C_don_reachable": boot(_gmean((sig(r["Dn"]) > 0.5).astype(float), r["mask_C_don"])),
                         "R_C_rec_unreachable": boot(_gmean((R < 0.5).astype(float), r["mask_C_rec"]))}
    # recipient-following index per graph: ΔC_rec − ΔC_don, for each edit
    for key in r:
        if key.startswith("edit_"):
            d = sig(r[key]) - R
            out[f"follow_index_{key[5:]}"] = boot(_gmean(d, r["mask_C_rec"]) - _gmean(d, r["mask_C_don"]))
    return out


def summarize_amplify(r):
    qb, qd, attached = r["q_branch"], r["q_dist"], r["attached"]
    pb = sig(r["base"])
    att_q = np.take_along_axis(attached, np.clip(qb, 0, None), 1) & (qb >= 0)
    out = {"W": int(r["W"]), "k": int(r["k"]), "lams": {}}
    for li, lam in enumerate(r["lams"]):
        e = {}
        own_deep, own_all, others = [], [], []
        for si in range(r["att_idx"].shape[1]):
            d = sig(r["att"][li, si]) - pb
            m = qb == r["att_idx"][:, [si]]
            own_deep.append(_gmean(d, m & (qd >= 3))); own_all.append(_gmean(d, m)); others.append(_gmean(d, att_q & ~m))
        e["own_deep"] = boot(np.nanmean(np.stack(own_deep), 0))
        e["own_all"] = boot(np.nanmean(np.stack(own_all), 0))
        e["other_attached"] = boot(np.nanmean(np.stack(others), 0))
        d = sig(r["dec"][li]) - pb
        e["decoy_self"] = boot(_gmean(d, qb == r["dec_idx"][:, None]))
        e["decoy_on_attached"] = boot(_gmean(d, att_q))
        d = sig(r["rnd"][li]) - pb
        e["random_on_attached"] = boot(_gmean(d, att_q))
        e["random_abs_attached"] = boot(_gmean(np.abs(d), att_q))
        out["lams"][float(lam)] = e
    return out


def summarize_upfront_cache(r):
    pb = sig(r["base"])
    clean = pb - sig(r["don"])
    out = {"W": int(r["W"]), "k": int(r["k"]), "clean_gap": boot(clean), "shares": {}, "by_dist": {}}
    for c in ("T", "L", "TL", "Gq", "TLGq", "clamp"):
        eff = pb - sig(r[c])
        out["shares"][c] = float(eff.mean() / clean.mean())
    for d in range(1, 7):
        m = r["target_dist"] == d
        if m.sum() >= 10 and clean[m].mean() > 0.05:
            out["by_dist"][d] = {"clean_gap": float(clean[m].mean()),
                                 **{c: float((pb[m] - sig(r[c][m])).mean() / clean[m].mean()) for c in ("T", "TL", "Gq")}}
    return out


def summarize_pair(r):
    R = sig(r["R"])
    out = {"W": int(r["W"]), "k": int(r["k"]), "j": int(r["j"]), "n": int(r["n"])}
    for e in ("both", "only1", "only2"):
        d = sig(r[f"edit_{e}"]) - R
        row = {s: boot(_gmean(d, r[f"mask_{s}"])) for s in ("C_rec1", "C_don1", "C_rec2", "C_don2")}
        row["follow1"] = boot(_gmean(d, r["mask_C_rec1"]) - _gmean(d, r["mask_C_don1"]))
        row["follow2"] = boot(_gmean(d, r["mask_C_rec2"]) - _gmean(d, r["mask_C_don2"]))
        out[e] = row
    nat = sig(r["Nt"]) - R
    out["natural"] = {s: boot(_gmean(nat, r[f"mask_{s}"])) for s in ("C_rec1", "C_rec2")}
    # additivity on each branch's recipient continuation: both vs single edit of that branch
    d_b, d1, d2 = (sig(r[f"edit_{e}"]) - R for e in ("both", "only1", "only2"))
    out["additivity_branch1"] = boot(_gmean(d_b, r["mask_C_rec1"]) - _gmean(d1, r["mask_C_rec1"]))
    out["additivity_branch2"] = boot(_gmean(d_b, r["mask_C_rec2"]) - _gmean(d2, r["mask_C_rec2"]))
    return out


# ------------------------------------------------------------------ cache factorial
def summarize_cache(r):
    qb = r["q_branch"]
    attached = r["attached"]
    S, N = r["A"].shape
    pb = sig(r["base"])
    out = {"W": int(r["W"]), "k": int(r["k"]), "conds": {}}
    clean = []
    for si in range(S):
        m = qb == r["A"][si][:, None]
        clean.append(np.nanmean(np.where(m, pb[si] - sig(r["don"][si]), np.nan), 1))
    cleang = np.nanmean(np.stack(clean), 0)
    out["clean_gap"] = boot(cleang)
    for cnd in list(r["conds"]) + ["don"]:
        cnd = str(cnd)
        tg, tl, co, cs = [], [], [], []
        for si in range(S):
            A = r["A"][si]
            m = qb == A[:, None]
            d = pb[si] - sig(r[cnd][si])
            tg.append(np.nanmean(np.where(m, d, np.nan), 1))
            tl.append(np.nanmean(np.where(m, r["base"][si] - r[cnd][si], np.nan), 1))
            inc = attached.copy(); inc[np.arange(N), A] = False
            c = collateral_stats(d, qb, inc)
            co.append(c[0]); cs.append(c[3])
        tgm = np.nanmean(np.stack(tg), 0)
        out["conds"][cnd] = {"targeted": boot(tgm), "targeted_logit": boot(np.nanmean(np.stack(tl), 0)),
                             "collateral_mean": boot(np.nanmean(np.stack(co), 0)),
                             "collateral_signed": boot(np.nanmean(np.stack(cs), 0)),
                             "fraction_of_clean": float(np.nanmean(tgm) / np.nanmean(cleang)),
                             "margin_by_step": np.nanmean(r[f"margin_{cnd}"], (0, 1)).tolist()}
    # depth-resolved route shares (post-hoc): fraction of the clean effect per node distance
    qd = r["q_dist"]
    out["by_depth"] = {}
    for d in range(1, 7):
        cl, per = [], {}
        for si in range(S):
            m = (qb == r["A"][si][:, None]) & (qd == d)
            cl.append(np.nanmean(np.where(m, pb[si] - sig(r["don"][si]), np.nan), 1))
        with np.errstate(all="ignore"):
            clm = np.nanmean(np.nanmean(np.stack(cl), 0))
        for cnd in [str(c) for c in r["conds"]]:
            vals = []
            for si in range(S):
                m = (qb == r["A"][si][:, None]) & (qd == d)
                vals.append(np.nanmean(np.where(m, pb[si] - sig(r[cnd][si]), np.nan), 1))
            with np.errstate(all="ignore"):
                per[cnd] = float(np.nanmean(np.nanmean(np.stack(vals), 0)) / clm) if clm > 0.05 else float("nan")
        out["by_depth"][d] = {"clean_gap": float(clm), "shares": per}
    # 2x2 interactions (T x L) on probability and logit scales
    C = out["conds"]
    if "TL" in C:
        for scale in ("targeted", "targeted_logit"):
            out[f"interaction_TxL_{scale}"] = C["TL"][scale][0] - C["T"][scale][0] - C["L"][scale][0]
            out[f"interaction_TxG_{scale}"] = C["TG"][scale][0] - C["T"][scale][0] - C["G"][scale][0]
    return out


# ------------------------------------------------------------------ attention budget
def summarize_attention(r):
    qb = r["q_branch"]
    attached = r["attached"]
    N = qb.shape[0]
    pb = sig(r["base"])
    att_q = np.take_along_axis(attached, np.clip(qb, 0, None), 1) & (qb >= 0)
    out = {"W": int(r["W"]), "doses": {}}
    for di, b in enumerate(r["doses"]):
        e = {}
        rows_self, rows_other = [], []
        for si in range(r["att_idx"].shape[1]):
            d = sig(r["att_boost"][di, si]) - pb
            self_m = qb == r["att_idx"][:, [si]]
            rows_self.append(_gmean(d, self_m)); rows_other.append(_gmean(d, att_q & ~self_m))
        e["att_boost_self"] = boot(np.nanmean(np.stack(rows_self), 0))
        e["att_boost_other_attached"] = boot(np.nanmean(np.stack(rows_other), 0))
        d = sig(r["dec_boost"][di]) - pb
        dm = qb == r["dec"][:, None]
        e["dec_boost_attached"] = boot(_gmean(d, att_q))
        e["dec_boost_self_decoy"] = boot(_gmean(d, dm))
        d = sig(r["all_boost"][di]) - pb
        e["all_boost_attached"] = boot(_gmean(d, att_q))
        e["all_boost_decoy"] = boot(_gmean(d, ~att_q & (qb >= 0)))
        out["doses"][float(b)] = e
    return out


def summarize_gain(r):
    qb = r["q_branch"]
    attached = r["attached"]
    N = qb.shape[0]
    att_q = np.take_along_axis(attached, np.clip(qb, 0, None), 1) & (qb >= 0)
    out = {"W": int(r["W"]), "k": int(r["k"])}
    for tag in ("plain", "restored"):
        pb = sig(r[f"{tag}_base"])
        tg, cg, co, cs = [], [], [], []
        for si in range(r["att_idx"].shape[1]):
            m = qb == r["att_idx"][:, [si]]
            d = pb - sig(r[f"{tag}_pat"][si])
            tg.append(_gmean(d, m)); cg.append(_gmean(pb - sig(r[f"{tag}_don"][si]), m))
            inc = attached.copy(); inc[np.arange(N), r["att_idx"][:, si]] = False
            c = collateral_stats(d, qb, inc); co.append(c[0]); cs.append(c[3])
        acc = np.where(att_q, pb > 0.5, pb < 0.5)
        out[tag] = {"targeted": boot(np.nanmean(np.stack(tg), 0)), "clean_gap": boot(np.nanmean(np.stack(cg), 0)),
                    "collateral_mean": boot(np.nanmean(np.stack(co), 0)),
                    "collateral_signed": boot(np.nanmean(np.stack(cs), 0)),
                    "accuracy": boot(np.nanmean(np.where(qb >= 0, acc, np.nan), 1))}
    return out


# ------------------------------------------------------------------ controllability
def jacobian_features(r):
    """Per (slot, graph): gain |a_i|, residual ratio |p|/|a_i|, linear-predicted and observed effects."""
    A = r["A"]                                            # (N, Q, rank)
    qb, attached = r["q_branch"], r["attached"]
    S, N = r["att_idx"].shape[1], A.shape[0]
    rows = []
    for si in range(S):
        for n in range(N):
            i = r["att_idx"][n, si]
            means = np.stack([A[n, qb[n] == b].mean(0) for b in range(10)])        # (NB, rank)
            a = means[i]
            O = np.delete(means, i, 0)
            P_ = O.T @ np.linalg.pinv(O @ O.T) @ O
            p = a - P_ @ a
            u = r["u"][si, n]
            pred = means @ u                                                      # linear logit change per branch
            obs = np.array([(r["obs_pca"][si, n] - r["base"][n])[qb[n] == b].mean() for b in range(10)])
            obs_full = np.array([(sig(r["obs_full"][si, n]) - sig(r["base"][n]))[qb[n] == b].mean() for b in range(10)])
            others_att = [b for b in range(10) if attached[n, b] and b != i]
            base_conf = sig(r["base"][n])[qb[n] == i].mean()
            rows.append({"W": int(r["W"]), "gain": float(np.linalg.norm(a)), "resid_ratio": float(np.linalg.norm(p) / (np.linalg.norm(a) + 1e-9)),
                         "diff_norm": float(r["diff_norm"][si, n]), "base_conf": float(base_conf),
                         "pred_target_logit": float(pred[i]), "obs_target_logit": float(obs[i]),
                         "pred_col_logit": float(np.abs(pred[others_att]).mean()) if others_att else float("nan"),
                         "obs_col_logit": float(np.abs(obs[others_att]).mean()) if others_att else float("nan"),
                         "obs_target_prob": float(-obs_full[i]),
                         "obs_col_prob": float(np.abs(obs_full[others_att]).mean()) if others_att else float("nan")})
    return rows
