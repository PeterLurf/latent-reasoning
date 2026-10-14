"""EMP gates (Theory seed-stability subagent, 2026-10-13): delta-gate statistics, open-loop delta contraction per pass, channel ablation by pass,
relative state increments, on one final model (D:\\Latent\\boxes\\runs_fsq\\<run>\\model.pt). READ-ONLY on the repo; CPU fp32; outputs to emp/.
Usage: python emp_gates.py <run> [n_per_n] [K] [--quick]"""
import os
import sys
import json
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
except Exception as e_:  # noqa: BLE001
    print("psutil nice failed", e_)
sys.path.insert(0, r"D:\Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(3)
torch.set_grad_enabled(False)
import boxes.deltamem as DM  # noqa: E402
import boxes.nested170 as N  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
RUN = sys.argv[1]
NPER = int(sys.argv[2]) if len(sys.argv) > 2 else 32
KMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 12
QUICK = "--quick" in sys.argv
VALIDATE = "--validate" in sys.argv
NEXT = int(sys.argv[4]) if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else 24   # per-n count of the extra n = 10 / 12 items
RD = os.path.join(r"D:\Latent\boxes\runs_fsq", RUN)
DEV = "cpu"
DH = DM.DH

# ---------------------------------------------------------------- recorder: wrap write_seq (the original computes the state; we recompute k, v, al, be)
REC = {"on": False, "data": None}
_orig_write_seq = DM.write_seq


def write_seq_rec(mem, h6, hw, S, mask=None):
    Snew = _orig_write_seq(mem, h6, hw, S, mask)
    if REC["on"]:
        H_ADD, H_DEL = mem.h_add, mem.h_del
        B, T, D = h6.shape; a6 = mem.ln(h6.float()); aw = mem.ln_w(hw.float()); gm = mem.gamma
        u = (1 - gm) * a6 + gm * aw; _, k, v = mem.qkv(u).split(D, -1); k, v = (t_.view(B, T, H_ADD + H_DEL, DH) for t_ in (k, v))
        g_ = mem.gate(u); al = torch.sigmoid(g_[..., :H_DEL]); be = 2 * torch.sigmoid(g_[..., H_DEL:])
        REC["data"] = {"al": al, "be": be, "kd": F.normalize(k[:, :, H_ADD:], dim=-1), "vd": v[:, :, H_ADD:], "mask": mask,
                       "be_logit": g_[..., H_DEL:], "al_logit": g_[..., :H_DEL]}
    return Snew


DM.write_seq = write_seq_rec


def load_model():
    a = json.load(open(os.path.join(RD, "args.json")))
    DM.FAST_ADD = bool(a.get("fast_add")); DM.CHUNK_DELTA = 0
    g = N.make_trunk({}, DEV); ML, HA, HD = N.mem_geometry(a)
    m = N.InjectN(g, ML, a.get("mem_write") or "top", HA, HD, a.get("add_norm") or None).to(DEV)
    sd = torch.load(os.path.join(RD, "model.pt"), map_location=DEV, weights_only=True)
    res = m.load_state_dict(sd, strict=True); del sd
    m.nopass = bool(a.get("aux_nopass")); m.eval()
    return m, a, str(res)


def make_items():
    rng = np.random.default_rng(19613); out = []
    for n in range(4, 9):
        for _ in range(NPER):
            r, t = N.generate(rng, n); s = N.render(r); out.append({"n": n, "syms": s, "t": t, "root": r})
    return out


def q(x, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    x = np.asarray(x, dtype=np.float64)
    return {"mean": float(x.mean()), **{"q%02d" % int(round(100 * p)): float(np.quantile(x, p)) for p in qs}}


def gate_stats(d):
    """(a) beta / alpha over real tokens, pooled and per head."""
    msk = d["mask"].bool(); be = d["be"][msk].double().numpy(); al = d["al"][msk].double().numpy()   # (Nreal, H)
    out = {"beta": q(be.ravel()), "alpha": q(al.ravel()),
           "beta_frac_lt0.2": float((be < 0.2).mean()), "beta_frac_0.8_1.2": float(((be >= 0.8) & (be <= 1.2)).mean()), "beta_frac_gt1.8": float((be > 1.8).mean()),
           "beta_frac_gt1.0": float((be > 1.0).mean()), "alpha_frac_lt0.5": float((al < 0.5).mean()), "alpha_frac_gt0.99": float((al > 0.99).mean()),
           "per_head": []}
    for h in range(be.shape[1]):
        out["per_head"].append({"beta_mean": float(be[:, h].mean()), "beta_q05": float(np.quantile(be[:, h], 0.05)), "beta_q50": float(np.quantile(be[:, h], 0.5)),
                                "beta_q95": float(np.quantile(be[:, h], 0.95)), "beta_frac_lt0.2": float((be[:, h] < 0.2).mean()),
                                "beta_frac_0.8_1.2": float(((be[:, h] >= 0.8) & (be[:, h] <= 1.2)).mean()), "beta_frac_gt1.8": float((be[:, h] > 1.8).mean()),
                                "alpha_mean": float(al[:, h].mean()), "alpha_q05": float(np.quantile(al[:, h], 0.05)), "alpha_q50": float(np.quantile(al[:, h], 0.5))})
    return out


def contraction(d, Sd_prev, Sd_new):
    """(b) P = prod_t al_t (I - be_t k_t k_t^T) over real tokens (right-multiplied, token order), float64, per item and head."""
    al, be, kd, vd, msk = (d["al"].double(), d["be"].double(), d["kd"].double(), d["vd"].double(), d["mask"].bool())
    B, T, H, _ = kd.shape; I = torch.eye(DH, dtype=torch.float64)
    P = I.expand(B, H, DH, DH).clone(); Sr = Sd_prev.double().clone()     # Sr: my own replay of the delta recurrence (sanity: must equal Sd_new)
    for t in range(T):
        k_ = kd[:, t]; a_ = al[:, t][..., None, None]; b_ = be[:, t][..., None, None]; m_ = msk[:, t][:, None, None, None]
        Pk = torch.einsum("bhij,bhj->bhi", P, k_); Pn = a_ * (P - b_ * Pk[..., :, None] * k_[..., None, :]); P = torch.where(m_, Pn, P)
        Sk = torch.einsum("bhvk,bhk->bhv", Sr, k_); Sn = a_ * (Sr - b_ * Sk[..., :, None] * k_[..., None, :]) + b_ * vd[:, t][..., :, None] * k_[..., None, :]
        Sr = torch.where(m_, Sn, Sr)
    replay_err = float(((Sr - Sd_new.double()).flatten(1).norm(dim=1) / Sd_new.double().flatten(1).norm(dim=1).clamp_min(1e-30)).max())
    sv = torch.linalg.svdvals(P)                                            # (B, H, 64) descending
    mfl = msk.double()                                                      # analytic: log|det P| / 64 = sum log al + (1/64) sum log|1 - be|
    mlsv_det = ((torch.log(al) + torch.log(torch.abs(1 - be).clamp_min(1e-300)) / DH) * mfl[..., None]).sum(1)   # (B, H)
    sum_log_al = (torch.log(al) * mfl[..., None]).sum(1)
    carried = torch.einsum("bhvk,bhkj->bhvj", Sd_prev.double(), P)
    pn = Sd_prev.double().flatten(2).norm(dim=2); cn = carried.flatten(2).norm(dim=2); nn_ = Sd_new.double().flatten(2).norm(dim=2)
    ok = pn > 0
    out = {"replay_rel_err_max": replay_err,
           "mean_log_sv_det": q(mlsv_det.numpy().ravel()), "sum_log_alpha": q(sum_log_al.numpy().ravel()),
           "mean_log_sv_svd": q(torch.log(sv.clamp_min(1e-300)).mean(-1).numpy().ravel()),
           "median_log_sv": q(torch.log(sv.clamp_min(1e-300)).median(-1).values.numpy().ravel()),
           "top_sv": q(sv[..., 0].numpy().ravel()), "n_sv_lt0.5": q((sv < 0.5).sum(-1).double().numpy().ravel()),
           "n_sv_gt0.9": q((sv > 0.9).sum(-1).double().numpy().ravel()), "frob2_over_64": q((sv ** 2).mean(-1).numpy().ravel()),
           "per_head_mean_log_sv_det": [float(x) for x in mlsv_det.mean(0)], "per_head_n_sv_lt0.5": [float(x) for x in (sv < 0.5).sum(-1).double().mean(0)],
           "per_head_top_sv": [float(x) for x in sv[..., 0].mean(0)]}
    if bool(ok.any()):
        out["state_retention"] = q((cn[ok] / pn[ok]).numpy())             # ||S_prev P|| / ||S_prev||: open-loop survival of the previous delta state
        out["carried_share"] = q((cn[ok] / nn_[ok].clamp_min(1e-30)).numpy())   # ||S_prev P|| / ||S_new||
        cosv = F.cosine_similarity(carried.flatten(2), Sd_new.double().flatten(2), dim=2)
        out["cos_carried_new"] = q(cosv[ok].numpy())
        out["per_head_state_retention"] = [float(x) for x in torch.where(ok, cn / pn.clamp_min(1e-30), torch.nan).nanmean(0)]
    # per-item arrays (B, H) for the by-n summaries (Audit request: alpha, alpha_eff^L, P spectrum by n)
    L = mfl.sum(1).numpy()                                                  # real tokens per item
    bem = (be * mfl[..., None]); alm = (al * mfl[..., None])
    pi = {"L": L, "alpha_mean": (alm.sum(1) / mfl.sum(1)[:, None]).numpy(), "sum_log_alpha": sum_log_al.numpy(), "mlsv_det": mlsv_det.numpy(),
          "top_sv": sv[..., 0].numpy(), "n_sv_lt05": (sv < 0.5).sum(-1).double().numpy(), "n_sv_gt09": (sv > 0.9).sum(-1).double().numpy(),
          "beta_mean": (bem.sum(1) / mfl.sum(1)[:, None]).numpy(),
          "cnt_be_lt02": ((be < 0.2).double() * mfl[..., None]).sum(1).numpy(), "cnt_be_08_12": (((be >= 0.8) & (be <= 1.2)).double() * mfl[..., None]).sum(1).numpy(),
          "cnt_be_gt18": ((be > 1.8).double() * mfl[..., None]).sum(1).numpy(),
          "retention": torch.where(ok, cn / pn.clamp_min(1e-30), torch.nan).numpy()}
    return out, pi


def by_n(pi, ns):
    """per-n summaries of the per-item arrays: alpha mean per head (token-weighted), alpha_eff^L = exp(mean over items of sum_t log alpha_t) per head
    and averaged over heads (exp of the head-mean log), P spectrum stats, beta fractions (token-weighted), state retention."""
    out = {}
    for n in sorted(set(int(v) for v in ns)):
        s = ns == n; Lw = pi["L"][s][:, None]; ntok = float(pi["L"][s].sum())
        am_h = (pi["alpha_mean"][s] * Lw).sum(0) / Lw.sum(); sla = pi["sum_log_alpha"][s]
        r_ = pi["retention"][s]
        out[str(n)] = {"n_items": int(s.sum()), "L_mean": float(pi["L"][s].mean()),
                       "alpha_mean_per_head": [float(v) for v in am_h], "alpha_mean": float(am_h.mean()),
                       "alpha_effL_per_head": [float(v) for v in np.exp(sla.mean(0))], "alpha_effL": float(np.exp(sla.mean())),
                       "alpha_effL_min_head": float(np.exp(sla.mean(0)).min()),
                       "mean_log_sv_det": float(pi["mlsv_det"][s].mean()), "mean_log_sv_det_per_head": [float(v) for v in pi["mlsv_det"][s].mean(0)],
                       "top_sv": float(pi["top_sv"][s].mean()), "top_sv_per_head": [float(v) for v in pi["top_sv"][s].mean(0)],
                       "n_sv_lt05": float(pi["n_sv_lt05"][s].mean()), "n_sv_lt05_per_head": [float(v) for v in pi["n_sv_lt05"][s].mean(0)],
                       "n_sv_gt09": float(pi["n_sv_gt09"][s].mean()),
                       "beta_mean": float(((pi["beta_mean"][s] * Lw).sum(0) / Lw.sum()).mean()),
                       "beta_frac_lt0.2": float(pi["cnt_be_lt02"][s].sum() / (ntok * pi["cnt_be_lt02"].shape[1])),
                       "beta_frac_0.8_1.2": float(pi["cnt_be_08_12"][s].sum() / (ntok * pi["cnt_be_lt02"].shape[1])),
                       "beta_frac_gt1.8": float(pi["cnt_be_gt18"][s].sum() / (ntok * pi["cnt_be_lt02"].shape[1])),
                       "state_retention": (float(np.nanmean(r_)) if np.isfinite(r_).any() else None)}
    return out


def aux_exact(m, S, items, t, abl, bs=32, roles=False):
    """(c) greedy exact match of e_{t+1} + EOT at pass t (items with n >= t), via the exactly equivalent teacher-forced criterion:
    every target position's argmax (full vocabulary) equals the target. mem.ablate = abl during the aux read only."""
    idx = [j for j, it in enumerate(items) if len(N.aux_of(it)) >= t]; ok, hd, cp = [], [], []
    m.mem.ablate = abl
    try:
        for i in range(0, len(idx), bs):
            jj = idx[i:i + bs]; sub = [items[j] for j in jj]
            lg, y = m.aux_logits(S[jj], sub, t, DEV); pr = lg.argmax(-1); good = (pr == y) | (y == -100)
            ok.extend(bool(v) for v in good.all(1)); g_ = good.numpy()
            if roles:
                for r_, it in enumerate(sub):
                    sy, hi = N.aux_of(it)[t - 1]; hd.append(bool(g_[r_, hi])); cp.extend(bool(g_[r_, k_]) for k_ in range(len(sy)) if k_ != hi)
    finally:
        m.mem.ablate = None
    return idx, ok, (hd, cp)


def answers(m, S, items, abl, bs=80):
    m.mem.ablate = abl; out = []
    try:
        for i in range(0, len(items), bs):
            ch = items[i:i + bs]; lg = m.dec_logits(S[i:i + bs], ch, DEV); out.append(lg[:, N.NUM_IDS].argmax(-1).numpy())
    finally:
        m.mem.ablate = None
    return np.concatenate(out)


def main():
    t0 = time.time(); m, a, lres = load_model(); items = make_items()
    if QUICK:
        items = items[::NPER][:5]
    B = len(items); x, msk = m.think_inputs(items, DEV); ns = np.array([it["n"] for it in items]); tt = np.array([it["t"] for it in items])
    H_ADD, H_DEL = m.mem.h_add, m.mem.h_del
    res = {"run": RUN, "args_subset": {k: a.get(k) for k in ("think_k", "add_norm", "aux_nopass", "aux_reweight", "aux_hold", "k_window", "seed", "fast_add", "train_gpu")},
           "load": lres, "fade": float(m.mem.fade), "gamma": float(m.mem.gamma), "g_add": float(torch.exp(m.mem.s_add)), "g_del": float(torch.exp(m.mem.s_del)),
           "gate_bias_alpha": [float(v) for v in m.mem.gate.bias[:H_DEL]], "gate_bias_beta": [float(v) for v in m.mem.gate.bias[H_DEL:]],
           "gate_w_norm_alpha": [float(v) for v in m.mem.gate.weight[:H_DEL].norm(dim=1)], "gate_w_norm_beta": [float(v) for v in m.mem.gate.weight[H_DEL:].norm(dim=1)],
           "items": {"n_per_n": NPER if not QUICK else 1, "n_range": [4, 8], "seed": 19613, "B": B, "n_real_tokens_mean": float(msk.sum(1).double().mean())},
           "passes": {}, "aux": {}, "answer": {}, "timing": {}}
    print(RUN, "loaded", res["fade"], res["gamma"], "B", B, "T", x.shape[1], "%.1fs" % (time.time() - t0), flush=True)
    # ---------------- clean trajectory with recording: (a), (b), (d); aux decodes none / aux-only add / aux-only delta
    trajs = {"none": None, "add": "add", "delta": "delta"}
    aux_rec = {}; ans_rec = {}
    for tname, tabl in trajs.items():
        S = m.mem.zero_state(B, DEV); tw = time.time()
        for t in range(1, KMAX + 1):
            REC["on"] = tname == "none"; m.mem.ablate = tabl
            try:
                Sn = m.write(S, x, msk)
            finally:
                m.mem.ablate = None; REC["on"] = False
            if tname == "none":
                d = REC["data"]; REC["data"] = None; pr = {}
                pr["gates"] = gate_stats(d)
                pr["contraction"], pi = contraction(d, S[:, H_ADD:H_ADD + H_DEL], Sn[:, H_ADD:H_ADD + H_DEL])
                pr["by_n"] = by_n(pi, ns)
                for blk, sl in (("add", slice(0, H_ADD)), ("delta", slice(H_ADD, H_ADD + H_DEL))):
                    a0 = S[:, sl].flatten(1).double(); a1 = Sn[:, sl].flatten(1).double(); n0 = a0.norm(dim=1); n1 = a1.norm(dim=1)
                    pr["norm_" + blk] = q(n1.numpy())
                    if t >= 2:
                        pr["rel_inc_" + blk] = q(((a1 - a0).norm(dim=1) / n0.clamp_min(1e-30)).numpy())
                        pr["cos_prev_" + blk] = q(F.cosine_similarity(a0, a1, dim=1).numpy())
                res["passes"][str(t)] = pr
                del d
                if VALIDATE and t in (1, 3, 5, 7):                          # teacher-forced criterion vs the module's own aux_greedy (nested170 aux_eval logic)
                    idx, ok, _ = aux_exact(m, Sn, items, t, None); jj = idx[:24]; sub = [items[j] for j in jj]
                    tg = [N.ids(N.aux_of(it)[t - 1][0]) + [N.EOT] for it in sub]; gen = m.aux_greedy(Sn[jj], t, max(len(z) for z in tg))
                    gok = [bool(list(gen[r_, :len(tg[r_])]) == tg[r_]) for r_ in range(len(sub))]
                    res.setdefault("validate_greedy", {})[str(t)] = {"n": len(jj), "agree": int(sum(a_ == b_ for a_, b_ in zip(gok, ok[:24]))), "greedy_acc": float(np.mean(gok)),
                                                                     "tf_acc": float(np.mean(ok[:24]))}
                    print("VALIDATE", t, res["validate_greedy"][str(t)], flush=True)
            if t <= 8:
                decs = [("aux_none", None), ("aux_add", "add"), ("aux_delta", "delta")] if tname == "none" else [("write_only", None), ("all", tabl)]
                for dname, dabl in decs:
                    idx, ok, (hd, cp) = aux_exact(m, Sn, items, t, dabl, roles=(tname == "none" and dname == "aux_none"))
                    key = "%s|%s" % (tname, dname)
                    aux_rec.setdefault(key, {})[str(t)] = {"acc": float(np.mean(ok)) if ok else None, "n": len(ok), "idx": idx, "ok": ok,
                                                           **({"handle_acc": float(np.mean(hd)), "copy_acc": float(np.mean(cp))} if hd else {})}
            if t in (4, 8, 12) and (tname == "none" or t == 8):
                decs = [("dec_none", None), ("dec_add", "add"), ("dec_delta", "delta")] if (tname == "none" and t == 8) else \
                       ([("dec_none", None)] if tname == "none" else [("write_only", None), ("all", tabl)])
                for dname, dabl in decs:
                    A = answers(m, Sn, items, dabl); key = "%s|%s" % (tname, dname)
                    ans_rec.setdefault(key, {})[str(t)] = {"acc": float((A == tt).mean()), "acc_by_n": {str(n): float((A[ns == n] == tt[ns == n]).mean()) for n in range(4, 9) if (ns == n).any()},
                                                           "pred": A.tolist()}
            S = Sn
            print(RUN, tname, "pass", t, "%.1fs" % (time.time() - t0), flush=True)
            if tname != "none" and t >= 8:
                break
        res["timing"][tname] = time.time() - tw
    res["aux"] = aux_rec; res["answer"] = ans_rec
    # ---------------- Audit addition: long items n = 10, 12 (clean trajectory only; gates / alpha_eff^L / P spectrum by n)
    if not QUICK and NEXT > 0:
        rng = np.random.default_rng(19614); ext = []
        for n in (10, 12):
            for _ in range(NEXT):
                r, t_ = N.generate(rng, n); ext.append({"n": n, "syms": N.render(r), "t": t_, "root": r})
        xe, me = m.think_inputs(ext, DEV); ne = np.array([it["n"] for it in ext]); S = m.mem.zero_state(len(ext), DEV); tw = time.time(); res["passes_ext"] = {}
        for t in range(1, KMAX + 1):
            REC["on"] = True
            try:
                Sn = m.write(S, xe, me)
            finally:
                REC["on"] = False
            d = REC["data"]; REC["data"] = None; c_, pi = contraction(d, S[:, H_ADD:H_ADD + H_DEL], Sn[:, H_ADD:H_ADD + H_DEL])
            pr = {"by_n": by_n(pi, ne), "replay_rel_err_max": c_["replay_rel_err_max"]}
            for blk, sl in (("add", slice(0, H_ADD)), ("delta", slice(H_ADD, H_ADD + H_DEL))):
                a0 = S[:, sl].flatten(1).double(); a1 = Sn[:, sl].flatten(1).double()
                if t >= 2:
                    pr["rel_inc_" + blk] = q(((a1 - a0).norm(dim=1) / a0.norm(dim=1).clamp_min(1e-30)).numpy())
            res["passes_ext"][str(t)] = pr; S = Sn; del d
            print(RUN, "ext pass", t, "%.1fs" % (time.time() - t0), flush=True)
        res["timing"]["ext"] = time.time() - tw; res["items_ext"] = {"n": [10, 12], "per_n": NEXT, "seed": 19614}
    res["timing"]["total"] = time.time() - t0
    res["item_n"] = ns.tolist(); res["item_t"] = tt.tolist()
    fn = os.path.join(OUT, "emp_%s%s.json" % (RUN, "_quick" if QUICK else ""))
    json.dump(res, open(fn, "w", newline="\n"), indent=0)
    print("WROTE", fn, "%.1fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
