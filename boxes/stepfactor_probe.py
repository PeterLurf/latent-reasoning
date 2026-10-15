"""§204 eval-only readings on a final model.pt (docs/THEORY_STEPFACTOR_DRAFT_2026-10-14.md §5.4 / §8.4 / gate 11, note 204-7). Never imported by training.
  clock : clock_A.py's protocol (theory/stepfactor/clock_A.py, copied here so Theory's file can change freely): 200 Aug-valid items (rng 20214), fit / test
          halves (rng 20215), M_t for t = 1..8 in the trained regime, reads at the 3 positions of the fixed prompt [EOT, ' Step', ' :']; pass probes t 1..4
          and 1..8 (L2 logistic, lambda by 5-fold CV on the fit items; nearest-centroid beside). Note 204-7: "pair_acc" = the consecutive-pair accuracies
          from the t 1..4 probe on the model's own decoder-visible read = delta read + additive read AS THE MODEL COMPUTES IT (post-gain; the model's own
          RMS for add_norm rms, raw otherwise); pair_acc_by_feature = delta-only, raw additive, gain-1 RMS additive.   -> <run>/eval_s204_clock.json
  probe : PROBE-LEAK (draft §5.3 item 1, note 204-7): features = the 768-d pre-W_o read [g_add phi(q)^T S_add,h ; g_del S_del,h q_hat_h] at the LAST token of
          [EOT, ' Step', ' :'] (a fixed linear image of S), from M_1, M_t and M_0 (zero memory, the floor), t in {2, 3, 4}. Targets of step t: the value's
          first token (top 200 of the train items' value first tokens + OTHER), the operator signature (s204 op vocab, + OTHER), pointer 1's first token id
          (top 200 + OTHER). Separate L2-logistic probes per (t, source, target), fit on 4,000 Aug-train items (rng 20470) with lambda from an inner 80 / 20
          split (rng 20471); reported on Aug valid (test descriptive). "plan" = the mean of the operator and pointer-1 accuracies; probe_leak[target][t] =
          acc(M_1) / acc(M_t) on l >= 2 R-steps (grounding 'R', i.e. an operand is an earlier result).        -> <run>/eval_s204_probe.json
    python -m boxes.stepfactor_probe <run dir> clock|probe [--device cpu|cuda:N]      (fp32 always)"""
import argparse
import json
import os
import time

os.environ.setdefault("BOXES_NB", "5")

import numpy as np
import torch
import torch.nn.functional as Fn

from boxes import s5
import boxes.deltamem as DM
import boxes.gsm8k_data as G
import boxes.gsm8k_factor as GF
from boxes.prosqa168s2 import COLON_ID, ENC, EOS, EOT, STEP_ID, ProsMem

PROMPT = [EOT, STEP_ID, COLON_ID]


def build(run, dev):
    a_ = json.load(open(os.path.join(run, "args.json"))); g = s5.load_gpt2(dev)
    new = (a_.get("aux_target") or "string") != "string" or bool(a_.get("aux_nopass")) or bool(a_.get("add_norm"))
    m = (GF.FactorMem(g, a_.get("add_norm") or None, bool(a_.get("aux_nopass")), a_.get("aux_target") == "factor", a_.get("seed", 1)) if new else ProsMem(g)).to(dev)
    m.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location=dev, weights_only=False)); m.eval()
    return g, m, a_


@torch.no_grad()
def prompt_q(g, m, dev):
    x6 = DM.gpt_lower(g, g.embed(torch.tensor([PROMPT], device=dev)), m.mem.layer)[0]; a = m.mem.ln(x6.float())
    return m.mem.qkv(a)[:, :m.mem.D].view(len(PROMPT), m.mem.h_add + m.mem.h_del, DM.DH)


@torch.no_grad()
def reads(m, S, q_pos):
    """post-gain additive / delta reads of S at the prompt queries -> oa, od (B, P, H, DH) (clock_A.reads)."""
    mem = m.mem; ha = mem.h_add
    pq = Fn.elu(q_pos[:, :ha]) + 1; oa = torch.exp(mem.s_add) * torch.einsum("phk,bhkv->bphv", pq, S[:, :ha].float())
    od = torch.exp(mem.s_del) * torch.einsum("bhvk,phk->bphv", S[:, ha:ha + mem.h_del].float(), Fn.normalize(q_pos[:, ha:], dim=-1))
    return oa, od


def rms(oa):
    return oa * torch.rsqrt(oa.pow(2).mean(-1, keepdim=True) + 1e-6)


def model_add(m, oa):
    """the additive read as the model's decoder sees it: deltamem's own RMS (with its learned gain) for add_norm rms, raw otherwise."""
    md = getattr(m.mem, "add_norm", None)                                  # deltamem.add_read: rms(phi^T S) * add_gain[h], then * exp(s_add) (rms is scale-free)
    if md == "rms":
        return rms(oa) * m.mem.add_gain.detach().to(oa.device).view(1, 1, -1, 1) * torch.exp(m.mem.s_add).to(oa.device)
    assert md is None, "stepfactor_probe: add_norm %r not supported" % md
    return oa


@torch.no_grad()
def states_reads(m, D, q, T, dev, bs=25):
    """M_t reads for t = 1..T -> (OA, OD): lists over t of (N, P, H, DH) on CPU."""
    OA, OD = [[] for _ in range(T)], [[] for _ in range(T)]
    for i in range(0, len(D), bs):
        ch = D[i:i + bs]; x, msk = m.think_inputs(ch, dev); hl = m.lower(x); S = m.mem.zero_state(len(ch), dev)
        for t in range(T):
            S = m.write(S, x, msk, hl); oa, od = reads(m, S, q); OA[t].append(oa.cpu()); OD[t].append(od.cpu())
    return [torch.cat(v) for v in OA], [torch.cat(v) for v in OD]


def logreg(Xtr, ytr, Xte, lam, ncls):
    W = torch.zeros(Xtr.shape[1], ncls, requires_grad=True); b = torch.zeros(ncls, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=300, line_search_fn="strong_wolfe")

    def cl():
        opt.zero_grad(); L = Fn.cross_entropy(Xtr @ W + b, ytr) + lam * (W ** 2).sum(); L.backward(); return L
    opt.step(cl)
    with torch.no_grad():
        return Xte @ W + b


def pass_probe(F, items_fit, items_te, ts):
    """clock_A.probe: F t -> (N, d); classes = ts; lambda by grouped 5-fold CV on the fit items -> acc, consecutive-pair accs, centroid acc."""
    def stack(idx):
        X = torch.cat([F[t][idx] for t in ts]); y = torch.cat([torch.full((len(idx),), k, dtype=torch.long) for k in range(len(ts))]); return X, y
    Xf, yf = stack(items_fit); mu, sd = Xf.mean(0), Xf.std(0).clamp_min(1e-8); Xf = (Xf - mu) / sd
    Xt, yt = stack(items_te); Xt = (Xt - mu) / sd
    folds = np.array_split(np.arange(len(items_fit)), 5); best = None
    for lam in (1e-4, 1e-3, 1e-2, 1e-1):
        acc = []
        for f in folds:
            tr = np.setdiff1d(np.arange(len(items_fit)), f)
            Xa, ya = stack(items_fit[tr]); Xb, yb = stack(items_fit[f]); Xa, Xb = (Xa - mu) / sd, (Xb - mu) / sd
            acc.append(float((logreg(Xa, ya, Xb, lam, len(ts)).argmax(-1) == yb).float().mean()))
        if best is None or np.mean(acc) > best[1]:
            best = (lam, float(np.mean(acc)))
    lg = logreg(Xf, yf, Xt, best[0], len(ts)); pr = lg.argmax(-1); acc = float((pr == yt).float().mean())
    C = torch.stack([Xf[yf == k].mean(0) for k in range(len(ts))]); cacc = float((torch.cdist(Xt, C).argmin(-1) == yt).float().mean()); pairs = {}
    for k in range(len(ts) - 1):
        mk = (yt == k) | (yt == k + 1)
        dec = torch.where(lg[mk, k + 1] > lg[mk, k], k + 1, k); pairs["%dv%d" % (ts[k], ts[k + 1])] = round(float((dec == yt[mk]).float().mean()), 4)
    return {"acc": round(acc, 4), "chance": round(1 / len(ts), 4), "lambda": best[0], "cv_acc": round(best[1], 4), "centroid_acc": round(cacc, 4), "pair_acc": pairs}


def run_clock(run, dev):
    T0 = time.time(); g, m, a_ = build(run, dev); q = prompt_q(g, m, dev)
    Rva, _ = G.load_aug("valid", filter_bad=False); idx = np.random.default_rng(20214).permutation(len(Rva))[:200]; D = [G.encode(Rva[i], ENC, EOS) for i in idx]
    sp = np.random.default_rng(20215).permutation(200); fit, te = np.sort(sp[:100]), np.sort(sp[100:])
    OA, OD = states_reads(m, D, q, 8, dev); MA = [model_add(m, v.to(dev)).cpu() for v in OA]
    feats = {"model_read": {t + 1: torch.cat([MA[t].flatten(1), OD[t].flatten(1)], 1) for t in range(8)}, "delta": {t + 1: OD[t].flatten(1) for t in range(8)},
             "add_raw": {t + 1: OA[t].flatten(1) for t in range(8)}, "add_rms_gain1": {t + 1: rms(OA[t]).flatten(1) for t in range(8)}}
    pr = {nm: {"t1-4": pass_probe(F, fit, te, [1, 2, 3, 4]), "t1-8": pass_probe(F, fit, te, list(range(1, 9)))} for nm, F in feats.items()}
    res = {"pair_acc": pr["model_read"]["t1-4"]["pair_acc"], "pair_acc_by_feature": {nm: v["t1-4"]["pair_acc"] for nm, v in pr.items()}, "probe": pr,
           "model_read_def": "post-gain delta read + additive read as computed (add_norm %r: %s)" % (a_.get("add_norm") or None, "model RMS" if a_.get("add_norm") == "rms" else "raw"),
           "protocol": "clock_A.py (200 Aug-valid items rng 20214, fit / test 100 / 100 rng 20215, trained regime, prompt [EOT, ' Step', ' :'] 3 positions)",
           "model": os.path.join(run, "model.pt"), "device": dev, "dtype": "fp32", "gamma": float(m.mem.gamma), "fade": float(m.mem.fade), "sec": round(time.time() - T0)}
    json.dump(res, open(os.path.join(run, "eval_s204_clock.json"), "w", newline="\n"), indent=1); print("CLOCK", res["pair_acc"], res["sec"], flush=True)


def top_classes(vals, k=200):
    c = {}
    for v in vals:
        c[v] = c.get(v, 0) + 1
    return {v: i for i, (v, _) in enumerate(sorted(c.items(), key=lambda kv: -kv[1])[:k])}


def run_probe(run, dev, n_train=4000, TS=(2, 3, 4)):
    T0 = time.time(); g, m, a_ = build(run, dev); q = prompt_q(g, m, dev); opv, _ = GF.load_op_vocab(a_.get("op_vocab") or None)
    Rtr, _ = G.load_aug("train"); Rva, _ = G.load_aug("valid", filter_bad=False); Rte, _ = G.load_aug("test", filter_bad=False)
    itr = np.random.default_rng(20470).permutation(len(Rtr)); Rtr_ = [Rtr[i] for i in itr if len(Rtr[i]["eqs"]) >= min(TS)][:n_train]

    def prep(R):
        D = [G.encode(r, ENC, EOS) for r in R]; FT = [GF.factor_targets(r, G.parse_recomputable(r)[0], opv) for r in R]; return D, FT
    sets = {"train": prep(Rtr_), "valid": prep(Rva), "test": prep(Rte)}
    tv = top_classes([st["v"][0] for st_ in sets["train"][1] for st in st_ if st["v"]]); tp = top_classes([st["p"][0][0] for st_ in sets["train"][1] for st in st_ if st["p"][0]])

    def target(st, kind):
        if kind == "value":
            return None if not st["v"] else tv.get(st["v"][0], len(tv))
        if kind == "op":
            return st["op"]
        return None if not st["p"][0] else tp.get(st["p"][0][0], len(tp))
    ncls = {"value": len(tv) + 1, "op": len(opv) + 1, "ptr1": len(tp) + 1}
    feat = {}
    for nm, (D, _) in sets.items():
        OA, OD = states_reads(m, D, q, max(TS), dev); feat[nm] = {t + 1: torch.cat([OA[t][:, -1].flatten(1), OD[t][:, -1].flatten(1)], 1) for t in range(max(TS))}
        feat[nm][0] = torch.zeros_like(feat[nm][1]); print("reads", nm, len(D), round(time.time() - T0), flush=True)
    out = {"acc": {}, "n": {}}; rng = np.random.default_rng(20471)
    for t in TS:
        for kind in ("value", "op", "ptr1"):
            rows = {nm: [(i, target(FT[i][t - 1], kind), FT[i][t - 1]["g"]) for i in range(len(FT)) if len(FT[i]) >= t and target(FT[i][t - 1], kind) is not None]
                    for nm, (_, FT) in sets.items()}
            tr = rows["train"]; perm = rng.permutation(len(tr)); cut = int(0.8 * len(tr)); ia, ib = perm[:cut], perm[cut:]
            for src, ts_ in (("M1", 1), ("Mt", t), ("M0", 0)):
                Xtr = feat["train"][ts_][[r[0] for r in tr]].float(); ytr = torch.tensor([r[1] for r in tr]); mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-8); Xs = (Xtr - mu) / sd
                best = None
                for lam in (1e-4, 1e-3, 1e-2, 1e-1):
                    a_c = float((logreg(Xs[ia], ytr[ia], Xs[ib], lam, ncls[kind]).argmax(-1) == ytr[ib]).float().mean())
                    if best is None or a_c > best[1]:
                        best = (lam, a_c)
                evs = {ev: [r for r in rows[ev]] for ev in ("valid", "test")}; Xall = torch.cat([feat[ev][ts_][[r[0] for r in evs[ev]]].float() for ev in evs])
                pred = logreg(Xs, ytr, (Xall - mu) / sd, best[0], ncls[kind]).argmax(-1); off = 0; P_ = {}
                for ev in evs:
                    P_[ev] = pred[off:off + len(evs[ev])]; off += len(evs[ev])
                for ev in ("valid", "test"):
                    for sub, keep in (("R", lambda r: r[2] == "R"), ("all", lambda r: True)):
                        mk = torch.tensor([keep(r) for r in evs[ev]], dtype=torch.bool)
                        if not mk.any():
                            continue
                        ye = torch.tensor([r[1] for r in evs[ev]])[mk]; acc = float((P_[ev][mk] == ye).float().mean()); rr = ye
                        out["acc"].setdefault(ev, {}).setdefault(sub, {}).setdefault(kind, {}).setdefault(str(t), {})[src] = round(acc, 4)
                        out["n"].setdefault(ev, {}).setdefault(sub, {}).setdefault(kind, {})[str(t)] = len(rr)
                print("probe t", t, kind, src, "lam", best[0], round(time.time() - T0), flush=True)
    A = out["acc"]["valid"]["R"]; pl = {"plan": {}, "value": {}, "op": {}, "ptr1": {}}
    for t in map(str, TS):
        for kind in ("value", "op", "ptr1"):
            pl[kind][t] = A[kind][t]["M1"] / A[kind][t]["Mt"] if A[kind][t]["Mt"] else None
        p1 = (A["op"][t]["M1"] + A["ptr1"][t]["M1"]) / 2; pt = (A["op"][t]["Mt"] + A["ptr1"][t]["Mt"]) / 2; pl["plan"][t] = p1 / pt if pt else None
    res = {"probe_leak": pl, "probe_acc": out["acc"], "n": out["n"], "classes": ncls, "n_train_items": len(Rtr_),
           "def": "note 204-7: plan = mean(op, ptr1) probe accuracy; ratio = acc(M_1)/acc(M_t) on l >= 2 R-steps (grounding 'R'), Aug valid; test descriptive; M0 = zero-memory floor",
           "features": "768-d post-gain [add ; delta] read at the last token of [EOT, ' Step', ' :']", "model": os.path.join(run, "model.pt"), "device": dev, "dtype": "fp32",
           "sec": round(time.time() - T0)}
    json.dump(res, open(os.path.join(run, "eval_s204_probe.json"), "w", newline="\n"), indent=1); print("PROBE", pl, res["sec"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("what", choices=["clock", "probe"]); ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_train", type=int, default=4000); a = ap.parse_args()
    os.environ.setdefault("BOXES_NB", "5"); torch.set_num_threads(3); DM.FAST_ADD = True; DM.CHUNK_DELTA = 64
    (run_clock if a.what == "clock" else lambda r, d: run_probe(r, d, a.n_train))(a.run, a.device)
