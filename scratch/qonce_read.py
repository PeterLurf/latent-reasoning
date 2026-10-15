"""§208 QUESTION-ONCE §5 readings, eval-only, fp32 (QONCE draft 1c2dc4d3 §5 / §12.3 / §16.2; note 208-1: a NEW file, no module edit until §208 lands).
Uses scratch/qonce_eval.py's replay machinery (G6-gated). -> <run>/eval_qonce.json (merge-on-write; each part dumped as it finishes)
  clean       clean_fp32 {acc, by_m, n_by_m}; kprime_fp32 {K': {acc, by_m}}; acc_by_nlit; R_QO
  meaniter    mean_iter {k: {acc, by_m}} (k = 1, 2, 3, 4, 6, 8; passes k+1..K = their test-set item means)
  transplant  PROGRESS-TRANSPLANT on TRAIN-REDRAW pairs: transplant {C_t2, C_t3, S_t2, S_t3: {n, P_hyb, P_own, P_don, hyb_null, by_m}, pool}
  probes      lit_recall {t: {Mt, M0, length_only, y_only: {all, nlit_le2, nlit_ge4, n}}}, int_progress {Ml1, M1, M0}, ans_lin
  clock       clock {pair_acc {2v3, 3v4, 5v6}, pair_acc_by_feature}
  g6          the G6 identities re-run on this model (gate before any ITER-GAIN read)
    python scratch/qonce_read.py <run dir> all|clean|meaniter|transplant|probes|clock|g6 [--device cuda:1]"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, "D:/Latent/scratch"); sys.path.insert(0, "D:/Latent/theory/ansbottle")
import qonce_eval as QE  # noqa: E402  (sets ROOT / cwd, BOXES_NB)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.gsm8k_data as G  # noqa: E402
import boxes.stepfactor_probe as SP  # noqa: E402
from boxes.gsm8k_stepaux import ans_ok  # noqa: E402
from boxes.prosqa168s2 import ANS, ENC, EOS, EOT  # noqa: E402
from bench.tasks import norm_number  # noqa: E402

DH = DM.DH
NOAC = lambda: torch.autocast("cuda", enabled=False)  # noqa: E731


@torch.no_grad()
def think_all(m, D, Kp, dev, bs=64, M0=None):
    out = []
    for i in range(0, len(D), bs):
        ch = D[i:i + bs]; x, msk = m.think_inputs(ch, dev); S = m.mem.zero_state(len(ch), dev) if M0 is None else M0[i:i + bs].to(dev)
        out.append(m.think(S, x, msk, Kp, cache=True).cpu())
    return torch.cat(out)


def decode(g, m, S, dev, bs=64):
    P = []
    for i in range(0, len(S), bs):
        P += QE.answers(g, m, S[i:i + bs].to(dev), len(S[i:i + bs]), dev, NOAC, bs)
    return P


def score(P, D):
    return np.array([ans_ok(p_, it["ans"]) for p_, it in zip(P, D)])


def strata(ok, D):
    mt = np.array([it["m"] for it in D]); ok = np.asarray(ok, bool)
    return {str(k): float(ok[mt == k].mean()) for k in sorted(set(mt.tolist()))}, {str(k): int((mt == k).sum()) for k in sorted(set(mt.tolist()))}


@torch.no_grad()
def mean_iter(g, m, D, K, dev, ks=(1, 2, 3, 4, 6, 8), bs=64):
    """MEAN-ITER(k): passes 1..k as run; passes k+1..K replaced by their test-set item means (additive increment sum; delta affine map P, C)."""
    sums = {k: None for k in ks}; n = 0
    for i in range(0, len(D), bs):
        _, st, maps = QE.think_record(m, D[i:i + bs], K, dev); n += len(D[i:i + bs])
        for k in ks:
            inc, P, C = QE.compose(maps[k:]); t_ = (inc.sum(0), P.sum(0), C.sum(0))
            sums[k] = t_ if sums[k] is None else tuple(a_ + b_ for a_, b_ in zip(sums[k], t_))
    means = {k: tuple(x_ / n for x_ in v) for k, v in sums.items()}; preds = {k: [] for k in ks}
    for i in range(0, len(D), bs):
        ch = D[i:i + bs]; _, st, _ = QE.think_record(m, ch, K, dev)
        for k in ks:
            Sa, Sd = QE.split(m.mem, st[k]); mi, mP, mC = means[k]; S = torch.cat([Sa + mi, Sd @ mP + mC], 1)
            preds[k] += QE.answers(g, m, S, len(ch), dev, NOAC, bs)
    return preds


def redraw_item(raw, tpl, rng):
    """an R_x redraw -> (new raw item, qv, values) or None (pairs_calc.redraw's loop, keeping the rendered item)."""
    import pairs_calc as PC
    for _ in range(G.TRIES):
        newq = {}
        for i in tpl["used_q"]:
            tok = next(tk for _, _, tk, v in tpl["qn"] if G._near(v, tpl["qvals"][i])); newq[i] = G.draw_literal(tok, rng)
        if not PC.ok_values(tpl, newq):
            continue
        res, vals = G.render(raw, tpl, newq)
        if res[0] is None:
            continue
        q, eqs, ans = res; return {"q": q, "eqs": eqs, "ans": ans}, newq, vals
    return None


def _num(p_):
    try:
        return float(norm_number(p_)) if p_ is not None else None
    except Exception:
        return None


@torch.no_grad()
def transplant(g, m, K, dev, n_tpl=4000, draws=16, bs=64):
    """PROGRESS-TRANSPLANT (draft §5.4): hybrid after pass t (t = 2, 3) = the recipient's pass-1 state + the donor's passes 2..t (replayed maps), passes
    t+1..K live, decoded. Targets C (donor steps 1..t-1) / S (donor steps 2..t, recipient v_1) via pairs_calc.chain; kept = y_hyb not in {y_own, y_don},
    y_own != y_don, strict = renderable. One pair per recipient: the first clean-correct instance of a template = recipient, the first other clean-correct
    instance strict-kept under that convention / t = donor. hyb_null = P(answer = y_hyb) on the unswapped recipient."""
    import pairs_calc as PC
    tr, _ = G.load_aug("train"); rng = np.random.default_rng(20711); idx = rng.permutation(len(tr))[:n_tpl]; groups = []
    for i in idx:
        r = tr[i]; tpl, _ = G.parse_recomputable(r)
        if tpl is None or len(tpl["eqt"]) < 2:
            continue
        inst = [x for x in (redraw_item(r, tpl, rng) for _ in range(draws)) if x is not None]
        if len(inst) >= 2:
            groups.append((tpl, inst))
    flat = [(gi, j) for gi, (_, inst) in enumerate(groups) for j in range(len(inst))]; D = [G.encode(groups[gi][1][j][0], ENC, EOS) for gi, j in flat]
    ok = score(decode(g, m, think_all(m, D, K, dev, bs), dev, bs), D); okmap = {k_: bool(o_) for k_, o_ in zip(flat, ok)}
    SETS = {"C": {2: [0], 3: [0, 1]}, "S": {2: [1], 3: [1, 2]}}; pairs = {(cv, t): [] for cv in SETS for t in (2, 3)}
    for gi, (tpl, inst) in enumerate(groups):
        cor = [j for j in range(len(inst)) if okmap[(gi, j)]]
        if len(cor) < 2:
            continue
        r_ = cor[0]; qvr, vr = inst[r_][1], inst[r_][2]; mm = len(tpl["eqt"])
        for cv, tt in SETS.items():
            for t, steps in tt.items():
                for d_ in cor[1:]:
                    vd = inst[d_][2]
                    if PC.tol_eq(vd[-1], vr[-1]):
                        continue
                    ch = PC.chain(tpl, qvr, {s_: vd[s_] for s_ in steps if s_ < mm}); y = ch[-1] if ch else None
                    if y is None or not math.isfinite(y) or PC.tol_eq(y, vr[-1]) or PC.tol_eq(y, vd[-1]) or PC.render_ans(y) is None:
                        continue
                    pairs[(cv, t)].append((gi, r_, d_, y, vr[-1], vd[-1], mm)); break
    out = {"pool": {"templates_sampled": int(n_tpl), "templates_used": len(groups), "instances": len(D), "clean_acc_instances": float(ok.mean()), "draws": draws, "rng": 20711}}
    hit = lambda p_, y: (_num(p_) is not None and PC.tol_eq(_num(p_), y))  # noqa: E731
    for (cv, t), L in pairs.items():
        key = "%s_t%d" % (cv, t)
        if not L:
            out[key] = {"n": 0}; continue
        Dr = [G.encode(groups[gi][1][r_][0], ENC, EOS) for gi, r_, *_ in L]; Dd = [G.encode(groups[gi][1][d_][0], ENC, EOS) for gi, _, d_, *_ in L]; Ph = []
        for i in range(0, len(L), bs):
            _, str_, _ = QE.think_record(m, Dr[i:i + bs], 1, dev); _, _, mdd = QE.think_record(m, Dd[i:i + bs], t, dev)
            Sa1, Sd1 = QE.split(m.mem, str_[1]); inc, P_, C_ = QE.compose(mdd[1:t]); S = torch.cat([Sa1 + inc, Sd1 @ P_ + C_], 1)
            S = QE.live_from(m, S, Dr[i:i + bs], t, K, dev); Ph += QE.answers(g, m, S, len(S), dev, NOAC, bs)
        Pr = decode(g, m, think_all(m, Dr, K, dev, bs), dev, bs)
        H = np.array([hit(p_, x[3]) for p_, x in zip(Ph, L)]); O = np.array([hit(p_, x[4]) for p_, x in zip(Ph, L)]); Dn = np.array([hit(p_, x[5]) for p_, x in zip(Ph, L)])
        Hn = np.array([hit(p_, x[3]) for p_, x in zip(Pr, L)]); ms = np.array([x[6] for x in L]); by = {}
        for k_ in sorted(set(ms.tolist())):
            sel = ms == k_; by[str(k_)] = {"n": int(sel.sum()), "P_hyb": float(H[sel].mean()), "P_own": float(O[sel].mean()), "P_don": float(Dn[sel].mean())}
        out[key] = {"n": len(L), "P_hyb": float(H.mean()), "P_own": float(O.mean()), "P_don": float(Dn.mean()), "hyb_null": float(Hn.mean()), "by_m": by}
    return out


@torch.no_grad()
def prompt_reads(g, m, S, ids, dev, bs=256):
    """post-gain [add ; delta] reads of S at the positions of a fixed prompt `ids` (queries item-independent) -> (N, len(ids) x 768) fp32 CPU."""
    x = torch.tensor([ids], device=dev); x6 = DM.gpt_lower(g, g.embed(x), m.mem.layer)[0]; a = m.mem.ln(x6.float())
    q = m.mem.qkv(a)[:, :m.mem.D].view(len(ids), m.mem.h_add + m.mem.h_del, DH); out = []
    for i in range(0, len(S), bs):
        oa, od = SP.reads(m, S[i:i + bs].to(dev), q); out.append(torch.cat([oa, od], 2).flatten(1).cpu())
    return torch.cat(out)


def pca_fit(X, k=1024):
    mu = X.mean(0); _, _, V = torch.pca_lowrank(X - mu, q=min(k, X.shape[1], X.shape[0] - 1), center=False, niter=4); return mu, V


def fit_probe(Xtr, ytr, ncls, seed):
    """L2 logistic (stepfactor_probe.logreg), lambda from an inner 80 / 20 split; features standardised on train -> predict fn."""
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-8); Xs = (Xtr - mu) / sd; rng = np.random.default_rng(seed); p_ = rng.permutation(len(Xs)); c_ = int(0.8 * len(Xs)); best = None
    for lam in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        a_ = float((SP.logreg(Xs[p_[:c_]], ytr[p_[:c_]], Xs[p_[c_:]], lam, ncls).argmax(-1) == ytr[p_[c_:]]).float().mean())
        if best is None or a_ > best[1]:
            best = (lam, a_)
    return (lambda X: SP.logreg(Xs, ytr, (X - mu) / sd, best[0], ncls).argmax(-1)), best[0]


def top_cls(vals, k=200):
    c = {}
    for v in vals:
        c[v] = c.get(v, 0) + 1
    return {v: i for i, (v, _) in enumerate(sorted(c.items(), key=lambda kv: -kv[1])[:k])}


@torch.no_grad()
def probes(g, m, K, dev, n_train=4000, bs=64):
    """LIT-RECALL(t) (t = 1, K; one probe per literal slot j, pooled; features = the 17 filler-position reads, PCA 1,024 on train), INT-PROGRESS (value
    first token of l >= 2 R-steps from M_{l+1} vs M_1 vs M_0, same features), ANS-LIN ((r0, r1) at K -> y's first token / the decoder's own accuracy).
    Floors: M_0, length-only (log question length), y-only (one-hot of y's first-token class). Fit on n_train fresh-R_x Aug-train items (rng 20720),
    read on the recomputable Aug-valid items. Literal / value classes: first BPE token of ' ' + the rendered number, top 200 + OTHER."""
    import pairs_calc as PC
    tr, _ = G.load_aug("train"); rng = np.random.default_rng(20720); TR = []
    for i in rng.permutation(len(tr)):
        tpl, _ = G.parse_recomputable(tr[i])
        if tpl is None:
            continue
        x = redraw_item(tr[i], tpl, rng)
        if x is not None:
            tp2, _ = G.parse_recomputable({**tr[i], **x[0]})
            if tp2 is not None:
                TR.append((x[0], tp2))
        if len(TR) >= n_train:
            break
    va, _ = G.load_aug("valid", filter_bad=False); VA = [(r, G.parse_recomputable(r)[0]) for r in va]; VA = [(r, t) for r, t in VA if t is not None]
    TS = sorted(set([1, K] + list(range(2, min(K, 6) + 1))))

    def feats(R):
        D = [G.encode(r, ENC, EOS) for r, _ in R]; Ss = {t: think_all(m, D, t, dev, bs) for t in TS}
        Fr = {t: prompt_reads(g, m, S, m.qo_ids, dev) for t, S in Ss.items()}; A = prompt_reads(g, m, Ss[K], [EOT, ANS], dev)
        return D, Fr, A, score(decode(g, m, Ss[K], dev, bs), D)
    Dt, Ft, At, okt = feats(TR); Dv, Fv, Av, okv = feats(VA); res = {"n_train": len(TR), "n_valid": len(VA), "train_rng": 20720}
    pcas = {t: pca_fit(Ft[t]) for t in Ft}

    def proj(Fs, t, rows):
        if t == 0:
            return torch.zeros(len(rows), 1024)
        mu, V = pcas[t]; return (Fs[t][rows] - mu) @ V
    lit = lambda tp, j: ENC.encode(" " + next(tk for _, _, tk, v in tp["qn"] if G._near(v, tp["qvals"][tp["used_q"][j]])))[0]   # noqa: E731  (the literal as written in the question)
    LR = {}
    for t in (1, K):
        rows_v = []
        for j in range(6):
            tr_ = [(i, lit(tp, j)) for i, (_, tp) in enumerate(TR) if j < len(tp["used_q"])]; va_ = [(i, lit(tp, j), len(tp["used_q"])) for i, (_, tp) in enumerate(VA) if j < len(tp["used_q"])]
            if len(tr_) < 200 or not va_:
                continue
            cls = top_cls([y for _, y in tr_]); yt = torch.tensor([cls.get(y, len(cls)) for _, y in tr_]); yv = torch.tensor([cls.get(y, len(cls)) for _, y, _ in va_])
            it_, iv_ = [i for i, _ in tr_], [i for i, _, _ in va_]
            for src, tt in (("Mt", t), ("M0", 0)):
                pr, _ = fit_probe(proj(Ft, tt, it_), yt, len(cls) + 1, 20721 + j); rows_v += [(src, nl, bool(h)) for (_, _, nl), h in zip(va_, (pr(proj(Fv, tt, iv_)) == yv).numpy())]
            Lt = torch.tensor([[math.log(len(Dt[i]["q"]))] for i in it_]); Lv = torch.tensor([[math.log(len(Dv[i]["q"]))] for i in iv_])
            pr, _ = fit_probe(Lt, yt, len(cls) + 1, 20727 + j); rows_v += [("length_only", nl, bool(h)) for (_, _, nl), h in zip(va_, (pr(Lv) == yv).numpy())]
            ycl = top_cls([Dt[i]["a1"][1] for i in it_]); Yt = F.one_hot(torch.tensor([ycl.get(Dt[i]["a1"][1], len(ycl)) for i in it_]), len(ycl) + 1).float()
            Yv = F.one_hot(torch.tensor([ycl.get(Dv[i]["a1"][1], len(ycl)) for i in iv_]), len(ycl) + 1).float(); pr, _ = fit_probe(Yt, yt, len(cls) + 1, 20733 + j)
            rows_v += [("y_only", nl, bool(h)) for (_, _, nl), h in zip(va_, (pr(Yv) == yv).numpy())]
        agg = {}
        for src in ("Mt", "M0", "length_only", "y_only"):
            sel = [r_ for r_ in rows_v if r_[0] == src]; le, ge = [h for _, nl, h in sel if nl <= 2], [h for _, nl, h in sel if nl >= 4]
            agg[src] = {"all": float(np.mean([h for *_, h in sel])) if sel else None, "nlit_le2": float(np.mean(le)) if le else None, "nlit_ge4": float(np.mean(ge)) if ge else None,
                        "n": len(sel), "n_le2": len(le), "n_ge4": len(ge)}
        LR[str(t)] = agg
    res["lit_recall"] = LR

    def vrows(R, D):
        out = []
        for i, (_, tp) in enumerate(R):
            for s_, l_ in enumerate(PC.levels(tp)):
                if l_ >= 2 and l_ + 1 <= min(K, 6) and s_ < len(D[i]["eqs"]):
                    out.append((i, l_, ENC.encode(" " + D[i]["eqs"][s_].split("=")[-1].strip())[0]))
        return out
    vt, vv = vrows(TR, Dt), vrows(VA, Dv); ip = {"n_train_steps": len(vt), "n_valid_steps": len(vv)}
    if vt and vv:
        cls = top_cls([y for *_, y in vt]); yt = torch.tensor([cls.get(y, len(cls)) for *_, y in vt]); yv = torch.tensor([cls.get(y, len(cls)) for *_, y in vv])

        def pick(Fs, rows, src):
            return torch.stack([proj(Fs, (l_ + 1) if src == "Ml1" else (1 if src == "M1" else 0), [i])[0] for i, l_, _ in rows])
        for src in ("Ml1", "M1", "M0"):
            pr, _ = fit_probe(pick(Ft, vt, src), yt, len(cls) + 1, 20740); ip[src] = float((pr(pick(Fv, vv, src)) == yv).float().mean())
    res["int_progress"] = ip
    ycl = top_cls([d["a1"][1] for d in Dt]); yt = torch.tensor([ycl.get(d["a1"][1], len(ycl)) for d in Dt]); yv = torch.tensor([ycl.get(d["a1"][1], len(ycl)) for d in Dv])
    pr, _ = fit_probe(At, yt, len(ycl) + 1, 20750); acc_ans = float((pr(Av) == yv).float().mean())
    res["ans_lin"] = {"acc_first_token": acc_ans, "decoder_acc_valid": float(okv.mean()), "ratio": acc_ans / max(float(okv.mean()), 1e-9)}
    return res


def clock(g, m, K, dev):
    """CLOCK pair accuracy on the aux-visible read ([EOT, ' Step', ' :']; note 204-7: post-gain add + delta), clock_A's protocol, pass probe t 1..8."""
    q = SP.prompt_q(g, m, dev); Rva, _ = G.load_aug("valid", filter_bad=False); idx = np.random.default_rng(20214).permutation(len(Rva))[:200]
    D = [G.encode(Rva[i], ENC, EOS) for i in idx]; sp = np.random.default_rng(20215).permutation(200); fit, te = np.sort(sp[:100]), np.sort(sp[100:])
    OA, OD = SP.states_reads(m, D, q, 8, dev); MA = [SP.model_add(m, v.to(dev)).cpu() for v in OA]
    feats = {"model_read": {t + 1: torch.cat([MA[t].flatten(1), OD[t].flatten(1)], 1) for t in range(8)}, "delta": {t + 1: OD[t].flatten(1) for t in range(8)},
             "add_raw": {t + 1: OA[t].flatten(1) for t in range(8)}}
    pr = {nm: SP.pass_probe(Fm, fit, te, list(range(1, 9))) for nm, Fm in feats.items()}
    return {"pair_acc": {k: pr["model_read"]["pair_acc"][k] for k in ("2v3", "3v4", "5v6")}, "pair_acc_by_feature": {nm: v["pair_acc"] for nm, v in pr.items()}, "probe_t1_8": pr}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("what"); ap.add_argument("--device", default="cuda:1"); a = ap.parse_args()
    torch.set_num_threads(3); dev = a.device; T0 = time.time(); g, m, A = QE.build(a.run, dev); K = int(A["think_k"])
    OUTP = os.path.join(a.run, "eval_qonce.json"); res = json.load(open(OUTP)) if os.path.exists(OUTP) else {}
    res.update({"model": os.path.join(a.run, "model.pt"), "dtype": "fp32", "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "K": K})
    R, _ = G.load_aug("test", filter_bad=False); D = [G.encode(r, ENC, EOS) for r in R]; parts = ["g6", "clean", "meaniter", "transplant", "probes", "clock"] if a.what == "all" else [a.what]

    def dump(tag):
        res.setdefault("sec", {})[tag] = round(time.time() - T0); json.dump(res, open(OUTP, "w", newline="\n"), indent=1, default=str); print("WROTE", tag, round(time.time() - T0), flush=True)
    for part in parts:
        if part == "g6":
            res["g6"] = QE.g6(a.run, dev, 64); dump("g6")
        elif part == "clean":
            kp = {}; okK = None
            for k in sorted({0, 1, 2, 3, 4, 6, 8, K, 12, 16}):
                ok = score(decode(g, m, think_all(m, D, k, dev), dev), D); by, nb = strata(ok, D); kp[str(k)] = {"acc": float(ok.mean()), "by_m": by}
                if k == K:
                    res["clean_fp32"] = {"acc": float(ok.mean()), "by_m": by, "n_by_m": nb}; okK = ok
            res["kprime_fp32"] = kp; nl = []
            for r in R:
                tp, _ = G.parse_recomputable(r); nl.append(len(tp["used_q"]) if tp is not None else -1)
            nl = np.array(nl); res["acc_by_nlit"] = {str(k): {"acc": float(okK[nl == k].mean()), "n": int((nl == k).sum())} for k in sorted(set(nl.tolist())) if k >= 0}
            le, ge = okK[(nl >= 1) & (nl <= 2)], okK[nl >= 4]
            res["R_QO"] = {"acc_nlit_le2": float(le.mean()), "acc_nlit_ge4": float(ge.mean()), "R_QO": float(ge.mean() / le.mean()) if le.mean() else None, "n_le2": len(le), "n_ge4": len(ge),
                           "def": "n_lit = len(parse_recomputable(item).used_q); non-recomputable test items excluded"}
            dump("clean")
        elif part == "meaniter":
            P = mean_iter(g, m, D, K, dev); res["mean_iter"] = {}
            for k, Pk in P.items():
                ok = score(Pk, D); by, _ = strata(ok, D); res["mean_iter"][str(k)] = {"acc": float(ok.mean()), "by_m": by}
            res["mean_iter_def"] = "passes 1..k as run; passes k+1..K replaced by their TEST-SET item means (additive increment sum; delta affine map P, C from delta_chunk on 0 / I)"
            dump("meaniter")
        elif part == "transplant":
            res["transplant"] = transplant(g, m, K, dev); dump("transplant")
        elif part == "probes":
            res["probes"] = probes(g, m, K, dev); dump("probes")
        elif part == "clock":
            res["clock"] = clock(g, m, K, dev); dump("clock")


if __name__ == "__main__":
    main()
