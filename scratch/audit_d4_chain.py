"""Audit $0 DESCRIPTIVE diagnostic (§234 P-4): where does the d = 4 chain break in boxes/runs_fsq/aw_plaw4_s1 (InjectAW, answer-position writes)?
Items: N TEST2-style items per d in {3, 4}: OW.Stream(23401, build_sets()['excl'], frames=FRAME_TRAIN, **OWKW).draw(d) (the eval's generator / exclusion), tok_aw(step cue,
answer write). K = d passes (pass t input = text + QUESTION(S_t) + ' The answer is:'; only ':' writes). Model fp32, no_grad, batch <= 64.
Per pass t = 1..d:
 (a) ':' argmax over the 11 number tokens vs v_t / v_{t-1} / v_{t+1} (categories overlap when values coincide; 'strict' = items with v_t != v_{t-1} and v_t != v_{t+1}).
 (b) probe v_t and v_{t-1} from u = mem.ln_w(h) at ':' (the write source) and from h6 = the residual entering block 6 at ':' (pre-read).
 (c) probe v_{t-1} from S_{t-1} (the state pass t reads; t >= 2), v_t from S_t, v_{t-1} from S_t; probe v_{t-1} from the pre-W_o read vector at ':' of pass t (read_vec on S_{t-1}).
 (d) ORACLE-STATE at pass t (t = 2..d; the d-th is the asked one): S_{t-1} := the state of a clean run on the re-rooted item R[t-1] (the generator's R field: S_{t-1}'s value given in the
     text, back-referenced constants kept), input = [EOT] + enc(' ' + R[t-1] minus its final question + ' ' + QUESTION(S_{t-1})) + ' The answer is:', either 1 pass from S_0 ('o1')
     or t-1 identical passes ('oh'), or 'hyb' = original passes 1..t-2 then pass t-1 on the re-rooted input; pass t then runs on the ORIGINAL item input; ':' argmax vs v_t. Controls: 'self' (S_{t-1} := its own),
     'text_only' (no state, pass 1 on the re-rooted input with QUESTION(S_t): v_t from the text alone).
Probe: sig split 60/20/20 (rng 0), standardised multinomial logistic (LBFGS), L2 wd in {1e-4,1e-3,1e-2} on val, test acc; chance = test majority share, 1/11.
Usage: python -B scratch/audit_d4_chain.py [device=cuda:1] [N=2000]"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("BOXES_NB", "5")
import boxes.inject_ow as IO  # noqa: E402
import boxes.inject_st as ST  # noqa: E402
import boxes.inject_aw as AW  # noqa: E402
import boxes.inject_aw_eval as EV  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
import boxes.ow_words as OW  # noqa: E402
from boxes.nested170 import EOT, NUM_IDS, pad  # noqa: E402

RUN = os.path.join(ROOT, "boxes/runs_fsq/aw_plaw4_s1"); OUT = os.path.join(ROOT, "scratch/audit_d4_chain/diag.json")
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000; P = 11; SEED = 23401; BS = 64
if dev == "cpu":
    torch.set_num_threads(3)


def fit_probe(X, y, tr, va, te):
    X = X.float(); mu = X[tr].mean(0, keepdim=True); sd = X[tr].std(0, keepdim=True) + 1e-6; Z = ((X - mu) / sd).to(dev); Y = torch.tensor(y, device=dev); best = None
    for wd in (1e-4, 1e-3, 1e-2):
        W = torch.zeros(Z.shape[1], P, device=dev, requires_grad=True); b = torch.zeros(P, device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=200, history_size=20, line_search_fn="strong_wolfe")

        def clo():
            opt.zero_grad(); l = F.cross_entropy(Z[tr] @ W + b, Y[tr]) + 0.5 * wd * (W ** 2).sum(); l.backward(); return l
        with torch.enable_grad():
            opt.step(clo)
        with torch.no_grad():
            pr = (Z @ W + b).argmax(-1); va_acc = float((pr[va] == Y[va]).float().mean())
            if best is None or va_acc > best[0]:
                best = (va_acc, wd, pr[te].cpu().numpy(), float((pr[tr] == Y[tr]).float().mean()))
    yt = y[te]
    del Z
    return {"test_acc": float((best[2] == yt).mean()), "val_acc": best[0], "train_acc": best[3], "wd": best[1], "n_test": int(len(te)),
            "chance_majority": float(np.bincount(yt, minlength=P).max() / len(yt)), "chance_uniform": 1 / P, "feature_dim": int(X.shape[1]), "test_pred": best[2].tolist()}


def reroot_ids(raw, k, cue_t):
    """[EOT] + enc(' ' + R[k] without its final question + ' ' + QUESTION(S_cue_t)) + ' The answer is:'; asserts S_k's given sentence is in the text."""
    R = raw["R"][k]; q = " " + raw["question"]; assert R.endswith(q), "R[k] must end with the final question"; tx = R[:-len(q)]
    g_ = OW.GIVEN.format(A=raw["names"]["spine"][k], v=raw["values"][k], u=raw["unit"]); assert g_ in tx, ("S_k given sentence missing in R[k]", k)
    return [EOT] + IO.ENC.encode(" " + tx + " " + ST.cue_str(raw, cue_t)) + IO.SFX


def run_d(d, m, g, sets):
    t0 = time.time(); mem = m.mem; num = torch.tensor(NUM_IDS, device=dev)
    st = OW.Stream(SEED, sets["excl"], frames=OW.FRAME_TRAIN, **IO.OWKW); raw = [st.draw(d) for _ in range(N)]
    T = [AW.tok_aw(it, "value", "step", "answer") for it in raw]; sigs = [it["sig"] for it in raw]; V = np.array([[int(v) for v in it["values"]] for it in raw])   # (N, d+1) v_0..v_d
    n = N; arg = np.zeros((n, d), int); decK = np.full(n, -1); U = {}; H6 = {}; RD = {}; SF = {}
    orc = {}; chk = 0.0
    for t in range(1, d + 1):
        U[t] = []; H6[t] = []; RD[t] = []; SF[t] = []
    for t in range(2, d + 1):
        orc[t] = {nm: np.zeros(n, int) for nm in ("self", "o1", "oh", "hyb", "text_only")}
    for i in range(0, n, BS):
        ch = T[i:i + BS]; rw = raw[i:i + BS]; B = len(ch); ar = torch.arange(B, device=dev); xs, msk = AW.cue_inputs(m, ch, dev); hls = [m.lower(x) for x in xs]
        S = mem.zero_state(B, dev); Sl = [S]
        for t in range(1, d + 1):
            Sp = Sl[-1]; S, h, pos = m.pass_aw(Sp, xs[t - 1], msk, hls[t - 1]); Sl.append(S)
            arg[i:i + B, t - 1] = m.colon_logits(h, pos)[:, num].argmax(-1).cpu().numpy()
            h6p = hls[t - 1][ar, pos].float(); U[t].append(mem.ln_w(h[ar, pos].float()).cpu()); H6[t].append(h6p.cpu()); SF[t].append(S.flatten(1).float().cpu())
            rv = EV.read_vec(mem, h6p, Sp); RD[t].append(rv.cpu())
            if i == 0 and t >= 2:                                                   # read_vec == read_fixed's own capture at the ':' row (as part_probe's check)
                rc_ = []; DM.read_fixed(mem, hls[t - 1][ar, pos][:, None], Sp, rc_)
                chk = max(chk, float((rc_[0] - rv[:, :DM.H_ADD * DM.DH]).abs().max() / rc_[0].abs().max().clamp_min(1e-30)))
        decK[i:i + B] = EV.pred_of(m.dec_logits(Sl[d], ch, dev))
        for t in range(2, d + 1):
            o = orc[t]; k = t - 1
            def run_pass(Sx):
                S2, h2, p2 = m.pass_aw(Sx, xs[t - 1], msk, hls[t - 1]); return m.colon_logits(h2, p2)[:, num].argmax(-1).cpu().numpy()
            o["self"][i:i + B] = run_pass(Sl[k])
            ids1, mk1 = pad([reroot_ids(r_, k, k) for r_ in rw]); x1 = torch.tensor(ids1, device=dev); m1 = torch.tensor(mk1, device=dev); hl1 = m.lower(x1)
            Sr, _, _ = m.pass_aw(mem.zero_state(B, dev), x1, m1, hl1); o["o1"][i:i + B] = run_pass(Sr)
            Sh = Sr
            for _ in range(k - 1):
                Sh, _, _ = m.pass_aw(Sh, x1, m1, hl1)
            o["oh"][i:i + B] = run_pass(Sh)
            Sy, _, _ = m.pass_aw(Sl[k - 1], x1, m1, hl1); o["hyb"][i:i + B] = run_pass(Sy)          # hybrid: passes 1..k-1 original, pass k on the re-rooted input (pass-index-matched key)
            ids2, mk2 = pad([reroot_ids(r_, k, t) for r_ in rw]); x2 = torch.tensor(ids2, device=dev); m2 = torch.tensor(mk2, device=dev)
            _, h3, p3 = m.pass_aw(mem.zero_state(B, dev), x2, m2); o["text_only"][i:i + B] = m.colon_logits(h3, p3)[:, num].argmax(-1).cpu().numpy()
        print("d", d, "batch", i, "/", n, flush=True)
    assert chk <= 1e-5, ("read_vec != read_fixed", chk)
    us = sorted(set(sigs)); perm = np.random.default_rng(0).permutation(len(us)); rank = {us[j]: r for r, j in enumerate(perm)}; r = np.array([rank[s_] for s_ in sigs]) / len(us)
    tr, va, te = [np.where(c)[0] for c in (r < 0.6, (r >= 0.6) & (r < 0.8), r >= 0.8)]
    y = np.array([it["ans"] for it in T]); res = {"d": d, "n": n, "seed": SEED, "split": [len(tr), len(va), len(te)], "read_vec_check_rel": chk, "dec_acc_at_Kprime_d": float((decK == y).mean()),
                                                  "decoder_answer_eq_v_d": float((decK == V[:, d]).mean())}
    pa = {}
    for t in range(1, d + 1):
        a = arg[:, t - 1]; vt = V[:, t]; vm = V[:, t - 1]; vp = V[:, t + 1] if t < d else np.full(n, -1)
        strict = (vt != vm) & (vt != vp)
        rec = {"eq_vt": float((a == vt).mean()), "eq_vtm1": float((a == vm).mean()), "eq_vtp1": float((a == vp).mean()) if t < d else None,
               "none_of_three": float(((a != vt) & (a != vm) & (a != vp)).mean()), "eq_vtm1_not_vt": float(((a == vm) & (a != vt)).mean()),
               "n_strict": int(strict.sum()), "strict_eq_vt": float((a[strict] == vt[strict]).mean()), "strict_eq_vtm1": float((a[strict] == vm[strict]).mean()),
               "strict_eq_vtp1": float((a[strict] == vp[strict]).mean()) if t < d else None, "frac_vt_eq_vtm1": float((vt == vm).mean()),
               "majority_vt": float(np.bincount(vt, minlength=P).max() / n)}
        if t >= 2:
            rec["eq_prev_pass_argmax"] = float((a == arg[:, t - 2]).mean())
        pa[str(t)] = rec
    res["a_argmax"] = pa
    res["a_pass_argmax_per_item"] = arg.tolist(); res["v_per_item"] = V.tolist(); res["sig"] = sigs; res["test_idx"] = te.tolist()
    pr = {}
    for t in range(1, d + 1):
        Ut = torch.cat(U[t]); Ht = torch.cat(H6[t]); St_ = torch.cat(SF[t]); row = {}
        for nm, X in (("u_write_src", Ut), ("h6_preread", Ht)):
            row[nm + "|v_t"] = fit_probe(X, V[:, t], tr, va, te); row[nm + "|v_t-1"] = fit_probe(X, V[:, t - 1], tr, va, te)
        row["S_t|v_t"] = fit_probe(St_, V[:, t], tr, va, te); row["S_t|v_t-1"] = fit_probe(St_, V[:, t - 1], tr, va, te)
        if t >= 2:
            row["read_at_colon|v_t-1"] = fit_probe(torch.cat(RD[t]), V[:, t - 1], tr, va, te); row["read_at_colon|v_t"] = fit_probe(torch.cat(RD[t]), V[:, t], tr, va, te)
            row["S_t-1|v_t-1"] = fit_probe(torch.cat(SF[t - 1]), V[:, t - 1], tr, va, te)   # = the state pass t reads
        pr[str(t)] = row
        print("d", d, "t", t, {k: round(v["test_acc"], 3) for k, v in row.items()}, flush=True)
        del Ut, Ht, St_
    res["bc_probes"] = pr
    res["d_oracle"] = {str(t): {nm: {"acc_vt": float((a == V[:, t]).mean()), "eq_vtm1": float((a == V[:, t - 1]).mean())} for nm, a in orc[t].items()} for t in range(2, d + 1)}
    # oracle conditional on the original chain being right / wrong at the preceding pass
    res["d_oracle_by_prev_ok"] = {str(t): {nm: {"prev_pass_argmax_correct": float((a[arg[:, t - 2] == V[:, t - 1]] == V[arg[:, t - 2] == V[:, t - 1], t]).mean()),
                                                "prev_pass_argmax_wrong": float((a[arg[:, t - 2] != V[:, t - 1]] == V[arg[:, t - 2] != V[:, t - 1], t]).mean()),
                                                "n_prev_correct": int((arg[:, t - 2] == V[:, t - 1]).sum())} for nm, a in orc[t].items()} for t in range(2, d + 1)}
    res["d_oracle_per_item"] = {str(t): {nm: a.tolist() for nm, a in orc[t].items()} for t in range(2, d + 1)}
    res["sec"] = round(time.time() - t0); return res


@torch.no_grad()
def main():
    t0 = time.time(); sets = IO.build_sets(); a, m, g = EV.load(RUN, dev); m.eval(); assert float(m.mem.gamma) == 1.0
    out = {"run": "aw_plaw4_s1", "device": dev, "dtype": "fp32", "think_k_trained": a["think_k"], "K_used": "d passes", "by_d": {}}
    for d in (3, 4):
        out["by_d"][str(d)] = run_d(d, m, g, sets)
    out["sec"] = round(time.time() - t0)
    out["notes"] = ("a: categories overlap where values coincide ('strict' removes that); u_write_src = mem.ln_w(h) at ':' of pass t (top residual, what write_seq consumes); h6_preread = "
                    "residual entering block 6 at ':'; read_at_colon = EV.read_vec(h6 at ':', S_{t-1}) (pre-W_o [oa||od], checked equal to read_fixed's capture); S_* = flat state (12x64x64); "
                    "oracle: hyb = original passes 1..k-1 then pass k on the re-rooted item (keeps the pass-k write slot); o1 = state after ONE pass on the re-rooted item (S_{t-1} given in text), oh = t-1 identical passes on it; text_only = no state; self = control")
    os.makedirs(os.path.dirname(OUT), exist_ok=True); open(OUT, "w", newline="\n").write(json.dumps(out)); print("WROTE", OUT, out["sec"], "s")


main()
