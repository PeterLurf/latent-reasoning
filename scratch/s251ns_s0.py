"""Note 251-15 UNSCAFFOLDED $0 stage (REVIEW_NS; eval only) on the 4060 (cuda:0), fp32, eval_bs --bs (one card / bs / dtype for every mode) -> scratch/s251ns_s0/<part>.json
Models rcb_plaw4_s1 / s2; REG TEST2 d 1-4 (1,000 each). Each eval part carries the named-cue, chain-order baseline (TEST2 as built) from the same call.
  gate    : (g1) re-render: NW._finish with the item's own fields and ORIGINAL order reproduces text / R / R_step / question exactly (every item, d 1-4);
            (g2) TEST-PERM: for d >= 2 the op sentences S_1..S_d are reassigned over their own slots by a seeded non-identity permutation (>= 1 inversion; S_0, givens
            G_t and distractors D_k keep their slots; d 1 has one op sentence -> unchanged), OW.solve(permuted text) == answer on every item, values / answer / names
            unchanged, the multiset of sentences unchanged; (g3) GENERIC-0: every think cue = [EOT] + enc(' ' + text + ' What is the next value?') + SFX, the text rows
            are the run's own think rows, the cue is 6 tokens (+ SFX), eval-only (inject_st's CUE_LEN untouched); (g4) the probe feature path: read_vec on S_t at the
            decoder ':' == read_fixed's own capture (rel <= 1e-5).
  perm    : (c) acc at K' = d - 1 and 8 and WS per pass 1-8 on TEST-PERM vs TEST2 (named cues as trained).
  generic : (a) the same on TEST2 with GENERIC-0 think cues (decoder unchanged) vs the named cues.
  probe   : (b) d 4 TEST2 items: per pass t = 1..8 the feature = the pre-W_o read [oa || od] at the decoder ':' from S_t (part_probe's read1 representation, 768-d);
            linear probes (Audit 231-1 protocol: standardise on train, LBFGS, wd in 1e-4 / 1e-3 / 1e-2 by val, splits by item sig 60 / 20 / 20) for the pass index t
            (8 classes; control: ||S_t|| alone), the spine identity of step t (the name of S_t, t <= 4; 40 names) and of S_min(t, 4) over t = 1..8. Descriptive.
    python -B scratch/s251ns_s0.py gate perm generic probe [--bs 100 --device cuda:0]"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
from boxes import inject_aw_eval as E  # noqa: E402
from boxes import inject_aw as AW  # noqa: E402
from boxes import inject_ow as IO  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
import boxes.ow_words as OW  # noqa: E402
import boxes.nested_words as NW  # noqa: E402
from boxes.nested170 import EOT, pad  # noqa: E402

OUT = ROOT + "/scratch/s251ns_s0"; K = 8; DS = (1, 2, 3, 4); SEED_PERM = 25150; GEN = " What is the next value?"; RUNS = ("rcb_plaw4_s1", "rcb_plaw4_s2")


def render(it, order):
    nm = it["names"]
    o = NW._finish(it["d"], it["ops"], it["operands"], it["src_j"], it["values"], it["mul_s_t"], nm["spine"], nm["givens"], nm["distractors"], it["unit"],
                   it["frames"], it["distractor_values"], order, it["n_distractors"])
    return OW._decorate(o, it["order_mode"], None, it.get("dag", True))


def permuted(it, rng):
    """the op sentences S_1..S_d reassigned over their own slots by a non-identity permutation (d >= 2); everything else keeps its slot."""
    d = it["d"]; order = list(it["order"])
    if d < 2:
        return render(it, order), order, 0
    slots = [i for i, x in enumerate(order) if x in {"S%d" % t for t in range(1, d + 1)}]; assert [order[i] for i in slots] == ["S%d" % t for t in range(1, d + 1)], "REG chain order"
    while True:
        p = rng.permutation(d)
        if (p != np.arange(d)).any():
            break
    new = list(order)
    for k, i in enumerate(slots):
        new[i] = "S%d" % (p[k] + 1)
    inv = sum(int(p[i] > p[j]) for i in range(d) for j in range(i + 1, d)); return render(it, new), new, inv


def perm_sets(raw):
    out = {}; info = {}
    for d in DS:
        rng = np.random.default_rng(SEED_PERM + d); its = []; invs = []
        for it in raw[d]:
            pit, order, inv = permuted(it, rng)
            for k_ in ("values", "answer", "names", "ops", "operands", "d", "question"):
                assert pit[k_] == it[k_], ("TEST-PERM changed", k_)
            assert sorted(pit["text"].split(". ")) == sorted(it["text"].split(". ")) or d == 1, "TEST-PERM: the same sentences"
            assert OW.solve(pit["text"]) == it["answer"], "TEST-PERM: OW.solve != answer"
            pit["sig"] = it["sig"] + ":perm"; its.append(pit); invs.append(inv)
        out[d] = its; info[str(d)] = {"n": len(its), "min_inversions": int(min(invs)), "mean_inversions": float(np.mean(invs)),
                                      "changed_text_share": float(np.mean([a_["text"] != b_["text"] for a_, b_ in zip(its, raw[d])]))}
    return out, info


def generic(T):
    """GENERIC-0 copies of tok_aw items: every think cue -> [EOT] + enc(' ' + text + GEN) + SFX (eval-only)."""
    gt = IO.ENC.encode(GEN); assert len(gt) == 6, ("GENERIC-0 cue tokens", len(gt)); out = {}
    for d, its in T.items():
        new = []
        for it in its:
            q = [EOT] + IO.ENC.encode(" " + it["raw"]["text"] + GEN) + AW.SFX
            assert q[:len(it["think"])] == it["think"] and q[len(it["think"]):] == gt + AW.SFX, "GENERIC-0: text rows / cue split"
            c = dict(it); c["think_cues"] = [q] * len(it["think_cues"]); new.append(c)
        out[d] = new
    return out


def run_eval(m, T, dev, bs):
    out = {}
    for d in DS:
        its = T[d]; y = np.array([it["ans"] for it in its]); ks = sorted({d - 1, K})
        A, _ = E.passes(m, its, dev, bs, set(ks)); r_, pv = AW.ws_readout(m, {d: its}, dev, K, bs, per_item=True)
        out[str(d)] = {"n": len(its), "acc_kprime": {str(k): float((A[k] == y).mean()) for k in ks}, "ws": r_[str(d)], "_ans": {str(k): A[k].tolist() for k in ks}, "_ws": pv[str(d)]}
    return out


def base(run, dev):
    a, m, g = E.load(ROOT + "/boxes/runs_fsq/" + run, dev); sets = IO.build_sets(mul_s_p=E.law(a)); raw = {d: sets["test2"][d] for d in DS}
    return a, m, g, raw, E.tokd({"test2": raw}, a, DS)


def gate(dev, bs):
    a, m, g, raw, T = base(RUNS[0], dev); o = {}
    o["g1_rerender_exact"] = all(render(it, list(it["order"]))[k_] == it[k_] for d in DS for it in raw[d] for k_ in ("text", "R", "R_step", "question"))
    try:
        P_, info = perm_sets(raw); o["g2_perm"] = info; o["g2_ok"] = all(info[str(d)]["min_inversions"] >= 1 for d in (2, 3, 4)) and info["1"]["changed_text_share"] == 0.0
    except AssertionError as e:
        o["g2_perm"] = repr(e); o["g2_ok"] = False
    try:
        generic(T); o["g3_ok"] = True
    except AssertionError as e:
        o["g3_ok"] = repr(e)
    with torch.no_grad():
        ch = T[4][:bs]; B = len(ch); ar = torch.arange(B, device=dev); xs, msk = AW.cue_inputs(m, ch, dev); S = m.mem.zero_state(B, dev)
        for t in range(1, 4):
            S = m.write(S, xs[t - 1], msk, None)
        ids_, _ = pad([it["dec"] for it in ch]); xd = torch.tensor(ids_, device=dev); dp = torch.tensor([len(it["dec"]) - 1 for it in ch], device=dev)
        hl = DM.gpt_lower(g, g.embed(xd), m.mem.layer)[ar, dp]; rv = E.read_vec(m.mem, hl, S); rc_ = []; DM.read_fixed(m.mem, hl[:, None], S, rc_)
        H_ADD = m.mem.h_add; o["g4_rel"] = float((rc_[0] - rv[:, :H_ADD * DM.DH]).abs().max() / rc_[0].abs().max().clamp_min(1e-30))
    o["pass"] = bool(o["g1_rerender_exact"] and o["g2_ok"] and o["g3_ok"] is True and o["g4_rel"] <= 1e-5); return o


def perm(dev, bs):
    o = {}
    for run in RUNS:
        a, m, g, raw, T = base(run, dev); P_, info = perm_sets(raw); TP = E.tokd({"test2": P_}, a, DS)
        with torch.no_grad():
            o[run] = {"test2": run_eval(m, T, dev, bs), "test_perm": run_eval(m, TP, dev, bs), "perm_info": info, "bs": bs, "dtype": "fp32", "gpu": torch.cuda.get_device_name(0),
                      "perm_items": {str(d): {"sig": [it["sig"] for it in P_[d]], "order": [it["order"] for it in P_[d]]} for d in DS}}
        del m, g; torch.cuda.empty_cache()
    o["def"] = ("Note 251-15 (c) TEST-PERM: REG TEST2 d 1-4 re-rendered (NW._finish) with the op sentences S_1..S_d reassigned over their slots by a seeded non-identity "
                "permutation (default_rng(25150 + d)); OW.solve == answer asserted; named cues as trained; acc_kprime {d-1, 8}, WS per pass; test2 = the same call's baseline")
    return o


def generic_part(dev, bs):
    o = {}
    for run in RUNS:
        a, m, g, raw, T = base(run, dev); TG = generic(T)
        with torch.no_grad():
            o[run] = {"named": run_eval(m, T, dev, bs), "generic0": run_eval(m, TG, dev, bs), "bs": bs, "dtype": "fp32", "gpu": torch.cuda.get_device_name(0)}
        del m, g; torch.cuda.empty_cache()
    o["def"] = "Note 251-15 (a) GENERIC-0: every think cue ' What is the next value? The answer is:' (decoder unchanged), REG TEST2 d 1-4; named = the same call's baseline"
    return o


def fit_probe_c(X, y, tr, va, te, C):
    """inject_aw_eval.fit_probe (Audit's 231-1 protocol) with C classes (fit_probe fixes 11), CPU."""
    import torch.nn.functional as Fn
    mu = X[tr].mean(0, keepdim=True); sd = X[tr].std(0, keepdim=True) + 1e-6; Z = (X - mu) / sd; Y = torch.tensor(y); best = None
    for wd in (1e-4, 1e-3, 1e-2):
        W = torch.zeros(Z.shape[1], C, requires_grad=True); b = torch.zeros(C, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=200, history_size=20, line_search_fn="strong_wolfe")

        def clo():
            opt.zero_grad(); l_ = Fn.cross_entropy(Z[tr] @ W + b, Y[tr]) + 0.5 * wd * (W ** 2).sum(); l_.backward(); return l_
        with torch.enable_grad():
            opt.step(clo)
        with torch.no_grad():
            pr = (Z @ W + b).argmax(-1); va_acc = float((pr[va] == Y[va]).float().mean())
            if best is None or va_acc > best[0]:
                best = (va_acc, wd, pr[te].numpy(), float((pr[tr] == Y[tr]).float().mean()))
    yt = y[te]
    return {"test_acc": float((best[2] == yt).mean()), "val_acc": best[0], "train_acc": best[3], "wd": best[1], "n_test": int(len(te)), "n_classes": C,
            "chance_majority": float(np.bincount(yt, minlength=C).max() / len(yt)), "feature_dim": int(X.shape[1])}


def probe(dev, bs):
    o = {}
    for run in RUNS:
        a, m, g, raw, T = base(run, dev); T4 = T[4]; feats = {t: [] for t in range(1, K + 1)}; norms = {t: [] for t in range(1, K + 1)}
        with torch.no_grad():
            for i in range(0, len(T4), bs):
                ch = T4[i:i + bs]; B = len(ch); ar = torch.arange(B, device=dev); xs, msk = AW.cue_inputs(m, ch, dev); hls = [m.lower(x_) for x_ in xs]; S = m.mem.zero_state(B, dev)
                ids_, _ = pad([it["dec"] for it in ch]); xd = torch.tensor(ids_, device=dev); dp = torch.tensor([len(it["dec"]) - 1 for it in ch], device=dev)
                hl = DM.gpt_lower(g, g.embed(xd), m.mem.layer)[ar, dp]
                for t in range(1, K + 1):
                    j = min(t, len(xs)) - 1; S = m.write(S, xs[j], msk, hls[j])
                    feats[t].append(E.read_vec(m.mem, hl, S).float().cpu()); norms[t].append(S.float().flatten(1).norm(dim=1)[:, None].cpu())
        sigs = [it["sig"] for it in T4]; us = sorted(set(sigs)); pm = np.random.default_rng(0).permutation(len(us)); rank = {us[j]: k for k, j in enumerate(pm)}
        rr = np.array([rank[s_] for s_ in sigs]) / len(us); tr0, va0, te0 = [np.where(c)[0] for c in (rr < 0.6, (rr >= 0.6) & (rr < 0.8), rr >= 0.8)]
        N = len(T4); nidx = {n_: k for k, n_ in enumerate(OW.NAMES)}

        def stack(ts):
            X = torch.cat([torch.cat(feats[t]) for t in ts]); Xn = torch.cat([torch.cat(norms[t]) for t in ts])
            sp = lambda ix: np.concatenate([ix + N * k for k in range(len(ts))])  # noqa: E731
            return X, Xn, sp(tr0), sp(va0), sp(te0)
        X, Xn, tr, va, te = stack(range(1, K + 1)); yt = np.concatenate([np.full(N, t - 1) for t in range(1, K + 1)])
        ysp = np.concatenate([np.array([nidx[it["raw"]["names"]["spine"][min(t, 4)]] for it in T4]) for t in range(1, K + 1)])
        X4, _, tr4, va4, te4 = stack(range(1, 5)); ysp4 = np.concatenate([np.array([nidx[it["raw"]["names"]["spine"][t]] for it in T4]) for t in range(1, 5)])
        o[run] = {"pass_index": fit_probe_c(X, yt, tr, va, te, K), "pass_index_norm_only": fit_probe_c(Xn, yt, tr, va, te, K),
                  "spine_id_t_le_d": fit_probe_c(X4, ysp4, tr4, va4, te4, len(OW.NAMES)), "spine_id_min_t_d": fit_probe_c(X, ysp, tr, va, te, len(OW.NAMES)),
                  "n_items": N, "split_items": [len(tr0), len(va0), len(te0)]}
        del m, g; torch.cuda.empty_cache()
    o["def"] = ("Note 251-15 (b) PROBE (descriptive): REG TEST2 d 4 x 1,000, feature per pass t = the pre-W_o read [oa || od] at the decoder ':' from S_t (part_probe's read1), "
                "Audit 231-1 protocol (fit_probe with C classes), item-sig splits 60 / 20 / 20 (seed 0) shared by every pass; pass index (8 classes; control ||S_t|| only), "
                "spine identity = the name of S_t (t <= 4) / S_min(t, 4) (t = 1..8), 40 names")
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("parts", nargs="+"); ap.add_argument("--bs", type=int, default=100); ap.add_argument("--device", default="cuda:0"); o_ = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); fns = {"gate": gate, "perm": perm, "generic": generic_part, "probe": probe}
    for p_ in o_.parts:
        r = fns[p_](o_.device, o_.bs); json.dump(r, open(os.path.join(OUT, p_ + ".json"), "w", newline="\n"), indent=1, default=str)
        print(p_, ("pass " + str(r["pass"])) if p_ == "gate" else "written", flush=True)
