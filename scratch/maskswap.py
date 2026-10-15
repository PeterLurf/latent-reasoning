"""THEORY notes 203-5 / 203-5a (CONSUMED-MASK) and 203-6 / 203-6a (INPUT-SWAP), $0 eval-only on the 16 D7 final models; a NEW script (boxes/*.py read
only). One card (3070), bf16 autocast trunk / fp32 memory branch, the original eval bs (250); card / dtype / bs recorded.
Per-pass inputs: the think pass t reads its own token tensor x_t (same length and padding mask as the clean input), so the memory writes see the
manipulated text; the decoder reads [EOT, tokens, " ="] through fwd_read (nested170 InjectN.dec_logits on explicit token ids).
CONSUMED-MASK (203-5): reductions in nested170.reductions' order; reduction j's SPAN = the tokens its subtree renders to in the original expression,
  including brackets that vanish when it is reduced (gate SPAN: replacing every maximal span consumed by reduction t with its value token reproduces
  aux_steps' e_t symbols exactly, on 10k generator items). At think pass t >= 2 the union of spans reduced at passes 1..t-1 is replaced by the filler
  " ?" (asserted one GPT-2 token, outside VOCAB). Arms: M-CONS; M-UNCONS (the same count per pass, from tokens outside every consumed span, ranked by
  the reduction index of the smallest subtree owning the token - latest first - ties shuffled by rng 20391); M-CONS-ANS (M-CONS + the decoder's
  input masked with the union consumed by pass K). Items: test n 3-8 (all <= 1,000 per n, permuted by rng 20381) the model solves clean at K' = K.
  Gate PLAIN: the per-pass runner with unmasked inputs == the standard clean answers, bitwise.
INPUT-SWAP (203-6 / 6a): k in {1, 2}; recipients = test n 3-8 (rng 20401 order) with height(root) >= 2k + 1 and a late leaf (the root's bare child,
  else the leaf of the latest reduction > k outside the reduced subtrees); donor = the recipient's tree with each maximal reduced-at-<=k subtree
  re-expanded (Alg. 1) at a new value (up to 10 tries until every node reduced at <= k differs) and the late leaf resampled; same shape -> same
  length (asserted). mirror item = the donor with those subtrees replaced by the recipient's (so y_mir = F(donor with nodes <= k at the recipient's
  values); every unreduced leaf, incl. the late leaf, is the donor's). Kept: y_mir not in {y_d, y_r}; div-by-0 -> retry / drop. Passes 1..k on
  x_r, passes k+1..K on x_d; decoder x_d (primary) or x_r (INPUT-SWAP-THINK). clean-ccc = x_r, x_d, mirror solved clean. Shares of y_mir
  (STATE-FOLLOW) / y_d (INPUT-FOLLOW) / y_r, overall and by n; floor = K' = 0 accuracy on x_d on the same ccc pairs. Gates: x_d = x_r -> answers ==
  clean bitwise; MIRROR: eval_tokens(render(mirror)) == y_mir and the shape match, on 10k generator items.
-> <run>/eval_consmask.json, <run>/eval_inputswap.json
    python scratch/maskswap.py <run> --device cuda:1 [--stage consmask|inputswap|both] [--bs 250]"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scratch")); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.halt173 as H  # noqa: E402
import boxes.nested170 as N  # noqa: E402
from boxes.deltas5 import fwd_read  # noqa: E402
import dasused as DU  # noqa: E402  (red_paths, at, reexpand, late_leaf, height, Runner)

FILL = N.ENC.encode(" ?"); assert len(FILL) == 1 and " ?" not in N.VOCAB and FILL[0] not in N.TID.values(), "filler must be one token outside VOCAB"
FILL = FILL[0]; EQ, DOT = N.TID[" ="], N.TID[" ."]


def render_spans(nd, spans, parent=None, right=False, off=0):
    """nested170.render with each node's [start, end) symbol span (incl. the brackets it adds) recorded in spans[id(node)]."""
    if nd.op is None:
        spans[id(nd)] = (off, off + 1); return [" %d" % nd.v]
    br = parent is not None and (N.PREC[nd.op] < N.PREC[parent] or (right and N.PREC[nd.op] == N.PREC[parent])); o0 = off + (1 if br else 0)
    L = render_spans(nd.l, spans, nd.op, False, o0); s = L + [" " + nd.op]; Rr = render_spans(nd.r, spans, nd.op, True, o0 + len(L) + 1); s = s + Rr
    if br:
        s = [" ("] + s + [" )"]
    spans[id(nd)] = (off, off + len(s)); return s


def red_spans(root):
    """(symbols, [span of reduction j's node, j = 1..n], [owner reduction index per symbol]) on the ORIGINAL tree."""
    sp = {}; syms = render_spans(root, sp); assert syms == N.render(root); paths = DU.red_paths(root); spans = [sp[id(DU.at(root, p))] for p in paths]
    owner = np.full(len(syms), 0)
    for j, (a, b) in enumerate(spans, 1):                                 # the smallest owning subtree = the earliest reduction whose span contains it
        for i in range(a, b):
            if owner[i] == 0:
                owner[i] = j
    return syms, spans, owner


def consumed_mask(spans, upto):
    m = np.zeros(spans[-1][1] if spans else 0, bool)
    for a, b in spans[:upto]:
        m[a:b] = True
    return m


def gate_spans(n_items=10000, seed=20380):
    rng = np.random.default_rng(seed); bad = 0
    for _ in range(n_items):
        r, _ = N.generate(rng, int(rng.integers(2, 9))); syms, spans, _ = red_spans(r); vals = N.reductions(r); aux = N.aux_steps(r)
        for t in range(1, len(spans)):
            mx = [j for j in range(t) if not any(spans[i][0] <= spans[j][0] and spans[j][1] <= spans[i][1] and i != j for i in range(t))]   # maximal spans
            s2 = list(syms)
            for j in sorted(mx, key=lambda j: -spans[j][0]):
                a, b = spans[j]; s2[a:b] = [" %d" % vals[j]]
            bad += s2 != aux[t - 1][0]
    return {"n_items": n_items, "mismatch": int(bad), "pass": bad == 0}


class PassRunner:
    def __init__(self, m, dev, bs):
        self.m, self.dev, self.bs = m, dev, bs; R_ = DU.Runner(m, dev, bs); self.ac, self.amp = R_.ac, R_.amp; self.R_ = R_

    @torch.no_grad()
    def run(self, toks_fn, dec_fn, n_items, K):
        """toks_fn(i, t) -> the think-token list (without EOT / ' .') of item i at pass t; dec_fn(i) -> the decoder's token list; K passes."""
        out = []
        for i0 in range(0, n_items, self.bs):
            idx = list(range(i0, min(i0 + self.bs, n_items))); S = None; m = self.m
            with self.ac():
                for t in range(0, K + 1):
                    if t == 0:
                        S = m.mem.zero_state(len(idx), self.dev); continue
                    ids_, msk = N.pad([[N.EOT] + toks_fn(i, t) + [DOT] for i in idx]); x = torch.tensor(ids_, device=self.dev); mk = torch.tensor(msk, device=self.dev)
                    S = m.write(S, x, mk)
                seqs = [[N.EOT] + dec_fn(i) + [EQ] for i in idx]; ids_, _ = N.pad(seqs); x = torch.tensor(ids_, device=self.dev)
                h = fwd_read(m.g, m.g.embed(x), m.mem, S); pos = torch.tensor([len(q) - 1 for q in seqs], device=self.dev)
                lg = m.g.logits(h[torch.arange(len(seqs), device=self.dev), pos]).float()
            out.append(lg[:, N.NUM_IDS].argmax(-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=int)


def acc_block(ok, nn):
    return {"acc": float(ok.mean()) if len(ok) else None, "n": int(len(ok)), "by_n": {str(n_): {"acc": float(ok[nn == n_].mean()), "n": int((nn == n_).sum())} for n_ in sorted(set(nn.tolist()))}}


def consmask(P, TE, K):
    its = [it for it in TE if 3 <= it["n"] <= 8]; its = [its[i] for i in np.random.default_rng(20381).permutation(len(its))]
    y = np.array([N.value(it["root"]) for it in its]); clean = P.R_.answers(its, K)
    toks = [N.ids(it["syms"]) for it in its]; plain = P.run(lambda i, t: toks[i], lambda i: toks[i], len(its), K)
    gate_plain = {"equal": int((plain == clean).sum()), "n": len(its), "pass": bool((plain == clean).all())}
    keep = np.where(clean == y)[0]; its = [its[i] for i in keep]; y = y[keep]; nn = np.array([it["n"] for it in its]); toks = [toks[i] for i in keep]
    info = [red_spans(it["root"]) for it in its]; rng = np.random.default_rng(20391); unc_order = []
    for syms, spans, owner in info:
        pr = rng.permutation(len(syms)); unc_order.append(sorted(range(len(syms)), key=lambda i: (-owner[i], int(np.where(pr == i)[0][0]))))
    cons = [[consumed_mask(sp, t - 1) for t in range(1, K + 1)] for _, sp, _ in info]   # cons[i][t-1]: consumed by pass t-1 (t = 1: none)
    short = [0]

    def mask_cons(i, t):
        mk = cons[i][min(t, len(cons[i])) - 1] if t >= 2 else None
        return toks[i] if mk is None or not mk.any() else [FILL if mk[j] else w for j, w in enumerate(toks[i])]

    def mask_unc(i, t):
        if t < 2:
            return toks[i]
        mk = cons[i][min(t, len(cons[i])) - 1]; c = int(mk.sum())
        if c == 0:
            return toks[i]
        av = [j for j in unc_order[i] if not mk[j]]; pick = set(av[:c]); short[0] += max(0, c - len(av))
        return [FILL if j in pick else w for j, w in enumerate(toks[i])]
    final = [cons[i][-1] if len(cons[i]) else np.zeros(len(toks[i]), bool) for i in range(len(its))]
    full = [consumed_mask(info[i][1], len(info[i][1])) for i in range(len(its))]        # all reductions -> the whole expression; use pass K's union
    dec_cons = lambda i: [FILL if cons[i][min(K, len(cons[i])) - 1][j] else w for j, w in enumerate(toks[i])] if K >= 2 else toks[i]  # noqa: E731
    a_cons = P.run(mask_cons, lambda i: toks[i], len(its), K); a_unc = P.run(mask_unc, lambda i: toks[i], len(its), K); a_ans = P.run(mask_cons, dec_cons, len(its), K)
    out = {"n_clean_items": len(its), "clean_acc_pool": float(len(keep) / len(clean)), "M-CONS": acc_block(a_cons == y, nn), "M-UNCONS": acc_block(a_unc == y, nn),
           "M-CONS-ANS": acc_block(a_ans == y, nn), "uncons_shortfall_tokens": int(short[0]), "gates": {"PLAIN": gate_plain, "FILLER": {"token": int(FILL), "text": " ?", "single": True, "outside_vocab": True}},
           "masked_tokens_mean_by_pass": {str(t): float(np.mean([cons[i][min(t, len(cons[i])) - 1].sum() if t >= 2 else 0 for i in range(len(its))])) for t in range(1, K + 1)},
           "readable": len(its) >= 100, "items": "test n 3-8, rng 20381 permutation, clean-correct at K' = K"}
    del final, full
    return out


def mirror_pair(it, k, rng, tries=10):
    root = it["root"]; paths = DU.red_paths(root)
    if len(paths) < k + 1 or DU.height(root) < 2 * k + 1:
        return None, "ineligible"
    ll = DU.late_leaf(root, paths, k)
    if ll is None:
        return None, "no_late_leaf"
    red = paths[:k]; mx = [p for p in red if not any(q != p and p[:len(q)] == q for q in red)]   # maximal reduced subtrees
    for _ in range(tries):
        d = N._copy(root)
        try:
            for p in mx:
                nd = DU.at(d, p); v0 = N.value(DU.at(root, p)); DU.reexpand(nd, int(rng.choice([x for x in range(N.P) if x != v0])), rng)
            if any(N.value(DU.at(d, p)) == N.value(DU.at(root, p)) for p in red):
                continue
            lf = DU.at(d, ll); lf.v = int(rng.choice([x for x in range(N.P) if x != lf.v]))
            mir = N._copy(d)
            for p in mx:
                setattr(DU.at(mir, p[:-1]), p[-1], N._copy(DU.at(root, p)))
            yd, yr, ym = N.value(d), N.value(root), N.value(mir)
        except KeyError:
            continue
        if ym in (yd, yr):
            return None, "mir_not_distinct"
        dd = {"syms": N.render(d), "t": yd, "root": d, "n": it["n"]}; mm = {"syms": N.render(mir), "t": ym, "root": mir, "n": it["n"]}
        assert len(dd["syms"]) == len(it["syms"]), "structure match: equal length"
        return (dd, mm), "ok"
    return None, "tries"


def gate_mirror(n_items=10000, seed=20400):
    rng = np.random.default_rng(seed); bad = n = 0
    for _ in range(n_items):
        r, t = N.generate(rng, int(rng.integers(3, 9))); it = {"root": r, "syms": N.render(r), "n": 0, "t": t}
        for k in (1, 2):
            pr, why = mirror_pair(it, k, rng)
            if pr is None:
                continue
            n += 1; d, mm = pr; ops = lambda s: [w for w in s if w.strip() in N.OPS or w in (" (", " )")]  # noqa: E731
            bad += N.eval_tokens(mm["syms"]) != mm["t"] or ops(d["syms"]) != ops(it["syms"]) or N.eval_tokens(d["syms"]) != d["t"]
    return {"n_pairs": n, "mismatch": int(bad), "pass": bad == 0}


def inputswap(P, TE, K):
    its = [it for it in TE if 3 <= it["n"] <= 8]; its = [its[i] for i in np.random.default_rng(20401).permutation(len(its))]; out = {}
    toks0 = [N.ids(it["syms"]) for it in its]; clean_all = P.R_.answers(its, K)
    same = P.run(lambda i, t: toks0[i], lambda i: toks0[i], len(its), K); out["gate_SAME"] = {"equal": int((same == clean_all).sum()), "n": len(its), "pass": bool((same == clean_all).all())}
    for k in (1, 2):
        rng = np.random.default_rng([20401, k]); R, cnt = [], {}
        for it in its:
            pr, why = mirror_pair(it, k, rng); cnt[why] = cnt.get(why, 0) + 1
            if pr is not None:
                R.append((it, pr[0], pr[1]))
        rec = [r[0] for r in R]; don = [r[1] for r in R]; mir = [r[2] for r in R]; nn = np.array([it["n"] for it in rec])
        yr = np.array([N.value(it["root"]) for it in rec]); yd = np.array([d["t"] for d in don]); ym = np.array([mm["t"] for mm in mir])
        cr, cd, cm = P.R_.answers(rec, K), P.R_.answers(don, K), P.R_.answers(mir, K); ccc = (cr == yr) & (cd == yd) & (cm == ym)
        tr = [N.ids(it["syms"]) for it in rec]; td = [N.ids(d["syms"]) for d in don]
        prim = P.run(lambda i, t: tr[i] if t <= k else td[i], lambda i: td[i], len(rec), K); thk = P.run(lambda i, t: tr[i] if t <= k else td[i], lambda i: tr[i], len(rec), K)
        floor = P.R_.answers(don, 0)

        def sh(got, mk):
            n_ = int(mk.sum())
            return {"n": n_, "y_mir": float((got == ym)[mk].mean()) if n_ else None, "y_d": float((got == yd)[mk].mean()) if n_ else None,
                    "y_r": float((got == yr)[mk].mean()) if n_ else None}
        blk = {"n_pairs": len(R), "drops": cnt, "n_ccc": int(ccc.sum()), "readable": int(ccc.sum()) >= 100,
               "clean_acc": {"x_r": float((cr == yr).mean()), "x_d": float((cd == yd).mean()), "mirror": float((cm == ym).mean())},
               "primary": {**sh(prim, ccc), "all": sh(prim, np.ones(len(rec), bool)), "by_n": {str(n_): sh(prim, ccc & (nn == n_)) for n_ in sorted(set(nn.tolist()))}},
               "think_only": {**sh(thk, ccc), "by_n": {str(n_): sh(thk, ccc & (nn == n_)) for n_ in sorted(set(nn.tolist()))}},
               "floor_kprime0_x_d": float((floor == yd)[ccc].mean()) if ccc.sum() else None}
        out[str(k)] = blk; print("INPUT-SWAP k", k, {kk: blk[kk] for kk in ("n_pairs", "n_ccc", "floor_kprime0_x_d")}, {kk: blk["primary"][kk] for kk in ("y_mir", "y_d", "y_r")}, flush=True)
    out["items"] = "test n 3-8, rng 20401 permutation; donors rng [20401, k]"
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--device", default="cuda:1"); ap.add_argument("--stage", default="both", choices=["consmask", "inputswap", "both"])
    ap.add_argument("--bs", type=int, default=250); a = ap.parse_args(); rd = os.path.join(ROOT, "boxes", "runs_fsq", a.run)
    kind, m, ra = H.load_model(rd, a.device); assert kind == "nested"; DM.FAST_ADD = bool(ra.get("fast_add")); DM.CHUNK_DELTA = 0; DM.READ_CLAMP = None; K = int(ra["think_k"])
    P = PassRunner(m, a.device, a.bs); TE = N.test_set(int(ra.get("per_n") or 1000))
    meta = {"card": torch.cuda.get_device_name(a.device), "dtype": "bf16 autocast trunk, fp32 memory branch" if P.amp is not None else "fp32", "bs": a.bs, "K": K,
            "torch": torch.__version__, "script_git": DU.git_hash("scratch/maskswap.py")}
    if a.stage in ("consmask", "both"):
        t0 = time.time(); g = gate_spans(); r = consmask(P, TE, K); r["gates"]["SPAN"] = g
        json.dump({**r, **meta, "sec": round(time.time() - t0), "def": "THEORY note 203-5 / 203-5a; see scratch/maskswap.py"}, open(os.path.join(rd, "eval_consmask.json"), "w", newline="\n"), indent=1)
        print(a.run, "CONSMASK", {k_: r[k_]["acc"] for k_ in ("M-CONS", "M-UNCONS", "M-CONS-ANS")}, "n", r["n_clean_items"], "gates", r["gates"]["PLAIN"]["pass"], g["pass"], flush=True)
    if a.stage in ("inputswap", "both"):
        t0 = time.time(); g = gate_mirror(); r = inputswap(P, TE, K); r["gate_MIRROR"] = g
        json.dump({**r, **meta, "sec": round(time.time() - t0), "def": "THEORY note 203-6 / 203-6a; see scratch/maskswap.py"}, open(os.path.join(rd, "eval_inputswap.json"), "w", newline="\n"), indent=1)
        print(a.run, "INPUTSWAP gates", r["gate_SAME"]["pass"], g, flush=True)


if __name__ == "__main__":
    main()
