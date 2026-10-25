"""THEORY §233 STEP-TOLD evaluation (ST_v1 §3; REVIEW_ST fixes 3-5) for boxes/inject_st.py runs. fp32 (no autocast); every pass's lower blocks cached per
DISTINCT input (inject_st's rule); TEST2 (boxes.inject_ow.build_sets: 1,000 per d, frames 0-5, seed 22960). Every per-item output is stored.
Pass inputs (seqs_of_t): LAT-ST pass t = the item's S_min(t, d) cue input; CUED-DIRECT = one pass on dec (the cues are in dec). Parts (-> <run>/eval_<part>.json,
then eval.json = the summary of the parts present). ds = 2..d_max of the run (P-2: d 2).
  acc        TEST2 d 1..12: inject -> the answer at every K' 0..14 (one sweep); direct -> one pass. by_d[d] {acc (K' = think_k), k_main, acc_kprime}; items
             {sig, ans, pred, L (pass-1 think tokens), s (decoder suffix)}.
  lp5        MEAN-AT-1 (LP5, fix 4): on TEST2 d in ds: S after pass 1 := the TEST2-d mean of S_1 (pass 1 on its own cue), passes 2..K on their own cues;
             answers at K' = d (binding) and K' = K (descriptive) + the intact answers at the same K' from the same code path; per item.
  mean_s     pooled MEAN-STATE (binding at P-2): the decoder reads the TEST2 mean of S_K pooled over d 1..12 (and, descriptive, the per-d mean); per item.
  cue_only   CUE-ONLY (fix 3) on TEST2 d in ds: every pass's think input = [EOT] + enc(" " + cue_min(t, d)) alone (no item text); decoder unchanged
             (text + " The answer is:"); the answer at K' = K and the teacher-forced aux value-token acc at passes 1 and 2; per item.
  cue_wrong  CUE-WRONG (fix 5) on TEST2 d in ds: the cue at pass t names S_{t+1} (t < d), else S_0 - arm "all" (every pass wrong) and arm "at_<t>"
             (only pass t wrong, t = 1..d); answers at K' = d and K' = K; per item (the intact answers are in lp5's items).
    python -m boxes.inject_st_eval --run boxes/runs_fsq/<run> --parts acc,lp5,mean_s,cue_only,cue_wrong [--device cuda:0 --eval_bs 100]"""
import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from boxes import inject_ow as IO  # noqa: E402
from boxes import inject_st as ST  # noqa: E402
from boxes.nested170 import NUM_IDS, EOT, pad  # noqa: E402

KMAX = 14; NULLC = contextlib.nullcontext
PARTS = ("acc", "lp5", "mean_s", "cue_only", "cue_wrong")


def pred_of(lg):
    return lg[:, NUM_IDS].argmax(-1).cpu().numpy()


def seq_std(it, t):
    """the run's own pass-t input: LAT-ST -> the cue min(t, d) input; no cues -> the plain think input."""
    tc = it.get("think_cues")
    return tc[min(t, len(tc)) - 1] if tc else it["think"]


def n_distinct(items):
    return max(len(it.get("think_cues", [None])) for it in items)


@torch.no_grad()
def passes(m, items, dev, bs, kset, seqs_of_t=seq_std, Td=None, edit=None, aux_ts=(), on_state=None):
    """the cue-aware pass runner: per batch, the distinct pass inputs x_1..x_Td (pass t > Td reads x_Td), one lower cache each; S from zero; edit(t, S, sl)
    -> a replacement S after pass t (or None); answers at every K' in kset; aux value-token acc (teacher-forced) after the passes in aux_ts; on_state(t, S,
    sl) sees every S_t. -> ({K': answers (N,)}, {t: [0/1 per item]})."""
    out = {k: np.full(len(items), -1) for k in kset}; auxv = {t: np.full(len(items), -1) for t in aux_ts}; Td = Td or n_distinct(items); kmax = max(kset)
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]; sl = slice(i, i + len(ch)); xs = []; m0 = None
        for t in range(1, Td + 1):
            ids_, mk = pad([seqs_of_t(it, t) for it in ch]); assert m0 is None or (mk == m0).all(), "one padding across passes"; m0 = mk
            xs.append(torch.tensor(ids_, device=dev))
        msk = torch.tensor(m0, device=dev); hls = [m.lower(x) for x in xs]; S = m.mem.zero_state(len(ch), dev)
        if 0 in kset:
            out[0][sl] = pred_of(m.dec_logits(S, ch, dev))
        for t in range(1, kmax + 1):
            j = min(t, Td) - 1; S = m.write(S, xs[j], msk, hls[j])
            if on_state is not None:
                on_state(t, S, sl)
            if edit is not None:
                S2 = edit(t, S, sl); S = S if S2 is None else S2
            if t in aux_ts:
                lg_a, y_a = m.aux_logits(S, ch, t, dev); pr = lg_a.argmax(-1); vi = torch.tensor([it["vidx"][t - 1] for it in ch], device=dev); ar = torch.arange(len(ch), device=dev)
                auxv[t][sl] = (pr[ar, vi] == y_a[ar, vi]).long().cpu().numpy()
            if t in kset:
                out[t][sl] = pred_of(m.dec_logits(S, ch, dev))
    return out, auxv


@torch.no_grad()
def answers_by_kprime(m, g, items, kmax, dev, bs, ac=None):
    """inject_ow_eval.answers_by_kprime's interface (the memory check): (N, kmax + 1) answers (inject) or (N, 1) (direct), cue-aware."""
    if m is None:
        out = np.full((len(items), 1), -1)
        for i in range(0, len(items), bs):
            ch = items[i:i + bs]; out[i:i + len(ch), 0] = pred_of(IO.direct_logits(g, ch, dev, ac or NULLC))
        return out
    o, _ = passes(m, items, dev, bs, set(range(kmax + 1))); return np.stack([o[k] for k in range(kmax + 1)], 1)


def tokd(sets, a, ds, key="test2"):
    return {d: [ST.tok_st(it, a.get("aux_target", "value"), a.get("cue", "none")) for it in sets[key][d]] for d in ds}


def items_rec(T, extra):
    return {"sig": [it["sig"] for it in T], "ans": [it["ans"] for it in T], **extra}


@torch.no_grad()                                                            # 10-24 (b345 mean_s OOM: a bare dec_logits built an autograd graph)
def part_acc(m, g, sets, a, dev, bs):
    T = tokd(sets, a, IO.TEST_DS); K = a["think_k"]; out = {"by_d": {}, "items": {}}
    for d in IO.TEST_DS:
        P = answers_by_kprime(m, g, T[d], KMAX if m is not None else 0, dev, bs); y = np.array([it["ans"] for it in T[d]]); km = K if m is not None else 0
        rec = {"acc": float((P[:, km] == y).mean()), "k_main": km}
        if m is not None:
            rec["acc_kprime"] = {str(k): float((P[:, k] == y).mean()) for k in range(KMAX + 1)}
        out["by_d"][str(d)] = rec
        out["items"][str(d)] = items_rec(T[d], {"pred": P.tolist() if m is not None else P[:, 0].tolist(), "L": [len(seq_std(it, 1)) for it in T[d]],
                                                 "s": [len(it["dec"]) - len(it["think"]) for it in T[d]]})
    out["acc_by_d"] = {d: v["acc"] for d, v in out["by_d"].items()}; return out


@torch.no_grad()                                                            # 10-24 (b345 mean_s OOM: a bare dec_logits built an autograd graph)
def part_lp5(m, g, sets, a, dev, bs, ds):
    K = a["think_k"]; T = tokd(sets, a, ds); out = {"by_d": {}, "items": {}}
    for d in ds:
        acc_ = {"sum": None}

        def grab(t, S, sl):
            if t == 1:
                s_ = S.float().sum(0); acc_["sum"] = s_ if acc_["sum"] is None else acc_["sum"] + s_
        passes(m, T[d], dev, bs, {1}, on_state=grab); mean1 = acc_["sum"] / len(T[d])
        ks = sorted({d, K}); y = np.array([it["ans"] for it in T[d]])
        I, _ = passes(m, T[d], dev, bs, set(ks))
        M, _ = passes(m, T[d], dev, bs, set(ks), edit=lambda t, S, sl: mean1.to(S.dtype)[None].expand(S.shape[0], *S.shape[1:]).contiguous() if t == 1 else None)
        out["by_d"][str(d)] = {"intact": {str(k): float((I[k] == y).mean()) for k in ks}, "mean_at_1": {str(k): float((M[k] == y).mean()) for k in ks},
                               "k_binding": d, "mean_state_norm": float(mean1.norm())}
        out["items"][str(d)] = items_rec(T[d], {"intact": {str(k): I[k].tolist() for k in ks}, "mean_at_1": {str(k): M[k].tolist() for k in ks}})
    out["def"] = "MEAN-AT-1: S after pass 1 := the TEST2-d mean of S_1; passes 2..K' on their own cues; LP5 binding at K' = d, K' = think_k descriptive"; return out


@torch.no_grad()                                                            # 10-24 (b345 mean_s OOM: a bare dec_logits built an autograd graph)
def part_mean_s(m, g, sets, a, dev, bs):
    K = a["think_k"]; T = tokd(sets, a, IO.TEST_DS); tot = None; n = 0; per = {}
    for d in IO.TEST_DS:
        acc_ = {"sum": None}

        def grab(t, S, sl):
            if t == K:
                s_ = S.float().sum(0); acc_["sum"] = s_ if acc_["sum"] is None else acc_["sum"] + s_
        passes(m, T[d], dev, bs, {K}, on_state=grab); per[d] = acc_["sum"] / len(T[d]); tot = acc_["sum"].clone() if tot is None else tot + acc_["sum"]; n += len(T[d])
    pooled = tot / n; out = {"by_d": {}, "items": {}, "mean_state_norm": {"pooled": float(pooled.norm())}}
    for d in IO.TEST_DS:
        y = np.array([it["ans"] for it in T[d]]); pr = {}
        for nm, Sm in (("pooled", pooled), ("per_d", per[d])):
            p_ = []
            for i in range(0, len(T[d]), bs):
                ch = T[d][i:i + bs]; p_.append(pred_of(m.dec_logits(Sm.to(dev)[None].expand(len(ch), *Sm.shape).contiguous(), ch, dev)))
            pr[nm] = np.concatenate(p_)
        out["by_d"][str(d)] = {nm: float((v == y).mean()) for nm, v in pr.items()}; out["items"][str(d)] = items_rec(T[d], {nm: v.tolist() for nm, v in pr.items()})
    out["def"] = "pooled: the decoder reads the TEST2 mean of S_K over d 1..12 (binding at P-2: acc_MEAN <= CD + 0.10); per_d: the TEST2-d mean (descriptive)"; return out


@torch.no_grad()                                                            # 10-24 (b345 mean_s OOM: a bare dec_logits built an autograd graph)
def part_cue_only(m, g, sets, a, dev, bs, ds):
    K = a["think_k"]; T = tokd(sets, a, ds); out = {"by_d": {}, "items": {}}

    def seq_cue(it, t):
        return [EOT] + IO.ENC.encode(" " + ST.cue_str(it["raw"], min(t, it["d"])))
    for d in ds:
        y = np.array([it["ans"] for it in T[d]]); A, X = passes(m, T[d], dev, bs, {K}, seqs_of_t=seq_cue, Td=d, aux_ts=(1, 2))
        out["by_d"][str(d)] = {"answer_acc": float((A[K] == y).mean()), "aux_vacc_pass1": float(X[1].mean()), "aux_vacc_pass2": float(X[2].mean()),
                               "majority": float(np.bincount(y, minlength=11).max() / len(y))}
        out["items"][str(d)] = items_rec(T[d], {"pred": A[K].tolist(), "aux_ok_1": X[1].tolist(), "aux_ok_2": X[2].tolist()})
    out["def"] = ("CUE-ONLY: every pass's think input = [EOT] + enc(' ' + cue_min(t, d)) (no item text); decoder unchanged; read on d >= 2 only; bar max(majority, "
                  "plain DIRECT same items) + 0.05 (the scorer's)"); return out


@torch.no_grad()                                                            # 10-24 (b345 mean_s OOM: a bare dec_logits built an autograd graph)
def part_cue_wrong(m, g, sets, a, dev, bs, ds):
    K = a["think_k"]; T = tokd(sets, a, ds); out = {"by_d": {}, "items": {}}

    def wrong_name(it, t):
        return t + 1 if t < it["d"] else 0

    def seq_wrong(which):
        def f(it, t):
            if which == "all" or which == t:
                return [EOT] + IO.ENC.encode(" " + it["raw"]["text"] + " " + ST.cue_str(it["raw"], wrong_name(it, min(t, it["d"]))))
            return seq_std(it, t)
        return f
    for d in ds:
        y = np.array([it["ans"] for it in T[d]]); ks = sorted({d, K}); rec = {}; items = {}
        for arm in ["all"] + list(range(1, d + 1)):
            A, _ = passes(m, T[d], dev, bs, set(ks), seqs_of_t=seq_wrong(arm), Td=d + (1 if arm == d else 0))
            nm = "all" if arm == "all" else "at_%d" % arm; rec[nm] = {str(k): float((A[k] == y).mean()) for k in ks}; items[nm] = {str(k): A[k].tolist() for k in ks}
        out["by_d"][str(d)] = rec; out["items"][str(d)] = items_rec(T[d], items)
    out["def"] = ("CUE-WRONG: the cue at pass t names S_{t+1} if t < d, else S_0 (min(t, d) indexes the hold passes); 'all' = every pass, 'at_t' = pass t only "
                  "(t = d: the hold passes > d keep the right S_d cue); answers at K' = d and K' = think_k; drop vs lp5 intact labels C1-TOLD (>= 0.30) / C1 (< 0.10)")
    return out


def load(run, dev):
    a = json.load(open(os.path.join(run, "args.json"))); import boxes.deltamem as DM; DM.FAST_ADD = bool(a.get("fast_add"))
    ns = argparse.Namespace(**a); torch.manual_seed(a["seed"]); m, g, params = IO.build_model(ns, dev)
    params.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location=dev, weights_only=True)); params.eval(); return a, m, g


def run(out, parts, dev, bs, model=None, sets=None, train=None):
    a = json.load(open(os.path.join(out, "args.json")))
    if model is None:
        a, m, g = load(out, dev)
    else:
        m, g = model; (m if m is not None else g).eval()
    if sets is None:
        sets = IO.build_sets()
    for k_, v_ in sets["shas"].items():
        assert a.get(k_) in (None, v_), ("§233: set sha mismatch vs args.json", k_)
    ds = list(range(2, a["d_max"] + 1)); parts = [p_ for p_ in parts if p_ in PARTS]
    if m is None:
        parts = [p_ for p_ in parts if p_ == "acc"]
    meta = {"eval_dtype": "fp32", "eval_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "eval_bs": bs, "arm": a["arm"], "cue": a.get("cue", "none"),
            "think_k": a["think_k"], "ds": ds, "module_sha256": IO.sha_file(os.path.abspath(__file__))}
    for p_ in parts:
        t1 = time.time()
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
        r_ = {"acc": lambda: part_acc(m, g, sets, a, dev, bs), "lp5": lambda: part_lp5(m, g, sets, a, dev, bs, ds), "mean_s": lambda: part_mean_s(m, g, sets, a, dev, bs),
              "cue_only": lambda: part_cue_only(m, g, sets, a, dev, bs, ds), "cue_wrong": lambda: part_cue_wrong(m, g, sets, a, dev, bs, ds)}[p_]()
        r_.update(meta); r_["sec"] = round(time.time() - t1)
        json.dump(r_, open(os.path.join(out, "eval_%s.json" % p_), "w", newline="\n")); print("PART", p_, "done", r_["sec"], "s", flush=True)
    summ = {"arm": a["arm"], "cue": a.get("cue", "none"), "think_k": a["think_k"], "d_max": a["d_max"], "seed": a["seed"], "parts": {}, **meta}
    tf = os.path.join(out, "train_summary.json"); summ["train"] = train if train is not None else (json.load(open(tf)) if os.path.exists(tf) else None)
    for p_ in PARTS:
        f_ = os.path.join(out, "eval_%s.json" % p_)
        if os.path.exists(f_):
            summ["parts"][p_] = {"file": "eval_%s.json" % p_, "sec": json.load(open(f_)).get("sec")}
    json.dump(summ, open(os.path.join(out, "eval.json"), "w", newline="\n"), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--parts", default="acc"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval_bs", type=int, default=100); ap.add_argument("--out", default="", help="queue runners append --out; ignored (outputs go to --run)")
    a = ap.parse_args(); run(a.run, [p_ for p_ in a.parts.split(",") if p_], a.device, a.eval_bs); print("DONE", a.run, flush=True)


if __name__ == "__main__":
    main()
