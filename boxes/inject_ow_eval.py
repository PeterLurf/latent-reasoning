"""THEORY §228 LAT-OW evaluation (LAT_v1.md e78d7211; draft §4 / §8.2) for boxes/inject_ow.py runs. fp32 (no autocast); every think pass uses the
cached lower blocks (m.lower once per batch; the LAT cells train with --cache_lower). Every per-item output is stored (store-items rule). Parts
(-> <run>/eval_<part>.json, then eval.json = the summary of every part present):
  acc       TEST2 (1,000 per d, d 1..12, frames 0-5, seed 22960): the answer (argmax over the 11 number tokens) at every K' 0..14 from one sweep
            (inject; direct: one pass). acc_by_d = K' = think_k; acc_by_d_kprime {K': {d: acc}} incl. "d+2"; K'0-NOBIAS (memory wo.bias zeroed; ||b_o||);
            per item: sig, d, ans, pred[K'] (inject) / pred, L (think tokens) and s (decoder suffix tokens) for R-C3.
  mean_s    MEAN-STATE (binding R2 one-pass reference): S after every pass := the TEST2 mean of S_8 -> the decoder reads that mean (the decoder sees only
            the final S, so this equals decoding on the mean state). Two means, both stored: per_d (the mean over the 1,000 items of that d) and pooled
            (over all 12,000). Descriptive mean_s_at_t (d 1..8): S_t := the per-d mean of S_t after pass t only, passes t+1..K own.
  swaps     PAIRS2 (d 3..8, kept <= 1,000 per d, t* = ceil(d / 2)): per pair the recipient's plain answer (own_pred), the donor's own answer at K
            (donor_pred); STATE-SWAP(t) for t in 1..d-1 (S after pass t := the donor's S_t on its own question; passes t+1..K and the decoder on the
            recipient); BUDGET-FRONT at t* (the answer after b = 0..K-t* passes past the swap); CACHE-SWAP(K) (the recipient's decoder on the donor's
            S_K); LEAK-SEG-H (the donor's lower-block (0-5) rows at the last 7 think positions spliced into the recipient's passes 1..t*, passes
            t*+1..K unspliced; exposure: rows_differ, last7_tokens_same); the matched floor (STATE-SWAP(t*) from each of 3 floor donors, seed 22975);
            SELF (donor = recipient for STATE / CACHE / LEAK: bitwise logit identity with the plain run). The permutation floor (seed 22980) is the
            scorer's (it needs only the stored preds and U = {ca, cb}).
  decode    descriptive C7: the read-only aux decoder, greedy after pass t <= d on the first 200 TEST2 items of d 1..8 (exact match, value token).
  transfer  descriptive: frames 6-7 (seed 22961), d 1..12, K' = think_k.
    python -m boxes.inject_ow_eval --run boxes/runs_fsq/<run> --parts acc,mean_s,swaps,decode,transfer [--device cuda:0 --eval_bs 100]"""
import argparse
import contextlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from boxes import inject_ow as IO  # noqa: E402
from boxes.nested170 import NUM_IDS, EOT  # noqa: E402

KMAX = 14                                                                   # the sweep: 0..12 plus d + 2 for d 12
LEAK_W = 7                                                                  # the last 7 think positions (draft §4 battery map)
NULLC = contextlib.nullcontext


def pred_of(lg):
    return lg[:, NUM_IDS].argmax(-1).cpu().numpy()


@torch.no_grad()
def answers_by_kprime(m, g, items, kmax, dev, bs, ac=None):
    """-> (N, kmax + 1) answers at K' = 0..kmax from one sweep (inject), or (N, 1) for the direct trunk (m None). ac: an autocast factory or None (fp32)."""
    ctx = ac or NULLC; out = np.full((len(items), (kmax + 1) if m is not None else 1), -1)
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]; r = slice(i, i + len(ch))
        if m is None:
            out[r, 0] = pred_of(IO.direct_logits(g, ch, dev, ctx)); continue
        with ctx():
            x, msk = m.think_inputs(ch, dev); hl = m.lower(x); S = m.mem.zero_state(len(ch), dev); out[r, 0] = pred_of(m.dec_logits(S, ch, dev))
            for t in range(1, kmax + 1):
                S = m.write(S, x, msk, hl); out[r, t] = pred_of(m.dec_logits(S, ch, dev))
    return out


def tokd(sets, key, ds):
    return {d: [IO.tok_ow(it) for it in sets[key][d]] for d in ds}


def part_acc(m, g, sets, a, dev, bs, key="test2"):
    T = tokd(sets, key, IO.TEST_DS); K = a["think_k"]; out = {"set": key, "by_d": {}, "items": {}}
    for d in IO.TEST_DS:
        P = answers_by_kprime(m, g, T[d], KMAX if m is not None else 0, dev, bs); y = np.array([it["ans"] for it in T[d]])
        rec = {"acc": float((P[:, K if m is not None else 0] == y).mean()), "k_main": K if m is not None else 0}
        if m is not None:
            rec["acc_kprime"] = {str(k): float((P[:, k] == y).mean()) for k in range(KMAX + 1)}; rec["acc_kprime"]["d+2"] = float((P[:, d + 2] == y).mean())
        out["by_d"][str(d)] = rec
        out["items"][str(d)] = {"sig": [it["sig"] for it in T[d]], "ans": y.tolist(), "pred": P.tolist() if m is not None else P[:, 0].tolist(),
                                "L": [len(it["think"]) for it in T[d]], "s": [len(it["dec"]) - len(it["think"]) for it in T[d]]}
    out["acc_by_d"] = {d: v["acc"] for d, v in out["by_d"].items()}
    if m is not None and key == "test2":                                    # K'0-NOBIAS: the memory branch's output bias zeroed (the plain trunk at K' = 0)
        b0 = m.mem.wo.bias.detach().clone(); out["wo_bias_norm"] = float(b0.norm())
        try:
            m.mem.wo.bias.data.zero_(); nb = {}
            for d in IO.TEST_DS:
                P0 = answers_by_kprime(m, g, T[d], 0, dev, bs)[:, 0]; nb[str(d)] = {"acc": float((P0 == np.array(out["items"][str(d)]["ans"])).mean()), "pred": P0.tolist()}
        finally:
            m.mem.wo.bias.data.copy_(b0)
        out["kprime0_nobias"] = nb
    out["kprime_def"] = "pred[k] = the answer after k think passes (k = 0: the decoder on S = 0); K' = think_k is the trained K; 'd+2' = pred[d + 2]"
    return out


@torch.no_grad()
def state_sums(m, items, dev, bs, K):
    """-> [sum over items of S_t for t = 1..K] (fp32, on dev): the per-pass state sums (no per-item states kept)."""
    sums = None
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]; x, msk = m.think_inputs(ch, dev); hl = m.lower(x); S = m.mem.zero_state(len(ch), dev); s_ = []
        for _ in range(K):
            S = m.write(S, x, msk, hl); s_.append(S.float().sum(0))
        sums = s_ if sums is None else [u + v for u, v in zip(sums, s_)]
    return sums


@torch.no_grad()
def part_mean_s(m, g, sets, a, dev, bs):
    T = tokd(sets, "test2", IO.TEST_DS); K = a["think_k"]; mean_d = {}; mean_dt = {}; tot = None; n = 0
    for d in IO.TEST_DS:
        sums = state_sums(m, T[d], dev, bs, K); mean_dt[d] = [s_ / len(T[d]) for s_ in sums]; mean_d[d] = mean_dt[d][-1]
        tot = sums[-1].clone() if tot is None else tot + sums[-1]; n += len(T[d])
    pooled = tot / n; out = {"by_d": {}, "items": {}, "mean_state_norm": {"pooled": float(pooled.norm()), **{str(d): float(v.norm()) for d, v in mean_d.items()}}}
    for d in IO.TEST_DS:
        y = np.array([it["ans"] for it in T[d]]); pr = {}
        for nm, Sm in (("per_d", mean_d[d]), ("pooled", pooled)):
            p_ = []
            for i in range(0, len(T[d]), bs):
                ch = T[d][i:i + bs]; p_.append(pred_of(m.dec_logits(Sm.to(dev)[None].expand(len(ch), *Sm.shape).contiguous(), ch, dev)))
            pr[nm] = np.concatenate(p_)
        out["by_d"][str(d)] = {nm: float((v == y).mean()) for nm, v in pr.items()}; out["items"][str(d)] = {nm: v.tolist() for nm, v in pr.items()}
    at = {}
    for d in range(1, 9):                                                   # descriptive: S_t := the per-d mean of S_t after pass t only
        means = mean_dt[d]; y = np.array([it["ans"] for it in T[d]]); at[str(d)] = {}
        for t in range(1, K + 1):
            p_ = []
            for i in range(0, len(T[d]), bs):
                ch = T[d][i:i + bs]; x, msk = m.think_inputs(ch, dev); hl = m.lower(x); S = m.mem.zero_state(len(ch), dev)
                for u in range(1, K + 1):
                    S = m.write(S, x, msk, hl)
                    if u == t:
                        S = means[t - 1].to(dev)[None].expand(len(ch), *S.shape[1:]).to(S.dtype).contiguous()
                p_.append(pred_of(m.dec_logits(S, ch, dev)))
            p_ = np.concatenate(p_); at[str(d)][str(t)] = {"acc": float((p_ == y).mean()), "pred": p_.tolist()}
    out["mean_s_at_t"] = at
    out["def"] = ("per_d / pooled: the decoder reads the TEST2 mean of S_K (per depth / over all 12,000 items) in place of the item's S (S := the mean after "
                  "every pass, which leaves the decoder's input = the mean); mean_s_at_t: S_t := the per-d mean of S_t after pass t, passes t+1..K own")
    return out


@torch.no_grad()
def run_from(m, S, x, msk, hl, t0, K, dev, items, front=False):
    """passes t0+1..K from S; -> (S_K, final logits, [answers after b = 0..K-t0 passes] if front)."""
    fr = [pred_of(m.dec_logits(S, items, dev))] if front else None
    for _ in range(t0 + 1, K + 1):
        S = m.write(S, x, msk, hl)
        if front:
            fr.append(pred_of(m.dec_logits(S, items, dev)))
    return S, m.dec_logits(S, items, dev), fr


@torch.no_grad()
def plain_states(m, x, msk, hl, K, B, dev):
    S = m.mem.zero_state(B, dev); out = [S]
    for _ in range(K):
        S = m.write(S, x, msk, hl); out.append(S)
    return out


@torch.no_grad()
def part_swaps(m, g, sets, a, dev, bs):
    K = a["think_k"]; out = {"by_d": {}, "items": {}, "pair_stats": sets["pair_stats"], "self_bitwise": {"state": True, "cache": True, "leak": True}}
    bs = max(1, bs // 2)
    for d in IO.PAIR_DS:
        P = sets["pairs"][d]; FL = sets["floors"][d]; ts = math.ceil(d / 2); rows = []
        if not P:
            out["by_d"][str(d)] = {"n_pairs": 0}; continue
        for i in range(0, len(P), bs):
            pc = P[i:i + bs]; fc = FL[i:i + bs]; R = [IO.tok_ow(p_["rec"]) for p_ in pc]; D = [IO.tok_ow(p_["don"]) for p_ in pc]; B = len(R)
            assert all(len(r_["think"]) == len(d_["think"]) for r_, d_ in zip(R, D)), "§228: exact-length pairs"
            xr, mr = m.think_inputs(R, dev); xd, md = m.think_inputs(D, dev); hr = m.lower(xr); hd = m.lower(xd)
            SR = plain_states(m, xr, mr, hr, K, B, dev); SD = plain_states(m, xd, md, hd, K, B, dev)
            lg_own = m.dec_logits(SR[K], R, dev); own_p = pred_of(lg_own); don_p = pred_of(m.dec_logits(SD[K], D, dev))
            sw = {}
            for t in range(1, d):
                _, lg_, fr_ = run_from(m, SD[t], xr, mr, hr, t, K, dev, R, front=(t == ts)); sw[t] = pred_of(lg_)
                if t == ts:
                    front = fr_
                _, lg_s, _ = run_from(m, SR[t], xr, mr, hr, t, K, dev, R)  # SELF (donor = recipient)
                out["self_bitwise"]["state"] &= bool(torch.equal(lg_s, lg_own))
            cache_p = pred_of(m.dec_logits(SD[K], R, dev)); out["self_bitwise"]["cache"] &= bool(torch.equal(m.dec_logits(SR[K], R, dev), lg_own))
            Ls = [len(r_["think"]) for r_ in R]; hl_leak = hr.clone(); hl_self = hr.clone(); differ = []; same7 = []
            for j, L_ in enumerate(Ls):
                hl_leak[j, L_ - LEAK_W:L_] = hd[j, L_ - LEAK_W:L_]; hl_self[j, L_ - LEAK_W:L_] = hr[j, L_ - LEAK_W:L_]
                differ.append(bool((hd[j, L_ - LEAK_W:L_] != hr[j, L_ - LEAK_W:L_]).any())); same7.append(R[j]["think"][-LEAK_W:] == D[j]["think"][-LEAK_W:])
            outs_ = []
            for hl_ in (hl_leak, hl_self):
                S = m.mem.zero_state(B, dev)
                for _ in range(ts):
                    S = m.write(S, xr, mr, hl_)
                _, lg_, _ = run_from(m, S, xr, mr, hr, ts, K, dev, R); outs_.append(lg_)
            leak_p = pred_of(outs_[0]); out["self_bitwise"]["leak"] &= bool(torch.equal(outs_[1], lg_own))
            flo = np.full((B, IO.FLOOR_K), -1)                              # the matched floor: STATE-SWAP(t*) from each floor donor
            for k in range(IO.FLOOR_K):
                idx = [j for j in range(B) if j < len(fc) and fc[j][k] is not None]
                if not idx:
                    continue
                F_ = [IO.tok_ow(fc[j][k]) for j in idx]; xf, mf = m.think_inputs(F_, dev); hf = m.lower(xf); Sf = m.mem.zero_state(len(F_), dev)
                for _ in range(ts):
                    Sf = m.write(Sf, xf, mf, hf)
                ii = torch.tensor(idx, device=dev); _, lg_, _ = run_from(m, Sf, xr[ii], mr[ii], hr[ii], ts, K, dev, [R[j] for j in idx]); flo[idx, k] = pred_of(lg_)
            for j, p_ in enumerate(pc):
                rows.append({"rec_sig": p_["rec"]["sig"], "don_sig": p_["don"]["sig"], "t": ts, "own": int(p_["rec"]["answer"]), "dfin": int(p_["don"]["answer"]),
                             "ca": int(p_["ca"]), "cb": int(p_["cb"]), "own_pred": int(own_p[j]), "donor_pred": int(don_p[j]),
                             "state_swap": {str(t): int(sw[t][j]) for t in sw}, "budget_front": [int(f_[j]) for f_ in front], "cache_swap": int(cache_p[j]),
                             "leak_seg": int(leak_p[j]), "leak_rows_differ": differ[j], "last7_tokens_same": bool(same7[j]),
                             "floor_preds": flo[j].tolist(), "floor_sigs": [(x_["sig"] if x_ is not None else None) for x_ in (fc[j] if j < len(fc) else [None] * IO.FLOOR_K)]})
        U = lambda r_, p: p in (r_["ca"], r_["cb"])  # noqa: E731
        st = {"n_pairs": len(rows), "t": ts, "follow_U": float(np.mean([U(r_, r_["state_swap"][str(ts)]) for r_ in rows])),
              "own_share": float(np.mean([r_["state_swap"][str(ts)] == r_["own"] for r_ in rows])), "donor_final_share": float(np.mean([r_["state_swap"][str(ts)] == r_["dfin"] for r_ in rows])),
              "cache_swap_own": float(np.mean([r_["cache_swap"] == r_["own"] for r_ in rows])), "cache_swap_donor_final": float(np.mean([r_["cache_swap"] == r_["dfin"] for r_ in rows])),
              "leak_follow_U": float(np.mean([U(r_, r_["leak_seg"]) for r_ in rows])), "leak_changed_output": float(np.mean([r_["leak_seg"] != r_["own_pred"] for r_ in rows])),
              "leak_rows_differ_share": float(np.mean([r_["leak_rows_differ"] for r_ in rows])), "last7_tokens_same_share": float(np.mean([r_["last7_tokens_same"] for r_ in rows]))}
        out["by_d"][str(d)] = st; out["items"][str(d)] = rows
    out["def"] = ("U = {ca (cf(a): states 0..t from the donor), cb (cf_b: S_t from the donor, earlier states the recipient's)}; follow_U / own / donor_final at "
                  "t* = ceil(d / 2) are descriptive summaries (the registered R3 = follow_U - max(matched, permutation floor) is the scorer's)")
    return out


@torch.no_grad()
def part_decode(m, g, sets, a, dev, bs, n_per_d=200):
    K = a["think_k"]; out = {"by_d": {}, "items": {}}
    for d in range(1, 9):
        T = [IO.tok_ow(it) for it in sets["test2"][d][:n_per_d]]; ex = {t: [] for t in range(1, d + 1)}; vv = {t: [] for t in range(1, d + 1)}; gen_all = {t: [] for t in range(1, d + 1)}
        for i in range(0, len(T), bs):
            ch = T[i:i + bs]; x, msk = m.think_inputs(ch, dev); hl = m.lower(x); S = m.mem.zero_state(len(ch), dev)
            for t in range(1, min(d, K) + 1):
                S = m.write(S, x, msk, hl); L_ = max(len(it["aux"][t - 1]) for it in ch); gen = m.aux_greedy(S, t, L_)
                for r_, it in enumerate(ch):
                    tg = it["aux"][t - 1]; gl = [int(v) for v in gen[r_, :len(tg)]]; ex[t].append(gl == tg); vi = it["vidx"][t - 1]; vv[t].append(gl[vi] == tg[vi]); gen_all[t].append(gl)
        out["by_d"][str(d)] = {str(t): {"exact": float(np.mean(ex[t])), "value_token": float(np.mean(vv[t]))} for t in ex if ex[t]}
        out["items"][str(d)] = {"sig": [it["sig"] for it in T], **{str(t): {"gen": gen_all[t], "exact": ex[t], "value_token": vv[t]} for t in ex if ex[t]}}
    out["def"] = "aux_greedy([EOT, ' Step', ' t', ' :']) on S_t (read-only), the target length; exact = every token incl. EOT; value_token = the token after ' has'"
    return out


PARTS = ("acc", "mean_s", "swaps", "decode", "transfer")


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
        t_ = time.time(); sets = IO.build_sets(); print("SETS", round(time.time() - t_), "s", flush=True)
    for k_, v_ in sets["shas"].items():
        assert a.get(k_) in (None, v_), ("§228: set sha mismatch vs args.json", k_)
    parts = [p_ for p_ in parts if p_ in PARTS]
    if m is None:
        parts = [p_ for p_ in parts if p_ in ("acc", "transfer")]
    meta = {"eval_dtype": "fp32", "eval_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "eval_bs": bs, "arm": a["arm"], "think_k": a["think_k"],
            "module_sha256": IO.sha_file(os.path.abspath(__file__))}
    for p_ in parts:
        t1 = time.time()
        r_ = {"acc": lambda: part_acc(m, g, sets, a, dev, bs), "transfer": lambda: part_acc(m, g, sets, a, dev, bs, key="transfer"),
              "mean_s": lambda: part_mean_s(m, g, sets, a, dev, bs), "swaps": lambda: part_swaps(m, g, sets, a, dev, bs),
              "decode": lambda: part_decode(m, g, sets, a, dev, bs)}[p_]()
        r_.update(meta); r_["sec"] = round(time.time() - t1)
        json.dump(r_, open(os.path.join(out, "eval_%s.json" % p_), "w", newline="\n")); print("PART", p_, "done", r_["sec"], "s", flush=True)
    summ = {"arm": a["arm"], "think_k": a["think_k"], "d_max": a["d_max"], "seed": a["seed"], "parts": {}, **meta}
    tf = os.path.join(out, "train_summary.json")
    summ["train"] = train if train is not None else (json.load(open(tf)) if os.path.exists(tf) else None)
    for p_ in PARTS:
        f_ = os.path.join(out, "eval_%s.json" % p_)
        if os.path.exists(f_):
            e_ = json.load(open(f_)); summ["parts"][p_] = {"file": "eval_%s.json" % p_, "sec": e_.get("sec")}
            if p_ in ("acc", "transfer"):
                summ["parts"][p_]["n_items"] = sum(len(v["ans"]) for v in e_["items"].values())
    json.dump(summ, open(os.path.join(out, "eval.json"), "w", newline="\n"), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--parts", default="acc"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval_bs", type=int, default=100); a = ap.parse_args()
    run(a.run, [p_ for p_ in a.parts.split(",") if p_], a.device, a.eval_bs); print("DONE", a.run, flush=True)


if __name__ == "__main__":
    main()
