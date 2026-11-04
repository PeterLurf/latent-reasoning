"""THEORY §271 N1 SHUFFLED-OBJECTS evaluation (N1_v1 §3.5-3.7, §6, §11; Note 271-1) for boxes/inject_so.py runs (LOOP-SI / LOOP-EN / GENERIC and the
one-pass arms DIRECT-OF / DIRECT-SI, `--arm direct`). A NEW module: inject_aw_eval / inject_aw / deltamem / so_words / so_tok are imported unchanged.
fp32; every part is its own @torch.no_grad() function and writes eval_<part>.json in the run dir; run() also writes eval.json (index + meta).
EVERY readout is so_tok.pick (argmax over the item's own people_ids, -> a NAMES_SO index), never full-vocabulary; BBH reads X's first BPE token.
passes_so() is inject_aw_eval.passes (line 55) copied with ONE edit: answers via pick (the cue / hold / edit / on_state mechanics unchanged).
Loop arms read at K' = max(think_k, m) (main) and K' = think_k; one-pass arms read [EOT] + text + " The answer is:" once (inject_ow.direct_logits).
Every output file carries the meta block (eval_dtype, eval_gpu, eval_bs, arm, cue, think_k, module_sha256, so_words_sha256, so_tok_sha256, n_smoke, set_shas, sec).
Parts not defined for the one-pass arms write {"arm": "direct", "n/a": true} (identical file list for every arm, §3.6).

OUTPUT FIELD PATHS ('/'-separated as in theory/score_n1.py SCHEMA (commit a41e40af), which this module matches; every key is a string;
<n> in 3,4,5; <m> 1..12; <mode> in NUM, ORD, TRANSFER (n 5, m 1-8, frames 4-5), OFFSET (n 5, m 4, labels 9-12); K' / t / L / k keys str(int)):
  eval_acc_so.json
    set_sha (the TEST set sha = so_test_sha), set_shas{so_test_sha, so_transfer_sha, so_offset_sha}
    by_cell/<n>/<m>/<mode>/{n_correct, n_items, n_correct_moved, n_moved, acc, acc_moved, acc_moved_k8, floor_hold, f, sm1_confusion, n_sm1, k_main,
                            by_k_touch/<k>/{n_correct, n, acc}, by_kprime/<K'>/{n_correct, n_correct_moved, acc, acc_moved} (loop only),
                            items/{sig, pred, pred_k/<K'> (loop only), ans, k_touch, moved, s0, sm1, s2, s4}}
      counts / acc / by_k_touch (ALL items) / sm1_confusion at K' main = max(think_k, m) (direct: the one pass); MOVED = S_m != S_0 and k_touch >= 2;
      acc_moved_k8 = MOVED accuracy at K' = think_k (direct: = acc_moved); floor_hold = P(S_m == S_0) on the cell (model-free);
      sm1_confusion = P(pred == S_{m-1} | S_{m-1} != S_m) over n_sm1 items (null if 0); f = floors_so.json floors/<n>/<m>/<mode>/f as given (null if absent);
      s2 / s4 = S_2 / S_4 (null if m < 2 / 4); item values are NAMES_SO indices (so_tok.val_index).
  eval_ws_so.json     (loop) by_m[m]{ws[t], cws[t], n[t], n_items} for n 5 NUM m 4 / 8 / 12, t = 1 .. max(think_k, m); ws[t] = P(pick at pass-t ':' == S_min(t, m));
                      cws[1] = ws[1], cws[t] = P(WS_t | WS_{t-1}) over the n[t] items with WS_{t-1} (null if 0); items[m]{sig, pred_by_pass[t]}.
                      (direct) {"arm": "direct", "n/a": true}
  eval_swap_so.json   (loop) by_cell[c]{n_pairs, n_tries, t, m, k_main, follow_cf, stay_own, follow_dfin, follow_stale, mean_floor, mean_stay_own, intact_acc,
                      by_kprime[K']{follow_cf, stay_own, follow_dfin, follow_stale, mean_floor, intact_acc}}, c in {"5_4_2", "5_8_4", "5_6_3"} (<n>_<m>_<t>);
                      items[c]{rec_sig, don_sig, own, dfin, stale, cf, swap, mean, intact} (at K' main). (direct) n/a
  eval_kcurve_so.json (loop) n5 m8 NUM: acc_moved[K'], acc[K'], acc_vs_sk[K'] (pred == S_min(K', m)), n_items, n_moved, K' in {0,1,2,4,6,7,8,9,12,16};
                      s_norm/<8|10|12>/{mean/<t>, r_median/<t>} (mean ||S_t||_F; median over items of ||S_t||_F / ||S_4||_F; m 8: t 1..16, m 10 / 12: t 1..12);
                      by_m/<10|12>/{acc/<K'>, acc_moved/<K'>, n_items, n_moved} at K' 8 / 12;
                      not_decoder/{acc_vs_s2, acc_vs_s4, n, n_stratum, n_cell, kprime} (M4: the acc_so TEST cell n5 m4 NUM itself (500 items), K' 2, on its
                      S_2 != S_4 stratum, n = the stratum size).
                      (direct) acc_moved["0"], acc["0"], by_m[m]{acc["0"], acc_moved["0"]}, n_items, n_moved, "arm": "direct" (the one pass = the K' 0 row)
  eval_long_so.json   (loop) by_m/<m>/{acc/<K'>, sm1_confusion/<K'>, n, n_items, n_sm1}, n5 NUM m 1-4, K' in {8, 12, 63, 64}. (direct) n/a
  eval_perm_label.json by_m/<m>/{acc, n, acc_chain, n_items, n_identity_redraws}; n5 NUM m 2-4, swap sentences in a non-identity order from
                      default_rng((SO.SEED_PERM, m, i)), labels = the true index; acc_chain = the same items in chain order; items[m]{sig, order, pred, ans}.
  eval_lookup.json    by_L[L]{acc, n}; n5 m12 NUM items, decoder on S = 0 (loop) / the one pass (direct) over [EOT] + enc(" " + text + " In swap L, who trades
                      with a_L?") + SFX -> b_L; items{sig, pred[L]}.
  eval_bbh.json       subsets/<three|five|seven>/{acc, n, n_parsed, m, per_item[{idx, pred, gold, m, X, parsed (bool), correct (bool)}]}; rule (string),
                      bbh_sha256{file: sha}. acc = #(parsed and correct) / n (unparsed items count wrong).
  eval_beta_so.json   (loop) share_gt1 (beta > 1 over every write pass t <= m, every delta head, every item, n5 NUM m 5-12), share_gt1_hold (passes t > m),
                      mean_beta, by_m[m]{share_gt1, mean_beta}, by_pass[t]{share_gt1, n}, by_head[h] share_gt1, recompute_maxabs. (direct) n/a
  eval_floors.json    status ("absent" | "loaded"), path, sha256, set_sha, set_sha_match (read-only echo of <runs>/n1_floors/floors_so.json; Theory writes it at G0).
  eval.json           {arm, cue, think_k, seed, parts{p: {file, sec}}, train, meta}
Smoke: env N_SMOKE=k truncates every cell / pair list / BBH subset to k items (and swap_pairs' cap to k).
    python -m boxes.inject_so_eval --run boxes/runs_fsq/<run> --parts acc_so,ws_so,swap_so,kcurve_so,long_so,perm_label,lookup,bbh,beta_so,floors [--device cuda:0 --eval_bs 250]"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from boxes import inject_ow as IO  # noqa: E402
from boxes import inject_aw as AW  # noqa: E402
from boxes import inject_aw_eval as AE  # noqa: E402
import boxes.so_words as SO  # noqa: E402
import boxes.so_tok as SOT  # noqa: E402
from boxes.inject_ow import ENC, SFX  # noqa: E402
from boxes.nested170 import EOT, pad  # noqa: E402

PARTS = ("acc_so", "ws_so", "swap_so", "kcurve_so", "long_so", "perm_label", "lookup", "bbh", "beta_so", "floors")
LOOP_ONLY = ("ws_so", "swap_so", "long_so", "beta_so")                     # n/a for the one-pass arms (kcurve_so: the K' 0 row only)
NA = {"arm": "direct", "n/a": True}
WS_MS = (4, 8, 12)
SWAP_CELLS = ((5, 4, 2), (5, 8, 4), (5, 6, 3))                              # §3.6 / §3.7: (n, m, t)
KC_KS = (0, 1, 2, 4, 6, 7, 8, 9, 12, 16); KC_EXTRA_MS = (10, 12); KC_EXTRA_KS = (8, 12)
LONG_KS = (8, 12, 63, 64); LONG_MS = (1, 2, 3, 4)
PERM_MS = (2, 3, 4); LOOKUP_M = 12; BETA_MS = tuple(range(5, 13))
BBH_SUBSETS = ("three", "five", "seven"); BBH_N = {"three": 3, "five": 5, "seven": 7}
BBH_DIR_DEFAULT = os.path.join(ROOT, "scratch", "n1_build", "bbh")
N_SMOKE = int(os.environ.get("N_SMOKE", "0") or 0)


def trunc(xs):
    return xs[:N_SMOKE] if N_SMOKE else xs


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def build_sets():
    """SO.test_sets() + their shas (the trainer's excl comes from the same function)."""
    s = SO.test_sets(); s["shas"] = set_shas(s); return s


def set_shas(s):
    return {"so_test_sha": SO.set_sha([it for k in sorted(s["test"]) for it in s["test"][k]]),
            "so_transfer_sha": SO.set_sha([it for m in sorted(s["transfer"]) for it in s["transfer"][m]]), "so_offset_sha": SO.set_sha(s["offset"])}


def cue_of(a):
    """the run's SO cue (tok_so: index / entity / generic); the one-pass arms never read cues (tok_so still needs one: index)."""
    c = a.get("cue", "index")
    return c if c in SOT.CUES else "index"


def toks(items, a):
    return [SOT.tok_so(it, cue_of(a)) for it in trunc(items)]


def vi(name):
    return SOT.val_index(name)


def wrap(x):
    return float(x) if x == x else None                                     # NaN -> null


def mean_or_none(mask_vals):
    v = np.asarray(mask_vals, dtype=float); return float(v.mean()) if len(v) else None


# ---------------------------------------------------------------------------------------------------- pass runners

@torch.no_grad()
def passes_so(m, items, dev, bs, kset, seqs_of_t=AE.seq_std, Td=None, edit=None, aux_ts=(), on_state=None):
    """inject_aw_eval.passes (55) copied; ONE edit: answers via so_tok.pick (argmax over the item's own people_ids) in place of pred_of (NUM_IDS).
    -> ({K': answers (N,) NAMES_SO indices}, {t: [0/1 per item]})."""
    out = {k: np.full(len(items), -1) for k in kset}; auxv = {t: np.full(len(items), -1) for t in aux_ts}; Td = Td or AE.n_distinct(items); kmax = max(kset)
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]; sl = slice(i, i + len(ch)); xs = []; m0 = None
        for t in range(1, Td + 1):
            ids_, mk = pad([seqs_of_t(it, t) for it in ch]); assert m0 is None or (mk == m0).all(), "one padding across passes"; m0 = mk
            xs.append(torch.tensor(ids_, device=dev))
        if seqs_of_t is AE.seq_std and any(it.get("cue") in ("generic", "target") for it in ch):   # §252 PLAN-FREE (NS_v1 §3) / §261 TC: one identical input at every pass
            assert all(it.get("cue") in ("generic", "target") for it in ch) and all(torch.equal(x_, xs[0]) for x_ in xs), "PLAN-FREE: generic / target-cue pass inputs differ across t"
        msk = torch.tensor(m0, device=dev); hls = [m.lower(x) for x in xs]; S = m.mem.zero_state(len(ch), dev)
        if 0 in kset:
            out[0][sl] = SOT.pick(m.dec_logits(S, ch, dev), ch)
        for t in range(1, kmax + 1):
            j = min(t, Td) - 1; S = m.write(S, xs[j], msk, hls[j])
            if on_state is not None:
                on_state(t, S, sl)
            if edit is not None:
                S2 = edit(t, S, sl); S = S if S2 is None else S2
            if t in aux_ts:
                lg_a, y_a = m.aux_logits(S, ch, t, dev); pr = lg_a.argmax(-1); vi_ = torch.tensor([it["vidx"][t - 1] for it in ch], device=dev); ar = torch.arange(len(ch), device=dev)
                auxv[t][sl] = (pr[ar, vi_] == y_a[ar, vi_]).long().cpu().numpy()
            if t in kset:
                out[t][sl] = SOT.pick(m.dec_logits(S, ch, dev), ch)
    return out, auxv


@torch.no_grad()
def colon_passes(m, items, dev, bs, kmax, on_colon):
    """the run's own passes (inject_st.cue_inputs: pass t > Td reads x_Td; one padding asserted) via InjectAW.pass_aw; on_colon(t, logits at the pass-t ':'
    (B, V) fp32, ch, sl) at every pass t = 1..kmax (inject_aw.ws_readout's mechanics; never fed back)."""
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]; sl = slice(i, i + len(ch)); xs, msk = AW.cue_inputs(m, ch, dev); hls = [m.lower(x_) for x_ in xs]; S = m.mem.zero_state(len(ch), dev)
        for t in range(1, kmax + 1):
            j = min(t, len(xs)) - 1; S, h, pos = m.pass_aw(S, xs[j], msk, hls[j]); on_colon(t, m.colon_logits(h, pos), ch, sl)


@torch.no_grad()
def direct_pick(g, items, dev, bs):
    """the one-pass arms: inject_ow.direct_logits on dec ([EOT] + text + " The answer is:"), fp32, -> pick."""
    out = []
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]; out.append(SOT.pick(IO.direct_logits(g, ch, dev, AE.NULLC), ch))
    return np.concatenate(out) if out else np.zeros(0, int)


@torch.no_grad()
def grab_states_so(m, items, dev, bs, ts):
    """S_t (fp32, CPU) for t in ts, per item (inject_aw_eval.grab_states on passes_so)."""
    St = {t: [] for t in ts}

    def grab(t, S, sl):
        if t in St:
            assert sl.start == sum(len(x_) for x_ in St[t]), "grab_states: batches in order"; St[t].append(S.float().cpu())
    passes_so(m, items, dev, bs, {max(ts)}, on_state=grab); return {t: torch.cat(v) for t, v in St.items()}


# ---------------------------------------------------------------------------------------------------- acc_so

def cell_record(T, P, kmain, Pk=None, floor=None):
    """T: tok_so items; P: answers at K' main (NAMES_SO indices); Pk: {K': answers} (loop) -> CELL (see the docstring)."""
    raw = [it["raw"] for it in T]; y = np.array([it["ans"] for it in T]); n = len(T)
    mv = np.array([r_["moved"] for r_ in raw], bool); kt = np.array([r_["k_touch"] for r_ in raw]); m_ = raw[0]["m"] if raw else 0
    s0 = np.array([vi(r_["values"][0]) for r_ in raw]); sm1 = np.array([vi(r_["values"][-2]) for r_ in raw]); ok = P == y
    smask = sm1 != y
    rec = {"n_correct": int(ok.sum()), "n_items": n, "n_correct_moved": int(ok[mv].sum()), "n_moved": int(mv.sum()), "acc": mean_or_none(ok),
           "acc_moved": mean_or_none(ok[mv]), "floor_hold": mean_or_none(s0 == y), "f": floor,
           "by_k_touch": {str(int(k)): {"n_correct": int(ok[kt == k].sum()), "n": int((kt == k).sum()), "acc": mean_or_none(ok[kt == k])}
                          for k in sorted(set(kt.tolist()))},
           "sm1_confusion": mean_or_none((P == sm1)[smask]), "n_sm1": int(smask.sum()), "k_main": kmain,
           "items": {"sig": [it["sig"] for it in T], "pred": P.tolist(), "ans": y.tolist(), "k_touch": kt.tolist(), "moved": mv.astype(int).tolist(), "s0": s0.tolist(),
                     "sm1": sm1.tolist(), "s2": [vi(r_["values"][2]) if m_ >= 2 else None for r_ in raw], "s4": [vi(r_["values"][4]) if m_ >= 4 else None for r_ in raw]}}
    if Pk is not None:
        rec["by_kprime"] = {str(k): {"n_correct": int((Pk[k] == y).sum()), "n_correct_moved": int((Pk[k] == y)[mv].sum()), "acc": mean_or_none(Pk[k] == y),
                                     "acc_moved": mean_or_none((Pk[k] == y)[mv])} for k in sorted(Pk)}
        rec["acc_moved_k8"] = rec["by_kprime"][str(min(Pk))]["acc_moved"]       # K' = think_k (8); equals acc_moved for m <= think_k
        rec["items"]["pred_k"] = {str(k): Pk[k].tolist() for k in sorted(Pk)}
    return rec


def eval_cell(m, g, T, a, dev, bs, floor=None):
    """one cell: loop -> K' {think_k, max(think_k, m)}; direct -> the one pass."""
    if not T:
        return {"n_correct": 0, "n_items": 0, "n_correct_moved": 0, "n_moved": 0, "acc": None, "acc_moved": None, "acc_moved_k8": None, "floor_hold": None, "f": floor, "by_k_touch": {}, "sm1_confusion": None, "n_sm1": 0, "items": {}}
    if m is None:
        r_ = cell_record(T, direct_pick(g, T, dev, bs), "direct", None, floor); r_["acc_moved_k8"] = r_["acc_moved"]; return r_
    K = a["think_k"]; km = max(K, T[0]["d"]); assert all(it["d"] == T[0]["d"] for it in T); Pk, _ = passes_so(m, T, dev, bs, {K, km})
    return cell_record(T, Pk[km], km, Pk, floor)


def floor_of(floors, n, mm, MD):
    """floors_so.json floors/<n>/<m>/<mode>/f, read as given (null if the file or the key is absent)."""
    try:
        return floors[str(n)][str(mm)][MD]["f"]
    except (TypeError, KeyError):
        return None


@torch.no_grad()
def part_acc_so(m, g, sets, a, dev, bs):
    out = {"by_cell": {}, "set_sha": sets["shas"]["so_test_sha"], "set_shas": sets["shas"]}; fl = sets.get("floors"); bc = out["by_cell"]

    def put(n, mm, MD, T):
        bc.setdefault(str(n), {}).setdefault(str(mm), {})[MD] = eval_cell(m, g, T, a, dev, bs, floor_of(fl, n, mm, MD))
    for (n, mm, md) in sorted(sets["test"]):
        put(n, mm, md.upper(), toks(sets["test"][(n, mm, md)], a))
    for mm in sorted(sets["transfer"]):
        put(5, mm, "TRANSFER", toks(sets["transfer"][mm], a))
    put(5, 4, "OFFSET", toks(sets["offset"], a))
    out["def"] = ("acc_so (§3.6): every TEST cell (n 3-5, m 1-12, num / ord; 500 items, n 5 / n 3 m 8 NUM 1,500), TRANSFER (frames 4-5, n 5 NUM m 1-8), OFFSET (n 5 m 4 NUM, "
                  "labels 9-12); pick over people_ids; loop at K' max(think_k, m) (main) and think_k; direct one pass; MOVED = S_m != S_0 and k_touch >= 2")
    return out


# ---------------------------------------------------------------------------------------------------- ws_so

@torch.no_grad()
def part_ws_so(m, g, sets, a, dev, bs):
    if m is None:
        return dict(NA)
    K = a["think_k"]; out = {"by_m": {}, "items": {}}
    for mm in WS_MS:
        T = toks(sets["test"][(5, mm, "num")], a); km = max(K, mm); pv = {t: np.full(len(T), -1) for t in range(1, km + 1)}

        def grab(t, lg, ch, sl):
            pv[t][sl] = SOT.pick(lg, ch)
        colon_passes(m, T, dev, bs, km, grab)
        V = np.array([[vi(it["raw"]["values"][min(t, mm)]) for t in range(1, km + 1)] for it in T]).reshape(len(T), km)
        W = np.stack([pv[t] for t in range(1, km + 1)], 1) == V; ws, cws, nn = {}, {}, {}
        for t in range(1, km + 1):
            ws[str(t)] = mean_or_none(W[:, t - 1])
            if t == 1:
                cws["1"] = ws["1"]; nn["1"] = len(T)
            else:
                c_ = W[:, t - 2]; cws[str(t)] = mean_or_none(W[c_, t - 1]); nn[str(t)] = int(c_.sum())
        out["by_m"][str(mm)] = {"ws": ws, "cws": cws, "n": nn, "n_items": len(T), "kmax": km}
        out["items"][str(mm)] = {"sig": [it["sig"] for it in T], "pred_by_pass": {str(t): pv[t].tolist() for t in pv}}
    out["def"] = ("ws_so (§3.6): pick at the pass-t ':' (InjectAW.colon_logits, argmax over people_ids) vs S_min(t, m), n 5 NUM m 4 / 8 / 12, t 1..max(think_k, m); "
                  "cws[t] = P(WS_t | WS_{t-1}) (n[t] conditioning items), cws[1] = ws[1]")
    return out


# ---------------------------------------------------------------------------------------------------- swap_so

def swap_build(n, mm, t, a):
    """so_words.swap_pairs (§3.7 keep rule; cap 1,000, N_SMOKE caps it) -> tokenised (R, Dn) and the pair records; token lengths asserted equal per pass."""
    cap = N_SMOKE if N_SMOKE else 1000; pairs, tries = SO.swap_pairs(n, mm, t, cap=cap)
    R = [SOT.tok_so(p_["rec"], cue_of(a)) for p_ in pairs]; Dn = [SOT.tok_so(p_["don"], cue_of(a)) for p_ in pairs]
    for r_, d_, p_ in zip(R, Dn, pairs):
        assert [len(q) for q in r_["think_cues"]] == [len(q) for q in d_["think_cues"]] and len(r_["dec"]) == len(d_["dec"]), "§3.7: rec / don token lengths differ"
        assert p_["cf"] not in (p_["rec"]["answer"], p_["don"]["answer"], p_["don"]["values"][t]), "§3.7 keep rule"
        if cue_of(a) == "index":
            assert all(SO.cue_si(p_["rec"], s) == SO.cue_si(p_["don"], s) for s in range(1, mm + 1)), "§3.7: SI cues equal"
    return R, Dn, pairs, tries


@torch.no_grad()
def part_swap_so(m, g, sets, a, dev, bs):
    if m is None:
        return dict(NA)
    K = a["think_k"]; out = {"by_cell": {}, "items": {}}
    for (n, mm, t) in SWAP_CELLS:
        c = "%d_%d_%d" % (n, mm, t); R, Dn, pairs, tries = swap_build(n, mm, t, a); km = max(K, mm); ks = sorted({K, km})
        if not pairs:
            out["by_cell"][c] = {"n_pairs": 0, "n_tries": tries, "t": t, "m": mm, "follow_cf": None, "mean_floor": None, "note": "no kept pair = VOID cell"}; continue   # Audit LOW (c)
        own = np.array([vi(p_["rec"]["answer"]) for p_ in pairs]); dfin = np.array([vi(p_["don"]["answer"]) for p_ in pairs])
        stale = np.array([vi(p_["don"]["values"][t]) for p_ in pairs]); cf = np.array([vi(p_["cf"]) for p_ in pairs])
        ST_ = grab_states_so(m, Dn, dev, bs, (t,))[t]; mt = ST_.mean(0)
        I, _ = passes_so(m, R, dev, bs, set(ks))
        M_, _ = passes_so(m, R, dev, bs, set(ks), edit=lambda t_, S, sl: ST_[sl].to(dev, S.dtype) if t_ == t else None)
        F_, _ = passes_so(m, R, dev, bs, set(ks), edit=lambda t_, S, sl: mt.to(dev, S.dtype)[None].expand(S.shape[0], *mt.shape).contiguous() if t_ == t else None)

        def row(k):
            return {"follow_cf": float((M_[k] == cf).mean()), "stay_own": float((M_[k] == own).mean()), "follow_dfin": float((M_[k] == dfin).mean()),
                    "follow_stale": float((M_[k] == stale).mean()), "mean_floor": float((F_[k] == cf).mean()), "mean_stay_own": float((F_[k] == own).mean()),
                    "intact_acc": float((I[k] == own).mean())}
        out["by_cell"][c] = {"n_pairs": len(pairs), "n_tries": tries, "t": t, "m": mm, "k_main": km, **row(km), "by_kprime": {str(k): row(k) for k in ks}}
        out["items"][c] = {"rec_sig": [p_["rec"]["sig"] for p_ in pairs], "don_sig": [p_["don"]["sig"] for p_ in pairs], "own": own.tolist(), "dfin": dfin.tolist(),
                           "stale": stale.tolist(), "cf": cf.tolist(), "swap": M_[km].tolist(), "mean": F_[km].tolist(), "intact": I[km].tolist()}
    out["def"] = ("SWAP-MID (§3.7): recipient's S after pass t := the donor's S_t (donor = same people / objects / h_0 / o* / frame / offset, redrawn swaps; "
                  "so_words.swap_pairs), the recipient's own passes t+1..K'; cf = rec's swaps t+1..m applied to don's S_t, kept iff cf not in {own, dfin, stale}; "
                  "mean_floor = follow_cf with S_t := the mean of the donors' S_t; top-level fields at K' main = max(think_k, m)")
    return out


# ---------------------------------------------------------------------------------------------------- kcurve_so

def snorm(chunks):
    """{t: [per-batch per-item ||S_t||_F]} -> {mean[t], r_median[t] = the median over items of ||S_t||_F / ||S_4||_F}."""
    N = {t: torch.cat(v) for t, v in chunks.items()}; ref = N[4].clamp_min(1e-30) if 4 in N else None
    return {"mean": {str(t): float(N[t].mean()) for t in sorted(N)},
            "r_median": {str(t): float((N[t] / ref).median()) for t in sorted(N)} if ref is not None else {}}


@torch.no_grad()
def part_kcurve_so(m, g, sets, a, dev, bs):
    T = toks(sets["test"][(5, 8, "num")], a); y = np.array([it["ans"] for it in T]); mv = np.array([it["raw"]["moved"] for it in T], bool)
    if m is None:
        P = direct_pick(g, T, dev, bs); out = {"arm": "direct", "acc_moved": {"0": mean_or_none((P == y)[mv])}, "acc": {"0": mean_or_none(P == y)},
                                               "n_items": len(T), "n_moved": int(mv.sum()), "by_m": {}}
        for mm in KC_EXTRA_MS:
            T2 = toks(sets["test"][(5, mm, "num")], a); y2 = np.array([it["ans"] for it in T2]); mv2 = np.array([it["raw"]["moved"] for it in T2], bool); P2 = direct_pick(g, T2, dev, bs)
            out["by_m"][str(mm)] = {"acc": {"0": mean_or_none(P2 == y2)}, "acc_moved": {"0": mean_or_none((P2 == y2)[mv2])}, "n_items": len(T2), "n_moved": int(mv2.sum())}
        out["def"] = "kcurve_so (direct): the one pass = the K' 0 row only"
        return out
    nrm = {}

    def grab_norm(t, S, sl):
        nrm.setdefault(t, []).append(AW.state_n(S).cpu())
    P, _ = passes_so(m, T, dev, bs, set(KC_KS), on_state=grab_norm)
    vk = {k: np.array([vi(it["raw"]["values"][min(k, 8)]) for it in T]) for k in KC_KS}
    out = {"acc_moved": {str(k): mean_or_none((P[k] == y)[mv]) for k in KC_KS}, "acc": {str(k): mean_or_none(P[k] == y) for k in KC_KS},
           "acc_vs_sk": {str(k): mean_or_none(P[k] == vk[k]) for k in KC_KS}, "n_items": len(T), "n_moved": int(mv.sum()),
           "s_norm": {"8": snorm(nrm)}, "by_m": {}, "items": {"sig": [it["sig"] for it in T], "pred": {str(k): P[k].tolist() for k in KC_KS}}}
    for mm in KC_EXTRA_MS:
        T2 = toks(sets["test"][(5, mm, "num")], a); y2 = np.array([it["ans"] for it in T2]); mv2 = np.array([it["raw"]["moved"] for it in T2], bool); ns2 = {}

        def grab2(t, S, sl):
            ns2.setdefault(t, []).append(AW.state_n(S).cpu())
        P2, _ = passes_so(m, T2, dev, bs, set(KC_EXTRA_KS), on_state=grab2)
        out["by_m"][str(mm)] = {"acc": {str(k): mean_or_none(P2[k] == y2) for k in KC_EXTRA_KS}, "acc_moved": {str(k): mean_or_none((P2[k] == y2)[mv2]) for k in KC_EXTRA_KS},
                                "n_items": len(T2), "n_moved": int(mv2.sum())}
        out["s_norm"][str(mm)] = snorm(ns2)
    T4 = toks(sets["test"][(5, 4, "num")], a); s2 = np.array([vi(it["raw"]["values"][2]) for it in T4]); s4 = np.array([it["ans"] for it in T4]); st = s2 != s4
    T4s = [it for it, k in zip(T4, st) if k]
    if T4s:
        P4, _ = passes_so(m, T4s, dev, bs, {2}); out["not_decoder"] = {"n_stratum": len(T4s), "acc_vs_s2": float((P4[2] == s2[st]).mean()), "acc_vs_s4": float((P4[2] == s4[st]).mean()), "n": len(T4s),
                              "n_cell": len(T4), "kprime": 2}
    else:
        out["not_decoder"] = {"n_stratum": 0, "n": 0, "n_cell": len(T4), "acc_vs_s2": None, "acc_vs_s4": None, "kprime": 2}
    out["def"] = ("kcurve_so (§3.6, §6.2 KC / C2-K / M4, §6.4): n 5 m 8 NUM, K' 0 = the decoder on the zero state; acc_vs_sk = P(pred == S_min(K', 8)); s_norm[t] = mean "
                  "||S_t||_F; by_m 10 / 12 at K' 8 / 12; not_decoder = n 5 m 4 NUM at K' 2 on the S_2 != S_4 stratum")
    return out


# ---------------------------------------------------------------------------------------------------- long_so

@torch.no_grad()
def part_long_so(m, g, sets, a, dev, bs):
    if m is None:
        return dict(NA)
    out = {"by_m": {}}
    for mm in LONG_MS:
        T = toks(sets["test"][(5, mm, "num")], a); y = np.array([it["ans"] for it in T]); sm1 = np.array([vi(it["raw"]["values"][-2]) for it in T]); sk = sm1 != y
        P, _ = passes_so(m, T, dev, bs, set(LONG_KS))
        out["by_m"][str(mm)] = {"acc": {str(k): mean_or_none(P[k] == y) for k in LONG_KS}, "sm1_confusion": {str(k): mean_or_none((P[k] == sm1)[sk]) for k in LONG_KS},
                                "n": len(T), "n_items": len(T), "n_sm1": int(sk.sum())}
    out["def"] = "long_so (§6.2 H1, M-5): n 5 NUM m 1-4 at K' 8 / 12 / 63 / 64; sm1_confusion = P(pred == S_{m-1} | S_{m-1} != S_m)"
    return out


# ---------------------------------------------------------------------------------------------------- perm_label

def perm_order(m_, i):
    r = np.random.default_rng((SO.SEED_PERM, m_, i)); k = 0
    while True:
        o = [int(x) for x in r.permutation(m_)]
        if o != list(range(m_)):
            return o, k
        k += 1


def permuted_item(it, order):
    """the item with its swap sentences printed in `order` (labels = the true index); solve() still == answer (asserted)."""
    p = dict(it); p["text"] = SO.render(it, order); p["sig"] = hashlib.sha1(p["text"].encode()).hexdigest(); p["perm_order"] = list(order)
    assert SO.solve(p["text"]) == p["answer"], "PERM-LABEL: solve != answer"; return p


@torch.no_grad()
def part_perm_label(m, g, sets, a, dev, bs):
    out = {"by_m": {}, "items": {}}
    for mm in PERM_MS:
        base = trunc(sets["test"][(5, mm, "num")]); ords = [perm_order(mm, i) for i in range(len(base))]
        Tp = [SOT.tok_so(permuted_item(it, o), cue_of(a)) for it, (o, _) in zip(base, ords)]; Tc = [SOT.tok_so(it, cue_of(a)) for it in base]
        y = np.array([it["ans"] for it in Tp])
        if m is None:
            Pp, Pc = direct_pick(g, Tp, dev, bs), direct_pick(g, Tc, dev, bs)
        else:
            km = max(a["think_k"], mm); Pp = passes_so(m, Tp, dev, bs, {km})[0][km]; Pc = passes_so(m, Tc, dev, bs, {km})[0][km]
        out["by_m"][str(mm)] = {"acc": mean_or_none(Pp == y), "acc_chain": mean_or_none(Pc == y), "n": len(Tp), "n_items": len(Tp), "n_identity_redraws": int(sum(k for _, k in ords))}
        out["items"][str(mm)] = {"sig": [it["sig"] for it in Tp], "order": [o for o, _ in ords], "pred": Pp.tolist(), "ans": y.tolist()}
    out["def"] = ("perm_label (§3.6, F10): NUM n 5 m 2-4 TEST items re-rendered with the swap sentences in a non-identity order (default_rng((SEED_PERM, m, i)), redrawn "
                  "while identity), labels = the true index; loop K' max(think_k, m); acc_chain = the same items in chain order")
    return out


# ---------------------------------------------------------------------------------------------------- lookup

@torch.no_grad()
def part_lookup(m, g, sets, a, dev, bs):
    base = trunc(sets["test"][(5, LOOKUP_M, "num")]); T0 = [SOT.tok_so(it, cue_of(a)) for it in base]; out = {"by_L": {}, "items": {"sig": [it["sig"] for it in T0], "pred": {}}}
    for L in range(1, LOOKUP_M + 1):
        TL = []
        for it, t0 in zip(base, T0):
            aL, bL = it["swaps"][L - 1]; q = [EOT] + ENC.encode(" " + it["text"] + " In swap %d, who trades with %s?" % (SO.label(it, L), aL)) + SFX
            TL.append({**t0, "dec": q, "ans": vi(bL)})
        y = np.array([x["ans"] for x in TL])
        if m is None:
            P = direct_pick(g, TL, dev, bs)
        else:
            P = np.concatenate([SOT.pick(m.dec_logits(m.mem.zero_state(len(TL[i:i + bs]), dev), TL[i:i + bs], dev), TL[i:i + bs]) for i in range(0, len(TL), bs)])
        out["by_L"][str(L)] = {"acc": mean_or_none(P == y), "n": len(TL)}; out["items"]["pred"][str(L)] = P.tolist()
    out["def"] = ("lookup (§3.6, F3): n 5 m 12 NUM TEST items; input [EOT] + enc(' ' + text + ' In swap L, who trades with a_L?') + ' The answer is:' -> b_L; loop: the "
                  "decoder on the zero state (K' 0 read); direct: the one pass")
    return out


# ---------------------------------------------------------------------------------------------------- bbh

BBH_FINAL = re.compile(r"At the end of the [^,]+, (\w+) (.+)$")
BBH_SWAP = re.compile(r"(?:First|Then|Finally), (\w+) and (\w+) (?:swap|switch|trade)\b")
BBH_RULE = ("BBH rendering (N1_v1 §3.5, implemented): input split at '\\nOptions:'; the options are the '(X) text' lines; the body's last 'At the end of the <..>, "
            "X <rest>' sentence is removed and parsed (X = the asked person, rest = BBH's own verb phrase: 'has the' / 'has' / 'is dancing with' / 'is playing'); "
            "newlines in the remaining text -> one space; per option o_j the pass-t input = [EOT] + enc(' ' + text + ' After swap L(min(t, m)), who ' + rest + ' ' + o_j "
            "+ '?') + ' The answer is:' (ORD labels L(s) = s; m = the parsed swap-sentence count); loop: the pass-m ':' logit of X's first BPE token (K' = m); direct: one "
            "pass on [EOT] + enc(' ' + text + ' After swap m, who ' + rest + ' ' + o_j + '?') + ' The answer is:', the same logit; answer = argmax_j (np.argmax: exact ties "
            "-> the first option); unparsed items count wrong")


def bbh_parse(inp, target):
    """-> dict {text, X, rest, options, gold, m, n, people, swaps, consistent} or None. Parsing = people (the first clause before ' are '), the holding clauses
    (each person's clause ends with exactly one option), the swap sentences in order, the final question; consistent = simulating the swaps gives gold."""
    try:
        body, ob = inp.split("\nOptions:\n")
        opts = [re.match(r"\(([A-G])\) (.+)$", ln).group(2).strip() for ln in ob.strip().split("\n")]
        k = body.rindex("At the end of the"); pre, fin = body[:k].strip(), body[k:].strip(); mf = BBH_FINAL.match(fin)
        X, rest = mf.group(1), mf.group(2).strip()
        head = pre.split(" are ")[0]; people = [p.strip() for p in re.split(r",\s*(?:and\s+)?|\s+and\s+", head) if p.strip()]
        seg = pre[pre.index(": ") + 2:pre.index("\n")].rstrip(".")
        starts = []; pos = 0
        for p in people:
            mm = re.compile(r"\b" + re.escape(p) + r" ").search(seg, pos); starts.append(mm.start()); pos = mm.end()
        hold = {}
        for j, p in enumerate(people):
            cl = seg[starts[j]:starts[j + 1] if j + 1 < len(people) else len(seg)].rstrip(" ,").rstrip()
            cl = re.sub(r",?\s*and$", "", cl).rstrip(" ,")
            hit = [o for o in opts if cl.endswith(" " + o)]; hit = sorted(hit, key=len)[-1:]
            assert len(hit) == 1, ("holding clause", cl); hold[p] = hit[0]
        assert sorted(hold.values()) == sorted(opts) and X in people
        swaps = BBH_SWAP.findall(pre); assert swaps and all(a_ in people and b_ in people for a_, b_ in swaps)
        h = dict(hold)
        for a_, b_ in swaps:
            h[a_], h[b_] = h[b_], h[a_]
        gold = "ABCDEFG".index(target.strip("()")); text = re.sub(r"\s*\n\s*", " ", pre)
        return {"text": text, "X": X, "rest": rest, "options": opts, "gold": gold, "m": len(swaps), "n": len(people), "people": people, "swaps": swaps,
                "consistent": h[X] == opts[gold]}
    except (AssertionError, AttributeError, ValueError, IndexError):
        return None


def bbh_dir(a):
    return a.get("bbh_dir") or os.environ.get("SO_BBH_DIR") or BBH_DIR_DEFAULT


def bbh_load(d):
    """-> ({subset: [(raw example, parsed or None)]}, {file: sha256}); shas checked against bbh_sha.json when present."""
    shas = {}; out = {}; ref = os.path.join(d, "bbh_sha.json"); assert os.path.exists(ref), ("Audit MED-2: bbh_sha.json missing", d)
    ref = json.load(open(ref, encoding="utf-8"))["sha256"]
    for s in BBH_SUBSETS:
        f = "tracking_shuffled_objects_%s_objects.json" % s; p = os.path.join(d, f); shas[f] = sha_file(p)
        assert ref.get(f) == shas[f], ("BBH sha mismatch vs bbh_sha.json", f)
        ex = json.load(open(p, encoding="utf-8"))["examples"]; out[s] = [(e, bbh_parse(e["input"], e["target"])) for e in ex]
    return out, shas


def bbh_check(d=BBH_DIR_DEFAULT):
    """G0 helper: 750 / 750 parse, every parse consistent with the gold, m == n per subset, and the X first-token / option-question token rules. -> summary dict."""
    data, shas = bbh_load(d); summ = {"shas": shas, "by_subset": {}}
    for s, rows in data.items():
        ok = [p for _, p in rows if p is not None]; cons = sum(p["consistent"] for p in ok)
        summ["by_subset"][s] = {"n": len(rows), "n_parsed": len(ok), "n_consistent": cons, "ms": sorted({p["m"] for p in ok}), "ns": sorted({p["n"] for p in ok}),
                                "rests": sorted({p["rest"] for p in ok})}
        assert len(ok) == len(rows) == 250 and cons == 250 and all(p["m"] == BBH_N[s] == p["n"] for p in ok), ("BBH parse", s, summ["by_subset"][s])
    first = {}
    for rows in data.values():
        for _, p in rows:
            for q in p["people"]:
                first.setdefault(ENC.encode(" " + q)[0], set()).add(q)
    assert all(len(v) == 1 for v in first.values()), ("BBH: first BPE tokens not distinct", first)
    summ["n_total"] = sum(v["n"] for v in summ["by_subset"].values()); summ["n_parsed_total"] = sum(v["n_parsed"] for v in summ["by_subset"].values()); return summ


def bbh_question(p, L, oj):
    return " After swap %d, who %s %s?" % (L, p["rest"], oj)


def bbh_rows(p):
    """per option: the loop item (think_cues t = 1..m, cue label min(t, m) = t) and the direct dec."""
    rows = []
    for oj in p["options"]:
        cues = [[EOT] + ENC.encode(" " + p["text"] + bbh_question(p, t, oj)) + SFX for t in range(1, p["m"] + 1)]
        assert len({len(q) for q in cues}) == 1 and all(q[-1] == SFX[-1] for q in cues), "BBH: one padding across passes"
        rows.append({"think_cues": cues, "dec": cues[-1], "xtok": ENC.encode(" " + p["X"])[0], "m": p["m"]})
    return rows


@torch.no_grad()
def part_bbh(m, g, sets, a, dev, bs):
    data, shas = bbh_load(bbh_dir(a)); out = {"subsets": {}, "rule": BBH_RULE, "bbh_sha256": shas}
    for s in BBH_SUBSETS:
        rows = trunc(data[s]); flat = []; owner = []
        for i, (_, p) in enumerate(rows):
            if p is not None:
                for j, r_ in enumerate(bbh_rows(p)):
                    flat.append(r_); owner.append((i, j))
        sc = np.full(len(flat), np.nan)
        if flat:
            if m is None:
                for i in range(0, len(flat), bs):
                    ch = flat[i:i + bs]; lg = IO.direct_logits(g, ch, dev, AE.NULLC)
                    sc[i:i + len(ch)] = lg[torch.arange(len(ch), device=lg.device), torch.tensor([r_["xtok"] for r_ in ch], device=lg.device)].cpu().numpy()
            else:
                for mm in sorted({r_["m"] for r_ in flat}):
                    idx = [k for k, r_ in enumerate(flat) if r_["m"] == mm]; sub = [flat[k] for k in idx]; got = np.full(len(sub), np.nan)

                    def read(t, lg, ch, sl):
                        if t == mm:
                            got[sl] = lg[torch.arange(len(ch), device=lg.device), torch.tensor([r_["xtok"] for r_ in ch], device=lg.device)].cpu().numpy()
                    colon_passes(m, sub, dev, bs, mm, read); sc[idx] = got
        per = []; nc = 0
        for i, (e, p) in enumerate(rows):
            if p is None:
                per.append({"idx": i, "pred": -1, "gold": "ABCDEFG".index(e["target"].strip("()")), "m": None, "X": None, "parsed": False, "correct": False}); continue
            v = np.array([sc[k] for k, (ii, _) in enumerate(owner) if ii == i]); pr = int(np.argmax(v)); nc += int(pr == p["gold"])
            per.append({"idx": i, "pred": pr, "gold": p["gold"], "m": p["m"], "X": p["X"], "parsed": True, "correct": bool(pr == p["gold"])})
        rec = {"acc": nc / len(rows) if rows else None, "n": len(rows), "n_parsed": int(sum(x["parsed"] for x in per)), "m": BBH_N[s], "per_item": per}
        out["subsets"][s] = rec
    return out


# ---------------------------------------------------------------------------------------------------- beta_so

@torch.no_grad()
def part_beta_so(m, g, sets, a, dev, bs):
    if m is None:
        return dict(NA)
    mem = m.mem; H = mem.h_del; rec = []; K = a["think_k"]; dmax = [0.0]

    def hook(mod, inp, outp):                                               # the ':' write input u (write_seq on the gathered ':' row) and gate(u)
        u = inp[0].detach().float(); be = 2 * torch.sigmoid(Fn.linear(u, mod.weight.float(), mod.bias.float())[..., H:])
        dmax[0] = max(dmax[0], float((be - 2 * torch.sigmoid(outp.detach().float()[..., H:])).abs().max())); rec.append(be.reshape(be.shape[0], -1, H)[:, -1])
    hd = mem.gate.register_forward_hook(hook)
    try:
        allw, allh, by_m, by_pass, by_head = [], [], {}, {}, []
        for mm in BETA_MS:
            T = toks(sets["test"][(5, mm, "num")], a); km = max(K, mm); per = {}

            def grab(t, S, sl):
                assert len(rec) == 1, ("beta_so: exactly one gate call per pass", len(rec)); per.setdefault(t, []).append(rec.pop().cpu())
            rec.clear(); passes_so(m, T, dev, bs, {km}, on_state=grab); rec.clear()
            B = {t: torch.cat(v) for t, v in per.items()}                   # (N, H) per pass
            w = torch.stack([B[t] for t in range(1, mm + 1)], 1); allw.append(w.reshape(-1, H))
            if km > mm:
                allh.append(torch.stack([B[t] for t in range(mm + 1, km + 1)], 1).reshape(-1, H))
            by_m[str(mm)] = {"share_gt1": float((w > 1).float().mean()), "mean_beta": float(w.mean())}
            for t in range(1, mm + 1):
                bp = by_pass.setdefault(str(t), [0.0, 0]); bp[0] += float((B[t] > 1).float().sum()); bp[1] += B[t].numel()
        W = torch.cat(allw); Hh = torch.cat(allh) if allh else None
        out = {"share_gt1": float((W > 1).float().mean()), "share_gt1_hold": float((Hh > 1).float().mean()) if Hh is not None else None, "mean_beta": float(W.mean()),
               "by_m": by_m, "by_pass": {t: {"share_gt1": v[0] / v[1], "n": v[1]} for t, v in by_pass.items()},
               "by_head": {str(h): float((W[:, h] > 1).float().mean()) for h in range(H)}, "recompute_maxabs": dmax[0],
               "def": ("beta_so (§3.6, E1): beta = 2 sigmoid(gate(u))[delta heads], recomputed from the ':' write input u captured by a forward hook on mem.gate "
                       "(deltamem.py untouched); n 5 NUM m 5-12, passes t <= m (share_gt1) and t > m (share_gt1_hold)")}
    finally:
        hd.remove()
    assert out["recompute_maxabs"] <= 1e-5, ("beta recompute", out["recompute_maxabs"]); return out


# ---------------------------------------------------------------------------------------------------- floors

FLOORS_REL = os.path.join("n1_floors", "floors_so.json")                  # score_n1.FLOORS_REL: <runs>/n1_floors/floors_so.json (Theory writes it at G0)


def floors_path(a, sets, out):
    """the frozen floors file: sets['floors_path'] > args 'so_floors' > <runs>/n1_floors/floors_so.json (runs = the run dir's parent). None if absent."""
    p = (sets or {}).get("floors_path") or a.get("so_floors") or os.path.join(os.path.dirname(os.path.abspath(out)), FLOORS_REL)
    p = p if os.path.isabs(p) else os.path.join(ROOT, p)
    return p if os.path.exists(p) else None


def load_floors(a, sets, out):
    """READ-ONLY (Note 271-2: Theory computes the floors at G0 with docs/research_n1_2026-11-04/n1_oc.py; nothing is computed here).
    -> (floors dict floors/<n>/<m>/<mode>/{f, floor_hold}, path, sha256, set_sha) or Nones; args 'so_floors_sha' (prefix) asserted when given."""
    p = floors_path(a, sets, out)
    if p is None:
        return None, None, None, None
    sha = sha_file(p); want = a.get("so_floors_sha"); assert not want or sha.startswith(want), ("floors sha", sha, want)
    d = json.load(open(p, encoding="utf-8")); return d.get("floors"), p, sha, d.get("set_sha")


@torch.no_grad()
def part_floors(m, g, sets, a, dev, bs):
    """a read-only echo of the frozen floors file (path, sha256, set_sha vs this eval's TEST set sha); status 'absent' when the file does not exist yet."""
    p, sha, fs = sets.get("floors_meta", (None, None, None))
    if p is None:
        return {"status": "absent", "path": None, "sha256": None, "set_sha": None, "set_sha_match": None,
                "def": "floors (§6.1, Note 271-2): read-only; the frozen file <runs>/n1_floors/floors_so.json was not found; nothing is computed here"}
    return {"status": "loaded", "path": p, "sha256": sha, "set_sha": fs, "set_sha_match": (fs == sets["shas"]["so_test_sha"]) if fs else None,
            "def": "floors (§6.1, Note 271-2): read-only echo of the frozen floors file; acc_so cells carry its floors/<n>/<m>/<mode>/f as 'f'"}


# ---------------------------------------------------------------------------------------------------- run

def load(run, dev):
    """inject_aw_eval.load (model.pt holds every weight; direct: the trunk)."""
    return AE.load(run, dev)


def run(out, parts, dev, bs, model=None, sets=None, train=None):
    a = json.load(open(os.path.join(out, "args.json"), encoding="utf-8"))
    if model is None:
        a, m, g = load(out, dev)
    else:
        m, g = model; (m if m is not None else g).eval()
    if a["arm"] == "direct":
        m = None
    if sets is None:
        sets = build_sets()
    if "shas" not in sets:
        sets = {**sets, "shas": set_shas(sets)}
    for k_, v_ in sets["shas"].items():
        assert a.get(k_) in (None, v_), ("§271: set sha mismatch vs args.json", k_)
    if a.get("so_set_shas") and not N_SMOKE:                                # inject_so.py's per-cell shas (args.json so_set_shas): every cell re-checked
        mine = {**{"test|%d|%d|%s" % k_: SO.set_sha(v_) for k_, v_ in sorted(sets["test"].items())},
                **{"transfer|%d" % k_: SO.set_sha(v_) for k_, v_ in sorted(sets["transfer"].items())}, "offset": SO.set_sha(sets["offset"])}
        assert all(mine.get(k_) == v_ for k_, v_ in a["so_set_shas"].items()), "§271: per-cell set sha mismatch vs args.json so_set_shas"
    f_, fp_, fsha_, fset_ = load_floors(a, sets, out)
    sets = {**sets, "floors": f_, "floors_meta": (fp_, fsha_, fset_)}
    parts = [p_ for p_ in parts if p_ in PARTS]
    dstr = str(dev)
    meta = {"eval_dtype": "fp32", "eval_gpu": torch.cuda.get_device_name(dev) if dstr.startswith("cuda") else "cpu", "eval_bs": bs, "arm": a["arm"],
            "cue": a.get("cue", "none"), "think_k": a.get("think_k", 0), "module_sha256": sha_file(os.path.abspath(__file__)),
            "so_words_sha256": sha_file(SO.__file__), "so_tok_sha256": sha_file(SOT.__file__), "n_smoke": N_SMOKE, "set_shas": sets["shas"]}
    fns = {"acc_so": part_acc_so, "ws_so": part_ws_so, "swap_so": part_swap_so, "kcurve_so": part_kcurve_so, "long_so": part_long_so,
           "perm_label": part_perm_label, "lookup": part_lookup, "bbh": part_bbh, "beta_so": part_beta_so, "floors": part_floors}
    for p_ in parts:
        t1 = time.time()
        if dstr.startswith("cuda"):
            torch.cuda.empty_cache()
        r_ = fns[p_](m, g, sets, a, dev, bs)
        r_.update(meta); r_["sec"] = round(time.time() - t1)
        with open(os.path.join(out, "eval_%s.json" % p_), "w", encoding="utf-8", newline="") as fh:
            json.dump(r_, fh)
        print("PART", p_, "done", r_["sec"], "s", flush=True)
    summ = {"arm": a["arm"], "cue": a.get("cue", "none"), "think_k": a.get("think_k", 0), "seed": a.get("seed"), "parts": {}, **meta}
    tf = os.path.join(out, "train_summary.json"); summ["train"] = train if train is not None else (json.load(open(tf, encoding="utf-8")) if os.path.exists(tf) else None)
    for p_ in PARTS:
        f2 = os.path.join(out, "eval_%s.json" % p_)
        if os.path.exists(f2):
            summ["parts"][p_] = {"file": "eval_%s.json" % p_, "sec": json.load(open(f2, encoding="utf-8")).get("sec")}
    with open(os.path.join(out, "eval.json"), "w", encoding="utf-8", newline="") as fh:
        json.dump(summ, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--parts", default=",".join(PARTS)); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval_bs", type=int, default=100); ap.add_argument("--out", default="", help="queue runners append --out; ignored (outputs go to --run)")
    ap.add_argument("--bbh_check", action="store_true", help="G0: parse the 750 BBH items (asserts 750 / 750) and print the summary; no model")
    a = ap.parse_args()
    if a.bbh_check:
        print(json.dumps(bbh_check(os.environ.get("SO_BBH_DIR") or BBH_DIR_DEFAULT), indent=1)); return
    run(a.run, [p_ for p_ in a.parts.split(",") if p_], a.device, a.eval_bs); print("DONE", a.run, flush=True)


if __name__ == "__main__":
    main()
