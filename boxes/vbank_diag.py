"""VB-2G Stage 0 (user-approved 2026-10-16 12:32 via Audit) eval-only diagnostics on the §211 cells: notes 211-15 (B: RS / PF) and 211-16 (REPAIR,
STEP rows, history arms, FOREIGN); definitions in THEORY notes 211-13 / 15 / 16 and docs/research_bottleneck_2026-10-16 (D §1.2, S §1.3, REDTEAM2).
Local fp32 (no autocast), --eval_bs 32 (note 211-7), the §170 test set (1,000 per n), card / bs / dtype recorded per output file. NEW module: imports
boxes.vbank / boxes.vbank_eval read-only and changes no interface. One output per part:
  rs   -> <run>/eval_rs.json   RESTART-SPLIT (D §1.2). Decode argmax R(V_t), VOIDs dropped; exact iff == cot_steps[t]. For n 1..16 x r in {2, 4, 6, 8}
       (t = n - r >= 1): INTACT-CONT(n, r) = P(correct at K' = n + 2 | exact at t); RESTART(n, r) = P(correct | the item reduced t times, run as a fresh
       problem at K' = r + 2 | exact at t) - the fresh problem is the ORIGINAL tree reduced t times, which renders identically when the decode is exact
       and keeps the remaining reduction order. Non-exact decodes split into parseable-wrong / unparseable (nested170.eval_tokens). LONG-STEP(m): every
       test item's chain re-rooted after every pass (1-pass rollouts on fresh inputs), stopping at the first inexact step; LONG-STEP(m) = exact /
       attempted single steps on fresh inputs with m operators. Readings (HISTORY-KEYED / HISTORY-FREE / HEAD-LIMITED) are Theory's; the counts are here.
    python -m boxes.vbank_diag --run boxes/runs_fsq/s211_vb_pa_s1 --part rs --device cuda:0"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402
import boxes.vbank_eval as VE  # noqa: E402

RS_R = (2, 4, 6, 8)


def n_ops(nd):
    return 0 if nd.op is None else 1 + n_ops(nd.l) + n_ops(nd.r)


def reduce_k(root, k):
    """a copy of root with its first k leftmost-ready reductions applied (cot_steps order)."""
    nd = N._copy(root)
    for _ in range(k):
        if nd.op is None:
            break
        path = N._leftmost_ready(nd); h = nd
        for p_ in path:
            h = getattr(h, p_)
        h.v = N.apply(h.op, h.l.v, h.r.v); h.op = h.l = h.r = None
    return nd


def classify(dec_syms, target_syms):
    """'exact' | 'wrong' (parses) | 'unparseable'."""
    if dec_syms == target_syms:
        return "exact"
    try:
        N.eval_tokens(dec_syms); return "wrong"
    except Exception:                                                       # noqa: BLE001  (any malformed decode)
        return "unparseable"


@torch.no_grad()
def part_rs(m, TE, dev, bs, do_long=True):
    num = torch.tensor(N.NUM_IDS, device=dev); out = {"cells": {}, "decode": {}}
    Nn = np.array([it["n"] for it in TE]); jobs = {r: [] for r in RS_R}       # restart jobs per r: (item index, t)
    intact_ok = np.zeros(len(TE), dtype=bool); exact = {}                   # exact[(i, t)] = bool
    for n in range(1, 17):
        for ch in VE.chunks(np.nonzero(Nn == n)[0], bs):
            its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev)
            r_ = m.rollout(x, pad, n + 2, coda_rows={n + 2: torch.arange(len(its), device=dev)})
            intact_ok[ch] = (m.answer_logits(r_["coda"][n + 2][1], lens)[:, num].argmax(-1).cpu().numpy() == np.array([it["t"] for it in its]))
            for r in RS_R:
                t = n - r
                if t < 1:
                    continue
                dec = m.rec_logits(r_["V"][t]).argmax(-1).cpu().numpy()
                for k_, j in enumerate(ch):
                    L = int(lens[k_]); ds = VB.rerender(list(dec[k_, :L])); tg = N.cot_steps(TE[j]["root"])[t]; c = classify(ds, tg)
                    d_ = out["decode"].setdefault("%d,%d" % (n, r), {"exact": 0, "wrong": 0, "unparseable": 0}); d_[c] += 1
                    if c == "exact":
                        exact[(j, t)] = True; jobs[r].append((j, t))
            del r_
    rest_ok = {}
    for r in RS_R:                                                          # fresh problems: the item reduced t times, K' = r + 2
        js = jobs[r]
        for c0 in range(0, len(js), bs):
            part = js[c0:c0 + bs]; its = [{"syms": N.render(reduce_k(TE[j]["root"], t)), "t": TE[j]["t"]} for j, t in part]
            assert all(len(it["syms"]) and N.eval_tokens(it["syms"]) == it["t"] for it in its), "RS: a re-rooted item's value changed"
            x, pad, lens = VB.batch_x(its, dev); rr = m.rollout(x, pad, r + 2, coda_rows={r + 2: torch.arange(len(its), device=dev)})
            a_ = m.answer_logits(rr["coda"][r + 2][1], lens)[:, num].argmax(-1).cpu().numpy(); del rr
            for k_, (j, t) in enumerate(part):
                rest_ok[(j, t)] = bool(a_[k_] == TE[j]["t"])
    for n in sorted({int(v) for v in Nn}):                                  # every n 1..16 on the full test set
        for r in RS_R:
            t = n - r
            if t < 1:
                continue
            js = [j for (j, tt) in jobs[r] if tt == t and TE[j]["n"] == n]
            ic = float(np.mean([intact_ok[j] for j in js])) if js else None; rs = float(np.mean([rest_ok[(j, t)] for j in js])) if js else None
            dc = out["decode"]["%d,%d" % (n, r)]; tot = sum(dc.values())
            out["cells"]["%d_%d" % (n, r)] = {"n_items": len(js), "intact": ic, "restart": rs, "unparse_rate": dc["unparseable"] / tot if tot else None,   # Theory's contract
                                             "n": n, "r": r, "t": t, "delta": (rs - ic) if js else None, "wrong_rate": dc["wrong"] / tot if tot else None,
                                             "exact_rate": dc["exact"] / tot if tot else None, "intact_all": float(intact_ok[Nn == n].mean())}
    out["long_step"] = long_step(m, TE, dev, bs) if do_long else None
    out["def"] = ("D §1.2 / note 211-15 B: cells '<n>_<r>' (t = n - r >= 1): n_items = items exact at t; intact = acc at K' = n + 2 on them; restart = acc of the "
                  "item reduced t times run fresh at K' = r + 2; unparse_rate / wrong_rate / exact_rate over all items of n; n <= 8 = control; long_step '<m>': "
                  "n_steps attempted single steps on fresh inputs with m operators, exact = their exact rate (chains stop at the first inexact step)")
    return out


@torch.no_grad()
def long_step(m, TE, dev, bs, by_t=False):
    """LONG-STEP(m): chains re-rooted after every pass; one pass on a fresh input, R(V_1) decoded and compared with its next step; stop at the first
    inexact step. -> {m: {attempted, exact, rate}} over every test item's chain."""
    cur = {j: N._copy(it["root"]) for j, it in enumerate(TE)}; stats = {}; stats_t = {}
    while cur:
        idx = sorted(cur, key=lambda j: len(N.render(cur[j])))
        nxt = {}
        for c0 in range(0, len(idx), bs):
            part = idx[c0:c0 + bs]; its = [{"syms": N.render(cur[j])} for j in part]; x, pad, lens = VB.batch_x(its, dev)
            rr = m.rollout(x, pad, 1); dec = m.rec_logits(rr["V"][1]).argmax(-1).cpu().numpy(); del rr
            for k_, j in enumerate(part):
                mo = n_ops(cur[j]); red = reduce_k(cur[j], 1); ok = VB.rerender(list(dec[k_, :int(lens[k_])])) == N.render(red)
                s_ = stats.setdefault(str(mo), [0, 0]); s_[0] += 1; s_[1] += int(ok)
                s2 = stats_t.setdefault("%d_%d" % (mo, len(its[k_]["syms"]) + 2), [0, 0]); s2[0] += 1; s2[1] += int(ok)   # Theory 12:4x: by token length T too
                if ok and red.op is not None:
                    nxt[j] = red
        cur = nxt
    if by_t:
        return {k: {"n_steps": v[0], "exact": v[1] / v[0], "n_exact": v[1]} for k, v in sorted(stats_t.items(), key=lambda kv: tuple(int(z) for z in kv[0].split("_")))}
    return {k: {"n_steps": v[0], "exact": v[1] / v[0], "n_exact": v[1]} for k, v in sorted(stats.items(), key=lambda kv: int(kv[0]))}   # contract: exact = rate


def gen_value(rng, k, v):
    """nested170.generate's Alg. 1 with the root value fixed to v: a tree with k operators and value v."""
    root = N.Node(v=v); leaves = [root]
    for _ in range(k):
        i = int(rng.integers(0, len(leaves))); leaf = leaves.pop(i); o = N.OPS[int(rng.integers(0, 4))]; prs = N.PAIRS[(o, leaf.v)]; a_, b_ = prs[int(rng.integers(0, len(prs)))]
        leaf.op, leaf.l, leaf.r, leaf.v = o, N.Node(v=a_), N.Node(v=b_), None; leaves.insert(i, leaf.r); leaves.insert(i, leaf.l)
    return root


def pf_items(r, n_draws=20000, k_max=12, seed=21400):
    """D §1.2 PREFIX-FACTORIAL: Q = a size-r item whose render starts with a number v; its leftmost leaf v replaced by P (k operators, value v).
    -> list of {syms, t, root, k, r, s, q_syms}; asserted: the state after k reductions renders as Q, the value is unchanged."""
    rng = np.random.default_rng(seed + r); out = []
    while len(out) < n_draws:
        q, y = N.generate(rng, r); qs = N.render(q)
        if not qs[0].strip().isdigit():
            continue
        k = int(rng.integers(1, k_max + 1)); it = N._copy(q); h = it
        while h.op is not None:
            h = h.l
        P_ = gen_value(rng, k, h.v); h.v, h.op, h.l, h.r = P_.v, P_.op, P_.l, P_.r
        syms = N.render(it); assert N.render(reduce_k(it, k)) == qs and N.value(it) == y, "PF: the state at k is not Q"
        out.append({"syms": syms, "t": y, "root": it, "k": k, "r": r, "s": len(syms) - len(qs) + 1, "q_syms": qs})
    return out


@torch.no_grad()
def part_pf(m, TE, dev, bs, n_draws=20000):
    num = torch.tensor(N.NUM_IDS, device=dev); grid = {}; S0 = {}
    for r in (2, 3):
        items = pf_items(r, n_draws); fresh_ok = []
        for c0 in range(0, len(items), bs):                                 # S0: the Q items fresh at K' = r + 2
            its = [{"syms": it["q_syms"], "t": it["t"]} for it in items[c0:c0 + bs]]; x, pad, lens = VB.batch_x(its, dev)
            rr = m.rollout(x, pad, r + 2, coda_rows={r + 2: torch.arange(len(its), device=dev)})
            fresh_ok += list(m.answer_logits(rr["coda"][r + 2][1], lens)[:, num].argmax(-1).cpu().numpy() == np.array([it["t"] for it in its])); del rr
        S0[str(r)] = float(np.mean(fresh_ok))
        for k in range(1, 13):
            ks = [it for it in items if it["k"] == k]; K = k + r + 2
            for c0 in range(0, len(ks), bs):
                its = ks[c0:c0 + bs]; x, pad, lens = VB.batch_x(its, dev); rr = m.rollout(x, pad, K, coda_rows={K: torch.arange(len(its), device=dev)})
                ok = m.answer_logits(rr["coda"][K][1], lens)[:, num].argmax(-1).cpu().numpy() == np.array([it["t"] for it in its])
                dec = m.rec_logits(rr["V"][k]).argmax(-1).cpu().numpy(); del rr
                for i_, it in enumerate(its):
                    c_ = grid.setdefault("%d_%d_%d" % (k, it["s"], r), {"n_items": 0, "n_correct": 0, "n_draws": 0, "r": r, "k": k, "s": it["s"]}); c_["n_draws"] += 1
                    if VB.rerender(list(dec[i_, :int(lens[i_])])) == it["q_syms"]:
                        c_["n_items"] += 1; c_["n_correct"] += int(ok[i_])
        print("PF r", r, "S0", S0[str(r)], flush=True)
    for c_ in grid.values():
        c_["S"] = c_["n_correct"] / c_["n_items"] if c_["n_items"] else None; c_["exact_rate"] = c_["n_items"] / c_["n_draws"]
    return {"grid": grid, "S0": S0, "def": "D §1.2 PF: keys '<k>_<s>_<r>'; S = P(correct at K' = k + r + 2 | R decodes Q exactly at k); n_items = exact-at-k "
            "items; S0 = accuracy of the fresh Q items at K' = r + 2; 20k draws per r, k uniform on 1..12 (seeds 21402 / 21403)"}


def _feat_mod():
    import importlib.util
    sp = importlib.util.spec_from_file_location("s_feat", os.path.join(ROOT, "docs", "research_bottleneck_2026-10-16", "scripts", "step", "feat.py"))
    md = importlib.util.module_from_spec(sp); sp.loader.exec_module(md); return md      # S's steps_features (its __main__ is guarded)


@torch.no_grad()
def part_x16(m, TE, dev, bs, rd):
    """note 211-16 parts that need no pin: STEP rows (S's features on exact-input STEP-CAP events, n 3..8, t 1..n), REPAIR, and the eval-only history
    arms V0-OFF (V_0 masked in core and gather for t >= 2) and recent-1 (only V_{t-1}; V_0 masked too) at K' = n + 2, n 1..16. Writes eval_step_rows.json,
    eval_repair.json, eval_hist_arms.json (floor2 is added once its implementation is pinned)."""
    F_ = _feat_mod(); num = torch.tensor(N.NUM_IDS, device=dev); Nn = np.array([it["n"] for it in TE]); rows = []
    rep = {"events": 0, "right_after": 0, "items_wrong": 0, "final_ok_given_wrong": 0, "events_consumed_next": 0}
    for n in range(3, 9):
        for ch in VE.chunks(np.nonzero(Nn == n)[0], bs):
            its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev); E = VB.rec_targets(its, n + 2, x.shape[1], dev); VS = VE.vslot_tensor(its, n, dev)
            r_ = m.rollout(x, pad, n + 2, coda_rows={n + 2: torch.arange(len(its), device=dev)})
            fin = (m.answer_logits(r_["coda"][n + 2][1], lens)[:, num].argmax(-1) == torch.tensor([it["t"] for it in its], device=dev)).cpu().numpy()
            dec = [m.rec_logits(r_["S"][t]).argmax(-1) for t in range(n + 2)]; del r_
            for k_, j in enumerate(ch):
                L = int(lens[k_]); feats = F_.steps_features(TE[j]["root"]); wrong_any = False
                for t in range(1, n + 1):
                    b = int(VS[k_, t]); Et, Ep = E[k_, t, :L], E[k_, t - 1, :L]; dt, dp = dec[t][k_, :L], dec[t - 1][k_, :L]
                    exact_in = bool((dp == Ep).all()); ok = bool(dt[b] == Et[b]); wrong_any |= not ok; f_ = feats[t - 1]
                    if exact_in:
                        rows.append({"item": int(j), "n": n, "t": t, "hh": f_["hh"], "alive": f_["alive"], "held": f_["held"], "prov": f_["prov"], "age_max": f_["age_max"],
                                     "bdepth": f_["bdepth"], "correct": int(ok), "age_min": f_["age_min"], "r_left": f_["r"], "op": f_["op"]})
                    skel = bool(((dt == Et) | (torch.arange(L, device=dev) == b)).all())
                    if not ok and skel and t < n:                           # REPAIR events: the value wrong at t, every other slot right
                        En = E[k_, t + 1, :L]
                        if int(En[b]) == int(Et[b]):
                            rep["events"] += 1; rep["right_after"] += int(dec[t + 1][k_, b] == En[b])
                        else:
                            rep["events_consumed_next"] += 1
                if wrong_any:
                    rep["items_wrong"] += 1; rep["final_ok_given_wrong"] += int(fin[k_])
    json.dump({"rows": rows, "n_rows": len(rows), "def": "note 211-16 §3 / Theory's contract: one row per exact-input STEP-CAP event (R(stored V_{t-1}) == E_{t-1} on every real "
               "slot), n 3..8, t 1..n; features = S's steps_features (docs/research_bottleneck_2026-10-16/scripts/step/feat.py); correct = R(V_t) at the value slot == E_t"},
              open(os.path.join(rd, "eval_step_rows.json"), "w", newline="\n"))
    repair = {"p_right_after_wrong": rep["right_after"] / rep["events"] if rep["events"] else None, "n_events": rep["events"],
              "p_final_correct_given_wrong": rep["final_ok_given_wrong"] / rep["items_wrong"] if rep["items_wrong"] else None, "n_items": rep["items_wrong"],
              "n_events_consumed_next": rep["events_consumed_next"],
              "def": "211-16 §2: events = (item, t < n) with R's value slot wrong at t and every other real slot right, and the slot not consumed at t + 1; right_after = R "
                     "right at that slot at t + 1. items = n 3..8 items with >= 1 wrong value slot at t <= n; final = the coda at K' = n + 2"}
    json.dump(repair, open(os.path.join(rd, "eval_repair.json"), "w", newline="\n"), indent=1)
    arms = {"v0off": (lambda t, nv: (torch.arange(nv) != 0) | torch.tensor(t < 2), None), "recent1": (lambda t, nv: torch.arange(nv) == t - 1, None),
            "floor2": (None, 2.0)}                                          # note 212-3: floor2 = age term min(gamma * age, -2 * age) on versions 1..t-2
    hist = {}
    for nm, (fk, fl) in arms.items():
        A = np.full(len(TE), -1)
        for n in range(1, 17):
            for ch in VE.chunks(np.nonzero(Nn == n)[0], bs):
                its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev); m.ver_keep = fk; m.age_floor = fl
                try:
                    r_ = m.rollout(x, pad, n + 2, coda_rows={n + 2: torch.arange(len(its), device=dev)})
                finally:
                    m.ver_keep = None; m.age_floor = None
                A[ch] = m.answer_logits(r_["coda"][n + 2][1], lens)[:, num].argmax(-1).cpu().numpy(); del r_
        Y = np.array([it["t"] for it in TE]); hist[nm] = {"acc_n2": {str(n): float((A == Y)[Nn == n].mean()) for n in range(1, 17)}}
        print("ARM", nm, {k: round(v, 3) for k, v in hist[nm]["acc_n2"].items()}, flush=True)
    hist["def"] = ("211-16 §4 eval-only arms at K' = n + 2: v0off = V_0 masked in core and gather for passes t >= 2 (the coda never reads V_0); recent1 = only "
                   "V_{t-1} readable in core, gather and coda (V_0 masked too); floor2 = note 212-3 slope floor: age term min(gamma * age, -2 * age) on versions "
                   "1..t-2 in core, coda and the PA gather (V_0 keeps beta0)")
    json.dump(hist, open(os.path.join(rd, "eval_hist_arms.json"), "w", newline="\n"), indent=1)
    return {"step_rows": len(rows), "repair": repair, "hist_arms": {k: v for k, v in hist.items() if k != "def"}}



@torch.no_grad()
def part_foreign(m, TE, dev, bs, rd, os_=(3, 6), seed=21500):
    """note 211-16 §5 / 212-3 FOREIGN pair at K' = n + 2 real passes, n 4..12: the foreign item = another test item of the same token length (rng seed),
    F_1..F_o = its own stored versions. INSERT: prefix [F_1..F_o] after V_0 (coda at o + n + 2); REPLACE: ver_sub (entry s read as F_s by passes > s + 1);
    intact beside. -> eval_foreign.json {insert / replace: {o: {acc_n2}}, intact: {acc_n2}}."""
    num = torch.tensor(N.NUM_IDS, device=dev); rng = np.random.default_rng(seed); Nn = np.array([it["n"] for it in TE]); bylen = {}
    for j, it in enumerate(TE):
        bylen.setdefault(len(it["syms"]), []).append(j)
    res = {"insert": {str(o): {"acc_n2": {}} for o in os_}, "replace": {str(o): {"acc_n2": {}} for o in os_}, "intact": {"acc_n2": {}}, "skipped_no_partner": {}}
    for n in range(4, 13):
        cnt = {"intact": 0, **{("insert", o): 0 for o in os_}, **{("replace", o): 0 for o in os_}}; tot = 0
        cand = [j for j in np.nonzero(Nn == n)[0] if len(bylen[len(TE[j]["syms"])]) > 1]                 # items with a length-matched partner
        res.setdefault("skipped_no_partner", {})[str(n)] = int((Nn == n).sum()) - len(cand)
        for ch in VE.chunks(np.array(cand, dtype=np.int64), bs):
            its = [TE[j] for j in ch]; fj = []
            for j in ch:
                pool = [k for k in bylen[len(TE[j]["syms"])] if k != j]; fj.append(int(pool[int(rng.integers(0, len(pool)))]))
            fits = [TE[k] for k in fj]; x, pad, lens = VB.batch_x(its, dev); xf, pf, lf = VB.batch_x(fits, dev)
            assert torch.equal(pad, pf), "FOREIGN: the foreign item is not length-matched"
            y = torch.tensor([it["t"] for it in its], device=dev); tot += len(its); K = n + 2
            Fs = m.rollout(xf, pf, max(os_))["S"]
            r0 = m.rollout(x, pad, K, coda_rows={K: torch.arange(len(its), device=dev)})
            cnt["intact"] += int((m.answer_logits(r0["coda"][K][1], lens)[:, num].argmax(-1) == y).sum()); del r0
            for o in os_:
                ri = m.rollout(x, pad, o + K, coda_rows={o + K: torch.arange(len(its), device=dev)}, prefix=[Fs[s_] for s_ in range(1, o + 1)])
                cnt[("insert", o)] += int((m.answer_logits(ri["coda"][o + K][1], lens)[:, num].argmax(-1) == y).sum()); del ri
                rr = m.rollout(x, pad, K, coda_rows={K: torch.arange(len(its), device=dev)}, ver_sub=[Fs[s_] for s_ in range(1, o + 1)])
                cnt[("replace", o)] += int((m.answer_logits(rr["coda"][K][1], lens)[:, num].argmax(-1) == y).sum()); del rr
            del Fs
        if not tot:
            continue
        res["intact"]["acc_n2"][str(n)] = cnt["intact"] / tot
        for o in os_:
            res["insert"][str(o)]["acc_n2"][str(n)] = cnt[("insert", o)] / tot; res["replace"][str(o)]["acc_n2"][str(n)] = cnt[("replace", o)] / tot
        print("FOREIGN n", n, {k if isinstance(k, str) else "%s%d" % k: round(v / tot, 3) for k, v in cnt.items()}, flush=True)
    res["def"] = ("211-16 §5 / 212-3: foreign = another test item of the same token length (rng 21500), F_1..F_o its own stored versions; INSERT = after V_0, "
                  "real passes then n + 2 (coda at o + n + 2; carry stays V_0); REPLACE = entry s in 1..o read as F_s by passes > s + 1 and the coda; intact beside")
    json.dump(res, open(os.path.join(rd, "eval_foreign.json"), "w", newline="\n"), indent=1)
    return {k: v for k, v in res.items() if k != "def"}



def fresh_items(m_ops, n_items, seed):
    rng = np.random.default_rng(seed); return [{"root": N.generate(rng, m_ops)[0]} for _ in range(n_items)]


@torch.no_grad()
def single_step(m, items, dev, bs):
    """one pass on fresh inputs; exact = R(V_1) VOIDs dropped == the next reduction step. -> list of (m_ops, T tokens, exact)."""
    out = []
    for c0 in range(0, len(items), bs):
        part = items[c0:c0 + bs]; its = [{"syms": N.render(it["root"])} for it in part]; x, pad, lens = VB.batch_x(its, dev)
        rr = m.rollout(x, pad, 1); dec = m.rec_logits(rr["V"][1]).argmax(-1).cpu().numpy(); del rr
        for k_, it in enumerate(part):
            out.append((n_ops(it["root"]), len(its[k_]["syms"]) + 2, VB.rerender(list(dec[k_, :int(lens[k_])])) == N.render(reduce_k(it["root"], 1))))
    return out


def agg(trip, key):
    d = {}
    for mo, T, ok in trip:
        k = key(mo, T); c_ = d.setdefault(k, [0, 0]); c_[0] += 1; c_[1] += int(ok)
    return {k: {"n_steps": v[0], "exact": v[1] / v[0]} for k, v in sorted(d.items(), key=lambda kv: tuple(int(z) for z in kv[0].split("_")))}


def part_steppos(m, TE, dev, bs):
    """Theory 12:5x (D3 form): (A) chain LONG-STEP keyed m_T; (B) a fresh single-step grid, 2,000 random items per m 1..16 (generator seed 21600 + m),
    exact by m_T (bracket count varies T at fixed m) and by m; (C) the position-offset test: fresh items m 1..8 (2,000 each, seed 21700 + m) at
    deltamem.POS_OFFSET o in {0, 8, 16} (prelude wpe and the PA gather's position queries both shifted), plain and with POS_KEEP0 (position 0 keeps
    wpe[0], the 187-5 sink fix); exact by m (m <= 5 = the control)."""
    import boxes.deltamem as DM
    res = {"long_step_m_T": long_step(m, TE, dev, bs, True)}; print("A done", flush=True)
    trip = []
    for mo in range(1, 17):
        trip += single_step(m, fresh_items(mo, 2000, 21600 + mo), dev, bs)
    res["fresh_m_T"] = agg(trip, lambda a_, b_: "%d_%d" % (a_, b_)); res["fresh_m"] = agg(trip, lambda a_, b_: str(a_)); print("B done", {k: round(v["exact"], 3) for k, v in res["fresh_m"].items()}, flush=True)
    res["offset"] = {}
    try:
        for o in (0, 8, 16):
            for keep0 in ((False,) if o == 0 else (False, True)):
                DM.POS_OFFSET, DM.POS_KEEP0 = o, keep0; trip = []
                for mo in range(1, 9):
                    trip += single_step(m, fresh_items(mo, 2000, 21700 + mo), dev, bs)
                res["offset"]["%d%s" % (o, "_keep0" if keep0 else "")] = agg(trip, lambda a_, b_: str(a_))
                print("C", o, keep0, {k: round(v["exact"], 3) for k, v in res["offset"]["%d%s" % (o, "_keep0" if keep0 else "")].items()}, flush=True)
    finally:
        DM.POS_OFFSET, DM.POS_KEEP0 = 0, False
    res["def"] = part_steppos.__doc__
    return res



def part_gsplice(m, TE, dev, bs, J=29):
    """note 212-6 GATHER-SPLICE ($0, Audit-approved): the fresh single-step grid of eval_steppos (B; 2,000 items per m 1..16, seed 21600 + m) under
    intact / gs (pass input at slots j >= 29 = the CARRY value N(V_{t-1}[j]) instead of the gather; at t = 1 that is N(V_0[j])) / mod31 (gather query position
    j mod 31). Readings (Theory's, pre-stated): GATHER-BOTTLENECK if gs raises pooled exact at T in {31, 33}, m 8-10 by >= 0.30 with the T <= 27 / m <= 8
    control >= -0.02; NOT-GATHER if < 0.10; UNREADABLE if the control drops > 0.02."""
    assert m.mode == "pa", "GATHER-SPLICE needs a PURE-ANNEAL cell"
    res = {}
    for arm, (sp, qm) in {"intact": (None, None), "gs": (J, None), "mod31": (None, 31)}.items():
        m.gather_splice, m.gather_qmod = sp, qm; trip = []
        try:
            for mo in range(1, 17):
                trip += single_step(m, fresh_items(mo, 2000, 21600 + mo), dev, bs)
        finally:
            m.gather_splice = m.gather_qmod = None
        res[arm] = {"fresh_m_T": agg(trip, lambda a_, b_: "%d_%d" % (a_, b_)), "fresh_m": agg(trip, lambda a_, b_: str(a_))}
        print("GS", arm, {k: round(v["exact"], 3) for k, v in res[arm]["fresh_m"].items()}, flush=True)

    def pooled(arm, cond):
        c_ = [(v["n_steps"], v["exact"]) for k, v in res[arm]["fresh_m_T"].items() if cond(*(int(z) for z in k.split("_")))]
        n_ = sum(a for a, _ in c_); return (sum(a * b for a, b in c_) / n_, n_) if n_ else (None, 0)
    tgt = lambda mo, T: 8 <= mo <= 10 and T in (31, 33)  # noqa: E731
    ctl = lambda mo, T: mo <= 8 and T <= 27  # noqa: E731
    res["summary"] = {arm: {"target_T31_33_m8_10": pooled(arm, tgt), "control_T27_m8": pooled(arm, ctl)} for arm in ("intact", "gs", "mod31")}
    res["J"] = J; res["def"] = part_gsplice.__doc__
    return res



@torch.no_grad()
def part_a15(m, TE, dev, bs):
    """note 211-15 A descriptives: (iii) RE-ROOT-VB - pass-1 STEP-CAP on render(E_{t-1}) as a fresh problem, test items n 3..8, t 2..n, by the original
    handle height hh (S's steps_features), exact = the full next-step decode; (v) the same after o in {2, 4} prepended V_0 copies (INSERT prefix of V_0,
    then one real pass); (iv, PA only) the gather mass split at passes 2..n of the intact rollout to n: on (t-1, j) / on (s < t-1, j) / on (., i != j)."""
    F_ = _feat_mod(); Nn = np.array([it["n"] for it in TE]); res = {"reroot": {}, "split": None}
    jobs = []
    for j in np.nonzero((Nn >= 3) & (Nn <= 8))[0]:
        fts = F_.steps_features(TE[j]["root"])
        for t in range(2, TE[j]["n"] + 1):
            jobs.append((int(j), t, min(fts[t - 1]["hh"], 5)))
    for o in (0, 2, 4):
        cnt = {}
        for c0 in range(0, len(jobs), bs):
            part = jobs[c0:c0 + bs]; roots = [reduce_k(TE[j]["root"], t - 1) for j, t, _ in part]; its = [{"syms": N.render(r_)} for r_ in roots]
            x, pad, lens = VB.batch_x(its, dev)
            if o:
                V0 = m.rollout(x, pad, 0)["V"][0]; rr = m.rollout(x, pad, o + 1, prefix=[V0] * o); Vn = rr["V"][o + 1]
            else:
                rr = m.rollout(x, pad, 1); Vn = rr["V"][1]
            dec = m.rec_logits(Vn).argmax(-1).cpu().numpy(); del rr
            for k_, (j, t, hh) in enumerate(part):
                ok = VB.rerender(list(dec[k_, :int(lens[k_])])) == N.render(reduce_k(roots[k_], 1)); c_ = cnt.setdefault(str(hh), [0, 0]); c_[0] += 1; c_[1] += int(ok)
        res["reroot"]["o%d" % o] = {h: {"n": v[0], "exact": v[1] / v[0]} for h, v in sorted(cnt.items())}
        print("REROOT o", o, {h: round(v[1] / v[0], 4) for h, v in sorted(cnt.items())}, flush=True)
    if m.mode == "pa":
        acc = {}
        for n in range(3, 9):
            for ch in VE.chunks(np.nonzero(Nn == n)[0], bs):
                its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev); m.gather_split_log = []
                try:
                    m.rollout(x, pad, n)
                    for t, a_, b_, c_ in m.gather_split_log:
                        if t >= 2:
                            z = acc.setdefault(str(t), [0, 0.0, 0.0, 0.0]); z[0] += len(its); z[1] += float(a_.sum()); z[2] += float(b_.sum()); z[3] += float(c_.sum())
                finally:
                    m.gather_split_log = None
        res["split"] = {t: {"n": z[0], "self_t1_j": z[1] / z[0], "older_same_j": z[2] / z[0], "cross_slot": z[3] / z[0]} for t, z in sorted(acc.items(), key=lambda kv: int(kv[0]))}
        tot = [sum(z[k] for z in acc.values()) for k in range(4)]; res["split_pooled"] = {"self_t1_j": tot[1] / tot[0], "older_same_j": tot[2] / tot[0], "cross_slot": tot[3] / tot[0]}
        print("SPLIT", res["split_pooled"], flush=True)
    res["def"] = part_a15.__doc__
    return res



def part_d3diag(m, TE, dev, bs, J=24):
    """note 212-11 ($0, eval-only, Audit-OK'd): the fresh single-step grid of eval_steppos B (2,000 items per m 1..16, seed 21600 + m) under intact /
    gs24 (gather_splice J = 24; PA only) / wpe_b1 (the fine-tuned wpe rows >= 24 replaced by PRETRAINED GPT-2 wpe rows) / wpe_b2 (every wpe row pretrained).
    The wpe arms swap the table's data at eval and restore an exact clone (off = bitwise default; asserted). summary: target = T in {31, 33}, m 8-10;
    control_exposed = m <= 8, T 25-29 (items with slots >= 24)."""
    import boxes.s5 as S5
    pre = torch.load(os.path.join(S5.ROOT, "prosqa", "ckpt", "gpt2_pretrained.bin"), map_location="cpu", weights_only=True)["wpe.weight"].to(dev)
    W = m.g.p["wpe_weight"]; orig = W.data.clone(); assert pre.shape == W.shape, (pre.shape, W.shape)
    arms = {"intact": None, "wpe_b1": "b1", "wpe_b2": "b2"}
    if m.mode == "pa":
        arms["gs24"] = "gs"
    res = {}
    for arm, how in arms.items():
        trip = []
        try:
            if how == "gs":
                m.gather_splice = J
            elif how == "b1":
                W.data[J:] = pre[J:].to(W.dtype)
            elif how == "b2":
                W.data[:] = pre.to(W.dtype)
            for mo in range(1, 17):
                trip += single_step(m, fresh_items(mo, 2000, 21600 + mo), dev, bs)
        finally:
            m.gather_splice = None; W.data.copy_(orig)
        assert torch.equal(W.data, orig), "d3diag: wpe not restored"
        res[arm] = {"fresh_m_T": agg(trip, lambda a_, b_: "%d_%d" % (a_, b_)), "fresh_m": agg(trip, lambda a_, b_: str(a_))}
        print("D3", arm, {k: round(v["exact"], 3) for k, v in res[arm]["fresh_m"].items()}, flush=True)

    def pooled(arm, cond):
        c_ = [(v["n_steps"], v["exact"]) for k, v in res[arm]["fresh_m_T"].items() if cond(*(int(z) for z in k.split("_")))]
        n_ = sum(a for a, _ in c_); return [sum(a * b for a, b in c_) / n_ if n_ else None, n_]
    res["summary"] = {arm: {"target": pooled(arm, lambda mo, T: 8 <= mo <= 10 and T in (31, 33)), "control_exposed": pooled(arm, lambda mo, T: mo <= 8 and 25 <= T <= 29)}
                      for arm in arms}
    res["J"] = J; res["def"] = part_d3diag.__doc__
    return res


@torch.no_grad()
def part_wpereset(m, TE, dev, bs):
    """Theory 10-16 ~14:00 (N1-WPE-RESET; $0, descriptive, for D3's FT-from-N1 reset choice): every wpe row PRETRAINED GPT-2 (d3diag's wpe_b2), arms
    intact / wpe_b2. Per arm: acc_n2 {n: acc at K' = n + 2} and acc_k24 {n: acc at K' = 24}, n 1..16, from one K' 24 rollout per chunk (vbank_eval.part_acc's
    chunking and coda rows -> the intact arm must equal eval_vb.json's acc_n2 / acc_k['24']; checked in main's caller); rs_n9_13 = part_rs's cells on the
    n 9-13 test items (no LONG-STEP). The table's data is swapped at eval and an exact clone restored (asserted)."""
    import boxes.s5 as S5
    pre = torch.load(os.path.join(S5.ROOT, "prosqa", "ckpt", "gpt2_pretrained.bin"), map_location="cpu", weights_only=True)["wpe.weight"].to(dev)
    W = m.g.p["wpe_weight"]; orig = W.data.clone(); assert pre.shape == W.shape, (pre.shape, W.shape)
    num = torch.tensor(N.NUM_IDS, device=dev); Nn = np.array([it["n"] for it in TE]); Y = np.array([it["t"] for it in TE]); res = {}
    sub = [it for it in TE if 9 <= it["n"] <= 13]
    for arm in ("intact", "wpe_b2"):
        A2 = np.full(len(TE), -1); A24 = np.full(len(TE), -1)
        try:
            if arm == "wpe_b2":
                W.data[:] = pre.to(W.dtype)
            for n in range(1, 17):
                for ch in VE.chunks(np.nonzero(Nn == n)[0], bs):
                    its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev)
                    r_ = m.rollout(x, pad, 24, coda_rows={t: torch.arange(len(its), device=dev) for t in set(VE.KS) | {n + 2}})
                    A2[ch] = m.answer_logits(r_["coda"][n + 2][1], lens)[:, num].argmax(-1).cpu().numpy()
                    A24[ch] = m.answer_logits(r_["coda"][24][1], lens)[:, num].argmax(-1).cpu().numpy(); del r_
            rs = part_rs(m, sub, dev, bs, do_long=False)["cells"]
        finally:
            W.data.copy_(orig)
        assert torch.equal(W.data, orig), "wpereset: wpe not restored"
        by_n = lambda a_: {str(n): float((a_ == Y)[Nn == n].mean()) for n in range(1, 17)}  # noqa: E731
        res[arm] = {"acc_n2": by_n(A2), "acc_k24": by_n(A24), "rs_n9_13": rs}
        print("WPE", arm, "acc_n2", {k: round(v, 3) for k, v in res[arm]["acc_n2"].items()}, flush=True)
    res["def"] = part_wpereset.__doc__
    return res


# ---------------------------------------------------------------- note 212-15: D3 Stage 0 rows (S0-1 SELECTOR, S0-2 LENGTH-FREE-K, S0-3 STAB-HIST)
def _cls_tok(c):
    """recode class id -> its token id (VOID -> None)."""
    nm = VB.CLS[int(c)]
    return None if nm == "VOID" else N.EOT if nm == "EOT" else N.TID[nm]


def _lower_masked(g, x, km):
    """the prelude (blocks 0..5 -> V_0) with the current-version keys at km (B, T) True removed (GAPPED's interior mask); block_bank without a bank
    is deltamem._block's computation with an explicit mask (G1)."""
    h = g.embed(x) + VB.DM_().wpe_slice(g, x.shape[1])[None]
    for i in range(VB.CORE[0]):
        h = VB.block_bank(g, i, h, None, None, None, km)
    return h


@torch.no_grad()
def part_d3sel(m, TE, dev, bs):
    """note 212-15 S0-1 SELECTOR (D3_v1 §6). Arms at K' = n + 2 on the §170 test set (part_rs's chunks):
    intact; MASKVOID = vbank slot_mask: a slot whose argmax R(V_t[j]) is VOID (t >= 1, the model's own decode; never EOT / ' =') is removed from every reader (core / coda
    bank entries of EVERY stored version, the current version's keys; PA: the gather) for every core pass > t and the coda after pass >= t;
    REFRESH = rollout(edit): at every pass t >= 1, slots whose argmax R(V_t[j]) equals their input class E_0[j] have their stored entry (and the carry)
    set to V_0[j]. RS items (exact decode at t = n - r, part_rs's classify): restart = part_rs's compact fresh run at K' = r + 2, computed over part_rs's
    full job lists for r in {2, 4, 6} (same batches -> bitwise with eval_rs.json); GAPPED = a fresh run at K' = r + 2 on x with each live slot of the
    reference E_t at its ORIGINAL position (token of its class) and the E_t VOID slots removed from every reader from V_0 on (prelude via an explicit
    interior key mask, then slot_mask). cells n 9-13 x r {4, 6}; gapped_control n 5-8 x r {2, 4}; acc_n2 per arm n 1..16; exposure = share of items with
    >= 1 masked (MASKVOID) / refreshed (REFRESH) slot at some pass <= n + 2, and the mean count of slot-events."""
    num = torch.tensor(N.NUM_IDS, device=dev); Nn = np.array([it["n"] for it in TE]); Y = np.array([it["t"] for it in TE])
    ARMS = ("intact", "maskvoid", "refresh"); acc = {a_: np.full(len(TE), -1) for a_ in ARMS}; RR = (2, 4, 6); jobs = {r: [] for r in RR}
    CELLS = [(n, r) for n in range(9, 14) for r in (4, 6)]; CTRL = [(n, r) for n in range(5, 9) for r in (2, 4)]; expo = {a_: {} for a_ in ("maskvoid", "refresh")}
    for n in range(1, 17):
        for ch in VE.chunks(np.nonzero(Nn == n)[0], bs):
            its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev); B, T = x.shape; E0 = VB.rec_targets(its, 0, T, dev)[:, 0]; rows = torch.arange(B, device=dev)
            r_ = m.rollout(x, pad, n + 2, coda_rows={n + 2: rows}); lg_ = m.answer_logits(r_["coda"][n + 2][1], lens); assert bool(torch.isfinite(lg_).all())
            acc["intact"][ch] = lg_[:, num].argmax(-1).cpu().numpy()
            for r in RR:
                t = n - r
                if t < 1:
                    continue
                dec = m.rec_logits(r_["V"][t]).argmax(-1).cpu().numpy()
                for k_, j in enumerate(ch):
                    if classify(VB.rerender(list(dec[k_, :int(lens[k_])])), N.cot_steps(TE[j]["root"])[t]) == "exact":
                        jobs[r].append((int(j), t))
            V0 = r_["V"][0]; del r_
            ex = {"any": torch.zeros(B, dtype=torch.bool, device=dev), "ev": 0}

            def mv(t, V, ex=ex, pad=pad, lens=lens):
                z = (m.rec_logits(V).argmax(-1) == VB.VOID) if t >= 1 else torch.zeros(V.shape[:2], dtype=torch.bool, device=V.device)
                z = z & pad; z[:, 0] = False; z[torch.arange(len(lens), device=z.device), lens - 1] = False   # Audit 14:4x (PROOFS_D3 row 1): EOT / ' =' never
                ex["any"] |= z.any(1); ex["ev"] += int(z.sum()); return z                                 # consumed (an all -inf row 0 at t <= 1 -> NaN)
            try:
                m.slot_mask = mv; rm_ = m.rollout(x, pad, n + 2, coda_rows={n + 2: rows})
            finally:
                m.slot_mask = None
            lg_ = m.answer_logits(rm_["coda"][n + 2][1], lens); assert bool(torch.isfinite(lg_).all()), "MASKVOID: non-finite answer logits"
            acc["maskvoid"][ch] = lg_[:, num].argmax(-1).cpu().numpy(); del rm_
            e_ = expo["maskvoid"].setdefault(str(n), [0, 0, 0]); e_[0] += int(ex["any"].sum()); e_[1] += ex["ev"]; e_[2] += B
            ex = {"any": torch.zeros(B, dtype=torch.bool, device=dev), "ev": 0}

            def rf(t, V, ex=ex, pad=pad, E0=E0, V0=V0):
                z = (m.rec_logits(V).argmax(-1) == E0) & pad; ex["any"] |= z.any(1); ex["ev"] += int(z.sum())
                return torch.where(z[..., None], V0, V)
            rr_ = m.rollout(x, pad, n + 2, coda_rows={n + 2: rows}, edit=rf)
            lg_ = m.answer_logits(rr_["coda"][n + 2][1], lens); assert bool(torch.isfinite(lg_).all()), "REFRESH: non-finite answer logits"
            acc["refresh"][ch] = lg_[:, num].argmax(-1).cpu().numpy(); del rr_
            e_ = expo["refresh"].setdefault(str(n), [0, 0, 0]); e_[0] += int(ex["any"].sum()); e_[1] += ex["ev"]; e_[2] += B
        print("D3SEL n", n, {a_: round(float((acc[a_] == Y)[Nn == n].mean()), 3) for a_ in ARMS}, flush=True)
    rest = {}
    for r in RR:                                                            # part_rs's restart, same job order and batches
        js = jobs[r]
        for c0 in range(0, len(js), bs):
            part = js[c0:c0 + bs]; its = [{"syms": N.render(reduce_k(TE[j]["root"], t)), "t": TE[j]["t"]} for j, t in part]
            x, pad, lens = VB.batch_x(its, dev); rr = m.rollout(x, pad, r + 2, coda_rows={r + 2: torch.arange(len(its), device=dev)})
            a_ = m.answer_logits(rr["coda"][r + 2][1], lens)[:, num].argmax(-1).cpu().numpy(); del rr
            for k_, (j, t) in enumerate(part):
                rest[(j, t)] = bool(a_[k_] == TE[j]["t"])
    gap = {}; need = set(CELLS) | set(CTRL)
    for r in RR:                                                            # GAPPED on the cells' / controls' items only
        js = [(j, t) for j, t in jobs[r] if (TE[j]["n"], r) in need]
        for c0 in range(0, len(js), bs):
            part = js[c0:c0 + bs]; x, pad, lens = VB.batch_x([TE[j] for j, _ in part], dev); km = torch.zeros_like(pad)
            for k_, (j, t) in enumerate(part):
                Et = VB.recode_targets(TE[j]["root"], t)[t]; assert len(Et) == int(lens[k_]), "GAPPED: E_t length != the item's"
                for p_, c in enumerate(Et):
                    tk = _cls_tok(c)
                    if tk is None:
                        km[k_, p_] = True
                    else:
                        x[k_, p_] = tk
            assert not bool((km & ~pad).any()) and not bool(km[:, 0].any()) and not bool(km[torch.arange(len(part), device=dev), lens - 1].any())
            try:
                m.slot_mask = lambda t, V, km=km: km if t == 0 else torch.zeros_like(km)
                rg = m.rollout(x, pad, r + 2, coda_rows={r + 2: torch.arange(len(part), device=dev)}, v0=_lower_masked(m.g, x, km))
            finally:
                m.slot_mask = None
            lg_ = m.answer_logits(rg["coda"][r + 2][1], lens); assert bool(torch.isfinite(lg_).all()), "GAPPED: non-finite answer logits"
            a_ = lg_[:, num].argmax(-1).cpu().numpy(); del rg
            for k_, (j, t) in enumerate(part):
                gap[(j, t)] = bool(a_[k_] == TE[j]["t"])
    mean_ = lambda v: float(np.mean(v)) if len(v) else None  # noqa: E731
    cells, ctrl = {}, {}
    for (n, r) in CELLS + CTRL:
        tt = n - r; js = [j for (j, t) in jobs[r] if t == tt and TE[j]["n"] == n]
        c_ = {"n_items": len(js), "restart": mean_([rest[(j, tt)] for j in js]), "gapped": mean_([gap[(j, tt)] for j in js])}
        if (n, r) in CELLS:
            c_.update({a_: mean_([bool(acc[a_][j] == Y[j]) for j in js]) for a_ in ARMS}); cells["%d_%d" % (n, r)] = c_
        else:
            ctrl["%d_%d" % (n, r)] = c_
    # GAPPED prelude path check: an all-False interior mask vs deltamem.gpt_lower on one chunk (explicit mask vs is_causal kernels)
    its = [TE[j] for j in np.nonzero(Nn == 9)[0][:bs]]; x, pad, _ = VB.batch_x(its, dev)
    pre_diff = float((_lower_masked(m.g, x, torch.zeros_like(pad)) - VB.DM_().gpt_lower(m.g, m.g.embed(x), VB.CORE[0])).abs().max())
    by_n = lambda a_: {str(n): float((a_ == Y)[Nn == n].mean()) for n in range(1, 17)}  # noqa: E731
    return {"cells": cells, "acc_n2": {a_: by_n(acc[a_]) for a_ in ARMS}, "gapped_control": ctrl, "gapped_supported": True,
            "exposure": {a_: {n: {"items_exposed": v[0] / v[2], "slot_events_per_item": v[1] / v[2]} for n, v in d.items()} for a_, d in expo.items()},
            "gapped_prelude_maxdiff_empty_mask": pre_diff, "def": part_d3sel.__doc__}


FAULT_P, FAULT_SEED, FAULT_J = 0.05, 21302, 33


@torch.no_grad()
def _lfk_core(m, TE, dev, bs, fault, p_=FAULT_P, seed=FAULT_SEED):
    """one LENGTH-FREE-K pass over n 9-16 (see part_lfk); fault: the 213-1 FAULT arm via vbank input_edit (p_, seed; False = intact).
    -> (result dict, per-event {(item, t): {bucket: (all-kind slot-events, strict correct)}} for the 213-2 pairing)."""
    Nn = np.array([it["n"] for it in TE]); KINDS = ("copy", "changed", "void", "all"); BK = (("le28", 0, 28), ("29_32", 29, 32), ("ge33", 33, 10 ** 6))
    slot = {"%s_%s" % (k, b[0]): [0, 0] for k in KINDS for b in BK}; step = {}; fw = {b[0]: [] for b in BK}; ge33_by_n = {}; nf = [0, 0]
    bname = lambda v: "le28" if v <= 28 else "29_32" if v <= 32 else "ge33"  # noqa: E731
    rng = np.random.default_rng(seed) if fault else None; ev = {}; rz = [0, 0]
    for n in range(9, 17):
        st = step.setdefault(str(n), {"events": 0, "strict_exact": 0, "voiddrop_exact": 0})
        for ch in VE.chunks(np.nonzero(Nn == n)[0], bs):
            its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev); B, T = x.shape; E = VB.rec_targets(its, n, T, dev)
            if fault:                                                       # draws per (item, t, j) in a fixed order: items in test-set order, t 2..n, j FAULT_J..L-1
                fm = torch.zeros(n + 1, B, T, dtype=torch.bool)
                for i_ in range(B):
                    for t in range(2, n + 1):
                        for j_ in range(FAULT_J, int(lens[i_])):
                            fm[t, i_, j_] = bool(rng.random() < p_)
                fm = fm.to(dev); nf[0] += int(fm.sum()); nf[1] += sum(max(0, int(L_) - FAULT_J) for L_ in lens) * (n - 1)

                def fe(t, U, fm=fm):
                    if t < 2 or not bool(fm[t].any()):
                        return U
                    sh = torch.cat([U[:, :1], U[:, :-1]], 1)                # sh[:, j] = U[:, j - 1]
                    return torch.where(fm[t][..., None], sh, U)
                m.input_edit = fe
            try:
                r_ = m.rollout(x, pad, n)
            finally:
                m.input_edit = None
            dec = [m.rec_logits(r_["V"][t]).argmax(-1) for t in range(n + 1)]; del r_
            col = torch.arange(T, device=dev)[None, :].expand(B, T)
            for t in range(2, n + 1):
                ok_in = ((dec[t - 1] == E[:, t - 1]) | ~pad).all(1)
                if not bool(ok_in.any()):
                    continue
                Et, Ep = E[:, t], E[:, t - 1]; cor = dec[t] == Et; base = pad & ok_in[:, None]; chg = Et != Ep
                km = {"changed": chg, "copy": ~chg & (Et != VB.VOID), "void": ~chg & (Et == VB.VOID), "all": torch.ones_like(chg)}
                for k in KINDS:
                    for b, lo, hi in BK:
                        sel = base & km[k] & (col >= lo) & (col <= hi); c_ = slot["%s_%s" % (k, b)]; c_[0] += int(sel.sum()); c_[1] += int((sel & cor).sum())
                ge33_by_n[str(n)] = ge33_by_n.get(str(n), 0) + int((base & (col >= 33)).sum())
                for b, lo, hi in BK:                                        # 213-2: per-event all-kind counts for the pairing
                    sel = base & (col >= lo) & (col <= hi); ne = sel.sum(1).cpu().numpy(); nc = (sel & cor).sum(1).cpu().numpy()
                    for i in np.nonzero(ne)[0]:
                        ev.setdefault((int(ch[i]), t), {})[b] = (int(ne[i]), int(nc[i]))
                if fault:                                                   # 213-2 realised: faulted ge33 slot-events; class(j-1) != class(j) in E_{t-1}
                    ff = base & fm[t] & (col >= 33); prev = torch.cat([Ep[:, :1], Ep[:, :-1]], 1)
                    rz[0] += int(ff.sum()); rz[1] += int((ff & (prev != Ep)).sum())
                strict = (cor | ~pad).all(1); dc = dec[t].cpu().numpy(); ec = Et.cpu().numpy(); okc = ok_in.cpu().numpy(); sc = strict.cpu().numpy()
                for i in np.nonzero(okc)[0]:
                    L = int(lens[i]); st["events"] += 1; st["strict_exact"] += int(sc[i]); st["voiddrop_exact"] += int(VB.rerender(list(dc[i, :L])) == VB.rerender(list(ec[i, :L])))
                    if not sc[i]:
                        fw[bname(L)].append(int(np.nonzero(dc[i, :L] != ec[i, :L])[0][0]))
        print("LFK", "fault" if fault else "intact", "n", n, st, flush=True)
    out = {"slot": {k: {"events": v[0], "strict_correct": v[1], "acc": v[1] / v[0] if v[0] else None} for k, v in slot.items()}, "step": step, "first_wrong": fw,
           "ge33_events_by_n": ge33_by_n}
    if fault:
        out["fault_draws"] = {"faulted": nf[0], "eligible": nf[1]}
        out["realised"] = {"ge33": {"faults": rz[0], "faults_class_differs": rz[1]}}
    return out, ev


def _lfk_paired(ev_i, ev_f):
    """213-2: over (item, t) events counted in both arms, per bucket: slot-events, intact / fault strict correct, diff = (fault - intact) / events_both."""
    out = {}
    for b in ("le28", "29_32", "ge33"):
        ks = [k for k in ev_i if k in ev_f and b in ev_i[k] and b in ev_f[k]]
        n_ = sum(ev_i[k][b][0] for k in ks); ci = sum(ev_i[k][b][1] for k in ks); cf = sum(ev_f[k][b][1] for k in ks)
        out[b] = {"events_both": n_, "intact_correct": ci, "fault_correct": cf, "diff": (cf - ci) / n_ if n_ else None}
    return out


def _lfk_sensitive(intact, fault):
    """213-1 LFK_SENSITIVE: FAULT readable (>= 500 ge33 slot-events), all-kind ge33 strict acc <= intact's - 0.02, and FAULT reads not-met (ge33 < le28 - 0.01)."""
    fs, is_ = fault["slot"], intact["slot"]
    if fs["all_ge33"]["events"] < 500 or is_["all_ge33"]["acc"] is None:
        return False
    return fs["all_ge33"]["acc"] <= is_["all_ge33"]["acc"] - 0.02 and fs["all_ge33"]["acc"] < fs["all_le28"]["acc"] - 0.01


@torch.no_grad()
def part_lfk(m, TE, dev, bs):
    """note 212-15 S0-2 LENGTH-FREE-K reference (D3_v1 §5 (a')): §170 test items n 9-16, the intact rollout (K' = n); every pass t in 2..n whose input
    V_{t-1} decodes STRICT exact (argmax R == E_{t-1} at every real slot, VOIDs in place); per real slot j (EOT and ' =' included) strict R(V_t[j]) == E_t[j].
    kind: changed (E_t[j] != E_{t-1}[j], consumption included) / copy (unchanged, non-VOID) / void (unchanged VOID) / all; bucket on the slot index j:
    le28 (j <= 28) / 29_32 / ge33. step {n: events, strict_exact, voiddrop_exact (rerender equality)}; first_wrong {T bucket of the item length T
    (tokens incl. EOT and ' ='): [first wrong slot j]} over events that are not strict exact; ge33_events_by_n = slot-events at j >= 33 per n.
    "fault" (note 213-1, the sensitivity control): the same reading with, at every pass t >= 2, each real slot j >= 33 (' =' included) independently
    with p 0.05 (np rng 21302, per (item, t, j): items in test-set order, t 2..n, j ascending) given slot j - 1's PASS INPUT (vbank input_edit:
    U_t[j] := U_t[j-1], after the carry's N(.) / the gather); stored bank entries untouched. 213-2: fault.paired {bucket: events_both, intact_correct,
    fault_correct, diff} over (item, t) events counted in both arms; fault.realised.ge33 {faults (faulted ge33 slot-events), faults_class_differs (E_{t-1}
    class at j - 1 != at j)}; fault.sensitive (LFK_SENSITIVE as computed here; the scorer decides). If not sensitive and faults_class_differs < 0.03 x the
    ge33 slot-events: "fault2" = the same at p 0.10, rng 21303."""
    out, ev_i = _lfk_core(m, TE, dev, bs, False); f_, ev_f = _lfk_core(m, TE, dev, bs, True); f_["paired"] = _lfk_paired(ev_i, ev_f); out["fault"] = f_
    sens = _lfk_sensitive(out, f_); f_["sensitive"] = sens
    if not sens and f_["realised"]["ge33"]["faults_class_differs"] < 0.03 * f_["slot"]["all_ge33"]["events"]:   # 213-2 branch: once at p 0.10, rng 21303
        f2, ev_2 = _lfk_core(m, TE, dev, bs, True, 0.10, 21303); f2["paired"] = _lfk_paired(ev_i, ev_2); f2["sensitive"] = _lfk_sensitive(out, f2); out["fault2"] = f2
    out["fault_def"] = ("213-1 FAULT: pass input U_t[j] := U_t[j-1] for real slots j >= %d (incl. ' ='), p %.2f per (item, t >= 2, j), np rng %d in test-set item "
                        "order, t ascending, j ascending; bank entries untouched; same items / reading as the top level" % (FAULT_J, FAULT_P, FAULT_SEED))
    out["def"] = part_lfk.__doc__
    return out


@torch.no_grad()
def part_stabhist(m, TE, dev, bs):
    """note 212-15 S0-3 STAB-HIST (D3_v1 §6): arms intact / self_ablate / other_ablate = vbank gather_hist_ablate None / "self" / "other" (the PA gather
    only: for query slot j the keys (s, j) resp. (s, i != j) at 1 <= s <= t-2 are masked; core / coda untouched; rt_d3/rt_hist2.py). acc_n2 (K' = n + 2)
    and acc_k24 (K' = 24) for n 3..8 from one K' 24 rollout per chunk (vbank_eval.part_acc's chunks; intact must equal eval_vb.json bitwise). check_split
    (descriptive): gather_split_log on the first chunk of n 8 - its older-same-slot mass INCLUDES V_0[j], which self_ablate keeps (the exact-zero mid-version
    check is scratch/gate_s212_d3.py g3)."""
    assert m.mode == "pa", "STAB-HIST needs the PA gather"
    num = torch.tensor(N.NUM_IDS, device=dev); Nn = np.array([it["n"] for it in TE]); Y = np.array([it["t"] for it in TE]); res = {}; chk = {}
    for arm, how in (("intact", None), ("self_ablate", "self"), ("other_ablate", "other")):
        A2 = np.full(len(TE), -1); A24 = np.full(len(TE), -1)
        try:
            m.gather_hist_ablate = how
            for n in range(3, 9):
                for c_i, ch in enumerate(VE.chunks(np.nonzero(Nn == n)[0], bs)):
                    its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev); rows = torch.arange(len(its), device=dev)
                    m.gather_split_log = [] if (n == 8 and c_i == 0) else None
                    r_ = m.rollout(x, pad, 24, coda_rows={n + 2: rows, 24: rows})
                    A2[ch] = m.answer_logits(r_["coda"][n + 2][1], lens)[:, num].argmax(-1).cpu().numpy()
                    A24[ch] = m.answer_logits(r_["coda"][24][1], lens)[:, num].argmax(-1).cpu().numpy(); del r_
                    if m.gather_split_log is not None:
                        chk[arm] = {"older_same_slot_incl_V0_mass_max": float(max(float(o.max()) for _, _, o, _ in m.gather_split_log[2:])),
                                    "self_mass_mean": float(np.mean([float(s_.mean()) for _, s_, _, _ in m.gather_split_log])),
                                    "cross_slot_mass_mean": float(np.mean([float(c.mean()) for _, _, _, c in m.gather_split_log]))}
                    m.gather_split_log = None
        finally:
            m.gather_hist_ablate = None; m.gather_split_log = None
        by_n = lambda a_: {str(n): float((a_ == Y)[Nn == n].mean()) for n in range(3, 9)}  # noqa: E731
        res[arm] = {"acc_n2": by_n(A2), "acc_k24": by_n(A24)}; print("STAB", arm, res[arm], flush=True)
    res["check_split_n8_chunk0"] = chk; res["def"] = part_stabhist.__doc__
    return res


@torch.no_grad()
def part_d3hist(m, TE, dev, bs):
    """note 213-1 eval_d3hist.json (§213; D3_v1 §5 (f) and its labels), local fp32. acc_n2 (K' = n + 2), n 1..12, vbank_eval.part_acc's chunks:
    intact; bank_ablate_slva = ver_keep vbank_eval.ablate_keep (versions 1..t-2 masked in SLVA, core and coda); self_ablate = gather_hist_ablate "self"
    (own-slot versions 1..t-2 in SLVA only); other_ablate = ablate_keep with SLVA.keep_bank_only (core and coda only). The intact arm must equal eval_vb.json
    acc_n2 bitwise (checked by the runner). swap: VALUE-SWAP class A pairs (vbank_eval.make_pairs, rng 21120), swap pass t, K' = n + 2, clean = recipient
    and donor correct (as vswap): NEWEST = the donor's V_t only as SLVA's newest version (SLVA's read at pass t + 1; slva_view); BANK = the donor's V_t only
    in the core / coda bank (stored entry = donor, SLVA's reads of entry t at passes >= t + 1 = the recipient's); OLD = the donor's V_1..V_t at every
    reader with reading index >= t + 2 (rollout ver_sub + ver_sub_from = t + 2; the coda after pass t + 1 reads index t + 2). follow = y_donor, own = y.
    Note 213-11 (the §211 cells, a known-answer harness check): PA maps SLVA's arms onto the gather (self_ablate = the gather's own-slot versions
    1..t-2; other_ablate = ver_keep with keep_bank_only; NEWEST / BANK via gather_view on the gather's version-t keys / values); CARRY: self_ablate
    omitted (no gather), other_ablate = ablate_keep (core / coda are its only version readers), NEWEST = edit -> (V_t, donor) (vswap iii_carry), BANK =
    edit -> (donor, V_t) (iii_bank). Arms a mode cannot express are listed in "omitted"."""
    num = torch.tensor(N.NUM_IDS, device=dev); Nn = np.array([it["n"] for it in TE]); Y = np.array([it["t"] for it in TE]); res = {"omitted": []}
    arms = (("intact", None), ("bank_ablate_slva", "keep"), ("self_ablate", "self"), ("other_ablate", "bank"))
    if m.mode == "carry":                                                   # 213-11: no gather -> no own-slot read to ablate; core / coda only == ablate_keep
        arms = (("intact", None), ("bank_ablate_slva", "keep"), ("other_ablate", "keep")); res["omitted"].append("self_ablate")
    vattr = "slva_view" if m.mode == "slva" else "gather_view"
    for arm, how in arms:
        A = np.full(len(TE), -1)
        try:
            if how in ("keep", "bank"):
                m.ver_keep = VE.ablate_keep; m.keep_bank_only = how == "bank"
            elif how == "self":
                m.gather_hist_ablate = "self"
            for n in range(1, 13):
                for ch in VE.chunks(np.nonzero(Nn == n)[0], bs):
                    its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev)
                    r_ = m.rollout(x, pad, n + 2, coda_rows={n + 2: torch.arange(len(its), device=dev)})
                    A[ch] = m.answer_logits(r_["coda"][n + 2][1], lens)[:, num].argmax(-1).cpu().numpy(); del r_
        finally:
            m.ver_keep = None; m.keep_bank_only = False; m.gather_hist_ablate = None
        res[arm] = {"acc_n2": {str(n): float((A == Y)[Nn == n].mean()) for n in range(1, 13)}}; print("D3HIST", arm, res[arm], flush=True)
    pairs, _ = VE.make_pairs(TE, cls="A"); cnt = {k: [0, 0] for k in ("newest", "bank", "old")}; npairs = 0
    for key in sorted({(p_["n"], p_["t"]) for p_ in pairs}):
        grp = [p_ for p_ in pairs if (p_["n"], p_["t"]) == key]; n, t = key; K = n + 2
        for ch in VE.chunks(list(range(len(grp))), bs):
            ps = [grp[j] for j in ch]; rec = [TE[p_["i"]] for p_ in ps]; don = [p_["donor"] for p_ in ps]; B = len(ps)
            x, pad, lens = VB.batch_x(rec, dev); xd, pdd, ld = VB.batch_x(don, dev); rows = {K: torch.arange(B, device=dev)}
            assert torch.equal(pad, pdd) and torch.equal(lens, ld), "swap: donor slots != recipient slots"
            rr = m.rollout(x, pad, K, coda_rows=rows); rd = m.rollout(xd, pdd, K, coda_rows=rows)
            y = torch.tensor([it["t"] for it in rec], device=dev); yd = torch.tensor([it["t"] for it in don], device=dev)
            clean = (m.answer_logits(rr["coda"][K][1], lens)[:, num].argmax(-1) == y) & (m.answer_logits(rd["coda"][K][1], ld)[:, num].argmax(-1) == yd)
            rms0 = rr["V"][0].pow(2).mean(-1, keepdim=True).sqrt(); Vr, Vd = rr["V"][t], rd["V"][t]
            kd, nd = (m.gkey(Vd) if m.mode != "carry" else None), m.nrm(Vd, rms0); kr, nr = (m.gkey(Vr) if m.mode != "carry" else None), m.nrm(Vr, rms0); dons = [rd["V"][s_] for s_ in range(1, t + 1)]
            for arm in ("newest", "bank", "old"):
                kw = {}
                try:
                    if m.mode == "carry" and arm == "newest":               # 213-11: the carry is CARRY's newest-version read (iii_carry)
                        kw["edit"] = lambda tt, V: (V, Vd.clone()) if tt == t else None
                    elif m.mode == "carry" and arm == "bank":               # iii_bank
                        kw["edit"] = lambda tt, V: (Vd.clone(), V) if tt == t else None
                    elif arm == "newest":
                        setattr(m, vattr, lambda tt, gk, NV: ((gk[:t] + [kd] + gk[t + 1:], NV[:t] + [nd] + NV[t + 1:]) if tt == t + 1 else (gk, NV)))
                    elif arm == "bank":
                        kw["edit"] = lambda tt, V: Vd.clone() if tt == t else None
                        setattr(m, vattr, lambda tt, gk, NV: ((gk[:t] + [kr] + gk[t + 1:], NV[:t] + [nr] + NV[t + 1:]) if tt >= t + 1 else (gk, NV)))
                    else:
                        kw["ver_sub"] = dons; m.ver_sub_from = t + 2
                    ra = m.rollout(x, pad, K, coda_rows=rows, **kw)
                finally:
                    setattr(m, vattr, None); m.ver_sub_from = None
                A_ = m.answer_logits(ra["coda"][K][1], lens)[:, num].argmax(-1); del ra
                cnt[arm][0] += int(((A_ == yd) & clean).sum()); cnt[arm][1] += int(((A_ == y) & clean).sum())
            npairs += int(clean.sum()); del rr, rd
    sw = {"n_pairs": npairs}
    for arm in ("newest", "bank", "old"):
        sw[arm + "_follow"] = cnt[arm][0] / npairs if npairs else None; sw[arm + "_own"] = cnt[arm][1] / npairs if npairs else None
    res["swap"] = sw; res["def"] = part_d3hist.__doc__; print("D3HIST swap", sw, flush=True)
    return res


@torch.no_grad()
def part_hold(m, TE, dev, bs):
    """note 213-14 S0-4 COPY-HOLD (on N1; $0, fp32, eval_bs 32). One K' 24 rollout per chunk and arm (vbank_eval.part_acc's chunks), coda rows n + 2 and 24:
    intact; hold = rollout(edit): at every pass t >= 2, real slots whose argmax R(V_t[j]) equals the decode of the previous stored entry S_{t-1}[j] and is
    not VOID get stored entry AND carry := S_{t-1}[j] (the previous stored value, possibly already held; S_{t-1} decodes as V_{t-1} at every slot that was
    held, since a slot is held only when its class repeats); hold_v = the same with VOID slots included. RS items = exact decode at t = n - r (part_rs's
    classify, from the intact rollout's V_t); cells n 9-13 x r {4, 6}: intact / hold / hold_v = acc at K' = n + 2 on them; restart = part_rs's compact
    fresh run at K' = r + 2 over part_rs's full job lists for r in {4, 6} (same batches -> bitwise eval_rs). acc_n2 / acc_k24 per arm, n 1..16;
    exposure (by n) = share of items with >= 1 held slot-event (passes <= 24) and mean held slot-events per item. Note 213-15: "unchanged" = argmax R(computed
    V_t[j]) == argmax R(stored S_{t-1}[j]) (!= VOID unless hold_v); sticky (a held slot keeps its entry while the decode repeats)."""
    num = torch.tensor(N.NUM_IDS, device=dev); Nn = np.array([it["n"] for it in TE]); Y = np.array([it["t"] for it in TE])
    ARMS = ("intact", "hold", "hold_v"); A2 = {a_: np.full(len(TE), -1) for a_ in ARMS}; A24 = {a_: np.full(len(TE), -1) for a_ in ARMS}; RR = (4, 6)
    jobs = {r: [] for r in RR}; expo = {a_: {} for a_ in ARMS[1:]}
    for n in range(1, 17):
        for ch in VE.chunks(np.nonzero(Nn == n)[0], bs):
            its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev); B = len(its); rows = torch.arange(B, device=dev)
            for arm in ARMS:
                ex = {"any": torch.zeros(B, dtype=torch.bool, device=dev), "ev": 0, "S": None, "D": None}
                edit = None
                if arm != "intact":
                    def edit(t, V, ex=ex, pad=pad, inc_void=(arm == "hold_v")):
                        d = m.rec_logits(V).argmax(-1)
                        if t >= 2:
                            z = (d == ex["D"]) & pad
                            if not inc_void:
                                z = z & (d != VB.VOID)
                            ex["any"] |= z.any(1); ex["ev"] += int(z.sum()); V = torch.where(z[..., None], ex["S"], V); d = torch.where(z, ex["D"], d)
                        ex["S"], ex["D"] = V, d
                        return V
                r_ = m.rollout(x, pad, 24, coda_rows={n + 2: rows, 24: rows}, edit=edit)
                l2 = m.answer_logits(r_["coda"][n + 2][1], lens); l24 = m.answer_logits(r_["coda"][24][1], lens)
                assert bool(torch.isfinite(l2).all() and torch.isfinite(l24).all()), arm
                A2[arm][ch] = l2[:, num].argmax(-1).cpu().numpy(); A24[arm][ch] = l24[:, num].argmax(-1).cpu().numpy()
                if arm == "intact":
                    for r in RR:
                        t = n - r
                        if t < 1:
                            continue
                        dec = m.rec_logits(r_["V"][t]).argmax(-1).cpu().numpy()
                        for k_, j in enumerate(ch):
                            if classify(VB.rerender(list(dec[k_, :int(lens[k_])])), N.cot_steps(TE[j]["root"])[t]) == "exact":
                                jobs[r].append((int(j), t))
                else:
                    e_ = expo[arm].setdefault(str(n), [0, 0, 0]); e_[0] += int(ex["any"].sum()); e_[1] += ex["ev"]; e_[2] += B
                del r_
        print("HOLD n", n, {a_: round(float((A2[a_] == Y)[Nn == n].mean()), 3) for a_ in ARMS}, flush=True)
    rest = {}
    for r in RR:                                                            # part_rs's restart, same job order and batches
        js = jobs[r]
        for c0 in range(0, len(js), bs):
            part = js[c0:c0 + bs]; its = [{"syms": N.render(reduce_k(TE[j]["root"], t)), "t": TE[j]["t"]} for j, t in part]
            x, pad, lens = VB.batch_x(its, dev); rr = m.rollout(x, pad, r + 2, coda_rows={r + 2: torch.arange(len(its), device=dev)})
            a_ = m.answer_logits(rr["coda"][r + 2][1], lens)[:, num].argmax(-1).cpu().numpy(); del rr
            for k_, (j, t) in enumerate(part):
                rest[(j, t)] = bool(a_[k_] == TE[j]["t"])
    mean_ = lambda v: float(np.mean(v)) if len(v) else None  # noqa: E731
    cells = {}
    for n in range(9, 14):
        for r in RR:
            tt = n - r; js = [j for (j, t) in jobs[r] if t == tt and TE[j]["n"] == n]
            c_ = {"n_items": len(js), "restart": mean_([rest[(j, tt)] for j in js])}; c_.update({a_: mean_([bool(A2[a_][j] == Y[j]) for j in js]) for a_ in ARMS})
            cells["%d_%d" % (n, r)] = c_
    by_n = lambda a_: {str(n): float((a_ == Y)[Nn == n].mean()) for n in range(1, 17)}  # noqa: E731
    return {"cells": cells, "acc_n2": {a_: by_n(A2[a_]) for a_ in ARMS}, "acc_k24": {a_: by_n(A24[a_]) for a_ in ARMS},
            "exposure": {a_: {n: {"items_exposed": v[0] / v[2], "held_events_per_item": v[1] / v[2]} for n, v in d.items()} for a_, d in expo.items()}, "def": part_hold.__doc__}


@torch.no_grad()
def part_stepbyt(m, TE, dev, bs):
    """note 213-18 STEP-BY-T ($0): eval_lfk's intact "step" events split by pass index t: step_by_t {n: {t: {events, strict_exact, voiddrop_exact}}}, n 9-16,
    t 2..n; an event = pass t whose input V_{t-1} decodes STRICT exact (VOIDs in place); strict_exact = every real slot of R(V_t) == E_t; the same chunks,
    rollout (K' = n) and decodes as _lfk_core's intact arm. step = the per-n sums (the runner checks them against eval_lfk.json's step block). Audit 213-18:
    per (n, t) also slot_<changed | copy | void>[_ge33] {events, strict_correct, acc} over the events' real slots (kinds as in lfk; _ge33 = slots j >= 33)."""
    Nn = np.array([it["n"] for it in TE]); out = {}; tot = {}
    for n in range(9, 17):
        d_ = out.setdefault(str(n), {})
        for ch in VE.chunks(np.nonzero(Nn == n)[0], bs):
            its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev); E = VB.rec_targets(its, n, x.shape[1], dev)
            r_ = m.rollout(x, pad, n); dec = [m.rec_logits(r_["V"][t]).argmax(-1) for t in range(n + 1)]; del r_
            for t in range(2, n + 1):
                ok_in = ((dec[t - 1] == E[:, t - 1]) | ~pad).all(1)
                if not bool(ok_in.any()):
                    continue
                Et = E[:, t]; strict = ((dec[t] == Et) | ~pad).all(1); dc = dec[t].cpu().numpy(); ec = Et.cpu().numpy(); c_ = d_.setdefault(str(t), {"events": 0, "strict_exact": 0, "voiddrop_exact": 0})
                Ep = E[:, t - 1]; chg = Et != Ep; cor = dec[t] == Et; base = pad & ok_in[:, None]   # Audit 213-18: slot-level strict by kind (all slots / j >= 33)
                col = torch.arange(x.shape[1], device=dev)[None, :]
                for kd, km in (("changed", chg), ("copy", ~chg & (Et != VB.VOID)), ("void", ~chg & (Et == VB.VOID))):
                    for tag, cm in ((kd, base & km), (kd + "_ge33", base & km & (col >= 33))):
                        z = c_.setdefault("slot_" + tag, [0, 0]); z[0] += int(cm.sum()); z[1] += int((cm & cor).sum())
                for i in np.nonzero(ok_in.cpu().numpy())[0]:
                    L = int(lens[i]); c_["events"] += 1; c_["strict_exact"] += int(strict[i]); c_["voiddrop_exact"] += int(VB.rerender(list(dc[i, :L])) == VB.rerender(list(ec[i, :L])))
        tot[str(n)] = {k: sum(v[k] for v in d_.values()) for k in ("events", "strict_exact", "voiddrop_exact")}; print("STEPBYT n", n, tot[str(n)], flush=True)
        for v in d_.values():                                               # slot_<kind>[_ge33]: [slot-events, strict correct] -> + acc
            for k_ in [k for k in v if k.startswith("slot_")]:
                v[k_] = {"events": v[k_][0], "strict_correct": v[k_][1], "acc": v[k_][1] / v[k_][0] if v[k_][0] else None}
            v["changed_events"] = v["slot_changed"]["events"]; v["changed_correct"] = v["slot_changed"]["strict_correct"]   # score_s213 a19fb453's field names
    return {"step_by_t": out, "step": tot, "def": part_stepbyt.__doc__}


@torch.no_grad()
def part_hist24(m, TE, dev, bs):
    """note 213-20 HIST-AT-K24 ($0): eval_d3hist's arms (intact / bank_ablate_slva / self_ablate / other_ablate, the same per-mode mapping as part_d3hist;
    CARRY omits self_ablate and its other_ablate = ablate_keep) read at K' = n + 2 AND K' = 24 from one K' 24 rollout per chunk (STAB-HIST's method;
    vbank_eval.part_acc's chunks), n 3..8 -> {arm: {acc_n2 {n}, acc_k24 {n}}} + "omitted". The intact arm must equal eval_vb.json acc_n2 / acc_k['24'].
    Note 213-21 DUP: dup_newest = vbank dup_newest (at pass t, every reader's versions 1..t-2 read as the newest stored V_{t-1}; entry count / positions kept).
    Note 213-22: dup_foreign = versions 1..t-2 read as a DONOR's stored S_{t-1} (donor = the next item of the same n in test order, wrapping; its own intact
    K' 24 rollout; recipients and donors padded together so slots align by position - a shorter donor contributes its padding rows at the recipient's
    extra slots; the recipient batch is then padded to the joint length, so this arm's batches differ from eval_vb's)."""
    num = torch.tensor(N.NUM_IDS, device=dev); Nn = np.array([it["n"] for it in TE]); Y = np.array([it["t"] for it in TE]); res = {"omitted": []}
    arms = (("intact", None), ("bank_ablate_slva", "keep"), ("self_ablate", "self"), ("other_ablate", "bank"), ("dup_newest", "dup"), ("dup_foreign", "dupf"))
    if m.mode == "carry":
        arms = (("intact", None), ("bank_ablate_slva", "keep"), ("other_ablate", "keep"), ("dup_newest", "dup"), ("dup_foreign", "dupf")); res["omitted"].append("self_ablate")
    for arm, how in arms:
        A2 = np.full(len(TE), -1); A24 = np.full(len(TE), -1)
        try:
            if how in ("keep", "bank"):
                m.ver_keep = VE.ablate_keep; m.keep_bank_only = how == "bank"
            elif how == "self":
                m.gather_hist_ablate = "self"
            elif how == "dup":                                              # 213-21 content control: versions 1..t-2 := V_{t-1} at every reader
                m.dup_newest = True
            for n in range(3, 9):
                pool = np.nonzero(Nn == n)[0]; nxt = {int(j): int(pool[(q + 1) % len(pool)]) for q, j in enumerate(pool)}   # 213-22 donor: next same-n item, wrapping
                for ch in VE.chunks(pool, bs):
                    its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev); rows = torch.arange(len(its), device=dev)
                    if how == "dupf":                                       # recipients + donors padded together (slots aligned by position), donor intact S_0..S_24
                        B = len(its); xa, pa_, la = VB.batch_x(its + [TE[nxt[int(j)]] for j in ch], dev); x, pad, lens = xa[:B], pa_[:B], la[:B]
                        m.dup_foreign = m.rollout(xa[B:], pa_[B:], 24)["S"]
                    r_ = m.rollout(x, pad, 24, coda_rows={n + 2: rows, 24: rows}); m.dup_foreign = None
                    A2[ch] = m.answer_logits(r_["coda"][n + 2][1], lens)[:, num].argmax(-1).cpu().numpy()
                    A24[ch] = m.answer_logits(r_["coda"][24][1], lens)[:, num].argmax(-1).cpu().numpy(); del r_
        finally:
            m.ver_keep = None; m.keep_bank_only = False; m.gather_hist_ablate = None; m.dup_newest = False; m.dup_foreign = None
        by_n = lambda a_: {str(n): float((a_ == Y)[Nn == n].mean()) for n in range(3, 9)}  # noqa: E731
        res[arm] = {"acc_n2": by_n(A2), "acc_k24": by_n(A24)}; print("HIST24", arm, res[arm], flush=True)
    res["def"] = part_hist24.__doc__
    return res


@torch.no_grad()
def part_ageproj(m, TE, dev, bs):
    """note 213-27 AGE-PROJ ($0; D3-T). FIT: §170 train-distribution items n 3..8, 300 per n (np rng 21327, N.generate with exactly n operators; test set +
    SERIAL pool excluded as in the trainer), rolled out to K' 20 with input_edit recording the pass input U_t (after SLVA / the carry's N(.), before block 6)
    at every real slot. Ridge regression of t on U_t over IDLE passes t in [n + 1, 20], with an unpenalised intercept, 5 folds by item (item index mod 5),
    lambda in {1e-2, 1, 1e2, 1e4} by pooled held-out R^2, computed exactly from per-fold fp64 sufficient statistics ([U, 1]^T [U, 1], [U, 1]^T t, sum t^2);
    the final w is refit on all folds; w_hat = w / ||w||. c5 = mean w_hat . U over in-range computing passes t = 5 of items n 6..8 at changed or copy real
    slots (E_5 != E_4, or unchanged and non-VOID). ARMS (input_edit at every pass t >= 2, every real slot; §170 test set n 1..14, vbank_eval.part_acc's chunks;
    one rollout per chunk to K' = max(10, n + 2), read at K' 10 and K' n + 2): intact; proj_out U - (U . w_hat) w_hat; age_clamp U - (U . w_hat - c5) w_hat;
    rand_clamp (note 213-28, size-matched): r = a random unit vector (np rng 21328, Gaussian, orthogonalised to w_hat); per slot and pass
    dU = -|U . w_hat - c5| * sign(U . r - c5_r) * r (toward its own t = 5 mean c5_r; sign(0) = +1), so |dU_rand| = |dU_age| exactly (asserted <= 1e-4 rel).
    fit also reports (descriptive, 213-28) r2_t_minus_n (target t - n on [U, 1]) and r2_t_given_n (t on [U, n, 1], n unpenalised); the rule stays R^2(t)."""
    num = torch.tensor(N.NUM_IDS, device=dev); D = m.D; LAMS = (1e-2, 1.0, 1e2, 1e4)
    excl = {tuple(it["syms"]) for it in TE}; r_sp = np.random.default_rng(17300)
    for _ in range(2200):
        excl.add(tuple(N.render(N.generate(r_sp, 8)[0])))
    rng = np.random.default_rng(21327); fit_items = []
    for n in range(3, 9):
        got = 0
        while got < 300:
            r_, y_ = N.generate(rng, n); s_ = N.render(r_)
            if n_ops(r_) == n and tuple(s_) not in excl:
                fit_items.append({"syms": s_, "t": y_, "root": r_, "n": n}); got += 1
    A3 = torch.zeros(5, D + 2, D + 2, dtype=torch.float64, device=dev); bt_ = torch.zeros(5, D + 2, dtype=torch.float64, device=dev)   # columns [U (D), n, 1]; target t
    bm_ = torch.zeros(5, D + 2, dtype=torch.float64, device=dev); yy = torch.zeros(5, dtype=torch.float64); sy = torch.zeros(5, dtype=torch.float64)  # bm_: target t - n
    yym = torch.zeros(5, dtype=torch.float64); sym = torch.zeros(5, dtype=torch.float64); cnt = torch.zeros(5, dtype=torch.float64); U5 = []
    for n in range(3, 9):
        idx = [i for i, it in enumerate(fit_items) if it["n"] == n]
        for c0 in range(0, len(idx), bs):
            ch = idx[c0:c0 + bs]; its = [fit_items[i] for i in ch]; x, pad, lens = VB.batch_x(its, dev); E = VB.rec_targets(its, 20, x.shape[1], dev)
            fold = torch.tensor([i % 5 for i in ch], device=dev); cap = {}

            def rec(t, U, cap=cap):
                cap[t] = U.detach(); return U
            m.input_edit = rec
            try:
                m.rollout(x, pad, 20)
            finally:
                m.input_edit = None
            for t in range(n + 1, 21):
                U = cap[t].double()
                for f in range(5):
                    sel = pad & (fold == f)[:, None]
                    if not bool(sel.any()):
                        continue
                    k_ = int(sel.sum()); X = torch.cat([U[sel], torch.full((k_, 1), float(n), dtype=torch.float64, device=dev), torch.ones(k_, 1, dtype=torch.float64, device=dev)], 1)
                    A3[f] += X.T @ X; bt_[f] += X.T @ torch.full((k_,), float(t), dtype=torch.float64, device=dev); bm_[f] += X.T @ torch.full((k_,), float(t - n), dtype=torch.float64, device=dev)
                    yy[f] += float(t) ** 2 * k_; sy[f] += float(t) * k_; yym[f] += float(t - n) ** 2 * k_; sym[f] += float(t - n) * k_; cnt[f] += k_
            if n >= 6:                                                      # t = 5 is in range (computing) for n 6..8
                E5, E4 = E[:, 5], E[:, 4]; sel = pad & ((E5 != E4) | (E5 != VB.VOID)); U5.append(cap[5][sel].double().cpu())
            del cap
    A3, bt_, bm_ = A3.cpu(), bt_.cpu(), bm_.cpu(); keep = list(range(D)) + [D + 1]                       # [U, 1] (the rule's design) drops the n column

    def cv(A, b, yy_, sy_, unpen):
        """pooled 5-fold held-out R^2 per lambda (exact, from fp64 statistics); unpen = unpenalised columns. -> ({lam: r2}, best lam)."""
        I_ = torch.eye(A.shape[1], dtype=torch.float64)
        for j in unpen:
            I_[j, j] = 0.0
        out = {}
        for lam in LAMS:
            sse = sst = 0.0
            for f in range(5):
                w = torch.linalg.solve(A.sum(0) - A[f] + lam * I_, b.sum(0) - b[f])
                sse += float(yy_[f] - 2 * w @ b[f] + w @ A[f] @ w); sst += float(yy_[f] - sy_[f] ** 2 / cnt[f])
            out[str(lam)] = 1 - sse / sst
        return out, max(LAMS, key=lambda l_: out[str(l_)]), I_
    A = A3[:, keep][:, :, keep]; bvec = bt_[:, keep]
    r2s, best, I_ = cv(A, bvec, yy, sy, [D])                                 # the rule: R^2(t) on [U, 1]
    r2m, bestm, _ = cv(A, bm_[:, keep], yym, sym, [D])                      # descriptive: target t - n
    r2g, bestg, _ = cv(A3, bt_, yy, sy, [D, D + 1])                         # descriptive: t with n as an (unpenalised) covariate
    w = torch.linalg.solve(A.sum(0) + best * I_, bvec.sum(0)); wv = w[:D]; wn = float(wv.norm()); w_hat = (wv / wv.norm()).float()
    U5 = torch.cat(U5, 0).float(); c5 = float((U5 @ w_hat).mean())
    g = np.random.default_rng(21328); rv = torch.tensor(g.standard_normal(D), dtype=torch.float32); rv = rv - (rv @ w_hat) * w_hat; rv = rv / rv.norm()
    c5r = float((U5 @ rv).mean()); fit = {"r2_heldout": r2s[str(best)], "r2_by_lambda": r2s, "lambda": best, "n_samples": int(cnt.sum()), "c5": c5, "w_norm": wn,
                                          "r2_t_minus_n": r2m[str(bestm)], "r2_t_minus_n_by_lambda": r2m, "r2_t_given_n": r2g[str(bestg)], "r2_t_given_n_by_lambda": r2g,
                                          "c5_rand": c5r, "n_t5_slots": int(U5.shape[0]), "w_hat_dot_rand": float(rv @ w_hat)}
    print("AGEPROJ fit", fit, flush=True)
    wd, rd = w_hat.to(dev), rv.to(dev)
    Nn = np.array([it["n"] for it in TE]); Y = np.array([it["t"] for it in TE]); arms = {}; chk = {"max_rel_size_mismatch": 0.0}

    def mk(kind, pad):
        def f(t, U):
            if t < 2:
                return U
            if kind == "proj_out":
                U2 = U - (U @ wd)[..., None] * wd
            elif kind == "age_clamp":
                U2 = U - ((U @ wd) - c5)[..., None] * wd
            else:                                                           # 213-28: |dU| = |U . w_hat - c5| (age_clamp's size), along r toward c5_r
                mag = ((U @ wd) - c5).abs(); sg = torch.where((U @ rd) - c5r >= 0, 1.0, -1.0).to(U.dtype)
                dU = (mag * sg)[..., None] * rd; U2 = U - dU                       # the size check is on dU itself: (U - dU) - U cancels in fp32
                dn = dU.norm(dim=-1); da = mag * float(wd.norm())                   # (|dU_age| = mag * |w_hat| = mag)
                chk["max_rel_size_mismatch"] = max(chk["max_rel_size_mismatch"], float(((dn - da).abs() / da.clamp_min(1e-12))[pad].max()))
            return torch.where(pad[..., None], U2.to(U.dtype), U)
        return f
    for arm in ("intact", "proj_out", "age_clamp", "rand_clamp"):
        A10 = np.full(len(TE), -1); A2 = np.full(len(TE), -1)
        for n in range(1, 15):
            for ch in VE.chunks(np.nonzero(Nn == n)[0], bs):
                its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev); rows = torch.arange(len(its), device=dev); K = max(10, n + 2)
                m.input_edit = None if arm == "intact" else mk(arm, pad)
                try:
                    r_ = m.rollout(x, pad, K, coda_rows={10: rows, n + 2: rows})
                finally:
                    m.input_edit = None
                A10[ch] = m.answer_logits(r_["coda"][10][1], lens)[:, num].argmax(-1).cpu().numpy(); A2[ch] = m.answer_logits(r_["coda"][n + 2][1], lens)[:, num].argmax(-1).cpu().numpy(); del r_
        by_n = lambda a_: {str(n): float((a_ == Y)[Nn == n].mean()) for n in range(1, 15)}  # noqa: E731
        arms[arm] = {"acc_k10": by_n(A10), "acc_n2": by_n(A2)}; print("AGEPROJ", arm, {k: [round(v, 3) for v in d_.values()] for k, d_ in arms[arm].items()}, flush=True)
    assert chk["max_rel_size_mismatch"] <= 1e-4, chk                       # 213-28: |dU_rand| == |dU_age| per slot and pass
    return {"fit": fit, "arms": arms, "rand_size_check": chk, "def": part_ageproj.__doc__}


@torch.no_grad()
def part_agerevert(m, TE, dev, bs):
    """D4_v1 §7 S0-3 AGE-REVERT + AGE-MCT ($0; D3-T). §170 test items n 1..8 (vbank_eval.part_acc's chunks). D in {2, 4}: passes 1..D compute V_t and then
    store an EDITED entry (rollout(edit=...); SLVA / the bank read the stored entry); normal passes follow; acc read at K' D + n - 1 (acc_k_pre) and
    K' D + n + 2 (acc_n2) from one rollout per chunk. REVERT: at pass t <= D, real slots whose argmax R(V_t[j]) differs from the decode of the previous
    STORED entry S_{t-1}[j] are set back to S_{t-1}[j] (S_0 = V_0; under CARRY the carry gets the same entry). MCT: at pass t <= D the stored entry is
    m.mct(V_t, E_0, pad, mu 1.0) (key "age_mct"; the no_grad minimum-norm edit in R's row space, so R decodes E_0 at every real slot; every slot keeps its aged vector).
    intact = acc at K' n + 2 (the runner gates it against eval_vb.json acc_n2, bitwise on the eval card). Exposure: reverted slot-events per item (REVERT),
    and the share of MCT stored entries decoding E_0 at every real slot."""
    num = torch.tensor(N.NUM_IDS, device=dev); Nn = np.array([it["n"] for it in TE]); Y = np.array([it["t"] for it in TE])
    res = {"intact": {}, "revert": {}, "age_mct": {}, "exposure": {}}; A0 = np.full(len(TE), -1)   # "age_mct": main() adds a top-level "mct" (the run's flag)
    for n in range(1, 9):
        for ch in VE.chunks(np.nonzero(Nn == n)[0], bs):
            its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev); rows = torch.arange(len(its), device=dev)
            r_ = m.rollout(x, pad, n + 2, coda_rows={n + 2: rows}); A0[ch] = m.answer_logits(r_["coda"][n + 2][1], lens)[:, num].argmax(-1).cpu().numpy(); del r_
    res["intact"]["acc_n2"] = {str(n): float((A0 == Y)[Nn == n].mean()) for n in range(1, 9)}
    for arm in ("revert", "age_mct"):
        for D in (2, 4):
            Ap = np.full(len(TE), -1); A2 = np.full(len(TE), -1); ex = {"events": 0, "items": 0, "mct_exact": 0, "mct_entries": 0}
            for n in range(1, 9):
                for ch in VE.chunks(np.nonzero(Nn == n)[0], bs):
                    its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev); B = len(its); rows = torch.arange(B, device=dev); K = D + n + 2
                    E0 = VB.rec_targets(its, 0, x.shape[1], dev)[:, 0]; st = {"prev": m.rollout(x, pad, 0)["V"][0]}

                    def edit(t, V, st=st, pad=pad, E0=E0, D=D, arm=arm, ex=ex):
                        if t > D:
                            return None
                        if arm == "revert":
                            ch_ = (m.rec_logits(V).argmax(-1) != m.rec_logits(st["prev"]).argmax(-1)) & pad
                            ex["events"] += int(ch_.sum()); out = torch.where(ch_[..., None], st["prev"], V)
                        else:
                            out = m.mct(V, E0, pad, 1.0)[0].to(V.dtype); ok = ((m.rec_logits(out).argmax(-1) == E0) | ~pad).all(1)
                            ex["mct_exact"] += int(ok.sum()); ex["mct_entries"] += int(ok.numel())
                        st["prev"] = out
                        return out
                    r_ = m.rollout(x, pad, K, coda_rows={D + n - 1: rows, K: rows}, edit=edit); ex["items"] += B
                    lp = m.answer_logits(r_["coda"][D + n - 1][1], lens); l2 = m.answer_logits(r_["coda"][K][1], lens)
                    assert bool(torch.isfinite(lp).all() and torch.isfinite(l2).all()), (arm, D)
                    Ap[ch] = lp[:, num].argmax(-1).cpu().numpy(); A2[ch] = l2[:, num].argmax(-1).cpu().numpy(); del r_
            by_n = lambda a_: {str(n): float((a_ == Y)[Nn == n].mean()) for n in range(1, 9)}  # noqa: E731
            res[arm][str(D)] = {"acc_k_pre": by_n(Ap), "acc_n2": by_n(A2)}
            res["exposure"]["%s_%d" % (arm, D)] = ({"reverted_slot_events_per_item": ex["events"] / ex["items"]} if arm == "revert"
                                                   else {"mct_entries_exact_share": ex["mct_exact"] / max(1, ex["mct_entries"])})
            print("AGEREVERT", arm, D, {k: [round(v, 3) for v in d_.values()] for k, d_ in res[arm][str(D)].items()}, res["exposure"]["%s_%d" % (arm, D)], flush=True)
    res["def"] = part_agerevert.__doc__
    return res


@torch.no_grad()
def part_carrier(m, TE, dev, bs, rd):
    """D4_v1 §7 S0-2 CARRIER ($0; D3-T). prestall s in {4, 8}: s copies of V_0 stored before pass 1 (rollout prefix = [V_0] * s; INSERT semantics, the
    CARRY stays V_0; SLVA / the gather read the stored copies), then n + 2 real passes (K' = s + n + 2), n 1..12 (vbank_eval.part_acc's chunks); intact =
    acc at K' n + 2 (gated against eval_vb by the runner). foreign = part_foreign at o 6 (its own pairs: n 4..12, length-matched partner, rng 21500 - the
    same partner draws as eval_foreign's (3, 6) run), insert and replace."""
    num = torch.tensor(N.NUM_IDS, device=dev); Nn = np.array([it["n"] for it in TE]); Y = np.array([it["t"] for it in TE]); res = {"prestall": {}}
    A0 = np.full(len(TE), -1); AS = {s_: np.full(len(TE), -1) for s_ in (4, 8)}
    for n in range(1, 13):
        for ch in VE.chunks(np.nonzero(Nn == n)[0], bs):
            its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev); rows = torch.arange(len(its), device=dev)
            r_ = m.rollout(x, pad, n + 2, coda_rows={n + 2: rows}); A0[ch] = m.answer_logits(r_["coda"][n + 2][1], lens)[:, num].argmax(-1).cpu().numpy(); V0 = r_["V"][0]; del r_
            for s_ in (4, 8):
                K = s_ + n + 2; r_ = m.rollout(x, pad, K, coda_rows={K: rows}, prefix=[V0] * s_)
                AS[s_][ch] = m.answer_logits(r_["coda"][K][1], lens)[:, num].argmax(-1).cpu().numpy(); del r_
    by_n = lambda a_: {str(n): float((a_ == Y)[Nn == n].mean()) for n in range(1, 13)}  # noqa: E731
    res["intact"] = {"acc_n2": by_n(A0)}; res["prestall"] = {str(s_): {"acc_n2": by_n(AS[s_])} for s_ in (4, 8)}
    print("CARRIER prestall", {k: [round(v, 3) for v in d_["acc_n2"].values()] for k, d_ in res["prestall"].items()}, flush=True)
    res["foreign_o6"] = part_foreign(m, TE, dev, bs, rd, os_=(6,)); res["def"] = part_carrier.__doc__
    return res


def _parse_tree(syms):
    """decoded symbols -> a nested170 Node tree (shunting-yard; precedence + left associativity + brackets), or None if the symbols are not one valid
    expression over values 0..P-1 (empty, unbalanced brackets, adjacent operands / operators, unknown tokens). Redundant brackets parse."""
    out, st = [], []
    prev = "op"                                                             # what the previous token allows next: "op" -> an operand or "(" may follow

    def red():
        if len(out) < 2:
            raise ValueError
        b, a = out.pop(), out.pop(); out.append(N.Node(op=st.pop(), l=a, r=b))
    try:
        for s_ in syms:
            w = s_.strip()
            if w.isdigit():
                if prev != "op" or not 0 <= int(w) < N.P:
                    return None
                out.append(N.Node(v=int(w))); prev = "val"
            elif w in N.PREC:
                if prev != "val":
                    return None
                while st and st[-1] in N.PREC and N.PREC[st[-1]] >= N.PREC[w]:
                    red()
                st.append(w); prev = "op"
            elif w == "(":
                if prev != "op":
                    return None
                st.append(w)
            elif w == ")":
                if prev != "val":
                    return None
                while st and st[-1] != "(":
                    red()
                if not st:
                    return None
                st.pop()
            else:
                return None
        if prev != "val":
            return None
        while st:
            if st[-1] == "(":
                return None
            red()
    except (ValueError, IndexError):
        return None
    return out[0] if len(out) == 1 else None


@torch.no_grad()
def part_chainlabel(m, TE, dev, bs):
    """note 213-33 CHAIN-LABEL ($0): §170 test items n 9..16. Per item cur = its expression; loop: a 1-pass rollout on batch_x(render(cur)) (batched by
    rendered length - no padding effects, as long_step), dec = argmax R(V_1), s = VB.rerender(dec) (VOIDs dropped); s must parse into one valid expression
    (_parse_tree) else FAIL; cur = the parsed tree; stop when cur has no operator (label = its value) or after 24 steps (FAIL). Always the model's own decode.
    chain_exact: every step's s equalled render(reduce_k(true current state, 1)) (so the chain followed the true trajectory to the end). agree: label ==
    the latent answer at K' n + 2 (recomputed here, vbank_eval.part_acc's chunks). by_n {items, precision (label == y; FAIL wrong), fail, chain_exact,
    agree, prec_agree (precision among items whose label agrees with the latent answer), n_agree, steps_mean}. Note 213-34: precision_strict (label == y AND
    exactly n steps), right_value_wrong_steps (label == y, steps != n), steps_minus_n {k: count} (k = steps - n; FAIL chains under "fail")."""
    num = torch.tensor(N.NUM_IDS, device=dev); Nn = np.array([it["n"] for it in TE]); Y = np.array([it["t"] for it in TE])
    idx = np.nonzero(Nn >= 9)[0]; lat = {}
    for n in range(9, 17):
        for ch in VE.chunks(np.nonzero(Nn == n)[0], bs):
            its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev)
            r_ = m.rollout(x, pad, n + 2, coda_rows={n + 2: torch.arange(len(its), device=dev)}); a_ = m.answer_logits(r_["coda"][n + 2][1], lens)[:, num].argmax(-1).cpu().numpy(); del r_
            for k_, j in enumerate(ch):
                lat[int(j)] = int(a_[k_])
    cur = {int(j): N._copy(TE[j]["root"]) for j in idx}; true = {int(j): N._copy(TE[j]["root"]) for j in idx}; exact = {int(j): True for j in idx}
    label, steps = {}, {int(j): 0 for j in idx}; active = set(cur)
    for _ in range(24):
        if not active:
            break
        order = sorted(active, key=lambda j: len(N.render(cur[j]))); nxt = set()
        for c0 in range(0, len(order), bs):
            part = order[c0:c0 + bs]; its = [{"syms": N.render(cur[j])} for j in part]; x, pad, lens = VB.batch_x(its, dev)
            rr = m.rollout(x, pad, 1); dec = m.rec_logits(rr["V"][1]).argmax(-1).cpu().numpy(); del rr
            for k_, j in enumerate(part):
                s_ = VB.rerender(list(dec[k_, :int(lens[k_])])); steps[j] += 1
                if true[j].op is not None:
                    tn = reduce_k(true[j], 1); exact[j] = exact[j] and s_ == N.render(tn); true[j] = tn
                else:
                    exact[j] = False
                t_ = _parse_tree(s_)
                if t_ is None:
                    label[j] = None; exact[j] = False; continue
                cur[j] = t_
                if t_.op is None:
                    label[j] = t_.v
                else:
                    nxt.add(j)
        active = nxt
    for j in active:                                                        # still operating after 24 steps
        label[j] = None; exact[j] = False
    out = {}
    for n in range(9, 17):
        js = [int(j) for j in np.nonzero(Nn == n)[0]]; L = [label.get(j) for j in js]
        ok = [l_ is not None and l_ == TE[j]["t"] for l_, j in zip(L, js)]; ag = [l_ is not None and l_ == lat[j] for l_, j in zip(L, js)]
        okag = [o for o, a_ in zip(ok, ag) if a_]; hist = {}
        for l_, j in zip(L, js):                                            # 213-34: steps - n per chain; FAIL chains under "fail"
            k_ = "fail" if l_ is None else str(steps[j] - n); hist[k_] = hist.get(k_, 0) + 1
        strict = [o and steps[j] == n for o, j in zip(ok, js)]; rvws = [o and steps[j] != n for o, j in zip(ok, js)]
        out[str(n)] = {"items": len(js), "precision": float(np.mean(ok)), "precision_strict": float(np.mean(strict)), "right_value_wrong_steps": float(np.mean(rvws)),
                       "steps_minus_n": dict(sorted(hist.items(), key=lambda kv: (kv[0] == "fail", int(kv[0]) if kv[0] != "fail" else 0))), "fail": float(np.mean([l_ is None for l_ in L])), "chain_exact": float(np.mean([exact[j] for j in js])),
                       "agree": float(np.mean(ag)), "prec_agree": float(np.mean(okag)) if okag else None, "n_agree": len(okag),
                       "latent_acc": float(np.mean([lat[j] == TE[j]["t"] for j in js])), "steps_mean": float(np.mean([steps[j] for j in js]))}
        print("CHAIN n", n, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[str(n)].items()}, flush=True)
    return {"by_n": out, "def": part_chainlabel.__doc__}


@torch.no_grad()
def part_chainlabel_fresh(m, TE, dev, bs, n_items=1000):
    """note 213-37 CHAIN-LABEL-FRESH ($0): part_chainlabel's chain (the model's own 1-pass decode re-rooted fresh at every hop, _parse_tree, length-grouped
    batches) on FRESH items: nested170.generate(rng, n), np rng 21337, 1,000 per n, n 9..20, excluding the §170 test set and the SERIAL pool by exact
    rendered syms (the trainer's EXCL) - a collision is redrawn and counted (collide_rate = collisions / draws). Step cap n + 4 (FAIL beyond). Gold
    (the generator's value) is for evaluation only. by_n {items, precision_strict (label == gold AND exactly n steps), fail, right_value_wrong_steps,
    steps_minus_n {k | "fail": count}, keep (every step parsed AND exactly n steps, whatever the label), prec_kept (precision among kept), collide_rate,
    sec_per_item}."""
    excl = {tuple(it["syms"]) for it in TE}; r_sp = np.random.default_rng(17300)
    for _ in range(2200):
        excl.add(tuple(N.render(N.generate(r_sp, 8)[0])))
    rng = np.random.default_rng(21337); out = {}
    for n in range(9, 21):
        t0 = time.time(); roots, gold, draws = [], [], 0
        while len(roots) < n_items:                                           # 1,000 as registered (n_items: smokes only)
            r_, y_ = N.generate(rng, n); draws += 1
            if tuple(N.render(r_)) in excl:
                continue
            roots.append(r_); gold.append(y_)
        cap = n + 4; cur = {j: N._copy(r_) for j, r_ in enumerate(roots)}; label, steps = {}, {j: 0 for j in cur}; active = set(cur)
        for _ in range(cap):
            if not active:
                break
            order = sorted(active, key=lambda j: len(N.render(cur[j]))); nxt = set()
            for c0 in range(0, len(order), bs):
                part = order[c0:c0 + bs]; its = [{"syms": N.render(cur[j])} for j in part]; x, pad, lens = VB.batch_x(its, dev)
                rr = m.rollout(x, pad, 1); dec = m.rec_logits(rr["V"][1]).argmax(-1).cpu().numpy(); del rr
                for k_, j in enumerate(part):
                    steps[j] += 1; t_ = _parse_tree(VB.rerender(list(dec[k_, :int(lens[k_])])))
                    if t_ is None:
                        label[j] = None; continue
                    cur[j] = t_
                    if t_.op is None:
                        label[j] = t_.v
                    else:
                        nxt.add(j)
            active = nxt
        for j in active:
            label[j] = None
        js = list(range(len(roots))); hist = {}
        for j in js:
            k_ = "fail" if label[j] is None else str(steps[j] - n); hist[k_] = hist.get(k_, 0) + 1
        ok = [label[j] is not None and label[j] == gold[j] for j in js]; kept = [label[j] is not None and steps[j] == n for j in js]
        pk = [o for o, k in zip(ok, kept) if k]
        out[str(n)] = {"items": len(js), "precision_strict": float(np.mean([o and steps[j] == n for o, j in zip(ok, js)])), "fail": float(np.mean([label[j] is None for j in js])),
                       "right_value_wrong_steps": float(np.mean([o and steps[j] != n for o, j in zip(ok, js)])),
                       "steps_minus_n": dict(sorted(hist.items(), key=lambda kv: (kv[0] == "fail", int(kv[0]) if kv[0] != "fail" else 0))),
                       "keep": float(np.mean(kept)), "prec_kept": float(np.mean(pk)) if pk else None, "n_kept": len(pk), "collide_rate": (draws - len(roots)) / draws,
                       "draws": draws, "sec_per_item": (time.time() - t0) / len(js), "cap": cap}
        print("CHAINFRESH n", n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[str(n)].items()}, flush=True)
    return {"by_n": out, "excl_size": len(excl), "te_items": len(TE), "def": part_chainlabel_fresh.__doc__}   # Audit: EXCL derives from TE (per_n)


@torch.no_grad()
def part_pause(m, TE, dev, bs):
    """§214 eval_pause.json (registered 7892a2dc; the ONLY eval file that applies pi = pause_vec). §170 test items, vbank_eval.part_acc's chunks, fp32.
    pause_inv: n 1..8, Q = 6 pauses per item = D 3 start pauses (passes 1..3) + 3 scattered pauses uniformly over the first n - 1 + 3 mid positions
      (lastR: the last mid position is a reduction; np rng 21410, drawn per item in test order), completion pass n + 6; one rollout per chunk to
      K' n + 8, read at every K' 0..n + 8 -> acc_k {K': acc}.
    start_d: n 9..12 x D in {0, 2, 4, 6, 8} with 12 <= n + D <= 20: D start pauses only (passes 1..D), completion n + D; read at K' 0..n + D + 2.
    items: per item and K' the argmax answer (pause_inv: ids by n; start_d: by "n_D")."""
    assert hasattr(m, "pause_vec"), "eval_pause needs a --pause model (pause_vec)"
    num = torch.tensor(N.NUM_IDS, device=dev); Nn = np.array([it["n"] for it in TE]); Y = np.array([it["t"] for it in TE]); rng = np.random.default_rng(21410)
    res = {"Q": 6, "pause_inv": {}, "start_d": {}, "items": {"pause_inv": {}, "start_d": {}}}

    def run(idx, Kmax, sched):
        """sched(j) -> bool list over passes 1..L (L <= Kmax) for test item j; -> {K': (len(idx),) preds}."""
        preds = {k: np.full(len(idx), -1) for k in range(Kmax + 1)}
        for c0 in range(0, len(idx), bs):
            ch = idx[c0:c0 + bs]; its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev); B = len(its)
            pm = torch.zeros(B, Kmax + 1, dtype=torch.bool, device=dev)
            for b, j in enumerate(ch):
                z = sched(int(j)); pm[b, 1:len(z) + 1] = torch.tensor(z, device=dev)
            rows = torch.arange(B, device=dev); r_ = m.rollout(x, pad, Kmax, coda_rows={k: rows for k in range(Kmax + 1)}, pause=pm)
            for k in range(Kmax + 1):
                preds[k][c0:c0 + B] = m.answer_logits(r_["coda"][k][1], lens)[:, num].argmax(-1).cpu().numpy()
            del r_
        return preds
    sch = {}
    for n in range(1, 9):                                                   # draws per item in test order (rng 21410)
        for j in np.nonzero(Nn == n)[0]:
            S = 3; L1 = n - 1 + S; mid = [False] * (L1 + 1)
            for q in rng.choice(L1, S, replace=False):
                mid[int(q)] = True
            sch[int(j)] = [True] * 3 + mid
    for n in range(1, 9):
        idx = np.nonzero(Nn == n)[0]; Kmax = n + 8; pr = run(idx, Kmax, lambda j: sch[j])
        res["pause_inv"][str(n)] = {"acc_k": {str(k): float((pr[k] == Y[idx]).mean()) for k in range(Kmax + 1)}, "completion": n + 6}
        res["items"]["pause_inv"][str(n)] = {"ids": [int(j) for j in idx], "pred": {str(k): [int(v) for v in pr[k]] for k in range(Kmax + 1)}}
        print("PAUSE_INV n", n, {k: round(v, 3) for k, v in res["pause_inv"][str(n)]["acc_k"].items() if int(k) >= n}, flush=True)
    for n in range(9, 13):
        for D in (0, 2, 4, 6, 8):
            if not 12 <= n + D <= 20:
                continue
            idx = np.nonzero(Nn == n)[0]; Kmax = n + D + 2; pr = run(idx, Kmax, lambda j, D=D: [True] * D + [False] * TE[j]["n"])
            key = "%d_%d" % (n, D); res["start_d"][key] = {"n": n, "D": D, "acc_k": {str(k): float((pr[k] == Y[idx]).mean()) for k in range(Kmax + 1)}, "completion": n + D}
            res["items"]["start_d"][key] = {"ids": [int(j) for j in idx], "pred": {str(k): [int(v) for v in pr[k]] for k in range(Kmax + 1)}}
            print("START_D", key, {k: round(v, 3) for k, v in res["start_d"][key]["acc_k"].items() if int(k) >= n + D - 1}, flush=True)
    res["def"] = part_pause.__doc__
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--part", required=True, choices=["rs", "pf", "x16", "longstep", "foreign", "steppos", "gsplice", "a15", "d3diag", "wpereset", "d3sel", "lfk", "stabhist", "d3hist", "hold", "stepbyt", "hist24", "ageproj", "agerevert", "carrier", "chainlabel", "chainlabel_fresh", "pause"])
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--eval_bs", type=int, default=32); ap.add_argument("--per_n", type=int, default=1000)
    a = ap.parse_args(); dev = a.device; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    m, args = VE.load(a.run, dev); TE = N.test_set(a.per_n); t0 = time.time()
    res = part_x16(m, TE, dev, a.eval_bs, a.run) if a.part == "x16" else part_foreign(m, TE, dev, a.eval_bs, a.run) if a.part == "foreign" else part_carrier(m, TE, dev, a.eval_bs, a.run) if a.part == "carrier" else {"rs": part_rs, "pf": part_pf, "steppos": part_steppos, "gsplice": part_gsplice, "a15": part_a15, "d3diag": part_d3diag, "wpereset": part_wpereset, "d3sel": part_d3sel, "lfk": part_lfk, "stabhist": part_stabhist, "d3hist": part_d3hist, "hold": part_hold, "stepbyt": part_stepbyt, "hist24": part_hist24, "ageproj": part_ageproj, "agerevert": part_agerevert, "chainlabel": part_chainlabel, "chainlabel_fresh": part_chainlabel_fresh, "pause": part_pause, "longstep": lambda m_, T_, d_, b_: {"long_step_m_T": long_step(m_, T_, d_, b_, True),
                                                                                                   "def": "LONG-STEP keyed '<m>_<T>' (m operators, T tokens incl. EOT and ' =')"}}[a.part](m, TE, dev, a.eval_bs)
    res.update({"run": a.run, "mode": m.mode, "mct": bool(args.get("mct")), "eval_bs": a.eval_bs, "dtype": "fp32", "per_n": a.per_n,
                "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "sec": round(time.time() - t0)})
    fo = os.path.join(a.run, "eval_%s.json" % a.part); json.dump(res, open(fo + ".tmp", "w", newline="\n"), indent=1); os.replace(fo + ".tmp", fo)
    print("DONE", fo, res["sec"], flush=True)


if __name__ == "__main__":
    main()
