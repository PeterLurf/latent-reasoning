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
def part_rs(m, TE, dev, bs):
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
    for n in range(1, 17):
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
    out["long_step"] = long_step(m, TE, dev, bs)
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--part", required=True, choices=["rs", "pf", "x16", "longstep", "foreign", "steppos", "gsplice", "a15", "d3diag"])
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--eval_bs", type=int, default=32); ap.add_argument("--per_n", type=int, default=1000)
    a = ap.parse_args(); dev = a.device; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    m, args = VE.load(a.run, dev); TE = N.test_set(a.per_n); t0 = time.time()
    res = part_x16(m, TE, dev, a.eval_bs, a.run) if a.part == "x16" else part_foreign(m, TE, dev, a.eval_bs, a.run) if a.part == "foreign" else {"rs": part_rs, "pf": part_pf, "steppos": part_steppos, "gsplice": part_gsplice, "a15": part_a15, "d3diag": part_d3diag, "longstep": lambda m_, T_, d_, b_: {"long_step_m_T": long_step(m_, T_, d_, b_, True),
                                                                                                   "def": "LONG-STEP keyed '<m>_<T>' (m operators, T tokens incl. EOT and ' =')"}}[a.part](m, TE, dev, a.eval_bs)
    res.update({"run": a.run, "mode": m.mode, "mct": bool(args.get("mct")), "eval_bs": a.eval_bs, "dtype": "fp32", "per_n": a.per_n,
                "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "sec": round(time.time() - t0)})
    fo = os.path.join(a.run, "eval_%s.json" % a.part); json.dump(res, open(fo + ".tmp", "w", newline="\n"), indent=1); os.replace(fo + ".tmp", fo)
    print("DONE", fo, res["sec"], flush=True)


if __name__ == "__main__":
    main()
