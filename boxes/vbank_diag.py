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
def long_step(m, TE, dev, bs):
    """LONG-STEP(m): chains re-rooted after every pass; one pass on a fresh input, R(V_1) decoded and compared with its next step; stop at the first
    inexact step. -> {m: {attempted, exact, rate}} over every test item's chain."""
    cur = {j: N._copy(it["root"]) for j, it in enumerate(TE)}; stats = {}
    while cur:
        idx = sorted(cur, key=lambda j: len(N.render(cur[j])))
        nxt = {}
        for c0 in range(0, len(idx), bs):
            part = idx[c0:c0 + bs]; its = [{"syms": N.render(cur[j])} for j in part]; x, pad, lens = VB.batch_x(its, dev)
            rr = m.rollout(x, pad, 1); dec = m.rec_logits(rr["V"][1]).argmax(-1).cpu().numpy(); del rr
            for k_, j in enumerate(part):
                mo = n_ops(cur[j]); red = reduce_k(cur[j], 1); ok = VB.rerender(list(dec[k_, :int(lens[k_])])) == N.render(red)
                s_ = stats.setdefault(str(mo), [0, 0]); s_[0] += 1; s_[1] += int(ok)
                if ok and red.op is not None:
                    nxt[j] = red
        cur = nxt
    return {k: {"n_steps": v[0], "exact": v[1] / v[0], "n_exact": v[1]} for k, v in sorted(stats.items(), key=lambda kv: int(kv[0]))}   # contract: exact = rate


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--part", required=True, choices=["rs"])
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--eval_bs", type=int, default=32); ap.add_argument("--per_n", type=int, default=1000)
    a = ap.parse_args(); dev = a.device; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    m, args = VE.load(a.run, dev); TE = N.test_set(a.per_n); t0 = time.time()
    res = {"rs": part_rs}[a.part](m, TE, dev, a.eval_bs)
    res.update({"run": a.run, "mode": m.mode, "mct": bool(args.get("mct")), "eval_bs": a.eval_bs, "dtype": "fp32", "per_n": a.per_n,
                "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "sec": round(time.time() - t0)})
    fo = os.path.join(a.run, "eval_%s.json" % a.part); json.dump(res, open(fo + ".tmp", "w", newline="\n"), indent=1); os.replace(fo + ".tmp", fo)
    print("DONE", fo, res["sec"], flush=True)


if __name__ == "__main__":
    main()
