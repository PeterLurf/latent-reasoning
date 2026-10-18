"""§217 CAUSAL-ACROSS-ROUNDS (registered 45d7d33c): eval_causal_deep.json = vbank_eval's VALUE-SWAP class A (make_pairs rng 21120, the same donor rule,
arms i / ii / iv, clean = recipient and donor correct at K' = n + 2) and part_stall (stored V_t := V_{t-1} at t = ceil(n / 2); stalled acc at K' n + 1 /
intact at K' n), COPIED from boxes/vbank_eval.py with ONLY an n-range parameter and per-item stores added (a separate module, so vbank_eval / vbank_diag
stay untouched while the §216 chains and deploys use them). The default range n 4..8 on test_set(1000) reproduces vbank_eval bitwise on the same card
and settings (smoke); here:
  gate: the same parts at n 4..8 on test_set(1000) (eval_vb's items and rng consumption) vs the run's committed eval_vb.json follow_ii / own_ii /
    stall recovery, |diff| <= 0.005 each (the card differs: the 4060 here, the 3070 there);
  vswap / stall: n 9..20 on test_set(1000, n_max=24) (never n >= 25: sealed, note 216-5): the by-n summaries in vbank_eval's format + by_n_iv and
    candidates_by_n; pairs [{i, n, t, clean, y, y_donor, ans_i, ans_ii, ans_iv}]; stall_items {n: {i, intact [0/1], stalled [0/1]}} (score_s217's contract).
Local RTX 4060, fp32, tf32 off, eval_bs 16 (the §217 card for every model).
    python -m boxes.vbank_causal --run boxes/runs_fsq/s215_ftself_s1 --device cuda:0 --eval_bs 16"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402
import boxes.vbank_eval as VE  # noqa: E402


def make_pairs(TE, seed=21120, cls="A", n_lo=4, n_hi=8):
    """vbank_eval.make_pairs with the n range a parameter (default 4..8 = vbank_eval's)."""
    rng = np.random.default_rng(seed); pairs = []; qual = {}
    for i, it in enumerate(TE):
        n = it["n"]
        if not n_lo <= n <= n_hi:
            continue
        hs = VE.handles(it["root"]); ok_t = [t for t in range(2, n) if hs[t - 1][1] == (cls == "A")]; qual.setdefault(str(n), [0, 0]); qual[str(n)][1] += 1
        if not ok_t:
            continue
        qual[str(n)][0] += 1; t = int(ok_t[int(rng.integers(0, len(ok_t)))]); path = hs[t - 1][0]
        for _ in range(200):
            d = N._copy(it["root"]); h = d
            for p_ in path:
                h = getattr(h, p_)
            hv0 = VE._val(h); lv = VE._leaves(h)
            for u in lv:
                u.v = int(rng.integers(0, N.P))
            hv, yd = VE._val(h), VE._val(d)
            if hv is not None and yd is not None and hv != hv0 and yd != it["t"]:
                pairs.append({"i": i, "n": n, "t": t, "donor": {"syms": N.render(d), "t": yd, "root": d, "n": n}}); break
    return pairs, {k: v[0] / v[1] for k, v in qual.items()}


@torch.no_grad()
def vswap_class(m, TE, dev, bs, cls="A", n_lo=4, n_hi=8, store=False):
    """vbank_eval.vswap_class with the n range a parameter; store = per-pair records {i, n, t, clean, y, y_donor, ans {arm}}."""
    pairs, qual = make_pairs(TE, cls=cls, n_lo=n_lo, n_hi=n_hi); res = {}; arms = ["i", "ii", "iv"] + (["iii_bank", "iii_carry"] if m.mode == "carry" else []); recs = []
    for key in sorted({(p["n"], p["t"]) for p in pairs}):
        grp = [p for p in pairs if (p["n"], p["t"]) == key]; n, t = key; K = n + 2
        for ch in VE.chunks(list(range(len(grp))), bs):
            ps = [grp[j] for j in ch]; rec = [TE[p["i"]] for p in ps]; don = [p["donor"] for p in ps]
            x, pad, lens = VB.batch_x(rec, dev); xd, pdd, ld = VB.batch_x(don, dev); B = len(ps); rows = {K: torch.arange(B, device=dev)}
            assert torch.equal(pad, pdd) and torch.equal(lens, ld), "VALUE-SWAP: donor slots != recipient slots"
            b = VE.vslot_tensor(rec, t, dev)[:, t]; assert bool((b > 0).all()) and torch.equal(b, VE.vslot_tensor(don, t, dev)[:, t]), "VALUE-SWAP: value slot"
            rr = m.rollout(x, pad, K, coda_rows=rows); rd = m.rollout(xd, pdd, K, coda_rows=rows); Vd = rd["V"][t]
            y = torch.tensor([it["t"] for it in rec], device=dev); yd = torch.tensor([it["t"] for it in don], device=dev)
            clean = (VE.ans(m, rr["coda"][K][1], lens) == y) & (VE.ans(m, rd["coda"][K][1], ld) == yd); arm_ans = {}
            del rr, rd; rr = rd = None                                      # memory only (8 GB card): Vd and clean are all the arms need; values unchanged
            for arm in arms:
                def edit(tt, V, arm=arm):
                    if tt != t:
                        return None
                    if arm == "i":
                        V2 = V.clone(); V2[torch.arange(B, device=dev), b] = Vd[torch.arange(B, device=dev), b]; return V2
                    if arm in ("ii", "iv"):
                        return Vd.clone()
                    return (Vd.clone(), V) if arm == "iii_bank" else (V, Vd.clone())
                ra = m.rollout(x, pad, K, coda_rows=rows, edit=edit, v0_off_after=t if arm == "iv" else None); A_ = VE.ans(m, ra["coda"][K][1], lens)
                del ra; d_ = res.setdefault(arm, {}).setdefault(str(n), {"follow": 0, "own": 0, "pairs": 0}); arm_ans[arm] = A_.cpu().numpy()
                d_["follow"] += int(((A_ == yd) & clean).sum()); d_["own"] += int(((A_ == y) & clean).sum()); d_["pairs"] += int(clean.sum())
            if store:
                cl = clean.cpu().numpy(); yy = y.cpu().numpy(); yy_d = yd.cpu().numpy()
                for q, p in enumerate(ps):
                    recs.append({"i": p["i"], "n": n, "t": t, "clean": bool(cl[q]), "y": int(yy[q]), "y_donor": int(yy_d[q]), "ans": {a: int(arm_ans[a][q]) for a in arms}})
            dd = res.setdefault("_all", {}).setdefault(str(n), [0, 0]); dd[0] += int(clean.sum()); dd[1] += B; del rr, rd, Vd
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
    tot = lambda arm, f: (sum(v[f] for v in res[arm].values()) / max(1, sum(v["pairs"] for v in res[arm].values()))) if arm in res else None  # noqa: E731
    npairs = sum(v["pairs"] for v in res.get("ii", {}).values())
    out = {"n_pairs": npairs, "follow_i": tot("i", "follow"), "own_i": tot("i", "own"), "follow_ii": tot("ii", "follow"), "own_ii": tot("ii", "own"),
           "follow_iv": tot("iv", "follow"), "own_iv": tot("iv", "own"),
           "by_n": {n: {"n_pairs": res["ii"][n]["pairs"], "follow_ii": res["ii"][n]["follow"] / max(1, res["ii"][n]["pairs"]), "own_ii": res["ii"][n]["own"] / max(1, res["ii"][n]["pairs"]),
                        "follow_i": res["i"][n]["follow"] / max(1, res["i"][n]["pairs"]), "own_i": res["i"][n]["own"] / max(1, res["i"][n]["pairs"]),
                        "follow_iv": res["iv"][n]["follow"] / max(1, res["iv"][n]["pairs"]), "own_iv": res["iv"][n]["own"] / max(1, res["iv"][n]["pairs"])}
                    for n in sorted(res.get("ii", {}), key=int)},
           "by_n_iv": {n: {"follow_iv": res["iv"][n]["follow"] / max(1, res["iv"][n]["pairs"]), "own_iv": res["iv"][n]["own"] / max(1, res["iv"][n]["pairs"])} for n in sorted(res.get("iv", {}), key=int)},
           "candidates_by_n": {str(n): sum(1 for p in pairs if p["n"] == n) for n in sorted({p["n"] for p in pairs})},
           "candidates": len(pairs), "clean_share_by_n": {n: v[0] / max(1, v[1]) for n, v in res.get("_all", {}).items()}, "qualifying_share_by_n": qual, "class": cls,
           "n_lo": n_lo, "n_hi": n_hi, "t_hist": {str(t): sum(1 for p in pairs if p["t"] == t) for t in sorted({p["t"] for p in pairs})}}
    if store:
        out["pairs"] = recs
    return out


@torch.no_grad()
def stall(m, TE, dev, bs, n_lo=4, n_hi=8, store=False):
    """vbank_eval.part_stall with the n range a parameter; store = per-item {i, n, intact (K' n), stalled (K' n + 1)} correctness."""
    Nn = np.array([it["n"] for it in TE]); by = {}; recs = []
    for n in range(n_lo, n_hi + 1):
        ts = math.ceil(n / 2); ok_i = ok_s = cnt = 0
        for ch in VE.chunks(np.nonzero(Nn == n)[0], bs):
            its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev); y = torch.tensor([it["t"] for it in its], device=dev); B = len(its); st = {}

            def edit(t, V):
                if t == ts:
                    return st[t - 1].clone()
                st[t] = V
                return None
            ri = m.rollout(x, pad, n, coda_rows={n: torch.arange(B, device=dev)}); ci = VE.ans(m, ri["coda"][n][1], lens) == y; ok_i += int(ci.sum())
            r0 = m.rollout(x, pad, 0); st[0] = r0["V"][0]
            rs = m.rollout(x, pad, n + 1, coda_rows={n + 1: torch.arange(B, device=dev)}, edit=edit); cs = VE.ans(m, rs["coda"][n + 1][1], lens) == y; ok_s += int(cs.sum()); cnt += B
            if store:
                for q, j in enumerate(ch):
                    recs.append({"i": int(j), "n": n, "intact": bool(ci[q]), "stalled": bool(cs[q])})
            del ri, rs, r0; st.clear()
        by[str(n)] = {"intact_kn": ok_i / cnt, "stalled_kn1": ok_s / cnt, "stall_t": ts, "items": cnt}
    pi = np.mean([v["intact_kn"] for v in by.values()]); ps = np.mean([v["stalled_kn1"] for v in by.values()])
    out = {"recovery": float(ps / pi) if pi > 0 else None, "by_n": by, "n_lo": n_lo, "n_hi": n_hi}
    if store:
        out["items"] = recs
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--n_lo", type=int, default=9); ap.add_argument("--n_hi", type=int, default=20); ap.add_argument("--out_dir", default=""); a = ap.parse_args(); dev = a.device
    assert a.n_hi <= 24, "n >= 25 is sealed (note 216-5)"
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; t0 = time.time()
    VE.NUM = torch.tensor(N.NUM_IDS, device=dev); VE.DEV = dev                # as vbank_eval.main sets them (VE.ans reads the module-level NUM)
    m, args = VE.load(a.run, dev); TE0 = N.test_set(1000); TE = N.test_set(1000, n_max=24)
    g_vs = vswap_class(m, TE0, dev, a.eval_bs, "A"); g_st = stall(m, TE0, dev, a.eval_bs)           # gate: n 4..8, eval_vb's items
    ref = json.load(open(os.path.join(a.run, "eval_vb.json"))); rv, rs_ = ref["vswap"], ref["stall"]
    gate = {"follow_ii": [g_vs["follow_ii"], rv["follow_ii"]], "own_ii": [g_vs["own_ii"], rv["own_ii"]], "stall_recovery": [g_st["recovery"], rs_["recovery"]]}
    gate = {k: {"here": v[0], "eval_vb": v[1], "abs_diff": abs(v[0] - v[1])} for k, v in gate.items()}; gate_ok = all(v["abs_diff"] <= 0.005 for v in gate.values())
    print("GATE n4..8", {k: round(v["abs_diff"], 4) for k, v in gate.items()}, "pass", gate_ok, flush=True)
    res = {"gate": {"vswap": g_vs, "stall": g_st, "terms": gate, "pass": gate_ok, "tolerance": 0.005, "ref_card": ref.get("card"), "ref_eval_bs": ref.get("eval_bs")}}
    vs = vswap_class(m, TE, dev, a.eval_bs, "A", a.n_lo, a.n_hi, store=True); st = stall(m, TE, dev, a.eval_bs, a.n_lo, a.n_hi, store=True)
    res["pairs"] = [{"i": r["i"], "n": r["n"], "t": r["t"], "clean": r["clean"], "y": r["y"], "y_donor": r["y_donor"], **{"ans_" + k: v for k, v in r["ans"].items()}} for r in vs.pop("pairs")]
    res["stall_items"] = {}
    for r in st.pop("items"):
        d = res["stall_items"].setdefault(str(r["n"]), {"i": [], "intact": [], "stalled": []}); d["i"].append(r["i"]); d["intact"].append(int(r["intact"])); d["stalled"].append(int(r["stalled"]))
    res["vswap"] = vs; res["stall"] = st
    res["meta"] = {"run": a.run, "mode": m.mode, "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "dtype": "fp32", "tf32": False, "eval_bs": a.eval_bs,
                   "n_lo": a.n_lo, "n_hi": a.n_hi, "test_n_max": 24, "test_seed": 17100, "commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip(),
                   "sec": round(time.time() - t0)}
    res["card"] = res["meta"]["card"]; res["def"] = __doc__
    fo = os.path.join(a.out_dir or a.run, "eval_causal_deep.json"); json.dump(res, open(fo + ".tmp", "w", newline="\n"), indent=1); os.replace(fo + ".tmp", fo)
    print("DONE", fo, res["meta"]["sec"], "gate", gate_ok, flush=True)


if __name__ == "__main__":
    main()
