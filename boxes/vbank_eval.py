"""THEORY §211 eval_vb.json (note 211-1 contract; read by theory/score_s211.py) for a landed VERSION-BANK cell: local fp32 (no autocast), --eval_bs (32, note 211-7)
items per batch (grouped by n; padded == per item to 5.7e-15 in fp64, scratch/gate_s211/maskk.json), the §170 test set (1,000 items per n, n 1..16), card
recorded. Parts (each merged into <run>/eval_vb.json as it finishes, so a late crash keeps the earlier parts):
  acc      acc_n2 {n: acc at K' = n + 2}; acc_k {K': {n: acc}} K' 0..16, 24; test_meta {n, height}; carry_likeness (PA: mean gather self-mass on (t-1, j),
           passes 2..n, n 3..8); stepcap {copy_slot_acc, value_stepcap_n6_8, stationary_slope, by_t_a}; r_acc_by_t; bank_ablate (descriptive)
  probe    probe {t: {acc_by_n, items, train_ce_final}}, t in 0 1 2 3 4 6 8 12: a fresh copy of blocks 10-11 + the coda's bank beta / gamma rows (ln_f and
           the tied LM head frozen) reads V_t + bank V_1..V_{t-1} (t = 0: V_0 alone), trained 2k steps, bs 64, AdamW 1e-4, answer CE, on a fixed pool
           of 20k n 3..8 train items (test + SERIAL pool excluded); the model frozen
  vswap    VALUE-SWAP (draft D3 / 211-1): recipients test n 4..8; t uniform over the t in 2..n-1 whose handle has two ORIGINAL-leaf operands; the donor
           resamples those two leaves (value changes, y_donor != y); clean = recipient and donor each correct at K' = n + 2; arms (i) value slot b,
           (ii) the whole V_t, (iii) N1 only: bank-swap / carry-swap, (iv) = (ii) with V_0 masked for passes > t
  stall    at t = ceil(n / 2) the stored entry is a copy of V_{t-1}; recovery = pooled acc(n 4..8, K' = n + 1, stalled) / pooled acc(n 4..8, K' = n)
  tf       (N3) acc_tf_n3_8: pooled n 3..8 at K' = n + 2 with MCT applied to every bank entry t >= 1
    python -m boxes.vbank_eval --run boxes/runs_fsq/s211_vb_pa_s1 --device cuda:0 [--parts acc,probe,vswap,stall,tf]   (eval_bs 32 for every reading)"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402

T_SET = (0, 1, 2, 3, 4, 6, 8, 12); KS = list(range(17)) + [24]; NUM = None; DEV = "cpu"


def height(nd):
    return 0 if nd.op is None else 1 + max(height(nd.l), height(nd.r))


def load(run, dev):
    from boxes import s5
    a = json.load(open(os.path.join(run, "args.json"))); m = VB.VBankN(s5.load_gpt2(dev), a["mode"]).to(dev)
    m.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location=dev, weights_only=True)); m.eval()
    for p_ in m.parameters():
        p_.requires_grad_(False)
    return m, a


def ablate_keep(t, nver):
    """BANK-ABLATE's kept versions at pass (or coda) t over stored versions 0..nver-1: V_0 and the newest (t - 1); versions 1..t-2 masked."""
    s = torch.arange(nver, device=DEV)
    return (s == 0) | (s >= t - 1)


def chunks(idx, bs):
    return [idx[i:i + bs] for i in range(0, len(idx), bs)]


def ans(m, h, lens):
    return m.answer_logits(h, lens)[:, NUM].argmax(-1)


def excl_set(TE):
    ex = {tuple(it["syms"]) for it in TE}; r_sp = np.random.default_rng(17300)
    for _ in range(2200):
        ex.add(tuple(N.render(N.generate(r_sp, 8)[0])))
    return ex


def ols(xs, ys):
    mx, my = np.mean(xs), np.mean(ys); den = sum((x - mx) ** 2 for x in xs)
    return float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) if den > 0 else None


def value_slots(root):
    """the slot (in [EOT] + render + [' =']) that takes reduction t's value, t = 1..n: the handle span's last slot, as vbank.recode_targets places it
    (independent of whether its symbol changes - e.g. 0 + 3 = 3 leaves that slot's class unchanged)."""
    nd = N._copy(root); slots = [1] * (len(N.render(nd)) + 2); out = []
    while nd.op is not None:
        live = [j for j in range(1, len(slots) - 1) if slots[j]]; own = VB._render_owned(nd); path = N._leftmost_ready(nd); h = nd
        for p_ in path:
            h = getattr(h, p_)
        span = [live[k] for k, (_, o) in enumerate(own) if o is h or o is h.l or o is h.r]
        h.v = N.apply(h.op, h.l.v, h.r.v); h.op = h.l = h.r = None
        for j in span:
            slots[j] = 0
        slots[span[-1]] = 1; out.append(span[-1])
    return out


def vslot_tensor(items, t_max, dev):
    """(B, t_max + 1) value slot of reduction t (column t; -1 where t > n or t = 0)."""
    V_ = np.full((len(items), t_max + 1), -1, dtype=np.int64)
    for r, it in enumerate(items):
        vs = value_slots(it["root"])
        for t in range(1, min(t_max, len(vs)) + 1):
            V_[r, t] = vs[t - 1]
    return torch.tensor(V_, device=dev)


# ---------------------------------------------------------------- acc / acc_k / test_meta / carry_likeness / stepcap / r_acc_by_t / bank_ablate
@torch.no_grad()
def part_acc(m, TE, dev, bs):
    Nn = np.array([it["n"] for it in TE]); Y = np.array([it["t"] for it in TE]); A = {k: np.full(len(TE), -1) for k in KS + ["n2"]}
    cl_sum, cl_cnt = 0.0, 0; sc = {}; copy_ok = copy_n = 0; val_ok = val_n = 0; racc = {}; abl = np.full(len(TE), -1); boff = np.full(len(TE), -1)
    for n in range(1, 17):
        for ch in chunks(np.nonzero(Nn == n)[0], bs):
            its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev); B, T = x.shape; E = VB.rec_targets(its, 24, T, dev)
            m.gather_log = [] if m.mode == "pa" else None
            r = m.rollout(x, pad, 24, coda_rows={t: torch.arange(B, device=dev) for t in set(KS) | {n + 2}})
            for k in KS:
                A[k][ch] = ans(m, r["coda"][k][1], lens).cpu().numpy()
            A["n2"][ch] = ans(m, r["coda"][n + 2][1], lens).cpu().numpy()
            if m.gather_log is not None and 3 <= n <= 8:
                for t, v in m.gather_log:
                    if 2 <= t <= n:
                        cl_sum += float(v.sum()); cl_cnt += len(v)
            m.gather_log = None
            dec = [m.rec_logits(r["V"][t]).argmax(-1) for t in range(0, n + 3)]; VS = vslot_tensor(its, n, dev)
            for t in range(0, n + 3):                                       # R slot accuracy by t (real slots)
                ok_ = (dec[t] == E[:, t])[pad]; d_ = racc.setdefault(str(t), {}).setdefault(str(n), [0, 0]); d_[0] += int(ok_.sum()); d_[1] += int(ok_.numel())
            for t in range(1, n + 1):
                Et, Ep = E[:, t], E[:, t - 1]; exact_prev = ((dec[t - 1] == Ep) | ~pad).all(1)
                unch = (Et == Ep) & pad
                if 3 <= n <= 8:
                    copy_ok += int(((dec[t] == Et) & unch).sum()); copy_n += int(unch.sum())
                b = VS[:, t]; assert bool((b > 0).all()); hit = (dec[t].gather(1, b[:, None])[:, 0] == Et.gather(1, b[:, None])[:, 0])
                alive = ((Ep != VB.VOID) & pad).sum(1) - 2                  # non-VOID content slots of E_{t-1} (EOT and ' =' excluded)
                for r_ in torch.nonzero(exact_prev).flatten().tolist():
                    k_ = "%d,%d" % (t, int(alive[r_])); d_ = sc.setdefault(k_, [0, 0]); d_[0] += int(hit[r_]); d_[1] += 1
                    if 6 <= n <= 8:
                        val_ok += int(hit[r_]); val_n += 1
            del r, dec, E                                                   # free the K' 24 rollout before the bank arms (06:2x 10-16: holding it OOMed n 16 on 8 GB)
            off0 = m.bank_off; m.bank_off = True                            # BANK-OFF (descriptive): every bank logit -inf, K' = n + 2
            rb = m.rollout(x, pad, n + 2, coda_rows={n + 2: torch.arange(B, device=dev)}); m.bank_off = off0
            boff[ch] = ans(m, rb["coda"][n + 2][1], lens).cpu().numpy(); del rb
            m.ver_keep = ablate_keep                                        # BANK-ABLATE as registered (draft §7; note 211-14): versions 1..t-2 masked
            rb = m.rollout(x, pad, n + 2, coda_rows={n + 2: torch.arange(B, device=dev)}); m.ver_keep = None
            abl[ch] = ans(m, rb["coda"][n + 2][1], lens).cpu().numpy(); del rb   # free this chunk's bank before the next (8 GB cards)
    by_n = lambda a_: {str(n): float((a_ == Y)[Nn == n].mean()) for n in range(1, 17)}  # noqa: E731
    by_ta = {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(sc.items(), key=lambda kv: tuple(int(z) for z in kv[0].split(",")))}
    slopes = []
    for a_ in sorted({int(k.split(",")[1]) for k in sc}):
        pts = [(int(k.split(",")[0]), v["acc"]) for k, v in by_ta.items() if int(k.split(",")[1]) == a_ and v["n"] >= 10]
        if len({t for t, _ in pts}) >= 3:
            slopes.append(ols([t for t, _ in pts], [y for _, y in pts]))
    return {"acc_n2": by_n(A["n2"]), "acc_k": {str(k): by_n(A[k]) for k in KS},
            "test_meta": {"n": [int(it["n"]) for it in TE], "height": [height(it["root"]) for it in TE]},
            "carry_likeness": (cl_sum / cl_cnt) if cl_cnt else None,
            "stepcap": {"copy_slot_acc": copy_ok / max(1, copy_n), "value_stepcap_n6_8": (val_ok / val_n) if val_n else None,
                        "stationary_slope": float(np.mean([s_ for s_ in slopes if s_ is not None])) if slopes else None, "by_t_a": by_ta,
                        "n_copy_slots": copy_n, "n_value_cond_n6_8": val_n,
                        "def": "STEP-CAP(t, a): R acc at the value slot of V_t on items whose R(V_{t-1}) decodes E_{t-1} exactly; a = non-VOID content slots of E_{t-1}; "
                               "t <= n; slope over a with >= 3 t values (cells with >= 10 items)"},
            "r_acc_by_t": {t: {n: v[0] / v[1] for n, v in d.items()} for t, d in racc.items()}, "bank_ablate": {"acc_n2": by_n(abl), "def": "draft §7 / note 211-14: versions 1..t-2 masked at every pass (core, gather) and in the coda; V_0 and the newest kept; K' = n + 2"},
            "hist_trunc": {"acc_n2": by_n(abl), "def": "alias of bank_ablate (note 211-14)"},
            "bank_off": {"acc_n2": by_n(boff), "def": "every bank logit -inf (core and coda; the gather still reads all versions), K' = n + 2"}}


# ---------------------------------------------------------------- PROBE(t)
class Src:
    """a weight source for vbank.block_bank / bank_kv: the probe's own copies of blocks 10-11, the frozen trunk otherwise (ln_f, wte, ...)."""
    def __init__(self, g, P):
        self.g, self.P, self.H = g, P, g.H

    def w(self, k):
        kk = k.replace(".", "_")
        return self.P[kk] if kk in self.P else self.g.w(k)


def probe_t(m, t, pool, TE, dev, bs, steps=2000, pbs=64, lr=1e-4, seed=21131):
    g = m.g; P = torch.nn.ParameterDict({k: torch.nn.Parameter(v.detach().clone()) for k, v in g.p.items() if k.startswith(("h_10_", "h_11_"))})
    beta = torch.nn.Parameter(m.beta[4:].detach().clone()); gam = torch.nn.Parameter(m.gam[4:].detach().clone()); beta0 = m.beta0[4:].detach()
    src = Src(g, P); opt = torch.optim.AdamW(list(P.parameters()) + [beta, gam], lr=lr, weight_decay=0.01); rng = np.random.default_rng(seed); ces = []

    def fwd(its):
        x, pad, lens = VB.batch_x(its, dev)
        with torch.no_grad():
            r = m.rollout(x, pad, t); S_ = r["S"]; Vt = r["V"][t]; del r                      # keep only the versions the coda reads
        kvs = {i: [(None, None)] + [VB.bank_kv(src, i, S_[s]) for s in range(1, t)] for i in VB.CODA}
        return m.answer_logits(m.coda(t, Vt, kvs, pad, src=src, bp=(beta, beta0, gam)), lens)
    for st in range(steps):
        its = [pool[j] for j in rng.integers(0, len(pool), pbs)]; y = torch.tensor(N.NUM_IDS[[it["t"] for it in its]], device=dev)
        loss = F.cross_entropy(fwd(its), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); ces.append(float(loss))
    Nn = np.array([it["n"] for it in TE]); Y = np.array([it["t"] for it in TE]); A = np.full(len(TE), -1)
    with torch.no_grad():
        for n in range(1, 17):
            for ch in chunks(np.nonzero(Nn == n)[0], bs):
                A[ch] = fwd([TE[j] for j in ch])[:, NUM].argmax(-1).cpu().numpy()
    C = A == Y
    return {"acc_by_n": {str(n): float(C[Nn == n].mean()) for n in range(1, 17)}, "items": [int(c) for c in C], "train_ce_final": float(np.mean(ces[-100:])),
            "train_ce_first": float(np.mean(ces[:100]))}


def part_probe(m, TE, dev, bs, ts=T_SET, steps=2000):
    ex = excl_set(TE); rng = np.random.default_rng(21130); pool = []
    while len(pool) < 20000:
        r_, y_ = N.generate(rng, int(rng.integers(3, 9))); s_ = N.render(r_)
        if tuple(s_) not in ex:
            pool.append({"syms": s_, "t": y_, "root": r_})
    out = {}
    for t in ts:
        t0 = time.time(); out[str(t)] = probe_t(m, t, pool, TE, dev, bs, steps); out[str(t)]["sec"] = round(time.time() - t0)
        print("PROBE", t, out[str(t)]["train_ce_final"], {k: round(v, 3) for k, v in out[str(t)]["acc_by_n"].items()}, flush=True)
    return {"probe": out, "probe_protocol": "fresh blocks 10-11 + coda bank beta / gamma copy (ln_f, wte frozen); 2k steps, bs 64, AdamW 1e-4 wd 0.01; "
                                            "pool 20k n 3..8 (rng 21130, test + SERIAL pool excluded); batch draws rng 21131; model frozen; fp32"}


# ---------------------------------------------------------------- VALUE-SWAP
def handles(root):
    """per reduction t = 1..n: (path to the handle, both operands original leaves?)."""
    nd = N._copy(root); orig = set(); st = [nd]
    while st:
        u = st.pop()
        if u.op is None:
            orig.add(id(u))
        else:
            st += [u.l, u.r]
    out = []
    while nd.op is not None:
        path = N._leftmost_ready(nd); h = nd
        for p_ in path:
            h = getattr(h, p_)
        out.append((path, id(h.l) in orig and id(h.r) in orig)); h.v = N.apply(h.op, h.l.v, h.r.v); h.op = h.l = h.r = None
    return out


def _leaves(nd):
    return [nd] if nd.op is None else _leaves(nd.l) + _leaves(nd.r)


def _val(nd):
    try:
        return N.value(nd)
    except KeyError:                                                        # a resampled '/' divisor evaluating to 0
        return None


def make_pairs(TE, seed=21120, cls="A"):
    """note 211-4. Class A (decisive): t uniform over the t in 2..n-1 whose handle has two ORIGINAL-leaf operands; the donor resamples those two leaves.
    Class B (descriptive): t uniform over the t whose handle has >= 1 operand computed earlier; the donor resamples every leaf under the handle's subtree.
    Either way the handle value changes and y_donor != y (<= 200 tries, else the item is dropped)."""
    rng = np.random.default_rng(seed); pairs = []; qual = {}
    for i, it in enumerate(TE):
        n = it["n"]
        if not 4 <= n <= 8:
            continue
        hs = handles(it["root"]); ok_t = [t for t in range(2, n) if hs[t - 1][1] == (cls == "A")]; qual.setdefault(str(n), [0, 0]); qual[str(n)][1] += 1
        if not ok_t:
            continue
        qual[str(n)][0] += 1; t = int(ok_t[int(rng.integers(0, len(ok_t)))]); path = hs[t - 1][0]
        for _ in range(200):
            d = N._copy(it["root"]); h = d
            for p_ in path:
                h = getattr(h, p_)
            hv0 = _val(h); lv = _leaves(h)
            for u in lv:
                u.v = int(rng.integers(0, N.P))
            hv, yd = _val(h), _val(d)
            if hv is not None and yd is not None and hv != hv0 and yd != it["t"]:
                pairs.append({"i": i, "n": n, "t": t, "donor": {"syms": N.render(d), "t": yd, "root": d, "n": n}}); break
    return pairs, {k: v[0] / v[1] for k, v in qual.items()}


@torch.no_grad()
def part_vswap(m, TE, dev, bs):
    out = vswap_class(m, TE, dev, bs, "A"); out["class_B"] = vswap_class(m, TE, dev, bs, "B")
    return {"vswap": out}


def vswap_class(m, TE, dev, bs, cls):
    pairs, qual = make_pairs(TE, cls=cls); res = {}; arms = ["i", "ii", "iv"] + (["iii_bank", "iii_carry"] if m.mode == "carry" else [])
    for key in sorted({(p["n"], p["t"]) for p in pairs}):
        grp = [p for p in pairs if (p["n"], p["t"]) == key]; n, t = key; K = n + 2
        for ch in chunks(list(range(len(grp))), bs):
            ps = [grp[j] for j in ch]; rec = [TE[p["i"]] for p in ps]; don = [p["donor"] for p in ps]
            x, pad, lens = VB.batch_x(rec, dev); xd, pdd, ld = VB.batch_x(don, dev); B = len(ps); rows = {K: torch.arange(B, device=dev)}
            assert torch.equal(pad, pdd) and torch.equal(lens, ld), "VALUE-SWAP: donor slots != recipient slots"
            b = vslot_tensor(rec, t, dev)[:, t]; assert bool((b > 0).all()) and torch.equal(b, vslot_tensor(don, t, dev)[:, t]), "VALUE-SWAP: value slot"
            rr = m.rollout(x, pad, K, coda_rows=rows); rd = m.rollout(xd, pdd, K, coda_rows=rows); Vd = rd["V"][t]
            y = torch.tensor([it["t"] for it in rec], device=dev); yd = torch.tensor([it["t"] for it in don], device=dev)
            clean = (ans(m, rr["coda"][K][1], lens) == y) & (ans(m, rd["coda"][K][1], ld) == yd)
            for arm in arms:
                def edit(tt, V, arm=arm):
                    if tt != t:
                        return None
                    if arm == "i":
                        V2 = V.clone(); V2[torch.arange(B, device=dev), b] = Vd[torch.arange(B, device=dev), b]; return V2
                    if arm in ("ii", "iv"):
                        return Vd.clone()
                    return (Vd.clone(), V) if arm == "iii_bank" else (V, Vd.clone())
                ra = m.rollout(x, pad, K, coda_rows=rows, edit=edit, v0_off_after=t if arm == "iv" else None); A_ = ans(m, ra["coda"][K][1], lens)
                del ra; d_ = res.setdefault(arm, {}).setdefault(str(n), {"follow": 0, "own": 0, "pairs": 0})
                d_["follow"] += int(((A_ == yd) & clean).sum()); d_["own"] += int(((A_ == y) & clean).sum()); d_["pairs"] += int(clean.sum())
            dd = res.setdefault("_all", {}).setdefault(str(n), [0, 0]); dd[0] += int(clean.sum()); dd[1] += B; del rr, rd, Vd
    tot = lambda arm, f: (sum(v[f] for v in res[arm].values()) / max(1, sum(v["pairs"] for v in res[arm].values()))) if arm in res else None  # noqa: E731
    npairs = sum(v["pairs"] for v in res.get("ii", {}).values())
    out = {"n_pairs": npairs, "follow_i": tot("i", "follow"), "own_i": tot("i", "own"), "follow_ii": tot("ii", "follow"), "own_ii": tot("ii", "own"),
           "follow_iv": tot("iv", "follow"), "own_iv": tot("iv", "own"),
           "by_n": {n: {"n_pairs": res["ii"][n]["pairs"], "follow_ii": res["ii"][n]["follow"] / max(1, res["ii"][n]["pairs"]), "own_ii": res["ii"][n]["own"] / max(1, res["ii"][n]["pairs"])}
                    for n in sorted(res.get("ii", {}))},
           "candidates": len(pairs), "clean_share_by_n": {n: v[0] / max(1, v[1]) for n, v in res.get("_all", {}).items()}, "qualifying_share_by_n": qual, "class": cls,
           "t_hist": {str(t): sum(1 for p in pairs if p["t"] == t) for t in sorted({p["t"] for p in pairs})},
           "def": "note 211-4 class A (top level, decisive): t uniform over 2..n-1 with two original-leaf operands (rng 21120), donor = those leaves resampled; class B "
                  "(vswap.class_B, descriptive): handles with >= 1 computed operand, donor = every leaf under the handle resampled; value and y change; "
                  "clean = recipient and donor correct at K' = n + 2; follow = y_donor, own = y; (iv) V_0 masked for passes > t (the coda never reads V_0)"}
    if m.mode == "carry":
        out.update(follow_iii_bank=tot("iii_bank", "follow"), own_iii_bank=tot("iii_bank", "own"), follow_iii_carry=tot("iii_carry", "follow"), own_iii_carry=tot("iii_carry", "own"))
    return out


# ---------------------------------------------------------------- STALL / TF
@torch.no_grad()
def part_stall(m, TE, dev, bs):
    Nn = np.array([it["n"] for it in TE]); by = {}
    for n in range(4, 9):
        ts = math.ceil(n / 2); ok_i = ok_s = cnt = 0
        for ch in chunks(np.nonzero(Nn == n)[0], bs):
            its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev); y = torch.tensor([it["t"] for it in its], device=dev); B = len(its); st = {}

            def edit(t, V):
                if t == ts:
                    return st[t - 1].clone()
                st[t] = V
                return None
            ri = m.rollout(x, pad, n, coda_rows={n: torch.arange(B, device=dev)}); ok_i += int((ans(m, ri["coda"][n][1], lens) == y).sum())
            r0 = m.rollout(x, pad, 0); st[0] = r0["V"][0]
            rs = m.rollout(x, pad, n + 1, coda_rows={n + 1: torch.arange(B, device=dev)}, edit=edit); ok_s += int((ans(m, rs["coda"][n + 1][1], lens) == y).sum()); cnt += B
            del ri, rs, r0; st.clear()
        by[str(n)] = {"intact_kn": ok_i / cnt, "stalled_kn1": ok_s / cnt, "stall_t": ts, "items": cnt}
    pi = np.mean([v["intact_kn"] for v in by.values()]); ps = np.mean([v["stalled_kn1"] for v in by.values()])
    return {"stall": {"recovery": float(ps / pi) if pi > 0 else None, "by_n": by, "def": "stored V_t := V_{t-1} at t = ceil(n/2); stalled acc at K' = n + 1 / intact acc at K' = n, pooled n 4..8"}}


@torch.no_grad()
def part_tf(m, TE, dev, bs):
    Nn = np.array([it["n"] for it in TE]); ok = cnt = 0; by = {}
    for n in range(3, 9):
        o_ = c_ = 0
        for ch in chunks(np.nonzero(Nn == n)[0], bs):
            its = [TE[j] for j in ch]; x, pad, lens = VB.batch_x(its, dev); E = VB.rec_targets(its, n + 2, x.shape[1], dev); y = torch.tensor([it["t"] for it in its], device=dev)
            r = m.rollout(x, pad, n + 2, coda_rows={n + 2: torch.arange(len(its), device=dev)}, edit=lambda t, V: m.mct(V, E[:, t], pad)[0])
            o_ += int((ans(m, r["coda"][n + 2][1], lens) == y).sum()); c_ += len(its); del r
        by[str(n)] = o_ / c_; ok += o_; cnt += c_
    return {"acc_tf_n3_8": ok / cnt, "acc_tf_by_n": by}


def main():
    global NUM, DEV
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--eval_bs", type=int, default=32)
    ap.add_argument("--parts", default="acc,probe,vswap,stall,tf"); ap.add_argument("--per_n", type=int, default=1000); ap.add_argument("--probe_ts", default=",".join(map(str, T_SET))); ap.add_argument("--probe_steps", type=int, default=2000, help="smokes only")
    a = ap.parse_args(); dev = a.device; DEV = dev; NUM = torch.tensor(N.NUM_IDS, device=dev); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    m, args = load(a.run, dev); TE = N.test_set(a.per_n); fo = os.path.join(a.run, "eval_vb.json")
    out = json.load(open(fo)) if os.path.exists(fo) else {}
    out.update({"run": a.run, "mode": m.mode, "mct": bool(args.get("mct")), "eval_bs": a.eval_bs, "dtype": "fp32",
                "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "per_n": a.per_n})
    for part in a.parts.split(","):
        if part == "tf" and not args.get("mct"):
            continue
        t0 = time.time()
        r = {"acc": lambda: part_acc(m, TE, dev, a.eval_bs), "probe": lambda: part_probe(m, TE, dev, a.eval_bs, tuple(int(v) for v in a.probe_ts.split(",")), a.probe_steps),
             "vswap": lambda: part_vswap(m, TE, dev, a.eval_bs), "stall": lambda: part_stall(m, TE, dev, a.eval_bs), "tf": lambda: part_tf(m, TE, dev, a.eval_bs)}[part]()
        if part == "probe" and "probe" in out:
            out["probe"].update(r.pop("probe"))
        out.update(r); out.setdefault("part_sec", {})[part] = round(time.time() - t0)
        json.dump(out, open(fo + ".tmp", "w", newline="\n")); os.replace(fo + ".tmp", fo); print("PART", part, "done", round(time.time() - t0), "s", flush=True)
    print("DONE", fo, flush=True)


if __name__ == "__main__":
    main()
