"""DAS-USED pilot (docs/THEORY_DASUSED_DRAFT_2026-10-14.md rev 2, 271db224, §7; THEORY notes 203-2 / 203-2a). A NEW file: boxes/*.py are imported
read-only (halt173.load_model, nested170, deltamem); DM.FAST_ADD = args.fast_add, DM.CHUNK_DELTA = 0, READ_CLAMP None. The state patch runs in this
script's own pass loop (no monkeypatch): S_t -> patch -> m.write for passes t+1..K -> m.dec_logits (= think(override=(t, .))).
Stage P0a (this file's first block; forward only): the CONDITIONED whole-state SERIAL (draft §2.5, notes 203-2 / 203-2a)
  (i)  s170 re-run: SPOOL (n = 8, rng 17300, 2,200 items), serial_items(rng 17200) (an exact copy that also returns the hybrid ITEM; gate SI: its
       recipients / donors / hybrid values equal nested170.serial_items'), swap the donor's whole state after pass t in {1, 2, 4, 6} (t < K), K passes.
       Per pair: the clean answers (K' = K, no swap) of recipient / donor / hybrid item and the swapped answer. GATE S170: the unconditioned hybrid /
       donor / recipient rates equal eval.json s170.serial[t] within 0.02 where eval has t.
  (ii) n 3-8 extension: the same donor construction on the test items (all <= 1,000 per n), rng 20371, swaps at t in {1, 2}; eligibility per t: the
       root has a bare leaf, h != root, height(root) >= 2t + 1, hybrid answer not in {own, donor}.
  Readings per (model, t[, n]): hybrid / own / donor / other shares on ALL rows and on CLEAN rows (ccc: recipient, donor and hybrid item all answered
  correctly without a swap), each with its N (note 203-2: N < 100 -> N/A).
Every block records card / dtype ("bf16 autocast trunk, fp32 memory branch") / bs_eval / torch / driver / script git hash / wall seconds.
-> <run>/eval_dasused.json (block "serial_cond") + <run>/eval_dasused_rows.npz (serial_i_t*, serial_ii_t*)
    python scratch/dasused.py <run> --device cuda:1 --stage P0 [--bs 250] [--memcheck]"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.halt173 as H  # noqa: E402
import boxes.nested170 as N  # noqa: E402

DTYPE = "bf16 autocast trunk, fp32 memory branch"


def git_hash(path):
    try:
        return subprocess.run(["git", "log", "-1", "--format=%H", "--", path], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:
        return None


def height(nd):
    return 0 if nd.op is None else 1 + max(height(nd.l), height(nd.r))


def serial_items_h(items, rng):
    """nested170.serial_items VERBATIM (same rng draws, same keep rule), additionally returning the hybrid ITEM (hb: h's operands replaced, the bare
    operand kept) and the handle path. -> (rec, don, hyb_value, hyb_item)."""
    rec, don, hyb, hyb_it = [], [], [], []
    for it in items:
        root = it["root"]
        if root.op is None:
            continue
        side = "l" if root.l.op is None else ("r" if root.r.op is None else None)
        path = N._leftmost_ready(root)
        if side is None or path is None or len(path) == 0:
            continue
        got = None
        for _ in range(10):
            d = N._copy(root); hn = d
            for p_ in path:
                hn = getattr(hn, p_)
            v = N.apply(hn.op, hn.l.v, hn.r.v); v2 = int(rng.choice([x for x in range(N.P) if x != v])); prs = N.PAIRS[(hn.op, v2)]; a_, b_ = prs[int(rng.integers(0, len(prs)))]
            hb = N._copy(root); hh = hb
            for p_ in path:
                hh = getattr(hh, p_)
            hh.l.v, hh.r.v = a_, b_; hn.l.v, hn.r.v = a_, b_; bare = getattr(d, side); bare.v = int(rng.choice([x for x in range(N.P) if x != bare.v]))
            try:
                got = (N.value(hb), N.value(d)); break
            except KeyError:
                continue
        if got is None:
            continue
        hyb_v, don_v = got
        if hyb_v != N.value(root) and hyb_v != don_v:
            rec.append(it); don.append({"syms": N.render(d), "t": don_v, "root": d, "n": it["n"]}); hyb.append(hyb_v)
            hyb_it.append({"syms": N.render(hb), "t": hyb_v, "root": hb, "n": it["n"]})
    return rec, don, hyb, hyb_it


# ---------------------------------------------------------------- P0b: the B sets (draft §2.1-2.5)
def red_paths(root):
    """the paths (in the ORIGINAL tree) of nested170.reductions' nodes, in order (reductions replace nodes in place, so paths stay valid)."""
    nd = N._copy(root); out = []
    while nd.op is not None:
        path = N._leftmost_ready(nd); tg = nd
        for p_ in path:
            tg = getattr(tg, p_)
        out.append(tuple(path)); tg.v = N.apply(tg.op, tg.l.v, tg.r.v); tg.op = tg.l = tg.r = None
    return out


def at(nd, path):
    for p_ in path:
        nd = getattr(nd, p_)
    return nd


def reexpand(nd, v, rng):
    """node nd (an op node) re-expanded top-down at value v by the generator's conditional (Alg. 1): (a, b) uniform among PAIRS[(op, v)]."""
    prs = N.PAIRS[(nd.op, v)]; a_, b_ = prs[int(rng.integers(0, len(prs)))]
    for ch, w in ((nd.l, a_), (nd.r, b_)):
        if ch.op is None:
            ch.v = w
        else:
            reexpand(ch, w, rng)


def forced(root, path, v):
    """F_t(x, v): the tree with the node at path replaced by a leaf of value v -> value (KeyError on division by 0)."""
    d = N._copy(root)
    if not path:
        return v
    setattr(at(d, path[:-1]), path[-1], N.Node(v=v))
    return N.value(d)


def late_leaf(root, paths, t):
    """the late leaf: the root's leaf child if it has one, else the leaf operand of the latest reduction > t whose node lies outside node t's subtree."""
    for side in ("l", "r"):
        if getattr(root, side).op is None:
            return (side,)
    pt = paths[t - 1]
    for p in reversed(paths[t:]):
        if p[:len(pt)] == pt:
            continue
        nd = at(root, p)
        for side in ("l", "r"):
            if getattr(nd, side).op is None:
                return p + (side,)
    return None


def make_pair(it, t, rng, same=False):
    """draft §2.3: (donor item, y', hybrid item) or None. same=True: the same-value control (v_t^d = v_t)."""
    root = it["root"]; paths = red_paths(root)
    if len(paths) < t + 1:
        return None, "n<t+1"
    pt = paths[t - 1]; ll = late_leaf(root, paths, t)
    if ll is None:
        return None, "no_late_leaf"
    try:
        y = N.value(root); vt = N.reductions(root)[t - 1]
        vd = vt if same else int(rng.choice([x for x in range(N.P) if x != vt]))
        d = N._copy(root); reexpand(at(d, pt), vd, rng); lf = at(d, ll); lf.v = int(rng.choice([x for x in range(N.P) if x != lf.v]))
        yd = N.value(d); yc = forced(root, pt, vd)
        hb = N._copy(root); par = at(hb, pt[:-1]) if pt else None
        if pt:
            setattr(par, pt[-1], N._copy(at(d, pt)))
        else:
            hb = N._copy(d)
        yh = N.value(hb)
    except KeyError:
        return None, "div0"
    if not same and len({y, yc, yd}) < 3:
        return None, "not_distinct"
    if same and (yc != y or yd == y):
        return None, "same_ctrl_invalid"
    don = {"syms": N.render(d), "t": yd, "root": d, "n": it["n"]}; hyb = {"syms": N.render(hb), "t": yh, "root": hb, "n": it["n"]}
    return (don, yc, hyb, {"vt": vt, "vd": vd, "path": pt, "late": ll}), "ok"


class Runner:
    def __init__(self, m, dev, bs):
        self.m, self.dev, self.bs = m, dev, bs
        amp = torch.bfloat16 if (dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8) else None
        self.ac = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
        self.amp = amp

    @torch.no_grad()
    def states(self, items, t):
        """S after t think passes from the zero state (nested170 inject_states), per chunk of bs -> list of chunk tensors."""
        out = []
        for i in range(0, len(items), self.bs):
            ch = items[i:i + self.bs]; x, msk = self.m.think_inputs(ch, self.dev)
            with self.ac():
                S = self.m.mem.zero_state(len(ch), self.dev)
                for _ in range(t):
                    S = self.m.write(S, x, msk)
            out.append(S)
        return out

    @torch.no_grad()
    def answers(self, items, K, override=None):
        """nested170.inject_answers: K passes from the zero state; override = (t, [chunk states]) replaces S after pass t; argmax over the 11 values."""
        out = []
        for ci, i in enumerate(range(0, len(items), self.bs)):
            ch = items[i:i + self.bs]; x, msk = self.m.think_inputs(ch, self.dev)
            with self.ac():
                S = self.m.mem.zero_state(len(ch), self.dev)
                for t in range(1, K + 1):
                    S = self.m.write(S, x, msk)
                    if override is not None and override[0] == t:
                        S = override[1][ci].to(S.dtype)
                lg = self.m.dec_logits(S, ch, self.dev)
            out.append(lg[:, N.NUM_IDS].argmax(-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=int)


def patch(Sr, Sd, R):
    """the DAS patch on the flattened state: (S_r - R R^T S_r) + R R^T S_d. R None / (D, 0) -> S_r exactly; R == "I" -> (S_r - S_r) + S_d = S_d exactly."""
    if isinstance(R, str) and R == "I":
        return (Sr - Sr) + Sd
    if R is None or R.shape[1] == 0:
        return Sr - torch.zeros_like(Sr) + torch.zeros_like(Sr)
    X, Y = Sr.flatten(1).float(), Sd.flatten(1).float(); Rf = R.float()
    return ((X - (X @ Rf) @ Rf.T) + (Y @ Rf) @ Rf.T).view_as(Sr).to(Sr.dtype)


@torch.no_grad()
def patched_answers(R_, rec, don, t, K, Rm):
    """per chunk: S_t of the recipients and the donors, patched with Rm, K passes total (override after pass t) -> answers."""
    out = []
    for i in range(0, len(rec), R_.bs):
        cr, cd = rec[i:i + R_.bs], don[i:i + R_.bs]; Sr, Sd = R_.states(cr, t)[0], R_.states(cd, t)[0]
        out.append(R_.answers(cr, K, override=(t, [patch(Sr, Sd, Rm)])))
    return np.concatenate(out) if out else np.zeros(0, dtype=int)


def share4(got, y, yc, yd, mask):
    n_ = int(mask.sum())
    if not n_:
        return {"n": 0}
    g = got[mask]
    return {"n": n_, "ycf": float((g == yc[mask]).mean()), "ydonor": float((g == yd[mask]).mean()), "yown": float((g == y[mask]).mean()),
            "other": float(((g != yc[mask]) & (g != yd[mask]) & (g != y[mask])).mean())}


def p0b(R_, TE, K, rows):
    """draft §7 P0 (rest): B sets at points t = 1, 2, 3 (every TE item n in [t+1, 8], one matched donor each, rng [20302, t]; same-value control
    donors rng [20302, t, 1]); gates CF / HYB / distinct / IDENT / WHOLE / CAP; clean answers of x / x_d / x_hyb -> ccc; the whole-state matched swap
    at each point; liveness (point-2 donors, whole-state swap at tau = 1..4)."""
    out = {"points": {}, "gates": {}}; Kt = K
    sm = R_.answers(TE[:4], K); out["gates"]["SMOKE_plain"] = bool(len(sm) == 4)
    for t in (1, 2, 3):
        rng, rng_s = np.random.default_rng([20302, t]), np.random.default_rng([20302, t, 1]); B, cnt = [], {}
        for it in TE:
            if not (t + 1 <= it["n"] <= 8):
                continue
            pr, why = make_pair(it, t, rng); cnt[why] = cnt.get(why, 0) + 1
            if pr is None:
                continue
            ps, why_s = make_pair(it, t, rng_s, same=True)
            B.append((it, pr, ps))
        rec = [b[0] for b in B]; don = [b[1][0] for b in B]; hyb = [b[1][2] for b in B]; info = [b[1][3] for b in B]
        y = np.array([N.value(it["root"]) for it in rec]); yc = np.array([b[1][1] for b in B]); yd = np.array([d["t"] for d in don]); nn = np.array([it["n"] for it in rec])
        g_cf = all(forced(it["root"], inf["path"], inf["vt"]) == yy for it, inf, yy in zip(rec, info, y))
        g_hyb = all(N.eval_tokens(h["syms"]) == c and h["t"] == c for h, c in zip(hyb, yc)); g_dist = bool(((y != yc) & (y != yd) & (yc != yd)).all())
        c_x, c_d, c_h = R_.answers(rec, Kt), R_.answers(don, Kt), R_.answers(hyb, Kt); ccc = (c_x == y) & (c_d == yd) & (c_h == yc)
        whole = patched_answers(R_, rec, don, t, Kt, "I"); blk = {"n_pairs": len(B), "drops": cnt, "n_same_ctrl": int(sum(b[2] is not None for b in B))}
        if t == 2:                                                          # G-IDENT / G-WHOLE on every B recipient at the primary point
            plain = c_x; ident = patched_answers(R_, rec, don, t, Kt, torch.zeros(1, 0)); sd_whole = []
            for i in range(0, len(rec), R_.bs):
                sd_whole.append(R_.answers(rec[i:i + R_.bs], Kt, override=(t, R_.states(don[i:i + R_.bs], t))))
            sd_whole = np.concatenate(sd_whole)
            out["gates"]["IDENT"] = {"equal": int((ident == plain).sum()), "n": len(rec), "pass": bool((ident == plain).all())}
            out["gates"]["WHOLE"] = {"equal": int((whole == sd_whole).sum()), "n": len(rec), "pass": bool((whole == sd_whole).all())}
            live = {}
            for tau in range(1, 5):
                if tau < Kt:
                    gl = patched_answers(R_, rec, don, tau, Kt, "I")
                    live[str(tau)] = {"all": share4(gl, y, yc, yd, np.ones(len(rec), bool)), "ccc": share4(gl, y, yc, yd, ccc),
                                      "by_n": {str(n_): share4(gl, y, yc, yd, ccc & (nn == n_)) for n_ in sorted(set(nn.tolist()))}}
            blk["liveness"] = live
        ns = sorted(set(nn.tolist())); nccc = {str(n_): int((ccc & (nn == n_)).sum()) for n_ in ns}
        blk.update({"gates": {"CF": bool(g_cf), "HYB": bool(g_hyb), "DISTINCT": g_dist},
                    "clean_acc_by_n": {str(n_): {"x": float((c_x == y)[nn == n_].mean()), "x_d": float((c_d == yd)[nn == n_].mean()), "x_hyb": float((c_h == yc)[nn == n_].mean()),
                                                 "n": int((nn == n_).sum())} for n_ in ns},
                    "n_ccc_by_n": nccc, "eligible_strata": [n_ for n_, c_ in nccc.items() if c_ >= 100], "n_ccc": int(ccc.sum()),
                    "b0_by_n": {str(n_): float((c_x == yc)[nn == n_].mean()) for n_ in ns},
                    "whole_state": {**share4(whole, y, yc, yd, ccc), "n_ccc": int(ccc.sum()), "all": share4(whole, y, yc, yd, np.ones(len(rec), bool)),
                                    "by_n": {str(n_): share4(whole, y, yc, yd, ccc & (nn == n_)) for n_ in ns}}})
        out["points"][str(t)] = blk
        rows["B_t%d" % t] = np.stack([np.arange(len(rec)), nn, y, yc, yd, c_x, c_d, c_h, whole, ccc.astype(int)], 1)
        rows["B_t%d_hash" % t] = np.array([int(hashlib.sha1(" ".join(map(str, it["syms"])).encode()).hexdigest()[:12], 16) for it in rec], dtype=np.int64)
        print("P0b t", t, "pairs", len(B), "drops", cnt, "gates", blk["gates"], "n_ccc", blk["n_ccc"], "whole", {k_: blk["whole_state"][k_] for k_ in ("ycf", "ydonor", "yown", "n")}, flush=True)
    out["gates"]["CAP"] = "ccc counts and eligible strata written before any IIA (P0)"; out["gates"]["DISJ"] = "pending P2 (A is built there); B recipients are TE items"
    out["row_cols"] = ["pair", "n", "y", "y_cf", "y_d", "clean_x", "clean_xd", "clean_xhyb", "whole_swap", "ccc"]
    out["def"] = ("draft §2.3-2.5 / §7 P0: matched donor = node t re-expanded at v_t^d != v_t by Alg. 1 + the late leaf resampled; y, y', y_d distinct; hybrid = x with "
                  "node t's subtree from x_d; ccc = x, x_d, x_hyb solved clean (K' = K); whole_state = S_t swapped whole (R = I) -> shares of y' / y_d / y on ccc pairs")
    return out


# ---------------------------------------------------------------- P1 / P2: the DAS fit core (draft §3, §7; das167's Rot, adapted to nested170)
class Rot(torch.nn.Module):
    """R = QR(W).Q, an orthonormal (D, k) basis of a free (D, k) parameter (das167.Rot)."""
    def __init__(self, D, k, dev, init=None, gen=None):
        super().__init__()
        self.W = torch.nn.Parameter(init.clone().to(dev) if init is not None else (torch.randn(D, k, generator=gen) / D ** 0.5).to(dev))

    def R(self):
        return torch.linalg.qr(self.W).Q


def cont_logits(R_, items, Sp, t, K, ckpt=True):
    """the patched state Sp (b, ...) after pass t -> passes t+1..K (frozen model, gradient to Sp) -> the decoder's 11 value logits at ' ='.
    No @torch.no_grad anywhere on this path (gradient-flow rule)."""
    from torch.utils.checkpoint import checkpoint
    m = R_.m; x, msk = m.think_inputs(items, R_.dev); S = Sp
    with R_.ac():
        for _ in range(t + 1, K + 1):
            S = checkpoint(m.write, S, x, msk, use_reentrant=False) if ckpt else m.write(S, x, msk)
        return m.dec_logits(S, items, R_.dev)[:, N.NUM_IDS].float()


def fit_step(R_, rot, items, Sr, Sd, ycf, t, K, micro=1, ckpt=True):
    """one DAS step on a fit batch (gradient accumulation over `micro` chunks): CE(patched logits, y') -> loss (backward done)."""
    tot = 0.0
    for idx in np.array_split(np.arange(len(items)), micro):
        R = rot.R(); X, Y = Sr[idx].flatten(1).float(), Sd[idx].flatten(1).float()
        Sp = ((X - (X @ R) @ R.T) + (Y @ R) @ R.T).view_as(Sr[idx])
        lg = cont_logits(R_, [items[i] for i in idx], Sp, t, K, ckpt)
        loss = torch.nn.functional.cross_entropy(lg, torch.tensor(ycf[idx], device=R_.dev)) * (len(idx) / len(items)); loss.backward(); tot += float(loss)
    return tot


def p1(R_, TE, K, fit_bs=64, steps=50):
    """draft §7 P1: G-MEM at this model's largest shape (t = 2, the longest n = 8 B rows, fit batch fit_bs, pass checkpointing) for fit_micro 1 / 2,
    G-GRAD at the first step, and --time_steps (s/step at the passing micro). k = 64 (the largest registered rank)."""
    m = R_.m
    for p_ in m.parameters():
        p_.requires_grad_(False)
    rng = np.random.default_rng([20302, 2]); B = []
    for it in TE:
        if it["n"] == 8:
            pr, _ = make_pair(it, 2, rng)
            if pr is not None:
                B.append((it, pr))
        if len(B) >= fit_bs:
            break
    items = [b[0] for b in B]; don = [b[1][0] for b in B]; ycf = np.array([b[1][1] for b in B])
    Sr, Sd = R_.states(items, 2)[0].detach(), R_.states(don, 2)[0].detach(); D = Sr[0].numel(); out = {"D": int(D), "k": 64, "fit_bs": fit_bs, "t": 2, "n": 8, "ckpt_passes": True}
    for micro in (1, 2):
        rot = Rot(D, 64, R_.dev, gen=torch.Generator().manual_seed(20381 + 64)); torch.cuda.reset_peak_memory_stats(R_.dev); torch.cuda.empty_cache()
        try:
            fit_step(R_, rot, items, Sr, Sd, ycf, 2, K, micro=micro); pk = torch.cuda.max_memory_allocated(R_.dev) / 2 ** 30
        except torch.OutOfMemoryError:
            pk = float("inf")
        out["peak_gib_micro%d" % micro] = pk
        if micro == 1 or "grad" not in out:
            gn = float(rot.W.grad.norm()) if rot.W.grad is not None else 0.0
            out["grad"] = {"R_grad_norm": gn, "model_params_frozen": all((not p_.requires_grad) and p_.grad is None for p_ in m.parameters()), "pass": gn > 0}
        if pk <= 6.5:
            out["fit_micro"] = micro; break
        del rot; torch.cuda.empty_cache()
    out["G_MEM"] = {"pass": "fit_micro" in out, "limit_gib": 6.5}
    if "fit_micro" in out:
        rot = Rot(D, 64, R_.dev, gen=torch.Generator().manual_seed(20381 + 64)); opt = torch.optim.Adam(rot.parameters(), lr=1e-3); torch.cuda.synchronize(); t0 = time.time()
        for _ in range(steps):
            opt.zero_grad(); fit_step(R_, rot, items, Sr, Sd, ycf, 2, K, micro=out["fit_micro"]); opt.step()
        torch.cuda.synchronize(); out["s_per_step"] = (time.time() - t0) / steps; out["time_steps"] = steps
    return out


def shares(got, own, don, hyb, mask):
    n_ = int(mask.sum())
    if not n_:
        return {"n": 0}
    g, o, d, h = got[mask], own[mask], don[mask], hyb[mask]
    return {"n": n_, "hybrid": float((g == h).mean()), "own": float((g == o).mean()), "donor": float((g == d).mean()), "other": float(((g != h) & (g != o) & (g != d)).mean()),
            "na": n_ < 100}


def run_serial(R, rec, don, hyb_v, hyb_it, ts, K, tag, rows):
    """clean answers once, then the swaps at each t -> {t: {all, clean, (by_n)}}; rows[tag_t*] = per-pair arrays."""
    own = np.array([N.value(it["root"]) for it in rec]); dv = np.array([it["t"] for it in don]); hv = np.array(hyb_v); nn = np.array([it["n"] for it in rec])
    c_rec, c_don, c_hyb = R.answers(rec, K), R.answers(don, K), R.answers(hyb_it, K); ccc = (c_rec == own) & (c_don == dv) & (c_hyb == hv); out = {}
    for t, elig in ts:
        Sd = R.states(don, t); got = R.answers(rec, K, override=(t, Sd)); e = np.array(elig, dtype=bool)
        blk = {"all": shares(got, own, dv, hv, e), "clean": shares(got, own, dv, hv, e & ccc),
               "clean_rec_don_only": shares(got, own, dv, hv, e & (c_rec == own) & (c_don == dv))}
        if len(set(nn.tolist())) > 1:
            blk["by_n"] = {str(n_): {"all": shares(got, own, dv, hv, e & (nn == n_)), "clean": shares(got, own, dv, hv, e & ccc & (nn == n_))} for n_ in sorted(set(nn.tolist()))}
        out[str(t)] = blk
        rows["%s_t%d" % (tag, t)] = np.stack([np.arange(len(rec)), nn, own, dv, hv, c_rec, c_don, c_hyb, got, e.astype(int)], 1)
    rows["%s_hash" % tag] = np.array([int(hashlib.sha1(" ".join(map(str, it["syms"])).encode()).hexdigest()[:12], 16) for it in rec], dtype=np.int64)
    out["clean_acc"] = {"recipient": float((c_rec == own).mean()), "donor": float((c_don == dv).mean()), "hybrid_item": float((c_hyb == hv).mean()), "n_ccc": int(ccc.sum()), "n": len(rec)}
    out["row_cols"] = ["pair", "n", "y_own", "y_donor", "y_hyb", "clean_rec", "clean_don", "clean_hyb", "swapped", "eligible"]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--device", default="cuda:1"); ap.add_argument("--stage", default="P0", choices=["P0", "P1", "P2", "B"])
    ap.add_argument("--bs", type=int, default=250); ap.add_argument("--memcheck", action="store_true"); ap.add_argument("--time_steps", type=int, default=0)
    a = ap.parse_args()
    rd = os.path.join(ROOT, "boxes", "runs_fsq", a.run); kind, m, ra = H.load_model(rd, a.device); assert kind == "nested", kind
    DM.FAST_ADD = bool(ra.get("fast_add")); DM.CHUNK_DELTA = 0; DM.READ_CLAMP = None; K = int(ra["think_k"]); t0 = time.time(); R = Runner(m, a.device, a.bs)
    meta = {"card": torch.cuda.get_device_name(a.device) if a.device.startswith("cuda") else "cpu", "dtype": DTYPE if R.amp is not None else "fp32", "bs_eval": a.bs,
            "torch": torch.__version__, "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.split()[0]
            if a.device.startswith("cuda") else None, "script_git": git_hash("scratch/dasused.py")}
    TE = N.test_set(int(ra.get("per_n") or 1000)); r_sp = np.random.default_rng(17300); SPOOL = []
    for _ in range(int(ra.get("serial_pool") or 2200)):
        r__, t__ = N.generate(r_sp, 8); s__ = N.render(r__); SPOOL.append({"n": 8, "syms": s__, "t": t__, "depth": N.depth(s__), "root": r__})
    if a.memcheck:                                                          # the largest shape: n = 8 rows at bs, K passes, decode
        torch.cuda.reset_peak_memory_stats(a.device); R.answers(SPOOL[:a.bs], K, override=(2, R.states(SPOOL[:a.bs], 2)))
        print(a.run, "memcheck peak %.2f GiB (bs %d, K %d)" % (torch.cuda.max_memory_allocated(a.device) / 2 ** 30, a.bs, K), flush=True); return
    fp = os.path.join(rd, "eval_dasused.json"); res = json.load(open(fp)) if os.path.exists(fp) else {}
    res.update({"run": a.run, "K": K, "add_norm": ra.get("add_norm") or None, "meta": meta}); rows = {}
    if a.stage == "P1":                                                     # draft §7 P1: G-MEM + G-GRAD + --time_steps
        tb = time.time(); res["p1"] = {**p1(R, TE, K, steps=a.time_steps or 50), **meta, "sec": round(time.time() - tb), "done": True}; print(a.run, "P1", res["p1"], flush=True)
    elif a.stage == "P0" and res.get("serial_cond", {}).get("done") and not res.get("p0b", {}).get("done"):   # P0a done: the P0 remainder (draft §7)
        tb = time.time(); res["p0b"] = {**p0b(R, TE, K, rows), **meta, "sec": round(time.time() - tb), "done": True}
    elif a.stage == "P0":
        rec, don, hyb_v, hyb_it = serial_items_h(SPOOL, np.random.default_rng(17200)); r0, d0, h0 = N.serial_items(SPOOL, np.random.default_rng(17200))
        gate_si = len(r0) == len(rec) and all(x is y for x, y in zip(r0, rec)) and [d["syms"] for d in d0] == [d["syms"] for d in don] and list(h0) == list(hyb_v)
        ts_i = [(t, [True] * len(rec)) for t in (1, 2, 4, 6) if t < K]
        s_i = run_serial(R, rec, don, hyb_v, hyb_it, ts_i, K, "serial_i", rows)
        ev = json.load(open(os.path.join(rd, "eval.json")))["s170"]["serial"]; g_ = {}
        for t in (str(x) for x, _ in ts_i):
            if t in ev:
                a_ = s_i[t]["all"]; g_[t] = {k_: (a_[{"recipient": "own"}.get(k_, k_)], ev[t][k_]) for k_ in ("hybrid", "donor", "recipient")}
        gate_s170 = {"pairs": g_, "max_abs_diff": max(abs(x - y) for d_ in g_.values() for x, y in d_.values()) if g_ else None}
        gate_s170["pass"] = gate_s170["max_abs_diff"] is not None and gate_s170["max_abs_diff"] <= 0.02
        print(a.run, "(i) n_pairs", len(rec), "SI", gate_si, "S170", gate_s170["max_abs_diff"], "clean", s_i["clean_acc"], flush=True)
        ext = [it for it in TE if 3 <= it["n"] <= 8]; rec2, don2, hv2, hi2 = serial_items_h(ext, np.random.default_rng(20371))
        hroot = [height(it["root"]) for it in rec2]; ts_ii = [(t, [h_ >= 2 * t + 1 for h_ in hroot]) for t in (1, 2)]
        s_ii = run_serial(R, rec2, don2, hv2, hi2, ts_ii, K, "serial_ii", rows)
        print(a.run, "(ii) n_pairs", len(rec2), {t: {"all": s_ii[t]["all"], "clean": s_ii[t]["clean"]} for t in ("1", "2")}, flush=True)
        res["serial_cond"] = {"s170": s_i, "ext": s_ii, "gates": {"SI_reproduces_serial_items": bool(gate_si), "S170": gate_s170},
                              "def": "notes 203-2 / 203-2a: whole-state swap of the donor's S_t, K passes; clean = recipient, donor and hybrid item correct without a swap; "
                                     "(ii) eligibility per t: bare root leaf, h != root, height(root) >= 2t+1, hybrid not in {own, donor}; n < 100 -> na",
                              "pool_i": "SPOOL n 8 rng 17300 (2,200), serial_items rng 17200", "pool_ii": "test items n 3-8 (all), rng 20371", **meta, "sec": round(time.time() - t0), "done": True}
    json.dump(res, open(fp, "w", newline="\n"), indent=1)
    rp = os.path.join(rd, "eval_dasused_rows.npz"); old = dict(np.load(rp)) if os.path.exists(rp) else {}; old.update(rows); np.savez_compressed(rp, **old)
    print(a.run, "P0 done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
