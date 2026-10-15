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


# ---------------------------------------------------------------- P2 (pilot scope, Theory 2026-10-14 21:4x): t = 2; one lr chosen on rms s1's validation
# pairs; k in {16, 64}; on-support refit for the best k; Haar + exact-energy + shuffled controls; resample necessity; G-DISRUPT; step cap 2,000.
def label(f):
    """draft §3.5 -> UR-FOUND[-REDUNDANT] / ILLUSION-SUSPECT / USED-UNVERIFIED / NONE-FOUND / FIT-FAILURE / N/A / PARTIAL."""
    if not f.get("eligible", True):
        return "N/A"
    c3 = f["iia_star"] >= 0.70 and f["iia_star"] - max(f["ctrl_max"], f["b0"]) >= 0.50 and f["shuffled"] <= 0.20
    if c3:
        if not f["supp_pass"]:
            return "USED-UNVERIFIED"
        if f["iia_on"] >= f["iia_star"] - 0.15 and f["per_n_floor_ok"]:
            ex = f.get("necessity_excess")
            return "UR-FOUND" + ("-REDUNDANT" if ex is not None and ex < 0.10 else "")
        if f["iia_on"] < f["iia_star"] - 0.30:
            return "ILLUSION-SUSPECT"
        return "PARTIAL"
    if f["iia_max_all_k"] < 0.30:
        return "NONE-FOUND" if f["gfit"] else "FIT-FAILURE"
    return "PARTIAL"


def gate_smoke_label():
    """G-SMOKE: label() on theory/dasused/toy_illusion.json (G-SUPP and the per-n floor set to pass; necessity excess vs the Haar mean)."""
    toy = json.load(open(os.path.join(ROOT, "theory", "dasused", "toy_illusion.json")))["models"]; want = {"USED": "UR-FOUND", "UNUSED": "ILLUSION-SUSPECT", "USED-LOW": "UR-FOUND"}; got = {}
    for nm, d in toy.items():
        if nm not in want:
            continue
        f = {"iia_star": d["iia"], "ctrl_max": max(d["haar_iia_max"], d["energy_iia_max"]), "b0": d["b0"], "shuffled": d.get("shuffled_iia", 0.0), "supp_pass": True,
             "iia_on": d["iia_on_support"], "per_n_floor_ok": True, "necessity_excess": d.get("necessity_resample_on_support_excess"),
             "iia_max_all_k": max(v["iia"] for v in d["by_k"].values()), "gfit": True}
        got[nm] = label(f)
    return {"got": got, "want": want, "pass": all(got.get(k_) == v for k_, v in want.items())}


def collect(R_, items, t):
    return np.concatenate([S.float().cpu().numpy() for S in R_.states(items, t)]) if items else None


@torch.no_grad()
def iia_eval(R_, items, Sr, Sd, ycf, t, K, Rm, mode="patch", So=None):
    """answers after the patch (mode patch: (I - RR^T) S_r + RR^T S_d; mode resample: S_r + RR^T (S_o - S_r)) -> predictions."""
    out = []
    for i in range(0, len(items), R_.bs):
        sl = slice(i, i + R_.bs); a_ = torch.tensor(Sr[sl], device=R_.dev); b_ = torch.tensor((So if mode == "resample" else Sd)[sl], device=R_.dev)
        out.append(R_.answers(items[sl], K, override=(t, [patch(a_, b_, Rm)])))
    return np.concatenate(out) if out else np.zeros(0, int)


SM = {"steps": 2000, "every": 100, "n_fit": 4000, "n_val": 500, "n_supp": 6000, "n_B": None}     # --smoke shrinks these (functional test only)


def fit_rot(R_, K, t, fit, val, k, lr, init=None, steps=None, bs=64, micro=1, every=None, patience=3, Vs=None):
    """Adam on R (or on W' with R = orth(Vs W') for the on-support refit); early stop on validation IIA; returns (R (D, k) detached, steps, val_best)."""
    steps = steps or SM["steps"]; every = every or SM["every"]; torch.manual_seed(20381 + k); rng = np.random.default_rng(20391 + k); its, Sr, Sd, y = fit; D = Sr[0].size
    if Vs is None:
        rot = Rot(D, k, R_.dev, init=init, gen=torch.Generator().manual_seed(20381 + k)); getR = rot.R
    else:
        rot = Rot(Vs.shape[1], k, R_.dev, init=init); getR = lambda: torch.linalg.qr(Vs @ rot.W).Q  # noqa: E731
    opt = torch.optim.Adam(rot.parameters(), lr=lr); best, bestR, bad, st = -1.0, None, 0, 0
    for st in range(1, steps + 1):
        b = rng.integers(0, len(its), bs); opt.zero_grad()
        for idx in np.array_split(b, micro):
            R = getR(); X = torch.tensor(Sr[idx], device=R_.dev); Y = torch.tensor(Sd[idx], device=R_.dev); Xf, Yf = X.flatten(1), Y.flatten(1)
            Sp = ((Xf - (Xf @ R) @ R.T) + (Yf @ R) @ R.T).view_as(X)
            lg = cont_logits(R_, [its[i] for i in idx], Sp, t, K); loss = torch.nn.functional.cross_entropy(lg, torch.tensor(y[idx], device=R_.dev)) * (len(idx) / len(b)); loss.backward()
        opt.step()
        if st % every == 0 or st == steps:
            with torch.no_grad():
                Rd = getR().detach()
            v = float((iia_eval(R_, val[0], val[1], val[2], val[3], t, K, Rd) == val[3]).mean())
            if v > best + 1e-4:
                best, bestR, bad = v, Rd.clone(), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    return (bestR if bestR is not None else getR().detach()), st, best


def build_A(R_, TE_syms, excl, t, K, n_fit=4000, n_val=500, max_raw=20000):
    """draft §2.4 fit set A: train_items (rng 20301, the run's exclusion set), n in [t+1, 8]; matched donors rng 20303; ccc pairs until n_fit + n_val."""
    rng, rd_ = np.random.default_rng(20301), np.random.default_rng(20303); acc = []; raw = 0
    while len(acc) < n_fit + n_val and raw < max_raw:
        batch = [it for it in N.train_items(rng, 500, 8, excl) if t + 1 <= it["n"] <= 8]; raw += len(batch); prs = []
        for it in batch:
            pr, _ = make_pair(it, t, rd_)
            if pr is not None and tuple(pr[0]["syms"]) not in TE_syms:
                prs.append((it, pr))
        if not prs:
            continue
        rec = [p[0] for p in prs]; don = [p[1][0] for p in prs]; hyb = [p[1][2] for p in prs]; yc = np.array([p[1][1] for p in prs])
        y = np.array([N.value(it["root"]) for it in rec]); yd = np.array([d["t"] for d in don])
        ok = (R_.answers(rec, K) == y) & (R_.answers(don, K) == yd) & (R_.answers(hyb, K) == yc)
        acc += [(rec[i], don[i], yc[i]) for i in np.where(ok)[0]]
    acc = acc[:n_fit + n_val]; return acc[:n_fit], acc[n_fit:], raw


def support(R_, excl, t, n_items=6000):
    """draft §3.3: clean S_t on n_items fresh items (rng 20311), centred; eigen-decomposition via the Gram matrix -> (mu, V (D, r), lam (r,)) on the GPU."""
    rng = np.random.default_rng(20311); its = []
    while len(its) < n_items:
        its += [it for it in N.train_items(rng, 1000, 8, excl) if t + 1 <= it["n"] <= 8]
    its = its[:n_items]; X = torch.tensor(collect(R_, its, t).reshape(n_items, -1), device=R_.dev); mu = X.mean(0, keepdim=True); X = X - mu
    G = (X @ X.T).double().cpu(); lam, U = torch.linalg.eigh(G); keep = lam > lam.max() * 1e-12; lam, U = lam[keep], U[:, keep]
    V = (X.T @ (U.float().to(R_.dev) / lam.sqrt().float().to(R_.dev))); lam = (lam / (n_items - 1)).float().to(R_.dev)
    order = torch.argsort(lam, descending=True); return mu, V[:, order], lam[order], X


def energy(Xc, R):
    return float((Xc @ R).pow(2).sum() / (Xc.shape[0] - 1))


def exact_energy_ctrl(V, lam, Xc, k, Estar, rng, eps_rel=1e-6, n=5):
    """draft §3.4 Prop E: R(beta) = orth((Sigma + eps I)^{beta/2} G), bisection on beta in [-12, 12] to |E - E*| / E* <= 1e-3 (redraw if not bracketed)."""
    eps = float(lam[0]) * eps_rel; out = []; draws = 0; lg_l = torch.log(lam.double() + eps); lg_e = float(np.log(eps))
    while len(out) < n and draws < 50:
        draws += 1; Gm = torch.tensor(rng.standard_normal((V.shape[0], k)), device=V.device, dtype=torch.float32); VG = V.T @ Gm; Gp = Gm - V @ VG   # Gp: G's part outside span(V)

        def R_of(beta):
            """orth((Sigma + eps I)^{beta/2} G) = orth(V diag((lam+eps)^{b/2}) V^T G + eps^{b/2} (I - V V^T) G), computed with the coefficients divided by
            their maximum in log space (QR is scale-invariant), so beta in [-12, 12] never overflows float32."""
            la = (beta / 2) * lg_l; le = (beta / 2) * lg_e; mx = max(float(la.max()), le)
            M = V @ (torch.exp(la - mx).float()[:, None] * VG) + float(np.exp(le - mx)) * Gp; return torch.linalg.qr(M).Q
        lo, hi = -12.0, 12.0; Elo, Ehi = energy(Xc, R_of(lo)), energy(Xc, R_of(hi))
        if not (np.isfinite(Elo) and np.isfinite(Ehi) and Elo <= Estar <= Ehi):
            continue
        for _ in range(60):
            mid = (lo + hi) / 2; Rm = R_of(mid); E = energy(Xc, Rm)
            if abs(E - Estar) / Estar <= 1e-3:
                break
            lo, hi = (mid, hi) if E < Estar else (lo, mid)
        out.append((Rm, abs(E - Estar) / Estar))
    return out                                                              # fewer than n after 50 draws -> G-EN reads the shortfall


def p2(R_, TE, K, ra, run, lr_select, fit_micro=1):
    t = 2; excl = {tuple(it["syms"]) for it in TE if it["n"] >= int(ra.get("excl_min_n", 1))}; r_sp = np.random.default_rng(17300)
    for _ in range(int(ra.get("serial_pool") or 2200)):
        r__, _t = N.generate(r_sp, 8); excl.add(tuple(N.render(r__)))
    TE_syms = {tuple(it["syms"]) for it in TE}; out = {"t": t, "scope": "pilot (Theory 2026-10-14): t 2; one lr from rms s1 val; k 16 / 64; on-support refit for best k; step cap 2000"}
    for p_ in R_.m.parameters():
        p_.requires_grad_(False)
    out["gates"] = {"SMOKE_label": gate_smoke_label()}
    # B (as P0b, rng [20302, 2]) + same-value controls
    rng, rng_s = np.random.default_rng([20302, t]), np.random.default_rng([20302, t, 1]); B = []
    for it in TE:
        if t + 1 <= it["n"] <= 8:
            pr, _ = make_pair(it, t, rng); ps, _ = make_pair(it, t, rng_s, same=True) if pr is not None else (None, None)
            if pr is not None:
                B.append((it, pr, ps))
    rec = [b[0] for b in B]; don = [b[1][0] for b in B]; hyb = [b[1][2] for b in B]; ycf = np.array([b[1][1] for b in B]); y = np.array([N.value(it["root"]) for it in rec])
    yd = np.array([d["t"] for d in don]); nn = np.array([it["n"] for it in rec])
    ccc = (R_.answers(rec, K) == y) & (R_.answers(don, K) == yd) & (R_.answers(hyb, K) == ycf); ix = np.where(ccc)[0]
    nc = {n_: int((nn[ix] == n_).sum()) for n_ in sorted(set(nn.tolist()))}; strata = [n_ for n_, c_ in nc.items() if c_ >= 100]
    ix = np.array([i for i in ix if nn[i] in strata], dtype=int)[:SM["n_B"]]; out["B"] = {"n_pairs": len(B), "n_ccc_by_n": {str(k_): v for k_, v in nc.items()}, "eligible_strata": strata, "n_ccc_eligible": int(len(ix))}
    if len(ix) < 100:
        out["label"] = "N/A"; return out
    Bi = [rec[i] for i in ix]; BSr, BSd = collect(R_, Bi, t), collect(R_, [don[i] for i in ix], t); Bycf, By, Bn = ycf[ix], y[ix], nn[ix]
    sv = [B[i][2] for i in ix]; sv_ok = np.array([s is not None for s in sv]); BSs = collect(R_, [s[0] for s in sv if s is not None], t)
    # A (fit / val) and G-DISJ
    A_fit, A_val, raw = build_A(R_, TE_syms, excl, t, K, SM["n_fit"], SM["n_val"]); out["A"] = {"n_fit": len(A_fit), "n_val": len(A_val), "raw_recipients": raw}
    disj = not ({tuple(a[0]["syms"]) for a in A_fit + A_val} & TE_syms) and not ({tuple(a[1]["syms"]) for a in A_fit + A_val} & {tuple(d["syms"]) for d in don})
    out["gates"]["DISJ"] = {"pass": bool(disj)}
    pack = lambda L: ([a[0] for a in L], collect(R_, [a[0] for a in L], t), collect(R_, [a[1] for a in L], t), np.array([a[2] for a in L]))  # noqa: E731
    fit, val = pack(A_fit), pack(A_val); D = fit[1][0].size; SH = fit[1].shape[1:]
    M = torch.tensor((fit[2] - fit[1]).reshape(len(fit[0]), -1), device=R_.dev); torch.manual_seed(20381); _, _, Vp = torch.svd_lowrank(M, q=64 + 16, niter=4)   # seeded: reproducible PCA init; Vp = Vp[:, :64].contiguous(); del M
    # lr (rms s1 chooses; the others read it)
    if lr_select:
        cand = {}
        for lr in (1e-3, 3e-3):
            _, st, vb = fit_rot(R_, K, t, fit, val, 16, lr, init=Vp[:, :16], micro=fit_micro); cand[str(lr)] = {"val": vb, "steps": st}
        lr = float(max(cand, key=lambda k_: cand[k_]["val"])); out["lr_selection"] = cand
    else:
        lr = float(json.load(open(os.path.join(ROOT, "boxes", "runs_fsq", "s185_rms_s1", "eval_dasused.json")))["p2"]["lr"]); out["lr_from"] = "s185_rms_s1 p2.lr"
    out["lr"] = lr; fits = {}; Rk = {}
    for k in (16, 64):
        init = Vp[:, :16] if k == 16 else torch.cat([Rk[16], Vp[:, :48]], 1)
        Rk[k], st, vb = fit_rot(R_, K, t, fit, val, k, lr, init=init, micro=fit_micro)
        pred = iia_eval(R_, Bi, BSr, BSd, Bycf, t, K, Rk[k]); trn = iia_eval(R_, fit[0][:1000], fit[1][:1000], fit[2][:1000], fit[3][:1000], t, K, Rk[k])
        fits[str(k)] = {"iia_ccc": float((pred == Bycf).mean()), "n_ccc": int(len(ix)), "iia_ccc_by_n": {str(n_): float((pred == Bycf)[Bn == n_].mean()) for n_ in strata if (Bn == n_).any()},
                        "iia_train": float((trn == fit[3][:1000]).mean()), "val_best": vb, "steps": st, "init": "pca" if k == 16 else "k16 + pca pad", "lr": lr}
        print(run, "P2 k", k, fits[str(k)], flush=True)
    out["k"] = fits; kstar = max((16, 64), key=lambda k_: (fits[str(k_)]["iia_ccc"], -k_)); Rs = Rk[kstar]; iia_star = fits[str(kstar)]["iia_ccc"]
    out.update({"best_k": kstar, "iia_star": iia_star})
    gfit = fits["64"]["iia_train"] >= fits["16"]["iia_train"] - 0.05; out["gates"]["FIT"] = {"pass": bool(gfit)}
    # support, on-support refit, controls
    mu, V, lam, Xc = support(R_, excl, t, SM["n_supp"]); tot = float(lam.sum()); out["support"] = {"eps_profile": {}}
    hold = torch.tensor(BSr[:1000].reshape(min(1000, len(BSr)), -1), device=R_.dev) - mu
    for eps in (1e-3, 1e-4, 1e-5, 1e-6):
        r_ = int((lam > lam[0] * eps).sum()); Vs = V[:, :r_]; cap = float((hold @ Vs).pow(2).sum() / hold.pow(2).sum()); out["support"]["eps_profile"][str(eps)] = {"m": r_, "heldout_capture": cap}
    r_ = int((lam > lam[0] * 1e-6).sum()); Vs = V[:, :r_]; cap = out["support"]["eps_profile"]["1e-06"]["heldout_capture"]; supp_pass = cap >= 0.999
    Rproj = torch.linalg.qr(Vs @ (Vs.T @ Rs)).Q; pproj = iia_eval(R_, Bi, BSr, BSd, Bycf, t, K, Rproj)
    Ron_W, st_on, vb_on = fit_rot(R_, K, t, fit, val, kstar, lr, init=(Vs.T @ Rs), micro=fit_micro, Vs=Vs); Ron = Ron_W
    pon = iia_eval(R_, Bi, BSr, BSd, Bycf, t, K, Ron); dd = torch.tensor((BSd - BSr)[:1000].reshape(min(1000, len(BSr)), -1), device=R_.dev) @ Rs @ Rs.T
    omega = float(((dd - (dd @ Vs) @ Vs.T).pow(2).sum()) / dd.pow(2).sum())
    out["support"].update({"eps": 1e-6, "m": r_, "heldout_capture": cap, "G_SUPP": supp_pass, "iia_proj": float((pproj == Bycf).mean()), "iia_on": float((pon == Bycf).mean()),
                           "iia_on_by_n": {str(n_): float((pon == Bycf)[Bn == n_].mean()) for n_ in strata if (Bn == n_).any()}, "omega": omega, "on_steps": st_on, "on_val": vb_on})
    Es = energy(Xc, Rs); rng_h = np.random.default_rng(20321); haar = []
    for _ in range(5):
        Q = torch.linalg.qr(torch.tensor(rng_h.standard_normal((D, kstar)), device=R_.dev, dtype=torch.float32)).Q; haar.append(Q)
    en = exact_energy_ctrl(V, lam, Xc, kstar, Es, np.random.default_rng(20322)); ctrl = [float((iia_eval(R_, Bi, BSr, BSd, Bycf, t, K, Q) == Bycf).mean()) for Q in haar + [e[0] for e in en]]
    perm = np.random.default_rng(20331).permutation(len(fit[3])); fit_sh = (fit[0], fit[1], fit[2], fit[3][perm]); vperm = np.random.default_rng(20332).permutation(len(val[3]))
    Rsh, _, _ = fit_rot(R_, K, t, fit_sh, (val[0], val[1], val[2], val[3][vperm]), 16, lr, init=Vp[:, :16], micro=fit_micro); bperm = np.random.default_rng(20333).permutation(len(Bycf))
    shuf = float((iia_eval(R_, Bi, BSr, BSd, Bycf, t, K, Rsh) == Bycf[bperm]).mean())
    out["controls"] = {"haar_iia_ccc": ctrl[:5], "energy_iia_ccc": ctrl[5:], "energy_rel_err": [e[1] for e in en], "E_star": Es, "E_total": tot, "var_share": Es / tot,
                       "shuffled_iia_ccc": shuf, "G_EN": len(en) == 5 and all(e[1] <= 0.2 for e in en)}
    out["gates"]["EN"] = {"pass": out["controls"]["G_EN"]}
    # necessity (resample, R_on and the 10 controls), G-DISRUPT, locus
    So = BSr[np.random.default_rng(20341).permutation(len(BSr))]
    def nec(Rm):
        return 1.0 - float((iia_eval(R_, Bi, BSr, None, By, t, K, Rm, mode="resample", So=So) == By).mean())
    n_on = nec(Ron); n_ctrl = [nec(Q) for Q in haar + [e[0] for e in en]]
    out["necessity"] = {"resample": {"R_on": n_on, "R_star": nec(Rs), "random_mean": float(np.mean(n_ctrl)), "excess": n_on - float(np.mean(n_ctrl))}}
    sidx = np.where(sv_ok)[0]; dis = {}
    for nm, Rm in (("R_star", Rs), ("R_on", Ron)):
        pr = iia_eval(R_, [Bi[i] for i in sidx], BSr[sidx], BSs, By[sidx], t, K, Rm); dis[nm] = 1.0 - float((pr == By[sidx]).mean())
    out["gates"]["DISRUPT"] = {"own_acc_drop": dis, "n": int(len(sidx)), "pass": all(v <= 0.05 for v in dis.values())}
    HA = R_.m.mem.h_add; blk = int(np.prod(SH[1:])); Rr = Ron.view(*SH, -1)
    out["locus"] = {"mass_add": float(Rr[:HA].pow(2).sum() / Ron.shape[1]), "mass_delta": float(Rr[HA:HA + R_.m.mem.h_del].pow(2).sum() / Ron.shape[1]),
                    "mass_z": float(Rr[HA + R_.m.mem.h_del:].pow(2).sum() / Ron.shape[1]) if SH[0] > HA + R_.m.mem.h_del else 0.0}
    b0 = 0.0; per_n_ok = all(v_ >= 0.5 for v_ in fits[str(kstar)]["iia_ccc_by_n"].values())
    f = {"iia_star": iia_star, "ctrl_max": max(ctrl), "b0": b0, "shuffled": shuf, "supp_pass": supp_pass, "iia_on": out["support"]["iia_on"], "per_n_floor_ok": per_n_ok,
         "necessity_excess": out["necessity"]["resample"]["excess"], "iia_max_all_k": max(v["iia_ccc"] for v in fits.values()), "gfit": gfit, "eligible": True}
    valid = all(out["gates"][g]["pass"] for g in ("DISJ", "EN", "DISRUPT"))
    out["label"] = label(f) if valid else "VOID"; out["label_inputs"] = f; out["b0"] = b0
    if SM["steps"] == 2000:
        np.save(os.path.join(ROOT, "boxes", "runs_fsq", run, "dasused_R_t2_on.npy"), Ron.cpu().numpy().astype(np.float16))
    return out


def ladder(R_, TE, K, ra, ks=(16, 64, 256, 1024), lr=3e-3):
    """Theory 2026-10-14 23:3x (descriptive pilot, rms s1): k-ladder at t = 2, each k warm-started from the previous k's R + PCA directions of S_d - S_r,
    lr 3e-3 (P2's selected), step cap 2,000 for k <= 64 and 5,000 for k >= 256 (early stop on validation IIA, patience 3 x 100 steps). Per k: train IIA
    (1,000 fit pairs), val best, B IIA_ccc (pooled over eligible strata; by n), steps; G-FIT (train IIA non-decreasing in k within 0.05).
    Rule (fixed before output): B IIA_ccc < 0.5 at k = 1024 -> 'no compact used subspace <= 1024-d found (distributed or fit-limited)'; some k >= 0.7 ->
    'compact subspace at k' (controls then run on that k)."""
    t = 2; excl = {tuple(it["syms"]) for it in TE if it["n"] >= int(ra.get("excl_min_n", 1))}; r_sp = np.random.default_rng(17300)
    for _ in range(int(ra.get("serial_pool") or 2200)):
        r__, _t = N.generate(r_sp, 8); excl.add(tuple(N.render(r__)))
    TE_syms = {tuple(it["syms"]) for it in TE}
    for p_ in R_.m.parameters():
        p_.requires_grad_(False)
    rng = np.random.default_rng([20302, t]); B = []
    for it in TE:
        if t + 1 <= it["n"] <= 8:
            pr, _ = make_pair(it, t, rng)
            if pr is not None:
                B.append((it, pr))
    rec = [b[0] for b in B]; don = [b[1][0] for b in B]; hyb = [b[1][2] for b in B]; ycf = np.array([b[1][1] for b in B]); y = np.array([N.value(it["root"]) for it in rec])
    yd = np.array([d["t"] for d in don]); nn = np.array([it["n"] for it in rec])
    ccc = (R_.answers(rec, K) == y) & (R_.answers(don, K) == yd) & (R_.answers(hyb, K) == ycf); ix = np.where(ccc)[0]
    nc = {n_: int((nn[ix] == n_).sum()) for n_ in sorted(set(nn.tolist()))}; strata = [n_ for n_, c_ in nc.items() if c_ >= 100]; ix = np.array([i for i in ix if nn[i] in strata], dtype=int)
    Bi = [rec[i] for i in ix]; BSr, BSd = collect(R_, Bi, t), collect(R_, [don[i] for i in ix], t); Bycf, Bn = ycf[ix], nn[ix]
    A_fit, A_val, raw = build_A(R_, TE_syms, excl, t, K)
    pack = lambda L: ([a[0] for a in L], collect(R_, [a[0] for a in L], t), collect(R_, [a[1] for a in L], t), np.array([a[2] for a in L]))  # noqa: E731
    fit, val = pack(A_fit), pack(A_val); kmax = max(ks)
    M = torch.tensor((fit[2] - fit[1]).reshape(len(fit[0]), -1), device=R_.dev); torch.manual_seed(20381); _, _, Vp = torch.svd_lowrank(M, q=kmax + 16, niter=4); Vp = Vp[:, :kmax].contiguous(); del M
    out = {"t": t, "lr": lr, "n_ccc": int(len(ix)), "strata": strata, "A": {"n_fit": len(A_fit), "n_val": len(A_val), "raw": raw}, "k": {}}; prev = None
    for k in ks:
        init = Vp[:, :k] if prev is None else torch.cat([prev, Vp[:, :k - prev.shape[1]]], 1)
        Rk, st, vb = fit_rot(R_, K, t, fit, val, k, lr, init=init, steps=2000 if k <= 64 else 5000)
        pred = iia_eval(R_, Bi, BSr, BSd, Bycf, t, K, Rk); trn = iia_eval(R_, fit[0][:1000], fit[1][:1000], fit[2][:1000], fit[3][:1000], t, K, Rk)
        out["k"][str(k)] = {"iia_ccc": float((pred == Bycf).mean()), "iia_ccc_by_n": {str(n_): float((pred == Bycf)[Bn == n_].mean()) for n_ in strata if (Bn == n_).any()},
                            "iia_train": float((trn == fit[3][:1000]).mean()), "val_best": vb, "steps": st, "step_cap": 2000 if k <= 64 else 5000,
                            "init": "pca" if prev is None else "k%d + pca pad" % prev.shape[1]}
        print("LADDER k", k, out["k"][str(k)], flush=True); prev = Rk
    tr = [out["k"][str(k)]["iia_train"] for k in ks]; out["G_FIT"] = {"train_by_k": tr, "pass": all(tr[i + 1] >= tr[i] - 0.05 for i in range(len(tr) - 1))}
    best = max(ks, key=lambda k_: out["k"][str(k_)]["iia_ccc"]); bi = out["k"][str(best)]["iia_ccc"]
    out["reading"] = ("compact subspace at k=%d (IIA_ccc %.3f >= 0.7; controls next)" % (best, bi) if bi >= 0.7 else
                      "no compact used subspace <= 1024-d found (distributed or fit-limited)" if out["k"][str(kmax)]["iia_ccc"] < 0.5 else "intermediate (0.5-0.7)")
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
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--device", default="cuda:1"); ap.add_argument("--stage", default="P0", choices=["P0", "P1", "P2", "B", "ladder"])
    ap.add_argument("--bs", type=int, default=250); ap.add_argument("--memcheck", action="store_true"); ap.add_argument("--time_steps", type=int, default=0)
    ap.add_argument("--lr_select", action="store_true", help="P2: choose the lr on this model's validation pairs (rms s1 only)"); ap.add_argument("--fit_micro", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="P2 functional smoke (tiny sizes; written to p2_smoke, never read)")
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
    res.update({"run": a.run, "K": K, "add_norm": ra.get("add_norm") or None, "meta": meta}); rows = {}; touched = set()
    if a.stage == "ladder":                                                 # Theory 2026-10-14 23:3x: rms s1 k-ladder (descriptive pilot)
        tb = time.time(); res["ladder"] = {**ladder(R, TE, K, ra), **meta, "sec": round(time.time() - tb), "done": True}; touched.add("ladder")
        print(a.run, "LADDER", {k_: res["ladder"]["k"][k_]["iia_ccc"] for k_ in res["ladder"]["k"]}, res["ladder"]["reading"], flush=True)
    elif a.stage == "P2":                                                   # pilot-scoped P2 (t = 2)
        if a.smoke:
            SM.update({"steps": 20, "every": 10, "n_fit": 200, "n_val": 60, "n_supp": 400, "n_B": 300})
        tb = time.time(); r2 = p2(R, TE, K, ra, a.run, a.lr_select, a.fit_micro); res["p2_smoke" if a.smoke else "p2"] = {**r2, **meta, "sm": dict(SM), "sec": round(time.time() - tb), "done": not a.smoke}; touched.add("p2_smoke" if a.smoke else "p2")
        print(a.run, "P2 label", r2.get("label"), {k_: r2.get(k_) for k_ in ("best_k", "iia_star", "lr")}, flush=True)
    elif a.stage == "P1":                                                   # draft §7 P1: G-MEM + G-GRAD + --time_steps
        tb = time.time(); res["p1"] = {**p1(R, TE, K, steps=a.time_steps or 50), **meta, "sec": round(time.time() - tb), "done": True}; touched.add("p1"); print(a.run, "P1", res["p1"], flush=True)
    elif a.stage == "P0" and res.get("serial_cond", {}).get("done") and not res.get("p0b", {}).get("done"):   # P0a done: the P0 remainder (draft §7)
        tb = time.time(); res["p0b"] = {**p0b(R, TE, K, rows), **meta, "sec": round(time.time() - tb), "done": True}; touched.add("p0b")
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
        touched.add("serial_cond")
    cur = json.load(open(fp)) if os.path.exists(fp) else {}                # merge-on-write: another stage may have written this file meanwhile
    cur.update({k_: v_ for k_, v_ in res.items() if k_ not in cur or k_ in touched or k_ in ("run", "K", "add_norm", "meta")}); json.dump(cur, open(fp, "w", newline="\n"), indent=1)
    rp = os.path.join(rd, "eval_dasused_rows.npz"); old = dict(np.load(rp)) if os.path.exists(rp) else {}; old.update(rows); np.savez_compressed(rp, **old)
    print(a.run, "P0 done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
