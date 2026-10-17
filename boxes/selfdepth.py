"""§215 SELF-DEPTH-CHAIN round 1 (registered c995604b; SELFDEPTH_v1.md at 5ded8703): the label-free chain labeller, the compose checks and the
self-pool record format, shared by the pool builder (scratch/s215_build_pool.py), GATE-1 (scratch/s215_gate1.py) and the trainer's loader.
  chain (213-33 / part_chainlabel_fresh's loop): cur <- input; repeat: x = [EOT] + render(cur) + [' =']; rollout K = 1; dec = argmax R(V_1); s =
    rerender(dec); cur <- _parse_tree(s); stop at no operator or the cap n + 4. The model's own decode only; no simulator call.
  per-step checks (docs/research_selfdepth_2026-10-17/scripts/compose.py check_step, copied): FRAME, SPAN, ORDER (the parsed input's leftmost-ready
    handle, shape only), PARSE, CANON. KEEP = PARSE at every step AND exactly n steps (n = # operator tokens of the input). CV = KEEP AND FRAME / SPAN /
    ORDER / CANON at every step. Kept = KEEP AND CV.
  record (pool): keys exactly {syms, n, chain_label, steps}; steps = per step [j_lo, j_hi, j_val, value] in the ORIGINAL frame (j_lo / j_hi the first /
    last VOIDed original slot of the span, j_val the value slot, value 0..P-1). No gold field.
  E_self (compose_E): E_0 = x's classes; step k VOIDs the live slots of E_{k-1} inside [j_lo, j_hi] and writes VAL0 + value at j_val; HOLD after n,
    padded to Kb. Under CV this equals vbank.recode_targets(gold root) at every t iff every step value is correct (PROOFS_SELFDEPTH §7)."""
import hashlib
import json

import numpy as np
import torch

import boxes.nested170 as N
import boxes.vbank as VB

EOT_C, EQ_C = VB.CID["EOT"], VB.CID[" ="]
VALC = set(range(VB.VAL0, VB.VAL0 + N.P))
REC_KEYS = {"syms", "n", "chain_label", "steps"}
GOLD_KEYS = {"root", "t", "y", "gold", "E", "recode", "E_self"}


def _parse_tree(syms):
    from boxes.vbank_diag import _parse_tree as pt
    return pt(syms)


def x_classes(syms):
    return [EOT_C] + [VB.CID[s] for s in syms] + [EQ_C]


def n_of(syms):
    return sum(1 for s in syms if s.strip() in N.PREC)


def handle_slots(tree):
    """slot indices (1-based, frame [EOT] + render + [' =']) owned by the leftmost-ready handle of `tree` (operands, operator, own brackets)."""
    own = VB._render_owned(tree); path = N._leftmost_ready(tree); h = tree
    for p_ in path:
        h = getattr(h, p_)
    return [k + 1 for k, (_, o) in enumerate(own) if o is h or o is h.l or o is h.r]


def check_step(cur_tree, dec):
    """compose.py check_step: -> (flags, next_tree or None, (lo, hi) of the span in the CURRENT frame or None)."""
    syms = N.render(cur_tree); x = x_classes(syms); T = len(x); f = {}
    f["FRAME"] = len(dec) == T and dec[0] == EOT_C and dec[-1] == EQ_C
    voids = [j for j in range(1, T - 1) if dec[j] == VB.VOID]
    other_changed = [j for j in range(1, T - 1) if dec[j] != VB.VOID and dec[j] != x[j]]
    if voids:
        lo, hi = voids[0], voids[-1] + 1
        f["SPAN"] = voids == list(range(lo, hi)) and len(voids) >= 2 and hi <= T - 2 and dec[hi] in VALC and all(j == hi for j in other_changed)
    else:
        lo = hi = None; f["SPAN"] = False
    f["ORDER"] = bool(f["SPAN"]) and list(range(lo, hi + 1)) == handle_slots(cur_tree)
    s_ = VB.rerender(dec); t_ = _parse_tree(s_)
    f["PARSE"] = t_ is not None
    f["CANON"] = f["PARSE"] and N.render(t_) == s_
    return f, t_, ((lo, hi) if f["SPAN"] else None)


@torch.no_grad()
def label_chains(m, syms_list, dev, bs=32):
    """the batched chain labeller over a list of input syms (length-grouped batches, as part_chainlabel_fresh). -> per input: {n, keep, cv, label,
    steps_n, steps (original-frame records, only when kept)}."""
    S = len(syms_list); n0 = [n_of(s_) for s_ in syms_list]; cur = [_parse_tree(s_) for s_ in syms_list]
    assert all(c is not None for c in cur), "an input does not parse"
    live = [[j for j in range(1, len(s_) + 1)] for s_ in syms_list]          # original-frame indices of the live interior slots, in order
    flags = [{k: True for k in ("FRAME", "SPAN", "ORDER", "PARSE", "CANON")} for _ in range(S)]; nst = [0] * S; recs = [[] for _ in range(S)]
    active = set(i for i in range(S) if cur[i].op is not None)
    for _ in range(max(n0) + 4):
        act = [i for i in active if nst[i] < n0[i] + 4]
        if not act:
            break
        act.sort(key=lambda i: len(N.render(cur[i]))); nxt = set()
        for c0 in range(0, len(act), bs):
            part = act[c0:c0 + bs]; its = [{"syms": N.render(cur[i])} for i in part]; x, pad, lens = VB.batch_x(its, dev)
            rr = m.rollout(x, pad, 1); dec = m.rec_logits(rr["V"][1]).argmax(-1).cpu().numpy(); del rr
            for k_, i in enumerate(part):
                d = [int(v) for v in dec[k_, :int(lens[k_])]]; f, t_, sp = check_step(cur[i], d); nst[i] += 1
                for kk in flags[i]:
                    flags[i][kk] = flags[i][kk] and f[kk]
                if not f["PARSE"]:
                    continue                                                # FAIL (chain stops)
                if sp is not None and f["FRAME"] and len(live[i]) == len(d) - 2:
                    lo, hi = sp; L = live[i]
                    recs[i].append([L[lo - 1], L[hi - 2], L[hi - 1], d[hi] - VB.VAL0])
                    live[i] = L[:lo - 1] + [L[hi - 1]] + L[hi:]            # the span's slots go VOID except the value slot
                else:
                    flags[i]["SPAN"] = False
                cur[i] = t_
                if t_.op is not None:
                    nxt.add(i)
        active = nxt
    out = []
    for i in range(S):
        fin = cur[i] is not None and cur[i].op is None
        keep = flags[i]["PARSE"] and fin and nst[i] == n0[i]
        cv = keep and all(flags[i][k] for k in ("FRAME", "SPAN", "ORDER", "CANON"))
        out.append({"n": n0[i], "keep": bool(keep), "cv": bool(cv), "label": int(cur[i].v) if (keep and fin) else None, "steps_n": nst[i],
                    "steps": recs[i] if cv else None})
    return out


def compose_E(syms, steps, Kb):
    """E_self over the original frame from a record's steps: (Kb + 1, T) class ids, HOLD after n (padded to Kb)."""
    E = [x_classes(syms)]
    for j_lo, j_hi, j_val, v in steps:
        e = list(E[-1])
        for j in range(j_lo, j_hi + 1):
            if e[j] != VB.VOID:
                e[j] = VB.VOID
        e[j_val] = VB.VAL0 + int(v); E.append(e)
    while len(E) < Kb + 1:
        E.append(list(E[-1]))
    return E[:Kb + 1]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_pool(path, meta_path):
    """the trainer's self-pool loader: sha256 == pool_meta.json's; every record's key set == REC_KEYS (no gold field); n recomputed from syms."""
    meta = json.load(open(meta_path)); sha = file_sha256(path); assert sha == meta["sha256"], ("pool sha mismatch", sha, meta["sha256"])
    recs = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln); assert set(r) == REC_KEYS, ("pool record keys", sorted(r)); assert not (set(r) & GOLD_KEYS)
            assert n_of(r["syms"]) == r["n"], "pool record n != #operators(syms)"
            recs.append(r)
    return recs, meta


def self_item(rec, Kb):
    """a pool record -> a vb_loss item: syms, n, t = chain_label (SD-LABEL-MAP), E_self (HOLD-padded to Kb). No root."""
    it = {"syms": rec["syms"], "n": rec["n"]}
    it["t"] = rec["chain_label"]  # SD-LABEL-MAP
    it["E_self"] = compose_E(rec["syms"], rec["steps"], Kb)
    return it


def _np_seed_check():
    return np.random.default_rng(0).integers(0, 10)
