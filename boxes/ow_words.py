"""ORDERED-WORDS (OW) generator, §226 stage 0 (registered 09368b14; moved here from docs/research_ordered_words_2026-10-23/ow_words.py, the prototype of
the $0 generator check, unchanged in behaviour for dag=True). §226 additions (Theory 10-23): `dag` (True = v1.2's DAG: add_s regular op + the forced
mul_s; False = a strict previous-state chain: no add_s, no forced mul_s, the regular op drawn from OPS_CHAIN) and the item keys "dag", "wrap_final"
(the final step's unreduced value left 0..10, i.e. the mod-11 reduction changed it) and "wrap_any" (any step did); computed from the drawn values, no
extra rng draws, so dag=True items are identical to the prototype's. Imports boxes/nested_words.py (NESTED-WORDS v1.2) and never edits it.
Exactly two changes vs v1.2 (Audit 10-23):
  (1) VALUES mod 11: every quantity is in Z_11 = 0..10 and every op is reduced mod 11 (no range rejection). Constants / givens of add/sub are
      1..10, of mul 2..10 (nonzero, not x1); S_0 is uniform 0..10; distractor values 1..10 (v1.2: 1..99). The forced cross-spine product mul_s
      keeps v1.2's rule (both factors >= 2, i.e. not 0 or 1; partner j computed, 1 <= j <= t-2). A mul with S_{t-1} = 0 is redrawn (as v1.2).
  (2) ORDER: order_mode "forward" (default) = S_0 first, then for t = 1..d: [G_t if the step reads a given] S_t; each distractor at a uniform
      random slot (distractors read nothing, so any slot is a dependency order). order_mode "topo" = v1.2's random linear extension
      (NW.topo_shuffle), for the VAL-only descriptive cell and the stage-2 axis.
Everything else is v1.2's own objects BY REFERENCE: FRAMES, NAMES, UNITS, GIVEN, QUESTION, FRAME_TRAIN / FRAME_ALL, OPS / OPS_REG / BASE,
_statements, _finish (rendering and R_t). Added item keys: "mod", "order_mode", "R_step" (the step sentences "<S_t> has <v_t> <u>." for
t = 1..d; R_step[-1] == R[d]) and "cot" (" ".join(R_step))."""
import hashlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import boxes.nested_words as NW  # noqa: E402

P = 11
FRAMES, NAMES, UNITS, GIVEN, QUESTION = NW.FRAMES, NW.NAMES, NW.UNITS, NW.GIVEN, NW.QUESTION
FRAME_TRAIN, FRAME_ALL, OPS, OPS_REG, BASE = NW.FRAME_TRAIN, NW.FRAME_ALL, NW.OPS, NW.OPS_REG, NW.BASE
OPS_CHAIN = tuple(op for op in OPS_REG if op != "add_s")                  # §226 dag=False: every step reads only S_{t-1} (+ a constant / given)
SEED_TEST, SEED_PAIRS, SEED_TRAIN = 22900, 22910, 22930      # unused elsewhere in boxes/*.py (grep 2026-10-23)


_NW_APPLY = NW.apply                                            # bound once: solve() swaps NW.apply for the call


def apply(op, a, b):
    return _NW_APPLY(op, a, b) % P


def _operand(rng, op):
    return int(rng.integers(2, P)) if BASE[op] == "mul" else int(rng.integers(1, P))


def _draw_program(rng, d, dag=True):
    """v1.2's _draw_program with values in Z_11 (no range rejection); dag=False: no forced mul_s (no ts draw) and regular ops from OPS_CHAIN.
    -> ops, bs, js, s, ts, restarts, raw (the unreduced value of each step)."""
    restarts = 0; reg = OPS_REG if dag else OPS_CHAIN
    while True:
        ts = int(rng.integers(3, d + 1)) if (dag and d >= 3) else None
        s = [int(rng.integers(0, P))]; ops, bs, js, raw = [], [], [], []; ok = True
        for t in range(1, d + 1):
            if t == ts:
                cand = [j for j in range(1, t - 1) if s[j] >= 2 and s[t - 1] >= 2]
                if not cand:
                    ok = False; break
                j = cand[int(rng.integers(0, len(cand)))]; ops.append("mul_s"); bs.append(s[j]); js.append(j); raw.append(_NW_APPLY("mul_s", s[t - 1], s[j])); s.append(apply("mul_s", s[t - 1], s[j])); continue
            for _ in range(50):
                op = "mul_v" if d == 1 else reg[int(rng.integers(0, len(reg)))]
                if op == "add_s":
                    if t < 2:
                        continue
                    j = int(rng.integers(0, t - 1)); b = s[j]
                else:
                    j = None; b = _operand(rng, op)
                if BASE[op] == "mul" and s[t - 1] == 0:
                    continue
                break
            else:
                ok = False; break
            ops.append(op); bs.append(b); js.append(j); raw.append(_NW_APPLY(op, s[t - 1], b)); s.append(apply(op, s[t - 1], b))
        if ok:
            return ops, bs, js, s, ts, restarts, raw
        restarts += 1


def forward_order(rng, ops, n_dis):
    seq = ["S0"]
    for t, op in enumerate(ops, 1):
        if op.endswith("_v"):
            seq.append("G%d" % t)
        seq.append("S%d" % t)
    for k in range(n_dis):                                                  # sequential uniform-slot insertion = a uniform interleaving
        seq.insert(int(rng.integers(0, len(seq) + 1)), "D%d" % k)
    return seq


def _decorate(it, order_mode, raw=None, dag=True):
    sp, s, u = it["names"]["spine"], it["values"], it["unit"]
    if raw is not None:                                                     # §226: the wrap flags (no rng)
        it["wrap_final"] = bool(raw[-1] != s[-1]); it["wrap_any"] = bool(any(r != v for r, v in zip(raw, s[1:])))
    it["dag"] = dag
    it["R_step"] = [GIVEN.format(A=sp[t], v=s[t], u=u) for t in range(1, it["d"] + 1)]; it["cot"] = " ".join(it["R_step"])
    it["mod"] = P; it["order_mode"] = order_mode; return it


def make(rng, d, frames=FRAME_TRAIN, stats=None, order_mode="forward", dag=True):
    """v1.2's make() draw sequence (program, n_dis, names, unit, frames, distractor values, order) with the two changes (+ §226 dag)."""
    ops, bs, js, s, ts, rs, raw = _draw_program(rng, d, dag)
    if stats is not None:
        stats["restarts"] = stats.get("restarts", 0) + rs; stats["items"] = stats.get("items", 0) + 1
    n_v = sum(op.endswith("_v") for op in ops); n_dis = int(rng.integers(0, 3))
    pick = rng.choice(len(NAMES), size=d + 1 + n_v + n_dis, replace=False); nm = [NAMES[i] for i in pick]
    spine = nm[:d + 1]; gnames = nm[d + 1:d + 1 + n_v]; dnames = nm[d + 1 + n_v:]; u = UNITS[int(rng.integers(0, len(UNITS)))]
    fr = [int(frames[int(rng.integers(0, len(frames)))]) for _ in ops]
    dis_vals = [int(rng.integers(1, P)) for _ in dnames]
    if order_mode == "forward":
        order = forward_order(rng, ops, n_dis)
    else:
        st, _ = NW._statements(ops, bs, js, s, spine, gnames, dnames, fr, dis_vals); order = NW.topo_shuffle(rng, st)
    return _decorate(NW._finish(d, ops, bs, js, s, ts, spine, gnames, dnames, u, fr, dis_vals, order, n_dis), order_mode, raw, dag)


def redraw(rng, it, tries=1000):
    """v1.2's redraw (a donor on the recipient's skeleton, same order and token length) with values in Z_11."""
    ops, js, d = it["ops"], it["src_j"], it["d"]
    for _ in range(tries):
        s = [int(rng.integers(0, P))]; bs = []; raw = []; ok = True
        for t, (op, j) in enumerate(zip(ops, js), 1):
            if op.endswith("_s"):
                b = s[j]
                if op == "mul_s" and (s[j] < 2 or s[t - 1] < 2):
                    ok = False; break
            else:
                b = _operand(rng, op)
            if BASE[op] == "mul" and s[t - 1] == 0:
                ok = False; break
            bs.append(b); raw.append(_NW_APPLY(op, s[t - 1], b)); s.append(apply(op, s[t - 1], b))
        if ok:
            dv = [int(rng.integers(1, P)) for _ in it["names"]["distractors"]]
            return _decorate(NW._finish(d, ops, bs, js, s, it["mul_s_t"], it["names"]["spine"], it["names"]["givens"], it["names"]["distractors"], it["unit"],
                                        it["frames"], dv, it["order"], it["n_distractors"]), it["order_mode"], raw, it.get("dag", True))
    return None


class Stream:
    def __init__(self, seed, excl=(), frames=FRAME_TRAIN, order_mode="forward", dag=True):
        self.rng = np.random.default_rng(seed); self.excl = set(excl); self.frames = frames; self.order_mode = order_mode; self.dag = dag; self.stats = {}

    def draw(self, d):
        while True:
            it = make(self.rng, d, self.frames, self.stats, self.order_mode, self.dag)
            if it["sig"] not in self.excl:
                return it


def test_set(per_d=1000, ds=range(1, 13), seed=SEED_TEST, order_mode="forward", dag=True):
    s = Stream(seed, frames=FRAME_ALL, order_mode=order_mode, dag=dag); return {d: [s.draw(d) for _ in range(per_d)] for d in ds}


def pair_candidates(per_d=3000, ds=range(1, 13), seed=SEED_PAIRS, order_mode="forward", dag=True):
    r = np.random.default_rng(seed); s = Stream(seed + 1, frames=FRAME_ALL, order_mode=order_mode, dag=dag); out = {}
    for d in ds:
        ps = []
        while len(ps) < per_d:
            rec = s.draw(d); don = redraw(r, rec)
            if don is not None and don["sig"] != rec["sig"]:
                ps.append((rec, don))
        out[d] = ps
    return out


def solve(text):
    """NW.solve (the independent regex solver) with its module-level apply swapped for the mod-11 one for the call."""
    old = NW.apply; NW.apply = apply
    try:
        return NW.solve(text)
    finally:
        NW.apply = old


def set_sha(items):
    return hashlib.sha256("\n".join(x["text"] for x in items).encode()).hexdigest()
