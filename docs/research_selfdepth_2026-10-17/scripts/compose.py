"""SELF-DEPTH-CHAIN check 4 ($0, CPU): the label-free composition of a chain's per-step fresh decodes into slot-aligned R2 targets E_self
over the ORIGINAL frame (the reference for the trainer's new code), plus the keep / compose-validity checks.
  chain step k: input x_k = [EOT] + render(cur_k) + [' ='] (T_k slots); dec_k = argmax R(V_1) over those slots (class ids).
  checks (all label-free; no gold value, no arithmetic):
    FRAME : dec_k[0] == EOT and dec_k[-1] == ' ='
    SPAN  : the slots dec_k marks VOID form one contiguous run j_lo..j_hi-1 (>= 2 slots), slot j_hi holds a value class, every other slot is unchanged
    ORDER : x_k[j_lo..j_hi] is exactly the leftmost-ready handle of the PARSED INPUT tree (its operands, operator and own brackets) - a convention
            computed from the input alone (which node is reduced), not its value
    CANON : rerender(dec_k) == render(parse(rerender(dec_k)))   (the next input is the decode itself, re-rendered without change)
    PARSE : parse(rerender(dec_k)) is one valid expression       (213-33's chain condition)
    STEPS : the chain ends (no operator) after exactly n steps     (Theory / Audit KEEP; n = #operator tokens of the input)
  KEEP (Theory, primary) = PARSE at every step AND STEPS. COMPOSE-VALID = KEEP AND FRAME, SPAN, ORDER, CANON at every step.
  E_self: E_0 = classes of the original x; E_k = E_{k-1} with its live slots (non-VOID, frame excluded; in order = the tokens of render(cur_{k-1}))
  overwritten by dec_k's interior slots (1..T_k-2). HOLD past n.
Gate (as vbank.gate_rerender): with 'perfect' decodes (the canonical one-step target of each re-rendered state), E_self == vbank.recode_targets(root)
on every t, every item; fixtures show which check catches which corruption."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "D:/Latent")
import numpy as np
import boxes.nested170 as N
import boxes.vbank as VB
from boxes.vbank_diag import _parse_tree, n_ops

EOT_C, EQ_C = VB.CID["EOT"], VB.CID[" ="]
VALC = set(range(VB.VAL0, VB.VAL0 + N.P))


def x_classes(syms):
    return [EOT_C] + [VB.CID[s] for s in syms] + [EQ_C]


def handle_slots(tree):
    """slot indices (1-based, in the frame [EOT] + render + [' =']) owned by the leftmost-ready handle of `tree` (operands, operator, own brackets)."""
    own = VB._render_owned(tree); path = N._leftmost_ready(tree); h = tree
    for p_ in path:
        h = getattr(h, p_)
    return [k + 1 for k, (_, o) in enumerate(own) if o is h or o is h.l or o is h.r]


def check_step(cur_tree, dec):
    """-> (ok_flags dict, next_tree or None)."""
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
    f["ONE_OP"] = f["PARSE"] and n_ops(t_) == n_ops(cur_tree) - 1
    return f, t_


def chain_compose(syms0, decoder, cap=24):
    """decoder(cur_tree) -> dec class list over x(cur). -> dict(keep, compose_valid, label, steps, E_self or None, flags)."""
    root = _parse_tree(syms0); n = sum(1 for s in syms0 if s.strip() in N.PREC)
    E = [x_classes(syms0)]; cur = root; steps = 0; allf = {k: True for k in ("FRAME", "SPAN", "ORDER", "PARSE", "CANON", "ONE_OP")}; label = None
    while cur.op is not None and steps < cap:
        dec = decoder(cur); f, nxt = check_step(cur, dec); steps += 1
        for k in allf:
            allf[k] = allf[k] and f[k]
        if not f["PARSE"]:
            break
        if f["FRAME"] and f["SPAN"]:                                         # compose onto the original frame's live slots
            live = [j for j in range(1, len(E[-1]) - 1) if E[-1][j] != VB.VOID]
            if len(live) == len(dec) - 2:
                e = list(E[-1])
                for k_, j in enumerate(live):
                    e[j] = dec[k_ + 1]
                E.append(e)
            else:
                allf["SPAN"] = False
        cur = nxt
        if cur.op is None:
            label = cur.v
    keep = allf["PARSE"] and cur is not None and cur.op is None and steps == n
    cv = keep and all(allf[k] for k in ("FRAME", "SPAN", "ORDER", "CANON"))
    return {"n": n, "keep": keep, "compose_valid": cv, "label": label if keep else None, "steps": steps, "E_self": E if cv else None, "flags": allf}


def perfect(cur):
    """the canonical one-step target of the current (re-rendered) state (fixtures: a '/ 0' handle, reachable only after a corrupted value,
    evaluates to 0 via N.INV[0] = 0, set after the gate; the generator's PAIRS table is built at import and is unaffected)."""
    return VB.recode_targets(cur, 1)[1]


def corrupt(kind, rng):
    def dec_fn(cur):
        d = list(perfect(cur))
        if rng.random() < 0.5:                                              # corrupt half of the steps
            return d
        T = len(d); voids = [j for j in range(1, T - 1) if d[j] == VB.VOID]; vj = voids[-1] + 1
        if kind == "value_first":                                           # value on the span's first slot, VOID on the last
            d[voids[0]], d[vj] = d[vj], VB.VOID
        elif kind == "wrong_value":
            d[vj] = VB.VAL0 + (d[vj] - VB.VAL0 + 1) % N.P
        elif kind == "frame":
            d[0] = VB.VOID
        elif kind == "keep_op":                                             # no reduction: copy the input (a 0-op step)
            d = x_classes(N.render(cur))
        return d
    return dec_fn


def main(n_items=100000, seed=21950):
    rng = np.random.default_rng(seed); bad = 0; tot_steps = 0
    for i in range(n_items):
        n = int(rng.integers(1, 17)); root, y = N.generate(rng, n); syms = N.render(root)
        r = chain_compose(syms, perfect)
        ref = VB.recode_targets(root)
        ok = r["keep"] and r["compose_valid"] and r["label"] == y and r["E_self"] == ref and r["n"] == n
        bad += not ok; tot_steps += r["steps"]
    out = {"gate": {"items": n_items, "steps": tot_steps, "failures": bad, "pass": bad == 0}}
    print("GATE", out["gate"], flush=True)
    fx = {}; N.INV[0] = 0
    for kind in ("value_first", "wrong_value", "frame", "keep_op"):
        rng2 = np.random.default_rng(seed + 1); c = {"items": 0, "keep": 0, "compose_valid": 0, "label_wrong_kept": 0, "E_wrong_valid": 0}
        for i in range(3000):
            n = int(rng2.integers(2, 13)); root, y = N.generate(rng2, n); r = chain_compose(N.render(root), corrupt(kind, rng2))
            c["items"] += 1; c["keep"] += r["keep"]; c["compose_valid"] += r["compose_valid"]; c["label_wrong_kept"] += bool(r["keep"] and r["label"] != y)
            c["E_wrong_valid"] += bool(r["compose_valid"] and r["E_self"] != VB.recode_targets(root))
        fx[kind] = c; print("FIXTURE", kind, c, flush=True)
    out["fixtures"] = fx
    json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[2]) if len(sys.argv) > 2 else 100000)
