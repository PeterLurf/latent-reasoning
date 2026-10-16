"""Vendored reference for the §211 E_t diff gate: render_spans + slot_targets, copied verbatim (definitions only) from the red team's
scratchpad/redteam/slot_targets.py (sha256 ebf25d3bbd30d0bd647b2aaae0419198721d8aec8f2a0e8bd48bcd9776f07be0; its module-level 20k check is omitted)."""
from boxes.nested170 import _copy, _leftmost_ready, apply, PREC  # noqa: F401


def render_spans(nd, parent=None, right=False, off=0):
    """render + for every internal node id(nd) -> (a, b) slot span INCLUDING its own brackets."""
    spans = {}
    if nd.op is None:
        return [" %d" % nd.v], spans
    ls, sl = render_spans(nd.l, nd.op, False, off)
    rs, sr = render_spans(nd.r, nd.op, True, off + len(ls) + 1)
    s = ls + [" " + nd.op] + rs; spans.update(sl); spans.update(sr)
    br = parent is not None and (PREC[nd.op] < PREC[parent] or (right and PREC[nd.op] == PREC[parent]))
    if br:
        s = [" ("] + s + [" )"]
        spans = {k: (a + 1, b + 1) for k, (a, b) in spans.items()}
    spans[id(nd)] = (off, off + len(s) - 1)
    return s, spans

def slot_targets(root):
    """A's slot-aligned E_t for t = 0..n over [EOT, syms, ' =']: classes as strings; VOID = 'V'."""
    nd = _copy(root); s0, spans = render_spans(nd)
    E = [["EOT"] + list(s0) + [" ="]]; cur = list(E[0])
    while nd.op is not None:
        path = _leftmost_ready(nd); tgt = nd
        for p in path:
            tgt = getattr(tgt, p)
        a, b = spans[id(tgt)]; v = apply(tgt.op, tgt.l.v, tgt.r.v)
        # span of tgt at THIS step in original slot coords: alive slots in [a,b] of the ORIGINAL render (spans fixed at t=0)
        cur = list(cur)
        for j in range(a + 1, b + 2):   # +1 for EOT offset
            cur[j] = "V"
        cur[b + 1] = " %d" % v
        tgt.v = v; tgt.op = tgt.l = tgt.r = None
        E.append(cur)
    return E
