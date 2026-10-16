"""D_depth_invariance support script (CPU, $0; reads boxes/nested170 read-only; PYTHONDONTWRITEBYTECODE=1).
(1) LOCAL-RULE: Feng's leftmost-ready reduction == a radius-2 rule in COMPACTED token coordinates + one prefix-OR ("no ready op to my left").
(2) COVERAGE: share of test-state windows (n 9..16, every step) that occur among training-state windows (n 1..8, every step),
    in compacted coordinates vs VB's raw VOID-laden slot coordinates, with and without global cues (t, absolute position, length).
(3) PREFIX: the prefix-factorial construction (prefix subtree P with k ops replacing Q's leading number) reaches Q exactly at step k."""
import sys, json, collections
sys.path.insert(0, "D:/Latent")
import numpy as np
import boxes.nested170 as N

PREC = N.PREC


def is_num(s):
    return s.strip().isdigit()


def ready_at(z, i):
    """z: compacted symbol list incl. frame ['EOT', ..., ' =']; i: index of an operator. Radius-2 readiness."""
    op = z[i].strip()
    if op not in PREC or not (is_num(z[i - 1]) and is_num(z[i + 1])):
        return False
    L, Rt = z[i - 2].strip(), z[i + 2].strip()
    okL = L in ("EOT", "(") or (L in PREC and PREC[L] < PREC[op])
    okR = Rt in ("=", ")") or (Rt in PREC and PREC[Rt] <= PREC[op])
    return okL and okR


def local_step(z):
    """one reduction with the local rule; the 'leftmost' is a prefix-OR over ready flags."""
    flags = [ready_at(z, i) if 2 <= i <= len(z) - 3 else False for i in range(len(z))]
    seen = False
    for i in range(len(z)):
        if flags[i] and not seen:          # ready and no ready op to the left (prefix-OR)
            a, b = int(z[i - 1]), int(z[i + 1]); v = N.apply(z[i].strip(), a, b)
            if z[i - 2].strip() == "(" and z[i + 2].strip() == ")":
                return z[:i - 2] + [" %d" % v] + z[i + 3:]
            return z[:i - 1] + [" %d" % v] + z[i + 2:]
        seen = seen or flags[i]
    return None


def check_local(n_items=100_000, seed=31000, n_max=16):
    rng = np.random.default_rng(seed); bad = 0; steps = 0
    for _ in range(n_items):
        root, y = N.generate(rng, int(rng.integers(1, n_max + 1)))
        st = N.cot_steps(root); z = ["EOT"] + st[0] + [" ="]
        for k in range(1, len(st)):
            z = local_step(z); steps += 1
            if z is None or z[1:-1] != st[k]:
                bad += 1; break
    return {"items": n_items, "steps": steps, "failures": bad}


# ---------------------------------------------------------------- coverage
def compact_states(root):
    return [["EOT"] + s + [" ="] for s in N.cot_steps(root)]


def raw_states(root):
    """VB's slot layout: consumed span -> VOID, the value on the span's last slot (vbank.recode_targets semantics, re-implemented on symbols)."""
    import boxes.vbank as VB
    E = VB.recode_targets(root)
    return [[VB.CLS[c] if c != VB.VOID else "V" for c in e] for e in E]


def handle_index(z, void="V"):
    """index (in z) of the operator of the leftmost ready handle, computed on the alive subsequence; None if done."""
    alive = [i for i, s in enumerate(z) if s != void]; za = [z[i] for i in alive]
    seen = False
    for j in range(len(za)):
        if 2 <= j <= len(za) - 3 and ready_at(za, j):
            return alive[j]
    return None


def win(z, c, R, pad="#"):
    return tuple(z[c + d] if 0 <= c + d < len(z) else pad for d in range(-R, R + 1))


def collect(items, kind, R, cues):
    """-> sets of windows (all slots, handle-centred) with optional global cues appended."""
    allw, hw = set(), set()
    for root in items:
        S = compact_states(root) if kind == "compact" else raw_states(root)
        for t, z in enumerate(S[:-1]):                       # states that still need a step
            T = len(z); h = handle_index(z)
            for c in range(T):
                w = win(z, c, R)
                key = w + tuple(cue(t, c, T) for cue in cues)
                allw.add(key)
                if c == h:
                    hw.add(key)
    return allw, hw


def coverage(train_sets, test_items_by_n, kind, R, cues):
    trA, trH = train_sets
    out = {}
    for n, items in test_items_by_n.items():
        tot = cov = htot = hcov = 0
        for root in items:
            S = compact_states(root) if kind == "compact" else raw_states(root)
            for t, z in enumerate(S[:-1]):
                T = len(z); h = handle_index(z)
                for c in range(T):
                    key = win(z, c, R) + tuple(cue(t, c, T) for cue in cues)
                    tot += 1; cov += key in trA
                    if c == h:
                        htot += 1; hcov += key in trH
        out[n] = {"all": round(cov / tot, 4), "handle": round(hcov / htot, 4)}
    return out


def prefix_check(n_trials=20000, seed=31200):
    """P (k ops) replaces the leading number v of Q (r ops); is cot_steps(full)[k] == render(Q)? and the (k, span) cells."""
    rng = np.random.default_rng(seed); ok = bad = 0; cells = collections.Counter()
    for _ in range(n_trials):
        r = int(rng.integers(2, 4)); k = int(rng.integers(1, 13))
        Q, yq = N.generate(rng, r)
        # find Q's leftmost leaf
        path = []; nd = Q
        while nd.op is not None:
            nd = nd.l; path.append("l")
        v = nd.v
        Pt, yp = N.generate(rng, k)
        # force P's value = v by regenerating with answer v: Alg.1 starts from t ~ U; emulate by retry
        tries = 0
        while N.value(Pt) != v and tries < 200:
            Pt, _ = N.generate(rng, k); tries += 1
        if N.value(Pt) != v:
            continue
        full = N._copy(Q); par = full
        for p_ in path[:-1]:
            par = getattr(par, p_)
        if path:
            setattr(par, path[-1], N._copy(Pt))
        else:
            full = N._copy(Pt)
        st = N.cot_steps(full)
        span = len(st[0]) - len(N.render(Q)) + 1
        if st[k] == N.render(Q):
            ok += 1; cells[(k, span)] += 1
        else:
            bad += 1
    return {"ok": ok, "bad": bad, "cells_k_span": {f"{k},{s}": c for (k, s), c in sorted(cells.items())}}


if __name__ == "__main__":
    out = {}
    out["local_rule"] = check_local()
    print("local rule:", out["local_rule"], flush=True)
    out["prefix"] = prefix_check()
    print("prefix:", {k: v for k, v in out["prefix"].items() if k != "cells_k_span"}, flush=True)
    rng = np.random.default_rng(31100)
    train = [N.generate(rng, int(rng.integers(1, 9)))[0] for _ in range(60000)]
    test = {n: [N.generate(rng, n)[0] for _ in range(300)] for n in (9, 10, 12, 14, 16)}
    CUES = {"none": lambda t, c, T: (), "t": lambda t, c, T: (t,), "abspos": lambda t, c, T: (c,), "len": lambda t, c, T: (T,)}
    out["coverage"] = {}
    for kind in ("compact", "raw"):
        trS = [compact_states(r) if kind == "compact" else raw_states(r) for r in train]
        teS = {n: [compact_states(r) if kind == "compact" else raw_states(r) for r in its] for n, its in test.items()}
        for R in (2, 3, 4):
            A = {c: set() for c in CUES}; H = {c: set() for c in CUES}
            for S in trS:
                for t, z in enumerate(S[:-1]):
                    T = len(z); h = handle_index(z)
                    for c in range(T):
                        w = win(z, c, R)
                        for cn, cue in CUES.items():
                            key = w + cue(t, c, T); A[cn].add(key)
                            if c == h: H[cn].add(key)
            for cn, cue in CUES.items():
                res = {}
                for n, SS in teS.items():
                    tot = cov = ht = hc = 0
                    for S in SS:
                        for t, z in enumerate(S[:-1]):
                            T = len(z); h = handle_index(z)
                            for c in range(T):
                                key = win(z, c, R) + cue(t, c, T); tot += 1; cov += key in A[cn]
                                if c == h: ht += 1; hc += key in H[cn]
                    res[n] = {"all": round(cov / tot, 4), "handle": round(hc / ht, 4)}
                out["coverage"][f"{kind}|R{R}|{cn}"] = res
                print(kind, R, cn, res, flush=True)
    # VOID-gap statistic in raw coordinates: distance (slots) from the handle's operator to its left / right alive operand
    gaps = {}
    for n, its in [(8, train[:20000])] + list(test.items()):
        mx = []; 
        for r in (its if n != 8 else [x for x in its]):
            for z in raw_states(r)[:-1]:
                h = handle_index(z)
                if h is None: continue
                l = h - 1
                while z[l] == "V": l -= 1
                rr = h + 1
                while z[rr] == "V": rr += 1
                mx.append(max(h - l, rr - h) - 1)
        gaps[n] = {"mean_void_gap": round(float(np.mean(mx)), 3), "p99": float(np.percentile(mx, 99)), "max": int(np.max(mx))}
    out["raw_handle_void_gap"] = gaps; print("void gaps", gaps, flush=True)
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "local_rule_cov.json", "w"), indent=1)
