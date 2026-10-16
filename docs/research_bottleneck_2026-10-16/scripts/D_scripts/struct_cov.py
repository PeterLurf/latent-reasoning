"""Structural window coverage (numbers -> 'N'; the value lookup a op b is a 4 x 11 x 11 table, all covered at n = 1), compacted vs raw VB slots,
radius 2..6, with / without global cues; and the parallel-ready pass count (reduce every ready handle per pass). CPU, $0."""
import sys, json
sys.path.insert(0, "D:/Latent")
sys.path.insert(0, sys.path[0] if False else __import__("os").path.dirname(__file__))
import numpy as np
import boxes.nested170 as N
from local_rule_cov import compact_states, raw_states, handle_index, ready_at, is_num


def S(z):
    return ["N" if is_num(s) else s for s in z]


def win(z, c, R):
    return tuple(z[c + d] if 0 <= c + d < len(z) else "#" for d in range(-R, R + 1))


def par_passes(root):
    z = ["EOT"] + N.render(root) + [" ="]; k = 0
    while len(z) > 3:
        idx = [i for i in range(2, len(z) - 2) if ready_at(z, i)]
        out = []; i = 0; idx = set(idx)
        while i < len(z):
            if i + 1 in idx:
                a, op, b = int(z[i]), z[i + 1].strip(), int(z[i + 2]); v = N.apply(op, a, b)
                if out and out[-1].strip() == "(" and i + 3 < len(z) and z[i + 3].strip() == ")":
                    out[-1] = " %d" % v; i += 4
                else:
                    out.append(" %d" % v); i += 3
            else:
                out.append(z[i]); i += 1
        z = out; k += 1
    assert int(z[1]) == N.value(root), (z, N.value(root))
    return k


if __name__ == "__main__":
    rng = np.random.default_rng(31300)
    train = [N.generate(rng, int(rng.integers(1, 9)))[0] for _ in range(60000)]
    test = {n: [N.generate(rng, n)[0] for _ in range(300)] for n in (9, 10, 12, 14, 16)}
    CUES = {"none": lambda t, c, T: (), "t": lambda t, c, T: (t,), "len": lambda t, c, T: (T,)}
    out = {"coverage": {}}
    for kind in ("compact", "raw"):
        f = compact_states if kind == "compact" else raw_states
        trS = [[(S(z), handle_index(z)) for z in f(r)] for r in train]; teS = {n: [[(S(z), handle_index(z)) for z in f(r)] for r in its] for n, its in test.items()}
        for R in ((2, 3, 4, 6) if kind == "compact" else (2, 4, 6, 10)):
            A = {c: set() for c in CUES}; H = {c: set() for c in CUES}
            for St in trS:
                for t, (z, h) in enumerate(St[:-1]):
                    T = len(z)
                    for c in range(T):
                        w = win(z, c, R)
                        for cn, cue in CUES.items():
                            k_ = w + cue(t, c, T); A[cn].add(k_)
                            if c == h:
                                H[cn].add(k_)
            for cn, cue in CUES.items():
                res = {}
                for n, SS in teS.items():
                    tot = cov = ht = hc = 0
                    for St in SS:
                        for t, (z, h) in enumerate(St[:-1]):
                            T = len(z)
                            for c in range(T):
                                k_ = win(z, c, R) + cue(t, c, T); tot += 1; cov += k_ in A[cn]
                                if c == h:
                                    ht += 1; hc += k_ in H[cn]
                    res[n] = {"all": round(cov / tot, 4), "handle": round(hc / ht, 4)}
                out["coverage"][f"{kind}|R{R}|{cn}"] = res; print(kind, R, cn, res, flush=True)
    pp = {}
    for n in (4, 8, 12, 16):
        ks = [par_passes(N.generate(rng, n)[0]) for _ in range(3000)]
        pp[n] = {"mean": round(float(np.mean(ks)), 3), "p90": float(np.percentile(ks, 90)), "max": int(max(ks))}
    out["parallel_ready_passes"] = pp; print("parallel passes", pp, flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
