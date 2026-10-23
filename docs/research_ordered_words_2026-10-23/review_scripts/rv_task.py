"""REVIEW_OW $0 task checks (CPU, 1 thread). Independent of ow_gen_check.py except for importing the draft generator ow_words.py.
(a) English-falsity (wrap) share of op statements; (b) held-out-frame exposure of test items by d; (c) kept pairs: cf(a) vs cf_b and
their coincidences; (d) per-d kept counts from the default 3000 candidates (subsampled). Writes rv_task.json next to this file.
    python -B docs/research_ordered_words_2026-10-23/review_scripts/rv_task.py"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import json, math, sys  # noqa: E401,E402
import numpy as np  # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import ow_words as OW  # noqa: E402

raw = lambda op, a, b: {"add": a + b, "sub": a - b, "mul": a * b}[OW.BASE[op]]  # noqa: E731


def cf(rec, don, t, mode):
    S = list(don["values"][:t + 1]) if mode == "a" else list(rec["values"][:t]) + [don["values"][t]]
    for k in range(t + 1, rec["d"] + 1):
        op, b, j = rec["ops"][k - 1], rec["operands"][k - 1], rec["src_j"][k - 1]
        S.append(OW.apply(op, S[k - 1], S[j] if op.endswith("_s") else b))
    return S[-1]


def main():
    out = {}
    # (a) wrap share by base op, OW forward train stream frames 0-5
    rng = np.random.default_rng(1); wr = {"add": [0, 0], "sub": [0, 0], "mul": [0, 0]}; item_any = {}
    for d in range(1, 9):
        anyw = []
        for _ in range(2000):
            it = OW.make(rng, d); w_item = False
            for t, (op, b) in enumerate(zip(it["ops"], it["operands"]), 1):
                a = it["values"][t - 1]; r = raw(op, a, b); w = not (0 <= r <= 10); base = OW.BASE[op]
                wr[base][0] += w; wr[base][1] += 1; w_item |= w
            anyw.append(w_item)
        item_any[str(d)] = float(np.mean(anyw))
    out["wrap_share_by_base_op"] = {k: v[0] / v[1] for k, v in wr.items()}
    out["share_items_with_ge1_false_statement"] = item_any
    # (b) held-out frames in the registered test set law (FRAME_ALL)
    s = OW.Stream(OW.SEED_TEST, frames=OW.FRAME_ALL); ho = {}
    for d in range(1, 13):
        its = [s.draw(d) for _ in range(500)]
        ho[str(d)] = {"share_items_with_heldout_frame": float(np.mean([any(f >= 6 for f in it["frames"]) for it in its])),
                      "analytic_1_minus_0.75^d": 1 - 0.75 ** d}
    out["test_heldout_frame_exposure"] = ho
    # (c)/(d) kept pairs, cf(a) vs cf_b (subsample of the pair law: 1200 candidates per d)
    pc = OW.pair_candidates(per_d=1200, ds=range(4, 9)); pr = {}
    for d in range(4, 9):
        t = math.ceil(d / 2); kept = [(r, dn) for r, dn in pc[d] if len({r["answer"], dn["answer"], cf(r, dn, t, "a")}) == 3]
        ca = np.array([cf(r, dn, t, "a") for r, dn in kept]); cb = np.array([cf(r, dn, t, "b") for r, dn in kept])
        own = np.array([r["answer"] for r, _ in kept]); dfin = np.array([dn["answer"] for _, dn in kept])
        # does a post-t step back-reference a j < t (where donor and recipient S_j differ)?
        bref = np.array([any(op.endswith("_s") and j is not None and j < t for op, j in zip(r["ops"][t:], r["src_j"][t:])) for r, _ in kept])
        pr[str(d)] = {"t": t, "kept_share": len(kept) / len(pc[d]), "projected_kept_of_3000": int(round(3000 * len(kept) / len(pc[d]))),
                      "share_post_t_backref_j_lt_t": float(bref.mean()), "share_cfb_ne_cfa": float((cb != ca).mean()),
                      "share_cfb_eq_own": float((cb == own).mean()), "share_cfb_eq_donorfinal": float((cb == dfin).mean()),
                      "share_cfb_distinct_from_own_dfin_cfa": float(((cb != ca) & (cb != own) & (cb != dfin)).mean())}
    out["pairs"] = pr
    with open(os.path.join(HERE, "rv_task.json"), "w", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
