"""IIT-SERIAL draft (docs/THEORY_IITSERIAL_DRAFT_2026-10-14.md §3) CHECK on the generator (nested170 Alg. 1, read-only): which share of TRAINING items
(nested170.train_items, n ~ U{1..8}, the run's exclusion set) yields a defined IIT-SERIAL pair, and a defined SIIT pair.
  eligibility: (e1) the root has a bare-number child a (the late leaf); (e2) the first leftmost-ready handle h is not the root; (e3) n >= 3 (then
  height(root) >= 3, checked here). IIT donor (as nested170.serial_items): h's operands re-drawn uniformly among the pairs giving v' != v_1, and a -> a' != a
  (up to 10 redraws on a division by 0). y = own answer, y_hyb = the recipient's tree with h := v' (a kept), y_d = the donor's answer.
  IIT pair kept iff all three are defined and y_hyb not in {y, y_d}. SIIT pair: text x' = x with h's operands re-drawn to v'' != v_1 (a kept); the memory
  is x's own S_1, the target is y; kept iff y(x') is defined and y(x') != y (so a text-reading model is wrong on it). kept_h5: kept IIT pairs with
  height(root) >= 5 (the tau = 2 eligibility rule height(root) >= 2 tau + 1).
-> theory/dasused/iit_hybrid_fraction.json      python theory/dasused/iit_hybrid_fraction.py"""
import ctypes
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "3")
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
except Exception:
    pass
sys.path.insert(0, "D:/Latent")
import numpy as np  # noqa: E402

import boxes.nested170 as N  # noqa: E402


def height(nd):
    return 0 if nd.op is None else 1 + max(height(nd.l), height(nd.r))


def node_at(root, path):
    nd = root
    for p_ in path:
        nd = getattr(nd, p_)
    return nd


def safe_value(nd):
    try:
        return N.value(nd)
    except KeyError:                                                        # INV[0]: a division by 0 somewhere
        return None


def main():
    TE = N.test_set(1000); EXCL = {tuple(it["syms"]) for it in TE}; r_sp = np.random.default_rng(17300)
    for _ in range(2200):
        r__, _t = N.generate(r_sp, 8); EXCL.add(tuple(N.render(r__)))
    items = N.train_items(np.random.default_rng(20501), 100000, 8, EXCL); rng = np.random.default_rng(20502)
    st = {n: {"items": 0, "e1": 0, "e1e2": 0, "elig": 0, "h_root_ge3": 0, "iit_defined": 0, "iit_kept": 0, "cf_eq_own": 0, "cf_eq_donor": 0, "siit_kept": 0, "kept_h5": 0}
          for n in range(1, 9)}
    for it in items:
        root, n = it["root"], it["n"]; s = st[n]; s["items"] += 1
        side = "l" if root.op is not None and root.l.op is None else ("r" if root.op is not None and root.r.op is None else None)
        if side is None:
            continue
        s["e1"] += 1; path = N._leftmost_ready(root)
        if path is None or len(path) == 0:
            continue
        s["e1e2"] += 1
        if n < 3:
            continue
        s["elig"] += 1; s["h_root_ge3"] += height(root) >= 3; y = it["t"]; h = node_at(root, path); v1 = N.apply(h.op, h.l.v, h.r.v); got = None
        for _ in range(10):
            v2 = int(rng.choice([x for x in range(N.P) if x != v1])); prs = N.PAIRS[(h.op, v2)]; a_, b_ = prs[int(rng.integers(0, len(prs)))]
            hb = N._copy(root); hh = node_at(hb, path); hh.l.v, hh.r.v = a_, b_
            d = N._copy(root); dh = node_at(d, path); dh.l.v, dh.r.v = a_, b_; bare = getattr(d, side); bare.v = int(rng.choice([x for x in range(N.P) if x != bare.v]))
            yh, yd = safe_value(hb), safe_value(d)
            if yh is not None and yd is not None:
                got = (yh, yd); break
        if got is not None:
            yh, yd = got; s["iit_defined"] += 1; s["cf_eq_own"] += yh == y; s["cf_eq_donor"] += yh == yd; s["iit_kept"] += (yh != y and yh != yd); s["kept_h5"] += (yh != y and yh != yd and height(root) >= 5)   # tau = 2 rule: height(root) >= 2 tau + 1
        for _ in range(10):                                                 # SIIT: resample h's operands in the TEXT only (memory = x's own S_1, target y)
            v3 = int(rng.choice([x for x in range(N.P) if x != v1])); prs = N.PAIRS[(h.op, v3)]; a_, b_ = prs[int(rng.integers(0, len(prs)))]
            xp = N._copy(root); xh = node_at(xp, path); xh.l.v, xh.r.v = a_, b_; yp = safe_value(xp)
            if yp is not None:
                s["siit_kept"] += yp != y; break
    tot = {k_: sum(st[n][k_] for n in st) for k_ in st[1]}
    out = {"def": __doc__.split("->")[0].strip(), "n_items": len(items), "by_n": st, "total": tot,
           "share_of_training_items": {"e1_root_has_leaf_child": tot["e1"] / tot["items"], "eligible_e1e2e3": tot["elig"] / tot["items"],
                                       "iit_kept": tot["iit_kept"] / tot["items"], "siit_kept": tot["siit_kept"] / tot["items"], "iit_kept_tau2_height5": tot["kept_h5"] / tot["items"]},
           "among_eligible": {"iit_defined": tot["iit_defined"] / tot["elig"], "iit_kept": tot["iit_kept"] / tot["elig"], "siit_kept": tot["siit_kept"] / tot["elig"],
                              "height_root_ge3": tot["h_root_ge3"] / tot["elig"]},
           "among_defined": {"cf_eq_own": tot["cf_eq_own"] / tot["iit_defined"], "cf_eq_donor": tot["cf_eq_donor"] / tot["iit_defined"],
                             "cf_ne_own_and_ne_donor": tot["iit_kept"] / tot["iit_defined"]},
           "kept_per_batch_of_64": 64 * tot["iit_kept"] / tot["items"]}
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "iit_hybrid_fraction.json"), "w", newline="\n"), indent=1)
    print(json.dumps({k_: v_ for k_, v_ in out.items() if k_ not in ("by_n", "def")}, indent=1))
    for n in range(1, 9):
        s = st[n]; print(n, s["items"], "elig %.3f" % (s["elig"] / s["items"]), "kept %.3f" % (s["iit_kept"] / s["items"]), "siit %.3f" % (s["siit_kept"] / s["items"]))


if __name__ == "__main__":
    main()
