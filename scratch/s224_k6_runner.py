"""§224 K6 helper (gate_s224.py k6): runs TREE's boxes.vbank trainer with far_branch wrapped so L_far and the logged far_ce carry a constant offset
(identical gradients). python s224_k6_runner.py <tree> <offset> <trainer args...>"""
import sys

tree, off = sys.argv[1], float(sys.argv[2]); sys.path.insert(0, tree)
import boxes.vbank as VB  # noqa: E402

orig = VB.far_branch


def far_off(*a, **k):
    Lf, Lp, st, r = orig(*a, **k); st["far_ce"] += off; return Lf + off, Lp, st, r


VB.far_branch = far_off; sys.argv = ["vbank"] + sys.argv[3:]; VB.main()
