"""Descriptive $0 read (Note 255-2): what do failing d 5-6 answers equal in the position-bound C1 loops? REG TEST2 regenerated (sigs asserted); eval_acc items pred[i][k] = answer at K' k; K' 8 used. REG runs only (rcb_plaw4_*)."""
import sys, json, collections
sys.path.insert(0, "D:/Latent")
import boxes.inject_ow as IO
M = 11
S = IO.build_sets(mul_s_p=0)["test2"]


def apply(op, a, opd, vals, srcj):
    o = opd
    if op.endswith("_c"):
        return {"add": (a + o), "sub": (a - o), "mul": (a * o)}[op[:3]] % M
    return {"add": (a + o), "sub": (a - o), "mul": (a * o)}[op[:3]] % M


import numpy as np
for run in sys.argv[1:]:
    A = json.load(open("D:/Latent/boxes/runs_fsq/%s/eval_acc.json" % run))["items"]
    for d in (5, 6):
        it = A[str(d)]; items = S[d]
        assert [x["sig"] for x in items] == it["sig"], ("sig mismatch", run, d)
        F = [(i, it["pred"][i][8]) for i, x in enumerate(items) if it["pred"][i][8] != int(x["answer"])]
        def share(pairs, which):
            k = 0
            for i, p in pairs:
                v = [int(t) for t in items[i]["values"]]
                k += (p in set(v[1:d])) if which == "earlier" else (p == v[4])
            return k / max(1, len(pairs))
        exp_e = np.mean([len(set(int(t) for t in items[i]["values"][1:d]) - {int(items[i]["answer"])}) / 10 for i, _ in F])   # Audit 255-2: wrong answer uniform over 10 non-answers
        exp_4 = np.mean([0.0 if int(items[i]["values"][4]) == int(items[i]["answer"]) else 0.1 for i, _ in F])
        rng = np.random.default_rng(0); perm = []
        for _ in range(200):                                               # permuted-pred control: failing preds shuffled across failing items (wrong-answer pairs kept)
            ps = rng.permutation([p for _, p in F]); pr = [(i, p) for (i, _), p in zip(F, ps) if p != int(items[i]["answer"])]
            perm.append((share(pr, "earlier"), share(pr, "v4")))
        pe, p4 = np.mean([a for a, _ in perm]), np.mean([b for _, b in perm])
        oe, o4 = share(F, "earlier"), share(F, "v4")
        print(run, "d", d, "fail n", len(F), "earlier-v obs %.3f exp %.3f perm %.3f ratio %.2f | v_4 obs %.3f exp %.3f perm %.3f ratio %.2f"
              % (oe, exp_e, pe, oe / exp_e, o4, exp_4, p4, o4 / max(1e-9, exp_4)))
