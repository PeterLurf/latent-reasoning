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


for run in sys.argv[1:]:
    A = json.load(open("D:/Latent/boxes/runs_fsq/%s/eval_acc.json" % run))["items"]
    for d in (5, 6):
        it = A[str(d)]; items = S[d]
        assert [x["sig"] for x in items] == it["sig"], ("sig mismatch", run, d)
        c = collections.Counter(); n_fail = 0
        for i, x in enumerate(items):
            p, y = it["pred"][i][8], int(x["answer"])
            if p == y:
                continue
            n_fail += 1; v = [int(t) for t in x["values"]]
            tags = []
            if p in v[1:d]:
                tags.append("earlier v_t (t<d)")
                if p == v[d - 1]:
                    tags.append("= v_{d-1} (stopped one short)")
                if p == v[4] and d > 4:
                    tags.append("= v_4 (last trained depth)")
            c.update(tags or ["other"])
        print(run, "d", d, "fail n", n_fail, {k: round(v_ / max(1, n_fail), 3) for k, v_ in c.items()})
