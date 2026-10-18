"""PROOFS_TS check P1b / P2b (CPU fp32, 2 threads, Idle): the draft's OWN precondition items (rng 23180 stream, first 200 per n, the same
exclusion as ts_label_check.py) at n 11 and 12: reproduce the draft's REROOT keep / kept-wrong with its TF code, and test TF vs KV-cache REROOT
generation on the items TF rejects (the informative EQUIV cases; the fresh 23187 set had almost none at n 9-12). No new items are read.
python -B pc_reroot_draft.py"""
import json
import os
import sys
import time

import pc_reroot as P   # sets threads / Idle priority, imports the draft's ts_label_check read-only

T = P.T; N = P.N; np = P.np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "pc_reroot_draft.json")


def main():
    t0 = time.time()
    excl = {tuple(it["syms"]) for it in N.test_set(1000, n_max=24)}; r_sp = np.random.default_rng(17300)
    for _ in range(2200):
        excl.add(tuple(N.render(N.generate(r_sp, 8)[0])))
    rng = np.random.default_rng(23180); items = {}
    for n in range(9, 21):
        items[n] = []
        while len(items[n]) < 200:
            r_, _ = N.generate(rng, n)
            if tuple(N.render(r_)) in excl:
                continue
            items[n].append(r_)
    D = json.load(open("D:/Latent/docs/research_textself_2026-10-18/scripts/ts_label_check_s1.json"))
    g = T.load("s170_cot_s1"); res = {"by_n": {}}
    for n in (11, 12):
        roots = items[n]; gv = [T.gold_values(r) for r in roots]
        tf = T.label_reroot(g, roots, 25); ge = P.reroot_gen(g, roots)
        rec = D["items"]["%d_reroot" % n]
        same_as_draft = sum(int(tf[i]["keep"]) == rec[i][0] and tf[i]["fail_step"] == rec[i][1] for i in range(200))
        kept = [i for i in range(200) if tf[i]["keep"]]; kw = [i for i in kept if tf[i]["values"] != gv[i]]
        rej = [i for i in range(200) if not tf[i]["keep"]]
        row = {"keep_tf": len(kept) / 200, "kept_wrong_tf": len(kw), "reproduces_draft_items": same_as_draft, "keep_gen": float(np.mean([x["keep"] for x in ge])),
               "agree_keep_and_values": sum((tf[i]["keep"] == ge[i]["keep"]) and (tf[i]["values"] == ge[i]["values"]) for i in range(200)),
               "tf_rejected": len(rej), "tf_rejected_gen_also_rejects": sum(not ge[i]["keep"] for i in rej),
               "tf_rejected_same_fail_step": sum(ge[i]["fail_step"] == tf[i]["fail_step"] for i in rej),
               "rejected_detail": [{"i": i, "L0": len(N.render(roots[i])), "tf_fail": tf[i]["fail_step"], "gen_fail": ge[i]["fail_step"], "gen_min_margin": ge[i]["min_margin"]} for i in rej],
               "kept_wrong_detail": [{"i": i, "values": tf[i]["values"], "gold": gv[i], "gen_values": ge[i]["values"]} for i in kw]}
        res["by_n"][str(n)] = row; print(n, {k: v for k, v in row.items() if not k.endswith("detail")}, flush=True)
        json.dump(res, open(OUT, "w", newline="\n"), indent=1)
    res["sec"] = time.time() - t0; json.dump(res, open(OUT, "w", newline="\n"), indent=1)
    open(os.path.join(HERE, "pc_reroot_draft.done"), "w", newline="\n").write("done\n")


if __name__ == "__main__":
    main()
