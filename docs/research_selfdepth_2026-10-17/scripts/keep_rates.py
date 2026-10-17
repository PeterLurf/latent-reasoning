"""SELF-DEPTH-CHAIN check 1 ($0, CPU, reads two JSONs): the label-free keep filter KEEP = parse at every step AND steps == n
(Theory/Audit) implies keep = 1 - fail - share(steps - n != 0). Compare with strict precision (label == y AND steps == n):
kept-but-wrong = keep - strict. Post-filter precision = strict / keep."""
import json, sys
out = {}
for cell in ("s213_d3t_s1", "s213_d3f_s1"):
    d = json.load(open("D:/Latent/boxes/runs_fsq/%s/eval_chainlabel.json" % cell))
    rows = {}
    for n, r in d["by_n"].items():
        h = r["steps_minus_n"]; items = r["items"]
        keep_cnt = h.get("0", 0)
        keep = keep_cnt / items; strict = r["precision_strict"]
        rows[n] = {"items": items, "fail": r["fail"], "steps_ne_n": sum(v for k, v in h.items() if k not in ("0", "fail")) / items,
                   "keep": keep, "strict": strict, "kept_but_wrong": keep - strict, "post_filter_precision": strict / keep if keep else None,
                   "rvws": r["right_value_wrong_steps"], "chain_exact": r["chain_exact"], "latent_acc": r["latent_acc"], "hist": h}
    out[cell] = {"card": d.get("card"), "eval_bs": d.get("eval_bs"), "by_n": rows}
    print(cell, d.get("card"))
    print(" n  keep   strict  kept_wrong post_prec chain_exact fail  steps!=n  hist")
    for n, v in rows.items():
        print("%2s %.3f  %.3f  %+.4f   %.4f   %.3f   %.3f  %.3f  %s" % (n, v["keep"], v["strict"], v["kept_but_wrong"], v["post_filter_precision"], v["chain_exact"], v["fail"], v["steps_ne_n"], v["hist"]))
json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1)
