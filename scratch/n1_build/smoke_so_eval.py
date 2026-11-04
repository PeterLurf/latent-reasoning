"""§271 N1 build: CPU smoke of boxes/inject_so_eval.py (G5 eval-part leg). Fresh models only (no run .pt is loaded: the local RAM rule):
(a) a fresh InjectAW (build_aw, GPT-2 pretrained trunk + fresh DeltaMem, gamma 1, think_k 8, cue index); (b) a direct model (the GPT-2 trunk).
Every part on N_SMOKE (default 8) items per cell; outputs in scratch/_tmp/n1_smoke/{inject,direct}/; checks every documented field path exists and every
proportion lies in [0, 1] (or is null); plus the ast check that every part_* function is decorated with torch.no_grad.
    python -B scratch/n1_build/smoke_so_eval.py [N]"""
import argparse
import ast
import json
import os
import sys
import time

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
os.environ["N_SMOKE"] = str(N)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import torch  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.inject_aw as AW  # noqa: E402
import boxes.inject_so_eval as E  # noqa: E402

BBH = os.environ.get("SO_BBH_DIR", "D:/Latent/scratch/n1_build/bbh")
OUT = os.path.join(ROOT, "scratch", "_tmp", "n1_smoke")


def ast_no_grad():
    tree = ast.parse(open(E.__file__, encoding="utf-8").read()); bad = []; n = 0
    for nd in tree.body:
        if isinstance(nd, ast.FunctionDef) and nd.name.startswith("part_"):
            n += 1; ok = any(isinstance(d, ast.Call) and ast.unparse(d.func) == "torch.no_grad" for d in nd.decorator_list)
            if not ok:
                bad.append(nd.name)
    for nm in ("passes_so", "colon_passes", "direct_pick", "grab_states_so"):
        fn = next(nd for nd in tree.body if isinstance(nd, ast.FunctionDef) and nd.name == nm)
        if not any(isinstance(d, ast.Call) and ast.unparse(d.func) == "torch.no_grad" for d in fn.decorator_list):
            bad.append(nm)
    assert n == len(E.PARTS) and not bad, ("no_grad", n, bad); return n


PROP = {"acc", "acc_moved", "floor_hold", "sm1_confusion", "ws", "cws", "follow_cf", "stay_own", "follow_dfin", "follow_stale", "mean_floor", "mean_stay_own",
        "intact_acc", "acc_vs_sk", "acc_vs_s2", "acc_vs_s4", "acc_chain", "share_gt1", "share_gt1_hold"}


def check_props(x, key=None, path="", bad=None):
    bad = [] if bad is None else bad
    if isinstance(x, dict):
        for k, v in x.items():
            if k in ("items", "per_item"):
                continue
            check_props(v, k if k in PROP else (key if key in PROP else None), path + "/" + str(k), bad)
    elif key in PROP and not (x is None or (isinstance(x, (int, float)) and 0.0 <= x <= 1.0)):
        bad.append((path, x))
    return bad


def get(d, path):
    if path == "n/a":                                                       # the literal key of the one-pass arms' n/a record
        assert d["n/a"] is True and d["arm"] == "direct"; return True
    for k in path.split("/"):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


def required(arm):
    """every score_n1.py SCHEMA path (a41e40af), expanded over its placeholders; plus the docstring extras."""
    loop = arm == "inject"; A = []
    for n in (3, 4, 5):
        for m in range(1, 13):
            for md in ("NUM", "ORD"):
                A += ["by_cell/%d/%d/%s/%s" % (n, m, md, f) for f in ("n_correct", "n_items", "n_correct_moved", "n_moved", "acc_moved_k8", "sm1_confusion", "f",
                                                                     "floor_hold", "items/pred", "items/ans", "items/k_touch", "items/moved", "items/s2", "items/s4")]
    A += ["set_sha", "by_cell/5/4/OFFSET/n_correct", "by_cell/5/4/OFFSET/n_items"]
    A += ["by_cell/5/%d/TRANSFER/%s" % (m, f) for m in range(1, 9) for f in ("n_correct", "n_items")]
    R = {"acc_so": A, "perm_label": ["by_m/%d/%s" % (m, f) for m in (2, 3, 4) for f in ("acc", "n")], "lookup": ["by_L/%d/%s" % (L, f) for L in range(1, 13) for f in ("acc", "n")],
         "bbh": ["subsets/%s/per_item" % s_ for s_ in ("three", "five", "seven")] + ["subsets/five/per_item/0/parsed", "subsets/five/per_item/0/correct"],
         "floors": ["status"]}
    if loop:
        R["acc_so"] += ["by_cell/5/12/NUM/by_kprime/12/n_correct", "by_cell/5/12/NUM/by_kprime/8/acc_moved"]
        R["ws_so"] = ["by_m/%d/%s/%d" % (m, f, t) for m in (4, 8, 12) for f in ("ws", "cws", "n") for t in range(1, max(8, m) + 1)]
        R["swap_so"] = ["by_cell/%s/%s" % (c, f) for c in ("5_4_2", "5_8_4", "5_6_3") for f in ("n_pairs", "follow_cf", "mean_floor", "stay_own", "follow_dfin", "follow_stale")]
        R["kcurve_so"] = (["acc_moved/%d" % k for k in (0, 1, 2, 4, 6, 7, 8, 9, 12, 16)] + ["n_moved"] + ["by_m/%d/acc_moved/%d" % (m, k) for m in (10, 12) for k in (8, 12)]
                          + ["by_m/%d/n_moved" % m for m in (10, 12)] + ["not_decoder/%s" % f for f in ("acc_vs_s2", "acc_vs_s4", "n")]
                          + ["s_norm/%d/r_median/%d" % (m, t) for m in (8, 12) for t in range(1, m + 1)])
        R["long_so"] = ["by_m/%d/acc/%d" % (m, k) for m in (1, 2, 3, 4) for k in (8, 12, 63, 64)] + ["by_m/%d/n" % m for m in (1, 2, 3, 4)] +             ["by_m/%d/sm1_confusion/%d" % (m, k) for m in (1, 2, 3, 4) for k in (63, 64)]
        R["beta_so"] = ["by_m/%d/share_gt1" % m for m in range(5, 13)]
    else:
        R.update({"ws_so": ["n/a"], "swap_so": ["n/a"], "long_so": ["n/a"], "beta_so": ["n/a"], "kcurve_so": ["acc_moved/0", "n_moved", "by_m/12/acc_moved/0"]})
    return R


def count_checks(d, path="", bad=None):
    """n_correct <= n_items, n_correct_moved <= n_moved, n_moved <= n_items in every acc_so cell."""
    bad = [] if bad is None else bad
    for n, d1 in d["by_cell"].items():
        for m, d2 in d1.items():
            for md, c in d2.items():
                if not all({"n", "n_correct"} <= set(v) for v in c["by_k_touch"].values()) or not c["by_k_touch"]:
                    bad.append(("by_k_touch", n, m, md))
                if not (0 <= c["n_correct"] <= c["n_items"] and 0 <= c["n_correct_moved"] <= c["n_moved"] <= c["n_items"]):
                    bad.append(("counts", n, m, md))
    return bad


def make_model(arm):
    ns = argparse.Namespace(arm=arm, mem_write="top", write_src="answer", base="gpt2", init_trunk="", init_sha="", seed=1, nk=0)
    torch.manual_seed(1); DM.FAST_ADD = True; m, g, params = AW.build_aw(ns, "cpu")
    if m is not None:
        m.mem.gamma.fill_(1.0)                                              # the AW recipe: gamma 1 from step 1
    params.eval(); return m, g


def main():
    t0 = time.time(); n_ng = ast_no_grad(); print("ast: %d part_* functions + 4 runners decorated torch.no_grad" % n_ng, flush=True)
    bb = E.bbh_check(BBH); print("bbh_check: %d / %d parsed" % (bb["n_parsed_total"], bb["n_total"]), flush=True)
    sets = E.build_sets(); print("sets built", round(time.time() - t0), "s", sets["shas"], flush=True); res = {}
    for arm in ("inject", "direct"):
        out = os.path.join(OUT, arm); os.makedirs(out, exist_ok=True)
        args = {"arm": arm, "think_k": 8 if arm == "inject" else 0, "cue": "index" if arm == "inject" else "none", "write_src": "answer", "seed": 1, "src": "so",
                "bbh_dir": BBH, "fast_add": True}
        with open(os.path.join(out, "args.json"), "w", encoding="utf-8", newline="") as fh:
            json.dump(args, fh, indent=1)
        m, g = make_model(arm); t1 = time.time()
        E.run(out, list(E.PARTS), "cpu", 64, model=(m, g), sets=sets, train={"smoke": True})
        miss, badp = [], []
        for p_, paths in required(arm).items():
            d = json.load(open(os.path.join(out, "eval_%s.json" % p_), encoding="utf-8"))
            for q in paths:
                try:
                    get(d, q)
                except (KeyError, TypeError, IndexError, AssertionError):
                    miss.append((p_, q))
            badp += [(p_,) + b for b in check_props(d)]
        badp += count_checks(json.load(open(os.path.join(out, "eval_acc_so.json"), encoding="utf-8")))
        ev = json.load(open(os.path.join(out, "eval.json"), encoding="utf-8"))
        assert sorted(ev["parts"]) == sorted(E.PARTS), ("eval.json parts", ev["parts"])
        res[arm] = {"sec": round(time.time() - t1), "missing": miss, "bad_props": badp}
        print(arm, json.dumps(res[arm]), flush=True)
        del m, g
    ok = all(not v["missing"] and not v["bad_props"] for v in res.values())
    print("SMOKE", "PASS" if ok else "FAIL", round(time.time() - t0), "s", flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
