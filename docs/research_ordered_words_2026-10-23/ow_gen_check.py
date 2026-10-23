"""$0 CPU check (OW_draft.md §3, F1-diff): ORDERED-WORDS v0 vs NESTED-WORDS v1.2 differ only in value range (mod 11) and premise order.
Single process, numpy only (threads pinned to 1), no GPU. Writes ow_gen_check.json next to this file.
    python -B docs/research_ordered_words_2026-10-23/ow_gen_check.py"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ow_words as OW  # noqa: E402
NW = OW.NW
N_STAT, N_SOLVE, N_PAIRS = 2000, 200, 1000
DS = range(1, 11)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def canonical(it):
    seq = ["S0"]
    for t, op in enumerate(it["ops"], 1):
        seq += (["G%d" % t] if op.endswith("_v") else []) + ["S%d" % t]
    return seq


def deps_of(it):
    dep = {"S0": []}
    for t, (op, j) in enumerate(zip(it["ops"], it["src_j"]), 1):
        dep["S%d" % t] = ["S%d" % (t - 1)] + (["G%d" % t] if op.endswith("_v") else []) + (["S%d" % j] if op.endswith("_s") else [])
        if op.endswith("_v"):
            dep["G%d" % t] = []
    return dep


def violations(it):
    pos = {x: i for i, x in enumerate(it["order"])}; dep = deps_of(it)
    return sum(1 for x, ds in dep.items() for y in ds if pos[y] > pos[x])


def given_gap(it):
    pos = {x: i for i, x in enumerate(it["order"])}
    return [pos["S%d" % t] - pos["G%d" % t] - 1 for t, op in enumerate(it["ops"], 1) if op.endswith("_v")]


def numbers(text):
    import re
    return [int(x) for x in re.findall(r"\b\d+\b", text)]


def cf_mod(rec, don, t, apply):
    S = list(don["values"][:t + 1])
    for k in range(t + 1, rec["d"] + 1):
        op, b, j = rec["ops"][k - 1], rec["operands"][k - 1], rec["src_j"][k - 1]
        S.append(apply(op, S[k - 1], S[j] if op.endswith("_s") else b))
    return S[-1]


def affine_drop_mod(it):
    x = it["values"][0]
    for op, b in zip(it["ops"], it["operands"]):
        if not op.endswith("_s"):
            x = OW.apply(op, x, b)
    return x


def bounded_variant(rng, d, lo=0, hi=10, tries=2000):
    """variant B (rejected design): bounded natural values 0..10 with v1.2-style rejection (no wrap) -> (restarts, answer) or (tries, None)."""
    for r in range(tries):
        ts = int(rng.integers(3, d + 1)) if d >= 3 else None; s = [int(rng.integers(1, hi + 1))]; ok = True
        for t in range(1, d + 1):
            if t == ts:
                cand = [j for j in range(1, t - 1) if s[j] >= 2 and s[t - 1] >= 2 and s[t - 1] * s[j] <= hi]
                if not cand:
                    ok = False; break
                s.append(s[t - 1] * s[cand[int(rng.integers(0, len(cand)))]]); continue
            for _ in range(50):
                op = "mul_v" if d == 1 else OW.OPS_REG[int(rng.integers(0, len(OW.OPS_REG)))]
                if op == "add_s" and t < 2:
                    continue
                b = s[int(rng.integers(0, t - 1))] if op == "add_s" else (int(rng.integers(2, hi)) if OW.BASE[op] == "mul" else int(rng.integers(1, hi)))
                if OW.BASE[op] == "mul" and s[t - 1] == 0:
                    continue
                v = NW.apply(op, s[t - 1], b)
                if lo <= v <= hi:
                    break
            else:
                ok = False; break
            s.append(v)
        if ok:
            return r, s[-1]
    return tries, None


def main():
    import tiktoken
    enc = tiktoken.get_encoding("gpt2"); out = {"note": "OW_draft.md §3 F1-diff ($0 CPU)", "nested_words_sha256": sha(NW.__file__), "ow_words_sha256": sha(OW.__file__)}
    # ---- 1. shared objects by reference, and which v1.2 functions OW re-implements
    shared = {k: getattr(OW, k) is getattr(NW, k) for k in ("FRAMES", "NAMES", "UNITS", "GIVEN", "QUESTION", "FRAME_TRAIN", "FRAME_ALL", "OPS", "OPS_REG", "BASE")}
    reused_fns = ["_statements", "_finish", "topo_shuffle", "solve (wrapped: apply -> mod 11)", "apply (wrapped: % 11)"]
    replaced = {}
    for fn in ("_operand", "_draw_program", "make", "redraw"):
        a, b = inspect.getsource(getattr(NW, fn)).splitlines(), inspect.getsource(getattr(OW, fn)).splitlines()
        replaced[fn] = {"nw_lines": len(a), "ow_lines": len(b)}
    out["identity"] = {"shared_by_reference": shared, "all_shared": all(shared.values()), "reused_functions": reused_fns, "reimplemented": replaced,
                       "added_item_keys": None}
    # ---- 2. item streams
    rng_ow, rng_topo, rng_nw = np.random.default_rng(1), np.random.default_rng(2), np.random.default_rng(3)
    st_ow, st_topo, st_nw = {}, {}, {}
    items = {"ow": {}, "topo": {}, "nw": {}}
    for d in DS:
        items["ow"][d] = [OW.make(rng_ow, d, OW.FRAME_ALL, st_ow) for _ in range(N_STAT)]
        items["topo"][d] = [OW.make(rng_topo, d, OW.FRAME_ALL, st_topo, order_mode="topo") for _ in range(N_STAT)]
        items["nw"][d] = [NW.make(rng_nw, d, NW.FRAME_ALL, st_nw) for _ in range(N_STAT)]
    k_ow, k_nw = set(items["ow"][3][0]), set(items["nw"][3][0]); out["identity"]["added_item_keys"] = sorted(k_ow - k_nw); out["identity"]["missing_item_keys"] = sorted(k_nw - k_ow)
    out["restarts_per_item"] = {"ow": st_ow["restarts"] / st_ow["items"], "topo": st_topo["restarts"] / st_topo["items"], "nw": st_nw["restarts"] / st_nw["items"]}
    # ---- 3. values and answer law
    val = {}
    for nm in ("ow", "nw"):
        val[nm] = {}
        for d in DS:
            its = items[nm][d]; ans = Counter(it["answer"] for it in its); allv = [v for it in its for v in it["values"]]
            half = len(its) // 2; tab = {}
            for it in its[:half]:
                tab.setdefault(it["ops"][-1], Counter())[it["answer"]] += 1
            hit = np.mean([tab.get(it["ops"][-1], Counter({-1: 1})).most_common(1)[0][0] == it["answer"] for it in its[half:]])
            val[nm][d] = {"min": min(allv), "max": max(allv), "n_answer_values": len(ans), "answer_max_share": max(ans.values()) / len(its),
                          "final_op_lookup_acc": float(hit), "chance": 1 / len(ans)}
    out["values"] = val
    out["values_ok"] = all(v["min"] >= 0 and v["max"] <= 10 and v["answer_max_share"] <= 0.15 for v in val["ow"].values())
    # ---- 4. order
    ordr = {}
    for nm in ("ow", "topo", "nw"):
        viol = sum(violations(it) > 0 for d in DS for it in items[nm][d]); n = sum(len(items[nm][d]) for d in DS)
        canon = np.mean([[x for x in it["order"] if not x.startswith("D")] == canonical(it) for d in DS for it in items[nm][d]])
        gaps = [g for d in DS for it in items[nm][d] for g in given_gap(it)]
        lay = {}
        for d in (3, 6):
            per = {}
            for it in items[nm][d]:
                per.setdefault(tuple(it["ops"]), set()).add(tuple(x for x in it["order"] if not x.startswith("D")))
            lay[str(d)] = float(np.mean([len(v) for v in per.values() if True]))
        ordr[nm] = {"items_with_dependency_violation": int(viol), "n_items": n, "share_program_layout_canonical": float(canon),
                    "given_to_use_gap_mean_sentences": float(np.mean(gaps)), "given_to_use_gap_max": int(max(gaps)), "mean_distinct_program_layouts_per_op_sequence": lay}
    out["order"] = ordr
    out["order_ok"] = ordr["ow"]["items_with_dependency_violation"] == 0 and ordr["ow"]["share_program_layout_canonical"] == 1.0
    # ---- 5. independent solver (mod 11) on the text and every R_t; step sentences; single tokens
    bad = []
    for nm in ("ow", "topo"):
        for d in DS:
            for it in items[nm][d][:N_SOLVE]:
                if OW.solve(it["text"]) != it["answer"]:
                    bad.append((nm, d, "text"))
                for t in range(1, d):
                    if OW.solve(it["R"][t]) != it["answer"]:
                        bad.append((nm, d, "R", t))
                if it["R_step"][-1] != it["R"][d] or len(it["R_step"]) != d:
                    bad.append((nm, d, "R_step"))
    tok1 = all(len(enc.encode(" %d" % v)) == 1 for v in range(11))
    out["solver"] = {"mismatches": len(bad), "first": bad[:5], "numbers_single_token": tok1}
    # ---- 6. frames: same law (uniform over the frames offered)
    fr = Counter((op, f) for d in DS for it in items["ow"][d] for op, f in zip(it["ops"], it["frames"]))
    per_op = {}
    for op in OW.OPS:
        c = [fr.get((op, f), 0) for f in OW.FRAME_ALL]; n = sum(c)
        if n:
            e = n / len(c); per_op[op] = {"n": n, "chi2_vs_uniform": float(sum((x - e) ** 2 / e for x in c)), "df": len(c) - 1}
    out["frames"] = per_op
    # ---- 7. shortcut panel (OW forward, all 8 frames), G-W0(i) analogue
    sc = {}
    maj = Counter(it["answer"] for d in DS for it in items["topo"][d]).most_common(1)[0][0]      # fitted on another stream
    for d in DS:
        its = items["ow"][d]; ns = [numbers(it["text"]) for it in its]
        rules = {"majority": [maj == it["answer"] for it in its],
                 "first_number": [n[0] == it["answer"] for n, it in zip(ns, its)],
                 "last_number": [n[-1] == it["answer"] for n, it in zip(ns, its)],
                 "sum_mod11": [sum(n) % 11 == it["answer"] for n, it in zip(ns, its)],
                 "product_mod11": [math.prod(x for x in n if x) % 11 == it["answer"] for n, it in zip(ns, its)],
                 "affine_drop_cross_spine": [affine_drop_mod(it) == it["answer"] for it in its]}
        sc[str(d)] = {k: float(np.mean(v)) for k, v in rules.items()}
        sc[str(d)]["affine_representable_share"] = float(np.mean([not any(op == "mul_s" and j is not None and j >= 1 for op, j in zip(it["ops"], it["src_j"])) for it in its]))
    out["shortcuts"] = sc
    out["shortcuts_ok"] = all(v <= 0.25 for d, r in sc.items() if int(d) >= 2 for k, v in r.items() if k not in ("affine_representable_share", "affine_drop_cross_spine"))
    out["shortcuts_note"] = "affine_drop_cross_spine is the exact solver for d 1-2 (no cross-spine product) and is descriptive, as v1.2"
    # ---- 8. pairs: kept share (own / donor-final / cf distinct at t = ceil(d/2)), cf floor, equal token length
    PC = OW.pair_candidates(per_d=N_PAIRS, ds=range(3, 11)); pr = {}; len_bad = 0
    for d, ps in PC.items():
        t = math.ceil(d / 2); kept = []
        for rec, don in ps:
            cf = cf_mod(rec, don, t, OW.apply)
            if len({rec["answer"], don["answer"], cf}) == 3:
                kept.append((rec, don, cf))
        fl = [cf_mod(kept[i][0], kept[(i + 1) % len(kept)][1], t, OW.apply) == kept[i][2] for i in range(len(kept))] or [float("nan")]
        for rec, don in ps[:100]:
            len_bad += len(enc.encode(rec["text"])) != len(enc.encode(don["text"]))
        pr[str(d)] = {"t": t, "kept_share": len(kept) / len(ps), "cf_floor_ideal_carrier": float(np.mean(fl)), "donor_final_eq_cf_excluded": True}
    out["pairs"] = {"per_d": pr, "token_length_mismatches_in_sample": len_bad,
                    "note": "d = 3: t = 2 and the forced mul_s sits at t = 3 with j = 1, so cf(a) = donor S_2 x donor S_1 = donor-final for every pair "
                            "(structurally degenerate; kept share 0, as in v1.2 / ps_eval); the serial reading pools d >= 4"}
    out["pairs_ok"] = all(v["kept_share"] >= 0.5 for d, v in pr.items() if int(d) >= 4) and pr["3"]["kept_share"] == 0.0 and len_bad == 0
    # ---- 9. exclusion collisions (train stream vs a test set, smallest space d = 1, 2)
    TE = OW.test_set(per_d=1000, ds=range(1, 3)); sig = {it["sig"] for d in TE for it in TE[d]}; rs = np.random.default_rng(OW.SEED_TRAIN)
    col = {str(d): float(np.mean([OW.make(rs, d)["sig"] in sig for _ in range(20000)])) for d in (1, 2)}
    out["exclusion"] = {"train_draws_hitting_test_sig_share": col, "test_dups_within": {str(d): len(TE[d]) - len({x["sig"] for x in TE[d]}) for d in TE}}
    # ---- 10. token lengths (question)
    out["question_tokens_mean"] = {nm: {str(d): float(np.mean([len(enc.encode(it["text"])) for it in items[nm][d][:300]])) for d in (1, 3, 6, 10)} for nm in ("ow", "nw")}
    # ---- 11. rejected variant B (bounded naturals 0..10, no wrap)
    rb = np.random.default_rng(4); vb = {}
    for d in (2, 3, 4, 6):
        res = [bounded_variant(rb, d) for _ in range(400)]; ans = Counter(a for _, a in res if a is not None)
        vb[str(d)] = {"restarts_per_item": float(np.mean([r for r, _ in res])), "failed": sum(a is None for _, a in res),
                      "answer_max_share": max(ans.values()) / max(1, sum(ans.values())) if ans else None}
    out["variant_B_bounded_0_10"] = vb
    out["PASS"] = bool(out["identity"]["all_shared"] and out["values_ok"] and out["order_ok"] and out["solver"]["mismatches"] == 0 and tok1
                       and out["shortcuts_ok"] and out["pairs_ok"] and not out["identity"]["missing_item_keys"])
    with open(os.path.join(HERE, "ow_gen_check.json"), "w", newline="\n") as f:
        json.dump(out, f, indent=1)
    print("OW-GEN-CHECK", "PASS" if out["PASS"] else "FAIL", json.dumps({k: out[k] for k in ("values_ok", "order_ok", "shortcuts_ok", "pairs_ok")}), out["solver"]["mismatches"])


if __name__ == "__main__":
    main()
