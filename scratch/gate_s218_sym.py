"""§218 LANG-VB gates for boxes/prosqa_sym.py (LANG_v1 §7): G-L0 SYM panel, G-L3 REACH re-render, G-L4 PREMISE-SWAP pairs, G-L4b CF pairs
(pair-maker part), G-L9 generator timing, plus prototype parity (v1_sym.py / v1_cf.py / v1_sym.json). CPU only; JSON to scratch/gate_s218/.
Every check re-derives its quantity independently of the module (statements parsed back from the rendered text, own BFS / layer expansion,
token offsets from tiktoken bytes).
    python -B scratch/gate_s218_sym.py sigs | parity | g_l0 train,pool | g_l0 gate,test | merge | g_l3 | g_l4 | g_l4b
"""
import collections
import json
import os
import random
import re
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "boxes"))
import prosqa_sym as S  # noqa: E402

OUT = os.path.join(ROOT, "scratch", "gate_s218"); os.makedirs(OUT, exist_ok=True)
DOCS = os.path.join(ROOT, "docs", "research_lang_2026-10-18")
STREAMS = {"train": S.SEED_TRAIN, "pool": S.SEED_POOL, "gate": S.SEED_GATE, "test": S.SEED_TEST}
N_BAND = 5000
GATED = ("first", "random", "maxdeg", "la1")


def dump(name, obj):
    with open(os.path.join(OUT, name), "w", newline="\n") as f:
        json.dump(obj, f, indent=1)
    print("wrote", name, flush=True)


def load_excl():
    return set(json.load(open(os.path.join(OUT, "excl_sigs.json")))["sigs"])


# ---------- independent re-derivations (no module code) ----------
STM = re.compile(r"(Every )?([A-Za-z]+) is a ([A-Za-z]+)\.")
QST = re.compile(r" Is ([A-Za-z]+) a ([A-Za-z]+) or ([A-Za-z]+)\?$")


def parse(text):
    """statements and question parsed back from the rendered text -> (edges by name, persons set, (q, o1, o2), premise name sequence)"""
    body, q = text.rsplit(" Is ", 1)
    st = STM.findall(body); m = QST.search(text)
    assert m and len(st) == body.count(".") and " ".join(("Every %s is a %s." if e else "%s is a %s.") % (a, b) for e, a, b in st) == body
    persons = {a for e, a, b in st if not e}
    return [(a, b) for e, a, b in st], persons, m.groups(), [w for e, a, b in st for w in (a, b)]


def layers(edges, src, t_max):
    """R_t by set expansion (R_0 = {src}); returns first-entry time per name."""
    succ = collections.defaultdict(set)
    for a, b in edges:
        succ[a].add(b)
    f = {src: 0}; cur = {src}
    for t in range(1, t_max + 1):
        new = {v for u in cur for v in succ[u]} - set(f)
        if not new:
            break
        for v in new:
            f[v] = t
        cur = new
    return f


def first_mentions(text, concepts):
    """index in [EOT] + enc(" " + text) of the last token of each concept name's first whole-word mention, from token byte offsets"""
    s = " " + text; ids = S.ENC.encode(s); ends = []; p = 0
    for t in ids:
        p += len(S.ENC.decode_single_token_bytes(t)); ends.append(p)
    out = {}
    for w in concepts:
        m = re.search(r" " + w + r"(?=[ .?])", s)
        if m is None:
            out[w] = None; continue
        e = m.end(); out[w] = 1 + next(j for j, x in enumerate(ends) if x >= e)
    return out, len(ids) + 1


def indep_invariants(it):
    """§2.2 item 8 re-derived from the rendered text."""
    nm = it["names"]; edges, persons, (q, o1, o2), seq = parse(it["text"]); f = []
    fR = layers(edges, nm[it["R"]], 10 ** 6); fP = layers(edges, nm[it["P"]], 10 ** 6)
    for k, (a, b) in enumerate(it["levels"]):
        if fR.get(nm[a]) != k + 1 or fP.get(nm[b]) != k + 1:
            f.append("dist level %d" % (k + 1))
    if nm[it["o_P"]] in fR or nm[it["o_R"]] in fP:
        f.append("cross reach")
    concepts = [w for w in nm if w[:1].islower()]
    if set(seq) != set(nm) or q != nm[it["R"]] or {o1, o2} != {nm[it["o_R"]], nm[it["o_P"]]}:
        f.append("mention / question")
    T = len(S.ENC.encode(" " + it["text"])) + 1
    if T > 1000 or T != it["T"]:
        f.append("T")
    if any(len(S.ENC.encode(" " + p)) != 1 for p in persons) or persons != {w for w in nm if w[:1].isupper()}:
        f.append("person tokens")
    if max(fR[w] for w in fR) > it["D"]:
        f.append("beyond D")
    if any(len(S.ENC.encode(" " + w)) != 2 for w in concepts):
        f.append("concept tokens")
    return f


def summ(v):
    v = sorted(v); return {"mean": round(sum(v) / len(v), 2), "p95": v[int(0.95 * len(v))], "max": v[-1], "min": v[0]}


# ---------- sigs (exclusion set; G-L9 "the exclusion set at full size") ----------
def cmd_sigs():
    t0 = time.perf_counter(); test = S.fixed_sigs(S.SEED_TEST, 1000, range(3, 17)); t1 = time.perf_counter()
    gate = S.fixed_sigs(S.SEED_GATE, 2000, range(9, 13)); t2 = time.perf_counter()
    dump("excl_sigs.json", {"n_test_sigs": len(test), "n_gate_sigs": len(gate), "test_gate_overlap": len(test & gate), "sec_test": round(t1 - t0, 1),
                            "sec_gate": round(t2 - t1, 1), "sigs": sorted(test | gate)})


# ---------- parity with the measured prototype ----------
def cmd_parity(band_only=False):
    sys.path.insert(0, os.path.join(DOCS, "v1_checks")); sys.path.insert(0, os.path.join(DOCS, "scripts")); argv = sys.argv; sys.argv = [argv[0]]
    import v1_sym as V  # noqa: E402  (read-only import of the prototype)
    from heur_panel import panel_for  # noqa: E402
    sys.argv = argv
    P = {"dlo": 0, "dhi": 0, "klo": "half", "rmax": 4, "cap": (lambda D: None if D < 9 else 2 * D + 6), "jmax": 2, "leaves": "last",
         "pextra": 0.1, "toplev": 0, "mlo": 1, "mhi": 1, "rr": 4, "rr_small": 5}                       # = v1_cf.py's v1 law
    out = json.load(open(os.path.join(OUT, "parity.json"))) if band_only else {}
    V.LEAD = " "; bad = collections.Counter(); n = 0
    for D in (range(3, 17) if not band_only else ()):              # (a) item parity, fresh seeds
        a, b = random.Random(9000 + D), random.Random(9000 + D)
        for _ in range(200):
            x = V.make(a, D, P); y = S.make_item(b, D); n += 1
            bad["text"] += x["question"] != " " + y["text"]; bad["edges"] += x["edges"] != y["edges"]; bad["names"] += x["idx_to_symbol"] != y["names"]
            bad["T"] += x["T"] != y["T"]; bad["order"] += x["order"] != y["order"]; bad["o1"] += x["o1"] != y["o1"]
            bad["levels"] += [tuple(l) for l in x["levels"]] != [tuple(l) for l in y["levels"]]
            conc = y["concepts"]; fm = V.first_mention_last_tok(x["question"], x["idx_to_symbol"], conc)
            bad["slot_idx"] += {c: fm[c] + 1 for c in conc} != y["slot_idx"]
            pv = {}
            for k, (u, v) in enumerate(x["levels"]):
                for h, val in panel_for(x, u, v).items():
                    pv["L%d_%s" % (k + 1, h)] = val
            mine = S.panel_levels(y); bad["panel"] += any(mine[k] != v for k, v in pv.items())
            s0 = random.getrandbits(32); h1, h2 = random.Random(s0), random.Random(s0)
            pc = [V.chase(x, h1, p) for p in ("first", "random", "maxdeg")] + [V.chase(x, h1, "la", h) for h in (1, 2, 3, 4)]
            mc = S.chase_set(y, h2); bad["chase"] += pc != [mc[k] for k in ("first", "random", "maxdeg", "la1", "la2", "la3", "la4")]
            bad["chase_rng_state"] += h1.getstate() != h2.getstate()
    if not band_only:
        out["item_parity"] = {"n": n, "mismatches": dict(bad), "pass": sum(bad.values()) == 0}; print(out["item_parity"], flush=True)
    # (b) exact replication of v1_cf.json's exposure table (seed 24300, 1,000 per D 4..8, the prototype's sequence)
    ref = json.load(open(os.path.join(DOCS, "v1_checks", "v1_cf.json")))["exposure"]; rng = random.Random(24300); diffs = []; tab = {}
    for D in (range(4, 9) if not band_only else ()):
        cnt = collections.Counter()
        for _ in range(1000):
            it = S.make_item(rng, D)
            for j in range(1, D + 1):
                r = S.cf_donor(it, j)
                if r is None:
                    continue
                cnt["n", j] += 1; cnt["flip", j] += r["flip"]; cnt["exposed", j] += r["exposed"]
        for j in range(1, D + 1):
            mine = {"n": cnt["n", j], "flip": cnt["flip", j], "exposed": cnt["exposed", j]}; tab["%d_%d" % (D, j)] = mine
            if mine != ref[str(D)]["single"][str(j)]:
                diffs.append({"D": D, "j": j, "mine": mine, "ref": ref[str(D)]["single"][str(j)]})
    if not band_only:
        out["v1_cf_exposure_replication"] = {"seed": 24300, "cells": len(tab), "diffs": diffs, "pass": not diffs}; print(out["v1_cf_exposure_replication"], flush=True)
    # (c) exact replication of v1_sym.json's band table (seed 24000, 5,000 per D 3..16, no lead space, expo_n 1000; prototype hr order)
    ref = json.load(open(os.path.join(DOCS, "v1_checks", "v1_sym.json")))["bands"]; rng = random.Random(24000); hr = random.Random(24001); rows = {}
    keys_bad = []
    for D in range(3, 17):
        acc = collections.defaultdict(list); beyond = 0; Sz = []; Pn = []; fail = 0
        for i in range(5000):
            it = S.make_item(rng, D); fail += bool(S.check(it))
            pl = S.panel_levels(it)
            for k in range(1, D + 1):
                for c in S.CUES:
                    acc["L%d_%s" % (k, c)].append(pl["L%d_%s" % (k, c)])
            cs = S.chase_set(it, hr)
            for p in ("first", "random", "maxdeg"):
                acc["HROOT_" + p].append(cs[p])
            for h in (1, 2, 3, 4):
                acc["HROOT_LA%d" % h].append(cs["la%d" % h])
            acc["HROOT_soft"].append(cs["soft"])
            ud = S._ud(it)
            for r, v in S.ubidir(it, ud).items():
                acc["uBIDIR%d" % r].append(v)
            acc["UDIST"].append(S.udist(it, ud)); beyond += S.beyond_D(it); Sz.append(len(it["concepts"])); Pn.append(len(it["persons"]))
            if 4 <= D <= 8 and i < 1000:
                hr.randint(3, D - 1)                                # the prototype's (i') exposure t draw (v1_sym.json: expo items 1,000)
        row = {k: round(max(sum(v) / len(v), 1 - sum(v) / len(v)), 4) for k, v in acc.items()}
        row.update({"invariant_fail": fail, "beyond_D_share": round(beyond / 5000, 4), "S_mean": round(sum(Sz) / len(Sz), 1), "S_max": max(Sz),
                    "persons_mean": round(sum(Pn) / len(Pn), 2), "persons_max": max(Pn)})
        rr = ref[str(D)]; d = {k: (v, rr.get(k)) for k, v in row.items() if rr.get(k) != v}
        rows[D] = {"n_keys": len(row), "diffs": d}; keys_bad += [(D, k) for k in d]
        print("v1_sym replication D", D, "diffs", d, flush=True)
    out["v1_sym_band_replication"] = {"seed": 24000, "n_per_D": 5000, "rows": rows, "pass": not keys_bad,
                                      "note": "v1_sym.json was run without the leading space (lead_space flag absent from its params); T is not compared"}
    dump("parity.json", out)


# ---------- G-L0 ----------
def cmd_g_l0(names):
    excl = load_excl()
    for sname in names.split(","):
        seed = STREAMS[sname]; st = S.Stream(seed, excl if sname in ("train", "pool") else set()); res = {"stream": sname, "seed": seed, "bands": {}}
        t0 = time.perf_counter()
        for D in range(3, 17):
            hr = random.Random(seed * 1000 + 500 + D); subsets = {"all": None, "testfile": 1000} if sname == "test" else {"all": None}
            acc = {k: collections.defaultdict(list) for k in subsets}; cnt = collections.Counter(); Ts, Sz, Pn, Xc = [], [], [], []; sigs = collections.Counter()
            fails = []
            for i in range(N_BAND):
                try:
                    it = st.draw(D)
                except S.SymInvariantError as e:
                    cnt["module_invariant_fail"] += 1; fails.append(str(e)[:200]); continue
                f = indep_invariants(it)
                if f:
                    cnt["indep_invariant_fail"] += 1; fails.append(f)
                pl = S.panel_levels(it); cs = S.chase_set(it, hr); ud = S._ud(it); ub = S.ubidir(it, ud); u = S.udist(it, ud)
                cnt["ubidir_not_half"] += any(v != 0.5 for v in ub.values()); cnt["udist_not_1"] += u != 1.0; cnt["beyond_D"] += S.beyond_D(it)
                Ts.append(it["T"]); Sz.append(len(it["concepts"])); Pn.append(len(it["persons"])); Xc.append(sum(it["n_cross"])); sigs[it["sig"]] += 1
                for sub, lim in subsets.items():
                    if lim is not None and i >= lim:
                        continue
                    a = acc[sub]
                    for k, v in pl.items():
                        a[k].append(v)
                    for k, v in cs.items():
                        a["chase_" + k].append(v)
                    for r, v in ub.items():
                        a["ubidir_%d" % r].append(v)
                    a["udist"].append(u)
            row = {"n": N_BAND, "module_invariant_fail": cnt["module_invariant_fail"], "indep_invariant_fail": cnt["indep_invariant_fail"],
                   "fail_examples": fails[:5], "ubidir_items_not_half": cnt["ubidir_not_half"], "udist_items_not_1": cnt["udist_not_1"],
                   "beyond_D_items": cnt["beyond_D"], "beyond_D_share": cnt["beyond_D"] / N_BAND, "T": summ(Ts), "T_le_1000": max(Ts) <= 1000,
                   "concepts": summ(Sz), "persons": summ(Pn), "cross_edges_mean": round(sum(Xc) / len(Xc), 3),
                   "sig_duplicates_within_band": sum(v - 1 for v in sigs.values())}
            for sub, a in acc.items():
                m = {k: sum(v) / len(v) for k, v in a.items()}; df = {k: round(max(x, 1 - x), 4) for k, x in m.items()}
                lv = {k: v for k, v in df.items() if k.startswith("L") and not k.startswith("L0_")}
                l0 = {k: v for k, v in df.items() if k.startswith("L0_")}
                wk = max(lv, key=lv.get)
                blk = {"n": len(a["udist"]), "cells": {k: df[k] for k in sorted(df) if k.startswith("L")},
                       "max_person_agnostic": lv[wk], "max_person_agnostic_which": wk, "max_L0_person": max(l0.values()),
                       "chase": {k[6:]: df[k] for k in df if k.startswith("chase_")},
                       "ubidir": {k[7:]: round(m[k], 4) for k in m if k.startswith("ubidir_")}, "udist": round(m["udist"], 4)}
                blk["gated_max"] = max(blk["chase"][g] for g in GATED); blk["gated_which"] = max(GATED, key=lambda g: blk["chase"][g])
                blk["gated_all_lt_0.6"] = all(blk["chase"][g] < 0.6 for g in GATED); blk["person_agnostic_lt_0.6"] = blk["max_person_agnostic"] < 0.6 and blk["max_L0_person"] < 0.6
                row[sub] = blk
            row["collisions_so_far"] = len(st.collisions); res["bands"][D] = row
            print(sname, "D", D, "inv", row["module_invariant_fail"], row["indep_invariant_fail"], "gated %.4f %s" % (row["all"]["gated_max"], row["all"]["gated_which"]),
                  "PA %.4f" % row["all"]["max_person_agnostic"], "L0 %.4f" % row["all"]["max_L0_person"], "ub!=.5", row["ubidir_items_not_half"],
                  "ud!=1", row["udist_items_not_1"], "beyond", row["beyond_D_items"], "T", row["T"], "%.0fs" % (time.perf_counter() - t0), flush=True)
        res["collisions"] = st.collisions; res["sec"] = round(time.perf_counter() - t0, 1)
        dump("g_l0_%s.json" % sname, res)


def cmd_merge():
    parts = {s: json.load(open(os.path.join(OUT, "g_l0_%s.json" % s))) for s in STREAMS}
    nc = S.name_checks(); names_ok = all(v for k, v in nc.items() if isinstance(v, bool)) and nc["sha256_persons"] == json.load(open(os.path.join(DOCS, "v1_checks", "v1_names.json")))["sha256_persons"]
    names_ok = names_ok and nc["sha256_nonce"] == json.load(open(os.path.join(DOCS, "v1_checks", "v1_names.json")))["sha256_nonce"]
    panel = {}; inv_total = 0; exact_ok = True; usable = []
    for D in range(3, 17):
        d = str(D); rows = {s: parts[s]["bands"][d] for s in STREAMS}
        inv = sum(r["module_invariant_fail"] + r["indep_invariant_fail"] for r in rows.values()); inv_total += inv
        ex = all(r["ubidir_items_not_half"] == 0 and r["udist_items_not_1"] == 0 and r["beyond_D_items"] == 0 for r in rows.values())
        ex = ex and all(all(v == 0.5 for v in r["all"]["ubidir"].values()) and r["all"]["udist"] == 1.0 for r in rows.values())
        exact_ok = exact_ok and ex
        gp = all(r["all"]["gated_all_lt_0.6"] for r in rows.values()); pa = all(r["all"]["person_agnostic_lt_0.6"] for r in rows.values())
        if gp and inv == 0 and ex:
            usable.append(D)
        panel[D] = {"n_gate": N_BAND, "invariant_failures": inv, "gated_max_by_stream": {s: r["all"]["gated_max"] for s, r in rows.items()},
                    "gated_which_by_stream": {s: r["all"]["gated_which"] for s, r in rows.items()},
                    "chase_by_stream": {s: r["all"]["chase"] for s, r in rows.items()},
                    "max_person_agnostic_by_stream": {s: r["all"]["max_person_agnostic"] for s, r in rows.items()},
                    "max_L0_person_by_stream": {s: r["all"]["max_L0_person"] for s, r in rows.items()},
                    "ubidir_by_stream": {s: r["all"]["ubidir"] for s, r in rows.items()}, "udist_by_stream": {s: r["all"]["udist"] for s, r in rows.items()},
                    "beyond_D_by_stream": {s: r["beyond_D_share"] for s, r in rows.items()}, "T_by_stream": {s: r["T"] for s, r in rows.items()},
                    "concepts_by_stream": {s: r["concepts"] for s, r in rows.items()}, "persons_by_stream": {s: r["persons"] for s, r in rows.items()},
                    "testfile_1000": rows["test"]["testfile"], "gate_pass": gp and inv == 0 and ex, "person_agnostic_all_lt_0.6": pa}
    out = {"spec": "LANG_v1 §7 G-L0; 5,000 draws per band per stream; gated = first / random / maxdeg / la1 (< 0.6 every stream)",
           "names": nc, "names_ok": names_ok, "invariant_failures_total": inv_total, "ubidir_udist_beyondD_exact": exact_ok,
           "usable_bands": usable, "collisions": {s: len(parts[s]["collisions"]) for s in STREAMS}, "sec": {s: parts[s]["sec"] for s in STREAMS},
           "pass": names_ok and inv_total == 0 and exact_ok and usable == list(range(3, 17)), "panel": panel,
           "full_rows": {s: parts[s]["bands"] for s in STREAMS}}
    dump("g_l0.json", out)
    print("G-L0 pass", out["pass"], "usable", usable, "inv", inv_total, "exact", exact_ok, "names", names_ok)


# ---------- G-L3 REACH part + G-L9 generator timing ----------
def cmd_g_l3():
    excl = load_excl(); st = S.Stream(S.SEED_TRAIN, excl); dr = random.Random(7); t_max = 24
    gen_sec = 0.0; byD = collections.Counter(); secD = collections.Counter(); diffs_items = 0; diff_cells = 0; ex = []; cells = 0
    for i in range(100000):
        D = dr.randint(3, 16); t0 = time.perf_counter(); it = st.draw(D); dt = time.perf_counter() - t0; gen_sec += dt; byD[D] += 1; secD[D] += dt
        E = S.reach_targets(it, t_max)
        edges, persons, (q, o1, o2), seq = parse(it["text"]); f = layers(edges, q, t_max)
        cn = [it["names"][c] for c in it["concepts"]]; big = 10 ** 9
        ref = [[1 if f.get(w, big) <= t and w not in persons else 0 for w in cn] for t in range(t_max + 1)]
        if ref != E or any(E[0]):
            diffs_items += 1; diff_cells += sum(a != b for ra, rb in zip(ref, E) for a, b in zip(ra, rb)); ex.append(it["text"][:120]) if len(ex) < 3 else None
        cells += len(cn) * (t_max + 1)
        if (i + 1) % 20000 == 0:
            print("g_l3", i + 1, "diff items", diffs_items, "%.0f draws/s" % ((i + 1) / gen_sec), flush=True)
    dump("g_l3_reach.json", {"spec": "LANG_v1 §7 G-L3 (REACH part): reach_targets vs layer expansion over statements parsed from the text, t 0..24",
                             "n_items": 100000, "D": "U{3..16} (rng 7) from Stream(22100) with the test + GATE exclusion", "cells": cells,
                             "diff_items": diffs_items, "diff_cells": diff_cells, "examples": ex, "pass": diffs_items == 0, "collisions": st.collisions})
    excl_meta = {k: v for k, v in json.load(open(os.path.join(OUT, "excl_sigs.json"))).items() if k != "sigs"}
    dump("g_l9_gen.json", {"spec": "LANG_v1 §7 G-L9 (generator part): 100k Stream(22100).draw(D) calls, D U{3..16}, exclusion set at full size (test + GATE)",
                           "n": 100000, "gen_sec": round(gen_sec, 2), "draws_per_s": round(100000 / gen_sec, 1),
                           "by_D": {D: {"n": byD[D], "draws_per_s": round(byD[D] / secD[D], 1)} for D in sorted(byD)},
                           "exclusion_set": excl_meta, "omp_threads": os.environ.get("OMP_NUM_THREADS"), "collisions": len(st.collisions)})


# ---------- G-L4 PREMISE-SWAP pairs ----------
def frontier_diff(it, t, cs):
    dR, dP = it["dist_R"], it["dist_P"]; big = 99
    return [c for c in cs if (dR.get(c, big) <= t) != (dP.get(c, big) <= t)]


def cmd_g_l4():
    st = S.Stream(S.SEED_TEST); rng = random.Random(S.SEED_SWAP); res = {}
    for D in range(4, 9):
        cnt = collections.Counter(); Lsz = []; Ldf = []; Lsz_i = []; Ldf_i = []; fails = []
        for _ in range(N_BAND):
            it = st.draw(D); t_ii = rng.randint(3, D - 1); t_i = rng.randint(3, D - 2) if D >= 5 else None
            try:
                don = S.premise_swap(it)
            except S.SymInvariantError as e:
                cnt["maker_fail"] += 1; fails.append(str(e)); continue
            nm = it["names"]; cn = [nm[c] for c in it["concepts"]]
            fr, Tr = first_mentions(it["text"], cn); fd, Td = first_mentions(don["text"], cn)
            ident = Tr == Td == it["T"] == don["T"] and fr == fd and all(v is not None for v in fr.values()) and fr == {nm[c]: p for c, p in it["slot_idx"].items()}
            cnt["identity"] += ident
            e_r, p_r, q_r, _ = parse(it["text"]); e_d, p_d, q_d, _ = parse(don["text"]); sw = {nm[it["R"]]: nm[it["P"]], nm[it["P"]]: nm[it["R"]]}
            same_rest = q_r == q_d and e_d == [(sw.get(a, a), b) for a, b in e_r]
            fR = layers(e_d, q_d[0], 10 ** 6); ok_ans = fR.get(nm[it["o_P"]]) == D and nm[it["o_R"]] not in fR
            fP = layers(e_d, nm[it["P"]], 10 ** 6); ok_ans = ok_ans and fP.get(nm[it["o_R"]]) == D and nm[it["o_P"]] not in fP
            cnt["donor_answer_oP"] += ok_ans; cnt["only_person_subjects_swapped"] += same_rest; cnt["sig_equal"] += don["sig"] == it["sig"]
            pdon = S.person_swap(it); cnt["person_identity"] += pdon["T"] == it["T"] and pdon["slot_idx"] == it["slot_idx"] and pdon["o_R"] == it["o_P"]
            conc = [c for c in it["concepts"]]
            ii = ident and len(frontier_diff(it, t_ii, conc)) > 0; cnt["ii_prime_exposed"] += ii; cnt["ii_exposed"] += ii; cnt["reset_exposed"] += ident
            cnt["ii_prime_exposed_all_t"] += ident and all(frontier_diff(it, t, conc) for t in range(3, D))
            L = [c for c in conc if c not in (it["o_R"], it["o_P"]) and min(it["dist_R"].get(c, 99), it["dist_P"].get(c, 99)) <= t_ii + 1]
            Lsz.append(len(L)); Ldf.append(len(frontier_diff(it, t_ii, L)))
            if t_i is not None:
                L = [c for c in conc if c not in (it["o_R"], it["o_P"]) and min(it["dist_R"].get(c, 99), it["dist_P"].get(c, 99)) <= t_i + 1]
                fdl = frontier_diff(it, t_i, L); outside = it["o_R"] not in L and it["o_P"] not in L
                cnt["i_prime_exposed"] += ident and len(fdl) > 0 and outside; Lsz_i.append(len(L)); Ldf_i.append(len(fdl))
                cnt["i_prime_exposed_all_t"] += ident and all(
                    frontier_diff(it, t, [c for c in conc if c not in (it["o_R"], it["o_P"]) and min(it["dist_R"].get(c, 99), it["dist_P"].get(c, 99)) <= t + 1])
                    for t in range(3, D - 1))
        n = N_BAND
        res[D] = {"n_pairs": n, "maker_fail": cnt["maker_fail"], "fail_examples": fails[:5], "identity": cnt["identity"],
                  "identity_share": cnt["identity"] / n, "donor_answer_oP": cnt["donor_answer_oP"], "only_person_subjects_swapped": cnt["only_person_subjects_swapped"],
                  "sig_equal": cnt["sig_equal"], "person_swap_identity": cnt["person_identity"],
                  "exposure": {"ii_prime": {"admissible_t": [3, D - 1], "n_candidates": n, "n_exposed": cnt["ii_prime_exposed"], "n_exposed_every_t": cnt["ii_prime_exposed_all_t"]},
                               "i_prime": ({"admissible_t": [3, D - 2], "n_candidates": n, "n_exposed": cnt["i_prime_exposed"], "n_exposed_every_t": cnt["i_prime_exposed_all_t"]}
                                           if D >= 5 else {"admissible_t": None, "n_candidates": n, "n_exposed": 0, "note": "D 4 excluded by construction"}),
                               "ii": {"n_candidates": n, "n_exposed": cnt["ii_exposed"]}, "person": {"n_candidates": n, "n_exposed": cnt["person_identity"]},
                               "reset": {"n_candidates": n, "n_exposed": cnt["reset_exposed"]}},
                  "L_t_at_ii_t": {"mean_size": round(statistics.mean(Lsz), 1), "mean_differing": round(statistics.mean(Ldf), 1)},
                  "L_t_at_i_t": ({"mean_size": round(statistics.mean(Lsz_i), 1), "mean_differing": round(statistics.mean(Ldf_i), 1)} if Lsz_i else None)}
        res[D]["pass"] = (cnt["maker_fail"] == 0 and cnt["identity"] == n and cnt["donor_answer_oP"] == n and cnt["only_person_subjects_swapped"] == n
                          and cnt["sig_equal"] == n and cnt["person_identity"] == n)
        print("g_l4 D", D, {k: v for k, v in res[D].items() if k not in ("exposure", "fail_examples")}, res[D]["exposure"], flush=True)
    dump("g_l4.json", {"spec": "LANG_v1 §7 G-L4: PREMISE-SWAP on 5,000 pairs per D 4..8 (recipients = the test stream 22200: the 1,000 test items + 4,000 more); "
                       "t ~ U{3..D-1} for (ii') / (ii) / RESET and U{3..D-2} for (i'), rng 22610", "pass": all(r["pass"] for r in res.values()), "by_D": res})


# ---------- G-L4b CF pair maker ----------
def cmd_g_l4b():
    ref = json.load(open(os.path.join(DOCS, "v1_checks", "v1_cf.json")))["exposure"]; st = S.Stream(S.SEED_TEST); res = {}; worst = 0; bad = collections.Counter()
    items_all = []
    for D in range(4, 9):
        items = [st.draw(D) for _ in range(1000)]; items_all += items; cnt = collections.Counter()
        for it in items:
            nm = it["names"]; cn = [nm[c] for c in it["concepts"]]; fr, Tr = first_mentions(it["text"], cn); e_r = parse(it["text"])[0]
            for j in range(1, D + 1):
                r = S.cf_donor(it, j)
                if r is None:
                    cnt["none", j] += 1; continue
                don = r["donor"]; cnt["n", j] += 1
                e_d, p_d, q_d, _ = parse(don["text"]); f = layers(e_d, q_d[0], 10 ** 6)
                flip = f.get(nm[it["o_P"]]) == D and nm[it["o_R"]] not in f
                fd, Td = first_mentions(don["text"], cn); same = Td == Tr and fd == fr
                src = nm[it["R"]] if j == 1 else nm[it["levels"][j - 2][0]]
                one = [i for i, (a, b) in enumerate(zip(e_r, e_d)) if a != b]
                bad["flip_mismatch"] += flip != r["flip"]; bad["position_mismatch"] += same != (r["same_len"] and r["same_slots"])
                bad["not_one_fact"] += not (len(one) == 1 and e_r[one[0]] == (src, nm[it["levels"][j - 1][0]]) and e_d[one[0]] == (src, nm[it["levels"][j - 1][1]]))
                bad["question_changed"] += q_d != parse(it["text"])[2]
                if r["exposed"]:
                    bad["exposed_identity_fail"] += not (Td == Tr == it["T"] and fd == fr and don["slot_idx"] == it["slot_idx"])
                cnt["flip", j] += r["flip"]; cnt["exposed", j] += r["exposed"]
        cells = {}
        for j in range(1, D + 1):
            rv = ref[str(D)]["single"][str(j)]; mine = {"n": cnt["n", j], "flip": cnt["flip", j], "exposed": cnt["exposed", j], "none": cnt["none", j]}
            dpts = 100.0 * (mine["exposed"] / 1000 - rv["exposed"] / rv["n"]); worst = max(worst, abs(dpts))
            cells[j] = {"mine": mine, "v1_cf": rv, "exposed_diff_points": round(dpts, 1), "within_5": abs(dpts) <= 5.0,
                        "decisive": 3 <= j <= D - 1, "se_diff_points": round(100 * (2 * 0.25 / 1000) ** 0.5, 1)}
        res[D] = cells
        print("g_l4b D", D, {j: (c["mine"]["exposed"], c["v1_cf"]["exposed"], c["exposed_diff_points"]) for j, c in cells.items()}, dict(bad), flush=True)
    sel = S.cf_select(items_all, js=lambda D: range(3, D)); selc = {"%d_%d" % k: len(v) for k, v in sel.items()}
    within = all(c["within_5"] for cells in res.values() for c in cells.values()); within_dec = all(c["within_5"] for cells in res.values() for c in cells.values() if c["decisive"])
    dump("g_l4b_pairs.json", {"spec": "LANG_v1 §7 G-L4b (pair-maker part): CF donor maker on the 1,000 test items per D 4..8 (seed 22200); flip re-derived by "
                              "layer expansion on x' parsed from text; T and first-mention slots from token byte offsets; exposure vs v1_cf.json (seed 24300) within +-5 points",
                              "checks": dict(bad), "checks_pass": sum(bad.values()) == 0, "within_5_all_cells": within, "within_5_decisive_cells": within_dec,
                              "max_abs_diff_points": round(worst, 1), "cf_select_up_to_400_decisive": selc, "by_D": res,
                              "pass": sum(bad.values()) == 0 and within,
                              "note": "the version-0 switch hook part of G-L4b is model-side and not run here; exact replication of v1_cf.json at its own seed is in parity.json"})


if __name__ == "__main__":
    c = sys.argv[1]
    {"sigs": cmd_sigs, "parity": cmd_parity, "parity_band": lambda: cmd_parity(True), "merge": cmd_merge, "g_l3": cmd_g_l3, "g_l4": cmd_g_l4, "g_l4b": cmd_g_l4b}.get(c, lambda: None)() if c != "g_l0" else cmd_g_l0(sys.argv[2])
