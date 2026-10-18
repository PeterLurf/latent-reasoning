"""§218 LANG-VB labeller gates (LANG_v1 §7) for boxes/langvb_label.py; each writes scratch/gate_s218/<name>.json. CPU only.
  sound   G-L3 (S-L' fuzz half): 100k SYM v1 items, D U{9..12} (+ 1,000 per D 3..16 for coverage), marks from 6 regimes incl. injected unsound
          marks; per trial M'_t subset of gold R_t (generator BFS), KEEP -> label == o_R and s >= D, no tie; gold / all-marks -> s == D and M' == R
          (maximality). Also: the text parse == the generator item (slots, concepts, statements) and an independent BFS on the parsed statements ==
          prosqa_sym.reach_targets. Pure Python, no model.                                                       -> g_l3_sound.json
  replay  G-L20 helper: the trainer's rng_s order replayed for seed 1 == 5,992 self steps / 383,488 records, by two independent replays
          (langvb_label.replay_self_demand and a cursor walk written from langvb.main's draw_step); seeds 2..21; g_l20_check fixtures.
                                                                                                                    -> g_l20_replay.json
  smoke   L-ROLL / L-RR / GATE-1-L on 8 GATE items with an UNTRAINED LangSLVA (CPU fp32); both chains with a text-parsing ORACLE model (BFS on its own
          input, so the re-render / chunk / frontier logic is checked end to end: s == D, M' == R_t, rollout counts) and a NOISY oracle (false marks
          incl. o_P at every pass: no wrong label); the pool builder (oracle, 52 draws, resume parity) into scratch/_tmp/label_smoke, assemble,
          load_pool / self_item round-trip; lvb_loss / backward on self items (and a plain + self mix). Files under _tmp deleted at the end.
                                                                                                                    -> g_label_smoke.json
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 19 python -B scratch/gate_s218_label.py sound|replay|smoke"""
import collections
import json
import math
import os
import random
import shutil
import sys
import time

sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.langvb_label as LL  # noqa: E402
import boxes.prosqa_sym as PS  # noqa: E402

OUT = "scratch/gate_s218"; TMP = "scratch/_tmp/label_smoke"
torch.set_num_threads(2)


def dump(name, obj):
    with open(os.path.join(OUT, name), "w", newline="\n") as f:
        json.dump(obj, f, indent=1)
    print("wrote", name, flush=True)


def bfs_layers(P):
    """independent of the module: first-entry time per name by layer expansion from R over the parsed statements."""
    succ = collections.defaultdict(set)
    for a, b in P["stmts"]:
        succ[a].add(b)
    f = {P["R"]: 0}; cur = {P["R"]}; t = 0
    while cur:
        t += 1; new = {v for u in cur for v in succ[u]} - set(f)
        for v in new:
            f[v] = t
        cur = new
    return f


# ---------------------------------------------------------------------- G-L3 S-L' fuzz
def trial_marks(regime, rng, cs, f, fP, D, oR, oP, K=16):
    """marks M_1..M_K (sets of names) for one regime; f / fP = gold first-entry from R / P (absent = unreachable)."""
    unreach = [c for c in cs if c not in f]
    if regime == "gold":
        return [{c for c in cs if c in f and f[c] <= t} for t in range(1, K + 1)]
    if regime == "all":
        return [set(cs) for _ in range(K)]
    if regime == "random":
        p = rng.uniform(0.02, 0.6); return [{c for c in cs if rng.random() < p} for _ in range(K)]
    if regime == "gold_inject":
        M = []
        for t in range(1, K + 1):
            s = {c for c in cs if c in f and f[c] <= t}
            s |= {c for c in unreach if rng.random() < 0.3}                       # unreachable (incl. o_P's world)
            s |= {c for c in cs if c in f and t < f[c] <= t + 3 and rng.random() < 0.5}   # over-reach
            if rng.random() < 0.5:
                s.add(oP)
            if t < D and rng.random() < 0.5:
                s.add(oR)                                                       # early option
            M.append(s)
        return M
    if regime == "lag_omit":
        ent = {c: f[c] + rng.randint(0, 3) for c in cs if c in f and rng.random() > 0.2}
        M = []
        for t in range(1, K + 1):
            s = {c for c, e in ent.items() if e <= t and rng.random() < 0.7}
            s |= {c for c in cs if rng.random() < 0.05}
            if rng.random() < 0.3:
                s.add(oP)
            M.append(s)
        return M
    if regime == "foil_world":                                                  # the wrong person's REACH + noise (o_P reached at D)
        return [{c for c in cs if c in fP and fP[c] <= t} | {c for c in cs if rng.random() < 0.05} for t in range(1, K + 1)]
    raise ValueError(regime)


REGIMES = ("gold", "all", "random", "gold_inject", "lag_omit", "foil_world")


def cmd_sound(n_main=100000, n_cov=1000):
    t0 = time.time(); rng_it = random.Random(21890); rng_D = random.Random(21891); rng_m = random.Random(21893); rng_cov = random.Random(21892)
    sets = [("main", [rng_D.randint(9, 12) for _ in range(n_main)], rng_it), ("coverage", [D for D in range(3, 17) for _ in range(n_cov)], rng_cov)]
    viol = collections.Counter(); ex = []; kept = collections.Counter(); trials = collections.Counter(); lag = collections.defaultdict(collections.Counter)
    parse_diff = 0; reach_diff = 0; maximal_fail = 0; nD = collections.Counter(); ties = 0
    for name, Ds, rng in sets:
        for k, D in enumerate(Ds):
            it = PS.make_item(rng, D); nD[(name, D)] += 1; nm = it["names"]
            u = LL.parse_ids(it["ids"]); cs = u["concepts"]; P = u["P"]
            li_cs = sorted(it["concepts"], key=lambda c: it["slot_idx"][c])
            if (u["slot_idx"] != [it["slot_idx"][c] for c in li_cs] or cs != [nm[c] for c in li_cs] or P["R"] != nm[it["R"]]
                    or sorted(map(tuple, P["stmts"])) != sorted((nm[a], nm[b]) for a, b in it["edges"])
                    or set(P["options"]) != {nm[it["o_R"]], nm[it["o_P"]]}):
                parse_diff += 1
            fb = bfs_layers(P); E = PS.reach_targets(it, 24)
            if any(E[t][i] != (1 if (t > 0 and nm[c] in fb and fb[nm[c]] <= t) else 0) for t in range(25) for i, c in enumerate(it["concepts"])):
                reach_diff += 1
            f = {nm[c]: d for c, d in it["dist_R"].items() if not nm[c][:1].isupper()}
            fP = {nm[c]: d for c, d in it["dist_P"].items() if not nm[c][:1].isupper()}
            oR, oP = nm[it["o_R"]], nm[it["o_P"]]; tabs = LL._closure_tables(P)
            for reg in REGIMES:
                M = trial_marks(reg, rng_m, cs, f, fP, D, oR, oP); sc = LL.sound_closure(P, M, 16, tabs); trials[reg] += 1
                bad = []
                for t, s_ in enumerate(sc["Mp"]):
                    if any(c not in f or f[c] > t for c in s_):
                        bad.append("M'_%d not in R_%d" % (t, t)); break
                if sc["tie"]:
                    bad.append("tie"); ties += 1
                if sc["keep"]:
                    kept[reg] += 1; lag[reg][sc["stop_s"] - D] += 1
                    if sc["label"] != oR:
                        bad.append("label != o_R")
                    if sc["stop_s"] < D:
                        bad.append("s < D")
                if reg in ("gold", "all"):
                    ok = sc["keep"] and sc["stop_s"] == D and all(sc["Mp"][t] == {c for c in cs if c in f and f[c] <= t} for t in range(D + 1))
                    maximal_fail += not ok
                for b in bad:
                    viol[(reg, b)] += 1
                    if len(ex) < 20:
                        ex.append({"set": name, "k": k, "D": D, "regime": reg, "violation": b})
            if (k + 1) % 20000 == 0:
                print(name, k + 1, "violations", sum(viol.values()), "%.0f s" % (time.time() - t0), flush=True)
    n_viol = sum(viol.values())
    out = {"spec": "LANG_v1 §7 G-L3, S-L' fuzz half (random marks incl. injected unsound ones; M' subset of R_t, label o_R, s >= D) + parse / BFS checks",
           "n_items": {"main_D9_12": n_main, "coverage_D3_16": 14 * n_cov}, "items_by_D": {"%s_%d" % k: v for k, v in sorted(nD.items())},
           "seeds": {"items_main": 21890, "D_main": 21891, "items_coverage": 21892, "marks": 21893, "note": "unregistered fuzz seeds, plain make_item draws"},
           "regimes": list(REGIMES), "trials": dict(trials), "violations": n_viol, "violations_by": {"%s | %s" % k: v for k, v in viol.items()}, "examples": ex,
           "ties": ties, "kept_by_regime": dict(kept), "stop_minus_D_hist": {r: dict(sorted(v.items())) for r, v in lag.items()},
           "maximality_fail_gold_or_all": maximal_fail, "parse_diff_items": parse_diff, "reach_bfs_diff_items": reach_diff,
           "pass": n_viol == 0 and maximal_fail == 0 and parse_diff == 0 and reach_diff == 0, "sec": round(time.time() - t0, 1)}
    dump("g_l3_sound.json", out)


# ---------------------------------------------------------------------- G-L20 replay
def cursor_walk(seed, steps=12000, p_self=0.5, bs=64, self_kextra=4, pool_len=4 * LL.KEPT_PER_D):
    """written from langvb.main's draw_step (plain batch first via rng / rng_k, not modelled; then one rng_s coin; on a self step the exhaustion
    check, cursor += bs, then one U per record)."""
    rng_s = np.random.default_rng(seed + 21800); cursor = 0; n_self = 0; exhausted = None; Us = collections.Counter()
    for step in range(1, steps + 1):
        coin = bool(rng_s.random() < p_self)
        if not coin:
            continue
        if cursor + bs > pool_len:
            exhausted = step; break
        cursor += bs; n_self += 1
        for _ in range(bs):
            Us[int(rng_s.integers(0, self_kextra + 1))] += 1
    return {"n_self_steps": n_self, "cursor": cursor, "exhausted_at": exhausted, "U_hist": dict(sorted(Us.items()))}


def cmd_replay():
    t0 = time.time(); r1 = LL.replay_self_demand(1); w1 = cursor_walk(1)
    others = {s: LL.replay_self_demand(s)["self_steps"] for s in range(2, 22)}
    v1 = json.load(open("docs/research_lang_2026-10-18/v1_checks/v1_pool.json"))
    per_D = {D: 0 for D in LL.POOL_D}
    for j in range(w1["cursor"]):                                             # interleaved pool: record j is band 9 + j % 4
        per_D[LL.POOL_D[j % 4]] += 1
    good = {"kept_by_D": {str(D): LL.KEPT_PER_D for D in LL.POOL_D}, "n_records": 4 * LL.KEPT_PER_D, "POOL_SHORT": False, "rng_replay_equal": True,
            "demand_replay": 383488, "key_set": sorted(LL.REC_KEYS), "sig_hits": {"test": 0, "gate": 0, "probe": 0}, "gold_key_hits": 0, "label_bug": False}
    bad = {"short": {"kept_by_D": {**good["kept_by_D"], "12": 105999}, "n_records": 4 * LL.KEPT_PER_D - 1, "POOL_SHORT": True},
           "sig_hit": {"sig_hits": {"test": 0, "gate": 1, "probe": 0}}, "gold_key": {"gold_key_hits": 1}, "tie": {"label_bug": True},
           "replay": {"rng_replay_equal": False}, "keys": {"key_set": sorted(LL.REC_KEYS | {"dist_R"})}}
    fx = {"good_meta": LL.g_l20_check(good)["pass"], **{"bad_" + k: LL.g_l20_check({**good, **v})["pass"] for k, v in bad.items()}}
    ok = (r1["self_steps"] == 5992 and r1["records"] == 383488 and w1["n_self_steps"] == 5992 and w1["cursor"] == 383488 and w1["exhausted_at"] is None
          and min(others.values()) == v1["seeds2_21_self_steps"]["min"] and max(others.values()) == v1["seeds2_21_self_steps"]["max"]
          and fx["good_meta"] and not any(v for k, v in fx.items() if k.startswith("bad_")) and all(v == 95872 for v in per_D.values()))
    dump("g_l20_replay.json", {"spec": "LANG_v1 §7 G-L20 (replay part): seed 1, 12,000 steps, bs 64, p_self 0.5, self_kextra 4; rng_s = default_rng(seed + 21800)",
                               "replay_self_demand_seed1": r1, "cursor_walk_seed1": w1, "spec_value": {"self_steps": 5992, "records": 383488},
                               "consumed_per_D_interleaved": {str(k): v for k, v in per_D.items()}, "pool_spare_per_D": LL.KEPT_PER_D - 95872,
                               "pool_total_over_demand": round(4 * LL.KEPT_PER_D / r1["records"], 4), "seeds2_21_self_steps": others,
                               "v1_pool_json_seeds2_21": v1["seeds2_21_self_steps"], "g_l20_check_fixtures": fx, "pass": ok, "sec": round(time.time() - t0, 1)})


# ---------------------------------------------------------------------- smoke
class Oracle(torch.nn.Module):
    """a text-parsing stand-in for LangSLVA: V_t[j] = 1 iff the slot's concept is within t hops of R in ITS OWN input (parsed back from x);
    noise > 0 adds random false marks and the unreachable option at every pass."""

    def __init__(self, noise=0.0, seed=0):
        super().__init__(); self.register_buffer("c_anneal", torch.tensor(0.0)); self.noise = noise; self.rng = random.Random(seed); self.n_calls = 0

    def rollout_ls(self, x, pad, si, sp, K, **kw):
        B, S = si.shape; dist = torch.full((B, S), 10 ** 6); extra = torch.zeros(B, S, dtype=torch.bool); self.n_calls += B
        for b in range(B):
            u = LL.parse_ids(x[b, :int(pad[b].sum())].tolist()); n = len(u["concepts"])
            assert si[b, :n].tolist() == u["slot_idx"] and int(sp[b].sum()) == n
            f = bfs_layers(u["P"])
            for j, c in enumerate(u["concepts"]):
                dist[b, j] = f.get(c, 10 ** 6)
            if self.noise:
                for o in u["P"]["options"]:
                    if o not in f:
                        extra[b, u["ord"][o]] = True
        V = [torch.zeros(B, S, 1)]
        for t in range(1, K + 1):
            v = dist <= t
            if self.noise:
                v = v | extra | (torch.rand(B, S, generator=torch.Generator().manual_seed(self.rng.randint(0, 2 ** 31))) < self.noise)
            V.append((v & sp).float()[..., None])
        return {"V": V}

    def rec_logits(self, V):
        return torch.cat([-V, V], -1)


def _layers_gold(it):
    nm = it["names"]; lay = collections.Counter(d for c, d in it["dist_R"].items() if not nm[c][:1].isupper())
    return lay


def cmd_smoke():
    import boxes.langvb as LV
    from boxes import s5
    t0 = time.time(); out = {"spec": "LANG_v1 §2.4 / §4.1 labeller smoke (CPU fp32): untrained LangSLVA + oracle models; pool builder; loader; lvb_loss"}
    ac = LL.make_ac("cpu"); fails = []
    gate = PS.gate_set(n_per_D=2, D_range=range(9, 13)); n_gate = sum(len(v) for v in gate.values())
    # (1) untrained LangSLVA
    torch.manual_seed(0); m = LV.LangSLVA(s5.load_gpt2("cpu")).to("cpu"); m.eval()
    utag = {"model": None, "model_sha256": "state:" + LL.state_sha256(m), "card": "cpu"}
    try:                                                                    # bf16 required by default: an fp32 ac must be refused
        LL.gate1_l(m, "cpu", ac, {9: gate[9][:1]}, bs=4, tag=utag); bf16_refused = False
    except AssertionError:
        bf16_refused = True
    g1 = LL.gate1_l(m, "cpu", ac, gate, bs=4, tag=utag, require_bf16=False)
    shp = []
    for c in LL.CHAINS:
        for D, lst in g1["items"][c].items():
            for r in lst:
                S = len(r["gold_dist"]); n_t = len(r["marks_hex"])
                ok = all(int(h, 16) < (1 << S) for h in r["marks_hex"]) and len(r["frontier"]) == n_t and isinstance(r["keep"], bool)
                ok = ok and ((n_t == 16) if c == "L-ROLL" else (1 <= n_t <= 16)) and (not r["keep"] or (len(r["marks_sound"]) == r["stop"] and r["label"] == r["gold"]))
                shp.append(ok)
    wrong = sum(g1["chains"][c][str(D)]["n_wrong_label"] for c in LL.CHAINS for D in range(9, 13))
    out["untrained"] = {"n_items": n_gate, "meta": g1["meta"], "shapes_ok": all(shp), "n_wrong_label": wrong, "label_bug": g1["label_bug"],
                        "bf16_refused_by_default": bf16_refused, "json_safe": json.loads(json.dumps(g1)) == g1,
                        "per_chain_D": {c: {D: {k: v for k, v in g1["chains"][c][D].items() if k in ("n", "kept", "yield", "pi_label", "false_in", "frontier_hist",
                                                                                              "rollouts_mean", "n_raw_marks", "n_tie", "n_unsound")} for D in g1["chains"][c]}
                                        for c in LL.CHAINS},
                        "choice": g1["choice"], "pass": g1["pass"], "sec": g1["sec"], "kept_vs_drawn_keys": sorted(g1["kept_vs_drawn"]["9"])}
    if not all(shp) or wrong or g1["label_bug"] or not bf16_refused or not out["untrained"]["json_safe"]:
        fails.append("untrained shapes / wrong label / bf16 / json")
    # (2) oracle: exact chains; rollout counts
    orc = Oracle(); items = [it for D in (9, 10, 11, 12) for it in PS.test_set(6, [D], seed=21894)[D]] + [it for v in gate.values() for it in v]
    inps = [LL.parse_ids(it["ids"]) for it in items]; ores = {}
    for c in LL.CHAINS:
        res = LL.run_chain(c, orc, inps, "cpu", ac, bs=8); bad = []
        for it, u, r in zip(items, inps, res):
            nm = it["names"]; f = {nm[k]: d for k, d in it["dist_R"].items() if not nm[k][:1].isupper()}; D = it["D"]; lay = _layers_gold(it)
            Mp = [set()]
            for a in r["Mp_add"]:
                Mp.append(Mp[-1] | {u["concepts"][j] for j in a})
            if not (r["keep"] and r["stop_s"] == D and r["label"] == nm[it["o_R"]]):
                bad.append("keep/stop/label")
            elif any(Mp[t] != {x for x in f if f[x] <= t} for t in range(D + 1)):
                bad.append("M' != R_t")
            if c == "L-RR":
                exp = 1 + sum(math.ceil(lay[t] / 4) for t in range(2, D))
                if r["n_rollouts"] != exp:
                    bad.append("n_rollouts %d != %d" % (r["n_rollouts"], exp))
                if r["frontier"] != [lay[t] for t in range(1, D + 1)]:
                    bad.append("frontier sizes")
        ores[c] = {"n": len(items), "bad": bad[:10], "n_bad": len(bad), "max_layer_width": max(max(_layers_gold(it).values()) for it in items),
                   "n_items_layer_gt4": sum(max(_layers_gold(it).values()) > 4 for it in items),
                   "rollouts_mean": float(np.mean([r["n_rollouts"] for r in res])), "model_rows": orc.n_calls}
        if bad:
            fails.append("oracle %s" % c)
    out["oracle_exact"] = ores
    # (3) noisy oracle: false marks everywhere incl. o_P at every pass -> never a wrong label
    noisy = Oracle(noise=0.1, seed=3); nz = {}
    for c in LL.CHAINS:
        res = LL.run_chain(c, noisy, inps, "cpu", ac, bs=8)
        wr = sum(r["keep"] and r["label"] != it["names"][it["o_R"]] for it, r in zip(items, res)); early = sum(r["keep"] and r["stop_s"] < it["D"] for it, r in zip(items, res))
        nz[c] = {"kept": sum(r["keep"] for r in res), "n": len(res), "wrong_label": int(wr), "stop_lt_D": int(early)}
        if wr or early:
            fails.append("noisy oracle %s" % c)
    out["oracle_noisy"] = nz
    # (4) pool builder (oracle chain L-RR, 13 draws per D = 52), resume parity, assemble, load_pool, self_item
    if os.path.isdir(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP); t1 = time.time(); excl = PS.exclusion_sigs(); sec_excl = time.time() - t1
    cached = json.load(open(os.path.join(OUT, "excl_sigs.json")))["sigs"]
    out["exclusion"] = {"n": len(excl), "equals_cached_excl_sigs_json": excl == set(cached), "sec": round(sec_excl, 1)}
    logs = []; otag = {"model": "oracle", "model_sha256": "state:" + LL.state_sha256(orc), "card": "cpu"}
    og1 = LL.gate1_l(orc, "cpu", ac, gate, bs=8, tag=otag, require_bf16=False, log=logs.append); g1p = os.path.join(TMP, "eval_gate1_oracle.json")
    LL._jdump(og1, g1p); ch = og1["choice"]; other = [c for c in LL.CHAINS if c != ch][0]
    kw = dict(kept_target=1000, chunk=5, bs=8, excl=excl, require_bf16=False, log=logs.append)
    neg = {}

    def refused(f, exc=AssertionError):
        try:
            f(); return False
        except exc:
            return True
    neg["wrong_chain"] = refused(lambda: LL.build_pool_part(orc, "cpu", ac, TMP, 9, other, g1p, otag, max_draws=1, **kw))
    neg["wrong_model"] = refused(lambda: LL.build_pool_part(orc, "cpu", ac, TMP, 9, ch, g1p, {**otag, "model_sha256": "x"}, max_draws=1, **kw))
    neg["fp32_without_flag"] = refused(lambda: LL.build_pool_part(orc, "cpu", ac, TMP, 9, ch, g1p, otag, max_draws=1,
                                                                 **{k: v for k, v in kw.items() if k != "require_bf16"}))
    for D in LL.POOL_D:
        LL.build_pool_part(orc, "cpu", ac, TMP, D, ch, g1p, otag, max_draws=13, **kw)
    dA = os.path.join(TMP, "resume"); os.makedirs(dA)                         # resume parity on D 9: 6 then 7 draws vs 13 in one go
    LL.build_pool_part(orc, "cpu", ac, dA, 9, ch, g1p, otag, max_draws=6, **kw)
    with open(os.path.join(dA, "part_D9.jsonl"), "ab") as f:                  # simulate a crash mid-write: junk past the recorded byte length
        f.write(b'{"partial":')
    neg["resume_other_bs"] = refused(lambda: LL.build_pool_part(orc, "cpu", ac, dA, 9, ch, g1p, otag, max_draws=7, **{**kw, "bs": 4}))
    neg["resume_other_chunk"] = refused(lambda: LL.build_pool_part(orc, "cpu", ac, dA, 9, ch, g1p, otag, max_draws=7, **{**kw, "chunk": 6}))
    neg["resume_other_card"] = refused(lambda: LL.build_pool_part(orc, "cpu", ac, dA, 9, ch, g1p, {**otag, "card": "NVIDIA GeForce RTX 3070"}, max_draws=7, **kw))
    open(os.path.join(dA, "part_D9.lock"), "w").write("held by test")
    neg["lock_held"] = refused(lambda: LL.build_pool_part(orc, "cpu", ac, dA, 9, ch, g1p, otag, max_draws=7, **kw), RuntimeError)
    os.remove(os.path.join(dA, "part_D9.lock"))
    LL.build_pool_part(orc, "cpu", ac, dA, 9, ch, g1p, otag, max_draws=7, **kw)
    same = open(os.path.join(dA, "part_D9.jsonl"), "rb").read() == open(os.path.join(TMP, "part_D9.jsonl"), "rb").read()
    st9 = json.load(open(os.path.join(dA, "part_D9.state.json")))
    neg["no_lock_left"] = not any(f_.endswith(".lock") for d_ in (TMP, dA) for f_ in os.listdir(d_))
    out["negative_tests"] = neg
    if not all(neg.values()):
        fails.append("negative tests %s" % neg)
    meta = LL.assemble_pool(TMP, g1p, LL.POOL_D, ch, kept_target=1000, require_done=False, probe_scale=0.002)
    out["state_D9_resumed"] = {k: st9[k] for k in ("bs", "chunk", "card", "dtype", "kept_target", "draw_cap", "n_tie", "drawn", "kept")}
    out["state_D9_resumed"]["n_resumes"] = len(st9["resumes"])
    t2 = time.time(); recs, meta2 = LL.load_pool(os.path.join(TMP, "pool.jsonl"), os.path.join(TMP, "pool_meta.json"), verify=True, compact=True); sec_load = time.time() - t2
    raw_lines = open(os.path.join(TMP, "pool.jsonl"), "rb").read().splitlines()
    keys_ok = all(set(json.loads(ln)) == LL.REC_KEYS for ln in raw_lines); crlf = b"\r" in open(os.path.join(TMP, "pool.jsonl"), "rb").read()
    # gold re-check of every record: the pool stream's kept items in order per band (Stream(22300) with the exclusion, the same draws)
    st = PS.Stream(PS.SEED_POOL, excl); gold_by_D = {D: [st.draw(D) for _ in range(13)] for D in LL.POOL_D}
    kept_gold = {D: [] for D in LL.POOL_D}
    for D in LL.POOL_D:
        for it in gold_by_D[D]:
            if any(it["sig"] == s_ for s_ in open(os.path.join(TMP, "part_D%d.sigs" % D)).read().split()):
                kept_gold[D].append(it)
    order = []; cur = {D: 0 for D in LL.POOL_D}
    while any(cur[D] < len(kept_gold[D]) for D in LL.POOL_D):
        for D in LL.POOL_D:
            if cur[D] < len(kept_gold[D]):
                order.append(kept_gold[D][cur[D]]); cur[D] += 1
    rt_bad = []
    for j, (rec, it) in enumerate(zip(recs, order)):
        si = LL.self_item(rec, 16); li = LV.lvb_item(it); nm = it["names"]
        if si["ids"] != li["ids"] or si["slot_idx"] != li["slot_idx"] or si["concepts"] != [nm[c] for c in li["concepts"]] or si["o_R"] != li["o_R"] or si["o_P"] != li["o_P"]:
            rt_bad.append((j, "fields"))
        if si["marks"] != [li["dist_R"].get(c) for c in li["concepts"]] or si["stop_s"] != it["D"]:
            rt_bad.append((j, "marks / stop vs gold"))
    sigs_pool = open(os.path.join(TMP, "pool_sigs.txt")).read().split()
    out["pool"] = {"meta": {k: v for k, v in meta.items() if k not in ("collisions", "builder_commit")}, "builder_commit_head": meta["builder_commit"]["head"],
                   "builder_commit_dirty": meta["builder_commit"]["dirty"], "collisions": meta["collisions"], "n_records": len(recs),
                   "keys_exact": keys_ok, "crlf": crlf, "resume_parity_D9": same, "load_sec": round(sec_load, 3),
                   "load_sec_per_record": sec_load / max(1, len(recs)), "roundtrip_bad": rt_bad, "n_order": len(order),
                   "sigs_disjoint_from_exclusion": not (set(sigs_pool) & excl), "g_l20_check_smoke_meta": LL.g_l20_check(meta2), "build_log": logs[:12],
                   "pool_bytes": os.path.getsize(os.path.join(TMP, "pool.jsonl")), "bytes_per_record": os.path.getsize(os.path.join(TMP, "pool.jsonl")) / max(1, len(recs))}
    if not (keys_ok and not crlf and same and not rt_bad and len(recs) == len(order) and len(recs) > 0):
        fails.append("pool / round-trip")
    # memory per record, compact vs plain (projection for 424k records)
    import tracemalloc
    for mode in (False, True):
        tracemalloc.start(); rr, _ = LL.load_pool(os.path.join(TMP, "pool.jsonl"), os.path.join(TMP, "pool_meta.json"), verify=False, compact=mode)
        cur_b, _ = tracemalloc.get_traced_memory(); tracemalloc.stop()
        out["pool"]["mem_bytes_per_record_compact_%s" % mode] = cur_b / max(1, len(rr)); out["pool"]["mem_GB_projected_424k_compact_%s" % mode] = round(cur_b / max(1, len(rr)) * 424000 / 1e9, 2)
        del rr
    # (5) the untrained pool builder: see cmd_e2e (scratch/_tmp/label_e2e) for the untrained-model pool, its timing and labeller_part
    # (6) lvb_loss on self items (+ a plain / self mix), backward
    sis = [LL.self_item(recs[j], 16) for j in range(2)]; rng_u = np.random.default_rng(1); Ks = [min(s_["stop_s"] + int(rng_u.integers(0, 5)), 16) for s_ in sis]
    m.train(); m.c_anneal.fill_(0.0); m.zero_grad(set_to_none=True)
    L, stt, _ = LV.lvb_loss(m, sis, Ks, "cpu", ac); L.backward()
    gr = {n_: float(p_.grad.norm()) for n_, p_ in m.named_parameters() if p_.grad is not None and n_ in ("rec.weight", "slva_wq")}
    E = LV.reach_targets(sis, max(Ks), max(len(s_["slot_idx"]) for s_ in sis), "cpu")
    e_ok = all(int(E[b, t, j]) == (1 if (s_["marks"][j] is not None and t >= s_["marks"][j]) else -100)
               for b, s_ in enumerate(sis) for t in range(max(Ks) + 1) for j in range(len(s_["slot_idx"])))
    m.zero_grad(set_to_none=True); plain = LV.lvb_item(PS.test_set(1, [5], seed=21895)[5][0])
    L2, st2, _ = LV.lvb_loss(m, [plain, sis[0]], [7, Ks[0]], "cpu", ac)
    out["lvb_loss"] = {"Ks": Ks, "stop_s": [s_["stop_s"] for s_ in sis], "loss": float(L.detach()), "stats": stt, "finite": math.isfinite(float(L.detach())), "grad_norms": gr,
                       "reach_targets_marks_ok": e_ok, "mix_plain_self": {"loss": float(L2.detach()), "stats": st2}}
    if not (math.isfinite(float(L.detach())) and math.isfinite(float(L2.detach())) and e_ok and stt["plain_cand_ce"] is None and st2["n_plain"] == 1 and all(v > 0 for v in gr.values())):
        fails.append("lvb_loss")
    m.zero_grad(set_to_none=True)
    shutil.rmtree(TMP)
    out["tmp_deleted"] = not os.path.exists(TMP); out["fails"] = fails; out["pass_smoke"] = not fails; out["pass"] = not fails
    out["sec"] = round(time.time() - t0, 1)
    dump("g_label_smoke.json", out)


E2E = "scratch/_tmp/label_e2e"


def _scorer():
    """theory/score_s218.py's pure readers (gate1_read, pool_read, p_labeller), imported read-only from the file; its main() is never run."""
    import importlib.util
    sp = importlib.util.spec_from_file_location("score_s218_ro", "theory/score_s218.py"); mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
    return mod


def cmd_e2e():
    """end-to-end on CPU with the untrained LangSLVA: gate1 (4 items per D) -> pool (build + assemble, ~40 records) -> labeller_part (n_audit /
    n_memo shrunk), in runs_fsq-style folders under scratch/_tmp/label_e2e (kept, for pointing the scorer at); readers of score_s218 applied."""
    import boxes.langvb as LV
    from boxes import s5
    t0 = time.time(); ac = LL.make_ac("cpu"); fails = []; res = {}
    if os.path.isdir(E2E):
        shutil.rmtree(E2E)
    dg, dp, da = (os.path.join(E2E, d_) for d_ in ("lvb_gate1_s1", "lvb_pool_s1", "lvb_self_s1"))
    for d_ in (dg, dp, da):
        os.makedirs(d_)
    torch.manual_seed(0); m = LV.LangSLVA(s5.load_gpt2("cpu")).to("cpu"); m.eval()
    tag = {"model": "untrained LangSLVA (torch seed 0, c 1e4)", "model_sha256": "state:" + LL.state_sha256(m), "card": "cpu"}
    gate = PS.gate_set(n_per_D=4, D_range=range(9, 13))
    g1 = LL.gate1_l(m, "cpu", ac, gate, bs=4, chunk=8, tag=tag, require_bf16=False); g1p = os.path.join(dg, "eval_gate1.json"); LL._jdump(g1, g1p)
    g1r = json.load(open(g1p)); kept_items = [x for c in LL.CHAINS for D in g1r["items"][c] for x in g1r["items"][c][D] if x["keep"]]
    lab_eq = all(isinstance(x["label"], str) and isinstance(x["gold"], str) and x["label"] == x["gold"] for x in kept_items)
    res["gate1"] = {"choice": g1r["choice"], "meta": {k: g1r["meta"][k] for k in ("card", "dtype", "bs", "chunk", "model_sha256")}, "label_bug": g1r["label_bug"],
                    "pass": g1r["pass"], "kept_by_chain_D": {c: {D: v["kept"] for D, v in g1r["chains"][c].items()} for c in LL.CHAINS},
                    "n_kept_items": len(kept_items), "kept_label_eq_gold_strings": lab_eq, "sec": g1r["sec_total"]}
    if not (lab_eq and len(kept_items) > 0 and g1r["label_bug"] is False):
        fails.append("gate1 label / gold")
    ch = g1r["choice"]; excl = PS.exclusion_sigs(); logs = []
    for D in LL.POOL_D:
        LL.build_pool_part(m, "cpu", ac, dp, D, ch, g1p, tag, kept_target=10, draw_cap=40, chunk=8, bs=4, excl=excl, require_bf16=False,
                           require_gate_pass=False, log=logs.append)
    meta = LL.assemble_pool(dp, g1p, LL.POOL_D, ch, kept_target=10, draw_cap=40, require_gate_pass=False, probe_scale=0.01)
    recs, _ = LL.load_pool(os.path.join(dp, "pool.jsonl"), os.path.join(dp, "pool_meta.json"))
    res["pool"] = {k: meta[k] for k in ("n_records", "kept_by_D", "draws_by_D", "total_kept", "demand_replay", "rng_replay_equal", "key_set", "sig_hits",
                                        "probe_draws_by_D", "gold_key_hits", "n_dup_sigs", "n_tie", "label_bug", "card_by_band", "bs", "chunk", "POOL_SHORT")}
    res["pool"]["builder_commit"] = {k: meta["builder_commit"][k] for k in ("head", "dirty", "dirty_files", "consistent")}
    res["pool"]["sec_per_drawn_cpu_fp32"] = {D: v["sec_per_drawn"] for D, v in meta["counts"].items()}; res["pool"]["build_log_tail"] = logs[-4:]
    if not (meta["n_records"] == len(recs) > 0 and meta["sig_hits"] == {"test": 0, "gate": 0, "probe": 0} and meta["gold_key_hits"] == 0
            and meta["rng_replay_equal"] and meta["demand_replay"] == 383488 and meta["key_set"] == sorted(LL.REC_KEYS) and meta["n_tie"] == 0):
        fails.append("pool meta")
    cursor = min(32, len(recs)); n_memo = 3
    lp = LL.labeller_part(m, "cpu", os.path.join(dp, "pool.jsonl"), os.path.join(dp, "pool_meta.json"), g1p, cursor, eval_bs=4, n_audit=20, n_memo=n_memo,
                          gate_items={D: v[:n_memo] for D, v in gate.items()}, card="cpu", log=logs.append)
    ev = {**lp, "card": "cpu", "dtype": "fp32"}; LL._jdump(ev, os.path.join(da, "eval_lvb.json"))
    la = lp["label_audit"]
    res["labeller_part"] = {"cursor": cursor, "label_audit": {k: la[k] for k in ("n", "n_wrong", "pi_label", "n_unsound", "n_no_gold")},
                            "memo": {D: {k: v[k] for k in ("acc_pool", "acc_fresh", "n_pool", "n_fresh", "K")} for D, v in lp["memo"].items()},
                            "gate1_copy_choice": lp["gate1"]["choice"], "meta": lp["meta"], "sec": lp["sec"]}
    if not (la["n"] == min(20, cursor) and la["n_wrong"] == 0 and la["n_unsound"] == 0 and lp["gate1"]["choice"] == ch):
        fails.append("labeller_part")
    try:                                                                    # the scorer's own readers on the three files (read-only)
        SC = _scorer()
        gr = SC.gate1_read(g1r); pr = SC.pool_read(json.load(open(os.path.join(dp, "pool_meta.json")))); pl = SC.p_labeller(json.load(open(os.path.join(da, "eval_lvb.json"))))
        res["scorer"] = {"gate1_read": {k: gr.get(k) for k in ("missing", "problems", "choice", "LABEL_BUG", "n_wrong", "yield_ok", "GATE_1_L", "why")},
                         "pool_read": {k: pr.get(k) for k in ("missing", "POOL_SHORT", "CONTAMINATED", "sig_hits", "gold_key_hits", "G_L20_fail")},
                         "p_labeller": {k: pl.get(k) for k in ("missing", "LABEL_BUG", "n_audit_flag", "gate1_copy_choice")}}
        if gr.get("missing") or pr.get("missing") or pl.get("missing") or gr.get("LABEL_BUG") or pr.get("CONTAMINATED") or pl.get("LABEL_BUG") \
                or gr.get("choice") != ch:
            fails.append("scorer readers")
    except Exception as e:                                                  # noqa: BLE001
        res["scorer"] = {"error": repr(e)}; fails.append("scorer import / read")
    res["files"] = sorted(os.path.relpath(os.path.join(r_, f_), E2E).replace(os.sep, "/") for r_, _, fs in os.walk(E2E) for f_ in fs)
    res["fails"] = fails; res["pass"] = not fails; res["sec"] = round(time.time() - t0, 1)
    sm = json.load(open(os.path.join(OUT, "g_label_smoke.json"))); sm["e2e"] = res
    sm["pass"] = bool(sm.get("pass_smoke", sm["pass"]) and res["pass"]); dump("g_label_smoke.json", sm)


if __name__ == "__main__":
    {"sound": cmd_sound, "replay": cmd_replay, "smoke": cmd_smoke, "e2e": cmd_e2e}[sys.argv[1]]()
