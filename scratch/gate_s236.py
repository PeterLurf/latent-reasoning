"""§236 REG-CHAIN build gates (RC_v1 §1 / §6 + Note 236-1 Audit conditions; for Audit's G-D). -> scratch/gate_s236/<part>.json ("pass"). 4060 (cuda:0) / CPU.
References kept: boxes/_ow_words_s235.py, _inject_ow_s235.py, _inject_aw_s235.py, _inject_aw_eval_s235.py (the modules before the mul_s_p edits, b03d4168..dd465e69).
  gen       flags-off generator: OW.Stream at the default AND explicit mul_s_p 1 vs _ow_words_s235 (OWKW, 10k draws per d 1..12, EXCL): items / rng state / stats
            equal; OW.test_set(**OWKW) and OW.pair_candidates(**OWKW) at default and mul_s_p 1 equal the old module's (sigs, per d); build_sets() == build_sets(1) ==
            aw_plaw4_s1/args.json's set shas + pair_stats + excl_n.
  train     flags-off training: _inject_aw_s235.py vs the new inject_aw.py at the P-4 args (LAT-AW, CD-AN), det wrap, fp32, micro 8, sched 0.02, stop_at 30: every
            ckpt field torch.equal, logs equal.
  evalq     flags-off eval: _inject_aw_eval_s235 vs the new eval, the 20-step P-4 model (scratch/_tmp/s234/p4/ev_lat_aw4), all 11 parts at eval_bs 250, TEST2 cut 250:
            every eval_<part>.json equal minus sec / module_sha256 / def.
  regsets   zero mul_s in every REG set: TEST2 / TRANSFER d 1..12, pairs (rec + don) / floors, the legacy exclusion sets (TE6 / PC6 / ED7), swap_k d 2, swap_kd d 3 / 4
            (+ d 3 t 2), probe, calibration (64, SEED_CAL) and memcheck (256 at d 4) items; the REG seeds unused before §236 (grep of boxes / scratch / theory / docs
            .py at build time, recorded); the REG set shas differ from the default-law ones.
  disjoint  Note 236-1 (1): the eval-set sha table + disjointness from the ACTUAL training draws: inject_aw's training EXCL (build_sets(0)["excl"] | reg_excl) and the
            training law (Stream(SEED_TRAIN, EXCL, mul_s_p 0), d ~ default_rng(seed + SEED_D) U{1..4}, bs 64) replayed for all 16,000 steps at seed 1 and seed 2:
            0 sigs in common with every eval set (per set, per seed).
  mi        Note 236-1 (2): from the first 50k seed-1 training draws: the (t, d) op table and the plug-in I(op_t; (t, d)) over passes t >= 2 at d >= 2 (expect ~0.0003
            bits) and over all (t, d) (expect ~0.049 bits); 0 mul_s.
  gradflow  fp64 CPU, perturbed, gamma 1, 4 REG d 3 + 4 REG d 4 items: answer CE and aux CE each nonzero on 148 GPT-2 + 12 memory tensors.
  smoke     REG LAT-AW / CD-AN (P-4 args + --mul_s_p 0) 50 steps (stop_at 50, micro 8) -> rc 9; LAT-AW bin ws_readout d 1..4; the run's args carry the REG shas + set_law.
  evreg     every part (11) at eval_bs 250 on a 20-step REG LAT-AW model (TEST2-REG cut 250 per d): files; swap_kd by_d 3 has swap_mid_t2 with n_pairs_t2 > 0, d 4 t 2;
            swap_k / probe ran on the REG streams (sigs == reg_eval_sigs); CD-AN acc at eval_bs 100.
  refreg    the reference driver (scratch/s236_reg_eval.py) on a copy of the 20-step P-4 model, FULL REG sets, eval_bs 100 (250 OOMs on 8 GB: REG items are longer): ws, acc, cue_only, swap_kd -> files; <out>/args.json REG shas; the
            source dir untouched (file list + mtimes).
  args      --args_only rc_plaw4_s1 / rc_plaw4_s2 / rc_pcd4_s1 vs scratch/s234_args_frozen_aw_plaw4_s1.json / _aw_pcd4_s1.json: the diff = {mul_s_p} + seed (s2) + the
            derived REG set keys (set shas, set_law, pair_stats, excl_n) + mul_s_p3 None -> 1.0 (the §235 flag at default) + inject_ow_sha256 (the module edit).
    python -B scratch/gate_s236.py gen regsets args mi train evalq gradflow smoke evreg refreg disjoint"""
import json
import math
import os
import shutil
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch")
import gate_s228 as G8  # noqa: E402
import gate_s234 as G4  # noqa: E402
import gate_s234_p4 as P4  # noqa: E402

OUT = ROOT + "/scratch/gate_s236"; TMP = ROOT + "/scratch/_tmp/s236"; DEV = "cuda:0"
P4_EVAL = P4.P4_EVAL; LAT4, CD4 = P4.LAT4, P4.CD4; REG = ["--mul_s_p", "0"]
RUN4 = ROOT + "/boxes/runs_fsq/aw_plaw4_s1"
SETKEYS = ["test2_sha", "transfer_sha", "pairs2_sha", "floors2_sha", "s226_test_sha", "s227_decode_sha"]
REG_SEEDS = [23800, 23801, 23802, 23803, 23804, 23810, 23811, 23812, 23823, 23824, 23833, 23834, 23843, 23853]


def reg_args(seed=1, arm="lat"):
    """vars() of the REG run's args as inject_aw builds them (only the keys reg_eval_sigs / tok_aw read)."""
    if arm == "lat":
        return {"d_max": 4, "aux_target": "value", "cue": "step", "write_src": "answer", "mul_s_p": 0, "seed": seed}
    return {"d_max": 4, "aux_target": "value", "cue": "all", "write_src": "all", "mul_s_p": 0, "seed": seed}


def gen():
    import boxes._ow_words_s235 as O0
    import boxes.inject_ow as IO
    import boxes.ow_words as OW
    S = IO.build_sets(); ex = S["excl"]; o = {"stream": {}}; ok = True
    for d in range(1, 13):
        for tag, kw in (("default", {}), ("explicit_1", {"mul_s_p": 1})):
            a = O0.Stream(23900 + d, ex, frames=O0.FRAME_TRAIN, **IO.OWKW); b = OW.Stream(23900 + d, ex, frames=OW.FRAME_TRAIN, **IO.OWKW, **kw)
            A = [a.draw(d) for _ in range(10000)]; B = [b.draw(d) for _ in range(10000)]
            r = [x["text"] for x in A] == [x["text"] for x in B] and a.rng.bit_generator.state == b.rng.bit_generator.state and a.stats == b.stats
            o["stream"]["%d_%s" % (d, tag)] = r; ok = ok and r
    for nm in ("test_set", "pair_candidates"):
        f0, f1 = getattr(O0, nm), getattr(OW, nm); r0 = f0(**IO.OWKW)
        for tag, kw in (("default", {}), ("explicit_1", {"mul_s_p": 1})):
            r1 = f1(**IO.OWKW, **kw)
            if nm == "test_set":
                eq = all([x["text"] for x in r0[d]] == [x["text"] for x in r1[d]] for d in r0) and sorted(r0) == sorted(r1)
            else:
                eq = all([(p[0]["text"], p[1]["text"]) for p in r0[d]] == [(p[0]["text"], p[1]["text"]) for p in r1[d]] for d in r0) and sorted(r0) == sorted(r1)
            o["%s_%s" % (nm, tag)] = eq; ok = ok and eq
    ref = json.load(open(RUN4 + "/args.json")); S1 = IO.build_sets(mul_s_p=1)
    o["build_sets_shas_equal_P4"] = all(S["shas"][k] == ref[k] == S1["shas"][k] for k in SETKEYS) and "set_law" not in S["shas"]
    o["pair_stats_equal_P4"] = S["pair_stats"] == ref["pair_stats"] == S1["pair_stats"]; o["excl_n"] = [len(S["excl"]), len(S1["excl"]), ref.get("excl_n")]
    o["pass"] = ok and o["build_sets_shas_equal_P4"] and o["pair_stats_equal_P4"] and len(S["excl"]) == len(S1["excl"]) == ref.get("excl_n"); return o


def regsets():
    import subprocess
    import boxes.inject_ow as IO
    import boxes.inject_aw_eval as AE
    import boxes.ow_words as OW
    S = IO.build_sets(mul_s_p=0); D = IO.build_sets(); a = reg_args(); has = lambda its: sum("mul_s" in it["ops"] for it in its)  # noqa: E731
    o = {"n_mul_s": {}}
    o["n_mul_s"]["test2"] = sum(has(v) for v in S["test2"].values()); o["n_mul_s"]["transfer"] = sum(has(v) for v in S["transfer"].values())
    o["n_mul_s"]["pairs"] = sum(has([p_["rec"], p_["don"]]) for ps in S["pairs"].values() for p_ in ps)
    o["n_mul_s"]["floors"] = sum(has([x for x in row if x is not None]) for fl in S["floors"].values() for row in fl)
    TE = OW.test_set(**IO.OWKW, mul_s_p=0); PC = OW.pair_candidates(**IO.OWKW, mul_s_p=0)
    o["n_mul_s"]["TE6_reg"] = sum(has(v) for v in TE.values()); o["n_mul_s"]["PC6_reg"] = sum(has(list(p)) for v in PC.values() for p in v)
    _, _, kp, _ = AE.swap_pairs(a); o["n_mul_s"]["swap_k_d2"] = sum(has(list(p[:2])) for p in kp)
    for d in (3, 4):
        _, _, kp, _, t = AE.swap_pairs_d(a, d); o["n_mul_s"]["swap_kd_d%d" % d] = sum(has(list(p[:2])) for p in kp); o["swap_kd_d%d_n" % d] = [len(kp), t]
    _, _, kp, tr, _ = AE.swap_pairs_d(a, 3, 2, (AE.SEED_SWAPD2_REC_REG + 3, AE.SEED_SWAPD2_DON_REG + 3)); o["n_mul_s"]["swap_kd_d3_t2"] = sum(has(list(p[:2])) for p in kp)
    o["swap_kd_d3_t2_n"] = [len(kp), tr]
    s = OW.Stream(AE.SEED_PROBE_REG, S["excl"], frames=OW.FRAME_TRAIN, **IO.OWKW, mul_s_p=0); o["n_mul_s"]["probe"] = has([s.draw(2) for _ in range(AE.PROBE_N)])
    s = OW.Stream(IO.SEED_CAL, frames=OW.FRAME_TRAIN, mul_s_p=0, **IO.OWKW); o["n_mul_s"]["calibration"] = has([s.draw(1 + i % 4) for i in range(64)])
    s = OW.Stream(IO.SEED_MEMCHECK, frames=OW.FRAME_TRAIN, mul_s_p=0, **IO.OWKW); o["n_mul_s"]["memcheck_d4"] = has([s.draw(4) for _ in range(256)])
    o["default_law_has_mul_s_test2_d3plus"] = sum(has(D["test2"][d]) for d in range(3, 13))
    o["reg_shas"] = {k: S["shas"][k] for k in SETKEYS}; o["reg_shas_differ_from_default"] = all(S["shas"][k] != D["shas"][k] for k in SETKEYS)
    o["set_law"] = S["shas"].get("set_law")
    o["seed_grep_before_build"] = ("the build-time grep (boxes / scratch / theory / docs/research_* .py, 10-27, before any §236 edit) found none of "
                                   + " ".join(str(x) for x in REG_SEEDS))
    o["pass"] = all(v == 0 for v in o["n_mul_s"].values()) and o["reg_shas_differ_from_default"] and o["set_law"] == "reg" and o["default_law_has_mul_s_test2_d3plus"] > 0
    return o


def train_replay(seed, steps=16000, dmax=4):
    import numpy as np
    import boxes.inject_ow as IO
    import boxes.inject_aw as AW
    import boxes.ow_words as OW
    import argparse
    S = IO.build_sets(mul_s_p=0); ns = argparse.Namespace(**reg_args(seed)); ex = S["excl"] | AW.reg_excl(ns, S["excl"])
    st = OW.Stream(OW.SEED_TRAIN, ex, frames=OW.FRAME_TRAIN, mul_s_p3=1.0, mul_s_p=0, **IO.OWKW); rd = np.random.default_rng(seed + IO.SEED_D)
    for _ in range(steps):
        d = int(rd.integers(1, dmax + 1))
        for _ in range(64):
            yield st.draw(d)


def eval_sets():
    import boxes.inject_ow as IO
    import boxes.inject_aw as AW  # noqa: F401
    import boxes.inject_aw_eval as AE
    S = IO.build_sets(mul_s_p=0); sets = {"test2_d%d" % d: {it["sig"] for it in S["test2"][d]} for d in IO.TEST_DS}
    sets.update({"transfer_d%d" % d: {it["sig"] for it in S["transfer"][d]} for d in IO.TEST_DS})
    sets["pairs2"] = {x["sig"] for ps in S["pairs"].values() for p_ in ps for x in (p_["rec"], p_["don"])}
    sets["floors2"] = {x["sig"] for fl in S["floors"].values() for row in fl for x in row if x is not None}
    sets.update(AE.reg_eval_sigs(reg_args(), S["excl"])); return S, sets


def disjoint():
    import boxes.ow_words as OW
    S, sets = eval_sets(); o = {"sha_table": {**{k: S["shas"][k] for k in SETKEYS}, "set_law": S["shas"].get("set_law")}, "set_sizes": {k: len(v) for k, v in sets.items()}}
    for k in ("swap_k_d2", "swap_kd_d3", "swap_kd_d4", "swap_kd_d3_t2", "probe"):
        o["sha_table"][k + "_sigsha"] = __import__("hashlib").sha256("\n".join(sorted(sets[k])).encode()).hexdigest()
    allsig = set().union(*sets.values()); o["overlap"] = {}; o["n_draws"] = {}; o["n_mul_s"] = {}
    for seed in (1, 2):
        seen = set(); n = 0; nm = 0
        for it in train_replay(seed):
            n += 1; nm += "mul_s" in it["ops"]
            if it["sig"] in allsig:
                seen.add(it["sig"])
        o["overlap"][str(seed)] = {k: len(seen & v) for k, v in sets.items()}; o["n_draws"][str(seed)] = n; o["n_mul_s"][str(seed)] = nm
    o["pass"] = all(v == 0 for s_ in o["overlap"].values() for v in s_.values()) and all(v == 16000 * 64 for v in o["n_draws"].values()) and all(v == 0 for v in o["n_mul_s"].values())
    return o


def mi():
    import collections
    tab = collections.Counter(); n = 0; nm = 0
    for it in train_replay(1):
        n += 1; nm += "mul_s" in it["ops"]
        for t, op in enumerate(it["ops"], 1):
            tab[(t, it["d"], op)] += 1
        if n >= 50000:
            break

    def info(keep):
        c = {k: v for k, v in tab.items() if keep(k[0], k[1])}; N = sum(c.values()); po = collections.Counter(); px = collections.Counter()
        for (t, d, op), v in c.items():
            po[op] += v; px[(t, d)] += v
        return sum(v / N * math.log2((v / N) / ((po[op] / N) * (px[(t, d)] / N))) for (t, d, op), v in c.items()), N
    i2, n2 = info(lambda t, d: t >= 2 and d >= 2); ia, na = info(lambda t, d: True)
    table = {"%d,%d" % (t, d): {} for (t, d, _) in tab}
    for (t, d, op), v in sorted(tab.items()):
        table["%d,%d" % (t, d)][op] = v
    return {"n_draws": n, "n_mul_s": nm, "I_bits_t2plus_d2plus": i2, "n_obs_t2plus_d2plus": n2, "I_bits_all": ia, "n_obs_all": na, "op_table": table,
            "expected": "~0.0003 bits (t >= 2, d >= 2), ~0.049 bits (all), RC_v1 §1", "pass": n == 50000 and nm == 0 and i2 < 0.005}


def train():
    o = {}
    for tag, fl in (("lat_aw4", LAT4), ("cd_an4", CD4)):
        com = fl + ["--fp32", "--micro", "8", "--sched_scale", "0.02", "--ckpt_every", "30", "--log_every", "10", "--stop_at", "30"]
        da, db = TMP + "/fo_s235mod_" + tag, TMP + "/fo_newmod_" + tag
        ra = G4.det("boxes/_inject_aw_s235.py", com, da); rb = G4.det("boxes/inject_aw.py", com, db); r = {"rc": [ra, rb]}
        if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
            r.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=("model", "opt", "rng_d", "np_rng", "torch_rng", "cuda_rng", "stream_rng", "stream_stats", "expo",
                                                                                     "binacc", "vbin", "vbin_d", "gn_log", "wg_hist")))
            strip = lambda p: [{k: v for k, v in json.loads(l_).items() if k != "sec"} for l_ in open(p)]  # noqa: E731
            r["log_equal"] = strip(da + "/log.jsonl") == strip(db + "/log.jsonl")
        r["pass"] = r["rc"] == [9, 9] and r.get("n_mismatch") == 0 and bool(r.get("log_equal")); o[tag] = r
    o["pass"] = all(v["pass"] for v in o.values()); return o


def evalq():
    import torch
    import boxes.inject_ow as IO
    import boxes._inject_aw_eval_s235 as E0
    import boxes.inject_aw_eval as E1
    src = ROOT + "/scratch/_tmp/s234/p4/ev_lat_aw4"; da, db = G8.fresh(TMP + "/eqev_s235"), G8.fresh(TMP + "/eqev_new")
    for d_ in (da, db):
        for f_ in ("args.json", "model.pt", "train_summary.json"):
            shutil.copy(os.path.join(src, f_), os.path.join(d_, f_))
    S = IO.build_sets(); S["test2"] = {k: v[:250] for k, v in S["test2"].items()}; torch.zeros(1, device=DEV)
    parts = P4_EVAL.split(","); E0.run(da, parts, DEV, 250, sets=S); E1.run(db, parts, DEV, 250, sets=S); res = {}; defs = {}
    for p_ in parts:
        ja, jb = (json.load(open(os.path.join(x, "eval_%s.json" % p_))) for x in (da, db))
        if ja.get("def") != jb.get("def"):
            defs[p_] = True
        for j_ in (ja, jb):
            j_.pop("sec", None); j_.pop("module_sha256", None); j_.pop("def", None)
        res[p_] = ja == jb
    return {"parts": res, "def_changed": sorted(defs), "pass": len(res) == 11 and all(res.values()) and set(defs) <= {"swap_kd"}}


def gradflow():
    with G8.F64():
        import torch
        import boxes.inject_ow as IO
        import boxes.inject_aw as AW
        import boxes.ow_words as OW
        torch.set_num_threads(2); m, g = G4.aw_model(torch.float64); s = OW.Stream(23960, frames=OW.FRAME_TRAIN, mul_s_p=0, **IO.OWKW)
        its = [AW.tok_aw(s.draw(d), "value", "step", "answer") for d in (3, 3, 3, 3, 4, 4, 4, 4)]; o = {"n_mul_s": sum("mul_s" in it["raw"]["ops"] for it in its)}
        for term in ("answer_ce", "aux_ce"):
            m.zero_grad(set_to_none=True); la, aux, _ = G4.aw_loss(m, its, 8, True); (la if term == "answer_ce" else aux).backward()
            gr = {n: (float(p.grad.norm()) if p.grad is not None else None) for n, p in m.named_parameters()}
            okf = lambda v: v is not None and v > 0 and math.isfinite(v)  # noqa: E731
            o[term] = {"gpt2_nonzero": sum(okf(v) for n, v in gr.items() if n.startswith("g.")), "mem_nonzero": sum(okf(v) for n, v in gr.items() if n.startswith("mem.")),
                       "zero_or_none": sorted(n for n, v in gr.items() if not okf(v))[:20]}
        o["pass"] = o["n_mul_s"] == 0 and all(o[t]["gpt2_nonzero"] == 148 and o[t]["mem_nonzero"] == 12 for t in ("answer_ce", "aux_ce")); return o


def smoke():
    o = {}
    for tag, fl in (("lat_aw4", LAT4 + REG), ("cd_an4", CD4 + REG)):
        d = G8.fresh(TMP + "/smoke_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--micro", "8", "--stop_at", "50", "--ckpt_every", "0", "--log_every", "10",
                                                                                                          "--device", DEV, "--out", d], d + "/stdout.log")
        o[tag] = {"rc": rc}
        if os.path.exists(d + "/log.jsonl"):
            br = [json.loads(l_) for l_ in open(d + "/log.jsonl") if '"by_d"' in l_][-1]; ar = json.load(open(d + "/args.json"))
            o[tag].update({"ws_ds": sorted(br.get("ws_readout", {})), "set_law": ar.get("set_law"), "mul_s_p": ar.get("mul_s_p"), "excl_n": ar.get("excl_n"),
                           "test2_sha": ar.get("test2_sha")})
    la, cd = o["lat_aw4"], o["cd_an4"]; ref = json.load(open(RUN4 + "/args.json"))
    o["pass"] = (la["rc"] == 9 and cd["rc"] == 9 and la.get("ws_ds") == ["1", "2", "3", "4"] and la.get("set_law") == "reg" == cd.get("set_law") and la.get("mul_s_p") == 0
                 and la.get("test2_sha") != ref["test2_sha"] and la.get("excl_n", 0) > ref["excl_n"]); return o


def evreg():
    import torch
    import boxes.inject_ow as IO
    import boxes.inject_aw_eval as AE
    o = {}
    for tag, fl, ebs, parts in (("lat_aw4", LAT4 + REG, 250, P4_EVAL.split(",")), ("cd_an4", CD4 + REG, 100, ["acc"])):
        d = G8.fresh(TMP + "/ev_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--steps", "20", "--micro", "8", "--eval_bs", str(ebs), "--eval_parts", "",
                                                                                                         "--ckpt_every", "0", "--log_every", "10", "--device", DEV, "--out", d], d + "/stdout.log")
        S = IO.build_sets(mul_s_p=0); S["test2"] = {k: v[:250] for k, v in S["test2"].items()}; torch.zeros(1, device=DEV); torch.cuda.reset_peak_memory_stats(DEV)
        AE.run(d, parts, DEV, ebs, sets=S); pk = torch.cuda.max_memory_allocated(DEV) / 2 ** 30
        fs = sorted(f_ for f_ in os.listdir(d) if f_.startswith("eval")); o[tag] = {"rc": rc, "files": fs, "peak_gib": pk}
        if tag == "lat_aw4":
            J = {p_: json.load(open(os.path.join(d, "eval_%s.json" % p_))) for p_ in ("swap_kd", "swap_k", "probe")}; rs = AE.reg_eval_sigs(reg_args(), S["excl"])
            kd = J["swap_kd"]["by_d"]
            o[tag].update({"swap_kd_3": {k: kd["3"].get(k) for k in ("n_pairs", "t_mid", "n_pairs_t2", "tried_t2")}, "swap_mid_t2_ks": sorted(kd["3"].get("swap_mid_t2", {})),
                           "swap_kd_4": {k: kd["4"].get(k) for k in ("n_pairs", "t_mid")}, "items_3_t2": "t2" in J["swap_kd"]["items"]["3"],
                           "swap_k_sigs_reg": set(J["swap_k"]["items"]["rec_sig"]) | set(J["swap_k"]["items"]["don_sig"]) == rs["swap_k_d2"],
                           "probe_sigs_reg": set(J["probe"]["items"]["sig"]) == rs["probe"]})
    la = o["lat_aw4"]
    o["pass"] = (la["rc"] == 0 and o["cd_an4"]["rc"] == 0 and set(la["files"]) == {"eval.json"} | {"eval_%s.json" % p_ for p_ in P4_EVAL.split(",")}
                 and o["cd_an4"]["files"] == ["eval.json", "eval_acc.json"] and la["swap_kd_3"]["n_pairs"] == 1000 and la["swap_kd_3"]["t_mid"] == 1
                 and (la["swap_kd_3"]["n_pairs_t2"] or 0) > 0 and la["swap_mid_t2_ks"] == ["3", "8"] and la["swap_kd_4"]["t_mid"] == 2 and la["items_3_t2"]
                 and la["swap_k_sigs_reg"] and la["probe_sigs_reg"] and la["peak_gib"] < 7.5)
    return o


def refreg():
    src = ROOT + "/scratch/_tmp/s234/p4/ev_lat_aw4"; snap = {f_: os.path.getmtime(os.path.join(src, f_)) for f_ in os.listdir(src)}; out = G8.fresh(TMP + "/refreg_out")
    shutil.rmtree(out)
    rc = G8.sh([sys.executable, "-B", "scratch/s236_reg_eval.py", "--run", "ev_lat_aw4", "--src_dir", src, "--out", out, "--parts", "ws,acc,cue_only,swap_kd", "--eval_bs", "100",
                "--device", DEV], TMP + "/refreg.log")
    import boxes.inject_ow as IO
    S = IO.build_sets(mul_s_p=0); fs = sorted(f_ for f_ in os.listdir(out)) if os.path.isdir(out) else []; ar = json.load(open(out + "/args.json")) if "args.json" in fs else {}
    o = {"rc": rc, "files": fs, "args_reg_shas": all(ar.get(k) == S["shas"][k] for k in SETKEYS) and ar.get("mul_s_p") == 0 and ar.get("reg_of") == "ev_lat_aw4",
         "src_untouched": {f_: os.path.getmtime(os.path.join(src, f_)) for f_ in os.listdir(src)} == snap}
    o["pass"] = rc == 0 and {"eval.json", "eval_ws.json", "eval_acc.json", "eval_cue_only.json", "eval_swap_kd.json", "model.pt", "args.json"} <= set(fs) and o["args_reg_shas"] and o["src_untouched"]
    return o


def args_part():
    import s228_args_check as C
    o = {}
    der = {"mul_s_p", "mul_s_p3", "set_law", "pair_stats", "excl_n", "inject_ow_sha256"} | set(SETKEYS)   # inject_ow_sha256: §236 edits inject_ow (build_sets)
    for tag, fl, ref, extra in (("rc_plaw4_s1", LAT4 + REG + ["--auto_micro", "--ckpt_every", "1000", "--eval_parts", P4_EVAL], "scratch/s234_args_frozen_aw_plaw4_s1.json", set()),
                                ("rc_plaw4_s2", LAT4 + REG + ["--seed", "2", "--auto_micro", "--ckpt_every", "1000", "--eval_parts", P4_EVAL], "scratch/s234_args_frozen_aw_plaw4_s1.json", {"seed"}),
                                ("rc_pcd4_s1", CD4 + REG + ["--auto_micro", "--ckpt_every", "1000", "--eval_parts", "acc"], "scratch/s234_args_frozen_aw_pcd4_s1.json", set())):
        d = G8.fresh(TMP + "/args_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
        dif = C.diff(json.load(open(ROOT + "/" + ref)), json.load(open(d + "/args.json")))
        o[tag] = {"rc": rc, "diff_keys": sorted(dif), "diff": {k: v for k, v in dif.items() if k not in ("pair_stats",)}, "args_file": d + "/args.json",
                  "pass": rc == 0 and set(dif) == der | extra and dif["mul_s_p"][1] == 0}
    o["pass"] = all(v["pass"] for v in o.values()); return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    fns = {"gen": gen, "regsets": regsets, "disjoint": disjoint, "mi": mi, "train": train, "evalq": evalq, "gradflow": gradflow, "smoke": smoke, "evreg": evreg,
           "refreg": refreg, "args": args_part}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
