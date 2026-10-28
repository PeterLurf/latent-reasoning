"""§238 DAG-FREE build gates (DF_v1 §1-3 / §8, b14bb1c8; Note 238-1 Audit conditions; for Audit's G-D). -> scratch/gate_s238/<part>.json ("pass"). 4060 (cuda:0) / CPU.
References kept: boxes/_{ow_words,inject_ow,inject_aw,inject_aw_eval}_s237.py (the §237 modules, c94f77f7..e4aab61d).
  gen       flags-off generator at laws 1 and 0: OW.Stream (default, explicit 1, explicit 0) vs _ow_words_s237, 10k draws per d 1..12 (OWKW, EXCL): items / rng state
            / stats equal; test_set / pair_candidates at default and 0 equal; build_sets(1) shas == aw_plaw4_s1's, build_sets(0) shas == rcb_plaw4_s1's (+ excl_n).
  law       DF law: 50,000 draws (d ~ U{3..8}, Stream(24500 + 0), OWKW, mul_s_p 2): realised P(mul_s | t) for t 3..6 within 0.125 +- 0.01 (per t, pooled over d);
            the per-(t, d) table; 0 mul_s at t <= 2; every mul_s partner valid (1 <= j <= t-2); no ts.
  mi        Note 238-1: I(op; (t, d)) over t >= 3 from the first 50k ACTUAL training draws (seed 1; EXCL as inject_aw builds it), plug-in; expected < 0.001 bits.
  classes   n >= 50 per op class at pass 3 (TEST2-DF d 3-4 pooled, ops[2]) and at pass 4 (OPCLASS-DF, ops[3]; mul_s split by j = 1 / 2).
  disjoint  0 sig overlap between the ACTUAL training draws (16,000 x 64, seeds 1 and 2, EXCL as built) and every DF eval set (TEST2 / TRANSFER d 1..12, pairs, floors,
            swap_k d 2, swap_kd d 3 / 4, probe, opclass, swap_prod); 0 overlap with the §236 REG / default TEST2; the sha table with the law tag "df".
  raise     unknown laws raise: inject_aw_eval.law({"mul_s_p": 3}) ValueError; OW.Stream(..., mul_s_p=3) AssertionError; build_sets(3) AssertionError; argparse 3 rc 2.
  train     flags-off training: _inject_aw_s237.py vs the new inject_aw.py at the §237 rcb args (LAT + init, CD + init), det, fp32, micro 8, sched 0.02, stop_at 30:
            every listed ckpt field torch.equal, logs equal.
  evalq     flags-off eval: _inject_aw_eval_s237 vs new, all 11 P-4 parts at eval_bs 250 (TEST2 cut 250) on the 20-step P-4 model (default law) and on the 20-step §236
            REG model (REG sets): every eval_<part>.json equal minus sec / module_sha256 / def.
  strict    the §238 init load (build_aw at the DF args + --init_model rcb_*): keys / tensors equal to the file (LAT, CD).
  gradflow  fp64 CPU, perturbed, gamma 1: 4 DF d 3 + 4 DF d 4 items, >= 2 with a mul_s step: answer CE and aux CE nonzero on 148 GPT-2 + 12 memory tensors.
  smoke     DF LAT-AW / CD-AN (+ init) 50 steps (stop_at 50, micro 8) -> rc 9; args set_law "df", mul_s_p 2; LAT bin ws_readout d 1..4.
  evdf      all 13 parts at eval_bs 250 on a 20-step DF LAT model (TEST2-DF cut 250): files; opclass n 3,000 with pred_by_pass 1..8; swap_prod n_pairs 1,000 with
            the flat fields + by_k {"3","8"}; swap_k / probe / swap_kd on the DF streams; CD acc at eval_bs 100.
  refdf     the driver (--law 2) on a copy of the 20-step P-4 model: acc, ws, opclass, swap_prod, cue_only at eval_bs 100 -> files; args DF shas; source untouched.
  args      --args_only df_plaw4_s1 / s2 / df_pcd4_s1 vs s237 frozen rcb_*: diff == {mul_s_p 0 -> 2, init_model, init_model_sha, init_model_sha256} + derived {set shas,
            set_law, pair_stats, excl_n, inject_ow_sha256, eval_parts (+ opclass, swap_prod: the new DF parts)} (+ seed for s2).
    python -B scratch/gate_s238.py gen law raise classes args mi strict train evalq gradflow smoke evdf refdf disjoint"""
import argparse
import collections
import json
import math
import os
import shutil
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch")
import gate_s228 as G8  # noqa: E402
import gate_s234 as G4  # noqa: E402
import gate_s234_p4 as P4  # noqa: E402
import gate_s237 as G7  # noqa: E402

OUT = ROOT + "/scratch/gate_s238"; TMP = ROOT + "/scratch/_tmp/s238"; DEV = "cuda:0"
P4_EVAL = P4.P4_EVAL; DF_EVAL = P4_EVAL + ",opclass,swap_prod"; DF = ["--mul_s_p", "2"]; REG = ["--mul_s_p", "0"]
SETKEYS = ["test2_sha", "transfer_sha", "pairs2_sha", "floors2_sha", "s226_test_sha", "s227_decode_sha"]
INIT = {"lat1": "boxes/runs_fsq/rcb_plaw4_s1/model.pt", "lat2": "boxes/runs_fsq/rcb_plaw4_s2/model.pt", "cd": "boxes/runs_fsq/rcb_pcd4_s1/model.pt"}


def initf(k):
    return ["--init_model", INIT[k], "--init_model_sha", G7.sha(INIT[k])[:16]]


LATD, CDD = P4.LAT4 + DF + initf("lat1"), P4.CD4 + DF + initf("cd")


def df_args(seed=1, arm="lat"):
    if arm == "lat":
        return {"d_max": 4, "aux_target": "value", "cue": "step", "write_src": "answer", "mul_s_p": 2, "seed": seed}
    return {"d_max": 4, "aux_target": "value", "cue": "all", "write_src": "all", "mul_s_p": 2, "seed": seed}


def gen():
    import boxes._ow_words_s237 as O0
    import boxes.inject_ow as IO
    import boxes.ow_words as OW
    S = IO.build_sets(); ex = S["excl"]; o = {"stream": {}}; ok = True
    for d in range(1, 13):
        for tag, kw0, kw1 in (("default", {}, {}), ("explicit_1", {"mul_s_p": 1}, {"mul_s_p": 1}), ("law_0", {"mul_s_p": 0}, {"mul_s_p": 0})):
            a = O0.Stream(24600 + d, ex, frames=O0.FRAME_TRAIN, **IO.OWKW, **kw0); b = OW.Stream(24600 + d, ex, frames=OW.FRAME_TRAIN, **IO.OWKW, **kw1)
            A = [a.draw(d) for _ in range(10000)]; B = [b.draw(d) for _ in range(10000)]
            r = [x["text"] for x in A] == [x["text"] for x in B] and a.rng.bit_generator.state == b.rng.bit_generator.state and a.stats == b.stats
            o["stream"]["%d_%s" % (d, tag)] = r; ok = ok and r
    for nm in ("test_set", "pair_candidates"):
        for tag, kw in (("default", {}), ("law_0", {"mul_s_p": 0})):
            r0 = getattr(O0, nm)(**IO.OWKW, **kw); r1 = getattr(OW, nm)(**IO.OWKW, **kw)
            f = (lambda x: x["text"]) if nm == "test_set" else (lambda p: (p[0]["text"], p[1]["text"]))
            eq = sorted(r0) == sorted(r1) and all([f(x) for x in r0[d]] == [f(x) for x in r1[d]] for d in r0); o["%s_%s" % (nm, tag)] = eq; ok = ok and eq
    a1 = json.load(open(ROOT + "/boxes/runs_fsq/aw_plaw4_s1/args.json")); a0 = json.load(open(ROOT + "/boxes/runs_fsq/rcb_plaw4_s1/args.json")); S0 = IO.build_sets(mul_s_p=0)
    o["law1_shas_eq_P4"] = all(S["shas"][k] == a1[k] for k in SETKEYS) and len(S["excl"]) == a1["excl_n"]
    o["law0_shas_eq_rcb"] = all(S0["shas"][k] == a0[k] for k in SETKEYS) and S0["shas"].get("set_law") == a0.get("set_law") == "reg"
    o["pass"] = ok and o["law1_shas_eq_P4"] and o["law0_shas_eq_rcb"]; return o


def law():
    import numpy as np
    import boxes.inject_ow as IO
    import boxes.ow_words as OW
    s = OW.Stream(24500, frames=OW.FRAME_TRAIN, mul_s_p=2, **IO.OWKW); rd = np.random.default_rng(24501); c = collections.Counter(); n = collections.Counter()
    ct = collections.Counter(); nt = collections.Counter(); bad = 0; early = 0; ts = 0
    for _ in range(50000):
        d = int(rd.integers(3, 9)); it = s.draw(d); ts += it["mul_s_t"] is not None
        for t, (op, j) in enumerate(zip(it["ops"], it["src_j"]), 1):
            n[(t, d)] += 1; c[(t, d)] += op == "mul_s"
            if t >= 3:
                nt[t] += 1; ct[t] += op == "mul_s"
            if op == "mul_s":
                early += t <= 2; bad += not (j is not None and 1 <= j <= t - 2)
    per_t = {str(t): ct[t] / nt[t] for t in sorted(nt)}; table = {"%d,%d" % k: c[k] / n[k] for k in sorted(n) if k[0] >= 3}
    return {"n_draws": 50000, "p_mul_s_by_t": per_t, "p_mul_s_by_t_d": table, "mul_s_at_t_le_2": early, "bad_partner": bad, "items_with_ts": ts,
            "pass": all(abs(per_t[str(t)] - 0.125) <= 0.01 for t in range(3, 7)) and early == 0 and bad == 0 and ts == 0}


def train_replay(seed, steps=16000, dmax=4, arm="lat"):
    import numpy as np
    import boxes.inject_ow as IO
    import boxes.inject_aw as AW
    import boxes.ow_words as OW
    S = IO.build_sets(mul_s_p=2); ns = argparse.Namespace(**df_args(seed, arm)); ex = S["excl"] | AW.reg_excl(ns, S["excl"])
    st = OW.Stream(OW.SEED_TRAIN, ex, frames=OW.FRAME_TRAIN, mul_s_p3=1.0, mul_s_p=2, **IO.OWKW); rd = np.random.default_rng(seed + IO.SEED_D)
    for _ in range(steps):
        d = int(rd.integers(1, dmax + 1))
        for _ in range(64):
            yield st.draw(d)


def mi():
    tab = collections.Counter(); n = 0
    for it in train_replay(1):
        n += 1
        for t, op in enumerate(it["ops"], 1):
            tab[(t, it["d"], op)] += 1
        if n >= 50000:
            break

    def info(keep):
        cc = {k: v for k, v in tab.items() if keep(k[0], k[1])}; N = sum(cc.values()); po = collections.Counter(); px = collections.Counter()
        for (t, d, op), v in cc.items():
            po[op] += v; px[(t, d)] += v
        return sum(v / N * math.log2((v / N) / ((po[op] / N) * (px[(t, d)] / N))) for (t, d, op), v in cc.items()), N
    i3, n3 = info(lambda t, d: t >= 3); ia, na = info(lambda t, d: True)
    table = collections.defaultdict(dict)
    for (t, d, op), v in sorted(tab.items()):
        table["%d,%d" % (t, d)][op] = v
    return {"n_draws": n, "I_bits_t3plus": i3, "n_obs_t3plus": n3, "I_bits_all": ia, "op_table": table, "expected": "< 0.001 bits over t >= 3 (DF_v1 §1)",
            "pass": n == 50000 and i3 < 0.001}


def classes():
    import boxes.inject_ow as IO
    S = IO.build_sets(mul_s_p=2); p3 = collections.Counter(it["ops"][2] for d in (3, 4) for it in S["test2"][d])
    p4 = collections.Counter((it["ops"][3] + ("_j%d" % it["src_j"][3] if it["ops"][3] == "mul_s" else "")) for it in IO.opclass_set())
    return {"pass3_test2_df_d34": dict(p3), "pass4_opclass_df": dict(p4), "min_pass3": min(p3.values()), "min_pass4": min(p4.values()),
            "pass": min(p3.values()) >= 50 and min(p4.values()) >= 50 and "mul_s" in p3 and {"mul_s_j1", "mul_s_j2"} <= set(p4)}


def eval_sets():
    import boxes.inject_ow as IO
    import boxes.inject_aw_eval as AE
    S = IO.build_sets(mul_s_p=2); sets = {"test2_d%d" % d: {it["sig"] for it in S["test2"][d]} for d in IO.TEST_DS}
    sets.update({"transfer_d%d" % d: {it["sig"] for it in S["transfer"][d]} for d in IO.TEST_DS})
    sets["pairs2"] = {x["sig"] for ps in S["pairs"].values() for p_ in ps for x in (p_["rec"], p_["don"])}
    sets["floors2"] = {x["sig"] for fl in S["floors"].values() for row in fl for x in row if x is not None}
    sets.update(AE.law_eval_sigs(df_args(), S["excl"])); return S, sets


def disjoint():
    import hashlib
    import boxes.inject_ow as IO
    S, sets = eval_sets(); o = {"sha_table": {**{k: S["shas"][k] for k in SETKEYS}, "set_law": S["shas"].get("set_law")}, "set_sizes": {k: len(v) for k, v in sets.items()}}
    for k in ("swap_k_d2", "swap_kd_d3", "swap_kd_d4", "probe", "opclass", "swap_prod"):
        o["sha_table"][k + "_sigsha"] = hashlib.sha256("\n".join(sorted(sets[k])).encode()).hexdigest()
    R0, R1 = IO.build_sets(mul_s_p=0), IO.build_sets()
    o["df_test2_vs_reg_default_overlap"] = len({it["sig"] for v in S["test2"].values() for it in v} & {it["sig"] for SS in (R0, R1) for v in SS["test2"].values() for it in v})
    allsig = set().union(*sets.values()); o["overlap"] = {}; o["n_draws"] = {}
    for seed in (1, 2):
        seen = set(); n = 0
        for it in train_replay(seed):
            n += 1
            if it["sig"] in allsig:
                seen.add(it["sig"])
        o["overlap"][str(seed)] = {k: len(seen & v) for k, v in sets.items()}; o["n_draws"][str(seed)] = n
    o["pass"] = (all(v == 0 for s_ in o["overlap"].values() for v in s_.values()) and all(v == 16000 * 64 for v in o["n_draws"].values()) and o["sha_table"]["set_law"] == "df")
    return o


def raise_():
    import boxes.inject_ow as IO
    import boxes.inject_aw_eval as AE
    import boxes.ow_words as OW
    o = {}
    for nm, fn, exc in (("eval_law_3", lambda: AE.law({"mul_s_p": 3}), ValueError), ("eval_law_str", lambda: AE.law({"mul_s_p": "2"}), ValueError),
                        ("stream_3", lambda: OW.Stream(1, mul_s_p=3), AssertionError), ("build_sets_3", lambda: IO.build_sets(mul_s_p=3), AssertionError)):
        try:
            fn(); o[nm] = False
        except exc:
            o[nm] = True
    d = G8.fresh(TMP + "/raise_args"); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + P4.LAT4 + ["--mul_s_p", "3", "--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
    o["argparse_3_rc"] = rc; o["eval_law_default"] = AE.law({}) == 1 and AE.reg_kw({}) == {} and AE.reg_kw({"mul_s_p": 2}) == {"mul_s_p": 2}
    o["pass"] = all(o[k] for k in ("eval_law_3", "eval_law_str", "stream_3", "build_sets_3", "eval_law_default")) and rc == 2; return o


def train():
    o = {}
    for tag, fl in (("rcb_lat", P4.LAT4 + REG + G7.initf("lat")), ("rcb_cd", P4.CD4 + REG + G7.initf("cd"))):
        com = fl + ["--fp32", "--micro", "8", "--sched_scale", "0.02", "--ckpt_every", "30", "--log_every", "10", "--stop_at", "30"]
        da, db = TMP + "/fo_s237mod_" + tag, TMP + "/fo_newmod_" + tag
        ra = G4.det("boxes/_inject_aw_s237.py", com, da); rb = G4.det("boxes/inject_aw.py", com, db); r = {"rc": [ra, rb]}
        if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
            r.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=("model", "opt", "rng_d", "np_rng", "torch_rng", "cuda_rng", "stream_rng", "stream_stats", "expo",
                                                                                     "binacc", "vbin", "vbin_d", "gnbin", "gn_log", "wg_hist", "dseq", "iseq")))
            strip = lambda p: [{k: v for k, v in json.loads(l_).items() if k != "sec"} for l_ in open(p)]  # noqa: E731
            r["log_equal"] = strip(da + "/log.jsonl") == strip(db + "/log.jsonl")
        r["pass"] = r["rc"] == [9, 9] and r.get("n_mismatch") == 0 and bool(r.get("log_equal")); o[tag] = r
    o["pass"] = all(v["pass"] for v in o.values()); return o


def evalq():
    import torch
    import boxes.inject_ow as IO
    import boxes._inject_aw_eval_s237 as E0
    import boxes.inject_aw_eval as E1
    o = {}; torch.zeros(1, device=DEV)
    for tag, src, law_ in (("default", ROOT + "/scratch/_tmp/s234/p4/ev_lat_aw4", 1), ("reg", ROOT + "/scratch/_tmp/s236/ev_lat_aw4", 0)):
        da, db = G8.fresh(TMP + "/eqev_s237_" + tag), G8.fresh(TMP + "/eqev_new_" + tag)
        for d_ in (da, db):
            for f_ in ("args.json", "model.pt", "train_summary.json"):
                shutil.copy(os.path.join(src, f_), os.path.join(d_, f_))
        S = IO.build_sets(mul_s_p=law_); S["test2"] = {k: v[:250] for k, v in S["test2"].items()}
        parts = P4_EVAL.split(","); E0.run(da, parts, DEV, 250, sets=S); E1.run(db, parts, DEV, 250, sets=S); res = {}; defs = []
        for p_ in parts:
            ja, jb = (json.load(open(os.path.join(x, "eval_%s.json" % p_))) for x in (da, db))
            if ja.get("def") != jb.get("def"):
                defs.append(p_)
            for j_ in (ja, jb):
                j_.pop("sec", None); j_.pop("module_sha256", None); j_.pop("def", None)
            res[p_] = ja == jb
        o[tag] = {"parts": res, "def_changed": defs, "pass": len(res) == 11 and all(res.values())}; torch.cuda.empty_cache()
    o["pass"] = all(v["pass"] for v in o.values()); return o


def strict():
    import torch
    import boxes.inject_aw as AW
    o = {}
    for k, arm in (("lat1", "lat"), ("lat2", "lat"), ("cd", "cd")):
        fl = (P4.LAT4 if arm == "lat" else P4.CD4)
        ns = argparse.Namespace(arm=fl[fl.index("--arm") + 1], mem_write="top", write_src="answer" if arm == "lat" else "all", init_trunk=fl[fl.index("--init_trunk") + 1],
                                init_sha=fl[fl.index("--init_sha") + 1])
        torch.manual_seed(1); m, g, params = AW.build_aw(ns, "cpu"); sd = torch.load(ROOT + "/" + INIT[k], map_location="cpu", weights_only=True); params.load_state_dict(sd, strict=True)
        ps = params.state_dict(); o[k] = {"keys_equal": sorted(ps) == sorted(sd), "all_equal": all(torch.equal(ps[x], sd[x]) for x in sd), "sha": G7.sha(INIT[k])[:16]}
    o["pass"] = all(v["keys_equal"] and v["all_equal"] for v in o.values()); return o


def gradflow():
    with G8.F64():
        import torch
        import boxes.inject_ow as IO
        import boxes.inject_aw as AW
        import boxes.ow_words as OW
        torch.set_num_threads(2); m, g = G4.aw_model(torch.float64); s = OW.Stream(24700, frames=OW.FRAME_TRAIN, mul_s_p=2, **IO.OWKW); its = []; nm = 0
        while len(its) < 8:
            d = 3 if len(its) < 4 else 4; it = s.draw(d); want = len(its) in (0, 4)
            if want and "mul_s" not in it["ops"]:
                continue
            its.append(AW.tok_aw(it, "value", "step", "answer")); nm += "mul_s" in it["ops"]
        o = {"n_with_mul_s": nm}
        for term in ("answer_ce", "aux_ce"):
            m.zero_grad(set_to_none=True); la, aux, _ = G4.aw_loss(m, its, 8, True); (la if term == "answer_ce" else aux).backward()
            gr = {n: (float(p.grad.norm()) if p.grad is not None else None) for n, p in m.named_parameters()}
            okf = lambda v: v is not None and v > 0 and math.isfinite(v)  # noqa: E731
            o[term] = {"gpt2_nonzero": sum(okf(v) for n, v in gr.items() if n.startswith("g.")), "mem_nonzero": sum(okf(v) for n, v in gr.items() if n.startswith("mem."))}
        o["pass"] = nm >= 2 and all(o[t]["gpt2_nonzero"] == 148 and o[t]["mem_nonzero"] == 12 for t in ("answer_ce", "aux_ce")); return o


def smoke():
    o = {}
    for tag, fl in (("lat", LATD), ("cd", CDD)):
        d = G8.fresh(TMP + "/smoke_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--micro", "8", "--stop_at", "50", "--ckpt_every", "0", "--log_every", "10",
                                                                                                          "--device", DEV, "--out", d], d + "/stdout.log")
        o[tag] = {"rc": rc}
        if os.path.exists(d + "/args.json"):
            ar = json.load(open(d + "/args.json")); o[tag].update({"set_law": ar.get("set_law"), "mul_s_p": ar.get("mul_s_p"), "excl_n": ar.get("excl_n")})
        if tag == "lat" and os.path.exists(d + "/log.jsonl"):
            br = [json.loads(l_) for l_ in open(d + "/log.jsonl") if '"by_d"' in l_][-1]; o[tag]["ws_ds"] = sorted(br.get("ws_readout", {}))
    o["pass"] = (o["lat"]["rc"] == 9 and o["cd"]["rc"] == 9 and o["lat"].get("set_law") == "df" == o["cd"].get("set_law") and o["lat"].get("mul_s_p") == 2
                 and o["lat"].get("ws_ds") == ["1", "2", "3", "4"]); return o


def evdf():
    import torch
    import boxes.inject_ow as IO
    import boxes.inject_aw_eval as AE
    o = {}
    for tag, fl, ebs, parts in (("lat", LATD, 250, DF_EVAL.split(",")), ("cd", CDD, 100, ["acc"])):
        d = G8.fresh(TMP + "/ev_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--steps", "20", "--micro", "8", "--eval_bs", str(ebs), "--eval_parts", "",
                                                                                                        "--ckpt_every", "0", "--log_every", "10", "--device", DEV, "--out", d], d + "/stdout.log")
        S = IO.build_sets(mul_s_p=2); S["test2"] = {k: v[:250] for k, v in S["test2"].items()}; torch.zeros(1, device=DEV); torch.cuda.reset_peak_memory_stats(DEV)
        AE.run(d, parts, DEV, ebs, sets=S); pk = torch.cuda.max_memory_allocated(DEV) / 2 ** 30
        fs = sorted(f_ for f_ in os.listdir(d) if f_.startswith("eval")); o[tag] = {"rc": rc, "files": fs, "peak_gib": pk}
        if tag == "lat":
            J = {p_: json.load(open(os.path.join(d, "eval_%s.json" % p_))) for p_ in ("opclass", "swap_prod", "swap_k", "probe", "swap_kd")}
            rs = AE.law_eval_sigs(df_args(), IO.build_sets(mul_s_p=2)["excl"]); sp = J["swap_prod"]; oc = J["opclass"]
            o[tag].update({"opclass_n": oc["n"], "opclass_passes": sorted(oc["items"]["pred_by_pass"], key=int), "opclass_items": len(oc["items"]["sig"]),
                           "swap_prod": {k: sp.get(k) for k in ("n_pairs", "tried", "k_flat")}, "swap_prod_fields": sorted(k for k in sp if k in ("follow_dfin", "follow_cb", "floor", "stay_own", "follow_U")),
                           "swap_prod_by_k": sorted(sp["by_k"]), "swap_k_reg": set(J["swap_k"]["items"]["rec_sig"]) | set(J["swap_k"]["items"]["don_sig"]) == rs["swap_k_d2"],
                           "probe_df": set(J["probe"]["items"]["sig"]) == rs["probe"], "swap_prod_df": set(sp["items"]["rec_sig"]) | set(sp["items"]["don_sig"]) == rs["swap_prod"],
                           "opclass_df": set(oc["items"]["sig"]) == rs["opclass"], "swap_kd_ds": sorted(J["swap_kd"]["by_d"]), "swap_kd_has_t2": "swap_mid_t2" in J["swap_kd"]["by_d"].get("3", {})})
        torch.cuda.empty_cache()
    la = o["lat"]
    o["pass"] = (la["rc"] == 0 and o["cd"]["rc"] == 0 and set(la["files"]) == {"eval.json"} | {"eval_%s.json" % p_ for p_ in DF_EVAL.split(",")} and o["cd"]["files"] == ["eval.json", "eval_acc.json"]
                 and la["opclass_n"] == 3000 and la["opclass_items"] == 3000 and la["opclass_passes"] == [str(t) for t in range(1, 9)] and la["swap_prod"]["n_pairs"] == 1000
                 and la["swap_prod_fields"] == ["floor", "follow_U", "follow_cb", "follow_dfin", "stay_own"] and la["swap_prod_by_k"] == ["3", "8"] and la["swap_k_reg"] and la["probe_df"]
                 and la["swap_prod_df"] and la["opclass_df"] and la["swap_kd_ds"] == ["3", "4"] and not la["swap_kd_has_t2"] and la["peak_gib"] < 7.5)
    return o


def refdf():
    import boxes.inject_ow as IO
    src = ROOT + "/scratch/_tmp/s234/p4/ev_lat_aw4"; snap = {f_: os.path.getmtime(os.path.join(src, f_)) for f_ in os.listdir(src)}; out = TMP + "/refdf_out"
    if os.path.isdir(out):
        shutil.rmtree(out)
    rc = G8.sh([sys.executable, "-B", "scratch/s236_reg_eval.py", "--run", "ev_lat_aw4", "--src_dir", src, "--out", out, "--law", "2", "--parts", "acc,ws,opclass,swap_prod,cue_only",
                "--eval_bs", "100", "--device", DEV], TMP + "/refdf.log")
    S = IO.build_sets(mul_s_p=2); fs = sorted(os.listdir(out)) if os.path.isdir(out) else []; ar = json.load(open(out + "/args.json")) if "args.json" in fs else {}
    o = {"rc": rc, "files": fs, "args_df_shas": all(ar.get(k) == S["shas"][k] for k in SETKEYS) and ar.get("mul_s_p") == 2 and ar.get("set_law") == "df",
         "src_untouched": {f_: os.path.getmtime(os.path.join(src, f_)) for f_ in os.listdir(src)} == snap}
    o["pass"] = rc == 0 and {"eval_acc.json", "eval_ws.json", "eval_opclass.json", "eval_swap_prod.json", "eval_cue_only.json"} <= set(fs) and o["args_df_shas"] and o["src_untouched"]
    return o


def args_part():
    import s228_args_check as C
    o = {}; der = {"mul_s_p", "init_model", "init_model_sha", "init_model_sha256", "set_law", "pair_stats", "excl_n", "inject_ow_sha256"} | set(SETKEYS)
    for tag, fl, ref, extra in (("df_plaw4_s1", P4.LAT4 + DF + initf("lat1") + ["--auto_micro", "--ckpt_every", "1000", "--eval_parts", DF_EVAL], "scratch/s237_args_frozen_rcb_plaw4_s1.json", {"eval_parts"}),
                                ("df_plaw4_s2", P4.LAT4 + DF + initf("lat2") + ["--seed", "2", "--auto_micro", "--ckpt_every", "1000", "--eval_parts", DF_EVAL], "scratch/s237_args_frozen_rcb_plaw4_s2.json", {"eval_parts"}),
                                ("df_pcd4_s1", P4.CD4 + DF + initf("cd") + ["--auto_micro", "--ckpt_every", "1000", "--eval_parts", "acc"], "scratch/s237_args_frozen_rcb_pcd4_s1.json", set())):
        d = G8.fresh(TMP + "/args_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
        new = json.load(open(d + "/args.json")); dif = C.diff(json.load(open(ROOT + "/" + ref)), new)
        o[tag] = {"rc": rc, "diff_keys": sorted(dif), "diff": {k: v for k, v in dif.items() if k != "pair_stats"}, "args_file": d + "/args.json",
                  "pass": rc == 0 and set(dif) == der | extra and dif["mul_s_p"] == [0, 2]}
    o["pass"] = all(v["pass"] for v in o.values()); return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    fns = {"gen": gen, "law": law, "mi": mi, "classes": classes, "disjoint": disjoint, "raise": raise_, "train": train, "evalq": evalq, "strict": strict,
           "gradflow": gradflow, "smoke": smoke, "evdf": evdf, "refdf": refdf, "args": args_part}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
