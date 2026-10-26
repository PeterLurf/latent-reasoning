"""§235 D4-DECOR build gates (D4_v1 §1 / §6 + Audit's additions via Theory 10-26; for Audit's G-D). -> scratch/gate_s235/<part>.json ("pass"). 4060 (cuda:0) / CPU.
References kept: boxes/_ow_words_s234.py (ow_words before the mul_s_p3 kwarg) and boxes/_inject_aw_s234.py (the P-4 module, 5357efd0).
  stream    flags-off: OW.Stream at the default AND explicit mul_s_p3 1.0 vs the old module, OWKW (topo, dag), 10k draws per d = 1..12 (one stream per d, seed
            23500 + d, EXCL = build_sets' excl): sigs / texts identical, final rng states equal, stats equal.
  share     Audit (1): the training draw law (Stream(SEED_TRAIN, EXCL, FRAME_TRAIN, **OWKW), d per step from default_rng(1 + SEED_D) U{1..4}, bs 64) for 800 steps
            (51,200 draws) at mul_s_p3 0.5 and 1.0: P(ops[2] == mul_s | d 3) at 0.5 in [0.47, 0.53] (1.0: == 1.0); per-position op shares at d 4 (and d 1 / 2)
            at 0.5 vs 1.0 within 0.03 (per-share |diff|); the d 3 stream stats keys == the counted shares; a d-4-only stream at 0.5 == at 1.0 bitwise (no coin).
  sets      Audit (2): --args_only P-4 + --mul_s_p3 0.5 -> the set shas (test2 / transfer / pairs2 / floors2 / s226 / s227) == aw_plaw4_s1/args.json; the eval items
            rebuilt with the new modules (TEST2 d 1..12 sigs, swap_k d 2 pairs, swap_kd d 3 / 4 pairs, probe stream) == the sigs stored in aw_plaw4_s1's eval files.
  flagsoff  Audit (3): the P-4 module (_inject_aw_s234.py) vs the new inject_aw.py, LAT-AW and CD-AN at the P-4 args (d <= 4, 16k), det wrap, fp32, micro 8,
            sched_scale 0.02, stop_at 30 (ckpt 30): every ckpt field torch.equal and logs (minus sec) equal.
  gradflow  fp64 CPU, perturbed, gamma 1, 4 d 3 items WITHOUT a mul_s step (drawn at mul_s_p3 0.5): answer CE and aux CE each nonzero on 148 GPT-2 + 12 memory tensors.
  smoke     LAT-AW / CD-AN at the §235 args (P-4 + --mul_s_p3 0.5), 50 steps (stop_at 50, micro 8, log 10) -> rc 9; LAT-AW's step-50 bin has stream_d3_mul_s
            [n_mul_s, n_d3] with n_d3 > 0 and ws_readout d 1..4.
  args      --args_only vs scratch/s234_args_frozen_aw_plaw4_s1.json / _aw_pcd4_s1.json: diff == {mul_s_p3} each.
    python -B scratch/gate_s235.py stream share sets args flagsoff gradflow smoke"""
import json
import math
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch")
import gate_s228 as G8  # noqa: E402
import gate_s234 as G4  # noqa: E402
import gate_s234_p4 as P4  # noqa: E402

OUT = ROOT + "/scratch/gate_s235"; TMP = ROOT + "/scratch/_tmp/s235"; DEV = "cuda:0"
P4_EVAL = P4.P4_EVAL; LAT4, CD4 = P4.LAT4, P4.CD4; FLAG = ["--mul_s_p3", "0.5"]
RUN4 = ROOT + "/boxes/runs_fsq/aw_plaw4_s1"


def stream():
    import boxes._ow_words_s234 as O0
    import boxes.inject_ow as IO
    import boxes.ow_words as OW
    ex = IO.build_sets()["excl"]; o = {"by_d": {}}; ok = True
    for d in range(1, 13):
        r = {}
        for tag, kw in (("default", {}), ("explicit_1.0", {"mul_s_p3": 1.0})):
            a = O0.Stream(23500 + d, ex, frames=O0.FRAME_TRAIN, **IO.OWKW); b = OW.Stream(23500 + d, ex, frames=OW.FRAME_TRAIN, **IO.OWKW, **kw)
            A = [a.draw(d) for _ in range(10000)]; B = [b.draw(d) for _ in range(10000)]
            eq = [x["sig"] for x in A] == [x["sig"] for x in B] and [x["text"] for x in A] == [x["text"] for x in B]
            rs = a.rng.bit_generator.state == b.rng.bit_generator.state; st = a.stats == b.stats
            r[tag] = {"items_equal": eq, "rng_state_equal": rs, "stats_equal": st}; ok = ok and eq and rs and st
        o["by_d"][str(d)] = r
    o["pass"] = ok; return o


def train_draws(p, steps=800, dmax=4):
    import numpy as np
    import boxes.inject_ow as IO
    import boxes.ow_words as OW
    ex = IO.build_sets()["excl"]; st = OW.Stream(OW.SEED_TRAIN, ex, frames=OW.FRAME_TRAIN, mul_s_p3=p, **IO.OWKW); rd = np.random.default_rng(1 + IO.SEED_D); its = []
    for _ in range(steps):
        d = int(rd.integers(1, dmax + 1)); its += [st.draw(d) for _ in range(64)]
    return its, st.stats


def op_shares(its, d):
    import collections
    sh = {}
    for t in range(d):
        c = collections.Counter(it["ops"][t] for it in its if it["d"] == d); n = sum(c.values()); sh[str(t + 1)] = {k: v / n for k, v in sorted(c.items())}
    return sh


def share():
    import boxes.inject_ow as IO
    import boxes.ow_words as OW
    I5, S5 = train_draws(0.5); I1, S1 = train_draws(1.0); o = {"n_draws": [len(I5), len(I1)]}
    d3_5 = [it for it in I5 if it["d"] == 3]; d3_1 = [it for it in I1 if it["d"] == 3]
    p5 = sum(it["ops"][2] == "mul_s" for it in d3_5) / len(d3_5); p1 = sum(it["ops"][2] == "mul_s" for it in d3_1) / len(d3_1)
    any5 = sum("mul_s" in it["ops"] for it in d3_5) / len(d3_5)
    o.update({"n_d3": [len(d3_5), len(d3_1)], "p_mul_s_step3_d3": {"0.5": p5, "1.0": p1}, "p_any_mul_s_d3_at_0.5": any5,
              "stats_d3_at_0.5": {"d3_items": S5.get("d3_items"), "d3_mul_s": S5.get("d3_mul_s")}, "stats_keys_at_1.0": sorted(S1)})
    diffs = {}
    for d in (1, 2, 4):
        a, b = op_shares(I5, d), op_shares(I1, d); mx = 0.0
        for t in a:
            for k in set(a[t]) | set(b[t]):
                mx = max(mx, abs(a[t].get(k, 0.0) - b[t].get(k, 0.0)))
        diffs[str(d)] = {"max_abs_share_diff": mx, "n": [sum(it["d"] == d for it in I5), sum(it["d"] == d for it in I1)]}
    o["op_share_diff_vs_1.0"] = diffs; o["d4_op_shares_0.5"] = op_shares(I5, 4)
    ex = IO.build_sets()["excl"]; a = OW.Stream(23600, ex, frames=OW.FRAME_TRAIN, mul_s_p3=0.5, **IO.OWKW); b = OW.Stream(23600, ex, frames=OW.FRAME_TRAIN, **IO.OWKW)
    o["d4_only_stream_equal"] = [a.draw(4)["sig"] for _ in range(5000)] == [b.draw(4)["sig"] for _ in range(5000)] and a.rng.bit_generator.state == b.rng.bit_generator.state
    o["pass"] = (0.47 <= p5 <= 0.53 and p1 == 1.0 and S5.get("d3_items") == len(d3_5) and S5.get("d3_mul_s") == sum("mul_s" in it["ops"] for it in d3_5)
                 and "d3_items" not in S1 and all(v["max_abs_share_diff"] <= 0.03 for v in diffs.values()) and o["d4_only_stream_equal"] and len(I5) >= 50000)
    return o


def sets():
    import boxes.inject_ow as IO
    import boxes.inject_aw_eval as AE
    import boxes.ow_words as OW
    d = G8.fresh(TMP + "/sets_args"); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + LAT4 + FLAG + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
    new = json.load(open(d + "/args.json")); ref = json.load(open(RUN4 + "/args.json"))
    keys = ["test2_sha", "transfer_sha", "pairs2_sha", "floors2_sha", "s226_test_sha", "s227_decode_sha"]
    o = {"rc": rc, "sha_equal": {k: new.get(k) == ref.get(k) and ref.get(k) is not None for k in keys}, "mul_s_p3": new.get("mul_s_p3"), "pair_stats_equal": new.get("pair_stats") == ref.get("pair_stats")}
    S = IO.build_sets(); E = lambda p: json.load(open(os.path.join(RUN4, "eval_%s.json" % p)))  # noqa: E731
    acc = E("acc"); o["test2_sigs_equal"] = all(acc["items"][str(dd)]["sig"] == [it["sig"] for it in S["test2"][dd]] for dd in IO.TEST_DS)
    R, Dn, kept, _ = AE.swap_pairs(ref); sk = E("swap_k")
    o["swap_k_sigs_equal"] = sk["items"]["rec_sig"] == [p[0]["sig"] for p in kept] and sk["items"]["don_sig"] == [p[1]["sig"] for p in kept]
    skd = E("swap_kd"); eqd = {}
    for dd in (3, 4):
        _, _, kp, _, _ = AE.swap_pairs_d(ref, dd); eqd[str(dd)] = skd["items"][str(dd)]["rec_sig"] == [p[0]["sig"] for p in kp] and skd["items"][str(dd)]["don_sig"] == [p[1]["sig"] for p in kp]
    o["swap_kd_sigs_equal"] = eqd
    s = OW.Stream(AE.SEED_PROBE, S["excl"], frames=OW.FRAME_TRAIN, **IO.OWKW); o["probe_sigs_equal"] = E("probe")["items"]["sig"] == [s.draw(2)["sig"] for _ in range(AE.PROBE_N)]
    for p_ in ("lp5", "lp5h", "ws", "cue_only", "cue_wrong", "mean_s"):
        J = E(p_); o[p_ + "_sigs_equal"] = all(J["items"][k]["sig"] == [it["sig"] for it in S["test2"][int(k)]] for k in J["items"])
    o["pass"] = (rc == 0 and all(o["sha_equal"].values()) and o["pair_stats_equal"] and o["mul_s_p3"] == 0.5 and o["test2_sigs_equal"] and o["swap_k_sigs_equal"]
                 and all(eqd.values()) and o["probe_sigs_equal"] and all(o[p_ + "_sigs_equal"] for p_ in ("lp5", "lp5h", "ws", "cue_only", "cue_wrong", "mean_s")))
    return o


def flagsoff():
    o = {}
    for tag, fl in (("lat_aw4", LAT4), ("cd_an4", CD4)):
        com = fl + ["--fp32", "--micro", "8", "--sched_scale", "0.02", "--ckpt_every", "30", "--log_every", "10", "--stop_at", "30"]
        da, db = TMP + "/fo_p4mod_" + tag, TMP + "/fo_newmod_" + tag
        ra = G4.det("boxes/_inject_aw_s234.py", com, da); rb = G4.det("boxes/inject_aw.py", com, db); r = {"rc": [ra, rb]}
        if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
            r.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=("model", "opt", "rng_d", "np_rng", "torch_rng", "cuda_rng", "stream_rng", "stream_stats", "expo",
                                                                                     "binacc", "vbin", "vbin_d", "gn_log", "wg_hist")))
            strip = lambda p: [{k: v for k, v in json.loads(l_).items() if k != "sec"} for l_ in open(p)]  # noqa: E731
            r["log_equal"] = strip(da + "/log.jsonl") == strip(db + "/log.jsonl")
        r["pass"] = r["rc"] == [9, 9] and r.get("n_mismatch") == 0 and bool(r.get("log_equal")); o[tag] = r
    o["pass"] = all(v["pass"] for v in o.values()); return o


def gradflow():
    with G8.F64():
        import torch
        import boxes.inject_ow as IO
        import boxes.inject_aw as AW
        import boxes.ow_words as OW
        torch.set_num_threads(2); m, g = G4.aw_model(torch.float64); s = OW.Stream(23700, frames=OW.FRAME_TRAIN, mul_s_p3=0.5, **IO.OWKW); its = []
        while len(its) < 4:
            it = s.draw(3)
            if "mul_s" not in it["ops"]:
                its.append(AW.tok_aw(it, "value", "step", "answer"))
        o = {"ops": [it["raw"]["ops"] for it in its]}
        for term in ("answer_ce", "aux_ce"):
            m.zero_grad(set_to_none=True); la, aux, _ = G4.aw_loss(m, its, 8, True); (la if term == "answer_ce" else aux).backward()
            gr = {n: (float(p.grad.norm()) if p.grad is not None else None) for n, p in m.named_parameters()}
            okf = lambda v: v is not None and v > 0 and math.isfinite(v)  # noqa: E731
            o[term] = {"gpt2_nonzero": sum(okf(v) for n, v in gr.items() if n.startswith("g.")), "mem_nonzero": sum(okf(v) for n, v in gr.items() if n.startswith("mem.")),
                       "zero_or_none": sorted(n for n, v in gr.items() if not okf(v))[:20]}
        o["pass"] = all(o[t]["gpt2_nonzero"] == 148 and o[t]["mem_nonzero"] == 12 for t in ("answer_ce", "aux_ce")); return o


def smoke():
    o = {}
    for tag, fl in (("lat_aw4", LAT4 + FLAG), ("cd_an4", CD4 + FLAG)):
        d = G8.fresh(TMP + "/smoke_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--micro", "8", "--stop_at", "50", "--ckpt_every", "0", "--log_every", "10",
                                                                                                          "--device", DEV, "--out", d], d + "/stdout.log")
        o[tag] = {"rc": rc}
        if os.path.exists(d + "/log.jsonl"):
            br = [json.loads(l_) for l_ in open(d + "/log.jsonl") if '"by_d"' in l_][-1]
            o[tag].update({"stream_d3_mul_s": br.get("stream_d3_mul_s"), "ws_ds": sorted(br.get("ws_readout", {})), "exposure_by_d": br.get("exposure_by_d")})
    la, cd = o["lat_aw4"], o["cd_an4"]
    o["pass"] = (la["rc"] == 9 and cd["rc"] == 9 and isinstance(la.get("stream_d3_mul_s"), list) and la["stream_d3_mul_s"][1] > 0 and la.get("ws_ds") == ["1", "2", "3", "4"]
                 and isinstance(cd.get("stream_d3_mul_s"), list)); return o


def args_part():
    import s228_args_check as C
    o = {}
    for tag, fl, ref in (("lat_aw4", LAT4 + FLAG + ["--auto_micro", "--ckpt_every", "1000", "--eval_parts", P4_EVAL], "scratch/s234_args_frozen_aw_plaw4_s1.json"),
                         ("cd_an4", CD4 + FLAG + ["--auto_micro", "--ckpt_every", "1000", "--eval_parts", "acc"], "scratch/s234_args_frozen_aw_pcd4_s1.json")):
        d = G8.fresh(TMP + "/args_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
        dif = C.diff(json.load(open(ROOT + "/" + ref)), json.load(open(d + "/args.json")))
        o[tag] = {"rc": rc, "diff": dif, "args_file": d + "/args.json", "pass": rc == 0 and set(dif) == {"mul_s_p3"} and dif["mul_s_p3"][1] == 0.5}
    o["pass"] = all(v["pass"] for v in o.values()); return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    fns = {"stream": stream, "share": share, "sets": sets, "flagsoff": flagsoff, "gradflow": gradflow, "smoke": smoke, "args": args_part}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
