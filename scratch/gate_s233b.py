"""§233 P-2 build gates (ST_v1 §3 / draft §6.1 item 3 / REVIEW_ST fixes 3, 8, 11; for Audit's G-D). -> scratch/gate_s233/<part>.json ("pass"). Runs under
scratch/_tmp/s233b; the 4060 (cuda:0) unless fp64 CPU.
  g1       FLAGS-OFF: boxes.inject_ow vs boxes.inject_st --cue none, det wrapper, fp32, micro 8, sched_scale 0.02, stop_at 30 (ckpt 30): (a) §228 LAT-2 flags,
           (b) --arm direct: ckpt model + opt + streams torch.equal, logs (minus sec) equal.
  g2       per-distinct-cue lower cache, fp64 CPU (Tensor.float -> double), a perturbed model (gate_s228.perturbed): cached vs uncached loss and every grad,
           LAT-ST cue inputs at d 1 / 2 / 4 / 8 (2 items each, K 8): max rel diff <= 1e-10.
  g2b      batch-of-one (fp64 CPU): each item's cached lower rows and its S after passes 1..d (+1 hold) inside a mixed-d batch == the item alone, uncached
           (rel <= 1e-10): no cross-item cache reuse.
  g3       per-term gradflow (fp64 CPU, perturbed, 4 d 2 items, cue step): answer CE and aux CE each nonzero on all 148 GPT-2 + 12 memory tensors.
  gcue     the no-digit cue fixture: 4,000 TOPO items d 1..8: tok_st's G-CUE asserts (no digit, no number token, 7 tokens) hold for every cue, and on 2,000
           same-skeleton redraw pairs (ow_words.redraw) the cue token lists are equal at every t (skeleton-only).
  g9       decoded fixtures: 3 items per d (1, 2, 4, 8): every pass t = 1..d + 1 decodes to " " + text + " " + QUESTION(u, S_min(t, d)), the decoder to
           " " + text + " The answer is:" (no cue), and CD's dec to " " + text + cues S_1..S_d + " The answer is:".
  g4_g10   plain-path / flag smokes on the 4060: inject_st --cue none (inject), --cue step (inject), --cue all (direct), 50 steps each (stop_at 50, log 10)
           -> rc 9; G10: in the cue-step run's step-50 bin, aux_vacc_by_d_pass pooled with its _n counts == aux_vacc_by_pass (|diff| <= 1e-12).
  g7       CUE-ONLY on an UNTRAINED LAT-ST model (inject_st build_model, no training) on TEST2 d 2 (200 items): answer / pass-1 / pass-2 aux acc reported
           (expected near chance: the guard's null).
  args     --args_only LAT-ST vs scratch/s228_args_frozen_s228_plat2_s1.json and CD vs scratch/s228_args_frozen_s228_pdir2c_s1.json (s228_args_check's
           CONTROL + inject_ow_sha256): == {cue} (+ the descriptive format string).
  f2       memory / timing at the pilot shapes on the 4060 (budget 6.0): LAT-ST d <= 2 and CD d <= 2, --auto_micro --time_steps 30 (the 3090 numbers come
           from the queue's K7 / G-TIME at admission; the eval probe = TEST2's longest cued d 12 item).
    python -B scratch/gate_s233b.py gcue g9 args g1 g2 g2b g3 g4_g10 g7 f2"""
import json
import math
import os
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch")
OUT = ROOT + "/scratch/gate_s233"; TMP = ROOT + "/scratch/_tmp/s233b"; DEV = "cuda:0"
import gate_s228 as G8  # noqa: E402

LAT2 = ["--arm", "inject", "--think_k", "8", "--aux_step", "1", "--aux_fade", "48000,48000", "--aux_reweight", "--fast_add", "--cache_lower", "--d_max", "2", "--steps", "12000"]
DIR2 = ["--arm", "direct", "--d_max", "2", "--steps", "12000"]


def det(mod_file, fl, d):
    G8.fresh(d); rc = G8.sh([sys.executable, "-B", "-m", "scratch.s222_det_wrap", mod_file] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")
    return rc


def g1():
    o = {}
    for tag, fl in (("lat", LAT2), ("direct", DIR2)):
        com = fl + ["--fp32", "--micro", "8", "--sched_scale", "0.02", "--ckpt_every", "30", "--log_every", "10", "--stop_at", "30"]
        da, db = TMP + "/g1_ow_" + tag, TMP + "/g1_st_" + tag
        ra = det("boxes/inject_ow.py", com, da); rb = det("boxes/inject_st.py", com + ["--cue", "none"], db); r = {"rc": [ra, rb]}
        if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
            r.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=("model", "opt", "rng_d", "stream_rng", "expo", "binacc", "vbin", "gn_log")))
            strip = lambda p: [{k: v for k, v in json.loads(l_).items() if k != "sec"} for l_ in open(p)]  # noqa: E731
            r["log_equal"] = strip(da + "/log.jsonl") == strip(db + "/log.jsonl")
        r["pass"] = r["rc"] == [9, 9] and r.get("n_mismatch") == 0 and bool(r.get("log_equal")); o[tag] = r
    o["pass"] = all(v["pass"] for v in o.values()); return o


def st_items(ds, n, seed=22997, cue="step"):
    import boxes.inject_ow as IO
    import boxes.inject_st as ST
    import boxes.ow_words as OW
    s = OW.Stream(seed, frames=OW.FRAME_TRAIN, **IO.OWKW); return {d: [ST.tok_st(s.draw(d), "value", cue) for _ in range(n)] for d in ds}


def st_loss(m, items, K, cached):
    """inject_st's LAT loss (answer CE + AUX_W-weighted aux) on the cued inputs, cache on / off."""
    import torch
    import torch.nn.functional as Fn
    import boxes.inject_ow as IO
    import boxes.inject_st as ST
    xs, msk = ST.cue_inputs(m, items, "cpu"); hls = [m.lower(x) for x in xs] if cached else None; dt = next(m.parameters()).dtype
    S = m.mem.zero_state(len(items), "cpu").to(dt); per = [[] for _ in items]; Ss = []
    for t in range(1, K + 1):
        i_ = min(t, len(xs)) - 1; S = m.write(S, xs[i_], msk, hls[i_] if hls else None); Ss.append(S)
        idx = [i for i, it in enumerate(items) if len(it["aux"]) >= t]
        if idx:
            lg_a, y_a = m.aux_logits(S[idx], [items[i] for i in idx], t, "cpu")
            ce = Fn.cross_entropy(lg_a.transpose(1, 2), y_a, ignore_index=-100, reduction="none").sum(1) / (y_a != -100).sum(1)
            for r_, i in enumerate(idx):
                per[i].append(ce[r_] * IO.AUX_W[t - 1])
    la = Fn.cross_entropy(m.dec_logits(S, items, "cpu"), torch.tensor(IO.NUM_IDS[[it["ans"] for it in items]]))
    aux = torch.stack([torch.stack(p_).mean() for p_ in per if p_]).mean(); return la, aux, Ss


def g2():
    with G8.F64():
        import torch
        torch.set_num_threads(2); m, g = G8.perturbed(torch.float64); its = st_items((1, 2, 4, 8), 2); o = {"by_d": {}}; ok = True
        rel = lambda u, v: float((u - v).abs().max() / v.abs().max().clamp_min(1e-300))  # noqa: E731
        for d, it_ in its.items():
            res = []
            for cached in (False, True):
                m.zero_grad(set_to_none=True); la, aux, Ss = st_loss(m, it_, 8, cached); (la + aux).backward()
                res.append((float(la + aux), [s.detach() for s in Ss], {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}))
            (l0, S0, G0), (l1, S1, G1) = res
            r = {"loss_rel": abs(l1 - l0) / abs(l0), "S_rel_max": max(rel(u, v) for u, v in zip(S1, S0)), "grad_rel_max": max(rel(G1[n], G0[n]) for n in G0 if G0[n].abs().max() > 0),
                 "n_distinct_inputs": len(it_[0]["think_cues"])}
            r["pass"] = max(r["loss_rel"], r["S_rel_max"], r["grad_rel_max"]) <= 1e-10; o["by_d"][str(d)] = r; ok = ok and r["pass"]
        o["pass"] = ok; return o


def g2b():
    with G8.F64():
        import torch
        import boxes.inject_st as ST
        torch.set_num_threads(2); m, g = G8.perturbed(torch.float64); its = sum(st_items((1, 2, 3, 4), 2).values(), [])   # a mixed-d batch
        xs, msk = ST.cue_inputs(m, its, "cpu"); hls = [m.lower(x) for x in xs]; S = m.mem.zero_state(len(its), "cpu").double(); Sb = []
        for t in range(1, len(xs) + 2):
            i_ = min(t, len(xs)) - 1; S = m.write(S, xs[i_], msk, hls[i_]); Sb.append(S)
        worst_low = worst_S = 0.0
        for r, it in enumerate(its):
            xs1, m1 = ST.cue_inputs(m, [it], "cpu"); L = int(m1.sum()); S1 = m.mem.zero_state(1, "cpu").double()
            for t in range(1, len(xs) + 2):
                i_ = min(t, len(xs)) - 1; j_ = min(t, len(xs1)) - 1
                lo_b = hls[i_][r, :L]; lo_1 = m.lower(xs1[j_])[0, :L]; worst_low = max(worst_low, float((lo_b - lo_1).abs().max() / lo_1.abs().max()))
                S1 = m.write(S1, xs1[j_], m1, None); worst_S = max(worst_S, float((Sb[t - 1][r] - S1[0]).abs().max() / S1.abs().max().clamp_min(1e-300)))
        return {"n_items": len(its), "ds": sorted({it["d"] for it in its}), "lower_rel_max": worst_low, "S_rel_max": worst_S, "pass": worst_low <= 1e-10 and worst_S <= 1e-10}


def g3():
    with G8.F64():
        import torch
        torch.set_num_threads(2); m, g = G8.perturbed(torch.float64); its = st_items((2,), 4)[2]; o = {}
        for term in ("answer_ce", "aux_ce"):
            m.zero_grad(set_to_none=True); la, aux, _ = st_loss(m, its, 8, True); (la if term == "answer_ce" else aux).backward()
            gr = {n: (float(p.grad.norm()) if p.grad is not None else None) for n, p in m.named_parameters()}
            okf = lambda v: v is not None and v > 0 and math.isfinite(v)  # noqa: E731
            o[term] = {"gpt2_nonzero": sum(okf(v) for n, v in gr.items() if n.startswith("g.")), "mem_nonzero": sum(okf(v) for n, v in gr.items() if n.startswith("mem.")),
                       "zero_or_none": sorted(n for n, v in gr.items() if not okf(v))[:20]}
        o["pass"] = all(o[t]["gpt2_nonzero"] == 148 and o[t]["mem_nonzero"] == 12 for t in ("answer_ce", "aux_ce")); return o


def gcue():
    import boxes.inject_ow as IO
    import boxes.inject_st as ST
    import boxes.ow_words as OW
    s = OW.Stream(22998, frames=OW.FRAME_TRAIN, **IO.OWKW); n = 0
    for d in range(1, 9):
        for _ in range(500):
            ST.tok_st(s.draw(d), "value", "step"); ST.tok_st(s.draw(d), "value", "all"); n += 2
    r = __import__("numpy").random.default_rng(23300); eq = tot = 0
    for d in range(2, 9):
        for _ in range(2000 // 7 + 1):
            it = s.draw(d); dn = OW.redraw(r, it)
            if dn is None:
                continue
            a_, b_ = ST.tok_st(it, "value", "step"), ST.tok_st(dn, "value", "step"); L = len(a_["think"]); tot += 1
            eq += all(qa[L:] == qb[len(b_["think"]):] for qa, qb in zip(a_["think_cues"], b_["think_cues"]))
    return {"items_asserted": n, "skeleton_pairs": tot, "cue_tokens_equal": eq, "pass": eq == tot and tot >= 1900}


def g9():
    import boxes.inject_ow as IO
    import boxes.inject_st as ST
    import boxes.ow_words as OW
    s = OW.Stream(22996, frames=OW.FRAME_TRAIN, **IO.OWKW); bad = []; fx = []
    for d in (1, 2, 4, 8):
        for _ in range(3):
            it = s.draw(d); o = ST.tok_st(it, "value", "step"); c = ST.tok_st(it, "value", "all"); sp = it["names"]["spine"]; q = lambda k: OW.QUESTION.format(u=it["unit"], A=sp[k])  # noqa: E731
            for t in range(1, d + 2):
                got = IO.ENC.decode(o["think_cues"][min(t, d) - 1][1:]); want = " " + it["text"] + " " + q(min(t, d))
                if got != want:
                    bad.append(("think", d, t))
            if IO.ENC.decode(o["dec"][1:]) != " " + it["text"] + " The answer is:":
                bad.append(("dec", d))
            if IO.ENC.decode(c["dec"][1:]) != " " + it["text"] + "".join(" " + q(k) for k in range(1, d + 1)) + " The answer is:":
                bad.append(("cd_dec", d))
            fx.append({"d": d, "pass1": IO.ENC.decode(o["think_cues"][0][1:])[-120:], "pass_hold": IO.ENC.decode(o["think_cues"][-1][1:])[-60:], "cd_dec": IO.ENC.decode(c["dec"][1:])[-200:]})
    return {"bad": bad, "fixtures": fx, "pass": not bad}


def g4_g10():
    o = {}
    for tag, fl in (("none_inject", LAT2 + ["--cue", "none"]), ("step_inject", LAT2 + ["--cue", "step"]), ("all_direct", DIR2 + ["--cue", "all"])):
        d = G8.fresh(TMP + "/g4_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_st"] + fl + ["--micro", "8", "--stop_at", "50", "--ckpt_every", "0", "--log_every", "10",
                                                                                                          "--device", DEV, "--out", d], d + "/stdout.log")
        o[tag] = {"rc": rc}
        if tag == "step_inject" and os.path.exists(d + "/log.jsonl"):
            br = [json.loads(l_) for l_ in open(d + "/log.jsonl") if '"by_d"' in l_][-1]; p = br.get("aux_vacc_by_pass", {}); bd = br.get("aux_vacc_by_d_pass", {}); bn = br.get("aux_vacc_by_d_pass_n", {})
            pooled = {t: sum(bd[dd][t] * bn[dd][t] for dd in bd if t in bd[dd]) / sum(bn[dd][t] for dd in bn if t in bn[dd]) for t in p}
            o["g10"] = {"aux_vacc_by_pass": p, "pooled_from_by_d": pooled, "max_abs_diff": max(abs(pooled[t] - p[t]) for t in p) if p else None, "d_keys": sorted(bd)}
    o["pass"] = all(v.get("rc") == 9 for k, v in o.items() if k != "g10") and o.get("g10", {}).get("max_abs_diff") is not None and o["g10"]["max_abs_diff"] <= 1e-12; return o


def g7():
    import argparse
    import numpy as np
    import torch
    import boxes.inject_ow as IO
    import boxes.inject_st_eval as SE
    torch.backends.cuda.matmul.allow_tf32 = False
    a = {"arm": "inject", "think_k": 8, "mem_write": "top", "seed": 1, "aux_target": "value", "cue": "step", "fast_add": True, "d_max": 2}
    torch.manual_seed(1); m, g, _ = IO.build_model(argparse.Namespace(**a), DEV); m.eval(); S = IO.build_sets(); S["test2"] = {2: S["test2"][2][:200]}
    r = SE.part_cue_only(m, g, S, a, DEV, 50, [2])["by_d"]["2"]; return {**r, "pass": r["answer_acc"] <= r["majority"] + 0.05 and r["aux_vacc_pass1"] <= 0.25 and r["aux_vacc_pass2"] <= 0.25,
                                                                     "note": "untrained model: memory wo = 0 at init, so the decoder reads the text alone (no memory signal)"}


def args_part():
    sys.path.insert(0, ROOT + "/scratch"); import s228_args_check as C
    o = {}
    for tag, fl, ref in (("lat_st", LAT2 + ["--cue", "step", "--auto_micro", "--ckpt_every", "1000", "--eval_parts", "acc,lp5,mean_s,cue_only,cue_wrong"], "scratch/s228_args_frozen_s228_plat2_s1.json"),
                         ("cd", DIR2 + ["--cue", "all", "--auto_micro", "--ckpt_every", "1000", "--eval_parts", "acc"], "scratch/s228_args_frozen_s228_pdir2c_s1.json")):
        d = G8.fresh(TMP + "/args_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_st"] + fl + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
        new = json.load(open(d + "/args.json")); dif = C.diff(json.load(open(ROOT + "/" + ref)), {k: v for k, v in new.items() if k != "inject_ow_sha256"})
        o[tag] = {"rc": rc, "diff_keys": sorted(dif), "diff": {k: v for k, v in dif.items() if k != "format"}, "args_file": d + "/args.json"}
        o[tag]["pass"] = rc == 0 and set(dif) - {"aux_target"} <= {"cue", "steps", "format", "eval_parts"} and dif.get("aux_target", [None, "value"])[1] == "value"   # aux_target: absent in the §228 frozen files (pre-§229), "value" = its default
    o["pass"] = all(v["pass"] for v in o.values()); return o


def f2():
    o = {}
    for tag, fl in (("lat_st_d2", LAT2 + ["--cue", "step"]), ("cd_d2", DIR2 + ["--cue", "all"])):
        d = G8.fresh(TMP + "/f2_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_st"] + fl + ["--auto_micro", "--mem_budget", "6.0", "--time_steps", "30", "--log_every", "10",
                                                                                                          "--device", DEV, "--out", d], d + "/stdout.log")
        r = {"rc": rc}
        for f_ in ("mem_check.json", "timing.json"):
            if os.path.exists(os.path.join(d, f_)):
                r[f_[:-5]] = json.load(open(os.path.join(d, f_)))
        o[tag] = r
    o["pass"] = all(v.get("rc") == 0 and "timing" in v for v in o.values()); return o


def evalsmoke():
    """every inject_st_eval part on a 20-step LAT-ST model and a 20-step CD model (the 4060; TEST2 truncated to 20 per d): files written, per-item counts,
    CUE-WRONG / CUE-ONLY inputs decoded (one fixture each) - functional only (no reading)."""
    import boxes.inject_ow as IO
    import boxes.inject_st as ST
    import boxes.inject_st_eval as SE
    o = {}
    for tag, fl in (("lat", LAT2 + ["--cue", "step"]), ("cd", DIR2 + ["--cue", "all"])):
        d = G8.fresh(TMP + "/evsmoke_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_st"] + fl + ["--steps", "20", "--micro", "8", "--eval_bs", "50", "--eval_parts", "",
                                                                                                                 "--ckpt_every", "0", "--log_every", "10", "--device", DEV, "--out", d], d + "/stdout.log")
        S = IO.build_sets(); S["test2"] = {k: v[:20] for k, v in S["test2"].items()}
        SE.run(d, list(SE.PARTS), DEV, 50, sets=S); fs = sorted(f_ for f_ in os.listdir(d) if f_.startswith("eval"))
        o[tag] = {"rc": rc, "files": fs, "n_items_acc_d2": len(json.load(open(d + "/eval_acc.json"))["items"]["2"]["ans"])}
        if tag == "lat":
            cw = json.load(open(d + "/eval_cue_wrong.json")); co = json.load(open(d + "/eval_cue_only.json")); lp = json.load(open(d + "/eval_lp5.json"))
            o[tag].update({"cue_wrong_arms": sorted(cw["items"]["2"]), "cue_only_keys": sorted(co["items"]["2"]), "lp5_keys": sorted(lp["items"]["2"]["mean_at_1"])})
            it = ST.tok_st(S["test2"][2][0], "value", "step"); r = it["raw"]; sp = r["names"]["spine"]
            o[tag]["fixture"] = {"cue_only_pass1": IO.ENC.decode(IO.ENC.encode(" " + ST.cue_str(r, 1))), "cue_wrong_pass1_names": sp[2], "cue_wrong_pass2_names": sp[0]}
    o["pass"] = (o["lat"]["rc"] == 0 and o["cd"]["rc"] == 0 and set(o["lat"]["files"]) >= {"eval.json"} | {"eval_%s.json" % p_ for p_ in SE.PARTS}
                 and o["cd"]["files"] == ["eval.json", "eval_acc.json"] and {"all", "at_1", "at_2"} <= set(o["lat"]["cue_wrong_arms"]) and o["lat"]["lp5_keys"] == ["2", "8"])
    return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    fns = {"g1": g1, "g2": g2, "g2b": g2b, "g3": g3, "gcue": gcue, "g9": g9, "g4_g10": g4_g10, "g7": g7, "args": args_part, "f2": f2, "evalsmoke": evalsmoke}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, "b_" + part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
