"""§262 NAME-KEYED MEMORY host gates (NK_v1 §7 G-D + Note 262-1) -> scratch/gate_s269/<part>.json ("pass"); run on each pilot host by scratch.s269_gates.
Flags: BUN = gate_s257.dch_flags(1, the bun_s1 copy) + --hold_inc 1.6 --hold_log (bun's recipe: cue step, --distract_p 1 --distract_mix); PILOT = BUN + --perm_p 0.5
--ask_p 0.25 (+ --nk 1 for the NK arm). det + fp32 + micro 1 for the bitwise parts (s222_det_wrap).
  offbit_aw    flags-off bitwise: the pre-§262 code (scratch/s269_ref.tar = git archive <pre-§262> boxes/*.py (top level) + scratch/s222_det_wrap.py, its own tree
               scratch/_ref269 with prosqa/ and boxes/runs_fsq/ linked) vs the new tree, BUN, 60 steps: ckpt (model / opt / rngs / stream) and logs equal.
  offbit_perm  the same on the --perm_p-only path (BUN without the distract flags + --perm_p 0.5).
  step0        in-process, det, fp32: CTRL (nk 0) and NK (nk 1) built from the PILOT args and the bun_s1 init (NK: strict except nk.*); on 50 REG TEST2 items per d 1-4,
               50 TEST-PERM items per d 3-4 and 50 ask items per d 1-4: at every pass 1..K the DeltaMem state S[:, :NB], the block-11 h and the ':' logits, the decoder
               logits at K' 0..K and the aux logits at t <= d are bitwise equal (W_on = 0); the hold stats (hold_step, no penalty) of CTRL == NK's on S[:, :NB] bitwise
               (NK's full-S stats reported: they include the NK block by design). fp64 is not used: deltamem's branch casts to fp32 itself; exact equality is required.
  ids          NK.read is called exactly once per forward of pass_aw / dec_logits / aux_logits / aux_greedy (one per generated token) with ids (B, T) == the hidden's;
               a read without ids and a write keyed by a non-name token raise.
  smoke_nk     boxes.inject_aw PILOT + --nk 1, stop 20, log every step: rc 9, losses finite, the stop-step bin carries ask_acc_by_pass (K passes) and ws_readout_perm.
  smoke_ctrl   the same without --nk.
  gradflow     a gate-only instrumented copy (boxes/_inject_aw_gradprobe_s269.py: after each step's backward, the grad norms of nk.W_n / W_vn / W_on), PILOT + --nk 1, det,
               stop 3: step 1 W_on > 0 (W_n / W_vn exactly 0: W_on = 0); steps 2-3 W_n, W_vn, W_on > 0.
    python -B scratch/gate_s269.py offbit_aw offbit_perm step0 ids smoke_nk smoke_ctrl gradflow      (host)"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile

ROOT = os.environ.get("LATENT_ROOT", "D:/Latent")
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
import gate_s251 as G1  # noqa: E402
import gate_s257 as G7  # noqa: E402

OUT = ROOT + "/scratch/gate_s269"; TMP = ROOT + "/scratch/_tmp/s269g"; DEV = os.environ.get("G269_DEV", "cuda:0"); KEYS = G1.KEYS; COM = ["--fp32", "--micro", "1"]
ST = int(os.environ.get("G269_STOP", "0"))                                  # local CPU smoke only: every training part stops at ST steps
OUT = OUT if not ST else ROOT + "/scratch/_tmp/s269g_local"                 # the local smoke never writes the shipped scratch/gate_s269/<part>.json
REFT = ROOT + "/scratch/_ref269"; INIT = ("boxes/runs_fsq/bun_s1_init/model.pt", "1ad3517b7635fddf")
recs = lambda p: [json.loads(l_) for l_ in open(p)] if os.path.exists(p) else []  # noqa: E731
strip = lambda p: [{k: v for k, v in r.items() if k != "sec"} for r in recs(p)]  # noqa: E731


def bun():
    return G7.dch_flags(1, *INIT) + ["--hold_inc", "1.6", "--hold_log"]


def pilot(nk):
    return bun() + ["--perm_p", "0.5", "--ask_p", "0.25"] + (["--nk", "1"] if nk else [])


def drop(fl, keys):
    o = list(fl)
    for k in keys:
        i = o.index(k); del o[i:i + 2]
    return o


def ref_tree():
    if not os.path.isdir(REFT + "/boxes"):
        os.makedirs(REFT, exist_ok=True); tarfile.open(ROOT + "/scratch/s269_ref.tar").extractall(REFT)
    lnk = lambda q: os.path.islink(q) or os.path.isjunction(q)  # noqa: E731   a junction (local Windows smoke) is a link too: never rmtree through it
    for sub in ("prosqa", "boxes/runs_fsq"):
        q = REFT + "/" + sub
        if os.path.isdir(q) and not lnk(q):
            shutil.rmtree(q)
        if not lnk(q):
            try:
                os.symlink(ROOT + "/" + sub, q)
            except OSError:                                                 # Windows without the symlink privilege (local smoke): a directory junction
                assert os.name == "nt"; subprocess.check_call(["cmd", "/c", "mklink", "/J", q.replace("/", "\\"), (ROOT + "/" + sub).replace("/", "\\")], stdout=subprocess.DEVNULL)
        assert lnk(q), ("ref_tree: not a link", q)
    return REFT


def det_in(tree, mod, fl, d):
    G8.fresh(d); return subprocess.call([sys.executable, "-B", "-m", "scratch.s222_det_wrap", mod] + fl + ["--device", DEV, "--out", d], cwd=tree, env=G8.ENV,
                                        stdout=open(d + "/stdout.log", "w"), stderr=subprocess.STDOUT)


def _offbit(fl, tag):
    fl = fl + COM + ["--ckpt_every", str(ST or 60), "--log_every", "10", "--stop_at", str(ST or 60)]; da, db = TMP + "/%s_ref" % tag, TMP + "/%s_new" % tag
    o = {"rc": [det_in(ref_tree(), "boxes/inject_aw.py", fl, da), det_in(ROOT, "boxes/inject_aw.py", fl, db)]}
    if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
        o.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=KEYS)); o["log_equal"] = strip(da + "/log.jsonl") == strip(db + "/log.jsonl")
    o["pass"] = o["rc"] == [9, 9] and o.get("n_mismatch") == 0 and bool(o.get("log_equal")); return o


def offbit_aw():
    return _offbit(bun(), "offaw")


def offbit_perm():
    return _offbit(drop(bun(), ("--distract_p", "--distract_mix")) + ["--perm_p", "0.5"], "offperm")


def _args(nk):
    d = G8.fresh(TMP + "/args_%d" % nk)
    rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + pilot(nk) + ["--stop_at", "2000", "--device", "cpu", "--args_only", "--out", d], d + "/stdout.log"); assert rc == 0, rc
    return json.load(open(d + "/args.json"))


def _models():
    import torch
    from boxes import inject_aw as AW
    out = {}
    for nk in (0, 1):
        a = _args(nk); torch.manual_seed(a["seed"]); m, g, params = AW.build_aw(argparse.Namespace(**{**a, "init_trunk": "", "init_sha": ""}), DEV)
        mk_ = params.load_state_dict(torch.load(INIT[0], map_location=DEV, weights_only=True), strict=not nk)
        assert not nk or (not mk_.unexpected_keys and sorted(mk_.missing_keys) == ["nk.W_n.weight", "nk.W_on.weight", "nk.W_vn.weight"]), mk_
        params.eval(); out[nk] = (a, m, g)
    return out


def _items(a):
    import numpy as np
    import boxes.inject_aw as AW
    import boxes.inject_aw_eval as EV
    import boxes.ow_words as OW
    import boxes.perm_render as PR
    from boxes.inject_ow import OWKW
    sets = __import__("boxes.inject_ow", fromlist=["build_sets"]).build_sets(mul_s_p=0); T = EV.tokd({"test2": {d: sets["test2"][d][:50] for d in (1, 2, 3, 4)}}, a, (1, 2, 3, 4))
    P_, _ = PR.test_perm_sets({d: sets["test2"][d][:50] for d in (3, 4)}, (3, 4)); TP = EV.tokd({"test2": P_}, a, (3, 4))
    s = OW.Stream(OW.SEED_TRAIN + 11, set(), frames=OW.FRAME_TRAIN, mul_s_p=0, **OWKW); AS = {}
    for d in (1, 2, 3, 4):
        AS[d] = []
        for k in range(50):
            r = s.draw(d)
            if d >= 2 and k % 2:
                r, _, _ = PR.permuted(r, np.random.default_rng((1, 98, d, k)))
            AS[d].append(AW.tok_ask(r, a["aux_target"], a["write_src"]))
    return [("test2_d%d" % d, d, T[d]) for d in T] + [("perm_d%d" % d, d, TP[d]) for d in TP] + [("ask_d%d" % d, d, AS[d]) for d in AS]


def step0():
    import torch
    import scratch.s222_det_wrap as DW
    import boxes.inject_aw as AW
    from boxes.inject_st import cue_inputs
    DW.apply(); M = _models(); a, mc, g0 = M[0]; _, mn, g1 = M[1]; K = a["think_k"]; NB = mn.NB; o = {"sets": {}, "hold": {}}; ok = True
    eq = lambda x, y: bool(torch.equal(x, y))  # noqa: E731
    with torch.no_grad():
        for name, d, its in _items(a):
            r = {"n": len(its), "S12": True, "h": True, "colon": True, "dec": True, "aux": True}; ask = name.startswith("ask")
            xs, msk = cue_inputs(mc, its, DEV); xn, mn_ = cue_inputs(mn, its, DEV); assert all(eq(p, q) for p, q in zip(xs, xn)) and eq(msk, mn_)
            Sc = mc.mem.zero_state(len(its), DEV); Sn = mn.mem.zero_state(len(its), DEV); assert Sn.shape[1] == NB + 1 and Sc.shape[1] == NB
            r["dec"] &= ask or eq(mc.dec_logits(Sc, its, DEV), mn.dec_logits(Sn, its, DEV)); keep_c = [Sc]; keep_n = [Sn]
            for t in range(1, K + 1):
                j = min(t, len(xs)) - 1; Sc, hc, pc = mc.pass_aw(Sc, xs[j], msk, mc.lower(xs[j])); Sn, hn, pn = mn.pass_aw(Sn, xs[j], msk, mn.lower(xs[j]))
                r["S12"] &= eq(Sc, Sn[:, :NB]); r["h"] &= eq(hc, hn); r["colon"] &= eq(mc.colon_logits(hc, pc), mn.colon_logits(hn, pn)); keep_c.append(Sc); keep_n.append(Sn)
                if not ask:
                    r["dec"] &= eq(mc.dec_logits(Sc, its, DEV), mn.dec_logits(Sn, its, DEV))
                    if t <= d:
                        lc, yc = mc.aux_logits(Sc, its, t, DEV); ln_, yn = mn.aux_logits(Sn, its, t, DEV); r["aux"] &= eq(lc, ln_) and eq(yc, yn)
            r["nk_block_norm_K"] = float(Sn[:, NB].float().norm(dim=(1, 2)).mean())
            if not ask and K >= 3:                                          # the hold stats (no penalty): CTRL vs NK on S[:, :NB] bitwise; NK full-S reported
                hc_ = AW.hold_step(mc, [(keep_c, xs, msk, [d] * len(its))], [], 0, K, len(its), False, 0.0, _nullc, DEV)
                hn12 = AW.hold_step(mn, [([s_[:, :NB] for s_ in keep_n], xs, msk, [d] * len(its))], [], 0, K, len(its), False, 0.0, _nullc, DEV)
                hnf = AW.hold_step(mn, [(keep_n, xs, msk, [d] * len(its))], [], 0, K, len(its), False, 0.0, _nullc, DEV)
                r["hold_equal_S12"] = hc_ == hn12; o["hold"][name] = {"ctrl": hc_, "nk_S12": hn12, "nk_full": hnf}; ok &= r["hold_equal_S12"]
            ok &= all(r[k_] for k_ in ("S12", "h", "colon", "dec", "aux")); o["sets"][name] = r
    o["pass"] = bool(ok); return o


def _nullc():
    import contextlib
    return contextlib.nullcontext()


def ids():
    import torch
    M = _models(); a, m, g = M[1]; its = [x for x in _items(a) if x[0] == "test2_d3"][0][2][:8]; from boxes.inject_st import cue_inputs
    calls = []; orig = m.nk.read

    def rd(g_, h, ids_, Sn):
        calls.append((tuple(ids_.shape) if ids_ is not None else None, tuple(h.shape[:2]))); return orig(g_, h, ids_, Sn)
    m.nk.read = rd; o = {}
    with torch.no_grad():
        xs, msk = cue_inputs(m, its, DEV); S = m.mem.zero_state(len(its), DEV)
        calls.clear(); S, h, pos = m.pass_aw(S, xs[0], msk, m.lower(xs[0])); o["pass_aw"] = list(calls)
        calls.clear(); m.dec_logits(S, its, DEV); o["dec_logits"] = list(calls)
        calls.clear(); m.aux_logits(S, its, 1, DEV); o["aux_logits"] = list(calls)
        calls.clear(); m.aux_greedy(S, 1, 3); o["aux_greedy"] = list(calls)
        m.nk.read = orig; neg = {}
        for nm, fn in (("read_no_ids", lambda: m.nk.read(g, torch.zeros(2, 5, g.w("wte.weight").shape[1], device=DEV), None, torch.zeros(2, 64, 64, device=DEV))),
                       ("write_non_name", lambda: m.nk.write(g, torch.zeros(1, 64, 64, device=DEV), torch.tensor([13], device=DEV), torch.zeros(1, g.w("wte.weight").shape[1], device=DEV)))):
            try:
                fn(); neg[nm] = False
            except AssertionError:
                neg[nm] = True
    one = lambda c: len(c) == 1 and c[0][0] == c[0][1]  # noqa: E731
    o["negative_raise"] = neg; o["pass"] = one(o["pass_aw"]) and one(o["dec_logits"]) and one(o["aux_logits"]) and len(o["aux_greedy"]) == 3 and all(c[0] == c[1] for c in o["aux_greedy"]) \
        and all(neg.values())
    return o


def _smoke(nk, tag):
    d = TMP + "/" + tag; G8.fresh(d); stop = ST or 20
    rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + pilot(nk) + ["--ckpt_every", "0", "--log_every", "1", "--stop_at", str(stop), "--device", DEV, "--out", d],
               d + "/stdout.log"); rs = recs(d + "/log.jsonl")
    ls = [r["loss"] for r in rs if isinstance(r.get("step"), int) and r.get("loss") is not None]; br = [r for r in rs if "bin" in r]
    a = json.load(open(d + "/args.json")) if os.path.exists(d + "/args.json") else {}
    o = {"rc": rc, "n_loss": len(ls), "finite": bool(ls) and all(math.isfinite(x) for x in ls), "nk": a.get("nk"), "ask_p": a.get("ask_p"),
         "ask_acc_by_pass_keys": sorted((br[-1].get("ask_acc_by_pass") or {}).keys(), key=int) if br else None, "ask_n": br[-1].get("ask_n") if br else None,
         "has_ws_readout_perm": bool(br) and "ws_readout_perm" in br[-1], "distract_n": br[-1].get("distract_n") if br else None}
    o["pass"] = (rc == 9 and o["finite"] and len(ls) >= stop and o["nk"] == nk and o["ask_acc_by_pass_keys"] == [str(t) for t in range(1, 9)] and o["has_ws_readout_perm"])
    return o


def smoke_nk():
    return _smoke(1, "smoke_nk")


def smoke_ctrl():
    return _smoke(0, "smoke_ctrl")


PROBE = '''        if os.environ.get("GRADPROBE_NK"):                                 # §262 GATE-ONLY (gate_s269 gradflow): the step's total NK grads
            _gn = lambda p_: (float(p_.grad.float().norm()) if p_.grad is not None else None)  # noqa: E731
            open(os.environ["GRADPROBE_NK"], "a").write(json.dumps({"step": step, "W_n": _gn(m.nk.W_n.weight), "W_vn": _gn(m.nk.W_vn.weight), "W_on": _gn(m.nk.W_on.weight)}) + "\\n")
'''


def gradflow():
    src = open(ROOT + "/boxes/inject_aw.py", encoding="utf-8").read(); anchor = "        if m is not None:\n            wwin.append(write_grad_norm(m))\n"; assert src.count(anchor) == 1
    open(ROOT + "/boxes/_inject_aw_gradprobe_s269.py", "w", encoding="utf-8", newline="\n").write(src.replace(anchor, PROBE + anchor))
    d = TMP + "/gradflow"; G8.fresh(d); lp = d + "/gradprobe.jsonl"; G8.ENV["GRADPROBE_NK"] = lp
    fl = pilot(1) + COM + ["--ckpt_every", "0", "--log_every", "1", "--stop_at", "3"]
    rc = G8.sh([sys.executable, "-B", "-m", "scratch.s222_det_wrap", "boxes/_inject_aw_gradprobe_s269.py"] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")
    G8.ENV.pop("GRADPROBE_NK", None); rs = recs(lp); by = {r["step"]: r for r in rs}
    o = {"rc": rc, "records": rs}
    o["pass"] = (rc == 9 and set(by) == {1, 2, 3} and (by[1]["W_on"] or 0) > 0 and by[1]["W_n"] == 0.0 and by[1]["W_vn"] == 0.0
                 and all((by[s]["W_n"] or 0) > 0 and (by[s]["W_vn"] or 0) > 0 and (by[s]["W_on"] or 0) > 0 for s in (2, 3)))
    return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    fns = {"offbit_aw": offbit_aw, "offbit_perm": offbit_perm, "step0": step0, "ids": ids, "smoke_nk": smoke_nk, "smoke_ctrl": smoke_ctrl, "gradflow": gradflow}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
