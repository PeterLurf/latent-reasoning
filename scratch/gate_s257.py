"""§257 DISTRACT-CHAIN build gates (DC2_v1) -> scratch/gate_s257d/<part>.json ("pass"). Reference trainer: boxes/_inject_aw_s256.py (= boxes/inject_aw.py before E257).
LOCAL (CPU):
  equiv256 boxes.insert_render (§257 edits) vs its HEAD~ text (git show <REF>:boxes/insert_render.py): insert_eval_sets texts + metas equal; held_out_sets texts equal (== the
           committed scratch/s256_ho_shas.json) and every old meta key / value kept (the new keys c / v / j / op / op_idx only added); distract() equal on 2,000 items.
  shas     held_out2_sets built in two fresh processes: identical -> scratch/s257_ho_shas.json (held_out2 + the per-set item metas' sha); solve inside.
  render   10,000 REG training-law items (Stream(25171, REG EXCL), d 1-4 cycling) through distract_mix(it, 1, i, 0): solve (inside); aux / ans / pass-cue tails == the undistracted
           item's; S_1..S_d keep their order; chain: the run of labels right after S_j holds it (only chain ops between); givop: its given directly before its op; s256: its given
           before S1 and before every new op; names new + distinct; d = 1: no chain; n ~ U{0,1,2} and types ~ U (d >= 2) within 0.03; no text in the INSERT / §256 / §257 eval sets.
  args     dch_s<k>'s --args_only vs dis_s<k>/args.json (the frozen dch args): the expected keys only.
HOST (the Vast cell's eval-only line, s257_guarded --gates; PILOT_INIT = the dis_s<k> copy; det + fp32 + micro 8):
  offbit   defaults: _inject_aw_s256.py vs the new module, 100 steps: ckpt (model / opt / rngs / stream) equal, logs equal.
  offbit_d --distract_p 1 (no mix) on both, 30 steps: ckpt equal, logs equal (the §256 path unchanged by E257).
  acts     --distract_p 1 vs --distract_p 1 --distract_mix s256,chain,givop, stop 3: rng_d / stream_rng / stream_stats equal, model differs.
  smoke    the dch flags (init dis_s<k>) + mix, log every step, stop 40: rc 9, losses finite, the stop bin's distract_types has s256 / chain / givop and distract_n 0 / 1 / 2.
  gradflow gate_s251's probe on the new module, the dch flags, det, stop 8: the answer-CE grad nonzero into mem.qkv / gate / ln_w / wo on every d >= 2 call after step 1.
    python -B scratch/gate_s257.py equiv256 shas render args      (local)        python -B scratch/gate_s257.py offbit offbit_d acts smoke gradflow   (host)"""
import hashlib
import json
import math
import os
import subprocess
import sys

ROOT = os.environ.get("LATENT_ROOT", "D:/Latent")
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
import gate_s251 as G1  # noqa: E402
import s228_args_check as C  # noqa: E402

OUT = ROOT + "/scratch/gate_s257d"; TMP = ROOT + "/scratch/_tmp/s257d"; DEV = "cuda:0"; RCB = list(G1.RCB); KEYS = G1.KEYS; COM = ["--fp32", "--micro", "8"]
MIX = "s256,chain,givop"; REF_COMMIT = os.environ.get("S257_REF", "c7db8d6d")   # a commit before the §257 insert_render edits


def dch_flags(k, init, sha):
    """dis_s<k>'s flags (s257_mkq.flags: RCB + seed + --distract_p 1, micro 2, eval_bs 250) with the init = dis_s<k>'s copy, + --distract_mix, --steps 8000."""
    fl = list(RCB); i = fl.index("--init_model"); fl[i + 1] = init; j = fl.index("--init_model_sha"); fl[j + 1] = sha; s = fl.index("--steps"); fl[s + 1] = "8000"
    return fl + (["--seed", "2"] if k == 2 else []) + ["--distract_p", "1", "--distract_mix", MIX, "--micro", "2", "--eval_bs", "250"]


def pilot_flags():
    return dch_flags(int(os.environ.get("PILOT_K", "1")), os.environ["PILOT_INIT"], os.environ["PILOT_INIT_SHA"])


def det(mod, fl, d):
    G8.fresh(d); return G8.sh([sys.executable, "-B", "-m", "scratch.s222_det_wrap", mod] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")


recs = lambda p: [json.loads(l_) for l_ in open(p)] if os.path.exists(p) else []  # noqa: E731
strip = lambda p, drop=(): [{k: v for k, v in r.items() if k != "sec" and k not in drop} for r in recs(p)]  # noqa: E731


def _old_ir():
    src = subprocess.check_output(["git", "show", REF_COMMIT + ":boxes/insert_render.py"]).decode("utf-8"); assert "def distract_mix" not in src, "REF_COMMIT already has §257"
    p = ROOT + "/boxes/_insert_render_s256ref.py"; open(p, "w", encoding="utf-8", newline="\n").write(src)
    import boxes._insert_render_s256ref as OLD
    return OLD


def equiv256():
    import boxes.insert_render as IR
    import boxes.ow_words as OW
    from boxes import inject_ow as IO
    OLD = _old_ir(); s0 = IO.build_sets(mul_s_p=0); raw = s0["test2"][4]
    a, ma = OLD.insert_eval_sets(raw); b, mb = IR.insert_eval_sets(raw)
    o = {"insert_texts_equal": all([x["text"] for x in a[c]] == [x["text"] for x in b[c]] for c in IR.CONDS), "insert_meta_equal": ma == mb}
    ha, hma = OLD.held_out_sets(raw); hb, hmb = IR.held_out_sets(raw)
    o["ho_texts_equal"] = OLD.set_shas(ha) == IR.set_shas(hb); o["ho_shas_eq_committed"] = IR.set_shas(hb) == json.load(open("scratch/s256_ho_shas.json"))["held_out"]
    o["ho_meta_superset"] = all(all(mo[k] == mn[k] for k in mo) for f in hma for p in hma[f] for mo, mn in zip(hma[f][p], hmb[f][p]))
    st = OW.Stream(25171, s0["excl"], frames=OW.FRAME_TRAIN, mul_s_p=0, **IO.OWKW); bad = 0
    for i in range(2000):
        it = st.draw(1 + i % 4); x, nx = OLD.distract(it, 1, i, 0); y, ny = IR.distract(it, 1, i, 0); bad += (nx != ny) or (x["text"] != y["text"])
    o["distract_mismatch"] = bad; o["pass"] = all(v is True for k, v in o.items() if k != "distract_mismatch") and bad == 0; return o


def _build2():
    import boxes.insert_render as IR
    from boxes import inject_ow as IO
    H, M = IR.held_out2_sets(IO.build_sets(mul_s_p=0)["test2"][4])
    return {"held_out2": IR.set_shas(H), "meta_sha": hashlib.sha256(json.dumps(M, sort_keys=True).encode()).hexdigest(), "n": {f: {p: len(v) for p, v in d.items()} for f, d in H.items()}}


def shas():
    outs = []
    for r in range(2):
        f = TMP + "/shas_build%d.json" % r
        rc = subprocess.call([sys.executable, "-B", "-c", "import sys, json; sys.argv = ['x']; sys.path.insert(0, 'scratch'); import gate_s257 as G; "
                              "json.dump(G._build2(), open(%r, 'w'))" % f]); outs.append(json.load(open(f)) if rc == 0 else None)
    o = {"identical": outs[0] is not None and outs[0] == outs[1], "n": outs[0] and outs[0]["n"]}
    if o["identical"]:
        json.dump({"held_out2": outs[0]["held_out2"], "meta_sha": outs[0]["meta_sha"], "n": outs[0]["n"], "built": "two fresh local processes, identical (gate_s257 shas)"},
                  open("scratch/s257_ho_shas.json", "w", newline="\n"), indent=1)
    o["pass"] = o["identical"]; return o


def render():
    import boxes.insert_render as IR
    import boxes.ow_words as OW
    from boxes import inject_ow as IO
    from boxes import inject_aw as AW
    s0 = IO.build_sets(mul_s_p=0); raw = s0["test2"][4]
    ev = ({x["text"] for v in IR.insert_eval_sets(raw)[0].values() for x in v} | {x["text"] for d_ in IR.held_out_sets(raw)[0].values() for v in d_.values() for x in v}
          | {x["text"] for d_ in IR.held_out2_sets(raw)[0].values() for v in d_.values() for x in v})
    st = OW.Stream(25171, s0["excl"], frames=OW.FRAME_TRAIN, mul_s_p=0, **IO.OWKW); cnt = {0: 0, 1: 0, 2: 0}; ty = {t: 0 for t in IR.MIX_TYPES}
    bad = {k: 0 for k in ("targets", "order", "chain_adjacent", "givop_adjacent", "s256_given", "names", "d1_chain", "ev_hit")}
    for i in range(10000):
        d = 1 + i % 4; it = st.draw(d); o, n = IR.distract_mix(it, 1, i, 0); cnt[n] += 1; bad["ev_hit"] += o["text"] in ev
        if not n:
            continue
        t1, t0 = AW.tok_aw(o, "value", "step", "answer"), AW.tok_aw(it, "value", "step", "answer")
        bad["targets"] += not (t1["aux"] == t0["aux"] and t1["ans"] == t0["ans"] and len(t1["think_cues"]) == len(t0["think_cues"])
                               and all(q1[len(t1["think"]):] == q0[len(t0["think"]):] for q1, q0 in zip(t1["think_cues"], t0["think_cues"])))
        lab = o["insert_labels"]; bad["order"] += [x for x in lab if x.startswith("S") and x != "S0"] != ["S%d" % t for t in range(1, d + 1)]
        used = set(it["names"]["spine"]) | set(it["names"]["givens"]) | set(it["names"]["distractors"]); new = []
        for k, m_ in enumerate(o["distract_meta"]):
            t = m_["type"]; ty[t] += d >= 2; no = lab.index("NO%d" % k); new += [m_["X"]] + ([m_["Y"]] if t != "chain" else [])
            if t == "chain":
                sj = lab.index("S%d" % m_["j"]); bad["d1_chain"] += d == 1
                bad["chain_adjacent"] += not (no > sj and all(x.startswith("NO") and o["distract_meta"][int(x[2:])]["type"] == "chain" and o["distract_meta"][int(x[2:])]["j"] == m_["j"]
                                                              for x in lab[sj + 1:no]))
            elif t == "givop":
                bad["givop_adjacent"] += lab.index("NG%d" % k) != no - 1
            else:
                ng = lab.index("NG%d" % k); bad["s256_given"] += not (ng < lab.index("S1") and all(ng < lab.index(x) for x in lab if x.startswith("NO")))
        bad["names"] += len(set(new)) != len(new) or bool(set(new) & used)
    tot = sum(ty.values())
    o = {"n": 10000, "n_dist": cnt, "types_d2plus": ty, "bad": bad}
    o["pass"] = not any(bad.values()) and all(abs(v / 10000 - 1 / 3) < 0.03 for v in cnt.values()) and all(abs(v / tot - 1 / 3) < 0.03 for v in ty.values()); return o


def args():
    o = {}
    for k in (1, 2):
        ref = json.load(open("boxes/runs_fsq/dis_s%d/args.json" % k)); d = G8.fresh(TMP + "/args_dch_s%d" % k)
        rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + dch_flags(k, *INIT_SHA[k]) + ["--keep_final_ckpt", "--ckpt_every", "1000", "--eval_parts", EVAL,
                   "--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
        new = json.load(open(d + "/args.json")); dif = {kk: v for kk, v in C.diff(ref, new).items() if kk != "pair_stats"}
        o["dch_s%d" % k] = {"rc": rc, "keys": sorted(dif), "diff": dif}
    exp = {"init_model", "init_model_sha", "init_model_sha256", "steps", "distract_mix"}
    o["pass"] = all(v["rc"] == 0 and set(v["keys"]) <= exp and {"steps", "distract_mix", "init_model"} <= set(v["keys"]) for v in o.values()); return o


EVAL = "acc,lp5,mean_s,cue_only,cue_wrong,swap_k,swap_1,probe,ws,lp5h,swap_kd,test_perm,insert_ho,ws_deep"
INIT_SHA = {}                                                               # filled by s257_mkq (k -> (path, sha16)) before args()


def offbit():
    fl = RCB + COM + ["--ckpt_every", "100", "--log_every", "10", "--stop_at", "100"]
    da, db = TMP + "/off_ref", TMP + "/off_new"; o = {"rc": [det("boxes/_inject_aw_s256.py", fl, da), det("boxes/inject_aw.py", fl, db)]}
    if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
        o.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=KEYS)); o["log_equal"] = strip(da + "/log.jsonl") == strip(db + "/log.jsonl")
    o["pass"] = o["rc"] == [9, 9] and o.get("n_mismatch") == 0 and bool(o.get("log_equal")); return o


def offbit_d():
    fl = RCB + COM + ["--distract_p", "1", "--ckpt_every", "30", "--log_every", "10", "--stop_at", "30"]
    da, db = TMP + "/offd_ref", TMP + "/offd_new"; o = {"rc": [det("boxes/_inject_aw_s256.py", fl, da), det("boxes/inject_aw.py", fl, db)]}
    if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
        o.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=KEYS)); o["log_equal"] = strip(da + "/log.jsonl") == strip(db + "/log.jsonl")
    o["pass"] = o["rc"] == [9, 9] and o.get("n_mismatch") == 0 and bool(o.get("log_equal")); return o


def acts():
    fl = RCB + COM + ["--distract_p", "1", "--ckpt_every", "3", "--log_every", "1", "--stop_at", "3"]
    da, db = TMP + "/acts_off", TMP + "/acts_on"; o = {"rc": [det("boxes/inject_aw.py", fl, da), det("boxes/inject_aw.py", fl + ["--distract_mix", MIX], db)]}
    if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
        o["stream"] = G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=("rng_d", "stream_rng", "stream_stats")); o["model"] = G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=("model",))
    o["pass"] = o["rc"] == [9, 9] and o.get("stream", {}).get("n_mismatch") == 0 and (o.get("model", {}).get("n_mismatch") or 0) > 0; return o


def smoke():
    d = TMP + "/smoke_mix"; fl = pilot_flags() + ["--micro", "8", "--ckpt_every", "0", "--log_every", "1", "--stop_at", "40"]
    G8.fresh(d); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")
    rs = recs(d + "/log.jsonl"); ls = [r["loss"] for r in rs if isinstance(r.get("step"), int) and r.get("loss") is not None]
    dn = [r["distract_n"] for r in rs if "distract_n" in r]; dt = [r["distract_types"] for r in rs if "distract_types" in r]
    o = {"rc": rc, "n_loss": len(ls), "finite": bool(ls) and all(math.isfinite(x) for x in ls), "distract_n": dn[-1] if dn else None, "distract_types": dt[-1] if dt else None}
    o["pass"] = rc == 9 and o["finite"] and len(ls) >= 40 and bool(dn) and all(k in dn[-1] for k in ("0", "1", "2")) and bool(dt) and all(t in dt[-1] for t in ("s256", "chain", "givop"))
    return o


def gradflow():
    src = open(ROOT + "/boxes/inject_aw.py", encoding="utf-8").read(); anchor = "            loss = la + lam_now[0] * aux\n"; assert src.count(anchor) == 1
    open(ROOT + "/boxes/_inject_aw_gradprobe_s257.py", "w", encoding="utf-8", newline="\n").write(src.replace(anchor, anchor + G1.PROBE.lstrip("\n")))
    d = TMP + "/gradflow"; G8.fresh(d); lp = d + "/gradprobe.jsonl"; G8.ENV["GRADPROBE_OUT"] = lp
    fl = pilot_flags() + COM + ["--ckpt_every", "0", "--log_every", "1", "--stop_at", "8"]
    rc = G8.sh([sys.executable, "-B", "-m", "scratch.s222_det_wrap", "boxes/_inject_aw_gradprobe_s257.py"] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")
    G8.ENV.pop("GRADPROBE_OUT", None); rs = recs(lp); rows = [dict(r, step=1 + i // 8) for i, r in enumerate(rs)]; el = [r for r in rows if r["d"] >= 2 and r["step"] >= 2]
    req = ("qkv.weight", "gate.weight", "ln_w.weight", "wo.weight")
    o = {"rc": rc, "n_calls": len(rs), "n_eligible": len(el), "all_nonzero": bool(el) and all(all((r["la_grad_mem"].get(k) or 0) > 0 for k in req) for r in el)}
    o["pass"] = rc == 9 and o["all_nonzero"]; return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    if os.environ.get("S257_INIT_SHA"):                                     # "1=<path>=<sha16>,2=<path>=<sha16>" for args()
        INIT_SHA.update({int(x.split("=")[0]): tuple(x.split("=")[1:]) for x in os.environ["S257_INIT_SHA"].split(",")})
    fns = {k: globals()[k] for k in ("equiv256", "shas", "render", "args", "offbit", "offbit_d", "acts", "smoke", "gradflow")}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
