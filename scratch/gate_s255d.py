"""Note 255-5 DISTRACT build gates (Audit's conditions a-d) -> scratch/gate_s255d/<part>.json ("pass"). The 4060 (cuda:0); det + fp32 + micro 8 for the bitwise parts.
Reference: boxes/_inject_aw_s252.py (the trainer before E255). Flags = rcb_plaw4_s1's (gate_s251.RCB); the pilot init = rcb_plaw4_s1/model.pt where stated.
  equiv    boxes.insert_render.insert_eval_sets == scratch/s255_insert.py's build (the G-D'd INSERT sets): every text and meta equal (7 conditions x 1,000).
  render   10,000 REG training-law items (Stream(25171, REG EXCL), d 1-4 cycling) through insert_render.distract(it, 1, i, 0): solve (inside), names disjoint (inside),
           every new given before every new op, S_1..S_d keep their relative order, n ~ U{0,1,2} shares; (a) tok_aw aux / ans / pass-cue tails == the undistracted item's;
           (b) no text in the INSERT eval texts.
  offbit   defaults: _inject_aw_s252.py vs the new module, det, 100 steps: ckpt (model / opt / rngs / stream) equal, logs equal.
  tiny     --distract_p 1e-12 (the distract path armed, no item ever changed = the inserts-0 builder) vs defaults, det, 30 steps: ckpt equal, logs equal minus distract_n.
  acts     --distract_p 1 vs defaults, det, stop 3: rng_d / stream_rng / stream_stats equal, model differs.
  smoke    the pilot flags (init rcb_plaw4_s1) the plain path, log every step, stop 40: (i) without the flag, (ii) with --distract_p 1: rc 9, losses finite; (ii) logs distract_n
           in the stop bin with n 0 / 1 / 2 all present.
  gradflow gate_s251's probe on the new module, the pilot flags + --distract_p 1, det, stop 8: the answer-CE grad nonzero into mem.qkv / gate / ln_w / wo on every d >= 2 call
           after step 1.
  memck    (d) --mem_check with --distract_p 1 (2-insert d_max items) at micro 8 / 16 on the 4060: train peaks.
  args     the pilot's --args_only vs rcb_plaw4_s1/args.json: {init_model(_sha / 256), distract_p} + module keys.
    python -B scratch/gate_s255d.py equiv render offbit tiny acts smoke gradflow memck args"""
import json
import math
import os
import sys

import numpy as np

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
import gate_s251 as G1  # noqa: E402
import s228_args_check as C  # noqa: E402

OUT = ROOT + "/scratch/gate_s255d"; TMP = ROOT + "/scratch/_tmp/s255d"; DEV = "cuda:0"; RCB = list(G1.RCB); KEYS = G1.KEYS; COM = ["--fp32", "--micro", "8"]


def pilot_flags():   # PILOT_INIT: the init path (the Vast cell ships an unbuffered-read copy, boxes/runs_fsq/rcb_plaw4_s1_init/model.pt; same sha)
    fl = list(RCB); i = fl.index("--init_model"); fl[i + 1] = os.environ.get("PILOT_INIT", "boxes/runs_fsq/rcb_plaw4_s1/model.pt"); j = fl.index("--init_model_sha"); fl[j + 1] = "67775e2492044a8a"; return fl


def det(mod, fl, d):
    G8.fresh(d); return G8.sh([sys.executable, "-B", "-m", "scratch.s222_det_wrap", mod] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")


recs = lambda p: [json.loads(l_) for l_ in open(p)] if os.path.exists(p) else []  # noqa: E731
strip = lambda p, drop=(): [{k: v for k, v in r.items() if k != "sec" and k not in drop} for r in recs(p)]  # noqa: E731


def equiv():
    import s255_insert as S
    import boxes.insert_render as IR
    from boxes import inject_ow as IO
    raw = IO.build_sets(mul_s_p=0)["test2"][4]; a, ma = S.build(raw); b, mb = IR.insert_eval_sets(raw)
    o = {"texts_equal": all([x["text"] for x in a[c]] == [x["text"] for x in b[c]] for c in S.CONDS), "meta_equal": ma == mb, "n": len(raw)}
    o["pass"] = o["texts_equal"] and o["meta_equal"]; return o


def render():
    import boxes.insert_render as IR
    import boxes.ow_words as OW
    from boxes import inject_ow as IO
    from boxes import inject_aw as AW
    s0 = IO.build_sets(mul_s_p=0); excl = s0["excl"]; ins = {it["text"] for v in IR.insert_eval_sets(s0["test2"][4])[0].values() for it in v}
    st = OW.Stream(25171, excl, frames=OW.FRAME_TRAIN, mul_s_p=0, **IO.OWKW); cnt = {0: 0, 1: 0, 2: 0}; bad = {"order": 0, "giv_before_op": 0, "targets": 0, "ins_hit": 0}
    for i in range(10000):
        d = 1 + i % 4; it = st.draw(d); o, n = IR.distract(it, 1, i, 0); cnt[n] += 1
        if n:
            sents_lab = None
            t1, t0 = AW.tok_aw(o, "value", "step", "answer"), AW.tok_aw(it, "value", "step", "answer")
            bad["targets"] += not (t1["aux"] == t0["aux"] and t1["ans"] == t0["ans"] and all(q1[len(t1["think"]):] == q0[len(t0["think"]):] for q1, q0 in zip(t1["think_cues"], t0["think_cues"])))
            sp = it["names"]["spine"]; pos = [o["text"].find(" " + sp[t] + " ") if False else None for t in range(1, d + 1)]; del pos, sents_lab
        bad["ins_hit"] += o["text"] in ins
    # order / given placement from the assembled label sequence (insert_render records it): S_1..S_d keep their order, every new given before S_1 and before every new op
    st = OW.Stream(25171, excl, frames=OW.FRAME_TRAIN, mul_s_p=0, **IO.OWKW)
    for i in range(10000):
        d = 1 + i % 4; it = st.draw(d); o, n = IR.distract(it, 1, i, 0)
        if not n:
            continue
        lab = o["insert_labels"]; sl = [x for x in lab if x.startswith("S") and x != "S0"]
        bad["order"] += sl != ["S%d" % t for t in range(1, d + 1)]
        ng = [lab.index("NG%d" % j) for j in range(n)]; no = [lab.index("NO%d" % j) for j in range(n)]
        bad["giv_before_op"] += not (max(ng) < lab.index("S1") and max(ng) < min(no))
    o = {"n": 10000, "n_dist": cnt, "bad": bad}; o["pass"] = not any(bad.values()) and all(abs(v / 10000 - 1 / 3) < 0.03 for v in cnt.values()); return o


def offbit():
    fl = RCB + COM + ["--ckpt_every", "100", "--log_every", "10", "--stop_at", "100"]
    da, db = TMP + "/off_ref", TMP + "/off_new"; o = {"rc": [det("boxes/_inject_aw_s252.py", fl, da), det("boxes/inject_aw.py", fl, db)]}
    if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
        o.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=KEYS)); o["log_equal"] = strip(da + "/log.jsonl") == strip(db + "/log.jsonl")
    o["pass"] = o["rc"] == [9, 9] and o.get("n_mismatch") == 0 and bool(o.get("log_equal")); return o


def tiny():
    fl = RCB + COM + ["--ckpt_every", "30", "--log_every", "10", "--stop_at", "30"]
    da, db = TMP + "/tiny_off", TMP + "/tiny_on"; o = {"rc": [det("boxes/inject_aw.py", fl, da), det("boxes/inject_aw.py", fl + ["--distract_p", "1e-12"], db)]}
    if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
        o.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=KEYS)); o["log_equal_minus_distract_n"] = strip(da + "/log.jsonl") == strip(db + "/log.jsonl", ("distract_n",))
    o["pass"] = o["rc"] == [9, 9] and o.get("n_mismatch") == 0 and bool(o.get("log_equal_minus_distract_n")); return o


def acts():
    fl = RCB + COM + ["--ckpt_every", "3", "--log_every", "1", "--stop_at", "3"]
    da, db = TMP + "/acts_off", TMP + "/acts_on"; o = {"rc": [det("boxes/inject_aw.py", fl, da), det("boxes/inject_aw.py", fl + ["--distract_p", "1"], db)]}
    if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
        o["stream"] = G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=("rng_d", "stream_rng", "stream_stats")); o["model"] = G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=("model",))
    o["pass"] = o["rc"] == [9, 9] and o.get("stream", {}).get("n_mismatch") == 0 and (o.get("model", {}).get("n_mismatch") or 0) > 0; return o


def smoke():
    o = {}
    for arm, ex in (("plain", []), ("distract", ["--distract_p", "1"])):
        d = TMP + "/smoke_" + arm; fl = pilot_flags() + ["--micro", "8"] + ex + ["--ckpt_every", "0", "--log_every", "1", "--stop_at", "40"]
        G8.fresh(d); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")
        rs = recs(d + "/log.jsonl"); ls = [r["loss"] for r in rs if isinstance(r.get("step"), int) and r.get("loss") is not None]; dn = [r["distract_n"] for r in rs if "distract_n" in r]
        o[arm] = {"rc": rc, "n_loss": len(ls), "finite": bool(ls) and all(math.isfinite(x) for x in ls), "distract_n": dn[-1] if dn else None}
        o[arm]["ok"] = rc == 9 and o[arm]["finite"] and len(ls) >= 40 and (arm == "plain" or (bool(dn) and all(k in dn[-1] for k in ("0", "1", "2"))))
    o["pass"] = o["plain"]["ok"] and o["distract"]["ok"]; return o


def gradflow():
    src = open(ROOT + "/boxes/inject_aw.py", encoding="utf-8").read(); anchor = "            loss = la + lam_now[0] * aux\n"; assert src.count(anchor) == 1
    open(ROOT + "/boxes/_inject_aw_gradprobe_s255d.py", "w", encoding="utf-8", newline="\n").write(src.replace(anchor, anchor + G1.PROBE.lstrip("\n")))
    d = TMP + "/gradflow"; G8.fresh(d); lp = d + "/gradprobe.jsonl"; G8.ENV["GRADPROBE_OUT"] = lp
    fl = pilot_flags() + COM + ["--distract_p", "1", "--ckpt_every", "0", "--log_every", "1", "--stop_at", "8"]
    rc = G8.sh([sys.executable, "-B", "-m", "scratch.s222_det_wrap", "boxes/_inject_aw_gradprobe_s255d.py"] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")
    G8.ENV.pop("GRADPROBE_OUT", None); rs = recs(lp); rows = [dict(r, step=1 + i // 8) for i, r in enumerate(rs)]; el = [r for r in rows if r["d"] >= 2 and r["step"] >= 2]
    req = ("qkv.weight", "gate.weight", "ln_w.weight", "wo.weight")
    o = {"rc": rc, "n_calls": len(rs), "n_eligible": len(el), "all_nonzero": bool(el) and all(all((r["la_grad_mem"].get(k) or 0) > 0 for k in req) for r in el)}
    o["pass"] = rc == 9 and o["all_nonzero"]; return o


def memck():
    o = {}
    for mi in (8, 16):
        d = G8.fresh(TMP + "/memck_%d" % mi)
        rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + pilot_flags() + ["--distract_p", "1", "--micro", str(mi), "--eval_bs", "50", "--mem_check", "--device", DEV, "--out", d],
                   d + "/stdout.log")
        mc = json.load(open(d + "/mem_check.json")) if os.path.exists(d + "/mem_check.json") else {}; o[str(mi)] = {"rc": rc, "peak_gib_train": mc.get("peak_gib_train")}
    o["pass"] = any((v["peak_gib_train"] or 99) <= 6.8 for v in o.values()); return o


def args():
    ref = json.load(open("boxes/runs_fsq/rcb_plaw4_s1/args.json")); d = G8.fresh(TMP + "/args_pilot")
    rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + pilot_flags() + ["--distract_p", "1", "--micro", "8", "--eval_bs", "250", "--stop_at", "2000", "--ckpt_every", "1000",
               "--log_every", "100", "--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
    new = json.load(open(d + "/args.json")); dif = {k: v for k, v in C.diff(ref, new).items() if k != "pair_stats"}
    o = {"rc": rc, "keys": sorted(dif), "diff": dif}; o["pass"] = rc == 0 and "distract_p" in dif; return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    fns = {k: globals()[k] for k in ("equiv", "render", "offbit", "tiny", "acts", "smoke", "gradflow", "memck", "args")}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
