"""§261 TARGET-CUE host gates (Audit G-D b412 condition (1)) -> scratch/gate_s268/<part>.json ("pass"); run on the pilot's host by scratch.s268_gates (the b412 gates line).
The pilot flags = gate_s257.dch_flags(1, the bun_s1 copy) + --hold_inc 1.6 --hold_log (bun_s1's recipe; its own cue is 'step'); det + fp32 + micro 1 for the bitwise parts.
  offbit_aw   flags-off bitwise: the pre-§261 code (scratch/s268_ref.tar = git archive cd814732^ boxes/*.py (top level) + scratch/s222_det_wrap.py, run from its own tree
              scratch/_ref268 with prosqa/ and boxes/runs_fsq/ linked) vs the new tree, boxes/inject_aw with the pilot flags at bun's cue (step), 60 steps: ckpt
              (model / opt / rngs / stream) and logs equal.
  offbit_gen  the same at --cue generic (the §252 / §260 path).
  smoke_plain boxes.inject_aw, the pilot flags at bun's cue (step), log every step, stop 20: rc 9, losses finite.
  smoke_tgt   the same at --cue target (the §261 path): rc 9, losses finite, args cue == target.
  gradflow    a gate-only instrumented copy (boxes/_inject_aw_gradprobe_s268.py = boxes/inject_aw.py + gate_s251.PROBE after 'loss = la + lam_now[0] * aux'), --cue target,
              det, stop 8: on every probed call with d >= 2 the answer loss alone gives nonzero grads on the memory's qkv / gate / ln_w / wo (write and read reached from
              the target-cue answer), and the aux grad is finite.
    python -B scratch/gate_s268.py offbit_aw offbit_gen smoke_plain smoke_tgt gradflow      (host)"""
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

OUT = ROOT + "/scratch/gate_s268"; TMP = ROOT + "/scratch/_tmp/s268g"; DEV = os.environ.get("G268_DEV", "cuda:0"); KEYS = G1.KEYS; COM = ["--fp32", "--micro", "1"]
ST = int(os.environ.get("G268_STOP", "0"))                                   # local CPU smoke only: every part stops at ST steps
OUT = OUT if not ST else ROOT + "/scratch/_tmp/s268g_local"                   # the local smoke never writes the shipped scratch/gate_s268/<part>.json
REFT = ROOT + "/scratch/_ref268"; INIT = ("boxes/runs_fsq/bun_s1_init/model.pt", "1ad3517b7635fddf")
recs = lambda p: [json.loads(l_) for l_ in open(p)] if os.path.exists(p) else []  # noqa: E731
strip = lambda p: [{k: v for k, v in r.items() if k != "sec"} for r in recs(p)]  # noqa: E731


def flags(cue=None):
    fl = G7.dch_flags(1, *INIT) + ["--hold_inc", "1.6", "--hold_log"]
    if cue:
        fl[fl.index("--cue") + 1] = cue
    return fl


def ref_tree():
    if not os.path.isdir(REFT + "/boxes"):
        os.makedirs(REFT, exist_ok=True); tarfile.open(ROOT + "/scratch/s268_ref.tar").extractall(REFT)
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


def _offbit(cue, tag):
    fl = flags(cue) + COM + ["--ckpt_every", str(ST or 60), "--log_every", "10", "--stop_at", str(ST or 60)]; da, db = TMP + "/%s_ref" % tag, TMP + "/%s_new" % tag
    o = {"cue": cue or "step (bun)", "rc": [det_in(ref_tree(), "boxes/inject_aw.py", fl, da), det_in(ROOT, "boxes/inject_aw.py", fl, db)]}
    if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
        o.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=KEYS)); o["log_equal"] = strip(da + "/log.jsonl") == strip(db + "/log.jsonl")
    o["pass"] = o["rc"] == [9, 9] and o.get("n_mismatch") == 0 and bool(o.get("log_equal")); return o


def offbit_aw():
    return _offbit(None, "offaw")


def offbit_gen():
    return _offbit("generic", "offgen")


def _smoke(cue, tag, want_cue):
    d = TMP + "/" + tag; G8.fresh(d)
    rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + flags(cue) + ["--ckpt_every", "0", "--log_every", "1", "--stop_at", str(ST or 20), "--device", DEV, "--out", d],
               d + "/stdout.log")
    ls = [r["loss"] for r in recs(d + "/log.jsonl") if isinstance(r.get("step"), int) and r.get("loss") is not None]
    a = json.load(open(d + "/args.json")) if os.path.exists(d + "/args.json") else {}
    o = {"rc": rc, "n_loss": len(ls), "finite": bool(ls) and all(math.isfinite(x) for x in ls), "cue": a.get("cue")}
    o["pass"] = rc == 9 and o["finite"] and len(ls) >= (ST or 20) and o["cue"] == want_cue; return o


def smoke_plain():
    return _smoke(None, "smoke_plain", "step")


def smoke_tgt():
    return _smoke("target", "smoke_tgt", "target")


def gradflow():
    src = open(ROOT + "/boxes/inject_aw.py", encoding="utf-8").read(); anchor = "            loss = la + lam_now[0] * aux\n"; assert src.count(anchor) == 1
    open(ROOT + "/boxes/_inject_aw_gradprobe_s268.py", "w", encoding="utf-8", newline="\n").write(src.replace(anchor, anchor + G1.PROBE.lstrip("\n")))
    d = TMP + "/gradflow"; G8.fresh(d); lp = d + "/gradprobe.jsonl"; G8.ENV["GRADPROBE_OUT"] = lp   # G8.sh runs with env=G8.ENV
    fl = flags("target") + COM + ["--ckpt_every", "0", "--log_every", "1", "--stop_at", str(ST or 8)]
    rc = G8.sh([sys.executable, "-B", "-m", "scratch.s222_det_wrap", "boxes/_inject_aw_gradprobe_s268.py"] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")
    G8.ENV.pop("GRADPROBE_OUT", None); rs = recs(lp); el = [r for r in rs if r["d"] >= 2]
    req = ("qkv.weight", "gate.weight", "ln_w.weight", "wo.weight")
    o = {"rc": rc, "n_calls": len(rs), "ds": sorted({r["d"] for r in rs}), "n_eligible": len(el), "eligible_la_mem_first": el[0]["la_grad_mem"] if el else None,
         "all_nonzero": bool(el) and all(all((r["la_grad_mem"].get(k) or 0) > 0 for k in req) for r in el),
         "aux_finite": bool(rs) and all(math.isfinite(r["aux_grad_norm"]) for r in rs)}
    o["pass"] = rc == 9 and o["all_nonzero"] and o["aux_finite"]; return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    fns = {"offbit_aw": offbit_aw, "offbit_gen": offbit_gen, "smoke_plain": smoke_plain, "smoke_tgt": smoke_tgt, "gradflow": gradflow}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
