"""§265 DENSE-DISTRACT host gates (Note 265-1) -> scratch/gate_s271/<part>.json ("pass"); run on a host by scratch.s271_gates.
BUN = gate_s257.dch_flags(1, the bun_s1 copy) + --hold_inc 1.6 --hold_log (bun's recipe: cue step, --distract_p 1 --distract_mix s256,chain,givop); det + fp32 + micro 8
(--micro = the chunk count) for the bitwise / gradflow parts (s222_det_wrap).
  offbit_aw   flags-off bitwise (--distract_nmax default 2): the pre-§265 code (scratch/s271_ref.tar = git archive <pre-§265> boxes/*.py (top level) + scratch/s222_det_wrap.py,
              its own tree scratch/_ref271 with prosqa/ and boxes/runs_fsq/ linked) vs the new tree, BUN, 60 steps: ckpt (model / opt / rngs / stream) and logs equal.
  smoke_dd    boxes.inject_aw BUN + --distract_nmax 8, stop 20, log every step: rc 9, losses finite, the stop-step bin's distract_n has some n > 2 (the nmax draw is live).
  gradflow    a gate-only instrumented copy (boxes/_inject_aw_gradprobe_s271.py = boxes/inject_aw.py + gate_s251.PROBE after 'loss = la + lam_now[0] * aux'), BUN +
              --distract_nmax 8, det, stop 8: every probed call with d >= 2 gives nonzero answer-loss grads on the memory's qkv / gate / ln_w / wo; the aux grad is finite.
    python -B scratch/gate_s271.py offbit_aw smoke_dd gradflow      (host)"""
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

OUT = ROOT + "/scratch/gate_s271"; TMP = ROOT + "/scratch/_tmp/s271g"; DEV = os.environ.get("G271_DEV", "cuda:0"); KEYS = G1.KEYS; COM = ["--fp32", "--micro", "8"]
ST = int(os.environ.get("G271_STOP", "0"))                                  # local CPU smoke only: every training part stops at ST steps
OUT = OUT if not ST else ROOT + "/scratch/_tmp/s271g_local"                 # the local smoke never writes the shipped scratch/gate_s271/<part>.json
REFT = ROOT + "/scratch/_ref271"; INIT = ("boxes/runs_fsq/bun_s1_init/model.pt", "1ad3517b7635fddf")
recs = lambda p: [json.loads(l_) for l_ in open(p)] if os.path.exists(p) else []  # noqa: E731
strip = lambda p: [{k: v for k, v in r.items() if k != "sec"} for r in recs(p)]  # noqa: E731


def bun():
    return G7.dch_flags(1, *INIT) + ["--hold_inc", "1.6", "--hold_log"]


def ref_tree():
    if not os.path.isdir(REFT + "/boxes"):
        os.makedirs(REFT, exist_ok=True); tarfile.open(ROOT + "/scratch/s271_ref.tar").extractall(REFT)
    lnk = lambda q: os.path.islink(q) or getattr(os.path, "isjunction", lambda _: False)(q)  # noqa: E731   never rmtree through a link; isjunction is 3.12+ (hosts: 3.11)
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


def offbit_aw():
    fl = bun() + COM + ["--ckpt_every", str(ST or 60), "--log_every", "10", "--stop_at", str(ST or 60)]; da, db = TMP + "/offaw_ref", TMP + "/offaw_new"
    o = {"rc": [det_in(ref_tree(), "boxes/inject_aw.py", fl, da), det_in(ROOT, "boxes/inject_aw.py", fl, db)]}
    if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
        o.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=KEYS)); o["log_equal"] = strip(da + "/log.jsonl") == strip(db + "/log.jsonl")
    o["pass"] = o["rc"] == [9, 9] and o.get("n_mismatch") == 0 and bool(o.get("log_equal")); return o


def smoke_dd():
    d = TMP + "/smoke_dd"; G8.fresh(d); stop = ST or 20
    rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + bun() + ["--distract_nmax", "8", "--ckpt_every", "0", "--log_every", "1", "--stop_at", str(stop), "--device", DEV,
                                                                          "--out", d], d + "/stdout.log")
    rs = recs(d + "/log.jsonl"); ls = [r["loss"] for r in rs if isinstance(r.get("step"), int) and r.get("loss") is not None]; br = [r for r in rs if "bin" in r]
    dn = br[-1].get("distract_n", {}) if br else {}; a = json.load(open(d + "/args.json")) if os.path.exists(d + "/args.json") else {}
    o = {"rc": rc, "n_loss": len(ls), "finite": bool(ls) and all(math.isfinite(x) for x in ls), "distract_n": dn, "distract_capped": br[-1].get("distract_capped") if br else None,
         "nmax": a.get("distract_nmax")}
    o["pass"] = rc == 9 and o["finite"] and len(ls) >= stop and o["nmax"] == 8 and any(int(k) > 2 for k in dn); return o


def gradflow():
    src = open(ROOT + "/boxes/inject_aw.py", encoding="utf-8").read(); anchor = "            loss = la + lam_now[0] * aux\n"; assert src.count(anchor) == 1
    open(ROOT + "/boxes/_inject_aw_gradprobe_s271.py", "w", encoding="utf-8", newline="\n").write(src.replace(anchor, anchor + G1.PROBE.lstrip("\n")))
    d = TMP + "/gradflow"; G8.fresh(d); lp = d + "/gradprobe.jsonl"; G8.ENV["GRADPROBE_OUT"] = lp
    fl = bun() + ["--distract_nmax", "8"] + COM + ["--ckpt_every", "0", "--log_every", "1", "--stop_at", str(ST or 8)]
    rc = G8.sh([sys.executable, "-B", "-m", "scratch.s222_det_wrap", "boxes/_inject_aw_gradprobe_s271.py"] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")
    G8.ENV.pop("GRADPROBE_OUT", None); rs = recs(lp); el = [r for r in rs if r["d"] >= 2]; req = ("qkv.weight", "gate.weight", "ln_w.weight", "wo.weight")
    o = {"rc": rc, "n_calls": len(rs), "ds": sorted({r["d"] for r in rs}), "n_eligible": len(el),
         "all_nonzero": bool(el) and all(all((r["la_grad_mem"].get(k) or 0) > 0 for k in req) for r in el), "aux_finite": bool(rs) and all(math.isfinite(r["aux_grad_norm"]) for r in rs)}
    o["pass"] = rc == 9 and o["all_nonzero"] and o["aux_finite"]; return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    fns = {"offbit_aw": offbit_aw, "smoke_dd": smoke_dd, "gradflow": gradflow}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
