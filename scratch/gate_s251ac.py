"""Note 251-10 ANSWER-AT-CUE build gates (REVIEW_AC #7) -> scratch/gate_s251ac/<part>.json ("pass"). The 4060 (cuda:0), det wrap, fp32, micro 8.
Reference: boxes/_inject_aw_s251.py (HEAD before EAC). Flags = the §244 recipe (ao_s1's queue line, gen_q_b366, run flags only).
  offbit    --cue_ans default: _inject_aw_s251.py vs the new module, det, 100 steps: ckpt (model / opt / rngs / stream) equal, logs equal.
  acts      --cue_ans 1 vs default from the same start, det, log every step, stop 1: the step-1 forward answer_ce equal (the S path unchanged: pass_aw()[0] is m.write's
            S), the step-1 ckpt differs (the term acts).
  smoke     --cue_ans 1, the plain (non-det) path, log every step, stop 40: rc 9, every loss finite, the stop bin carries acue_by_d for d 1 and d 2 (n > 0, finite ce).
  gradflow  a gate-only instrumented copy (boxes/_inject_aw_gradprobe_s251ac.py: after the ACUE add, autograd.grad of acue over every param, one record per loss_items
            call), --cue_ans 1, stop 4: acue grad nonzero into the trunk (g.*) and into the write params (mem.qkv / gate / ln_w) on every call; the norms per param group.
    python -B scratch/gate_s251ac.py offbit acts smoke gradflow"""
import json
import math
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402

OUT = ROOT + "/scratch/gate_s251ac"; TMP = ROOT + "/scratch/_tmp/s251ac"; DEV = "cuda:0"
AO = ("--arm inject --think_k 8 --aux_step 1 --aux_fade 0,0 --aux_reweight --fast_add --cache_lower --d_max 2 --steps 12000 --cue step --write_src answer "
      "--init_trunk boxes/runs_fsq/s233_askname_s1/model.pt --init_sha c00f7ec0bd7977fe --write_mix_start 0 --write_mix_end 0").split()
KEYS = ("model", "opt", "rng_d", "np_rng", "torch_rng", "cuda_rng", "stream_rng", "stream_stats", "gn_log")
COM = ["--fp32", "--micro", "8"]


def det(mod, fl, d):
    G8.fresh(d); return G8.sh([sys.executable, "-B", "-m", "scratch.s222_det_wrap", mod] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")


strip = lambda p: [{k: v for k, v in json.loads(l_).items() if k != "sec"} for l_ in open(p)]  # noqa: E731
recs = lambda p: [json.loads(l_) for l_ in open(p)] if os.path.exists(p) else []  # noqa: E731


def offbit():
    fl = AO + COM + ["--ckpt_every", "100", "--log_every", "10", "--stop_at", "100"]
    da, db = TMP + "/off_s251", TMP + "/off_new"; o = {"rc": [det("boxes/_inject_aw_s251.py", fl, da), det("boxes/inject_aw.py", fl, db)]}
    if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
        o.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=KEYS)); o["log_equal"] = strip(da + "/log.jsonl") == strip(db + "/log.jsonl")
    o["pass"] = o["rc"] == [9, 9] and o.get("n_mismatch") == 0 and bool(o.get("log_equal")); return o


def acts():
    fl = AO + COM + ["--ckpt_every", "1", "--log_every", "1", "--stop_at", "1"]
    da, db = TMP + "/acts_off", TMP + "/acts_on"; o = {"rc": [det("boxes/inject_aw.py", fl, da), det("boxes/inject_aw.py", fl + ["--cue_ans", "1"], db)]}
    ra = [r for r in recs(da + "/log.jsonl") if r.get("step") == 1 and "answer_ce" in r]; rb = [r for r in recs(db + "/log.jsonl") if r.get("step") == 1 and "answer_ce" in r]
    o["answer_ce_step1"] = [ra[0]["answer_ce"] if ra else None, rb[0]["answer_ce"] if rb else None]; o["loss_step1"] = [ra[0]["loss"] if ra else None, rb[0]["loss"] if rb else None]
    if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
        o.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=("model", "opt")))
    o["pass"] = (o["rc"] == [9, 9] and o["answer_ce_step1"][0] is not None and o["answer_ce_step1"][0] == o["answer_ce_step1"][1]
                 and o["loss_step1"][1] > o["loss_step1"][0] and (o.get("n_mismatch") or 0) > 0); return o


def smoke():
    d = TMP + "/smoke"; fl = AO + COM + ["--cue_ans", "1", "--ckpt_every", "0", "--log_every", "1", "--stop_at", "40"]
    G8.fresh(d); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")
    rs = recs(d + "/log.jsonl"); ls = [r["loss"] for r in rs if isinstance(r.get("step"), int) and "loss" in r and r.get("loss") is not None]
    bins = [r for r in rs if "acue_by_d" in r]; ab = bins[-1]["acue_by_d"] if bins else {}
    o = {"rc": rc, "n_loss": len(ls), "finite": bool(ls) and all(math.isfinite(x) for x in ls), "acue_by_d": ab}
    o["pass"] = (rc == 9 and o["finite"] and len(ls) >= 40 and all(k in ab and ab[k]["n"] > 0 and math.isfinite(ab[k]["ce"]) for k in ("1", "2"))); return o


PROBE = r'''
                if torch.is_grad_enabled() and acue.requires_grad and os.environ.get("GRADPROBE_OUT"):   # GATE-ONLY: the ACUE term's grad per param
                    _pl = [(n_, p_) for n_, p_ in params.named_parameters() if p_.requires_grad]
                    _ga = torch.autograd.grad(acue, [p_ for _, p_ in _pl], retain_graph=True, allow_unused=True)
                    open(os.environ["GRADPROBE_OUT"], "a").write(json.dumps({"d": items[0]["d"], "n_c": len(per_c), "acue": float(acue.detach()),
                        "norms": {n_: (float(x_.float().norm()) if x_ is not None else None) for (n_, _), x_ in zip(_pl, _ga)}}) + "\n")
'''


def gradflow():
    src = open(ROOT + "/boxes/inject_aw.py", encoding="utf-8").read()
    anchor = "                acue = torch.stack([per_c[i] for i in sorted(per_c)]).mean() if per_c else la * 0.0; loss = loss + a.cue_ans * acue\n"; assert src.count(anchor) == 1
    open(ROOT + "/boxes/_inject_aw_gradprobe_s251ac.py", "w", encoding="utf-8", newline="\n").write(src.replace(anchor, anchor + PROBE.lstrip("\n")))
    d = TMP + "/gradflow"; G8.fresh(d); lp = d + "/gradprobe.jsonl"; G8.ENV["GRADPROBE_OUT"] = lp   # G8.sh runs with env=G8.ENV
    fl = AO + COM + ["--cue_ans", "1", "--ckpt_every", "0", "--log_every", "1", "--stop_at", "4"]
    rc = G8.sh([sys.executable, "-B", "-m", "scratch.s222_det_wrap", "boxes/_inject_aw_gradprobe_s251ac.py"] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")
    G8.ENV.pop("GRADPROBE_OUT", None); rs = recs(lp)

    def grp(r):
        nz = {k: v for k, v in r["norms"].items() if v}
        return {"trunk_g": math.sqrt(sum(v ** 2 for k, v in nz.items() if k.startswith("g."))), "n_trunk_nonzero": sum(k.startswith("g.") for k in nz),
                "mem": {k: v for k, v in r["norms"].items() if k.startswith("mem.")}}
    gs = [dict(grp(r), d=r["d"], n_c=r["n_c"], acue=r["acue"]) for r in rs]
    req = ("mem.qkv.weight", "mem.gate.weight", "mem.ln_w.weight")
    o = {"rc": rc, "n_calls": len(rs), "calls": gs,
         "trunk_nonzero_all": bool(gs) and all(x["trunk_g"] > 0 for x in gs),
         "write_nonzero_all": bool(gs) and all(all((x["mem"].get(k) or 0) > 0 for k in req) for x in gs)}
    o["pass"] = rc == 9 and o["trunk_nonzero_all"] and o["write_nonzero_all"] and all(x["n_c"] > 0 for x in gs); return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    fns = {"offbit": offbit, "acts": acts, "smoke": smoke, "gradflow": gradflow}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
