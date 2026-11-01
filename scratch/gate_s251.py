"""§251 ANSWER-EXTEND build gates (AE_v1 §3) -> scratch/gate_s251/<part>.json ("pass"). The 4060 (cuda:0), det wrap, fp32 / CPU.
References: boxes/_inject_aw_s237.py (rcb_plaw4's module, sha c4c6fe7e), boxes/_inject_aw_s250.py (HEAD before E251). Flags = rcb_plaw4_s1's queue line (gen_q_b354).
  offbit    --aux_d_max default: _inject_aw_s237.py vs the new module, det, fp32, micro 8, the real schedule, 100 steps: ckpt (model / opt / rngs / stream) equal, logs equal.
  onbit     --aux_d_max 2 == default until the first d >= 3 step S3 (replayed: default_rng(seed + SEED_D).integers(1, 5) per step; gate-only --seed 4, since seeds 1 / 2
            draw d >= 3 at step 1): ckpt equal at S3 - 1 (stop_at S3 - 1);
            and the two differ at S3 (the flag acts).
  smoke     --aux_d_max 2 through a d 3 and a d 4 step (stop at the first step after both are seen, log every step): rc 9, every loss finite, aux_ce == 0 at every d >= 3
            step and > 0 at every d <= 2 step (the replayed d per step).
  gradflow  a gate-only instrumented copy (boxes/_inject_aw_gradprobe_s251.py: after loss = la + lam * aux, autograd.grad of aux over every param and of la over every
            DeltaMem param, one record per loss_items call) at --aux_d_max 2, micro 8, the first steps through d 4: aux grad norm == 0 on d >= 3 calls, > 0 on d <= 2 calls;
            at d 4 the answer-CE grad nonzero on the write (mem.qkv / gate / ln_w) and read (mem.wo / ln) params (every mem param norm reported).
    python -B scratch/gate_s251.py offbit onbit smoke gradflow"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.environ.get("LATENT_ROOT", "D:/Latent");   # LATENT_ROOT: the Vast host (Note 255-6 cell)
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402

OUT = ROOT + "/scratch/gate_s251"; TMP = ROOT + "/scratch/_tmp/s251"; DEV = "cuda:0"
RCB = ("--arm inject --think_k 8 --aux_step 1 --aux_fade 48000,48000 --aux_reweight --fast_add --cache_lower --d_max 4 --steps 16000 --cue step --write_src answer "
       "--init_trunk boxes/runs_fsq/s233_askname_s1/model.pt --init_sha c00f7ec0bd7977fe --write_mix_start 0 --write_mix_end 0 --mul_s_p 0 "
       "--init_model boxes/runs_fsq/aw_plaw2_s1/model.pt --init_model_sha 484dab1eab9e2c12").split()
KEYS = ("model", "opt", "rng_d", "np_rng", "torch_rng", "cuda_rng", "stream_rng", "stream_stats", "gn_log")
COM = ["--fp32", "--micro", "8"]


def dseq(seed=1, n=200):
    import boxes.inject_ow as IO
    r = np.random.default_rng(seed + IO.SEED_D); return [int(r.integers(1, 5)) for _ in range(n)]


def det(mod, fl, d):
    G8.fresh(d); return G8.sh([sys.executable, "-B", "-m", "scratch.s222_det_wrap", mod] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")


strip = lambda p: [{k: v for k, v in json.loads(l_).items() if k != "sec"} for l_ in open(p)]  # noqa: E731


def offbit():
    fl = RCB + COM + ["--ckpt_every", "100", "--log_every", "10", "--stop_at", "100"]
    da, db = TMP + "/off_s237", TMP + "/off_new"; o = {"rc": [det("boxes/_inject_aw_s237.py", fl, da), det("boxes/inject_aw.py", fl, db)]}
    if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
        o.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=KEYS)); o["log_equal"] = strip(da + "/log.jsonl") == strip(db + "/log.jsonl")
    o["pass"] = o["rc"] == [9, 9] and o.get("n_mismatch") == 0 and bool(o.get("log_equal")); return o


def onbit():
    ds = dseq(4); s3 = 1 + next(i for i, d in enumerate(ds) if d >= 3); o = {"seed": 4, "first_d_ge3_step": s3, "d_seq_head": ds[:12],   # seeds 1 / 2 open with a d >= 3 step
         "note": "gate-only seed 4 (d 2, 1, 1, 3, ...): the registered seeds 1 / 2 both draw d >= 3 at step 1, so the pre-S3 prefix is empty for them"}
    if s3 < 2:
        o["note"] = "S3 = 1: nothing precedes it"; o["pass"] = False; return o
    for tag, stop in (("pre", s3 - 1), ("at", s3)):
        fl = RCB + COM + ["--seed", "4", "--ckpt_every", str(stop), "--log_every", "1", "--stop_at", str(stop)]
        da, db = TMP + "/on_ctrl_" + tag, TMP + "/on_aex_" + tag; rc = [det("boxes/inject_aw.py", fl, da), det("boxes/inject_aw.py", fl + ["--aux_d_max", "2"], db)]
        r = {"rc": rc}
        if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
            r.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=("model", "opt")))
        o[tag] = r
    o["pass"] = (o["pre"]["rc"] == [9, 9] and o["pre"].get("n_mismatch") == 0 and o["at"]["rc"] == [9, 9] and (o["at"].get("n_mismatch") or 0) > 0); return o


def smoke():
    ds = dseq(); s3 = 1 + next(i for i, d in enumerate(ds) if d == 3); s4 = 1 + next(i for i, d in enumerate(ds) if d == 4); stop = max(s3, s4) + 1
    d = TMP + "/smoke"; fl = RCB + COM + ["--aux_d_max", "2", "--ckpt_every", "0", "--log_every", "1", "--stop_at", str(stop)]
    G8.fresh(d); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")
    recs = [r for r in (json.loads(l_) for l_ in open(d + "/log.jsonl")) if "aux_ce" in r and isinstance(r.get("step"), int) and r["step"] >= 1] if os.path.exists(d + "/log.jsonl") else []
    by = [(r["step"], ds[r["step"] - 1], r["aux_ce"], r.get("loss")) for r in recs]
    o = {"rc": rc, "stop": stop, "first_d3": s3, "first_d4": s4, "records": by,
         "finite": all(x[3] is not None and math.isfinite(x[3]) for x in by), "aux_zero_d_ge3": all(x[2] == 0 for x in by if x[1] >= 3),
         "aux_pos_d_le2": all(x[2] > 0 for x in by if x[1] <= 2), "n_d3": sum(x[1] == 3 for x in by), "n_d4": sum(x[1] == 4 for x in by)}
    o["pass"] = rc == 9 and o["finite"] and o["aux_zero_d_ge3"] and o["aux_pos_d_le2"] and o["n_d3"] > 0 and o["n_d4"] > 0; return o


PROBE = r'''
            if torch.is_grad_enabled() and loss.requires_grad and os.environ.get("GRADPROBE_OUT"):   # §251 GATE-ONLY: per-term grad norms (gate_s251 gradflow)
                _pl = [(n_, p_) for n_, p_ in params.named_parameters() if p_.requires_grad]
                _ga = torch.autograd.grad(aux, [p_ for _, p_ in _pl], retain_graph=True, allow_unused=True) if aux.requires_grad else [None] * len(_pl)
                _mp = [(n_, p_) for n_, p_ in m.mem.named_parameters() if p_.requires_grad]
                _gl = torch.autograd.grad(la, [p_ for _, p_ in _mp], retain_graph=True, allow_unused=True)
                _na = math.sqrt(sum(float((x_.float() ** 2).sum()) for x_ in _ga if x_ is not None))
                open(os.environ["GRADPROBE_OUT"], "a").write(json.dumps({"d": items[0]["d"], "n": len(items), "aux_grad_norm": _na, "aux": float(aux.detach()),
                    "la_grad_mem": {n_: (float(x_.float().norm()) if x_ is not None else None) for (n_, _), x_ in zip(_mp, _gl)}}) + "\n")
'''


def gradflow():
    src = open(ROOT + "/boxes/inject_aw.py", encoding="utf-8").read(); anchor = "            loss = la + lam_now[0] * aux\n"; assert src.count(anchor) == 1
    open(ROOT + "/boxes/_inject_aw_gradprobe_s251.py", "w", encoding="utf-8", newline="\n").write(src.replace(anchor, anchor + PROBE.lstrip("\n")))
    ds = dseq(); stop = max(1 + ds.index(3), 1 + ds.index(4)) + 1; d = TMP + "/gradflow"; G8.fresh(d); lp = d + "/gradprobe.jsonl"; G8.ENV["GRADPROBE_OUT"] = lp   # G8.sh runs with env=G8.ENV (os.environ changes would not reach the run)
    fl = RCB + COM + ["--aux_d_max", "2", "--ckpt_every", "0", "--log_every", "1", "--stop_at", str(stop)]
    rc = G8.sh([sys.executable, "-B", "-m", "scratch.s222_det_wrap", "boxes/_inject_aw_gradprobe_s251.py"] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")
    G8.ENV.pop("GRADPROBE_OUT", None)
    recs = [json.loads(l_) for l_ in open(lp)] if os.path.exists(lp) else []
    hi = [r for r in recs if r["d"] >= 3]; lo = [r for r in recs if r["d"] <= 2]; d4 = [r for r in recs if r["d"] == 4]
    req = ("qkv.weight", "gate.weight", "ln_w.weight", "wo.weight")
    o = {"rc": rc, "stop": stop, "n_calls": len(recs), "n_hi": len(hi), "n_lo": len(lo), "n_d4": len(d4),
         "aux_zero_hi": all(r["aux_grad_norm"] == 0.0 for r in hi), "aux_pos_lo": all(r["aux_grad_norm"] > 0 for r in lo),
         "d4_la_mem_norms": d4[0]["la_grad_mem"] if d4 else None,
         "d4_write_read_nonzero": bool(d4) and all(all((r["la_grad_mem"].get(k) or 0) > 0 for k in req) for r in d4)}
    o["pass"] = rc == 9 and o["n_hi"] > 0 and o["n_lo"] > 0 and o["aux_zero_hi"] and o["aux_pos_lo"] and o["d4_write_read_nonzero"]; return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    fns = {"offbit": offbit, "onbit": onbit, "smoke": smoke, "gradflow": gradflow}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
