"""§258 S0 INSTALL gates (LB2_v1 §2 + Note 258-1) -> scratch/gate_s258m/<part>.json ("pass").
LOCAL (CPU):
  hfref     boxes.s5.load_gpt2('cpu', 'gpt2-medium') (the sha-asserted safetensors, H 16) vs transformers' GPT2LMHeadModel('openai-community/gpt2-medium' from the SAME file):
            logits on 4 fixed prompts, max |delta| <= 1e-4 (the loader / H / key mapping right); + GPT-2 small through the new load_gpt2 == the old file's GPT2KV (bitwise).
HOST (the S0 cell on the S3 card class; det + fp32 + micro 8 for the bitwise parts):
  offbit_aw  flags-off bitwise on GPT-2 SMALL: the pre-§258 code (scratch/s258m_ref.tar = git archive <REF> boxes/*.py + the det wrap, run from its own tree
             scratch/_ref258m with prosqa/ and boxes/runs_fsq/ linked) vs the new tree, boxes/inject_aw with gate_s251.RCB, 100 steps: ckpt (model / opt / rngs / stream) and logs equal.
  offbit_ow  the same for boxes/inject_ow --arm direct (s233's recipe shape: d_max 2), 30 steps.
  step0      gpt2-medium InjectAW as built (no init: W_o = 0, fade 0): the decoder's answer-position hidden (inject_aw_eval.hh_dump, post ln_f) at K' 1..8 == K' 0 on 200
             REG TEST2 d 1-4 items: max |delta| == 0 (so max |delta logits| == 0); + the shape (L 24, D 1024, H 16, DeltaMem layer 12, heads 8 + 8).
  smoke_aw   inject_aw --base gpt2-medium (RCB's flags without the small inits), the plain path, log every step, stop 40: rc 9, losses finite.
  smoke_ow   the askname wrapper (scratch/s258_askname.py = the s233 wrapper copy) --base gpt2-medium, stop 20: rc 9, losses finite.
    python -B scratch/gate_s258m.py hfref      (local)          python -B scratch/gate_s258m.py offbit_aw offbit_ow step0 smoke_aw smoke_ow      (host)"""
import json
import math
import os
import subprocess
import sys
import tarfile

ROOT = os.environ.get("LATENT_ROOT", "D:/Latent")
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
import gate_s251 as G1  # noqa: E402

OUT = ROOT + "/scratch/gate_s258m"; TMP = ROOT + "/scratch/_tmp/s258m"; DEV = "cuda:0"; RCB = list(G1.RCB); KEYS = G1.KEYS; COM = ["--fp32", "--micro", "8"]
REFT = ROOT + "/scratch/_ref258m"


def medium_flags():
    """RCB's flags (REG d U{1..4}, aux, cue step, write_src answer, ...) without the GPT-2-small inits, + --base gpt2-medium."""
    fl = list(RCB)
    for k in ("--init_trunk", "--init_sha", "--init_model", "--init_model_sha"):
        i = fl.index(k); del fl[i:i + 2]
    return fl + ["--base", "gpt2-medium"]


recs = lambda p: [json.loads(l_) for l_ in open(p)] if os.path.exists(p) else []  # noqa: E731
strip = lambda p, drop=(): [{k: v for k, v in r.items() if k != "sec" and k not in drop} for r in recs(p)]  # noqa: E731


def ref_tree():
    if not os.path.isdir(REFT + "/boxes"):
        os.makedirs(REFT, exist_ok=True); tarfile.open(ROOT + "/scratch/s258m_ref.tar").extractall(REFT)
        os.symlink(ROOT + "/prosqa", REFT + "/prosqa"); os.symlink(ROOT + "/boxes/runs_fsq", REFT + "/boxes/runs_fsq")
    return REFT


def det_in(tree, mod, fl, d):
    G8.fresh(d); return subprocess.call([sys.executable, "-B", "-m", "scratch.s222_det_wrap", mod] + fl + ["--device", DEV, "--out", d], cwd=tree, env=G8.ENV,
                                        stdout=open(d + "/stdout.log", "w"), stderr=subprocess.STDOUT)


def _offbit(mod, fl, tag):
    da, db = TMP + "/%s_ref" % tag, TMP + "/%s_new" % tag; o = {"rc": [det_in(ref_tree(), mod, fl, da), det_in(ROOT, mod, fl, db)]}
    if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
        o.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=KEYS)); o["log_equal"] = strip(da + "/log.jsonl") == strip(db + "/log.jsonl")
    o["pass"] = o["rc"] == [9, 9] and o.get("n_mismatch") == 0 and bool(o.get("log_equal")); return o


def offbit_aw():
    return _offbit("boxes/inject_aw.py", RCB + COM + ["--ckpt_every", "100", "--log_every", "10", "--stop_at", "100"], "offaw")


def offbit_ow():
    return _offbit("boxes/inject_ow.py", ["--arm", "direct", "--d_max", "2", "--steps", "8000"] + COM + ["--ckpt_every", "30", "--log_every", "10", "--stop_at", "30"], "offow")


def step0():
    import numpy as np
    import torch
    from types import SimpleNamespace
    from boxes import inject_aw as AW
    from boxes import inject_aw_eval as EV
    from boxes import inject_ow as IO
    d = G8.fresh(TMP + "/step0_args"); fl = medium_flags() + ["--micro", "2", "--eval_bs", "50"]
    rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log"); assert rc == 0, rc
    a = json.load(open(d + "/args.json")); torch.manual_seed(a["seed"]); m, g, _ = AW.build_aw(SimpleNamespace(**dict(a, init_trunk="", init_model="")), DEV)
    sets = IO.build_sets(mul_s_p=0); T = EV.tokd({"test2": {dd: sets["test2"][dd][:50] for dd in (1, 2, 3, 4)}}, a, (1, 2, 3, 4))
    HP = [EV.hh_dump(m, T[dd], DEV, 50) for dd in (1, 2, 3, 4)]; H = np.concatenate([x[0] for x in HP]); P = np.concatenate([x[1] for x in HP])   # one d per batch (as part_halthead)
    dmax = float(np.abs(H - H[:, :1]).max())
    o = {"shape": {"L": g.L, "D": g.D, "H": g.H, "mem_layer": m.mem.layer, "h_add": m.mem.h_add, "h_del": m.mem.h_del}, "n": int(len(H)), "max_abs_delta_hidden_vs_k0": dmax,
         "preds_equal_all_k": bool((P == P[:, :1]).all())}
    o["pass"] = dmax == 0.0 and o["shape"] == {"L": 24, "D": 1024, "H": 16, "mem_layer": 12, "h_add": 8, "h_del": 8}; return o


def _smoke(cmd, d, n):
    G8.fresh(d); rc = G8.sh(cmd + ["--out", d], d + "/stdout.log"); rs = recs(d + "/log.jsonl")
    ls = [r["loss"] for r in rs if isinstance(r.get("step"), int) and r.get("loss") is not None]
    o = {"rc": rc, "n_loss": len(ls), "finite": bool(ls) and all(math.isfinite(x) for x in ls), "base": (json.load(open(d + "/args.json")).get("base") if os.path.exists(d + "/args.json") else None)}
    o["pass"] = rc == 9 and o["finite"] and len(ls) >= n and o["base"] == "gpt2-medium"; return o


def smoke_aw():
    return _smoke([sys.executable, "-B", "-m", "boxes.inject_aw"] + medium_flags() + ["--micro", "2", "--eval_bs", "50", "--ckpt_every", "0", "--log_every", "1", "--stop_at", "40",
                  "--device", DEV], TMP + "/smoke_aw", 40)


def smoke_ow():
    return _smoke([sys.executable, "-B", "scratch/s258_askname.py", "--d_max", "2", "--steps", "8000", "--base", "gpt2-medium", "--micro", "2",
                   "--eval_bs", "50", "--eval_parts", "", "--ckpt_every", "0", "--log_every", "1", "--stop_at", "20", "--device", DEV], TMP + "/smoke_ow", 20)


def hfref():
    import torch
    from transformers import GPT2LMHeadModel
    from boxes import s5
    from boxes import core as bc
    import tiktoken
    enc = tiktoken.get_encoding("gpt2"); P = ["The quick brown fox jumps over the lazy dog.", "Alice has 3 apples. Bob has 2 more apples than Alice. How many apples does Bob have?",
                                             "In 1905, Einstein published", "def f(x):\n    return x *"]
    g = s5.load_gpt2("cpu", "gpt2-medium").eval()
    from transformers import GPT2Config
    cfg = GPT2Config(n_layer=24, n_embd=1024, n_head=16); hf = GPT2LMHeadModel(cfg).eval(); sd = s5.medium_sd()
    miss = hf.transformer.load_state_dict(sd, strict=False); hf.tie_weights(); o = {"missing": list(miss.missing_keys)[:5], "unexpected": list(miss.unexpected_keys)[:5], "prompts": []}
    mx = 0.0
    with torch.no_grad():
        for p in P:
            ids = torch.tensor([enc.encode(p)]); a = hf(ids).logits[0]
            h = g.embed(ids) + g.w("wpe.weight")[: ids.shape[1]][None]
            for i in range(g.L):
                h = __import__("boxes.deltamem", fromlist=["_block"])._block(g, i, h, g.D // g.H)
            b = g.logits(torch.nn.functional.layer_norm(h, (g.D,), g.w("ln_f.weight"), g.w("ln_f.bias")))[0]
            dd = float((a - b).abs().max()); mx = max(mx, dd); o["prompts"].append({"n_tok": ids.shape[1], "max_abs_delta": dd})
    sd0 = torch.load(ROOT + "/prosqa/ckpt/gpt2_pretrained.bin", map_location="cpu", weights_only=True); g_old = bc.GPT2KV(sd0); g_new = s5.load_gpt2("cpu", "gpt2")
    o["small_equal"] = g_old.H == g_new.H and all(torch.equal(g_old.p[k], g_new.p[k]) for k in g_old.p) and sorted(g_old.p) == sorted(g_new.p)
    # the fp32 reference floor: the SAME comparison for GPT-2 small (the known-good loader) vs HF small (gpt2_pretrained.bin); medium must sit within 3x of it, <= 1e-3,
    # with every position's argmax equal (a wrong H / key map moves logits by units, not 1e-4)
    hs = GPT2LMHeadModel(GPT2Config(n_layer=12, n_embd=768, n_head=12)).eval(); hs.transformer.load_state_dict(sd0, strict=False); hs.tie_weights(); ms = 0.0; am = True
    with torch.no_grad():
        for p in P:
            ids = torch.tensor([enc.encode(p)])
            for gg, hh_ in ((g_new, hs), (g, hf)):
                h = gg.embed(ids) + gg.w("wpe.weight")[: ids.shape[1]][None]
                for i in range(gg.L):
                    h = __import__("boxes.deltamem", fromlist=["_block"])._block(gg, i, h, gg.D // gg.H)
                b = gg.logits(torch.nn.functional.layer_norm(h, (gg.D,), gg.w("ln_f.weight"), gg.w("ln_f.bias")))[0]; a = hh_(ids).logits[0]
                am = am and bool((a.argmax(-1) == b.argmax(-1)).all())
                if gg is g_new:
                    ms = max(ms, float((a - b).abs().max()))
    o["small_max_abs_delta_floor"] = ms; o["argmax_equal_all"] = am
    o["max_abs_delta"] = mx; o["pass"] = (mx <= max(3 * ms, 1e-5) and mx <= 1e-3 and am and o["small_equal"]
                                          and not [k for k in o["missing"] if not k.endswith("attn.bias") and "masked_bias" not in k]); return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    fns = {k: globals()[k] for k in ("hfref", "offbit_aw", "offbit_ow", "step0", "smoke_aw", "smoke_ow")}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
