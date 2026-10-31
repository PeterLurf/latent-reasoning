"""§253 LOOKUP-REBUILD (LB_v1 §2; Note 252-x; Audit approval guard $5) build / gates / step-0 kit -> scratch/gate_s253/<part>.json ("pass"). The 4060 (cuda:0).
Arms (Stage 1, Vast 3090, 12k, --keep_final_ckpt, micro FIXED = aw_plaw2_s1's realised 1, eval_bs = its realised 250, NO --auto_micro (Audit 4)):
  p2p_s1 / p2p_s2 : aw_plaw2_s1's queue flags (gen_q_b347) with --init_trunk boxes/runs_fsq/s252_askperm_s1/model.pt --init_sha cf35fc6bcea9035e + --perm_p 0.5, seeds 1 / 2
  p2p_old_s1      : the same with the askname init (s233_askname_s1, c00f7ec0bd7977fe) + --perm_p 0.5, seed 1
  eval_parts = aw_plaw2_s1's + test_perm.
Stage 2 (after Theory's Stage-1 read; not built here): lbr_s{k} = rcb_plaw4_s{k}'s args + --init_model p2p_s{k} + init_trunk askperm + --perm_p 0.5 + test_perm, micro 2.
Parts:
  perm     (Audit 1) the P2P training stream replayed on CPU for seeds 1 / 2 (the trainer's own draw law: Stream(OW.SEED_TRAIN, EXCL of law 1), d = rng_d per step, the per-item
           coin / permutation of perm_render.train_perm) for the first 800 steps x 64 items: solve asserted inside, >= 1 inversion, every text not in law-1 TEST-PERM d 1 / d 2.
  evalbit  (Audit 2) the eval module before (_inject_aw_eval_s252.py) / after the K' 2 edit on a rcb_plaw4_s1 copy, parts ws,test_perm: eval_ws.json equal (minus module sha),
           eval_test_perm.json equal except by_d.4.acc_kprime['2'] and items.4.ans_kprime['2'] (new) and module sha; the new field present.
  args     P2P / P2P-OLD --args_only vs aw_plaw2_s1/args.json (C.diff): {init_trunk, init_sha(256), perm_p, keep_final_ckpt, eval_parts, micro / auto_micro fixed} + module keys;
           frozen args -> scratch/s253_args_frozen_<run>.json (micro 1, eval_bs 250).
  step0    the step-0 model under P2P's args (seed 1): the trainer itself run 1 step at --lr 0 (AdamW at lr 0 changes no weight; the init + g_add calibration + the step-1
           fade / gamma buffers are the trainer's own) -> every trunk weight == the askperm model.pt's and every DeltaMem weight == its init (checked); model.pt + the real
           P2P args.json -> boxes/runs_fsq/lb_step0; then inject_aw_eval --parts test_perm (eval_bs 250, cuda:0) -> lb_step0/eval_test_perm.json (Theory reads by_d.2.ws.1).
    python -B scratch/s253_build.py perm evalbit args step0"""
import json
import os
import shutil
import sys

import numpy as np

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
import s228_args_check as C  # noqa: E402

OUT = ROOT + "/scratch/gate_s253"; TMP = ROOT + "/scratch/_tmp/s253"; DEV = "cuda:0"
AW2 = ("--arm inject --think_k 8 --aux_step 1 --aux_fade 48000,48000 --aux_reweight --fast_add --cache_lower --d_max 2 --steps 12000 --cue step --write_src answer "
       "--write_mix_start 0 --write_mix_end 0").split()
EVP = "acc,lp5,mean_s,cue_only,cue_wrong,swap_k,swap_1,probe,ws,test_perm"
INIT = {"askperm": ["--init_trunk", "boxes/runs_fsq/s252_askperm_s1/model.pt", "--init_sha", "cf35fc6bcea9035e"],
        "askname": ["--init_trunk", "boxes/runs_fsq/s233_askname_s1/model.pt", "--init_sha", "c00f7ec0bd7977fe"]}
ARMS = {"p2p_s1": ("askperm", 1), "p2p_s2": ("askperm", 2), "p2p_old_s1": ("askname", 1)}


def flags(run):
    init, seed = ARMS[run]
    return AW2 + INIT[init] + ["--perm_p", "0.5"] + (["--seed", str(seed)] if seed != 1 else []) + ["--micro", "1", "--eval_bs", "250"]


def tail(run):
    return ["--keep_final_ckpt", "--ckpt_every", "1000", "--eval_parts", EVP, "--args_ref", "scratch/s253_args_frozen_%s.json" % run]


def perm():
    import boxes.ow_words as OW
    import boxes.perm_render as PR
    from boxes import inject_ow as IO
    s1 = IO.build_sets(mul_s_p=1); tp, info = PR.test_perm_sets({d: s1["test2"][d] for d in (1, 2)}, (1, 2)); tpt = {it["text"] for v in tp.values() for it in v}; o = {"tp_info": info}
    for seed in (1, 2):
        st = OW.Stream(OW.SEED_TRAIN, s1["excl"], frames=OW.FRAME_TRAIN, mul_s_p=1, **IO.OWKW); rng_d = np.random.default_rng(seed + IO.SEED_D); n = 0; np_ = 0; bad = 0; hit = 0
        for step in range(1, 801):
            d = int(rng_d.integers(1, 3))
            for k in range(64):
                it = st.draw(d); n += 1
                if d >= 2 and np.random.default_rng((seed, PR.SEED_PERM_TRAIN + 1, step, k)).random() < 0.5:
                    it = PR.train_perm(it, seed, step, k); np_ += 1
                    oo = [int(x[1:]) for x in it["order"] if x.startswith("S") and x != "S0"]; bad += not any(oo[i] > oo[j] for i in range(d) for j in range(i + 1, d))
                hit += it["text"] in tpt
        o["seed%d" % seed] = {"items": n, "permuted": np_, "no_inversion": bad, "tp_text_hits": hit}
    o["pass"] = all(o["seed%d" % s]["no_inversion"] == 0 and o["seed%d" % s]["tp_text_hits"] == 0 and o["seed%d" % s]["permuted"] > 0 for s in (1, 2)); return o


def run_copy(d):
    G8.fresh(d); shutil.copy("boxes/runs_fsq/rcb_plaw4_s1/args.json", d + "/args.json"); shutil.copy("boxes/runs_fsq/rcb_plaw4_s1/train_summary.json", d + "/train_summary.json")
    os.link("boxes/runs_fsq/rcb_plaw4_s1/model.pt", d + "/model.pt")


def evalbit():
    da, db = TMP + "/eb_old", TMP + "/eb_new"; run_copy(da); run_copy(db); parts = "ws,test_perm"
    rc = [G8.sh([sys.executable, "-B", "-m", mod, "--run", d, "--parts", parts, "--eval_bs", "100", "--device", DEV], d + "/eval_stdout.log")
          for mod, d in (("boxes._inject_aw_eval_s252", da), ("boxes.inject_aw_eval", db))]
    o = {"rc": rc}
    if rc == [0, 0]:
        dr = lambda j: {k: v for k, v in j.items() if k != "module_sha256"}  # noqa: E731
        wa, wb = json.load(open(da + "/eval_ws.json")), json.load(open(db + "/eval_ws.json")); o["ws_equal"] = dr(wa) == dr(wb)
        ta, tb = json.load(open(da + "/eval_test_perm.json")), json.load(open(db + "/eval_test_perm.json"))
        o["new_field"] = tb["by_d"]["4"]["acc_kprime"].get("2") is not None and "2" in tb["items"]["4"]["ans_kprime"]
        tb2 = json.loads(json.dumps(tb)); tb2["by_d"]["4"]["acc_kprime"].pop("2", None); tb2["items"]["4"]["ans_kprime"].pop("2", None)
        o["test_perm_equal_minus_new"] = dr(ta) == dr(tb2); o["old_has_no_k2"] = "2" not in ta["by_d"]["4"]["acc_kprime"]
    o["pass"] = rc == [0, 0] and o.get("ws_equal") and o.get("test_perm_equal_minus_new") and o.get("new_field") and o.get("old_has_no_k2"); return o


def args():
    ref = json.load(open("boxes/runs_fsq/aw_plaw2_s1/args.json")); o = {}
    for run in ARMS:
        d = G8.fresh(TMP + "/args_" + run)
        rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + flags(run) + tail(run)[:-2] + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")   # no --args_ref (it is this file)
        new = json.load(open(d + "/args.json")); dif = {k: v for k, v in C.diff(ref, new).items() if k != "pair_stats"}
        json.dump(new, open("scratch/s253_args_frozen_%s.json" % run, "w", newline="\n"), indent=1)
        o[run] = {"rc": rc, "keys": sorted(dif), "diff": dif, "micro": new.get("micro"), "auto_micro": new.get("auto_micro"), "eval_bs": new.get("eval_bs"), "seed": new.get("seed")}
    o["pass"] = all(o[r]["rc"] == 0 and o[r]["micro"] == 1 and not o[r]["auto_micro"] and o[r]["eval_bs"] == 250 for r in ARMS); return o


def step0():
    import torch
    d = TMP + "/step0_train"; fl = flags("p2p_s1") + ["--micro", "8", "--lr", "0", "--stop_at", "1", "--ckpt_every", "1", "--log_every", "1"]   # micro 8 locally (8 GB; micro 1 OOMed): at lr 0 no weight moves
    G8.fresh(d); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--device", DEV, "--out", d], d + "/stdout.log"); o = {"train_rc": rc}
    ck = torch.load(d + "/ckpt_last.pt", map_location="cpu", weights_only=False); sd = ck["model"]
    tr = torch.load("boxes/runs_fsq/s252_askperm_s1/model.pt", map_location="cpu", weights_only=True)
    o["trunk_equal_askperm"] = all(("g." + k) in sd and torch.equal(sd["g." + k], v) for k, v in tr.items())
    o["n_trunk_keys"] = len(tr); o["model_keys_sample"] = sorted(sd)[:5]
    run = ROOT + "/boxes/runs_fsq/lb_step0"; os.makedirs(run, exist_ok=True)
    torch.save(sd, run + "/model.pt"); shutil.copy("scratch/s253_args_frozen_p2p_s1.json", run + "/args.json")
    json.dump({"step": 0, "how": "inject_aw 1 step at --lr 0 under P2P's flags (seed 1): the init + g_add calibration; AdamW at lr 0 updates nothing",
               "source_ckpt": d + "/ckpt_last.pt"}, open(run + "/train_summary.json", "w", newline="\n"), indent=1)
    o["eval_rc"] = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw_eval", "--run", run, "--parts", "test_perm", "--eval_bs", "250", "--device", DEV], run + "/eval_stdout.log")
    o["written"] = os.path.exists(run + "/eval_test_perm.json")
    o["pass"] = rc == 9 and o["trunk_equal_askperm"] and o["eval_rc"] == 0 and o["written"]; return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    fns = {"perm": perm, "evalbit": evalbit, "args": args, "step0": step0}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
