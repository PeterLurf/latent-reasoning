"""Note 255-6 DISTRACT pilot as ONE Vast cell (RTX 3090; the local box has RAM bit flips). Queue line: boxes/runs_fsq/pilot_dis_s1 scratch.s256_distract_cell --init <path>
--blob <path>=<git blob id> ... [--ship <files>] (+ --device / --out from the runner). Steps, all on the host:
  0. asserts: sha256(init) starts with 67775e2492044a8a (rcb_plaw4_s1, shipped as an unbuffered-read copy); the git blob id (sha1 of 'blob <len>\\0' + bytes) of every --blob file
     equals the committed one (boxes/inject_aw.py, inject_aw_eval.py, insert_render.py, perm_render.py, this script, scratch/gate_s255d.py, ...).
  1. build checks (Audit a-d) on this GPU: scratch/gate_s255d.py offbit tiny acts smoke gradflow (PILOT_INIT = the init) -> <out>/gates/*.json.
  2. step 0: the INSERT eval (boxes.insert_render.insert_eval_sets = the G-D'd s255_insert sets; E.passes K' 1, 2, 3, 8) on the init under rcb_plaw4_s1's args -> insert_step0.json
     (model key "rcb_plaw4_s1"; insert.json's format).
  3. micro 2 pinned (rcb's realised; the frozen args): --mem_check with the pilot flags + --distract_p 1 (2-insert d 4 items) must peak <= 21.5 GiB, else the cell aborts.
  4. train: rcb_plaw4_s1's flags + --init_model <init> + --distract_p 1 --steps 16000 --stop_at 2000 --ckpt_every 1000 --log_every 100 --eval_bs 250, seed 1 -> <out>
     (exit 9 at 2000; ckpt_prev = step 1000, ckpt_last = step 2000, both asserted by their 'step').
  5. the INSERT eval on the 1k / 2k ckpt weights (args = <out>/args.json) -> insert_1k.json / insert_2k.json (key "pilot_dis_s1").
  6. <out>/model.pt := the 2k weights, <out>/eval.json := the cell summary (the drain's done marker). Exit 0 only if every step passed."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
from boxes import inject_aw_eval as E  # noqa: E402
from boxes import inject_ow as IO  # noqa: E402
import boxes.insert_render as IR  # noqa: E402

RCB = ("--arm inject --think_k 8 --aux_step 1 --aux_fade 48000,48000 --aux_reweight --fast_add --cache_lower --d_max 4 --steps 16000 --cue step --write_src answer "
       "--init_trunk boxes/runs_fsq/s233_askname_s1/model.pt --init_sha c00f7ec0bd7977fe --write_mix_start 0 --write_mix_end 0 --mul_s_p 0").split()
INIT_SHA = "67775e2492044a8a"


def blob_id(p):
    b = open(p, "rb").read(); return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def insert_eval(run_dir, key, dev, bs=100):
    """scratch/s255_insert.py's evaluate() for one model dir (the sets from boxes.insert_render, gate equiv: == the driver's)."""
    raw = IO.build_sets(mul_s_p=0)["test2"][4]; sets, metas = IR.insert_eval_sets(raw); a, m, g = E.load(run_dir, dev); r = {}
    with torch.no_grad():
        for c in IR.CONDS:
            T = E.tokd({"test2": {4: sets[c]}}, a, (4,))[4]; A, _ = E.passes(m, T, dev, bs, {1, 2, 3, 8})
            v = np.array([it["values"] for it in sets[c]]); y = np.array([it["answer"] for it in sets[c]])
            r[c] = {"hop": {str(k): float((A[k] == v[:, k + 1]).mean()) for k in (1, 2, 3)}, "acc8": float((A[8] == y).mean()), "n": len(T),
                    "items": {"sig": [it["sig"] for it in sets[c]], "ans": {str(k): A[k].tolist() for k in (1, 2, 3, 8)}, "insert": metas[c]}}
    del m, g; torch.cuda.empty_cache()
    return {key: r, "meta": {"conds": IR.CONDS, "bs": bs, "dtype": "fp32", "gpu": torch.cuda.get_device_name(0)},
            "def": "Note 255-3 INSERT (boxes.insert_render; == scratch/s255_insert.py's sets): hop[k] = share with answer at K' = k == v_{k+1}, acc8 = acc at K' 8"}


def model_dir(sd, args_json, d):
    os.makedirs(d, exist_ok=True); shutil.copy(args_json, d + "/args.json"); torch.save(sd, d + "/model.pt"); return d


def run(cmd, log):
    with open(log, "w") as f:
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--init", required=True); ap.add_argument("--blob", nargs="+", default=[]); ap.add_argument("--ship", nargs="*", default=[])
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True); a = ap.parse_args(); dev = a.device; out = a.out
    stage = out + "_stage"; os.makedirs(stage, exist_ok=True); summ = {"steps": {}}
    # 0. shas
    s_init = hashlib.sha256(open(a.init, "rb").read()).hexdigest(); assert s_init.startswith(INIT_SHA), ("init sha", s_init)
    bl = {}
    for x in a.blob:
        p, want = x.split("="); got = blob_id(p); bl[p] = got == want; assert got == want, ("git blob", p, got, want)
    summ["steps"]["0_shas"] = {"init_sha256": s_init, "blobs_ok": bl}
    # 1. build checks on this GPU
    env = dict(os.environ, PILOT_INIT=a.init)
    rc = subprocess.call([sys.executable, "-B", "scratch/gate_s255d.py", "offbit", "tiny", "acts", "smoke", "gradflow"], env=env, stdout=open(stage + "/gates.log", "w"), stderr=subprocess.STDOUT)
    gd = stage + "/gates"; os.makedirs(gd, exist_ok=True); gp = {}
    for p_ in ("offbit", "tiny", "acts", "smoke", "gradflow"):
        f = "scratch/gate_s255d/%s.json" % p_
        if os.path.exists(f):
            shutil.copy(f, gd + "/%s.json" % p_); gp[p_] = json.load(open(f)).get("pass")
    summ["steps"]["1_gates"] = {"rc": rc, "pass": gp}; assert rc == 0 and all(gp.get(p_) for p_ in ("offbit", "tiny", "acts", "smoke", "gradflow")), ("gates", rc, gp)
    # 2. step-0 INSERT on the init under rcb_plaw4_s1's args
    d0 = model_dir(torch.load(a.init, map_location="cpu", weights_only=True), "boxes/runs_fsq/rcb_plaw4_s1/args.json", stage + "/m_step0")
    json.dump(insert_eval(d0, "rcb_plaw4_s1", dev), open(stage + "/insert_step0.json", "w", newline="\n"), indent=1); os.remove(d0 + "/model.pt")
    summ["steps"]["2_step0"] = "insert_step0.json"
    # 3. micro
    PF = RCB + ["--init_model", a.init, "--init_model_sha", INIT_SHA, "--distract_p", "1"]; M = 2   # micro pinned = rcb's realised 2 (the frozen args); the 2-insert check must fit
    d = stage + "/memck_2"; run([sys.executable, "-B", "-m", "boxes.inject_aw"] + PF + ["--micro", "2", "--eval_bs", "250", "--mem_check", "--device", dev, "--out", d], stage + "/memck_2.log")
    pk = json.load(open(d + "/mem_check.json"))["peak_gib_train"] if os.path.exists(d + "/mem_check.json") else None
    summ["steps"]["3_micro"] = {"micro": M, "peak_gib_train_2insert": pk}; assert pk is not None and pk <= 21.5, ("micro 2 does not fit with 2 inserts", pk)
    # 4. train
    rc = run([sys.executable, "-B", "-m", "boxes.inject_aw"] + PF + ["--micro", str(M), "--eval_bs", "250", "--stop_at", "2000", "--ckpt_every", "1000", "--log_every", "100",
                                                                     "--device", dev, "--out", out], stage + "/train.log")
    summ["steps"]["4_train"] = {"rc": rc}; assert rc == 9, ("train rc", rc)
    # 5. 1k / 2k
    for tag, f, step in (("1k", "ckpt_prev.pt", 1000), ("2k", "ckpt_last.pt", 2000)):
        ck = torch.load(out + "/" + f, map_location="cpu", weights_only=False); assert ck["step"] == step, (f, ck["step"])
        d = model_dir(ck["model"], out + "/args.json", stage + "/m_" + tag); del ck
        json.dump(insert_eval(d, "pilot_dis_s1", dev), open(out + "/insert_%s.json" % tag, "w", newline="\n"), indent=1)
        if tag == "2k":
            shutil.move(d + "/model.pt", out + "/model.pt")
        else:
            os.remove(d + "/model.pt")
    # 6. the stage outputs into <out>, the summary
    shutil.copy(stage + "/insert_step0.json", out + "/insert_step0.json"); shutil.copytree(gd, out + "/gates", dirs_exist_ok=True)
    for f in ("gates.log", "train.log"):
        shutil.copy(stage + "/" + f, out + "/cell_" + f)
    summ["ok"] = True; json.dump(summ, open(out + "/eval.json", "w", newline="\n"), indent=1); print("CELL DONE", flush=True); return 0


if __name__ == "__main__":
    sys.exit(main())
