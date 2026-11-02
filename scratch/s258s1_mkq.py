"""§258 S1 ASKNAME-M (LB2_v1 §2; Note 258-1): queues b398 askm_lr3e5 / b399 askm_lr1e4, one RTX A6000 each (the pinned S1-S3 class; HOSTPROBE 1.15) =
  <run>_mem / <run>_gtime  K7 mem_check + G-TIME of the direct arm on gpt2-medium (inject_ow, the s233 shape; the filler's admit measure);
  s1g_<tag>                the S0 host gates FIRST on this host (scratch.s258m_host: offbit_aw / offbit_ow bitwise small vs the pre-§258 tree, step0, medium smokes);
  <run>                    scratch.s258_guarded --require s1g_<tag>/gates_pass.json --entry scratch/s258_askname.py: the s233 recipe exactly (askname classes, --arm direct,
                           d_max 2, 8k, bs 64, seed 1, eval_bs 125, eval_parts acc, ckpt_every 2000) + --base gpt2-medium, --lr 3e-5 | 1e-4, micro 2 PINNED (S0 G-TIME's micro;
                           s233 used --auto_micro on a 4060 -> micro 1: disclosed) -> eval_acc.json + eval_st_classes.json.
Frozen args (--args_only through the wrapper) + diff vs s233_askname_s1/args.json. Pre-deploy: unbuffered blob ids == HEAD.
    python -B scratch/s258s1_mkq.py"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
import s228_args_check as C  # noqa: E402

BLOB = ["boxes/inject_aw.py", "boxes/inject_aw_eval.py", "boxes/insert_render.py", "boxes/perm_render.py", "boxes/s5.py", "boxes/nested170.py", "boxes/inject_ow.py",
        "boxes/deltamem.py", "boxes/core.py", "scratch/s257_guarded.py", "scratch/s258_guarded.py", "scratch/s258m_host.py", "scratch/gate_s258m.py",
        "scratch/s258_askname.py", "scratch/s256_ho_shas.json", "scratch/s257_ho_shas.json", "scratch/s258m_ref.tar"]
GSHIP = ("scratch/s258m_ref.tar scratch/gate_s258m.py scratch/gate_s228.py scratch/gate_s251.py scratch/s222_det_wrap.py scratch/s258_askname.py scratch/s257_guarded.py "
         "boxes/runs_fsq/aw_plaw2_s1/model.pt boxes/runs_fsq/s233_askname_s1/model.pt scratch/s256_ho_shas.json scratch/s257_ho_shas.json")
REC = ["--d_max", "2", "--steps", "8000", "--bs", "64", "--seed", "1", "--eval_bs", "125", "--eval_parts", "acc", "--ckpt_every", "2000", "--log_every", "100",
       "--base", "gpt2-medium", "--micro", "2"]


def nobuf_blob(p):
    src = open("scratch/s258_mkq.py", encoding="utf-8").read(); i = src.index("def nobuf_bytes(p):"); j = src.index("\nEVAL = "); g = {}
    exec("from nobuf_sha import k32, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN\nimport ctypes, os\n"
         "import ctypes.wintypes as W\n" + src[i:j], g)
    b = g["nobuf_bytes"](p); return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


sys.path.insert(0, ROOT + "/scratch/ram_scan")
head = {p: subprocess.check_output(["git", "rev-parse", "HEAD:" + p]).decode().strip() for p in BLOB}; pre = {p: {"head": head[p], "disk": nobuf_blob(p)} for p in BLOB}
ok = all(v["head"] == v["disk"] for v in pre.values()); out = {"blobs": pre, "predeploy_ok": ok, "args": {}}
assert ok, ("pre-deploy", pre)
blob = " ".join("%s=%s" % (p, head[p]) for p in BLOB); ref = json.load(open("boxes/runs_fsq/s233_askname_s1/args.json"))
for n, tag, lr in ((398, "lr3e5", "3e-5"), (399, "lr1e4", "1e-4")):
    run = "askm_" + tag; fl = REC + ["--lr", lr]; d = G8.fresh(ROOT + "/scratch/_tmp/s258s1/args_" + run)
    rc = G8.sh([sys.executable, "-B", "scratch/s258_askname.py"] + fl + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log"); assert rc == 0, rc
    new = json.load(open(d + "/args.json")); shutil.copy(d + "/args.json", "scratch/s260_args_frozen_%s.json" % run)
    dif = {k: v for k, v in C.diff(ref, new).items() if k != "pair_stats"}; out["args"][run] = {"keys": sorted(dif), "diff": dif}
    pf = " ".join(["--arm", "direct"] + fl)
    q = ("# THEORY §258 S1 ASKNAME-M (LB2_v1 §2): %s = the s233 askname recipe on gpt2-medium, lr %s, micro 2 pinned; this host's S0 gates first (gates_pass.json required). "
         "Frozen args scratch/s260_args_frozen_%s.json.\n# HOSTPROBE 1.15\n# P=1\n"
         "boxes/runs_fsq/%s_mem boxes.inject_ow %s --mem_check\n"
         "boxes/runs_fsq/%s_gtime boxes.inject_ow %s --time_steps 60\n"
         "boxes/runs_fsq/s1g_%s scratch.s258m_host --blob %s --gates offbit_aw,offbit_ow,step0,smoke_aw,smoke_ow --ship %s\n"
         "boxes/runs_fsq/%s scratch.s258_guarded --blob %s --require boxes/runs_fsq/s1g_%s/gates_pass.json --entry scratch/s258_askname.py -- %s --args_ref "
         "scratch/s260_args_frozen_%s.json\n") % (run, lr, run, run, pf, run, pf, tag, blob, GSHIP, run, blob, tag, " ".join(fl), run)
    open("scratch/gen_q_b%d.txt.hold" % n, "w", encoding="utf-8", newline="\n").write(q)
os.makedirs("scratch/gate_s258m", exist_ok=True); json.dump(out, open("scratch/gate_s258m/predeploy_s1.json", "w", newline="\n"), indent=1, default=str)
print({r: v["keys"] for r, v in out["args"].items()})
