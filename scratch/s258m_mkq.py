"""§258 S0 INSTALL cell (LB2_v1 §2 + Note 258-1): queue b397 on ONE RTX A6000 (the S1-S3 card class, pinned) =
  anchor_a6000_eqA_mem / _gtime  K7 mem_check + G-TIME at the S3 shape: inject_aw --base gpt2-medium with RCB's flags minus the small inits (REG d U{1..4}) + --distract_p 1
                       (2-insert mem items), --auto_micro --mem_budget 44 (48 GB card), eval_bs 100 -> the filler's admit measure and the S1-S3 projection;
  s0m_gates            scratch.s258m_host: git blob asserts, then gate_s258m offbit_aw / offbit_ow (flags-off bitwise on GPT-2 SMALL vs the pre-§258 tree s258m_ref.tar),
                       step0 (medium: decoder hidden at K' 1..8 == K' 0 at init, shape 24 / 1024 / 16, DeltaMem 12 / 8 + 8), smoke_aw / smoke_ow (medium plain paths);
  anchor_a6000_eqA     the same-class anchor (Note 258-1 (1)): bridge_s1's exact flags (scratch/s260_args_frozen_bridge_s1.json's run) --stop_at 2000 on the A6000
                       (s257_guarded train: blob + init sha asserted) -> its log.jsonl vs bridge_s1's first 2k on the 3090 (dir suffix _eqA = the drain's log-only marker).
Pre-deploy: unbuffered blob ids == HEAD (majority rule not needed for small files; the init's unbuffered sha, majority of 3).
    python -B scratch/s258m_mkq.py"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); sys.path.insert(0, ROOT + "/scratch/ram_scan"); os.chdir(ROOT)
import gate_s258m as GM  # noqa: E402
import s260_mkq_flags as BF  # noqa: E402
from nobuf_sha import sha_nobuf  # noqa: E402

BLOB = ["boxes/inject_aw.py", "boxes/inject_aw_eval.py", "boxes/insert_render.py", "boxes/perm_render.py", "boxes/s5.py", "boxes/nested170.py", "boxes/inject_ow.py",
        "boxes/deltamem.py", "boxes/core.py", "scratch/s257_guarded.py", "scratch/s258m_host.py", "scratch/gate_s258m.py", "scratch/s258_askname.py",
        "scratch/s256_ho_shas.json", "scratch/s257_ho_shas.json", "scratch/s258m_ref.tar"]
SHIP = ("scratch/s258m_ref.tar scratch/gate_s258m.py scratch/gate_s228.py scratch/gate_s251.py scratch/s222_det_wrap.py scratch/s258_askname.py scratch/s257_guarded.py "
        "boxes/runs_fsq/aw_plaw2_s1/model.pt boxes/runs_fsq/s233_askname_s1/model.pt scratch/s256_ho_shas.json scratch/s257_ho_shas.json")


def nobuf_blob(p):
    src = open("scratch/s258_mkq.py", encoding="utf-8").read(); i = src.index("def nobuf_bytes(p):"); j = src.index("\nEVAL = "); g = {}
    exec("from nobuf_sha import k32, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN\nimport ctypes, os\n"
         "import ctypes.wintypes as W\n" + src[i:j], g)
    b = g["nobuf_bytes"](p); return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


head = {p: subprocess.check_output(["git", "rev-parse", "HEAD:" + p]).decode().strip() for p in BLOB}
pre = {p: {"head": head[p], "disk": nobuf_blob(p)} for p in BLOB}; inits = [sha_nobuf(os.path.abspath(BF.INIT[0])) for _ in range(3)]
ok = all(v["head"] == v["disk"] for v in pre.values()) and sum(s_.startswith(BF.INIT[1]) for s_ in inits) >= 2
os.makedirs("scratch/gate_s258m", exist_ok=True); json.dump({"blobs": pre, "init_unbuffered_sha256_x3": inits, "predeploy_ok": ok}, open("scratch/gate_s258m/predeploy_b397.json", "w", newline="\n"), indent=1)
assert ok, ("pre-deploy", pre, inits)
blob = " ".join("%s=%s" % (p, head[p]) for p in BLOB); mf = " ".join(GM.medium_flags() + ["--distract_p", "1", "--auto_micro", "--mem_budget", "44", "--eval_bs", "100"])
af = " ".join(BF.flags() + ["--keep_final_ckpt", "--ckpt_every", "1000", "--stop_at", "2000", "--eval_parts", "acc"])
q = ("# THEORY §258 S0 INSTALL (LB2_v1 §2 + Note 258-1; RTX A6000 = the S1-S3 class): medium mem / G-TIME at the S3 shape, the host gates (bitwise small, step-0, medium smokes), "
     "the same-class 2k anchor of bridge_s1.\n# HOSTPROBE 1.15\n# P=1\n"   # the A6000 limit (b397 attempt 1: 0.987 s/step vs the 3090-calibrated 0.854; A6000 = GA102 at lower clocks)
     "boxes/runs_fsq/anchor_a6000_eqA_mem boxes.inject_aw %s --mem_check\n"
     "boxes/runs_fsq/anchor_a6000_eqA_gtime boxes.inject_aw %s --time_steps 60\n"
     "boxes/runs_fsq/s0m_gates scratch.s258m_host --blob %s --gates offbit_aw,offbit_ow,step0,smoke_aw,smoke_ow --ship %s\n"
     "boxes/runs_fsq/anchor_a6000_eqA scratch.s257_guarded --mode train --blob %s --init %s=%s -- %s\n") % (mf, mf, blob, SHIP, blob, BF.INIT[0], BF.INIT[1], af)
import gate_s228 as G8  # noqa: E402
import s228_args_check as C  # noqa: E402
d = G8.fresh(ROOT + "/scratch/_tmp/s258m/args_anchor"); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + af.split() + ["--device", "cpu", "--args_only", "--out", d],
                                                                   d + "/stdout.log"); assert rc == 0, rc
dif = {k: v for k, v in C.diff(json.load(open("scratch/s260_args_frozen_bridge_s1.json")), json.load(open(d + "/args.json"))).items() if k != "pair_stats"}
json.dump({"anchor_vs_bridge_frozen": dif}, open("scratch/gate_s258m/anchor_args_diff.json", "w", newline="\n"), indent=1, default=str)
shutil.copy(d + "/args.json", "scratch/s260_args_frozen_anchor_a6000_eqA.json")   # the filler cell = the anchor (post-admit checks its host args.json)
assert set(dif) <= {"stop_at", "eval_parts", "base", "inject_ow_sha256"}, ("anchor != bridge_s1 beyond stop_at / eval_parts / the new base key / the §258 S0 inject_ow sha", sorted(dif))
open("scratch/gen_q_b397.txt.hold", "w", encoding="utf-8", newline="\n").write(q); print("b397 written; pre-deploy OK; anchor diff", sorted(dif))
