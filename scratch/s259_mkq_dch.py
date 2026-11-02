"""Note 259-1: the dch cell of Note 257-2 (= scratch/s259_mkq.py with dis -> dch, b395 -> b400). Note 257-2 eval-only cell (scorer theory/score_hx.py): queue b400 on one RTX 3090 = K7 mem + G-TIME probe lines (dch_s1's flags; the filler's admit measure), then per
dch_s<k> one EVAL-ONLY line boxes/runs_fsq/dch_s<k>_hx259 through scratch.s259_guarded --mode evalonly (git blob ids + the unbuffered-read dch_s<k>_init copy's sha
asserted on the host): --parts halthead (§248 HALT-HEAD on the run's own law = REG: fit 6,000 fresh REG-stream items d U{1..4}, excl the eval sets, TEST2 d 1..7) and
--long 8,64,128:1,2,3,4,5,6:eval_long_k128.json (§239 run_long: by_d[d].acc / s_norm (mean ||S||) / s_cos / nonfinite per K'); host_shas.json after.
The landing copies eval_halthead.json / eval_long_k128.json into boxes/runs_fsq/dch_s<k>/ (score_hx's RUNS) after the host-sha check. Pre-deploy: unbuffered blob ids == HEAD,
init copies' unbuffered sha256 x3.
    python -B scratch/s259_mkq.py"""
import hashlib
import json
import os
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); sys.path.insert(0, ROOT + "/scratch/ram_scan"); os.chdir(ROOT)
import gate_s257 as G7  # noqa: E402
from nobuf_sha import sha_nobuf  # noqa: E402

INIT = {1: ("boxes/runs_fsq/dch_s1_init/model.pt", "0badc805792215b3"), 2: ("boxes/runs_fsq/dch_s2_init/model.pt", "d759f9a2d1014f21")}   # scratch/s261_build/mk_dch_copies.py
BLOB = ["boxes/inject_aw.py", "boxes/inject_aw_eval.py", "boxes/insert_render.py", "boxes/perm_render.py", "scratch/s259_guarded.py", "scratch/s256_ho_shas.json",
        "scratch/s257_ho_shas.json", "boxes/s5.py", "boxes/nested170.py", "boxes/inject_ow.py", "boxes/deltamem.py", "boxes/core.py"]   # + the §258-edited load path


def nobuf_blob(p):
    src = open("scratch/s258_mkq.py", encoding="utf-8").read(); i = src.index("def nobuf_bytes(p):"); j = src.index("\nEVAL = "); g = {}
    exec("from nobuf_sha import k32, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN\nimport ctypes, os\n"
         "import ctypes.wintypes as W\n" + src[i:j], g)
    b = g["nobuf_bytes"](p); return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


head = {p: subprocess.check_output(["git", "rev-parse", "HEAD:" + p]).decode().strip() for p in BLOB}
pre = {p: {"head": head[p], "disk": nobuf_blob(p)} for p in BLOB}; inits = {k: [sha_nobuf(os.path.abspath(v[0])) for _ in range(3)] for k, v in INIT.items()}
ok = all(v["head"] == v["disk"] for v in pre.values()) and all(all(s_.startswith(INIT[k][1]) for s_ in inits[k]) for k in INIT)
os.makedirs("scratch/gate_s259", exist_ok=True); json.dump({"blobs": pre, "init_unbuffered_sha256_x3": inits, "predeploy_ok": ok}, open("scratch/gate_s259/predeploy_b400.json", "w", newline="\n"), indent=1)
assert ok, ("pre-deploy", pre, inits)
blob = " ".join("%s=%s" % (p, head[p]) for p in BLOB); fs = " ".join(G7.dch_flags(1, *INIT[1]))
q = ("# THEORY Note 257-2 eval-only (score_hx): halthead (REG fit, TEST2 d 1..7) + run_long K' 8 / 64 / 128 d 1..6 on dch_s1 / dch_s2 (the unbuffered-read init copies, "
     "sha asserted on the host); scratch.s259_guarded; host_shas.json per dir.\n# HOSTPROBE\n# P=1\n"
     "boxes/runs_fsq/dch_s1_hx259_mem boxes.inject_aw %s --mem_check\n"
     "boxes/runs_fsq/dch_s1_hx259_gtime boxes.inject_aw %s --time_steps 60\n") % (fs, fs)   # the filler cell = dch_s1_hx259 (post-admit checks its args.json)
import shutil  # noqa: E402
shutil.copy("boxes/runs_fsq/dch_s1/args.json", "scratch/s261_args_frozen_dch_s1_hx259.json")   # the eval-only line copies dch_s1's args.json into the cell dir
for k in (1, 2):
    q += ("boxes/runs_fsq/dch_s%d_hx259 scratch.s259_guarded --mode evalonly --blob %s --model %s=%s --args boxes/runs_fsq/dch_s%d/args.json --parts halthead "
          "--long 8,64,128:1,2,3,4,5,6:eval_long_k128.json --ship scratch/s256_ho_shas.json scratch/s257_ho_shas.json\n") % (k, blob, INIT[k][0], INIT[k][1], k)
open("scratch/gen_q_b400.txt.hold", "w", encoding="utf-8", newline="\n").write(q); print("b400 written; pre-deploy OK")
