"""Note 259-5 C5-BUN eval-only cell (scorer theory/score_c5b.py): queue b407 on one RTX 3090 = K7 mem + G-TIME probe lines (dch_s1's flags on the bun_s1 copy; the filler's
admit measure), then per bun_s<k> one EVAL-ONLY line boxes/runs_fsq/bun_s<k>_c5b through scratch.s259_guarded --mode evalonly (git blobs + the streaming-copied
bun_s<k>_init sha asserted on the host): --parts halt_inc,halthead2 -> eval_halt_inc.json / eval_halthead2.json; host_shas.json after. The landing copies both into
boxes/runs_fsq/bun_s<k>/ after the host-sha check. Every model / data file shipped plainly (check_queue_ships).
    python -B scratch/s263_mkq.py"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); sys.path.insert(0, ROOT + "/scratch/ram_scan"); os.chdir(ROOT)
import gate_s257 as G7  # noqa: E402
from nobuf_sha import sha_nobuf  # noqa: E402

INIT = {1: ("boxes/runs_fsq/bun_s1_init/model.pt", "1ad3517b7635fddf"), 2: ("boxes/runs_fsq/bun_s2_init/model.pt", "09337f63aec57959")}   # scratch/ram_scan/nobuf_copy.py
BLOB = ["boxes/inject_aw.py", "boxes/inject_aw_eval.py", "boxes/insert_render.py", "boxes/perm_render.py", "scratch/s259_guarded.py", "scratch/s256_ho_shas.json",
        "scratch/s257_ho_shas.json", "boxes/s5.py", "boxes/nested170.py", "boxes/inject_ow.py", "boxes/deltamem.py", "boxes/core.py"]


def nobuf_blob(p):
    src = open("scratch/s258_mkq.py", encoding="utf-8").read(); i = src.index("def nobuf_bytes(p):"); j = src.index("\nEVAL = "); g = {}
    exec("from nobuf_sha import k32, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN\nimport ctypes, os\n"
         "import ctypes.wintypes as W\n" + src[i:j], g)
    b = g["nobuf_bytes"](p); return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


head = {p: subprocess.check_output(["git", "rev-parse", "HEAD:" + p]).decode().strip() for p in BLOB}
pre = {p: {"head": head[p], "disk": nobuf_blob(p)} for p in BLOB}; inits = {k: [sha_nobuf(os.path.abspath(v[0])) for _ in range(3)] for k, v in INIT.items()}
ok = all(v["head"] == v["disk"] for v in pre.values()) and all(sum(s_.startswith(INIT[k][1]) for s_ in inits[k]) >= 2 for k in INIT)
os.makedirs("scratch/gate_s263", exist_ok=True); json.dump({"blobs": pre, "init_unbuffered_sha256_x3": inits, "predeploy_ok": ok}, open("scratch/gate_s263/predeploy.json", "w", newline="\n"), indent=1)
assert ok, ("pre-deploy", pre, inits)
blob = " ".join("%s=%s" % (p, head[p]) for p in BLOB); fs = " ".join(G7.dch_flags(1, *INIT[1]))
q = ("# THEORY Note 259-5 C5-BUN eval-only (score_c5b): halt_inc (rel_inc halting, tau calibrated on a fresh REG fit set) + halthead2 (K' 0..14) on bun_s1 / bun_s2 "
     "(streaming-copied, sha asserted on the host); scratch.s259_guarded; host_shas.json per dir.\n# HOSTPROBE\n# P=1\n"
     "boxes/runs_fsq/bun_s1_c5b_mem boxes.inject_aw %s --mem_check\n"
     "boxes/runs_fsq/bun_s1_c5b_gtime boxes.inject_aw %s --time_steps 60\n") % (fs, fs)
shutil.copy("boxes/runs_fsq/bun_s1/args.json", "scratch/s262_args_frozen_bun_s1_c5b.json")   # the filler cell = bun_s1_c5b (post-admit checks its copied args.json)
for k in (1, 2):
    q += ("boxes/runs_fsq/bun_s%d_c5b scratch.s259_guarded --mode evalonly --blob %s --model %s=%s --args boxes/runs_fsq/bun_s%d/args.json --parts halt_inc,halthead2 "
          "--ship scratch/s256_ho_shas.json scratch/s257_ho_shas.json %s\n") % (k, blob, INIT[k][0], INIT[k][1], k, INIT[k][0])
open("scratch/gen_q_b407.txt.hold", "w", encoding="utf-8", newline="\n").write(q)
assert subprocess.call([sys.executable, "scratch/check_queue_ships.py", "scratch/gen_q_b407.txt.hold"]) == 0; print("b407 written; pre-deploy OK")
