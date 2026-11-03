"""§264 SCORECARD-CHECK + §265 gates (THEORY §264 / §265, Notes 264-1 / 265-1): queue b417 on one RTX 3090 (eval only; filler family s271 = template v5, drain gen10) =
  dd_gates     scratch.s271_gates: gate_s271 offbit_aw (flags-off bitwise vs the pre-§265 tree scratch/s271_ref.tar) / smoke_dd (--distract_nmax 8) / gradflow
               -> gates_pass.json (the §265 pilot TRAIN lines --require the landed copy);
  bun_s<k>_sc  scratch.s271_guarded --mode evalonly: bun_s<k> (the streaming copy == bun_s<k>/model.pt, sha on host) under bun_s<k>'s own args -> eval_cue_clock (§264),
               eval_op_count (§264), eval_attn_fade (§265 D1a) in its own dir (the landing copies them into bun_s<k>/: score_sc reads bun_s<k>/eval_cue_clock.json).
Pre-deploy: unbuffered blob ids == HEAD; the inits' streaming shas (majority of 3); ships checked.
    python -B scratch/s271_mkq.py"""
import hashlib
import json
import os
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); sys.path.insert(0, ROOT + "/scratch/ram_scan"); os.chdir(ROOT)
from nobuf_sha import sha_nobuf  # noqa: E402

INIT = {1: ("boxes/runs_fsq/bun_s1_init/model.pt", "1ad3517b7635fddf"), 2: ("boxes/runs_fsq/bun_s2_init/model.pt", "09337f63aec57959")}
BLOB = ["boxes/inject_aw.py", "boxes/inject_aw_eval.py", "boxes/sc_parts.py", "boxes/insert_render.py", "boxes/perm_render.py", "boxes/s5.py", "boxes/nested170.py",
        "boxes/inject_ow.py", "boxes/inject_st.py", "boxes/deltamem.py", "boxes/deltas5.py", "boxes/core.py", "boxes/ow_words.py", "boxes/nested_words.py",
        "scratch/s257_guarded.py", "scratch/s271_guarded.py", "scratch/s271_gates.py", "scratch/gate_s271.py", "scratch/gate_s257.py", "scratch/gate_s251.py", "scratch/gate_s228.py",
        "scratch/s222_det_wrap.py", "scratch/s271_ref.tar", "scratch/s256_ho_shas.json", "scratch/s257_ho_shas.json"]
GSHIP = ("scratch/s271_ref.tar scratch/gate_s271.py scratch/gate_s257.py scratch/gate_s251.py scratch/gate_s228.py scratch/s228_args_check.py scratch/s222_det_wrap.py "
         "scratch/s257_guarded.py boxes/runs_fsq/bun_s1_init/model.pt boxes/runs_fsq/s233_askname_s1/model.pt scratch/s256_ho_shas.json scratch/s257_ho_shas.json")
PARTS = "cue_clock,op_count,attn_fade"


def nobuf_blob(p):
    src = open("scratch/s258_mkq.py", encoding="utf-8").read(); i = src.index("def nobuf_bytes(p):"); j = src.index("\nEVAL = "); g = {}
    exec("from nobuf_sha import k32, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN\nimport ctypes, os\n"
         "import ctypes.wintypes as W\n" + src[i:j], g)
    b = g["nobuf_bytes"](p); return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


head = {p: subprocess.check_output(["git", "rev-parse", "HEAD:" + p]).decode().strip() for p in BLOB}; pre = {p: {"head": head[p], "disk": nobuf_blob(p)} for p in BLOB}
inits = {k: [sha_nobuf(os.path.abspath(v[0])) for _ in range(3)] for k, v in INIT.items()}
ok = all(v["head"] == v["disk"] for v in pre.values()) and all(sum(s_.startswith(INIT[k][1]) for s_ in inits[k]) >= 2 for k in INIT)
out = {"blobs": pre, "init_unbuffered_sha256_x3": inits, "predeploy_ok": ok}; assert ok, ("pre-deploy", out)
blob = " ".join("%s=%s" % (p, head[p]) for p in BLOB); SH = "scratch/s256_ho_shas.json scratch/s257_ho_shas.json"
sys.path.insert(0, ROOT + "/scratch")
import caps_check as CC  # noqa: E402
import gate_s257 as G7  # noqa: E402
import shutil  # noqa: E402
fl1 = " ".join(G7.dch_flags(1, *INIT[1]) + ["--hold_inc", "1.6", "--hold_log"]); W1 = "boxes/runs_fsq/bun_s1_sc"   # the b408 pattern: QS run = bun_s1_sc (its _mem / _gtime first)
shutil.copy("boxes/runs_fsq/bun_s1/args.json", "scratch/s271_args_frozen_bun_s1_sc.json")   # post_admit: the evalonly dir's args.json = a copy of bun_s1's
body = ("%s_mem boxes.inject_aw %s --mem_check\n%s_gtime boxes.inject_aw %s --time_steps 60\n"
        "boxes/runs_fsq/dd_gates scratch.s271_gates --blob %s --gates offbit_aw,smoke_dd,gradflow --ship %s\n") % (W1, fl1, W1, fl1, blob, GSHIP)
for k in (1, 2):
    body += ("boxes/runs_fsq/bun_s%d_sc scratch.s271_guarded --mode evalonly --blob %s --model %s=%s --args boxes/runs_fsq/bun_s%d/args.json --parts %s --ship %s "
             "boxes/runs_fsq/bun_s%d/args.json %s\n") % (k, blob, INIT[k][0], INIT[k][1], k, PARTS, INIT[k][0], k, SH)
nof = "# NOFETCH boxes/runs_fsq/bun_s1_sc/model.pt\n# NOFETCH boxes/runs_fsq/bun_s2_sc/model.pt\n"
caps = CC.caps_line([0.1, 0.1, 1.2, 0.4, 0.4], 0.8, 1.2, 1.65, 0.854, "RTX_3090", CC.fetch_gb(nof + body))   # template v6: the caps travel with the queue
q = ("# THEORY §264 SCORECARD-CHECK (score_sc; eval only on bun) + §265 DENSE-DISTRACT host gates (Note 265-1) + §265 D1a attn_fade on bun: one host; the §265 pilot "
     "TRAIN lines require this host's landed gates_pass.json.\n# HOSTPROBE\n# P=1\n%s\n%s%s") % (caps, nof, body)
open("scratch/gen_q_b417.txt.hold", "w", encoding="utf-8", newline="\n").write(q)
assert subprocess.call([sys.executable, "scratch/check_queue_ships.py", "scratch/gen_q_b417.txt.hold"]) == 0
assert subprocess.call([sys.executable, "-B", "scratch/queue_dryrun.py", "scratch/gen_q_b417.txt.hold"]) == 0   # template v6 item 2
os.makedirs("scratch/gate_s271", exist_ok=True); json.dump(out, open("scratch/gate_s271/predeploy.json", "w", newline="\n"), indent=1, default=str); print("b417 queue written")
