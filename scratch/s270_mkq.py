"""§263 ORDER-FREE (THEORY §263, Note 263-1; score_of): queues b415 of_s1 / b416 of_s2, one RTX 3090 each (filler family s270 = template v5, drain gen10) =
  <run>_mem / _gtime  K7 mem_check + G-TIME (the cell's flags; a crash refuses the host);
  <run>               scratch.s257_guarded --mode train: bun_s<k>'s flags (dch_flags + --hold_inc 1.6 --hold_log; the §257 insert mix) + the §262 CTRL recipe (--perm_p 0.5
                      perm-then-insert, --ask_p 0.25 one-hop ':'-CE co-training; no NK), --init_model the bun_s<k> streaming copy, --steps 8000, micro 2 pinned;
                      eval_parts acc,test_perm,insert_ho,swap_kd,lp5h,stab (the existing parts) in-process; no gates line: TRAIN --require's b413's landed gates_pass.json
                      (inject_aw.py blob == b413's; the §262 gates covered this exact trainer path incl. smoke_ctrl) - Audit to confirm at G-D;
  <run> (new parts)   boxes.inject_aw_eval --parts late_premise,name_swap,shuffle_graded (boxes/of_parts.py; their own line so a new-part crash cannot cost the
                      in-process evals' eval.json);
  <run> (long)        scratch.s262_long: run_long K' 8 / 64 / 128 on d 1-6 -> eval_long_k128.json.
Frozen args (scratch/s270_args_frozen_<run>.json) + diff vs bun_s<k>/args.json. Pre-deploy: unbuffered blob ids == HEAD; the inits' streaming shas (majority of 3); ships checked.
    python -B scratch/s270_mkq.py"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); sys.path.insert(0, ROOT + "/scratch/ram_scan"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
import gate_s257 as G7  # noqa: E402
import s228_args_check as C  # noqa: E402
from nobuf_sha import sha_nobuf  # noqa: E402

INIT = {1: ("boxes/runs_fsq/bun_s1_init/model.pt", "1ad3517b7635fddf"), 2: ("boxes/runs_fsq/bun_s2_init/model.pt", "09337f63aec57959")}
GP = "boxes/runs_fsq/pilot_nk_s1_gates/gates_pass.json"
BLOB = ["boxes/inject_aw.py", "boxes/inject_aw_eval.py", "boxes/of_parts.py", "boxes/ask_render.py", "boxes/insert_render.py", "boxes/perm_render.py", "boxes/s5.py",
        "boxes/nested170.py", "boxes/inject_ow.py", "boxes/inject_st.py", "boxes/deltamem.py", "boxes/deltas5.py", "boxes/core.py", "boxes/ow_words.py", "boxes/nested_words.py",
        "scratch/s257_guarded.py", "scratch/s262_long.py", "scratch/s256_ho_shas.json", "scratch/s257_ho_shas.json"]
EVAL = "acc,test_perm,insert_ho,swap_kd,lp5h,stab"; NEWP = "late_premise,name_swap,shuffle_graded"
CELLS = ((415, "of_s1", 1), (416, "of_s2", 2))


def nobuf_blob(p):
    src = open("scratch/s258_mkq.py", encoding="utf-8").read(); i = src.index("def nobuf_bytes(p):"); j = src.index("\nEVAL = "); g = {}
    exec("from nobuf_sha import k32, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN\nimport ctypes, os\n"
         "import ctypes.wintypes as W\n" + src[i:j], g)
    b = g["nobuf_bytes"](p); return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def flags(k):
    return G7.dch_flags(k, *INIT[k]) + ["--hold_inc", "1.6", "--hold_log", "--perm_p", "0.5", "--ask_p", "0.25"]


head = {p: subprocess.check_output(["git", "rev-parse", "HEAD:" + p]).decode().strip() for p in BLOB}; pre = {p: {"head": head[p], "disk": nobuf_blob(p)} for p in BLOB}
inits = {k: [sha_nobuf(os.path.abspath(v[0])) for _ in range(3)] for k, v in INIT.items()}
b413 = [t.split("=")[1] for t in open("scratch/gen_q_b413.txt", encoding="utf-8").read().split() if t.startswith("boxes/inject_aw.py=")][0]
ok = (all(v["head"] == v["disk"] for v in pre.values()) and all(sum(s_.startswith(INIT[k][1]) for s_ in inits[k]) >= 2 for k in INIT) and os.path.exists(GP)
      and head["boxes/inject_aw.py"] == b413)
out = {"blobs": pre, "init_unbuffered_sha256_x3": inits, "trainer_blob_eq_b413": head["boxes/inject_aw.py"] == b413, "predeploy_ok": ok, "args": {}}
assert ok, ("pre-deploy", out)
blob = " ".join("%s=%s" % (p, head[p]) for p in BLOB); SH = "scratch/s256_ho_shas.json scratch/s257_ho_shas.json"
for n, run, k in CELLS:
    fl = flags(k); tail = ["--ckpt_every", "1000", "--eval_parts", EVAL]; d = G8.fresh(ROOT + "/scratch/_tmp/s270/args_" + run)
    rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + tail + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log"); assert rc == 0, (run, rc)
    shutil.copy(d + "/args.json", "scratch/s270_args_frozen_%s.json" % run)
    dif = {kk: v for kk, v in C.diff(json.load(open("boxes/runs_fsq/bun_s%d/args.json" % k)), json.load(open(d + "/args.json"))).items() if kk != "pair_stats"}
    out["args"][run] = {"keys": sorted(dif), "diff": dif}; fs = " ".join(fl); W = "boxes/runs_fsq/" + run
    q = ("# THEORY §263 ORDER-FREE (score_of): %s = bun_s%d (streaming copy, sha on host) + bun's args + the §262 CTRL recipe (perm-then-insert --perm_p 0.5, one-hop "
         "':'-CE --ask_p 0.25, no NK), 8k, micro 2 pinned; TRAIN requires b413's gates_pass.json (same trainer blob); evals in-process (existing parts), then the new "
         "§263 parts, then run_long K' 8/64/128 d 1-6. Frozen args scratch/s270_args_frozen_%s.json.\n# HOSTPROBE\n# P=1\n"
         "%s_mem boxes.inject_aw %s --mem_check\n"
         "%s_gtime boxes.inject_aw %s --time_steps 60\n"
         "%s scratch.s257_guarded --mode train --blob %s --init %s=%s --require %s --ship %s %s %s -- %s %s --args_ref scratch/s270_args_frozen_%s.json\n"
         "%s boxes.inject_aw_eval --run %s --parts %s --eval_bs 250\n"
         "%s scratch.s262_long --kprime_list 8,64,128 --long_ds 1,2,3,4,5,6 --long_name eval_long_k128.json\n") % (
        run, k, run, W, fs, W, fs, W, blob, INIT[k][0], INIT[k][1], GP, GP, INIT[k][0], SH, fs, " ".join(tail), run, W, W, NEWP, W)
    open("scratch/gen_q_b%d.txt.hold" % n, "w", encoding="utf-8", newline="\n").write(q)
    assert subprocess.call([sys.executable, "scratch/check_queue_ships.py", "scratch/gen_q_b%d.txt.hold" % n]) == 0
os.makedirs("scratch/gate_s270", exist_ok=True); json.dump(out, open("scratch/gate_s270/predeploy.json", "w", newline="\n"), indent=1, default=str)
print({r: v["keys"] for r, v in out["args"].items()})
