"""§259 BUNDLE (Note 259-2; scorer theory/score_bu.py): queues b402 bun_s1 / b403 bun_s2 / b404 bunc_s1 / b405 bunc_s2, one RTX 3090 each (micro 2 pinned) =
  <run>_mem / _gtime  K7 mem_check (WITH the penalty + the §257 distract mix: the 2-insert memory items) + G-TIME (the filler's admit measure; a crash refuses the host);
  <run>               scratch.s257_guarded --mode train (git blobs + the dch_s<k> init copy's sha asserted on the host): dch_s<k>'s flags (gate_s257.dch_flags: dis_s<k>'s +
                      --distract_mix s256,chain,givop, --steps 8000) with --init_model the unbuffered dch_s<k> copy + BUN: --hold_inc 1.6 --hold_log (hold_start 500) |
                      BUNC: --hold_log only; --keep_final_ckpt; eval_parts BUN acc,swap_kd,lp5h,cue_only,ws_deep,halthead | BUNC acc,ws_deep (Audit);
  <run> (both arms)   scratch.s262_long: run_long K' 8 / 64 / 128 on d 1-6 -> eval_long_k128.json (acc + s_norm).
Fresh AdamW, 100-step warm-up + cosine over 8k, mem.fade re-ramp over 500 steps (as dis -> dch; disclosed). Frozen args + diff vs dch_s<k>/args.json.
Pre-deploy: unbuffered blob ids == HEAD; the init copies' unbuffered sha (majority of 3); every path=sha model shipped (check_queue_ships).
    python -B scratch/s262_mkq.py"""
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

INIT = {1: ("boxes/runs_fsq/dch_s1_init/model.pt", "0badc805792215b3"), 2: ("boxes/runs_fsq/dch_s2_init/model.pt", "d759f9a2d1014f21")}
BLOB = ["boxes/inject_aw.py", "boxes/inject_aw_eval.py", "boxes/insert_render.py", "boxes/perm_render.py", "boxes/s5.py", "boxes/nested170.py", "boxes/inject_ow.py",
        "boxes/deltamem.py", "boxes/core.py", "scratch/s257_guarded.py", "scratch/s262_long.py", "scratch/s256_ho_shas.json", "scratch/s257_ho_shas.json"]
EV_BUN = "acc,swap_kd,lp5h,cue_only,ws_deep,halthead"; EV_BUNC = "acc,ws_deep"   # Audit G-D change 1: BUNC + ws_deep + the long line (C2 / C6 attribution)


def nobuf_blob(p):
    src = open("scratch/s258_mkq.py", encoding="utf-8").read(); i = src.index("def nobuf_bytes(p):"); j = src.index("\nEVAL = "); g = {}
    exec("from nobuf_sha import k32, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN\nimport ctypes, os\n"
         "import ctypes.wintypes as W\n" + src[i:j], g)
    b = g["nobuf_bytes"](p); return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


head = {p: subprocess.check_output(["git", "rev-parse", "HEAD:" + p]).decode().strip() for p in BLOB}; pre = {p: {"head": head[p], "disk": nobuf_blob(p)} for p in BLOB}
inits = {k: [sha_nobuf(os.path.abspath(v[0])) for _ in range(3)] for k, v in INIT.items()}
ok = all(v["head"] == v["disk"] for v in pre.values()) and all(sum(s_.startswith(INIT[k][1]) for s_ in inits[k]) >= 2 for k in INIT)
out = {"blobs": pre, "init_unbuffered_sha256_x3": inits, "predeploy_ok": ok, "args": {}}; assert ok, ("pre-deploy", pre, inits)
blob = " ".join("%s=%s" % (p, head[p]) for p in BLOB)
for n, run, k, bun in ((402, "bun_s1", 1, True), (403, "bun_s2", 2, True), (404, "bunc_s1", 1, False), (405, "bunc_s2", 2, False)):
    fl = G7.dch_flags(k, *INIT[k]) + (["--hold_inc", "1.6", "--hold_log"] if bun else ["--hold_log"])
    tail = ["--keep_final_ckpt", "--ckpt_every", "1000", "--eval_parts", EV_BUN if bun else EV_BUNC]
    d = G8.fresh(ROOT + "/scratch/_tmp/s262/args_" + run)
    rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + tail + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log"); assert rc == 0, (run, rc)
    shutil.copy(d + "/args.json", "scratch/s262_args_frozen_%s.json" % run)
    dif = {kk: v for kk, v in C.diff(json.load(open("boxes/runs_fsq/dch_s%d/args.json" % k)), json.load(open(d + "/args.json"))).items() if kk != "pair_stats"}
    out["args"][run] = {"keys": sorted(dif), "diff": dif}; fs = " ".join(fl); W = "boxes/runs_fsq/" + run
    q = ("# THEORY §259 BUNDLE (Note 259-2; score_bu): %s = dch_s%d (unbuffered copy, sha on host) + dch's flags %s, 8k, micro 2 pinned, --keep_final_ckpt. "
         "Frozen args scratch/s262_args_frozen_%s.json.\n# HOSTPROBE\n# P=1\n"
         "%s_mem boxes.inject_aw %s --mem_check\n"
         "%s_gtime boxes.inject_aw %s --time_steps 60\n"
         "%s scratch.s257_guarded --mode train --blob %s --init %s=%s --ship %s -- %s %s --args_ref scratch/s262_args_frozen_%s.json\n") % (
        run, k, "+ --hold_inc 1.6 --hold_log" if bun else "+ --hold_log (no penalty)", run, W, fs, W, fs, W, blob, INIT[k][0], INIT[k][1], INIT[k][0], fs, " ".join(tail), run)
    if True:                                                                # Audit G-D change 1: the long line on BOTH arms
        q += "%s scratch.s262_long --kprime_list 8,64,128 --long_ds 1,2,3,4,5,6 --long_name eval_long_k128.json\n" % W
    open("scratch/gen_q_b%d.txt.hold" % n, "w", encoding="utf-8", newline="\n").write(q)
    assert subprocess.call([sys.executable, "scratch/check_queue_ships.py", "scratch/gen_q_b%d.txt.hold" % n]) == 0
os.makedirs("scratch/gate_s262", exist_ok=True); json.dump(out, open("scratch/gate_s262/predeploy.json", "w", newline="\n"), indent=1, default=str)
print({r: v["keys"] for r, v in out["args"].items()})
