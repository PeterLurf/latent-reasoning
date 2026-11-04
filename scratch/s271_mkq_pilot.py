"""§265 DENSE-DISTRACT pilot (THEORY §265 / Note 265-1; score_dd): queues b418 ddp_s1 / b419 ddp_s2 (--distract_nmax 8) and b420 ddc_s1 / b421 ddc_s2 (CTRL, nmax default 2),
one RTX 3090 host each (filler family s271), warm from bun_s<k> (bun_s<k>_init = the streaming copy of bun_s<k>/model.pt, sha on host) under bun's args; the 8k cosine
stopped at 2000 (the §262 pilot pattern: TRAIN --stop_at 2000, then scratch.s268_stopland evals acc / ws_deep / attn_fade). TRAIN --require's b417's landed
boxes/runs_fsq/dd_gates/gates_pass.json (offbit_aw / smoke_dd / gradflow on the same trainer blob; shipped). mem_check line at d 4 + 8 inserts for the nmax-8 cells (Note 265-1).
Frozen args scratch/s271_args_frozen_<run>.json (--args_only, local CPU) + their diff vs bun_s<k>/args.json -> scratch/gate_s271/pilot_args.json.
Pre-deploy: unbuffered blob ids == HEAD; the inits' streaming shas (majority of 3); gates_pass.json present; ships checked; queue dry-run.
    python -B scratch/s271_mkq_pilot.py"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); sys.path.insert(0, ROOT + "/scratch/ram_scan"); os.chdir(ROOT)
from nobuf_sha import sha_nobuf  # noqa: E402
import caps_check as CC  # noqa: E402
import gate_s228 as G8  # noqa: E402
import gate_s257 as G7  # noqa: E402
import s228_args_check as C  # noqa: E402

INIT = {1: ("boxes/runs_fsq/bun_s1_init/model.pt", "1ad3517b7635fddf"), 2: ("boxes/runs_fsq/bun_s2_init/model.pt", "09337f63aec57959")}
GP = "boxes/runs_fsq/dd_gates/gates_pass.json"
BLOB = ["boxes/inject_aw.py", "boxes/inject_aw_eval.py", "boxes/sc_parts.py", "boxes/insert_render.py", "boxes/perm_render.py", "boxes/s5.py", "boxes/nested170.py",
        "boxes/inject_ow.py", "boxes/inject_st.py", "boxes/deltamem.py", "boxes/deltas5.py", "boxes/core.py", "boxes/ow_words.py", "boxes/nested_words.py",
        "scratch/s257_guarded.py", "scratch/s271_guarded.py", "scratch/s268_stopland.py", "scratch/s256_ho_shas.json", "scratch/s257_ho_shas.json"]
PARTS = "acc,ws_deep,attn_fade"
CELLS = ((418, "ddp_s1", 1, 8), (419, "ddp_s2", 2, 8), (420, "ddc_s1", 1, None), (421, "ddc_s2", 2, None))


def nobuf_blob(p):
    src = open("scratch/s258_mkq.py", encoding="utf-8").read(); i = src.index("def nobuf_bytes(p):"); j = src.index("\nEVAL = "); g = {}
    exec("from nobuf_sha import k32, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN\nimport ctypes, os\n"
         "import ctypes.wintypes as W\n" + src[i:j], g)
    b = g["nobuf_bytes"](p); return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def flags(k, nmax):
    return G7.dch_flags(k, *INIT[k]) + ["--hold_inc", "1.6", "--hold_log"] + (["--distract_nmax", str(nmax)] if nmax else [])


head = {p: subprocess.check_output(["git", "rev-parse", "HEAD:" + p]).decode().strip() for p in BLOB}; pre = {p: {"head": head[p], "disk": nobuf_blob(p)} for p in BLOB}
inits = {k: [sha_nobuf(os.path.abspath(v[0])) for _ in range(3)] for k, v in INIT.items()}
ok = (all(v["head"] == v["disk"] for v in pre.values()) and all(sum(s_.startswith(INIT[k][1]) for s_ in inits[k]) >= 2 for k in INIT) and os.path.exists(GP)
      and json.load(open(GP)).get("pass", True) is not False)
out = {"blobs": pre, "init_unbuffered_sha256_x3": inits, "gates_pass": GP, "predeploy_ok": ok, "args": {}}
assert ok, ("pre-deploy", out)
blob = " ".join("%s=%s" % (p, head[p]) for p in BLOB); SH = "scratch/s256_ho_shas.json scratch/s257_ho_shas.json"
for n, run, k, nmax in CELLS:
    fl = flags(k, nmax); tail = ["--ckpt_every", "1000", "--stop_at", "2000", "--eval_parts", "acc"]; d = G8.fresh(ROOT + "/scratch/_tmp/s271p/args_" + run)
    rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + tail + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log"); assert rc == 0, (run, rc)
    shutil.copy(d + "/args.json", "scratch/s271_args_frozen_%s.json" % run)
    dif = {kk: v for kk, v in C.diff(json.load(open("boxes/runs_fsq/bun_s%d/args.json" % k)), json.load(open(d + "/args.json"))).items() if kk != "pair_stats"}
    out["args"][run] = {"keys": sorted(dif), "diff": dif}; fs = " ".join(fl); W = "boxes/runs_fsq/" + run
    memx = " --distract_nmax 8" if nmax else ""
    body = ("%s_mem boxes.inject_aw %s --mem_check\n%s_gtime boxes.inject_aw %s --time_steps 60\n"
            "%s scratch.s271_guarded --mode train --blob %s --init %s=%s --require %s --ship %s %s %s -- %s %s --args_ref scratch/s271_args_frozen_%s.json\n"
            "%s scratch.s268_stopland --step 2000 --parts %s\n") % (W, fs, W, fs, W, blob, INIT[k][0], INIT[k][1], GP, GP, INIT[k][0], SH, fs, " ".join(tail), run, W, PARTS)
    assert memx in body
    caps = CC.caps_line([0.1, 0.1, 1.0, 0.4], 0.6, 0.0, 1.9, 0.854, "RTX_3090", CC.fetch_gb(body))
    q = ("# THEORY §265 DENSE-DISTRACT pilot (score_dd): %s = bun_s%d (streaming copy, sha on host) + bun's args%s; 8k cosine stopped at 2000; TRAIN requires b417's "
         "landed %s; stopland evals %s. Frozen args scratch/s271_args_frozen_%s.json.\n# HOSTPROBE\n# P=1\n%s\n%s") % (
        run, k, " + --distract_nmax %d" % nmax if nmax else " (CTRL: nmax default 2)", GP, PARTS, run, caps, body)
    p = "scratch/gen_q_b%d.txt.hold" % n; open(p, "w", encoding="utf-8", newline="\n").write(q)
    assert subprocess.call([sys.executable, "scratch/check_queue_ships.py", p]) == 0
    assert subprocess.call([sys.executable, "-B", "scratch/queue_dryrun.py", p]) == 0
json.dump(out, open("scratch/gate_s271/pilot_args.json", "w", newline="\n"), indent=1, default=str); print(json.dumps({r: v["keys"] for r, v in out["args"].items()}))
