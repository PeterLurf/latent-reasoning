"""§270 FULL-PERM queues (FP_v1 §7; build serial s272, filler family s272; Note 270-1: cell cap $1.8, guard $4, RTX 3090 as of_sX, eval_gpu recorded):
b423 fp_s1 / b424 fp_s2 = of_s<k>'s recipe (s270_mkq.flags: bun_s<k>_init + bun's args + the §262 CTRL recipe) + --perm_scope all, 8k, micro 2 pinned; in-process evals
acc / test_perm / insert_ho / swap_kd / lp5h / stab, then late_premise / name_swap / shuffle_graded / full_perm, then run_long K' 8/64/128 d 1-6; then an eval-only
full_perm on of_s<k> (the paired same-item baseline, FP_v1 §3) into boxes/runs_fsq/of_s<k>_fp. Frozen args scratch/s272_args_frozen_fp_s<k>.json (+ diff vs of_s<k>).
Pre-deploy: unbuffered blob ids == HEAD; inits' streaming shas (majority of 3); ships checked; queue dry-run.
    python -B scratch/s272_mkq.py"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); sys.path.insert(0, ROOT + "/scratch/ram_scan"); os.chdir(ROOT)
from nobuf_sha import sha_nobuf  # noqa: E402
import gate_s228 as G8  # noqa: E402
import gate_s257 as G7  # noqa: E402
import s228_args_check as C  # noqa: E402

INIT = {1: ("boxes/runs_fsq/bun_s1_init/model.pt", "1ad3517b7635fddf"), 2: ("boxes/runs_fsq/bun_s2_init/model.pt", "09337f63aec57959")}
BLOB = ["boxes/inject_aw.py", "boxes/inject_aw_eval.py", "boxes/fp_parts.py", "boxes/of_parts.py", "boxes/sc_parts.py", "boxes/ask_render.py", "boxes/insert_render.py",
        "boxes/perm_render.py", "boxes/s5.py", "boxes/nested170.py", "boxes/inject_ow.py", "boxes/inject_st.py", "boxes/deltamem.py", "boxes/deltas5.py", "boxes/core.py",
        "boxes/ow_words.py", "boxes/nested_words.py", "scratch/s257_guarded.py", "scratch/s271_guarded.py", "scratch/s262_long.py", "scratch/s256_ho_shas.json",
        "scratch/s257_ho_shas.json"]
EVAL = "acc,test_perm,insert_ho,swap_kd,lp5h,stab"; NEWP = "late_premise,name_swap,shuffle_graded,full_perm"
CELLS = ((423, "fp_s1", 1), (424, "fp_s2", 2))


def nobuf_blob(p):
    src = open("scratch/s258_mkq.py", encoding="utf-8").read(); i = src.index("def nobuf_bytes(p):"); j = src.index("\nEVAL = "); g = {}
    exec("from nobuf_sha import k32, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN\nimport ctypes, os\n"
         "import ctypes.wintypes as W\n" + src[i:j], g)
    b = g["nobuf_bytes"](p); return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def flags(k):
    return G7.dch_flags(k, *INIT[k]) + ["--hold_inc", "1.6", "--hold_log", "--perm_p", "0.5", "--ask_p", "0.25", "--perm_scope", "all"]


head = {p: subprocess.check_output(["git", "rev-parse", "HEAD:" + p]).decode().strip() for p in BLOB}; pre = {p: {"head": head[p], "disk": nobuf_blob(p)} for p in BLOB}
OF = {k: ("boxes/runs_fsq/of_s%d/model.pt" % k) for k in INIT}
inits = {k: [sha_nobuf(os.path.abspath(v[0])) for _ in range(3)] for k, v in INIT.items()}
ofsha = {k: [sha_nobuf(os.path.abspath(OF[k])) for _ in range(3)] for k in OF}; OFS = {k: max(set(v), key=v.count)[:16] for k, v in ofsha.items()}
ok = (all(v["head"] == v["disk"] for v in pre.values()) and all(sum(s_.startswith(INIT[k][1]) for s_ in inits[k]) >= 2 for k in INIT)
      and all(ofsha[k].count(max(set(ofsha[k]), key=ofsha[k].count)) >= 2 for k in OF))
out = {"blobs": pre, "init_unbuffered_sha256_x3": inits, "of_unbuffered_sha256_x3": ofsha, "predeploy_ok": ok, "args": {}}
assert ok, ("pre-deploy", out)
sys.path.insert(0, ROOT + "/scratch")
import caps_check as CC  # noqa: E402
blob = " ".join("%s=%s" % (p, head[p]) for p in BLOB); SH = "scratch/s256_ho_shas.json scratch/s257_ho_shas.json"
for n, run, k in CELLS:
    fl = flags(k); tail = ["--ckpt_every", "1000", "--eval_parts", EVAL]; d = G8.fresh(ROOT + "/scratch/_tmp/s272/args_" + run)
    rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + tail + ["--device", "cpu", "--args_only", "--args_ref", "boxes/runs_fsq/of_s%d/args.json" % k,
               "--out", d], d + "/stdout.log"); assert rc == 0, (run, rc)
    shutil.copy(d + "/args.json", "scratch/s272_args_frozen_%s.json" % run)
    dif = {kk: v for kk, v in C.diff(json.load(open("boxes/runs_fsq/of_s%d/args.json" % k)), json.load(open(d + "/args.json"))).items() if kk != "pair_stats"}
    late = {kk for kk, v in dif.items() if kk == "distract_nmax" and list(v) == [None, 2]}   # §265's flag, absent in of_s<k>'s args; 2 = its default (the old rng.integers(0, 3) law)
    assert set(dif) - {"out", "args_ref"} - late == {"perm_scope"}, ("Audit MED-2: the args diff vs of_s%d must be perm_scope only" % k, sorted(dif))
    out["args"][run] = {"keys": sorted(dif), "diff": dif}; fs = " ".join(fl); W = "boxes/runs_fsq/" + run; WO = "boxes/runs_fsq/of_s%d_fp" % k
    body = ("%s_mem boxes.inject_aw %s --mem_check\n%s_gtime boxes.inject_aw %s --time_steps 60\n"
            "%s scratch.s257_guarded --mode train --blob %s --init %s=%s --ship %s %s -- %s %s --args_ref scratch/s272_args_frozen_%s.json\n"
            "%s boxes.inject_aw_eval --run %s --parts %s --eval_bs 250\n"
            "%s scratch.s262_long --kprime_list 8,64,128 --long_ds 1,2,3,4,5,6 --long_name eval_long_k128.json\n"
            "%s scratch.s271_guarded --mode evalonly --blob %s --model %s=%s --args boxes/runs_fsq/of_s%d/args.json --parts full_perm --ship %s "
            "boxes/runs_fsq/of_s%d/args.json %s\n") % (W, fs, W, fs, W, blob, INIT[k][0], INIT[k][1], INIT[k][0], SH, fs, " ".join(tail), run, W, W, NEWP, W,
                                                      WO, blob, OF[k], OFS[k], k, OF[k], k, SH)
    nof = "# NOFETCH %s/model.pt\n" % WO
    caps = CC.caps_line([0.1, 0.1, 3.9, 0.6, 0.4, 0.3], 1.0, 0.0, 1.7, 0.854, "RTX_3090", CC.fetch_gb(nof + body))
    q = ("# THEORY §270 FULL-PERM (score_fp): %s = of_s%d's recipe (bun_s%d_init streaming copy, sha on host; bun's args + perm-then-insert --perm_p 0.5 + one-hop "
         "':'-CE --ask_p 0.25) + --perm_scope all, 8k, micro 2 pinned; in-process evals, then late_premise / name_swap / shuffle_graded / full_perm, run_long "
         "K' 8/64/128 d 1-6; then eval-only full_perm on of_s%d -> %s (the paired baseline). Frozen args scratch/s272_args_frozen_%s.json.\n# HOSTPROBE\n# P=1\n%s\n%s%s") % (
        run, k, k, k, WO, run, caps, nof, body)
    p = "scratch/gen_q_b%d.txt.hold" % n; open(p, "w", encoding="utf-8", newline="\n").write(q)
    assert subprocess.call([sys.executable, "scratch/check_queue_ships.py", p]) == 0
    assert subprocess.call([sys.executable, "-B", "scratch/queue_dryrun.py", p]) == 0
os.makedirs("scratch/gate_s272", exist_ok=True); json.dump(out, open("scratch/gate_s272/predeploy.json", "w", newline="\n"), indent=1, default=str)
print({r: v["keys"] for r, v in out["args"].items()})
