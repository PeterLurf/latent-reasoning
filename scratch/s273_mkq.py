"""§268 LINEAGE-SEED queue for ONE stage (build serial s273, filler family s273; Note 268-1: one cell per stage, the landed model.pt is the next stage's init via a
streaming copy with a sha check). Uses scratch/s273_flags_<stage>_s3.json (scratch/s273_build/stage_flags.py: the s1 args with seed 3 / the s3 init chain; args diff
verified locally) and the stage's historical tree tar scratch/s273_tree_<commit>.tar (git archive, shipped, sha asserted on the host by scratch.s273_hist).
Lines: <run>_mem (--mem_check), <run>_gtime (--time_steps 60), <run> (train; the trainer's own end-of-run eval, the s1 stage's eval_parts).
Frozen args scratch/s273_args_frozen_<run>.json = the local --args_only args.json.
    python -B scratch/s273_mkq.py <stage> <queue n> [--sps <s/step>] [--init <model.pt>=<sha16> ...]"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); sys.path.insert(0, ROOT + "/scratch/ram_scan"); os.chdir(ROOT)
from nobuf_sha import sha_nobuf  # noqa: E402
import caps_check as CC  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("n", type=int); ap.add_argument("--sps", type=float, required=True)
ap.add_argument("--init", nargs="*", default=[]); x = ap.parse_args()
F = json.load(open("scratch/s273_flags_%s_s3.json" % x.stage)); assert F["ok"], F["extra_diff_keys"]; run, c = F["run"], F["commit"]; W = "boxes/runs_fsq/" + run
tar = "scratch/s273_tree_%s.tar" % c
if not os.path.exists(tar):
    shutil.copy("scratch/_tmp/s273/%s.tar" % c, tar)
tsha = sha_nobuf(os.path.abspath(tar))[:16]
trainer = {"boxes.inject_ow": "boxes/inject_ow.py", "boxes.inject_aw": "boxes/inject_aw.py"}[F["module"]]
msha = json.load(open("boxes/runs_fsq/%s/args.json" % run.replace("_s3", "_s1")))["module_sha256"][:16]
shutil.copy("scratch/_tmp/s273/args_%s/args.json" % x.stage, "scratch/s273_args_frozen_%s.json" % run)
for i in x.init:                                                            # local inits: majority-of-3 streaming shas
    p, s = i.split("="); assert sum(sha_nobuf(os.path.abspath(p)).startswith(s) for _ in range(3)) >= 2, ("init sha", p)
fl = " ".join(F["argv"]); ship = " ".join([tar] + [i.split("=")[0] for i in x.init])
key = "askname" if x.stage == "askname" else "m:" + F["module"]
pre = "scratch.s273_hist --tar %s=%s --trainer %s=%s%s --runner %s --ship %s --" % (tar, tsha, trainer, msha, "".join(" --init " + i for i in x.init), key, ship)
body = ("%s_mem %s %s --mem_check\n%s_gtime %s %s --time_steps 60\n%s %s %s\n") % (W, pre, fl, W, pre, fl, W, pre, fl)
steps = int(F["argv"][F["argv"].index("--steps") + 1]); train_h = round(steps * x.sps / 3600 + 0.3, 2)
caps = CC.caps_line([0.1, 0.1, train_h], 0.3, 0.0, x.sps, 0.854, "RTX_3090", CC.fetch_gb(body))
proj_h = 0.4 + 0.1 + 0.1 + train_h + 0.3                                    # setup + mem + gtime + train (+0.3 margin) + eval
hw = round(1.5 * proj_h, 2); assert hw >= 1.3 * proj_h, (hw, proj_h)       # Audit 11-04 LOW: host-side hard wall only as a backstop (>= 1.3 x projected)
caps = caps + "\n# HARD_WALL_H %s\n# IDLE_H 1.5" % hw
q = ("# THEORY §268 LINEAGE-SEED stage %s (score_x7): %s = the s1 stage's trainer version (git %s, %s sha %s) + its args with seed 3 and the s3 init chain "
     "(args diff vs %s_s1/args.json: %s). Frozen args scratch/s273_args_frozen_%s.json. Ships: %s.\n# HOSTPROBE\n# P=1\n%s\n%s") % (
    x.stage, run, c, trainer, msha, run.replace("_s3", ""), sorted(F["diff"]), run, ship, caps, body)
# the trainer line ships the tar + inits (queue_dryrun / check_queue_ships read --ship on s257-style lines; s273_hist lines name them via --tar / --init)
p = "scratch/gen_q_b%d.txt.hold" % x.n; open(p, "w", encoding="utf-8", newline="\n").write(q); print(p); print(q)
