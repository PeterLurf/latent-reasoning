"""§219 T1: the args.json the queued cell line writes, produced on CPU for Theory to freeze by sha (TS_v1 §2.5 recipe; the §218 218-9 practice).
boxes/textself_ft.py has no --args_only, so the trainer's own main() is run in-process on the EXACT tokens of the queue line (as gen_vast_job.sh runs
them: `python -u -m <mod> <rest> --device <dev> --out <out>`, here --device cpu) from a scratch tree that mirrors boxes/runs_fsq (hard links to
s170_cot_s1/{model.pt,args.json} and s219_pool_t1_s1/{pool_t1.jsonl,pool_meta.json}), so every relative path in args.json is the registered one and nothing
is written under boxes/. boxes.fsq120.make_opt is replaced by a sentinel that raises: main() has by then written args.json (+ args_diff.json vs --args_ref,
init_from.json) and nothing else (no log.jsonl, no model, no checkpoint, no optimiser step). The tree is removed afterwards (links only; the repo files
are untouched). CPU, 2 threads. Differences from the host's args.json are META_T only (device, train_gpu, train_driver, env keys, hostprobe_sps, micro).
    python -B scratch/s219_args_draft.py [--queue scratch/gen_q_b318.txt.hold] [--line 3] [--out_file scratch/s219_args_frozen_t1.json.draft]"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = "D:/Latent"
LINKS = ("boxes/runs_fsq/s170_cot_s1/model.pt", "boxes/runs_fsq/s170_cot_s1/args.json", "boxes/runs_fsq/s219_pool_t1_s1/pool_t1.jsonl",
         "boxes/runs_fsq/s219_pool_t1_s1/pool_meta.json")
BOOT = r"""
import json, sys
sys.path.insert(0, "D:/Latent")
import torch
torch.set_num_threads(2)
import boxes.fsq120 as F
class _ArgsWritten(Exception):
    pass
def _sentinel(*a, **k):
    raise _ArgsWritten()
F.make_opt = _sentinel
import boxes.textself_ft as T
argv = json.loads(sys.argv[1])
try:
    T.main(argv)
    print("UNEXPECTED: main returned without reaching make_opt"); sys.exit(4)
except _ArgsWritten:
    print("ARGS_WRITTEN (stopped at make_opt; no optimiser, no training)")
"""


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--queue", default="scratch/gen_q_b318.txt.hold"); ap.add_argument("--line", type=int, default=3)
    ap.add_argument("--out_file", default="scratch/s219_args_frozen_t1.json.draft"); ap.add_argument("--tmp", default="scratch/_tmp/s219_argsdraft")
    a = ap.parse_args(); os.chdir(ROOT)
    cmds = [l.split() for l in open(a.queue, encoding="utf-8").read().split("\n") if l.strip() and not l.startswith("#")]
    out, mod, rest = cmds[a.line - 1][0], cmds[a.line - 1][1], cmds[a.line - 1][2:]
    assert mod == "boxes.textself_ft", mod
    argv = rest + ["--device", "cpu", "--out", out]                        # gen_vast_job.sh: <rest> --device <dev> --out <out>
    tree = os.path.abspath(a.tmp)
    if os.path.exists(tree):
        shutil.rmtree(tree)
    for f in LINKS:
        os.makedirs(os.path.dirname(os.path.join(tree, f)), exist_ok=True)
        try:
            os.link(f, os.path.join(tree, f))
        except OSError:
            shutil.copy2(f, os.path.join(tree, f))
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    env.pop("HOSTPROBE_SPS", None)
    print("ARGV", " ".join(argv), flush=True)
    r = subprocess.run([sys.executable, "-B", "-c", BOOT, json.dumps(argv)], cwd=tree, env=env, capture_output=True, text=True)
    print(r.stdout[-3000:]); print(r.stderr[-3000:], file=sys.stderr)
    od = os.path.join(tree, out); files = sorted(os.listdir(od)) if os.path.isdir(od) else []
    print("FILES_WRITTEN", files)
    ok = r.returncode == 0 and "ARGS_WRITTEN" in r.stdout and set(files) <= {"args.json", "args_diff.json", "init_from.json"} and "args.json" in files
    if ok:
        shutil.copyfile(os.path.join(od, "args.json"), a.out_file)
        print("INIT_FROM", open(os.path.join(od, "init_from.json")).read().replace("\n", " "))
        print("TRAINER_ARGS_DIFF_KEYS", sorted(json.load(open(os.path.join(od, "args_diff.json")))["diff"]))
        print("DRAFT", a.out_file, "sha256", sha(a.out_file), "bytes", os.path.getsize(a.out_file))
    shutil.rmtree(tree)                                                     # hard links / copies only
    print("TREE_REMOVED", tree, "rc", r.returncode, "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
