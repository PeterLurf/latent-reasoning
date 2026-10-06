"""Queue wrapper so gen_vast_job + the drain can run and fetch scratch/replay_basin.py (claude-13 / Audit, 2026-10-05):
    python -m scratch.replay_wrap <replay_basin args> --device D --out DIR  ->  DIR/replay.jsonl, then DIR/eval.json on rc 0.
replay_basin.py itself is unchanged (md5 0fd7ea391c52b98e69a8fb1429ca6329); it runs with the repo root on PYTHONPATH."""
import json
import os
import subprocess
import sys

args = sys.argv[1:]; i = args.index("--out"); out = args[i + 1]; os.makedirs(out, exist_ok=True); args[i + 1] = os.path.join(out, "replay.jsonl")
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rc = subprocess.call([sys.executable, "-u", os.path.join(root, "scratch", "replay_basin.py")] + args, env=dict(os.environ, PYTHONPATH=root), cwd=root)
if rc == 0:
    json.dump({"replay": "done", "rc": rc, "args": args}, open(os.path.join(out, "eval.json"), "w", newline="\n"), indent=1)
sys.exit(rc)
