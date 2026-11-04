"""Finish s271_mkq_pilot after its run was killed during b421's checks (11-04): ships + dry-run on all 4 queues (headers gained
HARD_WALL_H / IDLE_H) and rebuild scratch/gate_s271/pilot_args.json from the frozen args the run already wrote (same diff code)."""
import json
import os
import subprocess
import sys

ROOT = "D:/Latent"; os.chdir(ROOT); sys.path.insert(0, ROOT + "/scratch")
src = open("scratch/s271_mkq_pilot.py", encoding="utf-8").read()
CELLS = eval(src[src.index("CELLS = ") + 8:].split("\n")[0])
import s228_args_check as C   # noqa: E402  (the generator's diff module; asserted below)
assert "import s228_args_check as C" in src
out = {"args": {}}
for n, run, k, nmax in CELLS:
    dif = {kk: v for kk, v in C.diff(json.load(open("boxes/runs_fsq/bun_s%d/args.json" % k)), json.load(open("scratch/s271_args_frozen_%s.json" % run))).items() if kk != "pair_stats"}
    out["args"][run] = {"keys": sorted(dif), "diff": dif}
    p = "scratch/gen_q_b%d.txt.hold" % n
    assert subprocess.call([sys.executable, "scratch/check_queue_ships.py", p]) == 0, p
    assert subprocess.call([sys.executable, "-B", "scratch/queue_dryrun.py", p]) == 0, p
json.dump(out, open("scratch/gate_s271/pilot_args.json", "w", newline="\n"), indent=1, default=str); print(json.dumps({r: v["keys"] for r, v in out["args"].items()}))
