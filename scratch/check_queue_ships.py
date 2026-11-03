"""Audit 11-02 (b395 lost dis_s2: its '--model <path>=<sha>' init never shipped; b402: the --blob'd sha jsons never shipped - auto_ship takes plain path tokens
only): (1) the path of every '--model' / '--init' '<path>=<hex>' value, and (2) every '--blob' path that is NOT a .py module (data files: .json / .tar / ...; .py modules
ship as queue-module dependencies), must also appear somewhere in the queue as a plain token. Exit 1 + the missing paths, else 0.
    python scratch/check_queue_ships.py scratch/gen_q_b<n>.txt"""
import re
import sys

lines = [ln.split() for ln in open(sys.argv[1], encoding="utf-8") if ln.strip() and not ln.startswith("#")]
plain = {t for ln in lines for t in ln}
need = set()
for ln in lines:
    flag = None
    for t in ln:
        if t.startswith("--"):
            flag = t; continue
        if re.match(r"^[^=]+=[0-9a-f]{8,}$", t) and (flag in ("--model", "--init") or (flag == "--blob" and not t.split("=")[0].endswith(".py"))):
            need.add(t.split("=")[0])
sys.path.insert(0, "scratch")                                               # Audit 11-03 (b412 attempt 5: scratch/s259_guarded.py was --blob'd by the gates line but no line
import check_py311 as P3  # noqa: E402                                      # ran or imported it after the step-0 line was dropped -> FileNotFoundError on the host)
shipped = set(P3.files_of(sys.argv[1]))
for ln in lines:
    flag = None
    for t in ln:
        if t.startswith("--"):
            flag = t; continue
        if flag == "--blob" and re.match(r"^scratch/[^=]+\.py=[0-9a-f]{8,}$", t) and t.split("=")[0] not in shipped and t.split("=")[0] not in plain:
            need.add(t.split("=")[0])
miss = sorted(need - plain)
if miss:
    print("NOT SHIPPED (path=sha only):", " ".join(miss)); sys.exit(1)
print("ships OK (%d model / init / data paths)" % len(need))
import subprocess  # noqa: E402   Audit 11-03 (b412 attempt 3): + the host-Python-3.11 check of every .py the queue runs (scratch/check_py311.py)
r_ = subprocess.run([sys.executable, "scratch/check_py311.py", sys.argv[1]], capture_output=True, text=True); print(r_.stdout.strip())
sys.exit(r_.returncode)
