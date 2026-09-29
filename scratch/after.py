"""Wait for a process to exit, then run a command (a minimal job queue).

    python scratch/after.py <pid> <logname> -- <python module args...>
"""
import os
import subprocess
import sys
import time

pid, name = int(sys.argv[1]), sys.argv[2]
cmd = sys.argv[sys.argv.index("--") + 1:]


def alive(p):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {p}", "/NH"], capture_output=True, text=True).stdout
    return str(p) in out


while alive(pid):
    time.sleep(30)
os.makedirs("logs", exist_ok=True)
with open(f"logs/{name}.out", "w") as o, open(f"logs/{name}.err", "w") as e:
    print(time.strftime("%H:%M:%S"), "start", cmd, file=o, flush=True)
    r = subprocess.run([sys.executable] + cmd, stdout=o, stderr=e)
    print(time.strftime("%H:%M:%S"), "exit", r.returncode, file=o, flush=True)
