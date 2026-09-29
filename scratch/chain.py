"""Wait for a process to exit, then run several python commands in sequence (a minimal job chain).

    python scratch/chain.py <pid> <logname> -- <cmd1 args...> :: <cmd2 args...> :: ...

Each command is run with the current interpreter (so "-m pkg.mod ..." or "scratch/x.py ..."). A failing
command is logged and the chain continues with the next one.
"""
import os
import subprocess
import sys
import time

pid, name = int(sys.argv[1]), sys.argv[2]
rest = sys.argv[sys.argv.index("--") + 1:]
cmds, cur = [], []
for tok in rest:
    if tok == "::":
        cmds.append(cur); cur = []
    else:
        cur.append(tok)
if cur:
    cmds.append(cur)


def alive(p):
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {p}", "/NH"], capture_output=True, text=True).stdout
        return str(p) in out
    try:
        os.kill(p, 0)
        return True
    except ProcessLookupError:
        return False


while pid and alive(pid):
    time.sleep(30)
os.makedirs("logs", exist_ok=True)
with open(f"logs/{name}.out", "a") as o, open(f"logs/{name}.err", "a") as e:
    for c in cmds:
        print(time.strftime("%H:%M:%S"), "start", c, file=o, flush=True)
        r = subprocess.run([sys.executable] + c, stdout=o, stderr=e)
        print(time.strftime("%H:%M:%S"), "exit", r.returncode, file=o, flush=True)
