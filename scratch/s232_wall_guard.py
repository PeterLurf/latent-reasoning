"""§232 Stage-A wall-clock cost guard for an EVAL-ONLY host (no log.jsonl steps for scratch/cost_guard.py to read). Every 10 min:
  spent = (now - t0) / 3600 x $/h;  progress = 'evaluated' lines in the host's queue log / queue lines;  projection = spent / max(progress, 1 / lines)
  (after >= 1 line done; before that, spent + the registered estimate).
  - projection > CAP: an ALERT line (no destroy; Audit decides).
  - spent >= CAP: STOP (GUARD_STOP=1: vdestroy the instance; the queue's finished dirs are lost unless the drain already ran).
  Exits when the instance is gone (drained / destroyed) or STOP. Read-only ssh otherwise.
    GUARD_STOP=1 nohup python scratch/s232_wall_guard.py <tag> <cap $> <instance> <host> <port> <$/h> <t0 epoch> <queue tag gen_b<n>> <n lines> <est $> > logs/<tag>.out 2>&1 &"""
import os
import subprocess
import sys
import time

TAG, CAP, IID, H, P, DPH, T0, QT, NL, EST = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]), float(sys.argv[7]), sys.argv[8], int(sys.argv[9]), float(sys.argv[10])
STOP = os.environ.get("GUARD_STOP") == "1"; KEY = os.path.expanduser("~/.ssh/runpod_ed25519"); os.chdir("D:/Latent")


def log(s):
    print("%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), s), flush=True)


def present():
    try:
        import json
        r = subprocess.run(["bash", "C:/Users/peter/.local/vastai/vast.sh", "show", "instances", "--raw"], capture_output=True, text=True, timeout=90)
        return int(IID) in {int(x["id"]) for x in json.loads(r.stdout)}
    except Exception:
        return None                                                         # unknown (CLI failure): keep guarding


def done_lines():
    try:                                                                    # subprocess timeout (a bare "timeout" may resolve to Windows timeout.exe)
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-i", KEY, "-p", P, "root@" + H, "grep -c ' evaluated ' /workspace/Latent/logs/queue_%s.out" % QT],
                           capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None
    try:
        return int(r.stdout.strip())
    except ValueError:
        return None


log("start: cap $%.2f, instance %s (%s:%s) $%.4f/h since %s, %d queue lines, estimate $%.2f, destroy at cap: %s" % (CAP, IID, H, P, DPH, time.strftime("%H:%M", time.localtime(T0)), NL, EST, "on" if STOP else "off"))   # no "ALERT" / "STOP" words: the filler greps them
alerted = False
while True:
    pr = present()
    if pr is False:
        log("guard done: instance %s gone (drained / destroyed); spent ~$%.3f" % (IID, (time.time() - T0) / 3600 * DPH)); break
    spent = (time.time() - T0) / 3600 * DPH; d = done_lines()
    proj = (spent / (d / NL)) if d else spent + EST
    log("CHECK spent $%.3f, lines %s / %d, projection $%.3f (cap $%.2f)" % (spent, d, NL, proj, CAP))
    if spent >= CAP:
        if STOP:
            r = subprocess.run(["bash", "scratch/vdestroy.sh", IID], capture_output=True, text=True); log("STOP spent $%.3f >= cap $%.2f: %s" % (spent, CAP, r.stdout.strip()[-200:]))
        else:
            log("ALERT spent $%.3f >= cap $%.2f (destroy at cap: off)" % (spent, CAP))
        break
    if proj > CAP and not alerted:
        log("ALERT projection $%.3f > cap $%.2f (no destroy; tell Audit)" % (proj, CAP)); alerted = True
    time.sleep(600)
