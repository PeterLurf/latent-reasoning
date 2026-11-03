"""v6 (Audit 11-03 LAUNCH_POSTMORTEM §3.1): + the DRAINING mode (below). §232 Stage-A wall-clock cost guard for an EVAL-ONLY host (no log.jsonl steps for scratch/cost_guard.py to read). Every 10 min:
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
VDESTROY = os.environ.get("GUARD_VDESTROY", "scratch/vdestroy.sh"); VASTCLI = os.environ.get("GUARD_VASTCLI", "C:/Users/peter/.local/vastai/vast.sh")   # stubs: the retry smoke only
RETRY_S = int(os.environ.get("GUARD_RETRY_S", "60"))
SSH = os.environ.get("GUARD_SSH", "ssh").split(); POLL_S = int(os.environ.get("GUARD_POLL_S", "600")); CEIL = float(os.environ.get("DRAIN_CEIL", "2.0"))   # v6 (stubs: the fixture)
DRAINLOG = os.environ.get("GUARD_DRAINLOG", "logs/auto_drain_%s.out" % sys.argv[8])                         # v6: the gen10 drain watcher's log for this queue tag
W = [float(x) for x in os.environ.get("GUARD_WEIGHTS", "").split(",") if x.strip()] or [1.0] * NL   # 10-24 (§232 s3 chain): per-line weights (est. hours)
assert len(W) == NL, ("GUARD_WEIGHTS needs one weight per queue line", len(W), NL)


def log(s):
    print("%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), s), flush=True)


def present():
    try:
        import json
        r = subprocess.run(["bash", VASTCLI, "show", "instances", "--raw"], capture_output=True, text=True, timeout=90)
        return int(IID) in {int(x["id"]) for x in json.loads(r.stdout)}
    except Exception:
        return None                                                         # unknown (CLI failure): keep guarding


def done_lines():
    try:                                                                    # subprocess timeout (a bare "timeout" may resolve to Windows timeout.exe)
        r = subprocess.run(SSH + ["-o", "ConnectTimeout=15", "-i", KEY, "-p", P, "root@" + H, "grep -c ' evaluated ' /workspace/Latent/logs/queue_%s.out" % QT],
                           capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None
    try:
        return int(r.stdout.strip())
    except ValueError:
        return None


def queue_done():
    """v6: the host queue log has QUEUE_DONE (None: unknown)."""
    try:
        r = subprocess.run(SSH + ["-o", "ConnectTimeout=15", "-i", KEY, "-p", P, "root@" + H, "grep -c QUEUE_DONE /workspace/Latent/logs/queue_%s.out" % QT],
                           capture_output=True, text=True, timeout=60)
        return int(r.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, ValueError):
        return None


def drain_state():
    """v6: the drain log's verdict since this guard started: 'DRAINED' / 'FAILED' / None."""
    try:
        s = open(DRAINLOG, "rb").read()[START_SIZE:].decode("utf-8", "replace")   # bytes: START_SIZE is a byte offset
    except OSError:
        return None
    return "FAILED" if "DRAIN FAILED" in s else ("DRAINED" if "ALL_DRAINED" in s else None)


try:
    START_SIZE = os.path.getsize(DRAINLOG)                                 # only verdicts written after this guard started count
except OSError:
    START_SIZE = 0
log("start: cap $%.2f, instance %s (%s:%s) $%.4f/h since %s, %d queue lines, estimate $%.2f, destroy at cap: %s" % (CAP, IID, H, P, DPH, time.strftime("%H:%M", time.localtime(T0)), NL, EST, "on" if STOP else "off"))   # no "ALERT" / "STOP" words: the filler greps them
alerted = False
while True:
    pr = present()
    if pr is False:
        log("guard done: instance %s gone (drained / destroyed); spent ~$%.3f" % (IID, (time.time() - T0) / 3600 * DPH)); break
    spent = (time.time() - T0) / 3600 * DPH; d = done_lines()
    proj = (spent / (sum(W[:d]) / sum(W))) if d else spent + EST           # default (no GUARD_WEIGHTS): uniform weights = spent / (d / NL), unchanged
    log("CHECK spent $%.3f, lines %s / %d, projection $%.3f (cap $%.2f)" % (spent, d, NL, proj, CAP))
    qd = queue_done() if spent >= CAP else None; ds = drain_state()
    if spent >= CAP and spent < CAP * CEIL and qd and ds != "FAILED":     # v6 DRAINING: the queue is done, the drain is fetching - no destroy mid-fetch
        log("ALERT DRAINING spent $%.3f >= cap $%.2f after QUEUE_DONE (drain %s; no destroy until DRAINED / DRAIN FAILED, ceiling $%.2f)" % (spent, CAP, ds or "running", CAP * CEIL))
        time.sleep(POLL_S); continue
    if spent >= CAP:
        if STOP:                                                            # Audit 10-24: retry until DESTROYED_VERIFIED or the instance is gone (never exit on STILL_LIVE / UNKNOWN)
            while True:
                r = subprocess.run(["bash", VDESTROY, IID], capture_output=True, text=True); ok = r.returncode == 0 and "DESTROYED_VERIFIED" in r.stdout
                log("STOP spent $%.3f >= cap $%.2f: vdestroy rc %d %s" % (spent, CAP, r.returncode, r.stdout.strip()[-200:]))
                if ok or present() is False:
                    break
                log("STOP retry in %d s (vdestroy not verified)" % RETRY_S); time.sleep(RETRY_S)
        else:
            log("ALERT spent $%.3f >= cap $%.2f (destroy at cap: off)" % (spent, CAP))
        break
    if proj > CAP and not alerted:
        log("ALERT projection $%.3f > cap $%.2f (no destroy; tell Audit)" % (proj, CAP)); alerted = True
    time.sleep(POLL_S)
