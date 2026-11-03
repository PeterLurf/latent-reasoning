"""Template v6 item 1 (Audit LAUNCH_POSTMORTEM §3.1, DRAIN budget): scratch/wall_guard_v6.py = scratch/s232_wall_guard.py + a DRAINING mode: once the host's queue log
has QUEUE_DONE, spent >= CAP only ALERTs (no vdestroy) until the local drain log (logs/auto_drain_<queue tag>.out) says ALL_DRAINED / DRAIN FAILED; DRAIN FAILED with the
instance still present -> STOP; a hard ceiling CAP x DRAIN_CEIL (default 2.0) STOPs in any mode. Stubs for the fixture: GUARD_SSH (the ssh program), GUARD_POLL_S (the
600 s poll), GUARD_DRAINLOG (the drain log path). Each edit asserted once.
    python -B scratch/v6_build/mk_wall_guard_v6.py"""
import sys

sys.path.insert(0, "scratch/s268_build")
from edits_s268_fn import edit  # noqa: E402

p = "scratch/wall_guard_v6.py"
edit(p, [
    ('"""§232 Stage-A wall-clock cost guard', '"""v6 (Audit 11-03 LAUNCH_POSTMORTEM §3.1): + the DRAINING mode (below). §232 Stage-A wall-clock cost guard'),
    ('RETRY_S = int(os.environ.get("GUARD_RETRY_S", "60"))\n',
     'RETRY_S = int(os.environ.get("GUARD_RETRY_S", "60"))\n'
     'SSH = os.environ.get("GUARD_SSH", "ssh"); POLL_S = int(os.environ.get("GUARD_POLL_S", "600")); CEIL = float(os.environ.get("DRAIN_CEIL", "2.0"))   # v6 (stubs: the fixture)\n'
     'DRAINLOG = os.environ.get("GUARD_DRAINLOG", "logs/auto_drain_%s.out" % sys.argv[8])                         # v6: the gen10 drain watcher\'s log for this queue tag\n'),
    ('        r = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-i", KEY, "-p", P, "root@" + H, "grep -c \' evaluated \' /workspace/Latent/logs/queue_%s.out" % QT],',
     '        r = subprocess.run([SSH, "-o", "ConnectTimeout=15", "-i", KEY, "-p", P, "root@" + H, "grep -c \' evaluated \' /workspace/Latent/logs/queue_%s.out" % QT],'),
    ('log("start: cap $%.2f',
     'def queue_done():\n'
     '    """v6: the host queue log has QUEUE_DONE (None: unknown)."""\n'
     '    try:\n'
     '        r = subprocess.run([SSH, "-o", "ConnectTimeout=15", "-i", KEY, "-p", P, "root@" + H, "grep -c QUEUE_DONE /workspace/Latent/logs/queue_%s.out" % QT],\n'
     '                           capture_output=True, text=True, timeout=60)\n'
     '        return int(r.stdout.strip()) > 0\n'
     '    except (subprocess.TimeoutExpired, ValueError):\n'
     '        return None\n\n\n'
     'def drain_state():\n'
     '    """v6: the drain log\'s verdict since this guard started: \'DRAINED\' / \'FAILED\' / None."""\n'
     '    try:\n'
     '        s = open(DRAINLOG, encoding="utf-8", errors="replace").read()[START_SIZE:]\n'
     '    except OSError:\n'
     '        return None\n'
     '    return "FAILED" if "DRAIN FAILED" in s else ("DRAINED" if "ALL_DRAINED" in s else None)\n\n\n'
     'try:\n'
     '    START_SIZE = os.path.getsize(DRAINLOG)                                 # only verdicts written after this guard started count\n'
     'except OSError:\n'
     '    START_SIZE = 0\n'
     'log("start: cap $%.2f'),
    ('    log("CHECK spent $%.3f, lines %s / %d, projection $%.3f (cap $%.2f)" % (spent, d, NL, proj, CAP))\n    if spent >= CAP:\n',
     '    log("CHECK spent $%.3f, lines %s / %d, projection $%.3f (cap $%.2f)" % (spent, d, NL, proj, CAP))\n'
     '    qd = queue_done() if spent >= CAP else None; ds = drain_state()\n'
     '    if spent >= CAP and spent < CAP * CEIL and qd and ds != "FAILED":     # v6 DRAINING: the queue is done, the drain is fetching - no destroy mid-fetch\n'
     '        log("ALERT DRAINING spent $%.3f >= cap $%.2f after QUEUE_DONE (drain %s; no destroy until DRAINED / DRAIN FAILED, ceiling $%.2f)" % (spent, CAP, ds or "running", CAP * CEIL))\n'
     '        time.sleep(POLL_S); continue\n'
     '    if spent >= CAP:\n'),
    ('    time.sleep(600)', '    time.sleep(POLL_S)')])
print("wall_guard_v6 written")
