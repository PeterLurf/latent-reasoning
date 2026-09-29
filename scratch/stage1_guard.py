"""Apply the PROTOCOL3 stage-1 restart rule to one training run.

    python scratch/stage1_guard.py <out_dir> <pid|0> <max_relaunch> -- <train args...>

Watches <out_dir>/log.jsonl. If step >= 4000 is logged with no curriculum advance, the process is
killed, the folder renamed <out_dir>_stage1fail_<n>, and the identical command relaunched.
Exits once the run has advanced (or the relaunch budget is spent). Events go to logs/guard_<name>.log.

Optional late-stage rule (PROTOCOL5 plan step 4; off unless given):  --late <T>
    after stage 1, relaunch from scratch if the run spends more than T steps in one of curriculum stages 2-4
    (folder archived as <out_dir>_latefail_<n>; shares the relaunch budget). results/stall_rule.json: T = 14000
    kills no attempt that later reached full depth and saves ~8% of steps per full-depth run; it is not a fix.
"""
import json
import os
import subprocess
import sys
import time

head = sys.argv[1:sys.argv.index("--")]
wait_end = "--wait" in head                    # keep running until training exits (for job chains)
head = [h for h in head if h != "--wait"]
late = None
if "--late" in head:
    i = head.index("--late"); late = int(head[i + 1]); head = head[:i] + head[i + 2:]
    wait_end = True
out, pid, max_re = head[0], int(head[1]), int(head[2])
cmd = sys.argv[sys.argv.index("--") + 1:]
name = os.path.basename(out)
glog = open(f"logs/guard_{name}.log", "a")
CHILDREN = {}                                  # pid -> Popen for runs this guard launched


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, file=glog, flush=True)


def launch():
    o = open(f"logs/{name}.out", "a"); e = open(f"logs/{name}.err", "a")
    flags = 0x08000000 if os.name == "nt" else 0          # CREATE_NO_WINDOW (Windows only)
    p = subprocess.Popen([sys.executable] + cmd, stdout=o, stderr=e, creationflags=flags)
    CHILDREN[p.pid] = p
    say("launched pid", p.pid)
    return p.pid


def status(full=False):
    f = os.path.join(out, "log.jsonl")
    if not os.path.exists(f):
        return (0, False, 0, False) if full else (0, False)
    step, adv, last_adv, done = 0, False, 0, False
    for l in open(f):
        x = json.loads(l)
        step = max(step, x.get("step", 0))
        if "curriculum_advance_to" in x:
            adv, last_adv = True, x.get("step", step)
            done |= x["curriculum_advance_to"] is None
    return (step, adv, last_adv, done) if full else (step, adv)


def alive(p):
    if p in CHILDREN:                        # our own child: poll() also reaps it (no zombie on POSIX)
        return CHILDREN[p].poll() is None
    if os.name == "nt":
        r = subprocess.run(["tasklist", "/FI", f"PID eq {p}", "/NH"], capture_output=True, text=True).stdout
        return str(p) in r
    try:
        os.kill(p, 0)
        return True
    except ProcessLookupError:
        return False


def kill(p):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(p), "/F"], capture_output=True)
    elif p in CHILDREN:
        CHILDREN[p].kill(); CHILDREN[p].wait()
    else:
        os.kill(p, 9)


say("guarding", out, "pid", pid, "max relaunch", max_re)
if pid == 0:
    pid = launch()
n = 0
while True:
    time.sleep(60)
    step, adv = status()
    if adv:
        say("stage 1 passed at step", step, "; waiting for the run to end" if wait_end else "; guard exits")
        relaunched = False
        while wait_end and alive(pid):
            time.sleep(60)
            if late is None:
                continue
            step, _, last_adv, done = status(full=True)
            if not done and step - last_adv > late:
                kill(pid)
                time.sleep(10)
                n += 1
                os.rename(out, f"{out}_latefail_{n}")
                say(f"late-stage stall: {step - last_adv} steps since the advance at {last_adv}; archived as {out}_latefail_{n}")
                if n > max_re:
                    say("relaunch budget spent")
                    break
                pid = launch()
                relaunched = True
                break
        if relaunched:
            continue
        break
    if not alive(pid):
        say("process exited without stage-1 advance at step", step)
        break
    if step >= 4000:
        kill(pid)
        time.sleep(10)
        n += 1
        os.rename(out, f"{out}_stage1fail_{n}")
        say(f"stage-1 failure at step {step}; archived as {out}_stage1fail_{n}")
        if n > max_re:
            say("relaunch budget spent")
            break
        pid = launch()
