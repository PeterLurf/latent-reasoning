"""Generic running cost guard (from scratch/s202_guard.py; §204: Audit's $3.90 guard). Cells / cap / steps / HIGH placeholders from the command line:
    python scratch/cost_guard.py <tag> <cap> <steps> <n>:<run dir>:<high $> [...]      e.g. s204 3.90 24000 295:gsm_lat_factor_s1:3.70
(the original §202 text follows) for W (b292) and C+ (b293), at >= 1k and >= 4k steps of each started cell,
re-project the pair's $ from realised s/step (each host's log step / sec fields only - no loss is read) x the host's $/h + EVAL_SETUP_H, with a not-yet-started
cell at Theory's §5.7 HIGH figure, plus $0.04 per BAD HOST deploy of b292 / b293 (8 min at <= $0.30/h). A pair projection above CAP ($4, the user's
pre-approval) writes an ALERT line and exits 3 - the pause itself is a decision for Audit / the user. Every check -> logs/s202_guard.log. Read-only ssh.
    python scratch/s202_guard.py   (runs until both cells passed 4k, or an ALERT)"""
import json
import os
import re
import subprocess
import time

import sys
ROOT = "D:/Latent"; os.chdir(ROOT); V = "C:/Users/peter/.local/vastai/vast.sh"; KEY = os.path.expanduser("~/.ssh/runpod_ed25519")
TAG, CAP, STEPS = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]); CELLS = {int(c.split(":")[0]): c.split(":")[1] for c in sys.argv[4:]}
HIGH = {int(c.split(":")[0]): float(c.split(":")[2]) for c in sys.argv[4:]}; EVAL_SETUP_H = 0.35
CSTEPS = {int(c.split(":")[0]): (int(c.split(":")[3]) if len(c.split(":")) > 3 else STEPS) for c in sys.argv[4:]}   # §216: optional per-cell steps (n:cell:high:steps); default STEPS
MARKS = tuple(int(x) for x in os.environ.get("GUARD_MARKS", "1000,4000").split(","))             # §206 (Audit): GUARD_MARKS=200,1000,4000 (G8 at J1's step 200)
RECENT_POLLS = int(os.environ.get("GUARD_RECENT_POLLS", "5"))                                    # Audit 10-21: projection from the RECENT rate over up to this many polls


def cell_marks(n):
    """Audit 10-21 (§218 A1 ran its last 8k of 24k steps unguarded): every cell's marks reach its final step - the given GUARD_MARKS plus 0.75 S, 0.9 S
    and S - 100 (S = the cell's steps; logged every 100), so the guard never exits while a cell is still training."""
    S = CSTEPS[n]; extra = {int(round(0.75 * S, -2)), int(round(0.9 * S, -2)), max(100, S - 100)}
    return tuple(sorted({m for m in MARKS if m < S} | extra))


def project(hist, S, dph):
    """Audit 10-21: $ projection from the RECENT rate. hist = [(step, sec), ...] of this cell's successive polls (oldest first). The recent s/step is taken
    over the oldest of the last RECENT_POLLS records with a smaller step (falls back to the run average sec / step until two distinct records exist);
    hours = (sec_now + (S - step_now) x recent_sps) / 3600 + EVAL_SETUP_H. -> (proj_usd, recent_sps, avg_sps)."""
    step, sec = hist[-1]; avg = sec / step; win = [r for r in hist[-RECENT_POLLS:] if r[0] < step]
    rec_sps = (sec - win[0][1]) / (step - win[0][0]) if win else avg
    return ((sec + max(0, S - step) * rec_sps) / 3600 + EVAL_SETUP_H) * dph, rec_sps, avg
MIRROR = {int(k): int(v) for k, v in (x.split("=") for x in os.environ.get("GUARD_MIRROR", "").split(",") if x)}   # n=m: n without data projects at m's figure
STOP = os.environ.get("GUARD_STOP", "") == "1"                                                   # §206 (Audit): on ALERT destroy every started cell (vdestroy)


def log(s):
    line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), s); open("logs/%s_guard.log" % TAG, "a", newline="\n").write(line + "\n"); print(line, flush=True)


def hosts():
    out = {}; bad = 0
    for l in open("logs/gen_deploy.log", encoding="utf-8", errors="replace"):
        m_ = re.search(r"gen host b(\d+) offer \d+: INSTANCE (\d+) (\S+) (\d+) .*STARTED", l)
        if m_ and int(m_.group(1)) in CELLS:
            out[int(m_.group(1))] = (m_.group(2), m_.group(3), m_.group(4))
        m2 = re.search(r"gen host b(\d+) offer \d+: INSTANCE .*BAD HOST", l)
        if m2 and int(m2.group(1)) in CELLS:
            bad += 1
    return out, bad


def prices():
    try:
        d = json.loads(subprocess.run(["bash", V, "show", "instances", "--raw"], capture_output=True, text=True, timeout=120).stdout)
        return {str(x["id"]): float(x["dph_total"]) for x in d}
    except Exception:
        return {}


def last_rec(h, p, cell):
    cmd = "grep -o '\"step\": [0-9]*, \"loss\": [^,]*, \"sec\": [0-9]*' /workspace/Latent/boxes/runs_fsq/%s/log.jsonl | tail -n 1 | sed 's/\"loss\": [^,]*, //'" % cell
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-i", KEY, "-p", p, "root@" + h, cmd], capture_output=True, text=True, timeout=60).stdout
        m_ = re.search(r'"step": (\d+), "sec": (\d+)', r); return (int(m_.group(1)), int(m_.group(2))) if m_ else None
    except Exception:
        return None


def main():
    done = {n: set() for n in CELLS}; proj = {}; info = {}; hist = {n: [] for n in CELLS}   # kept across polls: a landed (destroyed) cell keeps its last projection
    MK = {n: cell_marks(n) for n in CELLS}
    log("start: cap $%.2f, cells %s, marks %s, recent-rate polls %d" % (CAP, CELLS, MK, RECENT_POLLS))
    while True:
        H, bad = hosts(); P = prices()
        for n, cell in CELLS.items():
            if n not in H:
                continue
            iid, h, p = H[n]; rec = last_rec(h, p, cell); dph = P.get(iid)
            if rec is None or dph is None or rec[0] < 200:
                continue
            if not hist[n] or hist[n][-1][0] != rec[0]:
                hist[n].append(rec)
            proj[n], rsps, asps = project(hist[n], CSTEPS[n], dph)
            info[n] = {"step": rec[0], "s_per_step": round(rsps, 3), "s_per_step_avg": round(asps, 3), "dph": dph, "proj_usd": round(proj[n], 2)}
        pair = sum(proj.get(n, proj.get(MIRROR.get(n), HIGH[n])) for n in CELLS) + 0.04 * bad
        for n in CELLS:
            if n in info:
                for mark in MK[n]:
                    if info[n]["step"] >= mark and mark not in done[n]:
                        done[n].add(mark)
                        log("CHECK b%d @%d: %s | pair projection $%.2f (cells with data %s, others at HIGH %s; BAD HOSTs %d)" % (n, mark, info, pair, sorted(proj), HIGH, bad))
                        if pair > CAP:
                            log("ALERT pair projection $%.2f > cap $%.2f - pause decision to Audit / the user" % (pair, CAP))
                            if STOP:
                                for n2, (iid2, _, _) in hosts()[0].items():
                                    r_ = subprocess.run(["bash", "scratch/vdestroy.sh", iid2], capture_output=True, text=True, timeout=180); log("STOP b%d: vdestroy %s rc %d" % (n2, iid2, r_.returncode))
                            raise SystemExit(3)
        if all(set(MK[n]) <= done[n] for n in done):
            log("guard done: every cell past its last mark (its final step); last pair projection $%.2f" % pair); return
        time.sleep(600)


if __name__ == "__main__":
    main()
