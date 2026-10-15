"""§205 live cost guard on the post-gamma rate (Audit 2026-10-15 04:4x; replaces cost_guard s205, whose cumulative s/step mixed in the cheap gamma = 0
phase and whose marks stopped at 4k). Every 10 min, for each started seed: rate = s/step over the records after step 3000 (the IIT term runs only when
gamma > 0, i.e. from write_mix_start 2000), projection = (sec_now + (STEPS - step_now) x rate) / 3600 + EVAL_SETUP_H, times the host's $/h. A seed
without >= 1000 post-gamma steps uses the slowest post-gamma rate seen on the other seeds (cumulative if none); seed 1 not yet started is projected at
that rate x PMAX ($0.24/h). Family = the three + the billed BAD-HOST total (logs/s205_bad_cost.log).
  CHECK lines at every 4k mark per seed (4k .. 48k). Family > CAP: if seed 1 (b296) is running and before step 4000 -> vdestroy it (STOP line);
  otherwise an ALERT line only (Audit decides - stopping a seed late wastes its spend). Read-only ssh; log -> logs/s205_guard.log.
    nohup python scratch/s205_rate_guard.py > logs/s205_rate_guard.out 2>&1 &"""
import json
import os
import re
import subprocess
import time

ROOT = "D:/Latent"; os.chdir(ROOT); V = "C:/Users/peter/.local/vastai/vast.sh"; KEY = os.path.expanduser("~/.ssh/runpod_ed25519")
CELLS = {296: "s205_iit_s1", 297: "s205_iit_s2", 298: "s205_iit_s3"}; STEPS = 48000; CAP = 18.5; EVAL_SETUP_H = 0.35; PMAX = 0.28; G0 = 3000   # user 2026-10-15 ~12:0x (via Audit): family cap $18.50; s1 UNPAIRED, any non-580 driver
MARKS = tuple(range(4000, STEPS + 1, 4000)); S1 = 296; S1_STOP_BEFORE = 4000


def log(s):
    line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), s); open("logs/s205_guard.log", "a", newline="\n").write(line + "\n"); print(line, flush=True)


def hosts():
    out = {}
    for l in open("logs/gen_deploy.log", encoding="utf-8", errors="replace"):
        m_ = re.search(r"gen host b(\d+) offer \d+: INSTANCE (\d+) (\S+) (\d+) .*STARTED", l)
        if m_ and int(m_.group(1)) in CELLS:
            out[int(m_.group(1))] = (m_.group(2), m_.group(3), m_.group(4))
    return out


def bad_total():
    try:
        return sum(float(l.split()[-1]) for l in open("logs/s205_bad_cost.log") if l.strip() and not l.startswith("#"))
    except FileNotFoundError:
        return 0.0


def prices():
    try:
        d = json.loads(subprocess.run(["bash", V, "show", "instances", "--raw"], capture_output=True, text=True, timeout=120).stdout)
        return {str(x["id"]): float(x["dph_total"]) for x in d}
    except Exception:
        return None


def records(h, p, cell):
    cmd = "grep -o '\"step\": [0-9]*, \"loss\": [^,]*, \"sec\": [0-9]*' /workspace/Latent/boxes/runs_fsq/%s/log.jsonl | sed 's/\"loss\": [^,]*, //'" % cell
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-i", KEY, "-p", p, "root@" + h, cmd], capture_output=True, text=True, timeout=90).stdout
        return [(int(a), int(b)) for a, b in re.findall(r'"step": (\d+), "sec": (\d+)', r)]
    except Exception:
        return None


def main():
    done = {n: set() for n in CELLS}; last = {}; alerted = set(); drained = set()
    log("rate guard start: cap $%.2f, cells %s, post-gamma from step %d, s1 placeholder at $%.2f/h, marks every 4k" % (CAP, CELLS, G0, PMAX))
    while True:
        H = hosts(); P = prices(); bad = bad_total()
        dr = {l.strip() for l in open("logs/drained_hosts.txt")} if os.path.exists("logs/drained_hosts.txt") else set()
        for n, cell in CELLS.items():
            if n not in H:
                continue
            iid, h, p = H[n]
            if "%s:%s" % (h, p) in dr:
                drained.add(n); continue
            R = records(h, p, cell); dph = (P or {}).get(iid)
            if not R or dph is None:
                continue
            post = [r for r in R if r[0] >= G0]; st, sec = R[-1]
            rate = (post[-1][1] - post[0][1]) / (post[-1][0] - post[0][0]) if len(post) > 1 and post[-1][0] - post[0][0] >= 1000 else None
            last[n] = {"step": st, "sec": sec, "rate_post_gamma": round(rate, 3) if rate else None, "cum_rate": round(sec / max(st, 1), 3), "dph": round(dph, 4)}
        known = [v["rate_post_gamma"] for v in last.values() if v["rate_post_gamma"]]; fb = max(known) if known else None
        proj = {}
        for n in CELLS:
            if n in last:
                v = last[n]; r_ = v["rate_post_gamma"] or fb or v["cum_rate"]
                proj[n] = ((v["sec"] + (STEPS - v["step"]) * r_) / 3600 + EVAL_SETUP_H) * v["dph"]
            else:
                proj[n] = ((STEPS * (fb or 1.8)) / 3600 + EVAL_SETUP_H) * PMAX
        fam = sum(proj.values()) + bad
        for n in CELLS:
            if n in last:
                for mk in MARKS:
                    if last[n]["step"] >= mk and mk not in done[n]:
                        done[n].add(mk)
                        log("CHECK b%d @%d: %s | per-seed $%s | family $%.2f (incl. BAD-HOST $%.2f) vs cap $%.2f" % (n, mk, last, {k: round(v, 2) for k, v in proj.items()}, fam, bad, CAP))
        if fam > CAP:
            s1 = last.get(S1)
            if s1 is not None and s1["step"] < S1_STOP_BEFORE and S1 in H and S1 not in drained:
                log("ALERT family $%.2f > cap $%.2f with seed 1 at step %d < %d -> STOP seed 1 (Audit rule)" % (fam, CAP, s1["step"], S1_STOP_BEFORE))
                r_ = subprocess.run(["bash", "scratch/vdestroy.sh", H[S1][0]], capture_output=True, text=True, timeout=300); log("STOP b%d: vdestroy %s -> %s" % (S1, H[S1][0], r_.stdout.strip()[-200:]))
                return
            key = tuple(sorted((k, v["step"] // 4000) for k, v in last.items()))   # one ALERT per 4k bucket
            if key not in alerted:
                alerted.add(key); log("ALERT family $%.2f > cap $%.2f (%s) - no automatic stop (seed 1 past 4k or not started); Audit decides" % (fam, CAP, {k: round(v, 2) for k, v in proj.items()}))
        if drained >= set(CELLS) or (all(MARKS[-1] in done[n] for n in CELLS)):
            log("rate guard done: family $%.2f" % fam); return
        if os.environ.get("GUARD_ONESHOT"):
            print("ONESHOT", last, {k: round(v, 2) for k, v in proj.items()}, round(fam, 2)); return
        time.sleep(600)


if __name__ == "__main__":
    main()
