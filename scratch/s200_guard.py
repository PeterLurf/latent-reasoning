"""§200 running cost guard (Audit 2026-10-14): for the latent cells A (b287) and B (b288), at >= 2k and >= 6k steps, re-project the family's $ from
realised s/step (each host's log step / sec fields only - no loss is read) x each host's $/h, plus C / D (projected the same way from their logs, or
realised once landed) and b286's realised $. Read-only ssh. A family projection above CAP writes an ALERT line (and exits 3) - the pause itself
(read-only ckpt fetch, then vdestroy) is a decision for Audit / the user. Every check is logged to logs/s200_guard.log.
    python scratch/s200_guard.py   (runs until A and B have passed 6k, or an ALERT)"""
import json
import os
import re
import subprocess
import time

ROOT = "D:/Latent"; os.chdir(ROOT); CAP = 7.0; V = "C:/Users/peter/.local/vastai/vast.sh"; KEY = os.path.expanduser("~/.ssh/runpod_ed25519")
CELLS = {287: "gsm_lat_aug_s1", 288: "gsm_lat_augrs_s1", 289: "gsm_txt_aug_s1", 290: "gsm_txt_augrs_s1"}; STEPS = 24000; EVAL_SETUP_H = 0.35
B286 = {"dph": 1.0, "hours": 0.30}                                       # the fixed realised spend in $: b286 ~0.14 + slow hosts b287 / b288 / b290 ~0.15 (dph 1 x $)


def log(s):
    line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), s); open("logs/s200_guard.log", "a", newline="\n").write(line + "\n"); print(line, flush=True)


def hosts():
    out = {}
    for l in open("logs/gen_deploy.log", encoding="utf-8", errors="replace"):
        m_ = re.search(r"gen host b(\d+) offer \d+: INSTANCE (\d+) (\S+) (\d+) .*STARTED", l)
        if m_ and int(m_.group(1)) in CELLS:
            out[int(m_.group(1))] = (m_.group(2), m_.group(3), m_.group(4))
    return out


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
    done = {287: set(), 288: set()}; proj = {}; info = {}                  # kept across polls: a landed (destroyed) cell keeps its last projection
    while True:
        H = hosts(); P = prices()
        for n, cell in CELLS.items():
            if n not in H:
                continue
            iid, h, p = H[n]; rec = last_rec(h, p, cell); dph = P.get(iid)
            if rec is None or dph is None or rec[0] < 200:
                continue
            sps = rec[1] / rec[0]; hours = STEPS * sps / 3600 + EVAL_SETUP_H; proj[n] = hours * dph; info[n] = {"step": rec[0], "s_per_step": round(sps, 3), "dph": dph, "proj_usd": round(proj[n], 2)}
        missing = [n for n in CELLS if n not in proj]
        fam = sum(proj.values()) + B286["dph"] * B286["hours"]
        for n in (287, 288):
            if n in info:
                for mark in (2000, 6000):
                    if info[n]["step"] >= mark and mark not in done[n]:
                        done[n].add(mark)
                        log("CHECK b%d @%d: %s | family projection $%.2f (cells with data %s; b286 $%.2f)" % (n, mark, info, fam, sorted(proj), B286["dph"] * B286["hours"]))
                        if not missing and fam > CAP:
                            log("ALERT family projection $%.2f > cap $%.2f - pause decision to Audit / the user" % (fam, CAP)); raise SystemExit(3)
        if all({2000, 6000} <= done[n] for n in done):
            log("guard done: A and B past 6k; last family projection $%.2f" % fam); return
        time.sleep(600)


if __name__ == "__main__":
    main()
