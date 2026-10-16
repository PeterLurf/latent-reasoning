"""Off-host checkpoint mirror (user "checkpoints remember", relayed by Audit 21:5x 10-15): every --every_h hours, for each listed live cell, copy the host's
<run>/ckpt_last.pt (model + optimizer + scheduler / RNG state, so resume is exact) to scratch/ckpt_mirror/<run>/ckpt_last.pt. Read-only on the host.
Per fetch: remote sha256 + size -> scp to scratch/_tmp/ckmirror/<run>.part -> remote sha256 again (nested170 writes ckpt_last in place: a change during
the copy is a retry) -> local sha256 == both -> torch.load check (step present) -> os.replace into the mirror (the previous copy replaced: one copy per
run). A separate local dir: never the run dir a gen8 drain writes into (no-manual-fetch-during-drain lesson). Cells whose host is drained / destroyed are
skipped; their mirror is deleted once <run>/eval.json exists locally (retention: delete after landing + drain). -> logs/ckpt_mirror.log
    nohup python scratch/ckpt_mirror.py 296:s205_iit_s1 297:s205_iit_s2 303:gsm_codi_20ep_s1 [--every_h 2] > logs/ckpt_mirror.out 2>&1 &"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

ROOT = "D:/Latent"; os.chdir(ROOT); KEY = os.path.expanduser("~/.ssh/runpod_ed25519"); MIR = "scratch/ckpt_mirror"; TMP = "scratch/_tmp/ckmirror"


def log(s):
    line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), s); open("logs/ckpt_mirror.log", "a", newline="\n").write(line + "\n"); print(line, flush=True)


def host(n):
    h = None
    for l in open("logs/gen_deploy.log", encoding="utf-8", errors="replace"):
        m_ = re.search(r"gen host b%d offer \d+: INSTANCE (\d+) (\S+) (\d+) .*STARTED" % n, l)
        if m_:
            h = (m_.group(2), m_.group(3))
    return h


def ssh(h, p, cmd, t=120):
    return subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-i", KEY, "-p", p, "root@" + h, cmd], capture_output=True, text=True, timeout=t).stdout.strip()


def sha_local(f):
    h_ = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h_.update(b)
    return h_.hexdigest()


def mirror(n, run):
    hp = host(n)
    if hp is None:
        return "no host yet"
    h, p = hp; drained = os.path.exists("logs/drained_hosts.txt") and ("%s:%s" % (h, p)) in open("logs/drained_hosts.txt").read().split()
    md = os.path.join(MIR, run)
    if drained:
        if os.path.exists(os.path.join(ROOT, "boxes", "runs_fsq", run, "eval.json")) and os.path.isdir(md):
            shutil.rmtree(md); log("RETENTION %s landed + drained -> mirror deleted" % run)
        return "drained"
    rf = "/workspace/Latent/boxes/runs_fsq/%s/ckpt_last.pt" % run
    for attempt in range(3):
        s1 = ssh(h, p, "sha256sum %s 2>/dev/null | cut -c1-64" % rf, 600)
        if not s1:
            return "no ckpt_last on host yet"
        os.makedirs(TMP, exist_ok=True); part = os.path.join(TMP, run + ".part")
        r = subprocess.run(["scp", "-q", "-i", KEY, "-P", p, "root@%s:%s" % (h, rf), part], capture_output=True, text=True, timeout=7200)
        s2 = ssh(h, p, "sha256sum %s 2>/dev/null | cut -c1-64" % rf, 600)
        if r.returncode != 0 or not os.path.exists(part):
            log("MIRROR_RETRY %s scp rc %s" % (run, r.returncode)); continue
        sl = sha_local(part)
        if not (s1 == s2 == sl):
            log("MIRROR_RETRY %s sha changed / mismatch (attempt %d)" % (run, attempt + 1)); os.remove(part); time.sleep(60); continue
        import torch
        ck = torch.load(part, map_location="cpu", weights_only=False); step = ck.get("step") if isinstance(ck, dict) else None; del ck
        if step is None:
            log("MIRROR_FAIL %s no step in the checkpoint" % run); os.remove(part); return "bad ckpt"
        os.makedirs(md, exist_ok=True); os.replace(part, os.path.join(md, "ckpt_last.pt"))
        json.dump({"run": run, "queue": n, "host": "%s:%s" % (h, p), "step": step, "sha256": sl, "bytes": os.path.getsize(os.path.join(md, "ckpt_last.pt")), "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                  open(os.path.join(md, "meta.json"), "w", newline="\n"), indent=1)
        log("MIRROR %s step %s (%s:%s, %.2f GB, sha %s)" % (run, step, h, p, os.path.getsize(os.path.join(md, "ckpt_last.pt")) / 1e9, sl[:12])); return "ok"
    log("MIRROR_FAIL %s after 3 attempts" % run); return "fail"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cells", nargs="+"); ap.add_argument("--every_h", type=float, default=2.0); ap.add_argument("--once", action="store_true"); a = ap.parse_args()
    cells = [(int(c.split(":")[0]), c.split(":")[1]) for c in a.cells]; log("start: cells %s every %.1f h" % (cells, a.every_h))
    while True:
        for n, run in cells:
            try:
                st = mirror(n, run)
            except Exception as e:                                          # noqa: BLE001  (one bad cycle never kills the mirror)
                st = "error %r" % (e,); log("MIRROR_ERROR %s %r" % (run, e))
            print(time.strftime("%H:%M:%S"), run, st, flush=True)
        if a.once:
            return
        time.sleep(a.every_h * 3600)


if __name__ == "__main__":
    main()
