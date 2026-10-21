"""Off-host checkpoint mirror (user "checkpoints remember", relayed by Audit 21:5x 10-15): every --every_h hours, for each listed live cell, copy the host's
<run>/ckpt_last.pt (model + optimizer + scheduler / RNG state, so resume is exact) to scratch/ckpt_mirror/<run>/ckpt_last.pt. Read-only on the host.
Per fetch: remote sha256 + size -> scp to scratch/_tmp/ckmirror/<run>.part -> remote sha256 again (nested170 writes ckpt_last in place: a change during
the copy is a retry) -> local sha256 == the pre-copy sha (2026-10-16: s2 may differ when the trainer replaced the file mid-copy) -> torch.load check (step present) -> os.replace into the mirror (the previous copy replaced: one copy per
run). A separate local dir: never the run dir a gen8 drain writes into (no-manual-fetch-during-drain lesson). Cells whose host is drained / destroyed are
skipped; their mirror is deleted once <run>/eval.json exists locally (retention: delete after landing + drain). -> logs/ckpt_mirror.log
    nohup python scratch/ckpt_mirror.py 296:s205_iit_s1 297:s205_iit_s2 303:gsm_codi_20ep_s1 [--every_h 2] > logs/ckpt_mirror.out 2>&1 &
--companions f1,f2 (Note 218-15, 10-20): files beside ckpt_last.pt copied WITH it (e.g. ckpt_fingerprint.json,log.jsonl). They are staged on the host first
(cp into /workspace/_mirror_stage/<run>/, sha256 there), then scp'd and sha-checked locally; a companion with a "step" key (the fingerprint) must equal the
ckpt's step, else the whole copy retries (the trainer checkpointed meanwhile). --optional f3: copied if present. All land in the mirror dir together.
--local_host DIR (smokes only): DIR stands for the host's /workspace (commands run in local bash, cp instead of scp)."""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

ROOT = "D:/Latent"; os.chdir(ROOT); KEY = os.path.expanduser("~/.ssh/runpod_ed25519"); MIR = "scratch/ckpt_mirror"; TMP = "scratch/_tmp/ckmirror"
LOCAL = None; COMP = []; OPT = []                                       # set from the command line in main()


def log(s):
    line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), s); open("logs/ckpt_mirror.log", "a", newline="\n").write(line + "\n"); print(line, flush=True)


def host(n):
    if LOCAL:
        return ("local", "0")
    h = None
    for l in open("logs/gen_deploy.log", encoding="utf-8", errors="replace"):
        m_ = re.search(r"gen host b%d offer \d+: INSTANCE (\d+) (\S+) (\d+) .*STARTED" % n, l)
        if m_:
            h = (m_.group(2), m_.group(3))
    return h


def ssh(h, p, cmd, t=120):
    if LOCAL:
        return subprocess.run(["bash", "-c", cmd.replace("/workspace", LOCAL)], capture_output=True, text=True, timeout=t).stdout.strip()
    return subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-i", KEY, "-p", p, "root@" + h, cmd], capture_output=True, text=True, timeout=t).stdout.strip()


def sha_local(f):
    h_ = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h_.update(b)
    return h_.hexdigest()


def fetch(h, p, rf, lp):
    if LOCAL:
        return subprocess.run(["cp", rf.replace("/workspace", LOCAL), lp], capture_output=True, text=True, timeout=7200)
    return subprocess.run(["scp", "-q", "-i", KEY, "-P", p, "root@%s:%s" % (h, rf), lp], capture_output=True, text=True, timeout=7200)


def stage(h, p, run):
    """copy COMP (required) + OPT (if present) into /workspace/_mirror_stage/<run>/ on the host -> {file: sha256}, or None if a required one is missing."""
    out = ssh(h, p, "d=/workspace/_mirror_stage/%s; rm -rf $d; mkdir -p $d; cd /workspace/Latent/boxes/runs_fsq/%s || exit 0; for f in %s; do [ -f $f ] && cp $f $d/; done; "
                    "cd $d && sha256sum * 2>/dev/null" % (run, run, " ".join(COMP + OPT)), 600)
    sh = {l.split()[1].lstrip("*"): l.split()[0] for l in out.splitlines() if len(l.split()) == 2}
    return sh if all(f in sh for f in COMP) else None


def attempt_aside(md, run, n, h, p):
    """2026-10-21 (Audit; a relaunch of the same run name overwrote §218 attempt 1's kept mirror): if <md>/meta.json exists and names a different run /
    queue / host, rename <md> to <md>.attempt<k> (k = 1 + the highest existing attempt index) before the first copy - never delete. Refuses (returns False,
    logs MIRROR_ASIDE_REFUSED) if that target already exists. -> the new path, None (nothing to move) or False."""
    mf = os.path.join(md, "meta.json")
    if not os.path.exists(mf):
        return None
    m_ = json.load(open(mf))
    if m_.get("run") == run and m_.get("queue") == n and m_.get("host") == "%s:%s" % (h, p):
        return None
    ks = [int(x.rsplit("attempt", 1)[1]) for x in os.listdir(os.path.dirname(md)) if x.startswith(os.path.basename(md) + ".attempt") and x.rsplit("attempt", 1)[1].isdigit()]
    tgt = "%s.attempt%d" % (md, (max(ks) if ks else 0) + 1)
    if os.path.exists(tgt):
        log("MIRROR_ASIDE_REFUSED %s: %s exists" % (run, tgt)); return False
    os.rename(md, tgt); log("MIRROR_ASIDE %s: previous mirror (%s, %s, step %s) moved to %s" % (run, m_.get("host"), m_.get("queue"), m_.get("step"), tgt))
    return tgt


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
    if "QUEUE_DONE" in ssh(h, p, "tail -n 3 /workspace/Latent/logs/queue_gen_b%d.out 2>/dev/null" % n, 60):   # Audit 06:2x 10-16: the drain owns the host's
        return "queue done (the drain owns the upload)"                     # upload once the queue is done (a 06:01 copy competed with b304's drain)
    rf = "/workspace/Latent/boxes/runs_fsq/%s/ckpt_last.pt" % run
    for attempt in range(3):
        s1 = ssh(h, p, "sha256sum %s 2>/dev/null | cut -c1-64" % rf, 600)
        if not s1:
            return "no ckpt_last on host yet"
        os.makedirs(TMP, exist_ok=True); part = os.path.join(TMP, run + ".part")
        cs = stage(h, p, run) if (COMP or OPT) else {}                     # companions staged BEFORE the ckpt is read (fingerprint step checked below)
        if cs is None:
            log("MIRROR_RETRY %s companions missing on the host" % run); time.sleep(60); continue
        r = fetch(h, p, rf, part)
        s2 = ssh(h, p, "sha256sum %s 2>/dev/null | cut -c1-64" % rf, 600)
        if r.returncode != 0 or not os.path.exists(part):
            log("MIRROR_RETRY %s scp rc %s" % (run, r.returncode)); continue
        sl = sha_local(part)
        if sl != s1:                                                        # 2026-10-16: the trainer renames ckpt_last -> ckpt_prev, then writes a NEW file;
            # an scp that opened the old file keeps reading it unchanged, so local == the pre-copy sha (+ the torch.load / step check below) is a
            # consistent copy even if the path was replaced meanwhile (s2 != s1). §211 cells checkpoint every ~5.5 min; a 1.5 GB copy takes longer.
            log("MIRROR_RETRY %s sha changed / mismatch (attempt %d)" % (run, attempt + 1)); os.remove(part); time.sleep(60); continue
        import torch
        ck = torch.load(part, map_location="cpu", weights_only=False); step = ck.get("step") if isinstance(ck, dict) else None; del ck
        if step is None:
            log("MIRROR_FAIL %s no step in the checkpoint" % run); os.remove(part); return "bad ckpt"
        cl = {}
        for f, sh in cs.items():
            lp = os.path.join(TMP, run + "." + f); rr = fetch(h, p, "/workspace/_mirror_stage/%s/%s" % (run, f), lp)
            if rr.returncode != 0 or not os.path.exists(lp) or sha_local(lp) != sh:
                cl = None; break
            cl[f] = lp
        if cl is None:
            log("MIRROR_RETRY %s companion copy / sha mismatch (attempt %d)" % (run, attempt + 1)); os.remove(part); continue
        fsteps = {f: json.load(open(lp)).get("step") for f, lp in cl.items() if f.endswith(".json") and isinstance(json.load(open(lp)), dict)}
        if any(v is not None and v != step for v in fsteps.values()):
            log("MIRROR_RETRY %s companion step %s != ckpt step %s (attempt %d)" % (run, fsteps, step, attempt + 1)); os.remove(part); time.sleep(60); continue
        aside = attempt_aside(md, run, n, h, p)
        if aside is False:
            os.remove(part); [os.remove(lp) for lp in cl.values()]; return "refused (attempt-aside target exists)"
        os.makedirs(md, exist_ok=True)
        for f, lp in cl.items():                                            # companions first, the ckpt last: a ckpt in the mirror always has its set
            os.replace(lp, os.path.join(md, f))
        os.replace(part, os.path.join(md, "ckpt_last.pt"))
        json.dump({"run": run, "queue": n, "host": "%s:%s" % (h, p), "step": step, "sha256": sl, "companions": {f: sha_local(os.path.join(md, f)) for f in cl}, "bytes": os.path.getsize(os.path.join(md, "ckpt_last.pt")), "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                  open(os.path.join(md, "meta.json"), "w", newline="\n"), indent=1)
        log("MIRROR %s step %s (%s:%s, %.2f GB, sha %s)" % (run, step, h, p, os.path.getsize(os.path.join(md, "ckpt_last.pt")) / 1e9, sl[:12])); return "ok"
    log("MIRROR_FAIL %s after 3 attempts" % run); return "fail"


def main():
    global LOCAL, COMP, OPT
    ap = argparse.ArgumentParser(); ap.add_argument("cells", nargs="+"); ap.add_argument("--every_h", type=float, default=2.0); ap.add_argument("--once", action="store_true")
    ap.add_argument("--companions", default=""); ap.add_argument("--optional", default=""); ap.add_argument("--local_host", default=""); a = ap.parse_args()
    LOCAL = a.local_host or None; COMP = [x for x in a.companions.split(",") if x]; OPT = [x for x in a.optional.split(",") if x]
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
