"""§268 LINEAGE-SEED host runner (build serial s273; Note 268-1): one lineage stage on the s1 stage's exact trainer version. Queue line (the runner appends
--device <d> --out <dir>):
  <dir> scratch.s273_hist --tar scratch/s273_tree_<commit>.tar=<sha16> --trainer <path in tree>=<sha256 prefix> [--init <model.pt>=<sha16> ...] --runner <key>
        --ship <tar> [<inits>] -- <flags>      (<key>: askname = the §233 wrapper script in the tree; m:<module> = -m <module>; --ship is read by the deploy only)
Asserts the tar's sha256 prefix, extracts it to scratch/_hist/<commit>/ (boxes/runs_fsq and prosqa linked to /workspace/Latent's, so --init_* / GPT-2 resolve), asserts the
trainer file's sha256 (= the s1 args.json module_sha256) and every --init file's sha256 prefix, then runs `python -B <run prefix> <flags> --device <d> --out <abs dir>` in the
tree. host_shas.json (s257_guarded.manifest) after the run. Exit = the trainer's rc."""
import hashlib
import json
import os
import subprocess
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scratch"))


GPT2_SHA = "7c5d3f4b8b76583b"                                               # prosqa/ckpt/gpt2_pretrained.bin (gen_vast_job.sh)
RUNS = {"askname": ["docs/research_steptold_2026-10-24/st_ask_name_local.py"]}


def take(argv, flag):
    i = argv.index(flag); v = argv[i + 1]; del argv[i:i + 2]; return v


def multi(argv, flag):
    if flag not in argv:
        return []
    i = argv.index(flag) + 1; out = []
    while i < len(argv) and not argv[i].startswith("--"):
        out.append(argv[i]); i += 1
    return out


def manifest(out):
    """sha256 of every result json + log.jsonl in <out>, computed on the host -> <out>/host_shas.json (the s257_guarded.manifest rule)."""
    fs = sorted(f for f in os.listdir(out) if f.endswith(".json") and f != "host_shas.json") + [f for f in ("log.jsonl",) if os.path.exists(os.path.join(out, f))]
    m = {f: sha(os.path.join(out, f)) for f in fs}; json.dump(m, open(os.path.join(out, "host_shas.json"), "w"), indent=1); print("HOST_SHAS", json.dumps(m), flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    argv = sys.argv[1:]; i = argv.index("--"); own, rest = argv[:i], argv[i + 1:]
    dev = rest[rest.index("--device") + 1]; out = os.path.abspath(os.path.join(ROOT, rest[rest.index("--out") + 1]))
    k = rest.index("--device"); del rest[k:k + 2]; k = rest.index("--out"); del rest[k:k + 2]
    tar, tsha = take(own, "--tar").split("="); trainer, trsha = take(own, "--trainer").split("="); key = take(own, "--runner")
    run = RUNS[key] if key in RUNS else ["-m", key.split(":", 1)[1]]; assert key in RUNS or key.startswith("m:"), key; inits = multi(own, "--init")
    assert sha(os.path.join(ROOT, tar)).startswith(tsha), ("tar sha", tar)
    commit = os.path.basename(tar).split("_")[-1].split(".")[0]; t = os.path.join(ROOT, "scratch", "_hist", commit + "_" + tsha)   # Audit LOW (e): keyed by the tar sha
    if not os.path.isdir(os.path.join(t, "boxes")):
        os.makedirs(t, exist_ok=True); tarfile.open(os.path.join(ROOT, tar)).extractall(t)
    for sub in ("prosqa", os.path.join("boxes", "runs_fsq")):
        q = os.path.join(t, sub)
        if not os.path.lexists(q):
            try:
                os.symlink(os.path.join(ROOT, sub), q)
            except OSError:                                                 # local Windows smoke without the symlink privilege: a junction
                assert os.name == "nt"; subprocess.check_call(["cmd", "/c", "mklink", "/J", q, os.path.join(ROOT, sub)], stdout=subprocess.DEVNULL)
    assert sha(os.path.join(t, trainer)).startswith(trsha), ("trainer version", trainer)
    assert sha(os.path.join(ROOT, "prosqa", "ckpt", "gpt2_pretrained.bin")).startswith(GPT2_SHA), "GPT-2 weights sha"   # Audit LOW (d)
    for x in inits:
        p, s = x.split("="); assert sha(os.path.join(ROOT, p)).startswith(s), ("init sha", p)
    os.makedirs(out, exist_ok=True); print("S273_HIST", commit, trainer, " ".join(run), flush=True)
    rc = subprocess.call([sys.executable, "-B"] + run + rest + ["--device", dev, "--out", out], cwd=t)
    manifest(out); sys.exit(rc)


if __name__ == "__main__":
    main()
