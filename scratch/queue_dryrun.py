"""Template v6 item 2 (Audit 11-03 LAUNCH_POSTMORTEM §3.2): a queue dry-run on a fake host tree built from EXACTLY what a deploy ships
(scratch/auto_ship.py <queue> + gen_deploy_120's fixed EXTRA_FILES + vdeploy_toy's fixed tar members + the latent/ dir + the queue; GPT-2's prosqa/ckpt/gpt2_pretrained.bin,
which hosts have), hardlinked (.pt / .bin) or copied into scratch/_tmp/qdry/<tag>/. Per non-comment line, in that tree (cwd, python -B, CUDA_VISIBLE_DEVICES=''):
  import     the line's module imports;
  blob       every --blob <path>=<git blob> exists in the tree with that blob id;
  paths      every --model / --init / --init_model / --init_trunk / --args / --args_ref / --require / --run path exists in the tree or under an earlier line's out dir
             (produced on the host); <path>=<sha16> values: the file's streaming sha256 starts with it;
  flags      a boxes.inject_aw line, or the training flags after '--' of a train wrapper: boxes.inject_aw <flags> --args_only --device cpu runs (rc 0) - argparse /
             assert / init errors (distinct flag sets once);
  parts      --parts / --eval_parts names are inject_aw_eval.PARTS;
  gates      a scratch.sNNN_gates line: every scratch/gate_sNNN.py it names imports in the tree.
Exit 1 + 'DRYRUN FAIL <line> <check>: <why>' lines, else 'DRYRUN OK (<n> lines, <m> files)'.
    python -B scratch/queue_dryrun.py scratch/gen_q_b<n>.txt[.hold] [--keep]"""
import hashlib
import os
import re
import shutil
import subprocess
import sys

ROOT = "D:/Latent"; os.chdir(ROOT); sys.path.insert(0, ROOT + "/scratch/ram_scan")
FIXED = ["boxes/gen97.py", "boxes/mk108.py", "boxes/fsq120.py", "scratch/gen_vast_job.sh", "scratch/queue_worker2.sh", "scratch/stage1_guard.py", "scratch/eval_battery.sh",
         "prosqa/ckpt/gpt2_pretrained.bin"]
PATHF = ("--model", "--init", "--init_model", "--init_trunk", "--args", "--args_ref", "--require", "--run")


def blob_id(p):
    b = open(p, "rb").read(); return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def build(q, tag):
    import tempfile                                                         # a unique tree per run (Audit 11-03: a leftover tree raised FileExistsError)
    os.makedirs(ROOT + "/scratch/_tmp/qdry", exist_ok=True); T = tempfile.mkdtemp(prefix=tag + "_", dir=ROOT + "/scratch/_tmp/qdry").replace("\\", "/")
    files = subprocess.check_output([sys.executable, "scratch/auto_ship.py", q]).decode().split() + FIXED + [q]
    files += [os.path.join(dp, f).replace("\\", "/") for dp, _, fs in os.walk("latent") for f in fs if "__pycache__" not in dp]
    for f in sorted(set(files)):
        if not os.path.isfile(f):
            continue
        d = os.path.join(T, os.path.dirname(f)); os.makedirs(d, exist_ok=True)
        if f.endswith((".pt", ".bin", ".tar")):
            try:
                os.link(f, os.path.join(T, f)); continue
            except OSError:
                pass
        shutil.copy2(f, os.path.join(T, f))
    for d in ("logs", "boxes/runs_fsq"):
        os.makedirs(os.path.join(T, d), exist_ok=True)
    return T, len(set(files))


def run(T, args, timeout=900):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=""); r = subprocess.run([sys.executable, "-B"] + args, cwd=T, env=env, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr).strip().split("\n")[-1][:200]


def main():
    q = sys.argv[1]; tag = os.path.basename(q).replace(".", "_"); T, nf = build(q, tag); fails = []; outs = []; flag_seen = set()
    try:
        import boxes.inject_aw_eval as EV
        PARTS = set(EV.PARTS)
    except Exception:
        PARTS = None
    from nobuf_sha import sha_nobuf
    lines = [ln.split() for ln in open(q, encoding="utf-8") if ln.strip() and not ln.startswith("#")]
    for li, t in enumerate(lines):
        out, mod = t[0], t[1]; tagl = "%d %s %s" % (li + 1, out, mod)
        rc, msg = run(T, ["-c", "import importlib, sys; importlib.import_module(sys.argv[1])", mod])
        if rc:
            fails.append((tagl, "import", msg))
        flag = None
        for x in t[2:]:
            if x.startswith("--"):
                flag = x; continue
            if flag == "--blob" and "=" in x:
                p, want = x.split("=", 1)
                if not os.path.isfile(os.path.join(T, p)):
                    fails.append((tagl, "blob", "%s not shipped" % p))
                elif blob_id(os.path.join(T, p)) != want:
                    fails.append((tagl, "blob", "%s blob != %s" % (p, want[:12])))
            if flag in PATHF:
                p = x.split("=")[0]; produced = any(p == o or p.startswith(o + "/") for o in outs + [out])
                if not os.path.exists(os.path.join(T, p)) and not produced:
                    fails.append((tagl, "paths", "%s %s neither shipped nor produced by an earlier line" % (flag, p)))
                elif "=" in x and os.path.isfile(os.path.join(T, p)) and not sha_nobuf(os.path.abspath(os.path.join(T, p))).startswith(x.split("=")[1]):
                    fails.append((tagl, "paths", "%s sha256 != %s" % (p, x.split("=")[1])))
            if flag in ("--parts", "--eval_parts") and PARTS is not None:
                bad = [p_ for p_ in x.split(",") if p_ and p_ not in PARTS]
                if bad:
                    fails.append((tagl, "parts", "unknown parts %s" % bad))
            flag = flag if flag in ("--blob", "--ship") else None
        tf = t[2:] if mod == "boxes.inject_aw" else (t[t.index("--") + 1:] if "--" in t and mod.endswith("_guarded") and "train" in t else None)
        if tf is not None:
            tf = [x for i, x in enumerate(tf) if x not in ("--mem_check",) and not (i > 0 and tf[i - 1] == "--time_steps") and x != "--time_steps"]
            key = " ".join(tf)
            if key not in flag_seen:
                flag_seen.add(key); d_ = "scratch/_tmp/qdry_args"; rc, msg = run(T, ["-m", "boxes.inject_aw"] + tf + ["--device", "cpu", "--args_only", "--out", d_], timeout=1800)
                if rc:
                    fails.append((tagl, "flags", msg))
        if re.match(r"^scratch\.s\d+_gates$", mod) and os.path.isfile(os.path.join(T, mod.replace(".", "/") + ".py")):
            for g_ in sorted(set(re.findall(r"scratch/gate_s\d+\.py", open(os.path.join(T, mod.replace(".", "/") + ".py"), encoding="utf-8").read()))):
                rc, msg = run(T, ["-c", "import runpy, sys; sys.argv = ['x']; runpy.run_path(sys.argv[0] if False else %r)" % g_]) if os.path.isfile(os.path.join(T, g_)) else (1, "not shipped")
                if rc:
                    fails.append((tagl, "gates", "%s: %s" % (g_, msg)))
        outs.append(out)
    if "--keep" not in sys.argv:
        shutil.rmtree(T, ignore_errors=True)
    for f in fails:
        print("DRYRUN FAIL %s %s: %s" % f)
    print("DRYRUN %s (%d lines, %d files)" % ("FAIL" if fails else "OK", len(lines), nf)); return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
