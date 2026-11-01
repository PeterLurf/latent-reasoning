"""§256 DISTRACT host wrapper (DS_v1 §5; Note 255-6 RAM safeguards). Queue lines (the runner appends --device <d> --out <dir>, stripped from the passthrough here):
  <dir> scratch.s256_guarded --mode evalonly --blob <path>=<git blob id> ... --model <model.pt>=<sha16> --args <args.json> --parts <p1,p2> --ship <files>
        asserts the git blob ids and the model's sha256 prefix; <dir>/args.json := <args.json>, <dir>/model.pt := <model.pt>; boxes.inject_aw_eval --run <dir> --parts <...>
        --eval_bs 250; <dir>/log.jsonl := an eval-only stub (the drain's done marker). Exit = the eval's rc.
  <dir> scratch.s256_guarded --mode train --blob ... --init <model.pt>=<sha16> --require <file> ... -- <boxes.inject_aw flags>
        asserts the git blob ids, the init's sha256 prefix and that every --require file exists (the eval-only line's new parts ran on this host), then
        boxes.inject_aw <flags> --device <d> --out <dir>. Exit = the trainer's rc."""
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)


def blob_id(p):
    b = open(p, "rb").read(); return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def take(argv, flag):
    """pop the LAST '<flag> <value>' pair from argv (the runner's --device / --out come last) -> value."""
    i = len(argv) - 1 - argv[::-1].index(flag); v = argv[i + 1]; del argv[i:i + 2]; return v


def multi(argv, flag, stop=("--mode", "--blob", "--model", "--init", "--args", "--parts", "--require", "--ship", "--")):
    if flag not in argv:
        return []
    i = argv.index(flag) + 1; out = []
    while i < len(argv) and argv[i] not in stop:
        out.append(argv[i]); i += 1
    return out


def main():
    argv = sys.argv[1:]; dev = take(argv, "--device"); out = take(argv, "--out")
    pre, post = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    mode = multi(pre, "--mode")[0]
    for x in multi(pre, "--blob"):
        p, want = x.split("="); got = blob_id(p); assert got == want, ("git blob", p, got, want)
    if mode == "evalonly":
        mp, msha = multi(pre, "--model")[0].split("="); s = hashlib.sha256(open(mp, "rb").read()).hexdigest(); assert s.startswith(msha), ("model sha", mp, s)
        os.makedirs(out, exist_ok=True); shutil.copy(multi(pre, "--args")[0], out + "/args.json"); shutil.copy(mp, out + "/model.pt")
        rc = subprocess.call([sys.executable, "-u", "-B", "-m", "boxes.inject_aw_eval", "--run", out, "--parts", multi(pre, "--parts")[0], "--eval_bs", "250", "--device", dev])
        json.dump({"eval_only": True, "model": mp, "model_sha256": s}, open(out + "/log.jsonl", "w")); open(out + "/log.jsonl", "a").write("\n"); return rc
    ip, isha = multi(pre, "--init")[0].split("="); s = hashlib.sha256(open(ip, "rb").read()).hexdigest(); assert s.startswith(isha), ("init sha", ip, s)
    for f in multi(pre, "--require"):
        assert os.path.exists(f), ("required eval-only output missing", f)
    return subprocess.call([sys.executable, "-u", "-B", "-m", "boxes.inject_aw"] + post + ["--device", dev, "--out", out])


if __name__ == "__main__":
    sys.exit(main())
