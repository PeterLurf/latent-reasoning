"""§256 DISTRACT host wrapper (DS_v1 §5; Note 255-6 RAM safeguards). Queue lines (the runner appends --device <d> --out <dir>, stripped from the passthrough here):
  <dir> scratch.s256_guarded --mode evalonly --blob <path>=<git blob id> ... --model <model.pt>=<sha16> --args <args.json> --parts <p1,p2> --ship <files>
        asserts the git blob ids and the model's sha256 prefix; <dir>/args.json := <args.json>, <dir>/model.pt := <model.pt>; boxes.inject_aw_eval --run <dir> --parts <...>
        --eval_bs 250; <dir>/log.jsonl := an eval-only stub (the drain's done marker); with --gates also scratch/gate_s255d.py offbit / tiny / acts / smoke / gradflow on
        this host (PILOT_INIT = the model) -> <dir>/gates/*.json, and <dir>/gates_pass.json only if all 5 pass (TRAIN --require's it). Exit = the eval's rc.
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


def multi(argv, flag, stop=("--mode", "--blob", "--model", "--init", "--args", "--parts", "--require", "--ship", "--gates", "--")):
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
        json.dump({"eval_only": True, "model": mp, "model_sha256": s}, open(out + "/log.jsonl", "w")); open(out + "/log.jsonl", "a").write("\n")
        if "--gates" in pre:                                                # Audit (§256 G-D): the 5 build gates on THIS host before TRAIN; gates_pass.json only if all pass
            env = dict(os.environ, PILOT_INIT=mp, PILOT_INIT_SHA=msha, LATENT_ROOT=ROOT); parts = ("offbit", "tiny", "acts", "smoke", "gradflow")
            grc = subprocess.call([sys.executable, "-B", "scratch/gate_s255d.py"] + list(parts), env=env, stdout=open(out + "/gates.log", "w"), stderr=subprocess.STDOUT)
            os.makedirs(out + "/gates", exist_ok=True); gp = {}
            for p_ in parts:
                f = "scratch/gate_s255d/%s.json" % p_
                if os.path.exists(f):
                    shutil.copy(f, out + "/gates/%s.json" % p_); gp[p_] = json.load(open(f)).get("pass")
            if grc == 0 and all(gp.get(p_) for p_ in parts):
                json.dump({"pass": gp}, open(out + "/gates_pass.json", "w"))
        return rc
    ip, isha = multi(pre, "--init")[0].split("="); s = hashlib.sha256(open(ip, "rb").read()).hexdigest(); assert s.startswith(isha), ("init sha", ip, s)
    for f in multi(pre, "--require"):
        assert os.path.exists(f), ("required eval-only output missing", f)
    return subprocess.call([sys.executable, "-u", "-B", "-m", "boxes.inject_aw"] + post + ["--device", dev, "--out", out])


if __name__ == "__main__":
    sys.exit(main())
