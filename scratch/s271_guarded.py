"""s271 copy (Audit CODE_REVIEW 11-03 MED-2: the eval-only done-marker log.jsonl records eval_rc / status; a failed eval gets an explicit FAILED eval.json; a run_long skip is printed and recorded) of Note 257-2 eval-only host wrapper = scratch/s257_guarded.py + --long <K'list>:<ds>:<name> (after the parts: boxes.inject_aw_eval --kprime_list / --long_ds /
--long_name = the §239 run_long, same model / dir). The §257 doc follows. §257 DISTRACT-CHAIN host wrapper (DC2_v1 §5; + Audit G-D note 1: host_shas.json, the host sha256 of every result file, written after each mode) = scratch/s256_guarded.py with the §257 gates (scratch/gate_s257.py offbit / offbit_d / acts / smoke / gradflow,
--pilot_k <k> -> PILOT_K; outputs scratch/gate_s257d/). §256 doc: Queue lines (the runner appends --device <d> --out <dir>, stripped from the passthrough here):
  <dir> scratch.s256_guarded --mode evalonly --blob <path>=<git blob id> ... --model <model.pt>=<sha16> --args <args.json> --parts <p1,p2> --ship <files>
        asserts the git blob ids and the model's sha256 prefix; <dir>/args.json := <args.json>, <dir>/model.pt := <model.pt>; boxes.inject_aw_eval --run <dir> --parts <...>
        --eval_bs 250; <dir>/log.jsonl := an eval-only stub (the drain's done marker); with --gates also scratch/gate_s255d.py offbit / tiny / acts / smoke / gradflow on
        this host (PILOT_INIT = the model) -> <dir>/gates/*.json, and <dir>/gates_pass.json only if all pass (TRAIN --require's it). Exit = the eval's rc.
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


def multi(argv, flag, stop=("--mode", "--blob", "--model", "--init", "--args", "--parts", "--require", "--ship", "--gates", "--pilot_k", "--long", "--")):
    if flag not in argv:
        return []
    i = argv.index(flag) + 1; out = []
    while i < len(argv) and argv[i] not in stop:
        out.append(argv[i]); i += 1
    return out


def manifest(out):
    """Audit (§257 G-D note 1): sha256 of every result file in <out> computed ON THE HOST -> <out>/host_shas.json (+ stdout); the landing compares each local unbuffered sha."""
    fs = sorted(f for f in os.listdir(out) if f.endswith(".json") and f != "host_shas.json") + [f for f in ("log.jsonl", "gates.log") if os.path.exists(os.path.join(out, f))]
    m = {f: hashlib.sha256(open(os.path.join(out, f), "rb").read()).hexdigest() for f in fs}
    for sub in ("gates",):
        if os.path.isdir(os.path.join(out, sub)):
            m.update({sub + "/" + f: hashlib.sha256(open(os.path.join(out, sub, f), "rb").read()).hexdigest() for f in sorted(os.listdir(os.path.join(out, sub)))})
    json.dump(m, open(os.path.join(out, "host_shas.json"), "w"), indent=1); print("HOST_SHAS", json.dumps(m), flush=True)


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
        long_skipped = False
        if multi(pre, "--long"):                                            # Note 257-2: run_long on the same model / dir
            kp, lds, nm = multi(pre, "--long")[0].split(":")
            if rc == 0:
                rc = subprocess.call([sys.executable, "-u", "-B", "-m", "boxes.inject_aw_eval", "--run", out, "--kprime_list", kp, "--long_ds", lds, "--long_name", nm,
                                      "--eval_bs", "250", "--device", dev])
            else:                                                           # Audit CODE_REVIEW MED-2: never a silent skip
                long_skipped = True; print("RUN_LONG SKIPPED: the parts eval failed (rc %d); %s not written" % (rc, nm), flush=True)
        if rc != 0 and not os.path.exists(out + "/eval.json"):             # MED-2: an explicit FAILED eval.json (the drain fetches; the landing sees FAILED, never 'done')
            json.dump({"FAILED": True, "eval_rc": rc, "long_skipped": long_skipped}, open(out + "/eval.json", "w"))
        json.dump({"eval_only": True, "model": mp, "model_sha256": s, "eval_rc": rc, "status": "OK" if rc == 0 else "FAILED", "long_skipped": long_skipped},
                  open(out + "/log.jsonl", "w")); open(out + "/log.jsonl", "a").write("\n")   # MED-2: the done-marker carries the eval's rc / status
        if "--gates" in pre:                                                # Audit (§256 G-D): the 5 build gates on THIS host before TRAIN; gates_pass.json only if all pass
            env = dict(os.environ, PILOT_INIT=mp, PILOT_INIT_SHA=msha, PILOT_K=multi(pre, "--pilot_k")[0], LATENT_ROOT=ROOT); parts = ("offbit", "offbit_d", "acts", "smoke", "gradflow")
            grc = subprocess.call([sys.executable, "-B", "scratch/gate_s257.py"] + list(parts), env=env, stdout=open(out + "/gates.log", "w"), stderr=subprocess.STDOUT)
            os.makedirs(out + "/gates", exist_ok=True); gp = {}
            for p_ in parts:
                f = "scratch/gate_s257d/%s.json" % p_
                if os.path.exists(f):
                    shutil.copy(f, out + "/gates/%s.json" % p_); gp[p_] = json.load(open(f)).get("pass")
            if grc == 0 and all(gp.get(p_) for p_ in parts):
                json.dump({"pass": gp}, open(out + "/gates_pass.json", "w"))
        manifest(out); return rc
    ip, isha = multi(pre, "--init")[0].split("="); s = hashlib.sha256(open(ip, "rb").read()).hexdigest(); assert s.startswith(isha), ("init sha", ip, s)
    for f in multi(pre, "--require"):
        assert os.path.exists(f), ("required eval-only output missing", f)
    rc = subprocess.call([sys.executable, "-u", "-B", "-m", "boxes.inject_aw"] + post + ["--device", dev, "--out", out]); manifest(out); return rc


if __name__ == "__main__":
    sys.exit(main())
