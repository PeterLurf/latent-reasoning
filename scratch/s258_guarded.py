"""§258 S1+ host wrapper for a training ENTRY SCRIPT (queue line; the runner appends --device <d> --out <dir>, passed through to the entry):
  <dir> scratch.s258_guarded --blob <path>=<git blob id> ... [--init <file>=<sha16>] --require <file> ... --entry <script.py> [--ship <files>] -- <entry flags>
asserts the git blob ids, the init's sha256 prefix (if given) and that every --require file exists (the host's gates_pass.json), then runs
python -u -B <entry> <entry flags> --device <d> --out <dir>; host_shas.json after (s257_guarded.manifest). Exit = the entry's rc."""
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
import scratch.s257_guarded as SG  # noqa: E402


def main():
    argv = sys.argv[1:]; dev = SG.take(argv, "--device"); out = SG.take(argv, "--out")
    pre, post = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    stop = ("--blob", "--init", "--require", "--entry", "--ship", "--")
    for x in SG.multi(pre, "--blob", stop):
        p, want = x.split("="); got = SG.blob_id(p); assert got == want, ("git blob", p, got, want)
    for x in SG.multi(pre, "--init", stop):
        ip, isha = x.split("="); s = hashlib.sha256(open(ip, "rb").read()).hexdigest(); assert s.startswith(isha), ("init sha", ip, s)
    for f in SG.multi(pre, "--require", stop):
        assert os.path.exists(f), ("required file missing", f)
    entry = SG.multi(pre, "--entry", stop)[0]
    rc = subprocess.call([sys.executable, "-u", "-B", entry] + post + ["--device", dev, "--out", out]); SG.manifest(out); return rc


if __name__ == "__main__":
    sys.exit(main())
