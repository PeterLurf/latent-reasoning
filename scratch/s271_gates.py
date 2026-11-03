"""§265 DENSE-DISTRACT host gates wrapper (a copy of scratch/s269_gates.py running scratch/gate_s271.py) (queue line; the runner appends --device <d> --out <dir>):
  <dir> scratch.s271_gates --blob <path>=<git blob id> ... --gates <p1,p2,...>
asserts the git blob ids, then scratch/gate_s271.py <parts> on THIS host (LATENT_ROOT = the host root) ->
<dir>/gates/<part>.json, <dir>/gates.log, <dir>/eval.json = {"gates": {part: pass}, "all_pass"} (the runner's done marker), stub args.json / log.jsonl and a placeholder
model.pt (the drain's s224 done marker / need list; deleted at landing), <dir>/host_shas.json (s257_guarded.manifest).
Exit 0 iff every part passes."""
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)


def main():
    argv = sys.argv[1:]; dev = argv[argv.index("--device") + 1]; out = argv[argv.index("--out") + 1]; i = argv.index("--blob") + 1; blobs = []
    while i < len(argv) and not argv[i].startswith("--"):
        blobs.append(argv[i]); i += 1
    for x in blobs:
        p, want = x.split("="); b = open(p, "rb").read(); got = hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest(); assert got == want, ("git blob", p, got, want)
    parts = argv[argv.index("--gates") + 1].split(","); os.makedirs(out + "/gates", exist_ok=True)
    for p_ in parts:                                                        # no stale part json (a shipped working tree) can stand in for this run's
        if os.path.exists("%s/%s.json" % (os.environ.get("GATE_OUTDIR", "scratch/gate_s271"), p_)):
            os.remove("%s/%s.json" % (os.environ.get("GATE_OUTDIR", "scratch/gate_s271"), p_))
    env = dict(os.environ, LATENT_ROOT=ROOT, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", dev.split(":")[-1]))
    GS = os.environ.get("GATE_SCRIPT", "scratch/gate_s271.py"); GD = os.environ.get("GATE_OUTDIR", "scratch/gate_s271"); rcs = {}; gp = {}   # v6: GATE_* (the fixture only)
    with open(out + "/gates.log", "w") as lf:                               # v6 (Audit 11-03 §3.3): one subprocess per part - a part's memory dies with it
        for p_ in parts:
            lf.write("=== part %s\n" % p_); lf.flush()
            rcs[p_] = subprocess.call([sys.executable, "-B", GS, p_], env=env, stdout=lf, stderr=subprocess.STDOUT)
            lf.write("=== part %s rc %d\n" % (p_, rcs[p_])); lf.flush()
    rc = max(rcs.values()) if rcs else 1
    for p_ in parts:
        f = "%s/%s.json" % (GD, p_)
        if os.path.exists(f):
            shutil.copy(f, out + "/gates/%s.json" % p_); gp[p_] = json.load(open(f)).get("pass")
    ok = rc == 0 and all(gp.get(p_) is True for p_ in parts); json.dump({"gates": gp, "all_pass": ok, "gate_rc": rc, "part_rc": rcs}, open(out + "/eval.json", "w"))
    if ok:                                                                  # S1+ train lines --require <dir>/gates_pass.json (written only when every part passed)
        json.dump({"pass": gp}, open(out + "/gates_pass.json", "w"))
    json.dump({"gates_only": True, "parts": parts}, open(out + "/args.json", "w")); open(out + "/log.jsonl", "w").write(json.dumps({"gates_only": True}) + "\n")
    import torch                                                            # the drain's done marker / need list (s224_done_marker: model.pt + eval.json + log.jsonl + args.json)
    torch.save({"placeholder": "gates-only dir, no model"}, out + "/model.pt")
    import scratch.s257_guarded as SG
    SG.manifest(out); return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
