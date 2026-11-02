"""§258 S0 host wrapper (queue line; the runner appends --device <d> --out <dir>):
  <dir> scratch.s258m_host --blob <path>=<git blob id> ... --gates <p1,p2,...>
asserts the git blob ids, then scratch/gate_s258m.py <parts> on THIS host (LATENT_ROOT = the host root; the medium safetensors fetched and sha-asserted by the loader) ->
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
    env = dict(os.environ, LATENT_ROOT=ROOT, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", dev.split(":")[-1]))
    rc = subprocess.call([sys.executable, "-B", "scratch/gate_s258m.py"] + parts, env=env, stdout=open(out + "/gates.log", "w"), stderr=subprocess.STDOUT); gp = {}
    for p_ in parts:
        f = "scratch/gate_s258m/%s.json" % p_
        if os.path.exists(f):
            shutil.copy(f, out + "/gates/%s.json" % p_); gp[p_] = json.load(open(f)).get("pass")
    ok = rc == 0 and all(gp.get(p_) is True for p_ in parts); json.dump({"gates": gp, "all_pass": ok, "gate_rc": rc}, open(out + "/eval.json", "w"))
    json.dump({"gates_only": True, "parts": parts}, open(out + "/args.json", "w")); open(out + "/log.jsonl", "w").write(json.dumps({"gates_only": True}) + "\n")
    import torch                                                            # the drain's done marker / need list (s224_done_marker: model.pt + eval.json + log.jsonl + args.json)
    torch.save({"placeholder": "gates-only dir, no model"}, out + "/model.pt")
    import scratch.s257_guarded as SG
    SG.manifest(out); return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
