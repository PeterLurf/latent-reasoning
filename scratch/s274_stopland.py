"""s274 copy of scratch/s268_stopland.py: eval boxes.inject_so_eval (§271 N1 LOOP-SI stop at 4000); otherwise unchanged. Original doc follows."""
"""§261 TARGET-CUE pilot: queue line after a --stop_at TRAIN line, IN the run's own dir (the runner appends --device <d> --out <dir>; <dir> = the run dir):
  <dir> scratch.s268_stopland --step 2000 --parts acc,ws,insert_ho
asserts <dir>/ckpt_last.pt is at --step, writes <dir>/model.pt (its model state) + ckpt_stop_sha256.json, runs boxes.inject_aw_eval --run <dir> --parts <...> --eval_bs 250
under the run's own args (cue target) -> <dir>/eval.json + eval_<part>.json; the training log.jsonl is left as is (no stub: model.pt + eval.json make the dir a gen10 RUN,
so the drain fetches it whole except ckpt_last / ckpt_prev); host_shas.json (s257_guarded.manifest). Exit = the eval's rc."""
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
import scratch.s257_guarded as SG  # noqa: E402


def main():
    import torch
    argv = sys.argv[1:]; dev = SG.take(argv, "--device"); out = SG.take(argv, "--out"); st = int(SG.take(argv, "--step")); parts = SG.take(argv, "--parts")
    a = json.load(open(out + "/args.json")); assert a.get("stop_at") == st and a["steps"] > st, ("stopland: the run's stop_at / steps", a.get("stop_at"), a["steps"], st)
    cp = out + "/ckpt_last.pt"; ck = torch.load(cp, map_location="cpu", weights_only=False); assert ck["step"] == st, ("ckpt step", ck["step"], st)
    assert not os.path.exists(out + "/model.pt") and not os.path.exists(out + "/eval.json"), "stopland: the run dir already holds model.pt / eval.json"
    torch.save(ck["model"], out + "/model.pt"); del ck
    json.dump({"ckpt": cp, "step": st, "ckpt_sha256": hashlib.sha256(open(cp, "rb").read()).hexdigest()}, open(out + "/ckpt_stop_sha256.json", "w"))
    rc = subprocess.call([sys.executable, "-u", "-B", "-m", "boxes.inject_so_eval", "--run", out, "--parts", parts, "--eval_bs", "250", "--device", dev])
    SG.manifest(out); return rc


if __name__ == "__main__":
    sys.exit(main())
