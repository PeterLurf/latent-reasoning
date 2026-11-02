"""§258 S2 (LB2_v1 formation rule: WS read at 6k): queue line (the runner appends --device <d> --out <dir>):
  <dir> scratch.s258s2_ws6k --run boxes/runs_fsq/p2m_s1 --step 6000
asserts <run>/ckpt_last.pt is at --step, writes <dir>/model.pt (its model state) + <dir>/args.json (the run's) + ckpt_6k_sha256.json, runs boxes.inject_aw_eval
--parts acc,ws on it (eval_bs 250) -> <dir>/eval_acc.json / eval_ws.json; an eval-only log.jsonl stub; host_shas.json (s257_guarded.manifest). Exit = the eval's rc."""
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
import scratch.s257_guarded as SG  # noqa: E402


def main():
    import torch
    argv = sys.argv[1:]; dev = SG.take(argv, "--device"); out = SG.take(argv, "--out"); run = argv[argv.index("--run") + 1]; st = int(argv[argv.index("--step") + 1])
    cp = run + "/ckpt_last.pt"; ck = torch.load(cp, map_location="cpu", weights_only=False); assert ck["step"] == st, ("ckpt step", ck["step"], st)
    os.makedirs(out, exist_ok=True); torch.save(ck["model"], out + "/model.pt"); shutil.copy(run + "/args.json", out + "/args.json")
    json.dump({"ckpt": cp, "step": st, "ckpt_sha256": hashlib.sha256(open(cp, "rb").read()).hexdigest()}, open(out + "/ckpt_6k_sha256.json", "w")); del ck
    rc = subprocess.call([sys.executable, "-u", "-B", "-m", "boxes.inject_aw_eval", "--run", out, "--parts", "acc,ws", "--eval_bs", "250", "--device", dev])
    open(out + "/log.jsonl", "w").write(json.dumps({"eval_only": True, "from_ckpt_step": st}) + "\n"); SG.manifest(out); return rc


if __name__ == "__main__":
    sys.exit(main())
