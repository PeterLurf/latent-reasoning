"""note 196-1 probe gates (nested170 --probe_loss_every N), 1 CPU thread, inject + aux + rms at tiny scale, --steps 48000 --time_steps 4 --ckpt_every 2
(compared through ckpt_last.pt = step 4: model, optimiser, np_rng):
  a  flags-off == the pre-change module (argv[1], git HEAD copy): log (minus sec) and ckpt model + opt + np_rng bitwise
  b  flag on (--probe_loss_every 2) == flag off: ckpt model + opt + np_rng bitwise (the probe draws no training rng and leaves no state);
     log equal after dropping the probe records; probe records at steps 2 and 4 with finite probe_loss / probe_aux; one probe_items header
  c  with --anneal 2:1:2,3:2:48000 + probe: 4 steps == 2 + --resume (model + opt bitwise); the resumed run's probe_items_sha == the full run's and its
     step-4 probe_loss / probe_aux equal the full run's; probe records carry anneal_stage
    python scratch/gate_probe.py <pre nested170.py>   -> boxes/runs_fsq/gate_probe/gate.json"""
import json
import math
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import torch  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "gate_probe"); OUT = {}; SP = os.environ.get("GATE_TMP", os.path.join(ROOT, "scratch", "_gate_probe_tmp"))
BASE = ["--arm", "inject", "--aux_step", "1.0", "--aux_fade", "48000,48000", "--fast_add", "--add_norm", "rms", "--seed", "3", "--bs", "4", "--per_n", "2",
        "--serial_pool", "40", "--log_every", "1", "--eval_bs", "64", "--device", "cpu", "--steps", "48000", "--ckpt_every", "2"]


def dump():
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate.json"), "w", newline="\n"), indent=1, default=str)


def run(mod, out, args, clean=True):
    if clean:
        shutil.rmtree(out, ignore_errors=True)
    code = "import torch,sys,runpy;torch.set_num_threads(1);sys.argv=['x']+sys.argv[1:];runpy.run_module('%s',run_name='__main__')" % mod
    r = subprocess.run([sys.executable, "-c", code] + args + ["--out", out], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": ROOT, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    assert r.returncode == 0, (mod, args, r.stderr[-2500:])
    assert os.path.exists(os.path.join(out, "ckpt_last.pt")), ("no ckpt_last", out, os.listdir(out))
    return [json.loads(l) for l in open(os.path.join(out, "log.jsonl"))], torch.load(os.path.join(out, "ckpt_last.pt"), weights_only=False)


def teq(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(teq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(teq(x, y) for x, y in zip(a, b))
    return a == b


def train_log(L):
    return [{k: v for k, v in r.items() if k != "sec"} for r in L if "probe_loss" not in r and "probe_items" not in r]


def main():
    os.makedirs(SP, exist_ok=True); pre = os.path.join(ROOT, "boxes", "_pre_pr_nested170.py"); shutil.copy(sys.argv[1], pre)
    try:
        a4 = BASE + ["--think_k", "2", "--time_steps", "4"]
        l_new, c_new = run("boxes.nested170", os.path.join(SP, "off_new"), a4); l_pre, c_pre = run("boxes._pre_pr_nested170", os.path.join(SP, "off_pre"), a4)
        OUT["a"] = {"log_equal": train_log(l_new) == train_log(l_pre) and len(l_new) == len(l_pre), "model_equal": teq(c_new["model"], c_pre["model"]),
                    "opt_equal": teq(c_new["opt"], c_pre["opt"]), "np_rng_equal": c_new["np_rng"] == c_pre["np_rng"]}
        OUT["a_pass"] = all(OUT["a"].values()); dump(); print("a", OUT["a_pass"], flush=True)
        l_on, c_on = run("boxes.nested170", os.path.join(SP, "on"), a4 + ["--probe_loss_every", "2"])
        pr = [r for r in l_on if "probe_loss" in r]; hd = [r for r in l_on if "probe_items" in r]
        OUT["b"] = {"model_equal": teq(c_on["model"], c_new["model"]), "opt_equal": teq(c_on["opt"], c_new["opt"]), "np_rng_equal": c_on["np_rng"] == c_new["np_rng"],
                    "train_log_equal": train_log(l_on) == train_log(l_new), "probe_steps": [r["step"] for r in pr], "probe_finite": all(math.isfinite(r["probe_loss"]) and math.isfinite(r["probe_aux"]) for r in pr),
                    "header": hd, "probe_sample": pr}
        OUT["b_pass"] = (OUT["b"]["model_equal"] and OUT["b"]["opt_equal"] and OUT["b"]["np_rng_equal"] and OUT["b"]["train_log_equal"] and OUT["b"]["probe_steps"] == [2, 4]
                         and OUT["b"]["probe_finite"] and len(hd) == 1 and hd[0]["probe_items"] == 256); dump(); print("b", OUT["b_pass"], flush=True)
        AC = ["--anneal", "2:1:2,3:2:48000", "--n_max_train", "3", "--think_k", "2", "--probe_loss_every", "2"]
        lf, cf = run("boxes.nested170", os.path.join(SP, "an_full"), BASE + AC + ["--time_steps", "4"])
        lh, ch = run("boxes.nested170", os.path.join(SP, "an_half"), BASE + AC + ["--time_steps", "2"])
        cp = os.path.join(SP, "an_ck2.pt"); torch.save(ch, cp); rd = os.path.join(SP, "an_res"); shutil.rmtree(rd, ignore_errors=True); os.makedirs(rd)
        for f_ in ("args.json", "s170_init.json"):
            if os.path.exists(os.path.join(SP, "an_half", f_)):
                shutil.copy(os.path.join(SP, "an_half", f_), rd)
        lr, cr = run("boxes.nested170", rd, BASE + AC + ["--time_steps", "2", "--resume", cp], clean=False)
        sha = lambda L: [r["probe_items_sha"] for r in L if "probe_items_sha" in r]   # noqa: E731
        p4 = lambda L: [(r["probe_loss"], r["probe_aux"], r.get("anneal_stage")) for r in L if r.get("step") == 4 and "probe_loss" in r]   # noqa: E731
        OUT["c"] = {"model_equal": teq(cr["model"], cf["model"]), "opt_equal": teq(cr["opt"], cf["opt"]), "sha_full": sha(lf), "sha_resumed": sha(lr),
                    "probe4_full": p4(lf), "probe4_resumed": p4(lr), "stages_on_probe_records": [(r["step"], r.get("anneal_stage")) for r in lf if "probe_loss" in r]}
        OUT["c_pass"] = (OUT["c"]["model_equal"] and OUT["c"]["opt_equal"] and OUT["c"]["sha_full"] == OUT["c"]["sha_resumed"] and len(OUT["c"]["sha_full"]) == 1
                         and OUT["c"]["probe4_full"] == OUT["c"]["probe4_resumed"] and len(OUT["c"]["probe4_full"]) == 1 and OUT["c"]["stages_on_probe_records"] == [(2, 0), (4, 1)])
        dump(); print("c", OUT["c_pass"], flush=True)
    finally:
        os.remove(pre); shutil.rmtree(SP, ignore_errors=True)
        OUT["all_pass"] = bool(all(OUT.get(k + "_pass") for k in "abc")); dump(); print(json.dumps(OUT, default=str)[:2500], flush=True)


if __name__ == "__main__":
    main()
