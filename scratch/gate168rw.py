"""note 168-12 STEP-AUX-REWEIGHT gates (prosqa168s2 --aux_reweight), CPU fp32 -> boxes/runs_fsq/s168s2_gates/gate168rw.json:
  g1  the weight table = the train histogram's E_bar / E_t (asserted at import) and the full p8_na_train len(steps) histogram = HIST_168
  g2  flags off (1 CPU thread: multi-thread backward is nondeterministic): the post-change module == the pre-change module (git HEAD copy given as argv[1]) on a 3-step stepaux run (log losses, model.pt
      state_dict and eval.json bitwise, ignoring the new 'clip' / eval_sec)
  g3  gradient flow: the same 3 steps with --aux_reweight -> aux_ce logged differs at step 1 and the step-1 parameters differ from the flags-off run
      (the weights reach the gradient), every trainable parameter updated finite
  g4  eval.json clip {clip_rate, grad_norm_preclip median / p90, n_steps = 3} on both runs (s168s2.clip for stepaux)
    python scratch/gate168rw.py <pre-change prosqa168s2.py>"""
import json
import os
import shutil
import subprocess
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import torch  # noqa: E402

import boxes.prosqa168s2 as P  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "s168s2_gates"); OUT = {}; SP = os.environ.get("GATE_TMP", os.path.join(ROOT, "scratch", "_gate168rw_tmp"))
COMMON = ["--arm", "stepaux", "--think_k", "2", "--device", "cpu", "--steps", "3", "--log_every", "1", "--bs", "2", "--train_n", "40", "--test_n", "4",
          "--extras_n", "2", "--probe_n_fit", "8", "--eval_bs", "4", "--sched_scale", "0.0001", "--fast_add", "--no_normcap", "--ckpt_every", "0"]


def run(mod, out, extra=()):
    shutil.rmtree(out, ignore_errors=True)
    code = "import torch,sys,runpy;torch.set_num_threads(1);sys.argv=['x']+sys.argv[1:];runpy.run_module('%s',run_name='__main__')" % mod
    r = subprocess.run([sys.executable, "-c", code] + COMMON + list(extra) + ["--out", out], cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    assert r.returncode == 0, r.stderr[-2000:]
    return [json.loads(l) for l in open(os.path.join(out, "log.jsonl"))], json.load(open(os.path.join(out, "eval.json"))), torch.load(os.path.join(out, "model.pt"), weights_only=True)


def main():
    os.makedirs(SP, exist_ok=True); pre = os.path.join(ROOT, "boxes", "_pre168s2_gate.py"); shutil.copy(sys.argv[1], pre)
    try:
        D = P.load(os.path.join(ROOT, "prosqa", "p8data", "p8_na_train.json"))
        OUT["g1"] = {"table": list(P.AUX_W), "E": [round(e, 3) for e in P._E], "hist_equal": dict(Counter(it["depth"] for it in D)) == P.HIST_168}
        OUT["g1_pass"] = OUT["g1"]["hist_equal"]
        l_new, e_new, m_new = run("boxes.prosqa168s2", os.path.join(SP, "new"))
        l_pre, e_pre, m_pre = run("boxes._pre168s2_gate", os.path.join(SP, "pre"))
        strip = lambda e: {k: v for k, v in e.items() if k not in ("clip", "eval_sec", "train_sec")} | ({"s168s2": {k: v for k, v in e["s168s2"].items() if k != "clip"}} if "s168s2" in e else {})   # noqa: E731
        OUT["g2"] = {"log_equal": [{k: v for k, v in r.items() if k != "sec"} for r in l_new] == [{k: v for k, v in r.items() if k != "sec"} for r in l_pre],
                     "model_equal": set(m_new) == set(m_pre) and all(torch.equal(m_new[k], m_pre[k]) for k in m_new), "eval_equal": strip(e_new) == strip(e_pre)}
        OUT["g2_pass"] = all(OUT["g2"].values())
        l_rw, e_rw, m_rw = run("boxes.prosqa168s2", os.path.join(SP, "rw"), ["--aux_reweight"])
        a0 = [r.get("aux_ce") for r in l_new if "aux_ce" in r]; a1 = [r.get("aux_ce") for r in l_rw if "aux_ce" in r]
        diff = [k for k in m_new if not torch.equal(m_new[k], m_rw[k])]
        OUT["g3"] = {"aux_ce_off": a0, "aux_ce_rw": a1, "aux_ce_differs_step1": a0[0] != a1[0], "n_params_differ_after_3": len(diff), "n_params": len(m_new),
                     "all_finite": all(bool(torch.isfinite(v).all()) for v in m_rw.values()), "examples": diff[:5]}
        OUT["g3_pass"] = OUT["g3"]["aux_ce_differs_step1"] and len(diff) > 0 and OUT["g3"]["all_finite"]
        OUT["g4"] = {"off": e_new.get("clip"), "rw": e_rw.get("clip"), "s168s2_clip_equal": e_rw["s168s2"].get("clip") == e_rw.get("clip")}
        OUT["g4_pass"] = all((c or {}).get("grad_norm_preclip", {}).get("n_steps") == 3 for c in (e_new.get("clip"), e_rw.get("clip"))) and OUT["g4"]["s168s2_clip_equal"] and e_rw["clip"]["aux_reweight"]
    finally:
        os.remove(pre); shutil.rmtree(SP, ignore_errors=True)
    OUT["all_pass"] = bool(OUT.get("g1_pass") and OUT.get("g2_pass") and OUT.get("g3_pass") and OUT.get("g4_pass"))
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate168rw.json"), "w", newline="\n"), indent=1); print(json.dumps(OUT)[:2500], flush=True)


if __name__ == "__main__":
    main()
