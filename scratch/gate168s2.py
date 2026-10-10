"""THEORY §168 step 2 (note 168-7) pre-launch gates for boxes/prosqa168s2.py (CPU):
  (1) fixture: stepaux's aux targets at pass t are ids(" " + steps[t-1]) + EOT for every t <= depth (and t > depth is refused)
  (2) the aux decoder sees no question token: same memory, same steps, two different questions -> bitwise equal aux logits
  (3) gradflow (W_o perturbed, fade 1, gamma 0.5, K = 3): the aux CE at pass t alone gives nonzero gradient on every memory parameter and on S_t;
      the answer CE gives nonzero gradient on S_K and on the memory's read path
  (4) the three arms' CPU smokes (code paths): stepaux / currdirect / textcot run 4 steps + eval, stepaux writes every s168s2 field; currdirect's stage
      schedule is 12 x steps/12 with k = 0..10 then 10 (checked on the function)
  The memory check at the longest question (666 tokens) with K = 10 + the aux runs as the 3090 --time_steps run's --auto_micro.
    python scratch/gate168s2.py   -> boxes/runs_fsq/s168s2_gates/gate168s2.json"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import boxes.prosqa168s2 as P  # noqa: E402
from boxes import s5  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "s168s2_gates"); SP = os.environ.get("GATE168S2_TMP", os.path.join(GD, "tmp")); OUT = {}


def dump():
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate168s2.json"), "w", newline="\n"), indent=1)


def main():
    torch.set_num_threads(3); ok = True; D = P.load(os.path.join(ROOT, "prosqa", "p8data", "p8_na_train.json"))[:40]
    torch.manual_seed(1); m = P.ProsMem(s5.load_gpt2("cpu"))
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0); m.mem.gamma.fill_(0.5)
    its = D[:3]; S = torch.randn(3, 12, 64, 64) * 0.05
    # (1)
    g1 = {}
    with torch.no_grad():
        for t in range(1, 4):
            _, y = m.aux_logits(S, its, t, "cpu"); yy = y.numpy()
            g1["t%d" % t] = all(list(yy[r][yy[r] != -100]) == its[r]["step_ids"][t - 1] + [P.EOT] for r in range(3))
        try:
            m.aux_logits(S, [dict(its[0], depth=2)], 3, "cpu"); g1["refused_past_depth"] = False
        except AssertionError:
            g1["refused_past_depth"] = True
    OUT["g1"] = g1; OUT["g1_pass"] = all(g1.values()); ok &= OUT["g1_pass"]; dump()
    # (2)
    its2 = [dict(it, q=D[10 + r]["q"], pre=D[10 + r]["pre"]) for r, it in enumerate(its)]
    with torch.no_grad():
        l1, _ = m.aux_logits(S, its, 2, "cpu"); l2, _ = m.aux_logits(S, its2, 2, "cpu"); d1 = m.margins(S, its, "cpu"); d2 = m.margins(S, its2, "cpu")
    OUT["g2"] = {"aux_logits_equal_across_questions": bool(torch.equal(l1, l2)), "main_decoder_differs": bool(not np.allclose(d1, d2))}
    OUT["g2_pass"] = OUT["g2"]["aux_logits_equal_across_questions"] and OUT["g2"]["main_decoder_differs"]; ok &= OUT["g2_pass"]; dump()
    # (3)
    m.train(); x, msk = m.think_inputs(its, "cpu"); Sx = m.mem.zero_state(3, "cpu"); kept = []
    for t in range(1, 4):
        Sx = m.write(Sx, x, msk); Sx.retain_grad(); kept.append(Sx)
    g3 = {}
    for t in (1, 2, 3):
        m.zero_grad(set_to_none=True)
        for k_ in kept:
            k_.grad = None
        lg_a, y_a = m.aux_logits(kept[t - 1], its, t, "cpu"); Fn.cross_entropy(lg_a.transpose(1, 2), y_a, ignore_index=-100).backward(retain_graph=True)
        mz = [n_ for n_, p_ in m.mem.named_parameters() if p_.requires_grad and (p_.grad is None or float(p_.grad.norm()) == 0) and not n_.startswith("s_")]
        g3["aux_t%d" % t] = {"mem_zero_grad": mz, "S_t_grad": float(kept[t - 1].grad.norm()) if kept[t - 1].grad is not None else 0.0}
    m.zero_grad(set_to_none=True)
    for k_ in kept:
        k_.grad = None
    m.dec_loss(kept[-1], its, "cpu").backward()
    g3["answer"] = {"S_K_grad": float(kept[-1].grad.norm()) if kept[-1].grad is not None else 0.0, "S_1_grad": float(kept[0].grad.norm()) if kept[0].grad is not None else 0.0,
                    "wo_grad": float(m.mem.wo.weight.grad.norm()) if m.mem.wo.weight.grad is not None else 0.0}
    OUT["g3"] = g3; OUT["g3_pass"] = all(not v["mem_zero_grad"] and v["S_t_grad"] > 0 for k, v in g3.items() if k.startswith("aux")) and g3["answer"]["S_K_grad"] > 0 and g3["answer"]["wo_grad"] > 0
    ok &= OUT["g3_pass"]; dump()
    # (4)
    import types
    stages = [min(10, (s_ - 1) // (12000 // 12)) for s_ in (1, 1000, 1001, 10000, 10001, 11000, 11001, 12000)]
    OUT["g4"] = {"stage_samples": dict(zip(("1", "1000", "1001", "10000", "10001", "11000", "11001", "12000"), stages))}
    T = ["--train_n", "8", "--test_n", "6", "--extras_n", "4", "--probe_n_fit", "8", "--steps", "4", "--bs", "2", "--eval_bs", "2", "--log_every", "1", "--ckpt_every", "1000",
         "--device", "cpu", "--max_new", "30"]
    env = dict(os.environ, OMP_NUM_THREADS="3", PYTHONPATH=ROOT); rc = {}
    for arm, x_ in (("stepaux", ["--think_k", "2", "--fast_add"]), ("currdirect", []), ("textcot", [])):
        d_ = os.path.join(SP, arm); shutil.rmtree(d_, ignore_errors=True)
        r_ = subprocess.run([sys.executable, "-u", "-m", "boxes.prosqa168s2", "--arm", arm] + x_ + T + ["--out", d_], capture_output=True, text=True, env=env, cwd=ROOT)
        e_ = json.load(open(os.path.join(d_, "eval.json"))) if r_.returncode == 0 else {}
        rc[arm] = {"rc": r_.returncode, "err": r_.stderr[-400:] if r_.returncode else "", "fields": sorted(e_.get("s168s2", {})) if arm == "stepaux" else sorted(e_)[:6]}
        for f_ in ("model.pt",):
            os.path.exists(os.path.join(d_, f_)) and os.remove(os.path.join(d_, f_))
    need = {"acc_by_depth_kprime", "aux_exact_by_hop", "aux_acc_by_pass_ext", "delta_cos", "delta_cos_at_init", "aux_done", "path_indep", "step_recover"}
    OUT["g4"]["smokes"] = rc; OUT["g4_pass"] = all(v["rc"] == 0 for v in rc.values()) and need <= set(rc["stepaux"]["fields"]) and stages == [0, 0, 1, 9, 10, 10, 10, 10]
    ok &= OUT["g4_pass"]; OUT["all_pass"] = bool(ok); dump(); print("GATE168S2", json.dumps(OUT)[:2000], flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
