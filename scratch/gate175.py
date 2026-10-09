"""THEORY §175 STEP-AUX pre-launch gates (Theory 8965745e), boxes/nested170.py --aux_step:
  (0) the aux targets: on 5,000 random roots (n ~ U{1..16}) aux_steps' symbols are cot_steps[1:] and the marked index holds the t-th reduction's value
  (1) gradflow (fade 0.5, W_o perturbed, gamma 0.5, K = 3), from the aux CE ALONE (passes 1..3): every DeltaMem write parameter (ln_w, qkv's k / v rows,
      gate) gets gradient via M_t, and so does the shared trunk (block 0's c_attn); the memory after pass 1 gets gradient
  (2) the aux decoder writes nothing (S bitwise unchanged after aux_logits / aux_greedy) and sees no problem text: the embeddings it is fed are exactly
      [EOT, " Step", " t", " :"] + e_{t+1}[:-1] (teacher forcing), nothing else
  (3) lambda = 0 is bitwise nested170: 3 training steps with --aux_step 0 (and with no flag) give the same log losses and the same final weights as the
      committed HEAD's nested170 (a temporary copy boxes/_nested170_head_tmp.py, removed after)
    python scratch/gate175.py   -> boxes/runs_fsq/s175_gates/gate175.json"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.deltas5 as D5  # noqa: E402
from boxes import s5  # noqa: E402


def main():
    torch.set_num_threads(3); out = {}; ok = True; rng = np.random.default_rng(17500); bad = 0
    for _ in range(5000):
        r, t = N.generate(rng, int(rng.integers(1, 17))); st = N.cot_steps(r); ax = N.aux_steps(r); rv = N.reductions(r)
        bad += not ([s_ for s_, _ in ax] == st[1:] and all(s_[i] == " %d" % v for (s_, i), v in zip(ax, rv)))
    out["g0_bad"] = bad; out["g0_pass"] = bad == 0; ok &= out["g0_pass"]
    torch.manual_seed(0); g = s5.load_gpt2("cpu"); m = N.InjectN(g, 6, "top"); its = N.train_items(np.random.default_rng(1), 6, 8, set())
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02)
    m.mem.fade.fill_(0.5); m.mem.gamma.fill_(0.5); m.mem.s_add.data.fill_(-7.0); m.train(); m.zero_grad(set_to_none=True)
    x, msk = m.think_inputs(its, "cpu"); S = m.mem.zero_state(6, "cpu"); tot = 0; S1 = None
    for t in range(1, 4):
        S = m.write(S, x, msk)
        if t == 1:
            S.retain_grad(); S1 = S
        idx = [i for i, it in enumerate(its) if len(N.aux_of(it)) >= t]; lg, y = m.aux_logits(S[idx], [its[i] for i in idx], t, "cpu")
        tot = tot + Fn.cross_entropy(lg.transpose(1, 2), y, ignore_index=-100)
    tot.backward(); D = 768
    w = {"ln_w.weight": m.mem.ln_w.weight.grad, "ln_w.bias": m.mem.ln_w.bias.grad, "qkv.weight[k,v]": m.mem.qkv.weight.grad[D:], "qkv.bias[k,v]": m.mem.qkv.bias.grad[D:],
         "gate.weight": m.mem.gate.weight.grad, "gate.bias": m.mem.gate.bias.grad}
    out["g1"] = {k: (float(v.norm()) if v is not None else 0.0) for k, v in w.items()}
    out["g1"]["trunk_h0_c_attn"] = float(g.w("h.0.attn.c_attn.weight").grad.norm()) if g.w("h.0.attn.c_attn.weight").grad is not None else 0.0
    out["g1"]["S_after_pass1"] = float(S1.grad.norm()) if S1.grad is not None else 0.0
    out["g1_pass"] = min(out["g1"].values()) > 0; ok &= out["g1_pass"]; m.zero_grad(set_to_none=True)
    # (2)
    m.eval(); seen = []; f0 = D5.fwd_read

    def fr(g_, x_, mem, S_):
        seen.append(x_.detach().clone()); return f0(g_, x_, mem, S_)
    D5.fwd_read = fr
    try:
        with torch.no_grad():
            S = m.write(m.mem.zero_state(6, "cpu"), x, msk); Sc = S.clone(); lg, y = m.aux_logits(S, its, 1, "cpu"); m.aux_greedy(S, 1, 3)
    finally:
        D5.fwd_read = f0
    exp_ = [[N.EOT, N.STEP_ID, N.pass_tok(1), N.COLON_ID] + N.ids(N.aux_of(it)[0][0]) for it in its]; L_ = max(len(e) for e in exp_)
    exp_ids = torch.tensor([e + [N.EOT] * (L_ - len(e)) for e in exp_]); fed_ok = bool(torch.equal(seen[0], g.embed(exp_ids)))
    pre_ok = all(bool(torch.equal(s_[:, :4], g.embed(torch.tensor([[N.EOT, N.STEP_ID, N.pass_tok(1), N.COLON_ID]] * 6)))) for s_ in seen[1:])
    out["g2"] = {"S_unchanged": bool(torch.equal(S, Sc)), "teacher_forced_input_exact": fed_ok, "greedy_prefix_exact": pre_ok}
    out["g2_pass"] = all(out["g2"].values()); ok &= out["g2_pass"]
    # (3)
    head = subprocess.run(["git", "show", "HEAD:boxes/nested170.py"], capture_output=True, text=True, encoding="utf-8").stdout
    assert "def aux_steps" not in head, "commit the §175 nested170 only after this gate"
    tmp = os.path.join(ROOT, "boxes", "_nested170_head_tmp.py"); open(tmp, "w", encoding="utf-8", newline="\n").write(head); res = {}
    sp = os.environ.get("GATE175_OUT", os.path.join(ROOT, "boxes", "runs_fsq", "s175_gates"))
    try:
        common = ["--arm", "inject", "--think_k", "2", "--seed", "1", "--steps", "3", "--sched_scale", "0.002", "--log_every", "1", "--bs", "4", "--per_n", "2",
                  "--eval_bs", "8", "--serial_pool", "40", "--device", "cpu"]
        for nm, mod, extra in (("head", "boxes._nested170_head_tmp", []), ("aux0", "boxes.nested170", ["--aux_step", "0"]), ("noflag", "boxes.nested170", [])):
            od = os.path.join(sp, "lam0_" + nm); r_ = subprocess.run([sys.executable, "-m", mod] + common + ["--out", od], capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="3"))
            assert r_.returncode == 0, r_.stderr[-2000:]
            res[nm] = ([json.loads(l)["loss"] for l in open(os.path.join(od, "log.jsonl")) if '"loss"' in l], torch.load(os.path.join(od, "model.pt"), weights_only=True))
    finally:
        os.remove(tmp)
    same = {nm: bool(res[nm][0] == res["head"][0] and all(torch.equal(res[nm][1][k], res["head"][1][k]) for k in res["head"][1])) for nm in ("aux0", "noflag")}
    out["g3"] = {"losses_head": res["head"][0], "bitwise_equal": same}; out["g3_pass"] = all(same.values()); ok &= out["g3_pass"]
    out["all_pass"] = bool(ok); gd = os.path.join(ROOT, "boxes", "runs_fsq", "s175_gates"); os.makedirs(gd, exist_ok=True)
    json.dump(out, open(os.path.join(gd, "gate175.json"), "w", newline="\n"), indent=1); print("GATE175", json.dumps(out), flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
