"""THEORY §188 REWEIGHT x WINDOW gates (5b2ebd2e): boxes/nested170.kmatch_aux_loss(reweight=) + the --aux_reweight / --aux_hold guard + the step-1 tripwire.
  (a) fp64, a fixed batch (n 3..8, every K_i >= n + 1): the window path's reweighted aux == loss_items' reweighted aux (the §184 form, recomputed here
      verbatim: fixed passes, AUX_W[t-1] x token-mean CE of each real step inside the per-item mean, then the item mean) <= 1e-12; also unweighted
  (b) flags off: kmatch_aux_loss (reweight False; hold False / True) == the pre-change module's, bitwise (CE, aux, hold_aux, S), fp32
  (c) gradflow: the reweighted aux gives a nonzero gradient on every memory write parameter, different from the unweighted aux's
  (d) the tripwire: a 1-step --time_steps run with --k_window 1,4 --aux_reweight writes s188_tripwire.json with mean kept weight != 1.0; and
      --aux_reweight with --aux_hold is refused
    python scratch/gate188.py <pre-188 nested170 copy>   -> boxes/runs_fsq/s188_gates/gate188.json"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from contextlib import nullcontext

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

torch.set_num_threads(3)
import boxes.nested170 as N  # noqa: E402
from boxes import s5  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "s188_gates"); OUT = {}; SP = os.environ.get("GATE188_TMP", os.path.join(GD, "tmp"))


def model():
    g = s5.load_gpt2("cpu"); torch.manual_seed(1); m = N.InjectN(g, 6, "top", 6, 6, None)
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0); m.mem.gamma.fill_(0.5)
    return m


def items():
    TE = N.test_set(1000); return [[it for it in TE if it["n"] == n][0] for n in range(3, 9)]


def ref_aux(m, its, reweight):
    """loss_items' §175 / §184 aux (nested170 l.~497-505), verbatim."""
    x, msk = m.think_inputs(its, "cpu"); per = [[] for _ in its]; S = m.mem.zero_state(len(its), "cpu"); K = max(len(N.aux_of(it)) for it in its)
    for t in range(1, K + 1):
        S = m.write(S, x, msk); idx = [i for i, it in enumerate(its) if len(N.aux_of(it)) >= t]
        if idx:
            lg_a, y_a = m.aux_logits(S[idx], [its[i] for i in idx], t, "cpu")
            ce = Fn.cross_entropy(lg_a.transpose(1, 2), y_a, ignore_index=-100, reduction="none").sum(1) / (y_a != -100).sum(1)
            for r_, i in enumerate(idx):
                per[i].append(ce[r_] * N.AUX_W[t - 1] if reweight else ce[r_])
    return torch.stack([torch.stack(p_).mean() for p_ in per if p_]).mean()


def main():
    its = items(); kit = [it["n"] + 1 + (i % 4) for i, it in enumerate(its)]
    _f = torch.Tensor.float; torch.Tensor.float = lambda self, *a, **k: self.double()
    try:                                                                    # (a) fp64
        m = model().double()
        with torch.no_grad():
            r = {}
            for rw in (False, True):
                _, aw, _, _ = N.kmatch_aux_loss(m, its, kit, "cpu", nullcontext, False, rw); ar = ref_aux(m, its, rw)
                r["reweight%d" % rw] = {"window": float(aw), "loss_items": float(ar), "rel": float(abs(aw - ar) / abs(ar))}
        OUT["a"] = r; OUT["a_pass"] = all(v["rel"] <= 1e-12 for v in r.values())
    finally:
        torch.Tensor.float = _f
    spec = importlib.util.spec_from_file_location("pre_n", sys.argv[1]); PRE = importlib.util.module_from_spec(spec); spec.loader.exec_module(PRE)   # (b)
    m = model(); b = {}
    with torch.no_grad():
        for hold in (False, True):
            new = N.kmatch_aux_loss(m, its, kit, "cpu", nullcontext, hold); old = PRE.kmatch_aux_loss(m, its, kit, "cpu", nullcontext, hold)
            b["hold%d" % hold] = all(torch.equal(p, q) for p, q in zip(new, old))
    OUT["b"] = b; OUT["b_pass"] = all(b.values())
    gr = {}                                                                 # (c)
    for rw in (False, True):
        m.zero_grad(set_to_none=True); _, aw, _, _ = N.kmatch_aux_loss(m, its, kit, "cpu", nullcontext, False, rw); aw.backward()
        gr[rw] = {n_: p_.grad.detach().clone() for n_, p_ in m.mem.named_parameters() if p_.grad is not None}
    names = sorted(gr[True]); OUT["c"] = {"n_mem_params_with_grad": len(names), "all_nonzero": all(float(gr[True][k].abs().sum()) > 0 for k in names),
                                         "differs_from_unweighted": any(not torch.equal(gr[True][k], gr[False][k]) for k in names)}
    OUT["c_pass"] = OUT["c"]["all_nonzero"] and OUT["c"]["differs_from_unweighted"] and len(names) > 0
    shutil.rmtree(SP, ignore_errors=True); os.makedirs(SP, exist_ok=True); env = dict(os.environ, OMP_NUM_THREADS="3", PYTHONPATH=ROOT)   # (d)
    base = [sys.executable, "-m", "boxes.nested170", "--arm", "inject", "--think_k", "12", "--aux_step", "1.0", "--aux_fade", "48000,48000", "--k_window", "1,4",
            "--kmatch_impl", "masked", "--fast_add", "--seed", "1", "--steps", "48000", "--bs", "8", "--micro", "2", "--per_n", "2", "--serial_pool", "40", "--device", "cpu"]
    r1 = subprocess.run(base + ["--aux_reweight", "--time_steps", "1", "--out", os.path.join(SP, "rw")], capture_output=True, text=True, env=env, cwd=ROOT)
    tw = json.load(open(os.path.join(SP, "rw", "s188_tripwire.json"))) if os.path.exists(os.path.join(SP, "rw", "s188_tripwire.json")) else None
    r2 = subprocess.run(base + ["--aux_reweight", "--aux_hold", "--time_steps", "1", "--out", os.path.join(SP, "hold")], capture_output=True, text=True, env=env, cwd=ROOT)
    OUT["d"] = {"rc": r1.returncode, "tripwire": tw, "err": r1.stderr[-400:] if r1.returncode else "", "hold_refused": r2.returncode != 0 and "§188" in r2.stderr}
    OUT["d_pass"] = r1.returncode == 0 and tw is not None and tw["aux_w_mean_kept"] not in (None, 1.0) and OUT["d"]["hold_refused"]
    shutil.rmtree(SP, ignore_errors=True)
    OUT["all_pass"] = bool(OUT["a_pass"] and OUT["b_pass"] and OUT["c_pass"] and OUT["d_pass"])
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate188.json"), "w", newline="\n"), indent=1); print(json.dumps(OUT)[:2500])


if __name__ == "__main__":
    main()
