"""note 196-1 BUDGET-ANNEAL gates (nested170 --anneal 'nmax:K:until,...'), 1 CPU thread, inject + aux + rms at tiny scale, --steps 4 --time_steps 4:
  a  flags-off == the pre-change module (argv[1], git HEAD copy): log (minus sec) and ckpt_last model + opt bitwise
  b  transitions (in-process, N.main with train_items / InjectN.write wrapped): --anneal 2:1:2,3:2:4 --n_max_train 3 --think_k 2 -> steps 1-2 draw with
     n_max 2 and run 1 write per step, steps 3-4 draw with n_max 3 and run 2; stage records in the log at steps 1 and 3 (stage 0, 1)
  c  resume: 4 steps in one go == 2 steps + --resume (ckpt_last model + opt bitwise); the resumed run logs its stage (1) at step 3
  d  gradflow at each stage of the cell's schedule (4:4, 6:6, 8:8) on the rms model through kmatch_aux_loss with K = the stage's K on items n <= the
     stage's n_max: answer CE and aux CE each give nonzero mem and trunk grads
  e  memory: the last stage must equal the base args (asserted in nested170), so --auto_micro and the eval see the base cell's shape:
     s185_rms_s3's cloud check (micro 1, mem_check.json) applies unchanged
    python scratch/gate_anneal.py <pre nested170.py>   -> boxes/runs_fsq/gate_anneal/gate.json"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import torch  # noqa: E402

torch.set_num_threads(1)
import boxes.deltamem as DM  # noqa: E402
import boxes.nested170 as N  # noqa: E402
from boxes import s5  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "gate_anneal"); OUT = {}; SP = os.environ.get("GATE_TMP", os.path.join(ROOT, "scratch", "_gate_anneal_tmp"))
BASE = ["--arm", "inject", "--aux_step", "1.0", "--aux_fade", "48000,48000", "--fast_add", "--add_norm", "rms", "--seed", "3", "--bs", "4", "--per_n", "2",
        "--serial_pool", "40", "--log_every", "1", "--device", "cpu"]


def dump():
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate.json"), "w", newline="\n"), indent=1, default=str)


def run(mod, out, args, clean=True):
    if clean:
        shutil.rmtree(out, ignore_errors=True)
    code = "import torch,sys,runpy;torch.set_num_threads(1);sys.argv=['x']+sys.argv[1:];runpy.run_module('%s',run_name='__main__')" % mod
    r = subprocess.run([sys.executable, "-c", code] + args + ["--out", out], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": ROOT, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    assert r.returncode == 0, (mod, args, r.stderr[-2500:])
    assert os.path.exists(os.path.join(out, "ckpt_last.pt")), ("no ckpt_last", out, os.listdir(out), r.stdout[-1500:], r.stderr[-800:])
    return [json.loads(l) for l in open(os.path.join(out, "log.jsonl"))], torch.load(os.path.join(out, "ckpt_last.pt"), weights_only=False)


def teq(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(teq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(teq(x, y) for x, y in zip(a, b))
    return a == b


def nosec(L):
    return [{k: v for k, v in r.items() if k != "sec"} for r in L]


def main():
    os.makedirs(SP, exist_ok=True); pre = os.path.join(ROOT, "boxes", "_pre_an_nested170.py"); shutil.copy(sys.argv[1], pre)
    AN = ["--anneal", "2:1:2,3:2:4", "--n_max_train", "3", "--think_k", "2", "--steps", "4"]
    try:
        a4 = BASE + ["--think_k", "2", "--steps", "48000", "--time_steps", "4", "--ckpt_every", "2"]
        l_new, c_new = run("boxes.nested170", os.path.join(SP, "off_new"), a4); l_pre, c_pre = run("boxes._pre_an_nested170", os.path.join(SP, "off_pre"), a4)
        OUT["a"] = {"log_equal": nosec(l_new) == nosec(l_pre), "model_equal": teq(c_new["model"], c_pre["model"]), "opt_equal": teq(c_new["opt"], c_pre["opt"]),
                    "no_anneal_keys": not any("anneal_stage" in r for r in l_new)}
        OUT["a_pass"] = all(OUT["a"].values()); dump(); print("a", OUT["a_pass"], flush=True)
        ev = []; orig_ti, orig_w = N.train_items, N.InjectN.write

        def ti(rng, bs, n_max, excl):
            its = orig_ti(rng, bs, n_max, excl); ev.append(("draw", bs, n_max, max(it["n"] for it in its))); return its

        def wr(self, *a_, **k_):
            ev.append(("write",)); return orig_w(self, *a_, **k_)
        N.train_items, N.InjectN.write = ti, wr; ob = os.path.join(SP, "trans"); shutil.rmtree(ob, ignore_errors=True)
        argv0 = sys.argv
        try:
            sys.argv = ["x"] + BASE + AN + ["--time_steps", "4", "--ckpt_every", "2", "--out", ob]; N.main()
        finally:
            N.train_items, N.InjectN.write = orig_ti, orig_w; sys.argv = argv0
        draws = [i for i, e in enumerate(ev) if e[0] == "draw" and e[1] == 4]   # the training draws (bs 4); calibration draws 64
        per_step = [(ev[i][2], ev[i][3], sum(1 for e in ev[i + 1:(draws[j + 1] if j + 1 < len(draws) else len(ev))] if e[0] == "write")) for j, i in enumerate(draws)]
        lb = [json.loads(l) for l in open(os.path.join(ob, "log.jsonl"))]; st_recs = [(r["step"], r["anneal_stage"]) for r in lb if "anneal_stage" in r and "loss" not in r]
        OUT["b"] = {"per_step_(nmax_arg, max_n_drawn, writes)": per_step, "stage_records": st_recs, "logged_think_k": [(r["step"], r.get("think_k")) for r in lb if "loss" in r]}
        OUT["b_pass"] = (per_step[:2] == [(2, per_step[0][1], 1), (2, per_step[1][1], 1)] and all(p_[1] <= 2 for p_ in per_step[:2]) and len(per_step) == 4
                         and [p_[0] for p_ in per_step[2:]] == [3, 3] and [p_[2] for p_ in per_step[2:]] == [2, 2] and all(p_[1] <= 3 for p_ in per_step[2:])
                         and st_recs == [(1, 0), (3, 1)]); dump(); print("b", OUT["b_pass"], flush=True)
        AC = ["--anneal", "2:1:2,3:2:48000", "--n_max_train", "3", "--think_k", "2", "--steps", "48000"]   # c: last stage ends at --steps 48000 so ckpts land at steps 2 and 4
        g4 = BASE + AC + ["--time_steps", "4", "--ckpt_every", "2"]; lf, cf = run("boxes.nested170", os.path.join(SP, "an_full"), g4)
        _, ch = run("boxes.nested170", os.path.join(SP, "an_half"), BASE + AC + ["--time_steps", "2", "--ckpt_every", "2"])
        cp = os.path.join(SP, "an_ck2.pt"); torch.save(ch, cp); rd = os.path.join(SP, "an_res"); shutil.rmtree(rd, ignore_errors=True); os.makedirs(rd)
        for f_ in ("args.json", "s170_init.json"):
            if os.path.exists(os.path.join(SP, "an_half", f_)):
                shutil.copy(os.path.join(SP, "an_half", f_), rd)
        lr, cr = run("boxes.nested170", rd, BASE + AC + ["--time_steps", "2", "--ckpt_every", "2", "--resume", cp], clean=False)
        OUT["c"] = {"half_step": ch["step"], "resumed_step": cr["step"], "model_equal": teq(cr["model"], cf["model"]), "opt_equal": teq(cr["opt"], cf["opt"]),
                    "resumed_stage_records": [(r["step"], r["anneal_stage"]) for r in lr if "anneal_stage" in r and "loss" not in r]}
        OUT["c_pass"] = ch["step"] == 2 and cr["step"] == 4 and OUT["c"]["model_equal"] and OUT["c"]["opt_equal"] and OUT["c"]["resumed_stage_records"] == [(3, 1)]
        dump(); print("c", OUT["c_pass"], flush=True)
        DM.FAST_ADD = True; g = s5.load_gpt2("cpu"); torch.manual_seed(3); m = N.InjectN(g, 6, "top", 6, 6, "rms")
        with torch.no_grad():
            m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0); m.mem.gamma.fill_(0.5)
        TE = N.test_set(1000); d = {}
        for nm, kk in ((4, 4), (6, 6), (8, 8)):
            its = [it for n_ in (nm - 1, nm) for it in [x for x in TE if x["n"] == n_][:2]]; d[str(nm)] = {}
            for term in ("answer", "aux"):
                m.zero_grad(set_to_none=True); la, ax, hx, _ = N.kmatch_aux_loss(m, its, [kk] * len(its), "cpu", lambda: torch.autocast("cpu", enabled=False))
                (la if term == "answer" else ax).backward()
                d[str(nm)][term] = {"mem": float(sum(p.grad.abs().sum() for n, p in m.named_parameters() if n.startswith("mem.") and p.grad is not None)),
                                    "trunk": float(sum(p.grad.abs().sum() for n, p in m.named_parameters() if not n.startswith("mem.") and p.grad is not None))}
        OUT["d"] = d; OUT["d_pass"] = all(v > 0 for st in d.values() for t in st.values() for v in t.values()); dump(); print("d", OUT["d_pass"], flush=True)
        mc = json.load(open("boxes/runs_fsq/s185_rms_s3/mem_check.json"))
        OUT["e"] = {"s185_rms_s3_cloud": {"micro": mc.get("micro"), "peak_gib": mc.get("peak_gib")}, "note": "the last stage is asserted == the base args, so the cell's --auto_micro check and eval run the base shape; earlier stages are smaller (n_max, K <=)"}
        OUT["e_pass"] = mc.get("peak_gib", 99) < 21.0; dump()
    finally:
        os.remove(pre); shutil.rmtree(SP, ignore_errors=True)
        OUT["all_pass"] = bool(all(OUT.get(k + "_pass") for k in "abcde")); dump(); print(json.dumps(OUT, default=str)[:2500], flush=True)


if __name__ == "__main__":
    main()
