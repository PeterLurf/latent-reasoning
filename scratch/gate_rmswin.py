"""Gates for §193 RMS x WINDOW (Theory 8433d73f): s193_rmswin_s1 = s182_window_s1's args (--k_window 1,4 --kmatch_impl masked, think_k 12, fast_add, aux 1.0
to 48k, seed 1) + --add_norm rms (nested170). No code change for this cell.
  (a) flags-off: nested170 / deltamem / deltas5 / halt173 equal git HEAD (byte-identical), so the flag-off path is the gated one
  (b) combined construct + 1 CPU step (both flags, --time_steps 1): rc 0, args carry add_norm rms + k_window 1,4 + kmatch_impl masked; on a fixture through
      kmatch_aux_loss (masked, items at mixed K in n + 1..4), every additive read's post-gain per-head norm is constant across tokens / items (<= 1e-3 rel)
      on BOTH the write path (read_fixed inside m.write) and the aux reads (fwd_read inside m.aux_logits), zero-state reads (pass 1) excluded
  (c) gradflow on the combined model through kmatch_aux_loss: answer CE alone and aux CE alone each give nonzero grads on mem.add_gain and on the
      additive heads' q / k / v rows of mem.qkv.weight (and the delta heads' q rows)
  (d) memory: s182_window_s1's cloud check (micro 2, 12.77 GiB) x the chunk-4 peak ratio (--add_norm rms vs off, WINDOW args, --mem_check --bs 64
      --micro 16 --eval_bs 8) on argv[1] (default cuda:0); projection < 21 GiB
  (e) smoke_plain (run separately; SMOKE_PLAIN_RESULT)
    python scratch/gate_rmswin.py [cuda device]   -> boxes/runs_fsq/gate_rmswin/gate.json"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

torch.set_num_threads(3)
import boxes.deltamem as DM  # noqa: E402
import boxes.nested170 as N  # noqa: E402
from boxes import s5  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "gate_rmswin"); OUT = {}; SP = os.environ.get("GATE_TMP", os.path.join(GD, "tmp"))
WIN = ["--arm", "inject", "--think_k", "12", "--aux_step", "1.0", "--aux_fade", "48000,48000", "--fast_add", "--k_window", "1,4", "--kmatch_impl", "masked", "--seed", "1", "--steps", "48000"]


def dump():
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate.json"), "w", newline="\n"), indent=1, default=str)


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    OUT["a"] = {f: subprocess.run(["git", "diff", "--quiet", "HEAD", "--", f], cwd=ROOT).returncode == 0 for f in ("boxes/nested170.py", "boxes/deltamem.py", "boxes/deltas5.py", "boxes/halt173.py")}
    OUT["a_pass"] = all(OUT["a"].values()); dump()
    shutil.rmtree(SP, ignore_errors=True); os.makedirs(SP, exist_ok=True)
    cmd = [sys.executable, "-m", "boxes.nested170"] + WIN + ["--add_norm", "rms", "--bs", "4", "--micro", "2", "--per_n", "2", "--serial_pool", "40", "--time_steps", "1", "--device", "cpu", "--out", os.path.join(SP, "c")]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="3", PYTHONPATH=ROOT))
    a_ = json.load(open(os.path.join(SP, "c", "args.json"))) if os.path.exists(os.path.join(SP, "c", "args.json")) else {}
    DM.FAST_ADD = True; g = s5.load_gpt2("cpu"); torch.manual_seed(1); m = N.InjectN(g, 6, "top", 6, 6, "rms")
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0); m.mem.gamma.fill_(0.5); m.mem.add_gain.uniform_(0.5, 1.5)
    TE = N.test_set(1000); its = [it for n_ in (3, 5, 7) for it in [x for x in TE if x["n"] == n_][:2]]; ks = [it["n"] + 1 + (i % 4) for i, it in enumerate(its)]
    rec = {"write": [], "aux": []}; cur = {"p": None}; orig_add, orig_write, orig_aux = DM.add_read, m.write, m.aux_logits

    def add_read_rec(mem, pq, Sa, z=None):
        oa = orig_add(mem, pq, Sa, z)
        if cur["p"] is not None:
            nz = Sa.detach().abs().flatten(1).sum(1) > 0                   # zero-state reads (pass 1) are 0 by construction: excluded
            v = (torch.exp(mem.s_add.detach()).view(1, 1, -1, 1) * oa.detach() if oa.dim() == 4 else torch.exp(mem.s_add.detach()).view(1, -1, 1) * oa.detach()).norm(dim=-1)
            if bool(nz.any()):
                rec[cur["p"]].append(v[nz].reshape(-1, v.shape[-1]))
        return oa

    def write_rec(*a, **k):
        cur["p"] = "write"; o = orig_write(*a, **k); cur["p"] = None; return o

    def aux_rec(*a, **k):
        cur["p"] = "aux"; o = orig_aux(*a, **k); cur["p"] = None; return o
    DM.add_read = add_read_rec; m.write = write_rec; m.aux_logits = aux_rec
    try:
        with torch.no_grad():
            N.kmatch_aux_loss(m, its, ks, "cpu", lambda: torch.autocast("cpu", enabled=False))
    finally:
        DM.add_read = orig_add; m.write = orig_write; m.aux_logits = orig_aux
    spread = {}
    for p_, L in rec.items():
        R = torch.cat(L) if L else torch.zeros(0, 6); spread[p_] = {"n_reads": int(R.shape[0]), "max_rel_spread_per_head": float(((R.max(0).values - R.min(0).values) / R.mean(0)).max()) if len(R) else None,
                                                                      "per_head_norm": R.mean(0).tolist() if len(R) else None}
    OUT["b"] = {"rc": r.returncode, "err": r.stderr[-600:] if r.returncode else "", "args": {k: a_.get(k) for k in ("add_norm", "k_window", "kmatch_impl", "think_k", "fast_add", "aux_step")},
                "fixture_K": ks, "fixture_n": [it["n"] for it in its], "rms_read_norm": spread}
    OUT["b_pass"] = (r.returncode == 0 and a_.get("add_norm") == "rms" and a_.get("k_window") == "1,4" and a_.get("kmatch_impl") == "masked"
                     and all(v["n_reads"] > 0 and v["max_rel_spread_per_head"] <= 1e-3 for v in spread.values())); dump()
    gr = {}
    for term in ("answer", "aux"):
        m.zero_grad(set_to_none=True); la, ax, hx, _ = N.kmatch_aux_loss(m, its, ks, "cpu", lambda: torch.autocast("cpu", enabled=False))
        (la if term == "answer" else ax).backward(); W = m.mem.qkv.weight.grad; D_ = W.shape[1]
        gr[term] = {"add_gain": float(m.mem.add_gain.grad.abs().sum()), "q_add": float(W[:384].abs().sum()), "k_add": float(W[D_:D_ + 384].abs().sum()),
                    "v_add": float(W[2 * D_:2 * D_ + 384].abs().sum()), "q_del": float(W[384:768].abs().sum()), "loss": float(la if term == "answer" else ax)}
    OUT["c"] = gr; OUT["c_pass"] = all(v > 0 for t_ in gr.values() for k_, v in t_.items() if k_ != "loss"); dump()
    pk = {}
    for tag, ex in (("off", []), ("rms", ["--add_norm", "rms"])):
        o_ = os.path.join(SP, "mem_" + tag)
        rr = subprocess.run([sys.executable, "-m", "boxes.nested170"] + WIN + ex + ["--bs", "64", "--micro", "16", "--eval_bs", "8", "--mem_check", "--device", dev, "--out", o_],
                            capture_output=True, text=True, cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT))
        pk[tag] = json.load(open(os.path.join(o_, "mem_check.json"))) if rr.returncode == 0 and os.path.exists(os.path.join(o_, "mem_check.json")) else {"err": rr.stderr[-600:]}
    mc = json.load(open("boxes/runs_fsq/s182_window_s1/mem_check.json")); ratio = (pk["rms"]["peak_gib"] / pk["off"]["peak_gib"]) if all("peak_gib" in pk[t] for t in pk) else None
    OUT["d"] = {"chunk4": pk, "ratio_rms_over_off": ratio, "window_s1_cloud": {"micro": mc["micro"], "peak_gib": mc["peak_gib"]}, "gpu": torch.cuda.get_device_name(dev),
                "proj_3090_micro2_gib": mc["peak_gib"] * ratio if ratio else None, "note": "the cell runs --auto_micro (budget 21 GiB): its own cloud check decides micro"}
    OUT["d_pass"] = bool(ratio) and OUT["d"]["proj_3090_micro2_gib"] < 21.0; dump()
    OUT["e"] = {"smoke_plain": os.environ.get("SMOKE_PLAIN_RESULT", "not run")}; OUT["e_pass"] = "PASS" in OUT["e"]["smoke_plain"]
    shutil.rmtree(SP, ignore_errors=True)
    OUT["all_pass"] = bool(all(OUT.get(k + "_pass") for k in "abcde")); dump(); print(json.dumps(OUT, default=str)[:3000])


if __name__ == "__main__":
    main()
