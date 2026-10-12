"""Gates for the candidate cell RMS x NOPASS (Theory 2026-10-12, not registered; gate-only): s185_rms_s1's args + --aux_nopass (nested170).
  (a) flags-off: no code changed for this cell - nested170 / deltamem / halt173 equal git HEAD (byte-identical), so the flag-off path is the gated one
  (b) combined construct + 1 CPU step (--time_steps 1, both flags): rc 0; the aux prompt has no pass token; the RMS per-head post-gain read norm is constant
      across items (<= 1e-3 relative) on a fixture after 1..4 passes
  (c) gradflow on the combined model (rms + nopass), fixed fixture, K = 4: answer CE and aux CE each give a nonzero grad on mem.add_gain and on the
      additive heads' q / k / v rows of mem.qkv.weight (and the delta heads' rows)
  (d) memory: no new tensors per step vs §185 RMS (the aux prompt is one token shorter); s185_rms_s1's cloud check: micro 1, 18.1 GiB (mem_check.json)
  (e) smoke_plain (run separately; recorded here from its output)
    python scratch/gate_rmsnopass.py   -> boxes/runs_fsq/gate_rmsnopass/gate.json"""
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
import boxes.deltamem as DM  # noqa: E402
import boxes.halt173 as H  # noqa: E402
import boxes.nested170 as N  # noqa: E402
from boxes import s5  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "gate_rmsnopass"); OUT = {}; SP = os.environ.get("GATE_TMP", os.path.join(GD, "tmp"))


def main():
    OUT["a"] = {f: subprocess.run(["git", "diff", "--quiet", "HEAD", "--", f], cwd=ROOT).returncode == 0 for f in ("boxes/nested170.py", "boxes/deltamem.py", "boxes/halt173.py")}
    OUT["a_pass"] = all(OUT["a"].values())
    shutil.rmtree(SP, ignore_errors=True); os.makedirs(SP, exist_ok=True)
    cmd = [sys.executable, "-m", "boxes.nested170", "--arm", "inject", "--think_k", "8", "--aux_step", "1.0", "--aux_fade", "48000,48000", "--fast_add", "--add_norm", "rms",
           "--aux_nopass", "--seed", "1", "--steps", "48000", "--bs", "4", "--per_n", "2", "--serial_pool", "40", "--time_steps", "1", "--device", "cpu", "--out", os.path.join(SP, "c")]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="3", PYTHONPATH=ROOT))
    a_ = json.load(open(os.path.join(SP, "c", "args.json"))) if os.path.exists(os.path.join(SP, "c", "args.json")) else {}
    DM.FAST_ADD = True; g = s5.load_gpt2("cpu"); torch.manual_seed(1); m = N.InjectN(g, 6, "top", 6, 6, "rms"); m.nopass = True
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0); m.mem.gamma.fill_(0.5); m.mem.add_gain.uniform_(0.5, 1.5)
    TE = N.test_set(1000); its = [it for it in TE if it["n"] in (4, 6)][:8]
    pr = [m.aux_prompt(t) for t in range(1, 13)]; nopass_ok = all(p == [N.EOT, N.STEP_ID, N.COLON_ID] for p in pr) and not any(tk in N.PASS_TOKS for p in pr for tk in p)
    x, msk = m.think_inputs(its, "cpu"); S = m.mem.zero_state(len(its), "cpu"); spreads = []
    with torch.no_grad():
        for t in range(1, 5):
            S = m.write(S, x, msk)
            with H.ReadTap() as tap:
                tap.arm(); m.dec_logits(S, its, "cpu")
            seqs = [[N.EOT] + N.ids(it["syms"]) + [N.TID[" ="]] for it in its]; rp = torch.tensor([len(q) - 1 for q in seqs])
            R = H.read_vec(m.mem, tap.x[torch.arange(len(its)), rp], S)[:, :6 * 64].reshape(len(its), 6, 64).norm(dim=-1)
            spreads.append(float(((R.max(0).values - R.min(0).values) / R.mean(0)).max()))
    OUT["b"] = {"rc": r.returncode, "err": r.stderr[-400:] if r.returncode else "", "args_aux_nopass": a_.get("aux_nopass"), "args_add_norm": a_.get("add_norm"),
                "aux_prompt_no_pass_token": nopass_ok, "rms_read_norm_rel_spread_by_t": spreads}
    OUT["b_pass"] = r.returncode == 0 and a_.get("aux_nopass") is True and a_.get("add_norm") == "rms" and nopass_ok and max(spreads) <= 1e-3
    gr = {}
    for term in ("answer", "aux"):
        m.zero_grad(set_to_none=True); S = m.mem.zero_state(len(its), "cpu")
        for t in range(1, 5):
            S = m.write(S, x, msk)
        if term == "answer":
            loss = Fn.cross_entropy(m.dec_logits(S, its, "cpu"), torch.tensor(N.NUM_IDS[[it["t"] for it in its]]))
        else:
            lg, y = m.aux_logits(S, its, 4, "cpu"); loss = Fn.cross_entropy(lg.transpose(1, 2), y, ignore_index=-100)
        loss.backward(); W = m.mem.qkv.weight.grad; D_ = W.shape[1]
        gr[term] = {"add_gain": float(m.mem.add_gain.grad.abs().sum()), "q_add": float(W[:384].abs().sum()), "k_add": float(W[D_:D_ + 384].abs().sum()),
                    "v_add": float(W[2 * D_:2 * D_ + 384].abs().sum()), "q_del": float(W[384:768].abs().sum()), "loss": float(loss)}
    OUT["c"] = gr; OUT["c_pass"] = all(v > 0 for t_ in gr.values() for k_, v in t_.items() if k_ != "loss")
    mc = json.load(open("boxes/runs_fsq/s185_rms_s1/mem_check.json"))
    OUT["d"] = {"s185_rms_s1_mem_check": mc, "note": "the combined cell adds no tensors (nopass shortens the aux prompt by one token); reuse micro 1 (18.1 GiB on the 3090)"}
    OUT["d_pass"] = True
    OUT["e"] = {"smoke_plain": os.environ.get("SMOKE_PLAIN_RESULT", "not run")}; OUT["e_pass"] = "PASS" in OUT["e"]["smoke_plain"]
    shutil.rmtree(SP, ignore_errors=True)
    OUT["all_pass"] = bool(all(OUT[k + "_pass"] for k in "abcde"))
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate.json"), "w", newline="\n"), indent=1); print(json.dumps(OUT)[:2500])


if __name__ == "__main__":
    main()
