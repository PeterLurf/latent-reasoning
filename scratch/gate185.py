"""THEORY §185 ADD-NORM pre-launch gates (2a1aac15; Theory's den extras) + §186's inject169 K' change (95e5dfbf) + note 185-1's halt173 --normcap:
  (1) flags off bitwise: deltamem read_fixed / write_seq vs the pre-185 copy (GPT-2 DeltaMem, FAST_ADD off / on, mask none / some); nested170 NOFADE path
      (inject K2 + aux) vs the pre-185 nested170 copy; inject169 --interleave 1 vs the pre-186 copy (logs + every eval field); smoke_plain
  (2) den: z == sum_t mask * phi(k_t) recomputed from the same projections (write_seq, FAST_ADD off and on)
  (3) den: ||read|| <= max_i ||v_i|| x 1.001 per head on a fixture (the den read is a convex combination of the written v's)
  (4) rms: each head's read has unit RMS over DH before the gain (gain = 1)
  (5) gradflow (den and rms): every memory parameter (incl. rms's add_gain) gets nonzero gradient from a decoder loss after 2 passes
  (6) den runtime asserts: S.shape[1] == h_add + h_del + 1; block 12 rows h_add..DH-1 exactly 0 after a training step and after eval; the delta read from
      the full S == from S[:, h_add:h_add + h_del] alone (ablate "add", bitwise)
  (7) static check: every S-slicing site in boxes/*.py, with its status (patched / not needed: module never runs add_norm)
  (8) cell smokes: nested170 --add_norm den / rms (s185 fields) + halt173 --normcap on the den smoke (loads, writes eval_s185_normcap.json)
    python scratch/gate185.py <pre-185 deltamem> <pre-185 nested170> <pre-186 inject169>   -> boxes/runs_fsq/s185_gates/gate185.json"""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.nested170 as N  # noqa: E402
from boxes import s5  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "s185_gates"); SP = os.environ.get("GATE185_TMP", os.path.join(GD, "tmp")); OUT = {}
BENCH = os.path.join(ROOT, "bench", "data", "mult_4x4_test.txt")


def dump():
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate185.json"), "w", newline="\n"), indent=1)


def run_many(jobs):
    procs = []
    for mod, args, out, script in jobs:
        shutil.rmtree(out, ignore_errors=True); env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=ROOT, NESTED170_PROBE_N_FIT="64")
        procs.append(subprocess.Popen([sys.executable, "-u"] + ([script] if script else ["-m", mod]) + args + ["--device", "cpu", "--out", out],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=ROOT))
    res = []
    for p_ in procs:
        o_, e_ = p_.communicate(); res.append((p_.returncode, (o_ or "")[-300:] + (e_ or "")[-800:]))
    return res


def flat(d, pre=""):
    o = {}
    for k, v in d.items():
        o.update(flat(v, pre + k + ".")) if isinstance(v, dict) else o.__setitem__(pre + k, v)
    return o


def cmp(dn, do):
    Ln = [json.loads(x).get("loss") for x in open(os.path.join(dn, "log.jsonl"))]; Lo = [json.loads(x).get("loss") for x in open(os.path.join(do, "log.jsonl"))]
    fn, fo = flat(json.load(open(os.path.join(dn, "eval.json")))), flat(json.load(open(os.path.join(do, "eval.json"))))
    return {"log_losses_equal": Ln == Lo, "eval_fields_differing": sorted(k for k in set(fn) | set(fo) if not k.endswith("_sec") and fn.get(k) != fo.get(k))[:20], "n_fields": len(fn)}


def main():
    torch.set_num_threads(3); old_dm, old_n, old_i = sys.argv[1], sys.argv[2], sys.argv[3]; ok = True
    NT = ["--arm", "inject", "--think_k", "2", "--aux_step", "1.0", "--aux_fade", "48000,48000", "--fast_add", "--seed", "1", "--steps", "6", "--sched_scale", "0.002",
          "--log_every", "2", "--bs", "4", "--per_n", "2", "--eval_bs", "8", "--serial_pool", "40", "--kprimes", "0,1,2,4"]
    IT = ["--think_k", "0", "--interleave", "1", "--pass_index", "sin_gain", "--fast_add", "--seed", "1", "--steps", "6", "--sched_scale", "0.002", "--log_every", "2",
          "--bs", "4", "--test_n", "4", "--ext_n", "4", "--probe_n_fit", "32", "--eval_bs", "4", "--ckpt_every", "1000", "--bench", BENCH]
    jobs = [("boxes.nested170", NT, os.path.join(SP, "n_new"), None), ("boxes.nested170", NT, os.path.join(SP, "n_old"), old_n),
            ("boxes.inject169", IT, os.path.join(SP, "i_new"), None), ("boxes.inject169", IT, os.path.join(SP, "i_old"), old_i),
            ("boxes.nested170", NT + ["--add_norm", "den"], os.path.join(SP, "cell_den"), None), ("boxes.nested170", NT + ["--add_norm", "rms"], os.path.join(SP, "cell_rms"), None)]
    R = run_many(jobs)
    # (1) deltamem bitwise vs the pre-185 copy
    spec = importlib.util.spec_from_file_location("dm_pre", old_dm); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
    gp = s5.load_gpt2("cpu"); torch.manual_seed(0); mem = DM.DeltaMem(gp, 6, "top")
    with torch.no_grad():
        mem.gamma.fill_(0.5); mem.fade.fill_(1.0); mem.wo.weight.normal_(0, 0.02); mem.gate.weight.normal_(0, 0.01)
    h6, hw, S5 = torch.randn(3, 9, 768), torch.randn(3, 9, 768), torch.randn(3, 12, 64, 64); msk = torch.ones(3, 9, dtype=torch.bool); msk[1, 5:] = False; dm1 = {}
    for fa in (False, True):
        DM.FAST_ADD = P.FAST_ADD = fa
        dm1["fast%d" % fa] = {"write": bool(torch.equal(DM.write_seq(mem, h6, hw, S5), P.write_seq(mem, h6, hw, S5))),
                              "write_mask": bool(torch.equal(DM.write_seq(mem, h6, hw, S5, msk), P.write_seq(mem, h6, hw, S5, msk))),
                              "read": bool(torch.equal(DM.read_fixed(mem, h6, S5), P.read_fixed(mem, h6, S5)))}
    DM.FAST_ADD = P.FAST_ADD = False
    g1 = {"deltamem": dm1}
    g1["nested170"] = cmp(jobs[0][2], jobs[1][2]) if R[0][0] == 0 and R[1][0] == 0 else {"rc": [R[0][0], R[1][0]], "err": (R[0][1] + R[1][1])[-600:]}
    g1["inject169_il1"] = cmp(jobs[2][2], jobs[3][2]) if R[2][0] == 0 and R[3][0] == 0 else {"rc": [R[2][0], R[3][0]], "err": (R[2][1] + R[3][1])[-600:]}
    sp = subprocess.run([sys.executable, "scratch/smoke_plain.py"], capture_output=True, text=True, cwd=ROOT); g1["smoke_plain"] = sp.stdout.strip().splitlines()[-1:]
    OUT["g1"] = g1
    OUT["g1_pass"] = all(all(v.values()) for v in dm1.values()) and all(g1[k].get("log_losses_equal") and not g1[k].get("eval_fields_differing") for k in ("nested170", "inject169_il1")) and sp.returncode == 0
    ok &= OUT["g1_pass"]; dump()
    # (2) den z == sum mask * phi(k)
    torch.manual_seed(1); md = DM.DeltaMem(gp, 6, "top", add_norm="den")
    with torch.no_grad():
        md.load_state_dict(mem.state_dict(), strict=False)
    S0 = md.zero_state(3, "cpu"); g2 = {}
    for fa in (False, True):
        DM.FAST_ADD = fa; S1 = DM.write_seq(md, h6, hw, S0, msk)
        u = (1 - md.gamma) * md.ln(h6) + md.gamma * md.ln_w(hw); _, k, _ = md.qkv(u).split(768, -1); pk = F.elu(k.view(3, 9, 12, 64)[:, :, :6]) + 1
        zr = (pk * msk[:, :, None, None]).sum(1); g2["fast%d" % fa] = {"shape": list(S1.shape), "z_maxdiff": float((DM.state_z(md, S1) - zr).abs().max()), "z_scale": float(zr.abs().max())}
    DM.FAST_ADD = False
    OUT["g2"] = g2; OUT["g2_pass"] = all(v["shape"][1] == 13 and v["z_maxdiff"] <= 1e-5 * max(1.0, v["z_scale"]) for v in g2.values()); ok &= OUT["g2_pass"]; dump()
    # (3) den read norm <= max ||v|| x 1.001 (fixture: S_add = sum phi(k) v^T, z = sum phi(k))
    torch.manual_seed(2); B_, T_ = 4, 7; pk = F.elu(torch.randn(B_, T_, 6, 64)) + 1; v = torch.randn(B_, T_, 6, 64); Sa = torch.einsum("bthk,bthv->bhkv", pk, v); z = pk.sum(1)
    pq = F.elu(torch.randn(B_, 5, 6, 64)) + 1; rd = DM.add_read(md, pq, Sa, z); vmax = v.norm(dim=-1).max(1).values      # (B, 6)
    OUT["g3"] = {"max_ratio": float((rd.norm(dim=-1) / vmax[:, None, :]).max())}; OUT["g3_pass"] = OUT["g3"]["max_ratio"] <= 1.001; ok &= OUT["g3_pass"]; dump()
    # (4) rms unit RMS before the gain
    mr = DM.DeltaMem(gp, 6, "top", add_norm="rms"); rr = DM.add_read(mr, pq, Sa)
    OUT["g4"] = {"gain": mr.add_gain.tolist(), "rms_min": float(rr.pow(2).mean(-1).sqrt().min()), "rms_max": float(rr.pow(2).mean(-1).sqrt().max())}
    OUT["g4_pass"] = OUT["g4"]["gain"] == [1.0] * 6 and abs(OUT["g4"]["rms_min"] - 1) < 1e-3 and abs(OUT["g4"]["rms_max"] - 1) < 1e-3; ok &= OUT["g4_pass"]; dump()
    # (5) gradflow den / rms and (6) den runtime asserts (InjectN, a training step + eval)
    g5 = {}; g6 = {}
    for an in ("den", "rms"):
        torch.manual_seed(3); m = N.InjectN(gp, 6, "top", 6, 6, an)
        with torch.no_grad():
            m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0); m.mem.gamma.fill_(0.5)
        its = N.train_items(np.random.default_rng(4), 4, 4, set()); x, mk = m.think_inputs(its, "cpu"); opt = torch.optim.AdamW(m.mem.parameters(), lr=1e-3)
        S = m.think(m.mem.zero_state(4, "cpu"), x, mk, 2); lg = m.dec_logits(S, its, "cpu"); F.cross_entropy(lg, torch.tensor(N.NUM_IDS[[it["t"] for it in its]])).backward()
        g5[an] = {"mem_zero_grad": [n_ for n_, p_ in m.mem.named_parameters() if p_.requires_grad and (p_.grad is None or float(p_.grad.norm()) == 0)],
                  "has_add_gain": hasattr(m.mem, "add_gain")}
        opt.step()
        if an == "den":
            with torch.no_grad():
                S2 = m.think(m.mem.zero_state(4, "cpu"), x, mk, 3); m.dec_logits(S2, its, "cpu")
                ab = m.mem.ablate; m.mem.ablate = "add"; mp = DM.DeltaMem(gp, 6, "top"); mp.load_state_dict({k_: v_ for k_, v_ in m.mem.state_dict().items()}, strict=False); mp.ablate = "add"
                h_ = torch.randn(4, 5, 768); r_full = DM.read_fixed(m.mem, h_, S2); r_cut = DM.read_fixed(mp, h_, S2[:, :12]); m.mem.ablate = ab
            g6 = {"shape1": int(S2.shape[1]), "zblock_pad_rows_max_after_train_and_eval": float(S[:, 12, 6:].abs().max().detach()) + float(S2[:, 12, 6:].abs().max()),
                  "delta_read_equal_full_vs_cut": bool(torch.equal(r_full, r_cut))}
    OUT["g5"] = g5; OUT["g5_pass"] = all(not v["mem_zero_grad"] for v in g5.values()) and g5["rms"]["has_add_gain"] and not g5["den"]["has_add_gain"]; ok &= OUT["g5_pass"]
    OUT["g6"] = g6; OUT["g6_pass"] = g6["shape1"] == 13 and g6["zblock_pad_rows_max_after_train_and_eval"] == 0.0 and g6["delta_read_equal_full_vs_cut"]; ok &= OUT["g6_pass"]; dump()
    # (7) static check
    pat = re.compile(r"H_ADD:\]|h_add:\]|\[:, *6:\]|S\[:, *H_ADD|S\[:, *h_add|ha_?:\]|S\.flatten|S_?\.flatten|\.flatten\(1\)")
    runs_den = {"deltamem.py", "nested170.py", "halt173.py"}; hits = []
    for f_ in sorted(os.listdir(os.path.join(ROOT, "boxes"))):
        if not f_.endswith(".py") or f_.startswith("_"):
            continue
        for ln, line in enumerate(open(os.path.join(ROOT, "boxes", f_), encoding="utf-8"), 1):
            if pat.search(line) and ("S" in line or "St" in line):
                st = ("patched / explicit end or q-k-v heads (checked)" if f_ in runs_den else "not needed: module never builds an add_norm='den' memory")
                hits.append({"file": f_, "line": ln, "status": st, "text": line.strip()[:120]})
    OUT["g7"] = {"n_hits": len(hits), "hits": hits, "den_modules": sorted(runs_den)}
    unpatched = [h for h in hits if h["file"] in runs_den and re.search(r"S\[:, *(DM\.)?H_ADD:\]|S\[:, *(m\.mem\.)?h_add:\]|Sf\[:, *ha_:\]", h["text"])]
    OUT["g7_pass"] = not unpatched; OUT["g7"]["unpatched_in_den_modules"] = unpatched; ok &= OUT["g7_pass"]; dump()
    # (8) cell smokes + halt173 --normcap on the den smoke
    g8 = {}
    for nm, (rc, err) in (("den", R[4]), ("rms", R[5])):
        d_ = jobs[4 if nm == "den" else 5][2]
        if rc:
            g8[nm] = {"rc": rc, "err": err[-600:]}; continue
        e_ = json.load(open(os.path.join(d_, "eval.json"))); s_ = e_.get("s185", {})
        g8[nm] = {"rc": 0, "args_add_norm": json.load(open(os.path.join(d_, "args.json"))).get("add_norm"), "s185_keys": sorted(s_), "read_norm_by_pass": s_.get("read_norm_by_pass")}
    if g8.get("den", {}).get("rc") == 0:
        hr = subprocess.run([sys.executable, "-m", "boxes.halt173", "--run", jobs[4][2], "--normcap", "--bs", "16", "--device", "cpu"], capture_output=True, text=True,
                            env=dict(os.environ, OMP_NUM_THREADS="3", PYTHONPATH=ROOT), cwd=ROOT)
        nf = os.path.join(jobs[4][2], "eval_s185_normcap.json"); g8["normcap"] = {"rc": hr.returncode, "err": hr.stderr[-400:] if hr.returncode else "",
                                                                                  "C": json.load(open(nf))["normcap"]["C"] if os.path.exists(nf) else None}
    OUT["g8"] = g8; need = {"clip_rate", "grad_norm_preclip", "grad_add", "read_norm_by_pass"}
    OUT["g8_pass"] = all(g8.get(n, {}).get("rc") == 0 and g8[n]["args_add_norm"] == n and need <= set(g8[n]["s185_keys"]) for n in ("den", "rms")) and g8.get("normcap", {}).get("rc") == 0
    ok &= OUT["g8_pass"]
    for j in jobs:
        for f_ in ("model.pt", "ckpt_last.pt", "ckpt_prev.pt"):
            os.path.exists(os.path.join(j[2], f_)) and os.remove(os.path.join(j[2], f_))
    OUT["all_pass"] = bool(ok); dump(); print("GATE185", json.dumps({k: v for k, v in OUT.items() if k.endswith("pass")}), flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
