"""THEORY §177 PASS-DISTINCT pre-launch gates (4947f034, notes 177-1..3), boxes/inject169.py (a) tidx / (d) fadetidx, boxes/nested170.py (f) kmatch:
  (1) (a) gradflow (fade 0.5, W_o perturbed, gamma 0.5, K = 3, offset o = 5): the answer CE puts gradient on the index rows o + 1 .. o + 3 (and none
      elsewhere) and on every DeltaMem parameter; the bound assert fires at o + t >= rows
  (2) (d) masked mixture: a padded pair row inside a mixed batch writes exactly what the same pair segment writes alone (max abs diff < 1e-4, fp32);
      p = 0 is the pure pair schedule (no padding); gradflow at p = 0.5 into the index rows and every DeltaMem parameter; pair_seg = multpair's
      schedule / tokens (column order, ' ;' on a column's first pair)
  (3) (f) kmatch: a 3-step CPU run with --k_window 0,2 writes kmatch_counts with every K - n in {0, 1, 2} and one K per n-bucket per step
  (4) flags off: 3 steps of inject169 (no §177 flag) and of nested170 inject (no --k_window) equal the committed HEAD's (temporary copies, removed
      after): logged losses identical, weights within 1e-9 (HEAD's own CPU runs differ at ~4e-12 in wte, gate175's addendum)
    python scratch/gate177.py   -> boxes/runs_fsq/s177_gates/gate177.json"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import boxes.inject169 as I  # noqa: E402
import boxes.multpair as MP  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.fixedchunk import DT, product_digits  # noqa: E402
from boxes.multcol import gen_operands  # noqa: E402


def prep(m):
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02)
    m.mem.fade.fill_(0.5); m.mem.gamma.fill_(0.5); m.mem.s_add.data.fill_(-7.0); m.train(); m.zero_grad(set_to_none=True)


def main():
    torch.set_num_threads(3); out = {}; ok = True; torch.manual_seed(0); g = s5.load_gpt2("cpu")
    A, B = gen_operands(np.random.default_rng(3), 4, 4), gen_operands(np.random.default_rng(4), 4, 3); ans = torch.tensor(DT[product_digits(A, B)])
    # (1)
    m = I.Inject(g, 6, "top"); m.pidx = torch.nn.Embedding(72, 768); torch.nn.init.zeros_(m.pidx.weight); prep(m); m.p_off = 5
    S = m.think(m.mem.zero_state(4, "cpu"), A, B, 3); lg = m.dec_logits(S, A, B, ans); Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), ans.reshape(-1)).backward()
    rn = m.pidx.weight.grad.norm(dim=1); used = [i for i in range(72) if rn[i] > 0]
    mz = [n_ for n_, p_ in m.mem.named_parameters() if p_.grad is None or float(p_.grad.norm()) == 0]
    try:
        m.p_off = 70; m.write(m.mem.zero_state(4, "cpu"), m.ids(torch.tensor(I.think_text(A, B))), 2); bound = False
    except AssertionError:
        bound = True
    out["g1"] = {"index_rows_with_grad": used, "mem_zero": mz, "bound_assert": bound}; out["g1_pass"] = used == [6, 7, 8] and not mz and bound; ok &= out["g1_pass"]
    # (2)
    m2 = I.Inject(g, 6, "top"); m2.pidx = torch.nn.Embedding(72, 768); prep(m2)
    with torch.no_grad():
        m2.pidx.weight.normal_(0, 0.02)
    ok_seg = True; MP_m = type("X", (), {})()
    for na, nb in ((4, 3), (4, 5)):
        A_, B_ = gen_operands(np.random.default_rng(5), 3, na), gen_operands(np.random.default_rng(6), 3, nb); pairs, col, _ = MP.schedule(na, nb)
        newcol = np.r_[True, col[1:] != col[:-1]]
        for t in range(1, na * nb + 1):
            i, j = pairs[t - 1]; ref = np.stack([DT[A_[:, i]], DT[B_[:, j]], np.full(3, MP.NEWTOK if newcol[t - 1] else MP.SAMETOK)], 1)
            ok_seg &= bool(np.array_equal(I.pair_seg(A_, B_, t), ref))

    class R:                                                                # a fixed per-item draw: rows 0 / 2 full text, rows 1 / 3 pair segments
        def random(self, n):
            return np.array([0.0, 0.9, 0.0, 0.9])[:n]
    with torch.no_grad():
        m2.eval(); m2.p_off = 0; Smix = m2.think_fade(m2.mem.zero_state(4, "cpu"), A, B, 0.5, R())
        Spair = m2.think_fade(m2.mem.zero_state(4, "cpu"), A, B, 0.0, np.random.default_rng(0))
        Sfull = m2.think_fade(m2.mem.zero_state(4, "cpu"), A, B, 1.0, np.random.default_rng(0))
    d_pair = float((Smix[[1, 3]] - Spair[[1, 3]]).abs().max()); d_full = float((Smix[[0, 2]] - Sfull[[0, 2]]).abs().max())
    prep(m2); m2.p_off = 2; S = m2.think_fade(m2.mem.zero_state(4, "cpu"), A, B, 0.5, np.random.default_rng(1)); lg = m2.dec_logits(S, A, B, ans)
    Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), ans.reshape(-1)).backward(); rn2 = m2.pidx.weight.grad.norm(dim=1)
    mz2 = [n_ for n_, p_ in m2.mem.named_parameters() if p_.grad is None or float(p_.grad.norm()) == 0]
    out["g2"] = {"pair_seg_equals_multpair": ok_seg, "mixed_vs_pure_pair_rows": d_pair, "mixed_vs_full_rows": d_full, "index_rows_with_grad": [i for i in range(72) if rn2[i] > 0],
                 "mem_zero": mz2}
    out["g2_pass"] = ok_seg and d_pair < 1e-4 and d_full < 1e-4 and out["g2"]["index_rows_with_grad"] == list(range(3, 3 + 12)) and not mz2; ok &= out["g2_pass"]
    # (3) + (4)
    sp = os.environ.get("GATE177_OUT", os.path.join(ROOT, "boxes", "runs_fsq", "s177_gates")); res = {}; env = dict(os.environ, OMP_NUM_THREADS="3")
    common_n = ["--arm", "inject", "--think_k", "10", "--seed", "1", "--steps", "3", "--sched_scale", "0.002", "--log_every", "1", "--bs", "8", "--per_n", "2", "--eval_bs", "8",
                "--serial_pool", "40", "--device", "cpu"]
    od = os.path.join(sp, "kmatch"); r_ = subprocess.run([sys.executable, "-m", "boxes.nested170"] + common_n + ["--k_window", "0,2", "--out", od], capture_output=True, text=True, env=env)
    assert r_.returncode == 0, r_.stderr[-2000:]
    kc = json.load(open(os.path.join(od, "kmatch_counts.json"))); e7 = json.load(open(os.path.join(od, "eval.json")))["s177"]
    win = all(0 <= int(k_) - int(n_) <= 2 for n_, d in kc.items() for k_ in d); per = all(sum(d.values()) <= 3 for d in kc.values())
    out["g3"] = {"kmatch_counts": kc, "window_ok": win, "one_K_per_bucket_per_step": per, "s177_fields": sorted(e7)}
    out["g3_pass"] = win and per and {"acc_by_n_kprime", "skip_pass", "train_loss_tail", "kmatch_counts"} <= set(e7); ok &= out["g3_pass"]
    common_i = ["--think_k", "2", "--seed", "1", "--steps", "3", "--sched_scale", "0.002", "--log_every", "1", "--bs", "4", "--test_n", "4", "--ext_n", "4", "--probe_n_fit", "32",
                "--eval_bs", "4", "--device", "cpu"]
    common_n2 = [x for x in common_n]; common_n2[common_n2.index("--think_k") + 1] = "2"
    tmps = []
    try:
        for mod in ("inject169", "nested170"):
            head = subprocess.run(["git", "show", "HEAD:boxes/%s.py" % mod], capture_output=True, text=True, encoding="utf-8").stdout
            tmp = os.path.join(ROOT, "boxes", "_%s_head177_tmp.py" % mod); open(tmp, "w", encoding="utf-8", newline="\n").write(head); tmps.append(tmp)
        for nm, mod, args_ in (("i_head", "boxes._inject169_head177_tmp", common_i), ("i_new", "boxes.inject169", common_i), ("n_head", "boxes._nested170_head177_tmp", common_n2),
                               ("n_new", "boxes.nested170", common_n2)):
            od = os.path.join(sp, "off_" + nm); r_ = subprocess.run([sys.executable, "-m", mod] + args_ + ["--out", od], capture_output=True, text=True, env=env)
            assert r_.returncode == 0, (nm, r_.stderr[-2000:])
            res[nm] = ([json.loads(l)["loss"] for l in open(os.path.join(od, "log.jsonl")) if '"loss"' in l], torch.load(os.path.join(od, "model.pt"), weights_only=True))
    finally:
        for tmp in tmps:
            os.remove(tmp)
    g4 = {}
    for a_, b_ in (("i_new", "i_head"), ("n_new", "n_head")):
        mx = max(float((res[a_][1][k].float() - res[b_][1][k].float()).abs().max()) for k in res[b_][1])
        g4[a_] = {"losses_equal": res[a_][0] == res[b_][0], "max_weight_diff": mx}
    out["g4"] = g4; out["g4_pass"] = all(v["losses_equal"] and v["max_weight_diff"] < 1e-9 for v in g4.values()); ok &= out["g4_pass"]
    out["all_pass"] = bool(ok); gd = os.path.join(ROOT, "boxes", "runs_fsq", "s177_gates"); os.makedirs(gd, exist_ok=True)
    json.dump(out, open(os.path.join(gd, "gate177.json"), "w", newline="\n"), indent=1, default=str); print("GATE177", json.dumps(out, default=str), flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
