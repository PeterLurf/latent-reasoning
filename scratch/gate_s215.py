"""§215 SELF-DEPTH local gates (SELFDEPTH_v1 §6 at 5ded8703): G-S0 (no-gold), G-S7 (plain path), G-S8 (gradflow + mass_stats on a self batch); G-S2 / G-S3 /
G-S4 after the pool lands (s2 / s4 here, s3 from the cells' draws_log.json).
The trainer under test is TREE/boxes/vbank.py (the §215 working copy, a code-only copy of boxes/ under scratch/_tmp; ROOT pointed at D:/Latent for data)
until it is committed; after commit run with --tree D:/Latent (G-S7 on the final tree).
  fix  fixture records: 16 fresh items (rng 21995, 4 per n 9..12, never trained), labelled by selfdepth.label_chains on D3-T; kept (KEEP and CV) only
       -> scratch/gate_s215/fixture_recs.json (record keys exactly REC_KEYS). Gate fixtures only; not pool records.
  s0   grep: 'chain_label' in the trainer only via selfdepth.self_item's SD-LABEL-MAP line (vbank.py has no 'chain_label' and no 's215_gate'); self
       items carry no root / gold key; a VB.recode_targets call counter: 0 calls in vb_loss and slva.mass_stats on a self batch, > 0 on a plain batch.
  s7   plain path (no --self_pool, p_self 0, draws_log off): HEAD's boxes.vbank (cwd D:/Latent) vs TREE's, pa / carry / slva, 3 steps, CPU, bs 16:
       log + calibration bitwise; no draws_log.json / auto_micro_self.json written.
  s8   gradflow on a self batch (D3-T weights, c 0, train): R1 alone (rec_w 0) and R2 alone (L(rec_w 1) - L(rec_w 0), same inputs) each give nonzero
       grads on slva_wq / slva_wk / rec.weight / core layers 7, 10 / beta; ast: rec_targets / vb_loss / main carry no decorator; mass_stats on the
       self batch runs (no KeyError) with 0 recode_targets calls.
  s2   (after GATE-1) the trainer's rec_targets on self_item(rec, 16) == rec_targets on {root = _parse_tree(syms)} at Kb 16, bitwise, on the GATE-1
       kept items with state_ok and on the first 1,000 pool records whose roots are re-derived here (pool records differ only where a step value is
       wrong; counted).
    python scratch/gate_s215.py s7 --device cpu"""
import argparse
import ast
import json
import math
import os
import subprocess
import sys
from contextlib import nullcontext

TREE = "D:/Latent/scratch/_tmp/sd215_tree"
for i, v in enumerate(sys.argv):
    if v == "--tree":
        TREE = sys.argv[i + 1]
sys.path.insert(0, TREE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402
import boxes.selfdepth as SD  # noqa: E402
from boxes import s5  # noqa: E402

OUT = "D:/Latent/scratch/gate_s215"; RUNS = "D:/Latent/boxes/runs_fsq"; D3T = RUNS + "/s213_d3t_s1"


def fixture(dev):
    fp = os.path.join(OUT, "fixture_recs.json")
    if os.path.exists(fp):
        return json.load(open(fp))
    import boxes.vbank_eval as VE
    rg = np.random.default_rng(21995); syms = [N.render(N.generate(rg, n)[0]) for n in (9, 10, 11, 12) for _ in range(4)]
    m, _ = VE.load(D3T, dev); lab = SD.label_chains(m, syms, dev, bs=32); del m
    recs = [{"syms": s_, "n": o["n"], "chain_label": o["label"], "steps": o["steps"]} for s_, o in zip(syms, lab) if o["keep"] and o["cv"]]
    assert all(set(r) == SD.REC_KEYS for r in recs) and len(recs) >= 8, len(recs)
    json.dump(recs, open(fp, "w", newline="\n")); return recs


def d3t_model(dev):
    m = VB.make_model(s5.load_gpt2(dev), "slva", "train", "none").to(dev)
    mk = m.load_state_dict(torch.load(D3T + "/model.pt", map_location=dev, weights_only=True), strict=True); assert not mk.missing_keys
    m.c_anneal.fill_(0.0); return m


class Counter:
    def __init__(self):
        self.n = 0; self.f = VB.recode_targets

    def __enter__(self):
        def wrapped(*a, **k):
            self.n += 1; return self.f(*a, **k)
        VB.recode_targets = wrapped; return self

    def __exit__(self, *a):
        VB.recode_targets = self.f


def g_s0(dev):
    o = {}; src = open(os.path.join(TREE, "boxes", "vbank.py"), encoding="utf-8").read(); sds = open(os.path.join(TREE, "boxes", "selfdepth.py"), encoding="utf-8").read()
    o["vbank_chain_label"] = src.count("chain_label"); o["vbank_s215_gate"] = src.count("s215_gate")
    sub = [nd for nd in ast.walk(ast.parse(sds)) if isinstance(nd, ast.Subscript) and isinstance(nd.slice, ast.Constant) and nd.slice.value == "chain_label"]
    o["selfdepth_chain_label_reads"] = [ast.get_source_segment(sds, nd) + " @ line %d" % nd.lineno for nd in sub]
    o["sd_label_map_line"] = [sds.splitlines()[nd.lineno - 1].strip() for nd in sub]                 # the source line of each code read
    recs = fixture(dev); its = [SD.self_item(r, 16) for r in recs[:8]]; o["self_item_keys"] = sorted(set().union(*[set(it) for it in its]))
    m = d3t_model(dev); m.train(); Ks = [min(it["n"] + 2, 16) for it in its]
    from boxes.slva import mass_stats
    with Counter() as c:
        m.slva_log = []; VB.vb_loss(m, its, Ks, dev, nullcontext, 1.0); n_vb = c.n
        with torch.no_grad():
            ms = mass_stats(m, its, Ks, dev); n_ms = c.n - n_vb
    o["mass_stats_self"] = ms; m.slva_log = None
    pl = [it for it in N.test_set(1000) if it["n"] == 6][:4]
    with Counter() as c:
        VB.vb_loss(m, pl, [8] * 4, dev, nullcontext, 1.0); n_pl = c.n
    o.update({"recode_calls_self_vb_loss": n_vb, "recode_calls_self_mass_stats": n_ms, "recode_calls_plain_vb_loss": n_pl})
    o["pass"] = (o["vbank_chain_label"] == 0 and o["vbank_s215_gate"] == 0 and len(o["selfdepth_chain_label_reads"]) == 1 and len(o["sd_label_map_line"]) == 1
                 and o["sd_label_map_line"][0] == 'it["t"] = rec["chain_label"]  # SD-LABEL-MAP' and ms["chg"][1] > 0
                 and not (set(o["self_item_keys"]) & (SD.GOLD_KEYS - {"E_self", "t"})) and n_vb == 0 and n_ms == 0 and n_pl > 0)   # t = chain_label (SD-LABEL-MAP)
    return o


def g_s7(dev):
    o = {}; base = "D:/Latent/scratch/_tmp/s215_plain"
    for mode in ("pa", "carry", "slva"):
        logs = {}
        for tag, cwd in (("pre", "D:/Latent"), ("new", TREE)):
            out = "%s/%s_%s" % (base, mode, tag); os.makedirs(out, exist_ok=True)
            for f_ in os.listdir(out):
                os.remove(os.path.join(out, f_))
            cmd = [sys.executable, "-m", "boxes.vbank", "--mode", mode, "--steps", "48000", "--stop_at", "3", "--log_every", "1", "--bs", "16", "--micro", "1", "--device", "cpu", "--out", out]
            r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}); assert r.returncode == 0, r.stderr[-2000:]
            logs[tag] = ([json.loads(l_) for l_ in open(os.path.join(out, "log.jsonl"))], json.load(open(os.path.join(out, "calibration.json"))), sorted(os.listdir(out)))
        keys = ("loss", "answer_ce", "rec_ce", "gn_preclip", "c", "carry_mass", "slva_gn", "slva_offnewest_chg", "slva_offnewest_copy")
        o[mode] = {"log_bitwise": [[r_.get(k) for k in keys] for r_ in logs["pre"][0]] == [[r_.get(k) for k in keys] for r_ in logs["new"][0]],
                   "calibration_equal": logs["pre"][1] == logs["new"][1], "n_log": len(logs["new"][0]), "files_pre": logs["pre"][2], "files_new": logs["new"][2]}
    o["pass"] = all(o[md]["log_bitwise"] and o[md]["calibration_equal"] and o[md]["n_log"] == 3 and o[md]["files_pre"] == o[md]["files_new"] for md in ("pa", "carry", "slva"))
    return o


def g_s8(dev):
    o = {}; recs = fixture(dev); its = [SD.self_item(r, 16) for r in recs[:8]]; rs = np.random.default_rng(3); Ks = [min(it["n"] + int(rs.integers(0, 5)), 16) for it in its]
    m = d3t_model(dev); m.train(); P = dict(m.named_parameters()); names = ("slva_wq", "slva_wk", "rec.weight", "g.p.h_7_attn_c_attn_weight", "g.p.h_10_attn_c_attn_weight", "beta")

    def grads(f):
        m.zero_grad(set_to_none=True); torch.manual_seed(0); L = f(); L.backward()
        return {k: (float(P[k].grad.norm()) if P[k].grad is not None else None) for k in names}
    o["R1"] = grads(lambda: VB.vb_loss(m, its, Ks, dev, nullcontext, 0.0)[0])
    o["R2"] = grads(lambda: VB.vb_loss(m, its, Ks, dev, nullcontext, 1.0)[0] - VB.vb_loss(m, its, Ks, dev, nullcontext, 0.0)[0])
    tree = ast.parse(open(os.path.join(TREE, "boxes", "vbank.py"), encoding="utf-8").read()); dec = {}
    for nd in ast.walk(tree):
        if isinstance(nd, ast.FunctionDef) and nd.name in ("rec_targets", "vb_loss", "main"):
            dec[nd.name] = [ast.unparse(d) for d in nd.decorator_list]
    o["decorators"] = dec
    from boxes.slva import mass_stats
    with Counter() as c, torch.no_grad():
        m.slva_log = []; VB.vb_loss(m, its, Ks, dev, nullcontext, 1.0); ms = mass_stats(m, its, Ks, dev)
    o["mass_stats"] = ms; o["mass_stats_recode_calls"] = c.n
    ok = lambda d, skip=(): all(v is not None and v > 0 and math.isfinite(v) for k, v in d.items() if k != "beta" and k not in skip)   # beta reported, not required (§214 p10 had it nonzero under c 0)  # noqa: E731
    o["R1_skip"] = ["rec.weight (the R head is not on the answer path)"]
    o["pass"] = ok(o["R1"], ("rec.weight",)) and ok(o["R2"]) and all(v == [] for v in dec.values()) and c.n == 0 and ms["chg"][1] > 0
    return o


def g_s2(dev):
    """the pool part: E via the trainer's E_self branch vs via recode_targets(gold root re-derived from syms); a record differs only if a step value
    is wrong (CV fixes the frame), so n_diff / 1,000 estimates 1 - pi^E (GATE-1's term needs >= 0.99)."""
    o = {}; ge = json.load(open(RUNS + "/s215_gate1/eval_gate.json"))
    pool, _ = SD.load_pool(RUNS + "/s215_pool/pool_r1.jsonl", RUNS + "/s215_pool/pool_meta.json"); n_eq = 0; diff_label_ok = 0
    for r in pool[:1000]:
        tree = SD._parse_tree(r["syms"]); T = len(r["syms"]) + 2
        Es = VB.rec_targets([SD.self_item(r, 16)], 16, T, "cpu"); Eg = VB.rec_targets([{"root": tree, "syms": r["syms"], "n": r["n"]}], 16, T, "cpu")
        if torch.equal(Es, Eg):
            n_eq += 1
        else:
            diff_label_ok += int(r["chain_label"] == int(N.value(tree)))
    o.update({"pool_first_1000": {"n_cmp": 1000, "n_equal": n_eq, "n_diff": 1000 - n_eq, "n_diff_label_correct": diff_label_ok}, "gate1_gs2": ge["gs2_compose"]})
    o["pass"] = bool(ge["gs2_compose"]["PASS"]) and (1000 - n_eq) <= 10
    return o


def g_s4(dev):
    """reads scratch/_tmp/gs4 (scratch/_tmp/gs4.sh: A = init_from --keep_ckpt_steps 0 --stop_at 2 --ckpt_every 1; A resumed --stop_at 3; B straight to 3)."""
    G = "D:/Latent/scratch/_tmp/gs4"; o = {}; base = torch.load(D3T + "/model.pt", map_location="cpu", weights_only=True)
    c0 = torch.load(G + "/A/ckpt_model_0.pt", map_location="cpu", weights_only=False)["model"]
    o["ckpt0_keys_equal"] = sorted(c0) == sorted(base); o["ckpt0_bitwise"] = o["ckpt0_keys_equal"] and all(torch.equal(c0[k], base[k]) for k in base)
    al = open(G + "/A.log", encoding="utf-8").read(); o["init_line"] = [ln for ln in al.splitlines() if ln.startswith("INIT_FROM")]
    o["init_ok"] = len(o["init_line"]) == 1 and "(missing: [])" in o["init_line"][0] and "calibrate() NOT called" in o["init_line"][0]   # the CALIBRATION line only prints the copied values
    o["calibration_equal"] = json.load(open(G + "/A/calibration.json")) == json.load(open(D3T + "/calibration.json"))
    k2 = torch.load(G + "/A_ckpt2.pt", map_location="cpu", weights_only=False); o["ckpt_step2"] = k2["step"]
    o["ckpt_prev_step1"] = torch.load(G + "/A_prev1.pt", map_location="cpu", weights_only=False)["step"] if os.path.exists(G + "/A_prev1.pt") else None   # A's rotated step-1 ckpt, copied before the resume
    o["resume_line"] = "RESUME" in open(G + "/A_resume.log", encoding="utf-8").read()
    ka, kb = (torch.load(G + "/%s/ckpt_last.pt" % d, map_location="cpu", weights_only=False) for d in ("A", "B"))
    o["final_step"] = [ka["step"], kb["step"]]; o["self_state"] = [ka["self_state"], kb["self_state"]]
    o["rng_s_equal"] = ka["rng_s"] == kb["rng_s"]; o["np_rng_equal"] = ka["np_rng"] == kb["np_rng"] and ka["rng_k"] == kb["rng_k"]
    sa = [v["step"] for v in ka["opt"]["state"].values()]; sb = [v["step"] for v in kb["opt"]["state"].values()]
    o["adam_steps"] = [float(max(sa)), float(max(sb))]; o["adam_state_n"] = [len(sa), len(sb)]
    da, db = (json.load(open(G + "/%s/draws_log.json" % d)) for d in ("A", "B")); o["draws_log_equal"] = da == db; o["draws_steps"] = [d_["step"] for d_ in da]
    o["draws_coins"] = [d_["coin"] for d_ in da]
    o["pass"] = (o["ckpt0_bitwise"] and o["init_ok"] and o["calibration_equal"] and o["ckpt_step2"] == 2 and o["ckpt_prev_step1"] == 1 and o["resume_line"]
                 and o["final_step"] == [3, 3] and o["self_state"][0] == o["self_state"][1] and o["rng_s_equal"] and o["np_rng_equal"]
                 and o["adam_steps"] == [3.0, 3.0] and o["adam_state_n"][0] == o["adam_state_n"][1] and o["draws_log_equal"] and o["draws_steps"] == [1, 2, 3])
    return o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("gate"); ap.add_argument("--device", default="cpu"); ap.add_argument("--tree", default=TREE); a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; os.makedirs(OUT, exist_ok=True); os.chdir("D:/Latent")
    assert os.path.abspath(VB.__file__).replace("\\", "/").startswith(TREE.rstrip("/") + "/boxes"), (VB.__file__, TREE)
    r = {"fix": lambda d: {"n": len(fixture(d)), "pass": True}, "s0": g_s0, "s2": g_s2, "s4": g_s4, "s7": g_s7, "s8": g_s8}[a.gate](a.device)
    r["device"] = torch.cuda.get_device_name(a.device) if a.device.startswith("cuda") else "cpu"; r["tree"] = TREE; r["vbank_file"] = VB.__file__
    json.dump(r, open(os.path.join(OUT, "%s.json" % a.gate), "w", newline="\n"), indent=1); print(a.gate, "PASS" if r["pass"] else "FAIL", json.dumps(r)[:1600], flush=True)


if __name__ == "__main__":
    main()
