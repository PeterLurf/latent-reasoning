"""THEORY §163 SHAPE-MIX-2D pre-launch gates (0045d6a1), CPU, on the real training batches (main()'s batch(), captured at MultDelta.chain):
  (1) labels / schedule for m != n: schedule(m, n) emits at each column's last pair + the flush (m + n emissions) and accumulators()' digits are
      a x b's digits LSB first, every (m, n) in {2..5}^2 plus 8x4 / 4x8 / 10x10; captured --shape_mix_2d batches: (m, n) in the list, labels at
      the emission steps = those digits, several shapes seen, no row equal to a 4x4 / 5x5 bench test item (either order)
  (2) the exclusion itself: gen2d redraws every row that hits the excluded set
  (3) both flags off: the captured batches (items, labels, trace mask, operands, every step's tokens) and the step losses bitwise equal to the
      reference module's (MULTPAIR_REF, default boxes.multpair) under §155 R's arguments
  (4) --mem_check with --shape_mix_2d runs (the 5x5 training shape + the 8x8 eval chain; peak on CPU is None - the real check runs on the host)
    MULTPAIR_MOD=boxes._multpair163_tmp python scratch/gate163.py   -> boxes/runs_fsq/s163_gates/gate163.json (exit 1 on a failed gate)"""
import importlib
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

MP = importlib.import_module(os.environ.get("MULTPAIR_MOD", "boxes.multpair"))
from scratch.gate161 import capture, digits_lsb  # noqa: E402
from boxes.multcol import DTOK, gen_operands, load_bench  # noqa: E402


def excl_set():
    E = set()
    for t in ("4x4", "5x5"):
        A, B = load_bench(os.path.join(ROOT, "bench", "data", "mult_%s_test.txt" % t))
        E |= {(x.tobytes(), y.tobytes()) for x, y in zip(A, B)} | {(y.tobytes(), x.tobytes()) for x, y in zip(A, B)}
    return E


def main():
    torch.set_num_threads(3); out = {"module": MP.__name__}; ok = True; rng = np.random.default_rng(0); r1 = {}
    for ma in range(2, 6):
        for mb in range(2, 6):
            r1["%dx%d" % (ma, mb)] = None
    for k in list(r1) + ["8x4", "4x8", "10x10"]:
        ma, mb = (int(v) for v in k.split("x")); A, Bd = gen_operands(rng, 64, ma), gen_operands(rng, 64, mb); acc, dig, _, pairs, col, emit = MP.accumulators(A, Bd)
        a_ = sum(A[:, i].astype(object) * 10 ** i for i in range(ma)); b_ = sum(Bd[:, j].astype(object) * 10 ** j for j in range(mb)); P = digits_lsb(np.array(a_ * b_, dtype=object), ma + mb).astype(np.int64)
        last = [t for t in range(len(pairs)) if t == len(pairs) - 1 or col[t + 1] != col[t]]
        r1[k] = {"n_emit": int(emit.sum()), "emit_ok": list(np.nonzero(emit)[0]) == last, "digits_exact": bool((dig[:, emit] == P).all())}
    out["g1a"] = r1; out["g1a_pass"] = all(v["n_emit"] == sum(int(x) for x in k.split("x")) and v["emit_ok"] and v["digits_exact"] for k, v in r1.items()); ok &= out["g1a_pass"]
    E = excl_set(); gb = capture(MP, ["--shape_mix_2d", "2,3,4,5"], 40); lut = {t: i for i, t in enumerate(DTOK)}; seen = set(); bad = []; hits = 0; full = 0
    for k, b in enumerate(gb):
        A = np.vectorize(lut.get)(b["adig"]).astype(np.int64); Bd = np.vectorize(lut.get)(b["bdig"]).astype(np.int64); seen.add((A.shape[1], Bd.shape[1]))
        _, dig, _, pairs, col, emit = MP.accumulators(A, Bd); T = b["items"].shape[1]
        if not (A.shape[1] in (2, 3, 4, 5) and Bd.shape[1] in (2, 3, 4, 5) and (b["tm"][0] == emit[:T]).all() and (b["labels"][:, emit[:T]] == dig[:, :T][:, emit[:T]]).all() and (b["pairs"] == pairs).all()):
            bad.append(k)
        hits += sum((x.tobytes(), y.tobytes()) in E for x, y in zip(A, Bd)); full += int(T == len(pairs))
    out["g1b"] = {"batches": len(gb), "shapes_seen": sorted("%dx%d" % s_ for s_ in seen), "bad": bad, "excluded_hits": int(hits), "full_chains": full}
    out["g1b_pass"] = not bad and len(seen) >= 8 and hits == 0; ok &= out["g1b_pass"]
    # (2) the exclusion: with the excluded set = the first draw's own rows, a redraw from the same state must avoid all of them
    r_ = np.random.default_rng(5); A0, B0 = MP.gen2d(r_, 256, 4, 4, None); E0 = {(x.tobytes(), y.tobytes()) for x, y in zip(A0, B0)}
    r_ = np.random.default_rng(5); A1, B1 = MP.gen2d(r_, 256, 4, 4, E0); hit1 = sum((x.tobytes(), y.tobytes()) in E0 for x, y in zip(A1, B1))
    out["g2"] = {"excluded_hits_after_redraw": int(hit1), "n_excl": len(E), "excl_has_4x4_and_5x5": len(E) >= 4 * 1000 - 50}
    out["g2_pass"] = hit1 == 0 and out["g2"]["excl_has_4x4_and_5x5"]; ok &= out["g2_pass"]
    # (3) flags off vs the reference module
    REF = importlib.import_module(os.environ.get("MULTPAIR_REF", "boxes.multpair")); torch.use_deterministic_algorithms(True, warn_only=True)
    ga = capture(MP, [], 12); gr = capture(REF, [], 12); torch.use_deterministic_algorithms(False); same = len(ga) == len(gr)
    for x, y in zip(ga, gr):
        same &= all(np.array_equal(x[k], y[k]) for k in ("items", "labels", "tm", "adig", "bdig")) and all(np.array_equal(p_, q_) for p_, q_ in zip(x["toks"], y["toks"])) and x["loss"] == y["loss"]
    out["g3"] = {"batches": [len(ga), len(gr)], "ref": REF.__name__}; out["g3_pass"] = bool(same) and REF.__name__ != MP.__name__; ok &= out["g3_pass"]
    # (4) the mem_check path
    d = tempfile.mkdtemp(prefix="g163m_"); sys.argv = ["multpair", "--family", "mult_pair", "--task", "4x4", "--seed", "1", "--carrier", "deltamem", "--mem_write", "top", "--shape_mix_2d", "2,3,4,5",
                                                       "--bs", "4", "--eval_bs", "2", "--train_n", "500", "--test_n", "8", "--mem_check", "--device", "cpu", "--out", d]
    MP.main(); mc = json.load(open(os.path.join(d, "mem_check.json"))); out["g4"] = mc; out["g4_pass"] = True; ok &= out["g4_pass"]
    out["all_pass"] = bool(ok); dd = os.path.join(ROOT, "boxes", "runs_fsq", "s163_gates"); os.makedirs(dd, exist_ok=True)
    json.dump(out, open(os.path.join(dd, "gate163.json"), "w", newline="\n"), indent=1, default=int)
    print("GATE163", json.dumps({k: v for k, v in out.items() if k.endswith("_pass") or k in ("g1b", "g2")}, default=int), flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
