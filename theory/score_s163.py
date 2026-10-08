"""THEORY §163 scoring (rules fixed in §163; written before any §163 implementation or run; Theory is the only scorer).
SHAPE-MIX-2D: §155 R's recipe (DeltaMem-R, top writes, gamma 2k -> 6k) trained on a x b with BOTH lengths m, n ~ U{2..5} (drawn per batch, 161-1), 48k,
RTX 3090; runs only if §161 MIX s2 reads 4x4 test_acc >= 0.99 and ext_4x8 >= 0.9.
EXACT-ID(run): 4x4 bench >= 0.99 and 5x5 bench >= 0.99. Slot 1 EXACT-ID opens s2 / s3. GEN-2D(run): EXACT-ID and ext_8x8 >= 0.90.
BOUNDED(run): EXACT-ID and ext_6x6 >= 0.90 but ext_8x8 < 0.90. Arm: GEN-2D if >= 2 / 3 seeds GEN-2D; BOUNDED if >= 2 / 3 at least BOUNDED; else NOT GENERAL.
Descriptive: SERIAL on 5x5 (row level, pairs 6 / 12 / 18), state-carried (5x5 in-progress colsum_cls units >= 0.9), 7x7 / 10x10 / 4x8 / 8x4, decay profiles,
C3 at 5x5 vs TC-M 5x5 (1.000; 2.33e10; 87) beside ICoT-SI 5x5 (0.996; 5.76e9; 15), compute at 8x8; fallback trigger = note 162-2's outcome rule.
Dirs boxes/runs_fsq/s163_mix2d_s{seed}.
    python theory/score_s163.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s155 as S155  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
TCM5 = {"acc": 1.0, "flops": 23314827264.0, "passes": 87}; ICOT5 = {"acc": 0.996, "flops": 5764879872.0, "passes": 15}   # §98 / §159
BASE = {"family": "mult_pair", "carrier": "deltamem", "mem_write": "top", "write_mix_start": 2000, "write_mix_end": 6000, "shape_mix_2d": "2,3,4,5", "steps": 48000}
ERRORS = []
load = S155.load


def ta(r, k):
    return (r.get(k) or {}).get("test_acc")


def trigger(d, r):
    """Note 162-2's outcome rule: any non-finite step, or a logged window loss > 2x the logged loss 1k steps earlier, after 2k, within one curriculum stage
    (same c_max). Gradient spikes (155-1 / 160-1) descriptive."""
    sp = ((r.get("s155") or {}).get("write_grad_spikes_floor")) or []; nf = int(r.get("n_nonfinite") or 0); pts, jumps = {}, []
    p = os.path.join(R, d, "log.jsonl")
    if os.path.exists(p):
        for ln in open(p, encoding="utf-8"):
            try:
                x = json.loads(ln)
            except ValueError:
                continue
            if isinstance(x.get("step"), int) and isinstance(x.get("loss"), (int, float)):
                pts[x["step"]] = (float(x["loss"]), x.get("c_max"))
    for st, (lo, cm) in pts.items():
        pv = pts.get(st - 1000)
        if st > 2000 and pv and pv[1] == cm and pv[0] > 0 and lo > 2 * pv[0]:
            jumps.append(st)
    return {"loss_jumps": sorted(jumps), "n_nonfinite": nf, "fires": bool(jumps or nf), "grad_floor_spikes_descriptive": len(sp)}


def run(s):
    d = "s163_mix2d_s%d" % s; r = load(d); a = load(d, "args.json") or {}
    if r is None:
        return {"status": "running" if a else "open"}
    if a.get("mem_check") or a.get("train_n"):
        return {"status": "smoke"}
    bad = {k: a.get(k) for k, v in BASE.items() if a.get(k) != v}
    if bad:
        ERRORS.append((d, bad)); return {"status": "error", "error": bad}
    x = r.get("s163") or {}
    t44, t55, e66, e88 = r.get("test_acc"), ta(r, "test_5x5"), ta(r, "ext_6x6"), ta(r, "ext_8x8")
    exact = bool((t44 or 0) >= 0.99 and (t55 or 0) >= 0.99)
    kind = "GEN-2D" if exact and (e88 or 0) >= 0.90 else ("BOUNDED" if exact and (e66 or 0) >= 0.90 else ("EXACT-ID only" if exact else "not exact in distribution"))
    ws = (x.get("what_state_5x5") or {}).get("operand_probe") or {}
    inprog = {}
    for P, v in ws.items():                                                  # the in-progress column at each probe point: the largest column index probed there
        cc = v.get("colsum_cls") or {}
        if cc:
            k = max(cc, key=int); inprog[P] = (cc[k].get("units") or {}).get("acc")
    c5 = x.get("compute_5x5") or {}; c8 = x.get("compute_8x8") or {}
    beats5 = bool((t55 or 0) >= TCM5["acc"] - 0.01 and c5.get("flops") and c5["flops"] < TCM5["flops"] and (c5.get("sequential_passes") or 1e9) < TCM5["passes"])
    return {"status": "done", "test_4x4": t44, "test_5x5": t55, "exact_id": exact, "kind": kind,
            "ext": {k: ta(r, "ext_" + k) for k in ("6x6", "7x7", "8x8", "10x10", "4x8", "8x4")},
            "ext_8x8_per_column": ((r.get("ext_8x8") or {}).get("acc_by_column") or {}).get("per_column"),
            "serial_5x5": S155.S154.serial(x.get("donor_5x5")), "serial_4x4": S155.S154.serial(r.get("donor")),
            "inprogress_units_5x5": inprog, "state_carried_5x5": bool(inprog) and all((v or 0) >= 0.9 for v in inprog.values()),
            "compute_5x5": {"flops": c5.get("flops"), "passes": c5.get("sequential_passes"), "vs_tcm": c5.get("flops") and c5["flops"] / TCM5["flops"],
                            "vs_icot": c5.get("flops") and c5["flops"] / ICOT5["flops"]},
            "compute_8x8": {"flops": c8.get("flops"), "passes": c8.get("sequential_passes")}, "latent_beats_text_5x5": beats5,
            "value_range": x.get("value_range"), "test_4x4_at_24k": x.get("test_4x4_at_24k"),          # descriptive (Literature / Audit)
            "fallback_trigger": trigger(d, r), "gpu": r.get("train_gpu")}


def main():
    g = S155.load("s161_mix_4x4_s2"); ga = S155.load("s161_mix_4x4_s2", "args.json") or {}
    if g is None:
        cond = "pending (§161 MIX s2 not read)"
    else:
        ok = (g.get("test_acc") or 0) >= 0.99 and ((g.get("ext_4x8") or {}).get("test_acc") or 0) >= 0.9
        cond = ("RUN" if ok else "NOT RUN") + " (§161 MIX s2: 4x4 %s, ext_4x8 %s)" % (g.get("test_acc"), (g.get("ext_4x8") or {}).get("test_acc"))
    print("CONDITION:", cond); res = {"condition": cond}
    runs = {s: run(s) for s in (1, 2, 3)}; res["runs"] = runs
    s1 = runs[1]
    gate = None if s1.get("status") != "done" else s1["exact_id"]
    done = [v for v in runs.values() if v.get("status") == "done"]
    n_gen = sum(1 for v in done if v["kind"] == "GEN-2D"); n_bnd = sum(1 for v in done if v["kind"] in ("GEN-2D", "BOUNDED"))
    if gate is False:
        v = "gate closed: 2-D mixing breaks in-distribution exactness at 48k (s1 4x4 %s, 5x5 %s)" % (s1["test_4x4"], s1["test_5x5"])
    elif len(done) < 3 and not (n_gen >= 2 or len(done) - n_bnd >= 2):
        v = "pending (%d / 3 seeds)" % len(done)
    else:
        v = "GEN-2D" if n_gen >= 2 else ("BOUNDED" if n_bnd >= 2 else "NOT GENERAL")
    res["verdict"] = v
    for s, x in runs.items():
        if x.get("status") != "open":
            print("s%d" % s, {k: x.get(k) for k in ("status", "test_4x4", "test_5x5", "kind", "ext", "serial_5x5", "state_carried_5x5", "latent_beats_text_5x5",
                                                     "test_4x4_at_24k", "fallback_trigger", "error")})
    rd = {"GEN-2D": "latent-memory multiplication generalises over both operands without a scratchpad (C2 at the literature frontier; C3 vs ICoT-SI's per-size models)",
          "BOUNDED": "bounded 2-D generalisation (6x6 holds, 8x8 does not); the decay profile is the result",
          "NOT GENERAL": "2-D mixing keeps in-distribution exactness but does not generalise past 5x5"}.get(v, v)
    res["reading"] = rd; print("VERDICT:", v, "| READING:", rd)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s163.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
