"""RED-TEAM SELF-DEPTH ($0, CPU, eval-only, no repo writes): COMPOSE-VALID / SPAN / FRAME / ORDER / CANON and pi^E on D3-T's REAL fresh decodes.
Items: exactly part_chainlabel_fresh's (213-37/38) fresh items: excl = test_set(1000) syms + SERIAL (rng 17300, 2200 x n 8), rng 21337, n 9..NMAX drawn
in order, 1,000 per n (all drawn so the rng stream matches); the chain runs on the first PER items of each n (CPU budget).
Chain: as part_chainlabel_fresh (1-pass rollout on batch_x(render(cur)), argmax rec_logits(V_1) over the item's own len, VB.rerender, _parse_tree;
length-sorted batches of BS; cap n + 4), but the full decoded class vector of every step is kept and scored with compose.py's check_step /
composition (imported read-only from docs/research_selfdepth_2026-10-17/scripts/compose.py). Gold (generator root / value) for evaluation only.
usage: python real_compose.py OUT.json PER NMIN NMAX THREADS"""
import os, sys, json, time
THREADS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
os.environ["OMP_NUM_THREADS"] = str(THREADS); os.environ["MKL_NUM_THREADS"] = str(THREADS)
sys.dont_write_bytecode = True; sys.path.insert(0, "D:/Latent"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
torch.set_num_threads(THREADS)
import boxes.nested170 as N
import boxes.vbank as VB
import boxes.vbank_eval as VE
from boxes.vbank_diag import _parse_tree, n_ops
import compose_ref as C                                                      # a byte copy of the draft's compose.py (check_step, x_classes, handle_slots)

OUT = sys.argv[1]; PER = int(sys.argv[2]); NMIN = int(sys.argv[3]); NMAX = int(sys.argv[4]); BS = 32
RUN = "D:/Latent/boxes/runs_fsq/s213_d3t_s1"


def main():
    t0 = time.time(); dev = "cpu"
    m, args = VE.load(RUN, dev)
    TE = N.test_set(1000); excl = {tuple(it["syms"]) for it in TE}; r_sp = np.random.default_rng(17300)
    for _ in range(2200):
        excl.add(tuple(N.render(N.generate(r_sp, 8)[0])))
    rng = np.random.default_rng(21337); items = {}
    for n in range(9, NMAX + 1):                                             # same draw order as part_chainlabel_fresh
        roots, gold = [], []
        while len(roots) < 1000:
            r_, y_ = N.generate(rng, n)
            if tuple(N.render(r_)) in excl:
                continue
            roots.append(r_); gold.append(y_)
        if n >= NMIN:
            items[n] = (roots[:PER], gold[:PER])
    print("items ready", round(time.time() - t0, 1), flush=True)
    res = {}
    for n, (roots, gold) in items.items():
        t1 = time.time(); J = len(roots)
        cur = {j: N._copy(roots[j]) for j in range(J)}; decs = {j: [] for j in range(J)}; label = {}; steps = {j: 0 for j in range(J)}; active = set(cur)
        prev = {j: N._copy(roots[j]) for j in range(J)}                     # the tree the step's input was rendered from
        for _ in range(n + 4):
            if not active:
                break
            order = sorted(active, key=lambda j: len(N.render(cur[j]))); nxt = set()
            for c0 in range(0, len(order), BS):
                part = order[c0:c0 + BS]; its = [{"syms": N.render(cur[j])} for j in part]; x, pad, lens = VB.batch_x(its, dev)
                with torch.no_grad():
                    rr = m.rollout(x, pad, 1); dec = m.rec_logits(rr["V"][1]).argmax(-1).cpu().numpy(); del rr
                for k_, j in enumerate(part):
                    d_ = [int(v) for v in dec[k_, :int(lens[k_])]]; decs[j].append((cur[j], d_)); steps[j] += 1
                    t_ = _parse_tree(VB.rerender(d_))
                    if t_ is None:
                        label[j] = None; continue
                    cur[j] = t_
                    if t_.op is None:
                        label[j] = t_.v
                    else:
                        nxt.add(j)
            active = nxt
        for j in active:
            label[j] = None
        # ---- score with compose.py's per-step checks, replaying the recorded decodes
        agg = {"items": J, "keep": 0, "compose_valid": 0, "kept_wrong": 0, "cv_wrong": 0, "cv_E_wrong": 0, "cv_E_right": 0,
               "keep_not_cv": 0, "keep_not_cv_valright": 0, "fail_flag_among_keep": {k: 0 for k in ("FRAME", "SPAN", "ORDER", "CANON", "ONE_OP")},
               "steps_total": 0, "step_flag_fail": {k: 0 for k in ("FRAME", "SPAN", "ORDER", "CANON", "ONE_OP")},
               "value_slot": {"last": 0, "first": 0, "other": 0, "none": 0}, "frame_bad_steps": 0, "examples": []}
        for j in range(J):
            seq = decs[j]; it = iter(seq)
            replay = lambda cur_tree, _it=it: next(_it)[1]                   # decoder returning the recorded decode for this step
            r = C.chain_compose(N.render(roots[j]), replay, cap=n + 4)
            keep = label.get(j) is not None and steps[j] == n
            assert keep == r["keep"], (n, j, keep, r["keep"], steps[j], r["steps"])
            assert (r["label"] if keep else None) == (label.get(j) if keep else None)
            # per-step flags on every recorded step
            for (ct, d_) in seq:
                f, _ = C.check_step(ct, d_); agg["steps_total"] += 1
                for k in agg["step_flag_fail"]:
                    agg["step_flag_fail"][k] += not f[k]
                xs = C.x_classes(N.render(ct)); T = len(xs); vo = [q for q in range(1, T - 1) if d_[q] == VB.VOID]
                if not (d_[0] == C.EOT_C and d_[-1] == C.EQ_C):
                    agg["frame_bad_steps"] += 1
                if vo and ct.op is not None:
                    hs = C.handle_slots(ct)                               # canonical span slots (value on hs[-1])
                    if d_[hs[-1]] in C.VALC and all(d_[q] == VB.VOID for q in hs[:-1]):
                        agg["value_slot"]["last"] += 1
                    elif d_[hs[0]] in C.VALC and all(d_[q] == VB.VOID for q in hs[1:]):
                        agg["value_slot"]["first"] += 1
                    else:
                        agg["value_slot"]["other"] += 1
                else:
                    agg["value_slot"]["none"] += 1
            if keep:
                agg["keep"] += 1; right = r["label"] == gold[j]; agg["kept_wrong"] += not right
                if r["compose_valid"]:
                    agg["compose_valid"] += 1; agg["cv_wrong"] += not right
                    Eg = VB.recode_targets(roots[j]); okE = r["E_self"] == Eg
                    agg["cv_E_right"] += okE; agg["cv_E_wrong"] += not okE
                    if not okE:                                             # value-right label, wrong intermediate state (R2 target error)
                        tb = [t for t in range(len(Eg)) if r["E_self"][t] != Eg[t]]
                        agg.setdefault("examples_E", []).append({"j": j, "input": "".join(N.render(roots[j])), "label_right": bool(right), "t_bad": tb,
                                                                  "self_states": ["".join(VB.rerender(r["E_self"][t])) for t in range(len(Eg))],
                                                                  "gold_states": ["".join(VB.rerender(Eg[t])) for t in range(len(Eg))]})
                else:
                    agg["keep_not_cv"] += 1; agg["keep_not_cv_valright"] += right
                    for k in agg["fail_flag_among_keep"]:
                        agg["fail_flag_among_keep"][k] += not r["flags"][k]
                    if len(agg["examples"]) < 6:                             # first failing step, for the report
                        for (ct, d_) in seq:
                            f, _ = C.check_step(ct, d_)
                            if not all(f[k] for k in ("FRAME", "SPAN", "ORDER", "CANON")):
                                agg["examples"].append({"input": "".join(N.render(ct)), "x": C.x_classes(N.render(ct)), "dec": d_, "flags": f,
                                                        "canon_target": VB.recode_targets(ct, 1)[1]})
                                break
        agg["keep_rate"] = agg["keep"] / J; agg["cv_rate"] = agg["compose_valid"] / J
        agg["prec_kept"] = 1 - agg["kept_wrong"] / agg["keep"] if agg["keep"] else None
        agg["prec_cv"] = 1 - agg["cv_wrong"] / agg["compose_valid"] if agg["compose_valid"] else None
        agg["piE"] = agg["cv_E_right"] / agg["compose_valid"] if agg["compose_valid"] else None
        agg["sec"] = round(time.time() - t1, 1)
        res[str(n)] = agg
        print("n", n, {k: v for k, v in agg.items() if k not in ("examples",)}, flush=True)
        json.dump({"run": RUN, "device": "cpu fp32", "threads": THREADS, "bs": BS, "per_n": PER, "by_n": res}, open(OUT, "w", newline="\n"), indent=1)
    print("DONE", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
