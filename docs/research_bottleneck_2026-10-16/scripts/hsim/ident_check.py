# Identification check for the NO-OP-ALIVE predicate on the VB recode targets (read-only import of the repo's generator).
import sys, os
os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"
sys.dont_write_bytecode = True
sys.path.insert(0, "D:/Latent")
import numpy as np, torch
torch.set_num_threads(2)
import boxes.vbank as VB, boxes.nested170 as N
OPC = {VB.CID[" " + o] for o in N.OPS}; BRC = {VB.CID[" ("], VB.CID[" )"]}
VALC = {VB.CID[" %d" % v] for v in range(N.P)}; FRAME = {VB.CID["EOT"], VB.CID[" ="], VB.VOID}
rng = np.random.default_rng(1); lone = {o: 0 for o in OPC}; done_bad = 0; notdone_noop = 0; br_noop = 0; steps_ok = 0; tot = 0
for _ in range(20000):
    n = int(rng.integers(1, 9)); root, y = N.generate(rng, n); E = VB.recode_targets(root, n + 4)
    for t, e in enumerate(E):
        s = set(e); ops = [c for c in e if c in OPC]; nv = sum(c in VALC for c in e)
        done = t >= n
        if done:
            done_bad += not (s <= (FRAME | VALC) and nv == 1)
        else:
            notdone_noop += (len(ops) == 0)
            if len(ops) == 1 and not (s & BRC):
                lone[ops[0]] += 1
        if any(c in BRC for c in e) and not ops:
            br_noop += 1
        if 1 <= t <= n:
            prev = E[t - 1]; a0 = sum(c in OPC for c in prev); a1 = len(ops); steps_ok += (a0 - a1 == 1); tot += 1
print({"lone_op_noBracket_counts": {VB.CLS[k]: v for k, v in lone.items()}, "done_states_not_single_value": done_bad,
       "notdone_with_no_op": notdone_noop, "bracket_without_op": br_noop, "alive_op_drop_exactly_1_per_step": steps_ok / tot})
