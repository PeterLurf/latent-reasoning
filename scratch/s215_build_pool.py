"""§215 round-1 pool builder (SELFDEPTH_v1 §2.1-§2.2 at 5ded8703). Local 3070 (cuda:1), fp32, bs 32, labeller = s213_d3t_s1/model.pt.
  1. exclusion set (before any patch): test_set(1000, n_max 20) syms + the SERIAL pool (rng 17300, 2,200 x generate(8)) + the 213-37 fresh items
     (rng 21337, 1,000 per n 9..20, logged separately).
  2. GOLD PATCH: N.apply, N.value, N.eval_tokens, N.cot_steps, VB.recode_targets, VB.rec_targets and vbank_diag.reduce_k raise from here on.
  3. GATE-1 items: default_rng(21960), 2,000 per n 9..12 (n order), redrawn on an exclusion hit -> boxes/runs_fsq/s215_gate1/gate_items.json
     (syms + n only); added to the exclusion set.
  4. pool: default_rng(21900), per n 9..12 (n order) chunks of 4,096 draws, each labelled by selfdepth.label_chains; kept = KEEP AND CV; exclusion hits
     and within-pool duplicates redrawn (counted); until 110,000 kept per n. The generator's (root, t) are dropped right after render; n(syms) is
     asserted == the generator n.
  5. one fixed interleaved order: default_rng(21901).permutation over the 440,000 kept records -> boxes/runs_fsq/s215_pool/pool_r1.jsonl (one record
     per line, keys exactly {syms, n, chain_label, steps}); pool_meta.json {seed, perm_seed, per_n, draws_n, collide_n, dup_n, keep_n, sha256,
     builder_commit, card, dtype, bs, ...}.
    python scratch/s215_build_pool.py --device cuda:1"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402
import boxes.vbank_diag as VD  # noqa: E402
import boxes.vbank_eval as VE  # noqa: E402
import boxes.selfdepth as SD  # noqa: E402

OUTP, OUTG = "boxes/runs_fsq/s215_pool", "boxes/runs_fsq/s215_gate1"; PER_N, NS, GATE_N = 110_000, (9, 10, 11, 12), 2000


def patch_gold():
    def boom(*a, **k):
        raise RuntimeError("G-S0: a gold function was called in the pool builder")
    for mod, nm in ((N, "apply"), (N, "value"), (N, "eval_tokens"), (N, "cot_steps"), (VB, "recode_targets"), (VB, "rec_targets"), (VD, "reduce_k")):
        setattr(mod, nm, boom)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:1"); ap.add_argument("--per_n", type=int, default=PER_N, help="smokes only")
    ap.add_argument("--gate_n", type=int, default=GATE_N, help="smokes only"); ap.add_argument("--out_suffix", default=""); a = ap.parse_args(); dev = a.device
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; t0 = time.time()
    outp, outg = OUTP + a.out_suffix, OUTG + a.out_suffix; os.makedirs(outp, exist_ok=True); os.makedirs(outg, exist_ok=True)
    m, _ = VE.load("boxes/runs_fsq/s213_d3t_s1", dev)
    excl = {tuple(it["syms"]) for it in N.test_set(1000, n_max=20)}; n_test = len(excl); r_sp = np.random.default_rng(17300)
    for _ in range(2200):
        excl.add(tuple(N.render(N.generate(r_sp, 8)[0])))
    n_serial = len(excl) - n_test; r37 = np.random.default_rng(21337); ex37 = set()
    for n in range(9, 21):
        got = 0
        while got < 1000:
            r_, _ = N.generate(r37, n); s_ = tuple(N.render(r_))
            if s_ in excl:
                continue
            ex37.add(s_); got += 1
    patch_gold()                                                            # G-S0: no gold function from here on
    rg = np.random.default_rng(21960); gate = []; gcoll = {}
    for n in NS:
        got = 0; gcoll[str(n)] = 0
        while got < a.gate_n:
            r_, _ = N.generate(rg, n); s_ = N.render(r_); del r_
            if tuple(s_) in excl or tuple(s_) in ex37:
                gcoll[str(n)] += 1; continue
            assert SD.n_of(s_) == n; gate.append({"syms": s_, "n": n}); excl.add(tuple(s_)); got += 1
    json.dump({"items": gate, "seed": 21960, "per_n": a.gate_n, "collide_n": gcoll}, open(os.path.join(outg, "gate_items.json"), "w", newline="\n"))
    rp = np.random.default_rng(21900); kept, draws, coll, dup, keepn, seen = [], {}, {}, {}, {}, set(); at_target = {}; labn, dropn = {}, {}
    for n in NS:
        draws[str(n)] = coll[str(n)] = dup[str(n)] = 0; kn = 0; nlab = 0; dropn[str(n)] = 0
        while kn < a.per_n:
            batch = []; bdraw = []
            while len(batch) < 4096:
                r_, _ = N.generate(rp, n); s_ = N.render(r_); del r_; draws[str(n)] += 1
                ts = tuple(s_)
                if ts in excl or ts in ex37:
                    coll[str(n)] += 1; continue
                if ts in seen:
                    dup[str(n)] += 1; continue
                assert SD.n_of(s_) == n; seen.add(ts); batch.append(s_); bdraw.append(draws[str(n)])
            lab = SD.label_chains(m, batch, dev, bs=32)
            for q, (s_, o) in enumerate(zip(batch, lab)):
                if o["keep"] and o["cv"] and kn >= a.per_n:
                    dropn[str(n)] += 1                                      # kept-eligible beyond the target in the last chunk: dropped, never trained
                if o["keep"] and o["cv"] and kn < a.per_n:
                    kept.append({"syms": s_, "n": n, "chain_label": o["label"], "steps": o["steps"]}); kn += 1
                    if kn == a.per_n:                                       # both counts logged (Theory 10-17): up to the target's kept item
                        at_target[str(n)] = {"draws": bdraw[q], "labelled": nlab + q + 1, "keep": kn / (nlab + q + 1)}
            nlab += len(batch)
        keepn[str(n)] = (kn + dropn[str(n)]) / nlab; labn[str(n)] = nlab
        print("POOL n", n, "kept", kn, "labelled", nlab, "keep %.4f" % keepn[str(n)], "draws", draws[str(n)], "coll", coll[str(n)], "dup", dup[str(n)], "%.0f s" % (time.time() - t0), flush=True)
    perm = np.random.default_rng(21901).permutation(len(kept)); fp = os.path.join(outp, "pool_r1.jsonl")
    with open(fp, "w", encoding="utf-8", newline="\n") as f:
        for i in perm:
            r = kept[int(i)]; assert set(r) == SD.REC_KEYS; f.write(json.dumps(r, separators=(",", ":")) + "\n")
    meta = {"seed": 21900, "perm_seed": 21901, "per_n": a.per_n, "ns": list(NS), "n_records": len(kept), "draws_n": draws, "collide_n": coll, "dup_n": dup,
            "keep_all": keepn, "keep_all_def": "kept-eligible (KEEP and CV) / all labelled, incl. the last chunk's overshoot", "keep_to_target": {k: v["keep"] for k, v in at_target.items()},
            "at_target": at_target, "labelled_n": labn, "dropped_extra_n": dropn, "gate_seed": 21960, "gate_per_n": a.gate_n, "gate_collide_n": gcoll, "excl": {"test_n_max20": n_test, "serial": n_serial, "s213_37": len(ex37)},
            "sha256": SD.file_sha256(fp), "builder_commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
            "labeller": "boxes/runs_fsq/s213_d3t_s1/model.pt", "card": torch.cuda.get_device_name(dev), "dtype": "fp32", "bs": 32, "cap": "n + 4",
            "filter": "KEEP and COMPOSE-VALID", "sec": round(time.time() - t0)}
    json.dump(meta, open(os.path.join(outp, "pool_meta.json"), "w", newline="\n"), indent=1); print("DONE", meta["n_records"], meta["sha256"][:16], meta["sec"], flush=True)


if __name__ == "__main__":
    main()
