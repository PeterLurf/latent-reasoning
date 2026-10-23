"""§227 decode-OOM fix gates (10-23; the TEXT run OOMed in decode_eval at step 16,000). OLD = the committed boxes/codi_ps.py, NEW = scratch/_tmp/s227b_tree.
On the RTX 4060 (cuda:0), the 14k checkpoint of scratch/_tmp/s227_text/TEXT loaded into build_full (fp32 master, bf16 autocast as in training):
  equiv   OLD vs NEW greedy_decode on the run's held-out set (the same 22950 build) for d 1..4, the full 200-item batch: every generated token list,
          terminated and overflow flag identical.
  mem     NEW decode_eval for d 1..8 (d_max 7) at the full batch with the training state resident (AdamW state loaded from the checkpoint and every
          parameter's .grad allocated, as between steps): no OOM; peak allocated recorded. No accuracy values are printed or stored (only shape fields).
    python -B scratch/gate_s227b.py equiv mem   -> scratch/gate_s227/fix_<part>.json"""
import importlib.util
import json
import os
import sys

import torch

DEV = "cuda:0"; RUN = "D:/Latent/scratch/_tmp/s227_text/TEXT"; OUT = "D:/Latent/scratch/gate_s227"


def load_mod(root, name):
    sys.path.insert(0, root); sp = importlib.util.spec_from_file_location(name, os.path.join(root, "boxes", "codi_ps.py"))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def setup(PS):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2"); PS.VALUE_TOKS.update(PS.value_tokens(enc)); a = json.load(open(os.path.join(RUN, "args.json")))
    m, prj = PS.build_full(DEV, a["seed"]); c = torch.load(os.path.join(RUN, "ckpt_last.pt"), map_location=DEV, weights_only=False); TP = PS.CP.trainable(m, prj)
    for n, p_ in TP:
        p_.data.copy_(c["tp"][n])
    TE = PS.OW.test_set(order_mode=a["order_mode"], dag=a["dag"] == "on"); PC = PS.OW.pair_candidates(order_mode=a["order_mode"], dag=a["dag"] == "on")
    excl = {it["sig"] for d, its in TE.items() for it in its}
    es = PS.OW.Stream(22950, excl, frames=PS.NW.FRAME_TRAIN, order_mode=a["order_mode"], dag=a["dag"] == "on"); EDraw = {d: [es.draw(d) for _ in range(200)] for d in range(1, 9)}
    sha = PS.OW.set_sha([it for d in sorted(EDraw) for it in EDraw[d]]); assert sha == a["text_eval_sha"], (sha, a["text_eval_sha"])
    ED = {d: [PS.tok_item(it, enc, True) for it in EDraw[d]] for d in EDraw}
    return m, prj, TP, c, ED, enc


def equiv():
    OLD = load_mod("D:/Latent", "codi_old"); NEW = load_mod("D:/Latent/scratch/_tmp/s227b_tree", "codi_new")
    m, prj, TP, c, ED, enc = setup(NEW); m.eval(); same = {}
    for d in range(1, 5):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a_ = OLD.greedy_decode(m, ED[d], DEV, 24 * d + 16); torch.cuda.empty_cache(); b_ = NEW.greedy_decode(m, ED[d], DEV, 24 * d + 16)
        same[str(d)] = {"n": len(a_), "identical": sum(x == y for x, y in zip(a_, b_))}
    o = {"by_d": same, "pass": all(v["identical"] == v["n"] == 200 for v in same.values())}; return o


def mem():
    NEW = load_mod("D:/Latent/scratch/_tmp/s227b_tree", "codi_new"); m, prj, TP, c, ED, enc = setup(NEW)
    import argparse
    a = argparse.Namespace(lr=1e-4, wd=0.1, warmup=0.03); opt, sch, _ = NEW.make_opt(a, TP, 48000); opt.load_state_dict(c["opt"])
    for _, p_ in TP:
        p_.grad = torch.zeros_like(p_)
    torch.cuda.reset_peak_memory_stats(DEV); m.train(); prj.train()
    try:
        rec = NEW.decode_eval(m, prj, ED, enc, DEV, 7, 0); ok = True; err = ""
    except torch.OutOfMemoryError as e:
        rec = None; ok = False; err = str(e)[:300]
    pk = torch.cuda.max_memory_allocated(DEV) / 2 ** 30
    o = {"ok": ok, "err": err, "peak_gib": pk, "ds": sorted(rec["by_d"]) if rec else None, "n_per_d": [rec["by_d"][d]["n"] for d in sorted(rec["by_d"])] if rec else None,
         "card": torch.cuda.get_device_name(DEV)}
    o["pass"] = ok and o["ds"] == [str(k) for k in range(1, 9)] and pk < 7.6; return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for part in [x for x in sys.argv[1:] if x in ("equiv", "mem")]:
        r = {"equiv": equiv, "mem": mem}[part](); json.dump(r, open(os.path.join(OUT, "fix_%s.json" % part), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
