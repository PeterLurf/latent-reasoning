"""201-4 drift test: halt173.halt_fields from the code tree argv[1] (sys.path first), bf16 / fp32, on a run dir (absolute) -> json (scratchpad only)."""
import json, os, sys, time
code, run, dev, prec, out = sys.argv[1:6]; BS = int(sys.argv[6]) if len(sys.argv) > 6 else 250; sys.path.insert(0, code); os.environ.setdefault("BOXES_NB", "5")
import torch
import boxes.deltamem as DM
import boxes.halt173 as H
assert os.path.dirname(os.path.dirname(os.path.abspath(H.__file__))).replace("\\", "/").lower() == code.replace("\\", "/").lower().rstrip("/"), ("wrong tree", H.__file__)
kind, m, a = H.load_model(run, dev); t0 = time.time()
_, _, by_t, nn = H.halt_fields(kind, m, a, dev, torch.bfloat16 if prec == "bf16" else None, 1000, BS)
json.dump({"code": code, "module": H.__file__, "by_t": by_t, "n": nn, "prec": prec, "bs": BS, "card": torch.cuda.get_device_name(dev), "torch": torch.__version__, "cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(), "dm_flags": {"FAST_ADD": bool(DM.FAST_ADD), "CHUNK_DELTA": int(getattr(DM, "CHUNK_DELTA", 0))}, "sec": round(time.time() - t0)},
          open(out, "w", newline="\n"), indent=1)
print("done", code, prec, round(time.time() - t0), flush=True)
