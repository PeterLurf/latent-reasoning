"""Runs scratch/s251ns_s0.py's parts with inject_aw_eval.pad wrapped by an OverflowError reporter (the first run died in pad with 'int too big to convert'; not
reproduced in isolation). The wrapper is transparent on success (it calls the original pad and returns its result unchanged).
    python -B scratch/s251ns_run_diag.py perm generic probe"""
import json
import os
import sys
import traceback

import numpy as np

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import s251ns_s0 as N  # noqa: E402
from boxes import inject_aw_eval as E  # noqa: E402
from boxes import inject_st as ST  # noqa: E402

P0 = E.pad


def pad(seqs, fill=E.EOT):
    try:
        return P0(seqs, fill)
    except OverflowError:
        print("PAD OVERFLOW: n", len(seqs), "fill", fill, type(fill), "lens", sorted({len(q) for q in seqs})[:5], flush=True)
        for q in seqs:
            bad = [x for x in q if not isinstance(x, (int, np.integer)) or int(x) > 60000 or int(x) < 0]
            if bad:
                print("BAD elems", bad[:3], [type(x) for x in bad[:3]], flush=True); break
        else:
            print("no bad elem; elem types", {type(x) for q in seqs for x in q}, flush=True)
        traceback.print_stack(); raise


E.pad = pad; ST.pad = pad
if __name__ == "__main__":
    os.makedirs(N.OUT, exist_ok=True); fns = {"perm": N.perm, "generic": N.generic_part, "probe": N.probe}
    for p_ in sys.argv[1:]:
        r = fns[p_]("cuda:0", 100); json.dump(r, open(os.path.join(N.OUT, p_ + ".json"), "w", newline="\n"), indent=1, default=str); print(p_, "written", flush=True)
