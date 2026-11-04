"""§271 G0 unseen-key test (Note 271-2): with empty fit tables every bag / last-2 key is unseen, so both MAP shortcuts must score exactly 1/(n-1) on MOVED items,
and the unseen shares must be 1.0; then a real fit on (5, 8) reports its unseen share. A throwaway draw (default_rng(27198)); the frozen floors file is not touched.
    python -B scratch/n1_build/g0_unseen.py   -> scratch/gate_s271n1/g0_unseen.json"""
import json, os, sys
NEW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.insert(0, NEW); os.chdir(NEW)
import numpy as np  # noqa: E402
import boxes.so_words as SO  # noqa: E402
out = {}
for n in (3, 5):
    its = [SO.make(np.random.default_rng((27198, n, i)), n, 8, "num") for i in range(3000)]
    SO._FIT[(n, 8)] = ({}, {}); r = SO.floors(its); u = 1.0 / (n - 1)
    out["empty_fit_n%d" % n] = {"bag_MAP": r["by_shortcut"]["bag_MAP"], "last2_MAP": r["by_shortcut"]["last2_MAP"], "expected": u, "unseen": r["unseen"],
                                "ok": abs(r["by_shortcut"]["bag_MAP"] - u) < 1e-12 and abs(r["by_shortcut"]["last2_MAP"] - u) < 1e-12 and r["unseen"] == {"bag": 1.0, "last2": 1.0}}
    del SO._FIT[(n, 8)]
r = SO.floors([SO.make(np.random.default_rng((27198, 9, i)), 5, 8, "num") for i in range(3000)]); out["real_fit_5_8"] = {"f": r["f"], "arg": r["arg"], "unseen": r["unseen"]}
out["pass"] = all(v["ok"] for k, v in out.items() if k.startswith("empty"))
os.makedirs("scratch/gate_s271n1", exist_ok=True); json.dump(out, open("scratch/gate_s271n1/g0_unseen.json", "w", newline="\n"), indent=1); print(json.dumps(out))
