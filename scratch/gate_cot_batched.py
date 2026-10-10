"""prosqa168s2.generate_cot_batched gate (Audit 2026-10-10): on CPU fp32 with pretrained GPT-2, the batched KV-cached greedy CoT equals generate_cot
(batch 1, no cache) token for token on p8 test prompts: 3 shared-length groups (batched rows) + singletons, max_new 40; run 1 with the real stop
tokens (ANS / EOS), run 2 with ANS mapped to "." (rows stop at different steps: tests the per-row cut). -> boxes/runs_fsq/s168s2_gates/gate_cot_batched.json
    python scratch/gate_cot_batched.py"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import torch  # noqa: E402

import boxes.prosqa168s2 as P  # noqa: E402
from boxes import s5  # noqa: E402

torch.set_num_threads(3); g = s5.load_gpt2("cpu").eval(); D = P.load(os.path.join(ROOT, "prosqa", "p8data", "p8_na_test.json")); by = {}
for it in D:
    by.setdefault(len(it["pre"]), []).append(it)
grp = sorted((v for v in by.values() if len(v) >= 4), key=len, reverse=True)[:3]; sing = [v[0] for v in by.values() if len(v) == 1][:4]
items = [it for v in grp for it in v[:4]] + sing; pres = [it["pre"] for it in items]; res = {"n_items": len(items), "group_lens": [len(v[0]["pre"]) for v in grp]}
for run, ans in (("real_stops", P.ANS), ("ans_is_period", P.ENC.encode(".")[0])):
    P.ANS = ans; ref = [P.generate_cot(g, p, 40, "cpu", None) for p in pres]; bat = P.generate_cot_batched(g, pres, 40, "cpu", None, 4)
    res[run] = {"identical": sum(int(a == b) for a, b in zip(ref, bat)), "gen_lens": [len(r) - len(p) for r, p in zip(ref, pres)]}
    print(run, res[run], flush=True)
res["pass"] = all(res[r]["identical"] == len(items) for r in ("real_stops", "ans_is_period")) and len(set(res["ans_is_period"]["gen_lens"])) > 1
os.makedirs("boxes/runs_fsq/s168s2_gates", exist_ok=True); json.dump(res, open("boxes/runs_fsq/s168s2_gates/gate_cot_batched.json", "w", newline="\n"), indent=1); print("PASS", res["pass"])
